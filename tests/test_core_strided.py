"""
Partial reads of fixed-size values stored in one block of a file.

`StridedProxy` reads only the slabs of the slowest stored axis that a
selection needs, or memory-maps a local file that is not compressed.
These tests check its values against NumPy for every storage order,
compression, axis order, flip and scaling, that a read gets only the
bytes it needs, and that the MRtrix, AFNI and NRRD proxies built on it
give Dask blocks of the storage.
"""

import gzip
import itertools

import numpy as np
import pytest

from brainhops._core.strided import StridedProxy

SHAPE = (5, 6, 7)
"""The stored shape: small, and no two axes of the same length."""

HEADER = b"HEADER__"
"""Bytes that come before the values, which `offset` skips."""


def _values() -> np.ndarray:
    return np.random.default_rng(0).normal(size=SHAPE).astype(">f4")


def _write(path, order: str, compressed: bool) -> None:  # noqa: ANN001
    content = HEADER + _values().tobytes(order=order)
    with (gzip.open if compressed else open)(path, "wb") as f:
        f.write(content)


def _expected(axes: tuple, flips: tuple) -> np.ndarray:
    expected = np.transpose(_values().astype("f4"), axes)
    for axis in flips:
        expected = np.flip(expected, axis)
    return 2 * expected + 1


INDEXES = [
    (),
    (slice(1, 4), 2, slice(None, None, 2)),
    (0,),
    (slice(None, None, -1), slice(2, 5), 3),
    (slice(4, 1, -2),),
    (Ellipsis, 1),
]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    "axes, flips", [((0, 1, 2), ()), ((2, 0, 1), (1,)), ((1, 2, 0), (0, 2))]
)
def test_the_values_are_those_of_numpy(
    tmp_path,  # noqa: ANN001
    order: str,
    compressed: bool,
    axes: tuple,
    flips: tuple,
) -> None:
    path = tmp_path / "values.bin"
    _write(path, order, compressed)
    expected = _expected(axes, flips)
    with open(path, "rb") as stream:
        for source in (path, stream):
            proxy = StridedProxy(
                source,
                ">f4",
                SHAPE,
                offset=len(HEADER),
                order=order,
                axes=axes,
                flips=flips,
                slope=2.0,
                inter=1.0,
            )
            assert proxy.shape == expected.shape
            assert proxy.dtype == np.float32
            for index in INDEXES:
                np.testing.assert_allclose(
                    np.asarray(proxy[index]), expected[index], rtol=1e-6
                )


@pytest.mark.parametrize("order", ["C", "F"])
def test_a_selection_reads_only_its_slabs(
    tmp_path,  # noqa: ANN001
    order: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "values.bin.gz"
    _write(path, order, compressed=True)
    reads = []
    read_bytes = StridedProxy._read_bytes

    def recorded(proxy, position, nbytes):  # noqa: ANN001, ANN202
        reads.append((position, nbytes))
        return read_bytes(proxy, position, nbytes)

    monkeypatch.setattr(StridedProxy, "_read_bytes", recorded)
    proxy = StridedProxy(path, ">f4", SHAPE, offset=len(HEADER), order=order)
    slowest = 0 if order == "C" else 2
    slab = int(np.prod(SHAPE)) // SHAPE[slowest] * 4
    index = [slice(None)] * 3
    index[slowest] = slice(1, 3)
    values = np.asarray(proxy[tuple(index)])
    np.testing.assert_array_equal(values, _values()[tuple(index)])
    assert reads == [(slab, 2 * slab)]


def test_a_local_file_is_memory_mapped(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "values.bin"
    _write(path, "C", compressed=False)
    proxy = StridedProxy(path, ">f4", SHAPE, offset=len(HEADER))
    values = np.asarray(proxy[1:3])
    assert isinstance(values, np.memmap) or isinstance(values.base, np.memmap)
    np.testing.assert_array_equal(values, _values()[1:3])


def test_dask_blocks_are_slabs_of_the_slowest_axis(tmp_path) -> None:  # noqa: ANN001
    dask = pytest.importorskip("dask")
    path = tmp_path / "values.bin.gz"
    _write(path, "F", compressed=True)
    slab_bytes = SHAPE[0] * SHAPE[1] * 4
    with dask.config.set({"array.chunk-size": f"{2 * slab_bytes}B"}):
        proxy = StridedProxy(
            path, ">f4", SHAPE, offset=len(HEADER), order="F", axes=(2, 0, 1)
        )
        assert proxy.base_chunks == (2, SHAPE[0], SHAPE[1])
        lazy = proxy.to_dask()
    assert lazy.chunks[0] == (2, 2, 2, 1)
    np.testing.assert_array_equal(
        lazy.compute(), np.transpose(_values(), (2, 0, 1))
    )


def test_a_short_file_is_reported(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "values.bin.gz"
    with gzip.open(path, "wb") as f:
        f.write(HEADER + b"\0" * 8)
    proxy = StridedProxy(path, ">f4", SHAPE, offset=len(HEADER))
    with pytest.raises(ValueError, match="ends before"):
        np.asarray(proxy)


def test_a_scaling_per_volume_follows_the_selection(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "values.bin"
    _write(path, "F", compressed=False)
    slope = np.arange(1, SHAPE[2] + 1, dtype=float)
    proxy = StridedProxy(
        path, ">f4", SHAPE, offset=len(HEADER), order="F", slope=slope
    )
    expected = _values().astype("f4") * slope
    for index in itertools.product(
        [slice(None), 1], [slice(2, 4)], [3, slice(1, 6, 2)]
    ):
        np.testing.assert_allclose(
            np.asarray(proxy[index]), expected[index], rtol=1e-6
        )


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


def test_an_mrtrix_image_reads_slabs(tmp_path) -> None:  # noqa: ANN001
    dask = pytest.importorskip("dask")
    from brainhops import io
    from brainhops.backends import backend
    from brainhops.io.images.mrtrix import MrtrixImage

    values = np.random.default_rng(1).normal(size=(8, 9, 10)).astype("f4")
    path = tmp_path / "image.mif.gz"
    MrtrixImage(data=values).save(path)
    with dask.config.set({"array.chunk-size": f"{8 * 9 * 4 * 3}B"}):
        with backend("dask"):
            data = io.load(path).data
    assert data.chunks[2] == (3, 3, 3, 1)
    np.testing.assert_allclose(np.asarray(data), values)


@pytest.mark.parametrize("name", ["image+orig.HEAD", "image.nrrd"])
def test_an_image_gives_dask_blocks_of_its_storage(
    tmp_path,  # noqa: ANN001
    name: str,
) -> None:
    dask = pytest.importorskip("dask")
    from brainhops import io
    from brainhops.backends import backend
    from brainhops.datamodel.images import SingleScaleImage

    values = np.random.default_rng(2).normal(size=(8, 9, 10)).astype("f4")
    path = tmp_path / name
    io.save(SingleScaleImage(data=values), path)
    with dask.config.set({"array.chunk-size": f"{8 * 9 * 4 * 3}B"}):
        with backend("dask"):
            data = io.load(path).data
    assert max(len(chunks) for chunks in data.chunks) > 1
    np.testing.assert_allclose(np.asarray(data), values, rtol=1e-6)
