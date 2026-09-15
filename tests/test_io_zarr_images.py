"""Tests for the Zarr and OME-Zarr image readers and writers.

The tests favour write-then-read round-trips, since a round-trip exercises
the reader, the writer, and the axis-order seam between them at once. The
seam maps the brainhops F-order ``(x, y, z, t, c)`` to the OME-Zarr C-order
``(t, c, z, y, x)`` at the boundary.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io.images as images
from brainhops.datamodel.axes import (
    ChannelAxis,
    SpatialAxis,
    TimeAxis,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import WriterError
from brainhops.io.images.zarr import (
    OmeImageError,
    OmeZarrImage,
    ZarrImage,
    _axisorder,
)

abczarr = pytest.importorskip("abczarr")


def _diag_affine(
    scale: tx.Sequence[float], translation: tx.Sequence[float]
) -> Affine:
    """A voxel-to-world affine with a per-axis scale and translation."""
    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    ndim = scale.shape[0]
    matrix = np.zeros((ndim, ndim + 1))
    matrix[np.arange(ndim), np.arange(ndim)] = scale
    matrix[:, -1] = translation
    return Affine(matrix=matrix)


def _spatial_axes() -> tx.List[SpatialAxis]:
    return [
        SpatialAxis(name="x"),
        SpatialAxis(name="y"),
        SpatialAxis(name="z"),
    ]


# ---- the axis-order seam ---------------------------------------------


def test_seam_maps_storage_order_to_canonical_and_back() -> None:
    stored = [
        TimeAxis(name="t"),
        ChannelAxis(name="c"),
        SpatialAxis(name="z"),
        SpatialAxis(name="y"),
        SpatialAxis(name="x"),
    ]
    to_canon = _axisorder.to_canonical(stored)
    canonical = _axisorder.permute(stored, to_canon)
    assert [a.name for a in canonical] == ["x", "y", "z", "t", "c"]

    to_store = _axisorder.to_storage(canonical)
    restored = _axisorder.permute(canonical, to_store)
    assert [a.name for a in restored] == ["t", "c", "z", "y", "x"]


def test_seam_round_trips_any_permutation() -> None:
    stored = [
        ChannelAxis(name="c"),
        SpatialAxis(name="y"),
        SpatialAxis(name="x"),
    ]
    perm = _axisorder.to_canonical(stored)
    canonical = _axisorder.permute(stored, perm)
    back = _axisorder.permute(canonical, _axisorder.to_storage(canonical))
    assert [a.name for a in back] == [a.name for a in stored]


# ---- pure Zarr array -------------------------------------------------


def test_pure_array_round_trip(tmp_path: Path) -> None:
    data = np.arange(24, dtype="float32").reshape(2, 3, 4)
    path = str(tmp_path / "plain.zarr")

    ZarrImage(data=data).save(path)
    back = images.load(path)

    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back), data)
    assert back.dtype == data.dtype


def test_pure_array_defaults_to_identity_geometry(tmp_path: Path) -> None:
    data = np.zeros((2, 3), dtype="uint8")
    path = str(tmp_path / "id.zarr")

    ZarrImage(data=data).save(path)
    back = ZarrImage.load(path)

    assert list(back.transformations) == []


def test_pure_array_reads_supplied_geometry(tmp_path: Path) -> None:
    data = np.zeros((3, 3), dtype="float32")
    path = str(tmp_path / "supplied.zarr")
    ZarrImage(data=data).save(path)

    affine = _diag_affine([2.0, 3.0], [1.0, 4.0])
    back = ZarrImage.load(path, transformation=affine)

    assert back.transformation is affine


def test_pure_array_preserves_chunks(tmp_path: Path) -> None:
    data = np.arange(64, dtype="float32").reshape(8, 8)
    path = str(tmp_path / "chunked.zarr")

    ZarrImage(data=data).save(path, chunks=(4, 4))
    stored = abczarr.open(path, mode="r")

    assert tuple(stored.chunks) == (4, 4)


# ---- dispatch --------------------------------------------------------


def test_dispatch_selects_array_reader_for_a_plain_array(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "arr.zarr")
    ZarrImage(data=np.zeros((2, 2), "f4")).save(path)

    assert images.sniff(path) is ZarrImage
    assert isinstance(images.load(path), ZarrImage)


def test_dispatch_selects_ome_reader_for_a_multiscale_group(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "pyr.zarr")
    _small_pyramid().save(path)

    assert images.sniff(path) is OmeZarrImage
    assert isinstance(images.load(path), OmeZarrImage)


# ---- OME-Zarr multiscale ---------------------------------------------


def _small_pyramid() -> OmeZarrImage:
    data0 = np.random.default_rng(0).random((4, 5, 6)).astype("float32")
    data1 = data0[::2, ::2, ::2].copy()
    level0 = SingleScaleImage(
        data=data0,
        transformations=[_diag_affine([1.0, 2.0, 3.0], [10, 20, 30])],
    )
    level1 = SingleScaleImage(
        data=data1,
        transformations=[_diag_affine([2.0, 4.0, 6.0], [10, 20, 30])],
    )
    return OmeZarrImage(images=[level0, level1], axes=_spatial_axes())


def test_multiscale_round_trip_preserves_data_and_geometry(
    tmp_path: Path,
) -> None:
    original = _small_pyramid()
    path = str(tmp_path / "pyr.zarr")

    original.save(path)
    back = images.load(path)

    assert isinstance(back, OmeZarrImage)
    assert back.nscales == 2
    for level in range(2):
        assert np.array_equal(
            np.asarray(back.images[level].data),
            np.asarray(original.images[level].data),
        )
        np.testing.assert_allclose(
            np.asarray(back.images[level].transformation.matrix),
            np.asarray(original.images[level].transformation.matrix),
        )


def test_multiscale_stores_axes_in_ome_order(tmp_path: Path) -> None:
    axes = [
        SpatialAxis(name="x"),
        SpatialAxis(name="y"),
        SpatialAxis(name="z"),
        TimeAxis(name="t"),
        ChannelAxis(name="c"),
    ]
    data = np.random.default_rng(2).random((4, 5, 6, 2, 3)).astype("float32")
    affine = _diag_affine([1, 2, 3, 1, 1], [10, 20, 30, 0, 0])
    image = SingleScaleImage(data=data, transformations=[affine])
    path = str(tmp_path / "p5.zarr")

    OmeZarrImage(images=[image], axes=axes).save(path)

    stored = abczarr.open(path, mode="r")
    block = dict(stored.attrs)["multiscales"][0]
    assert [a["name"] for a in block["axes"]] == ["t", "c", "z", "y", "x"]
    # The stored array is transposed to the OME order, so the shape is the
    # F-order shape read back to front.
    assert tuple(stored["0"].shape) == (2, 3, 6, 5, 4)

    back = images.load(path)
    assert tuple(back.images[0].data.shape) == (4, 5, 6, 2, 3)
    assert np.array_equal(np.asarray(back.images[0].data), data)


def test_single_scale_written_as_one_level_pyramid(tmp_path: Path) -> None:
    data = np.arange(60, dtype="float32").reshape(3, 4, 5)
    image = SingleScaleImage(
        data=data, transformations=[_diag_affine([2, 2, 2], [0, 0, 0])]
    )
    path = str(tmp_path / "single.zarr")

    OmeZarrImage(images=[image], axes=_spatial_axes()).save(path)
    back = images.load(path)

    assert back.nscales == 1
    assert np.array_equal(np.asarray(back.images[0].data), data)


def test_reader_builds_geometry_from_coordinate_transformations(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "hand.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones((6, 5, 4), "float32"))
    group.update_attributes(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [3.0, 2.0, 1.0]},
                                {
                                    "type": "translation",
                                    "translation": [30.0, 20.0, 10.0],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    )

    back = images.load(path)
    matrix = np.asarray(back.images[0].transformation.matrix)
    # The stored (z, y, x) scale (3, 2, 1) becomes the canonical (x, y, z)
    # scale (1, 2, 3), and likewise the translation.
    np.testing.assert_allclose(np.diag(matrix[:, :-1]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix[:, -1], [10.0, 20.0, 30.0])


def test_reader_treats_missing_transformations_as_identity(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "bare.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones((3, 3, 2), "float32"))
    group.update_attributes(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ]
        }
    )

    back = images.load(path)
    matrix = np.asarray(back.images[0].transformation.matrix)
    np.testing.assert_allclose(matrix, np.eye(3, 4))


def test_writer_refuses_a_non_axis_aligned_geometry(tmp_path: Path) -> None:
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.5
    image = SingleScaleImage(
        data=np.zeros((3, 3, 3), "float32"),
        transformations=[Affine(matrix=matrix)],
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())

    with pytest.raises(WriterError):
        pyramid.save(str(tmp_path / "rotated.zarr"))


def test_reader_refuses_unsupported_coordinate_transformation(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "weird.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones((3, 3), "float32"))
    group.update_attributes(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "affine", "affine": [1, 0, 0, 1]}
                            ],
                        }
                    ],
                }
            ]
        }
    )

    # The reader raises the specific error. Through the dispatcher, that
    # error is reported as a general parse failure instead, so the reader
    # is exercised directly here.
    with pytest.raises(OmeImageError):
        OmeZarrImage.load(path)
