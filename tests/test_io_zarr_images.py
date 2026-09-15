"""Tests for the Zarr and OME-Zarr image readers and writers.

The tests favour write-then-read round-trips, since a round-trip exercises
the reader, the writer, and the axis-order seam between them at once. The
seam maps the brainhops F-order ``(x, y, z, t, c)`` to the OME-Zarr C-order
``(t, c, z, y, x)`` at the boundary.
"""

import sys
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

# The zarr I/O runs through abczarr's zarr-python driver, and that driver
# requires zarr-python 3, which itself requires Python 3.11 or newer. On an
# older interpreter the driver cannot be installed at all, so the whole
# feature is unavailable there. These tests are gated on the interpreter
# version, not on driver presence: on every supported interpreter a driver
# is guaranteed by the test dependencies, so a missing driver there is a
# real failure rather than a skip.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="zarr I/O requires zarr-python 3, which needs Python 3.11 or newer",
)


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


def test_reader_refuses_missing_transformations(tmp_path: Path) -> None:
    # A level with no coordinate transformations is malformed metadata.
    # abczarr rejects it while parsing, so the reader reports the fault
    # rather than reading the level with a silent identity geometry.
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

    with pytest.raises(OmeImageError):
        OmeZarrImage.load(path)


def test_reader_refuses_misspelled_transform_type(tmp_path: Path) -> None:
    # A transformation whose type is misspelled is malformed metadata.
    # abczarr rejects it rather than skipping it, so a wrong geometry is
    # never produced silently.
    path = str(tmp_path / "typo.zarr")
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
                                {"type": "scal", "scale": [2.0, 2.0]}
                            ],
                        }
                    ],
                }
            ]
        }
    )

    with pytest.raises(OmeImageError):
        OmeZarrImage.load(path)


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


# ---- public read/write API -------------------------------------------


def test_from_store_and_to_store_round_trip(tmp_path: Path) -> None:
    data = np.arange(12, dtype="float32").reshape(3, 4)
    path = str(tmp_path / "api.zarr")

    ZarrImage(data=data).to_store(path)
    back = ZarrImage.from_store(path)

    assert np.array_equal(np.asarray(back.data), data)


def test_from_store_accepts_a_pathlike(tmp_path: Path) -> None:
    data = np.arange(6, dtype="float32").reshape(2, 3)
    path = tmp_path / "pathlike.zarr"
    ZarrImage(data=data).to_store(path)

    back = ZarrImage.from_store(path)

    assert np.array_equal(np.asarray(back.data), data)


def test_from_node_reads_an_opened_abczarr_node(tmp_path: Path) -> None:
    data = np.arange(6, dtype="float32").reshape(2, 3)
    path = str(tmp_path / "node.zarr")
    ZarrImage(data=data).to_store(path)

    node = abczarr.open(path, mode="r")
    back = ZarrImage.from_node(node)

    assert np.array_equal(np.asarray(back.data), data)


def test_from_node_wraps_a_driver_native_array(tmp_path: Path) -> None:
    # A raw driver object, here a zarr-python array, is wrapped in an
    # abczarr node before it is read.
    zarrpy = pytest.importorskip("zarr")
    data = np.arange(6, dtype="float32").reshape(2, 3)
    path = str(tmp_path / "native.zarr")
    ZarrImage(data=data).to_store(path)

    native = zarrpy.open(path, mode="r")
    assert isinstance(native, zarrpy.Array)
    back = ZarrImage.from_node(native)

    assert np.array_equal(np.asarray(back.data), data)


def test_to_node_writes_into_an_existing_array(tmp_path: Path) -> None:
    data = np.arange(6, dtype="float32").reshape(2, 3)
    path = str(tmp_path / "target.zarr")
    node = abczarr.open(path, mode="w", shape=(2, 3), dtype="float32")

    ZarrImage(data=data).to_node(node)
    back = ZarrImage.from_store(path)

    assert np.array_equal(np.asarray(back.data), data)


def test_multiscale_from_store_to_store_round_trip(tmp_path: Path) -> None:
    original = _small_pyramid()
    path = str(tmp_path / "ms.zarr")

    original.to_store(path)
    back = OmeZarrImage.from_store(path)

    assert back.nscales == original.nscales
    for level in range(back.nscales):
        assert np.array_equal(
            np.asarray(back.images[level].data),
            np.asarray(original.images[level].data),
        )


def test_multiscale_to_node_writes_into_a_group(tmp_path: Path) -> None:
    path = str(tmp_path / "grp.zarr")
    group = abczarr.open_group(path, mode="w")

    _small_pyramid().to_node(group)
    back = OmeZarrImage.from_store(path)

    assert back.nscales == 2


# ---- laziness --------------------------------------------------------


def test_opening_a_pyramid_does_not_read_its_levels(tmp_path: Path) -> None:
    path = str(tmp_path / "lazy.zarr")
    _small_pyramid().save(path)

    back = OmeZarrImage.from_store(path)

    # No level's data has been materialized yet: each level holds its array
    # handle and reads it only on access.
    for level in back.images:
        assert getattr(level, "_data", None) is None
    # Accessing one level reads that level, and leaves the others untouched.
    _ = np.asarray(back.images[0].data)
    assert getattr(back.images[0], "_data", None) is not None
    assert getattr(back.images[1], "_data", None) is None


# ---- per-level chunks ------------------------------------------------


def test_per_level_chunks_are_applied_independently(tmp_path: Path) -> None:
    path = str(tmp_path / "chunked.zarr")
    # Chunks are given per level, in the brainhops axis order, and stored in
    # the OME order after transposition.
    _small_pyramid().save(path, chunks=[(2, 2, 2), (1, 1, 1)])

    group = abczarr.open(path, mode="r")
    assert tuple(group["0"].chunks) == (2, 2, 2)
    assert tuple(group["1"].chunks) == (1, 1, 1)


# ---- the vector component axis ---------------------------------------


def _displacement_field() -> OmeZarrImage:
    from brainhops.datamodel.axes import DisplacementAxis

    axes = [
        SpatialAxis(name="x"),
        SpatialAxis(name="y"),
        SpatialAxis(name="z"),
        DisplacementAxis(name="v"),
    ]
    data = np.zeros((2, 3, 4, 3), dtype="float32")
    # Each component encodes the spatial axis it belongs to.
    data[..., 0] = 10.0
    data[..., 1] = 20.0
    data[..., 2] = 30.0
    image = SingleScaleImage(
        data=data, transformations=[_diag_affine([1, 1, 1, 1], [0, 0, 0, 0])]
    )
    return OmeZarrImage(images=[image], axes=axes)


def test_vector_axis_is_grouped_with_the_channel_position() -> None:
    from brainhops.datamodel.axes import DisplacementAxis

    axes = [
        SpatialAxis(name="x"),
        SpatialAxis(name="y"),
        SpatialAxis(name="z"),
        DisplacementAxis(name="v"),
    ]
    stored = _axisorder.permute(axes, _axisorder.to_storage(axes))
    # The component axis leads, in the position a channel axis would take,
    # then the spatial axes in z, y, x.
    assert [a.name for a in stored] == ["v", "z", "y", "x"]


def test_field_components_are_not_reordered_by_the_axis_permutation(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "field.zarr")
    _displacement_field().save(path, version="0.6rc0")

    stored = np.asarray(abczarr.open(path, mode="r")["0"][...])
    # The array axes are transposed to (v, z, y, x), but the component
    # values are not reordered: a field's components are expressed in its
    # output coordinate system, which the seam keeps fixed. So the leading
    # component still holds the x value.
    np.testing.assert_allclose(stored[:, 0, 0, 0], [10.0, 20.0, 30.0])

    back = OmeZarrImage.from_store(path)
    read = np.asarray(back.images[0].data)
    # The round-trip returns the field unchanged.
    np.testing.assert_allclose(read[0, 0, 0, :], [10.0, 20.0, 30.0])
    assert np.array_equal(
        read, np.asarray(_displacement_field().images[0].data)
    )


# ---- the OME version option ------------------------------------------


def test_write_version_defaults_to_a_bare_envelope(tmp_path: Path) -> None:
    path = str(tmp_path / "default.zarr")
    _small_pyramid().save(path)

    attrs = dict(abczarr.open(path, mode="r").attrs)
    # With no source version and scale-and-translation content, the leanest
    # version that carries it is written, whose metadata sits directly in the
    # attributes.
    assert "multiscales" in attrs
    assert "ome" not in attrs


def test_write_version_can_be_requested(tmp_path: Path) -> None:
    path = str(tmp_path / "explicit.zarr")
    _small_pyramid().save(path, version="0.6rc0")

    node = abczarr.open(path, mode="r")
    assert node.ome.version == "0.6rc0"


def test_write_version_falls_back_to_the_source_version(
    tmp_path: Path,
) -> None:
    source = str(tmp_path / "source.zarr")
    _small_pyramid().save(source, version="0.6rc0")

    # A pyramid read from a 0.6rc0 store is written back in 0.6rc0 without
    # the version being restated.
    back = OmeZarrImage.from_store(source)
    target = str(tmp_path / "target.zarr")
    back.save(target)

    assert abczarr.open(target, mode="r").ome.version == "0.6rc0"


# ---- axes are derived from the OME metadata --------------------------


def test_read_pyramid_derives_axes_from_ome(tmp_path: Path) -> None:
    axes = [
        SpatialAxis(name="x"),
        SpatialAxis(name="y"),
        SpatialAxis(name="z"),
        TimeAxis(name="t"),
        ChannelAxis(name="c"),
    ]
    data = np.random.default_rng(3).random((4, 5, 6, 2, 3)).astype("float32")
    image = SingleScaleImage(
        data=data, transformations=[_diag_affine([1, 2, 3, 1, 1], [0] * 5)]
    )
    source = str(tmp_path / "src.zarr")
    OmeZarrImage(images=[image], axes=axes).save(source)

    back = OmeZarrImage.from_store(source)
    # A read pyramid stores no separate axis list; its axes come from `ome`.
    assert back.axes is None
    assert back.ome is not None

    # Re-saving derives the axes from `ome`, so their stored order is kept.
    target = str(tmp_path / "resaved.zarr")
    back.save(target)
    block = dict(abczarr.open(target, mode="r").attrs)["multiscales"][0]
    assert [a["name"] for a in block["axes"]] == ["t", "c", "z", "y", "x"]
