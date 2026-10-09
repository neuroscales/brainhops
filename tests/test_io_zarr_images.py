"""Tests for the Zarr and OME-Zarr image readers and writers.

Most tests write and read back, exercising the reader, the writer and the
axis-order seam together. The seam maps the F-ordered (x, y, z, t, c) axes
of brainhops to the C-ordered (t, c, z, y, x) axes of OME-Zarr.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io.images as images
from brainhops.datamodel.axes import (
    ChannelAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import WriterError

# The module imports abczarr itself, so the skip must come first.
abczarr = pytest.importorskip("abczarr")

from brainhops.io.images.zarr import (  # noqa: E402
    OmeImageError,
    OmeZarrImage,
    ZarrImage,
    _axisorder,
)

# Field readers find the component axis in OME metadata or in zarr v3
# dimension_names, which only zarr-python 3 persists, on Python 3.11+.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="OME-Zarr field metadata needs zarr-python 3, which needs "
    "Python 3.11 or newer",
)


def _diag_affine(
    scale: tx.Sequence[float], translation: tx.Sequence[float]
) -> Affine:
    """A vox-to-world affine with a scale and translation per axis."""
    scale = np.asarray(scale, dtype=float)
    translation = np.asarray(translation, dtype=float)
    ndim = scale.shape[0]
    matrix = np.zeros((ndim, ndim + 1))
    matrix[np.arange(ndim), np.arange(ndim)] = scale
    matrix[:, -1] = translation
    return Affine(matrix=matrix)


def _world_matrix(transformation: tx.Any) -> np.ndarray:
    return np.asarray(transformation.compute().to(Affine).matrix)


def _spatial_axes() -> tx.List[SpaceAxis]:
    return [
        SpaceAxis(name="x"),
        SpaceAxis(name="y"),
        SpaceAxis(name="z"),
    ]


# Axis-order seam


# ---- the axis-order seam ---------------------------------------------


def test_seam_maps_storage_order_to_canonical_and_back() -> None:
    stored = [
        TimeAxis(name="t"),
        ChannelAxis(name="c"),
        SpaceAxis(name="z"),
        SpaceAxis(name="y"),
        SpaceAxis(name="x"),
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
        SpaceAxis(name="y"),
        SpaceAxis(name="x"),
    ]
    perm = _axisorder.to_canonical(stored)
    canonical = _axisorder.permute(stored, perm)
    back = _axisorder.permute(canonical, _axisorder.to_storage(canonical))
    assert [a.name for a in back] == [a.name for a in stored]


# Pure Zarr array


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


# Dispatch


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


# OME-Zarr multiscale


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
            _world_matrix(back.images[level].transformation),
            _world_matrix(original.images[level].transformation),
        )


def _stored_axis_names(path: str) -> tx.List[str]:
    """The axis names a written pyramid stores, in stored order.

    OME-NGFF up to 0.5 keeps them on the multiscale and 0.6 on each coordinate
    system. Both are read, so that order tests do not pin the version.
    """
    attrs = dict(abczarr.open(path, mode="r").attrs)
    block = attrs.get("ome", attrs)["multiscales"][0]
    axes = (
        block["axes"]
        if "axes" in block
        else (block["coordinateSystems"][0]["axes"])
    )
    return [a["name"] for a in axes]


def test_multiscale_stores_axes_in_ome_order(tmp_path: Path) -> None:
    axes = [
        SpaceAxis(name="x"),
        SpaceAxis(name="y"),
        SpaceAxis(name="z"),
        TimeAxis(name="t"),
        ChannelAxis(name="c"),
    ]
    data = np.random.default_rng(2).random((4, 5, 6, 2, 3)).astype("float32")
    affine = _diag_affine([1, 2, 3, 1, 1], [10, 20, 30, 0, 0])
    image = SingleScaleImage(data=data, transformations=[affine])
    path = str(tmp_path / "p5.zarr")

    OmeZarrImage(images=[image], axes=axes).save(path)

    assert _stored_axis_names(path) == ["t", "c", "z", "y", "x"]
    # The stored shape is the F-order shape reversed.
    stored = abczarr.open(path, mode="r")
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
    # Scale and translation are kept as a sequence, not collapsed.
    from brainhops.datamodel.transformations import (
        Scaling,
        Sequence,
        Translation,
    )

    geometry = back.images[0].transformation
    assert isinstance(geometry, Sequence)
    assert isinstance(geometry.transformations[0], Scaling)
    assert isinstance(geometry.transformations[1], Translation)
    # The stored (z, y, x) scale (3, 2, 1) is (1, 2, 3) in (x, y, z).
    np.testing.assert_allclose(
        geometry.transformations[0].scale, [1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(
        geometry.transformations[1].translation, [10.0, 20.0, 30.0]
    )


def test_reader_refuses_missing_transformations(tmp_path: Path) -> None:
    # A level without transformations is malformed and is reported.
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
    # A misspelled transformation type is rejected.
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


def test_writer_writes_a_non_axis_aligned_geometry_as_an_affine(
    tmp_path: Path,
) -> None:
    # OME carries a sheared placement as a full affine.
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.5
    image = SingleScaleImage(
        data=np.zeros((3, 3, 3), "float32"),
        transformations=[Affine(matrix=matrix)],
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())
    path = str(tmp_path / "sheared.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.images[0].transformation
    assert isinstance(geometry, Affine)
    np.testing.assert_allclose(_world_matrix(geometry), matrix)


def test_writer_refuses_a_rich_geometry_in_a_scale_only_version(
    tmp_path: Path,
) -> None:
    # OME-NGFF 0.4 has no shear, so 0.4 is refused rather than dropping it.
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.5
    image = SingleScaleImage(
        data=np.zeros((3, 3, 3), "float32"),
        transformations=[Affine(matrix=matrix)],
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())

    with pytest.raises(WriterError):
        pyramid.save(str(tmp_path / "rotated.zarr"), version="0.4")


def test_writer_round_trips_a_rotation_placement(tmp_path: Path) -> None:
    # A rotation round-trips as an OME rotation.
    from brainhops.datamodel.transformations import Rotation

    theta = 0.4
    linear = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    image = SingleScaleImage(
        data=np.zeros((3, 4, 5), "float32"),
        transformations=[Rotation(matrix=linear)],
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())
    path = str(tmp_path / "rot.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.images[0].transformation
    assert isinstance(geometry, Rotation)
    np.testing.assert_allclose(
        _world_matrix(geometry), _world_matrix(image.transformation)
    )


def test_writer_round_trips_a_sequence_placement(tmp_path: Path) -> None:
    # A sequence round-trips with the same kinds, not collapsed.
    from brainhops.datamodel.transformations import (
        Rotation,
        Sequence,
        Translation,
    )

    theta = 0.3
    linear = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    placement = Sequence(
        [Rotation(matrix=linear), Translation(translation=[10.0, 20.0, 30.0])]
    )
    image = SingleScaleImage(
        data=np.zeros((3, 4, 5), "float32"), transformations=[placement]
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())
    path = str(tmp_path / "seq.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.images[0].transformation
    assert isinstance(geometry, Sequence)
    kinds = [type(part).__name__ for part in geometry.transformations]
    assert kinds == ["Rotation", "Translation"]
    np.testing.assert_allclose(
        _world_matrix(geometry), _world_matrix(image.transformation)
    )


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

    # Dispatch reports a general parse failure, so the reader is tested
    # directly.
    with pytest.raises(OmeImageError):
        OmeZarrImage.load(path)


# Public read and write API


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
    # A driver-native array is wrapped in an abczarr node before reading.
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


# Laziness


# ---- laziness --------------------------------------------------------


def test_opening_a_pyramid_does_not_read_its_levels(tmp_path: Path) -> None:
    path = str(tmp_path / "lazy.zarr")
    _small_pyramid().save(path)

    back = OmeZarrImage.from_store(path)

    # No level is read yet: each holds an array handle.
    for level in back.images:
        assert getattr(level, "_cache_data", None) is None
    # Accessing one level reads only that level.
    _ = np.asarray(back.images[0].data)
    assert getattr(back.images[0], "_cache_data", None) is not None
    assert getattr(back.images[1], "_cache_data", None) is None


# Chunking


# ---- chunking --------------------------------------------------------


def test_one_chunking_is_applied_to_every_level(tmp_path: Path) -> None:
    path = str(tmp_path / "chunked.zarr")
    # One chunk shape, in brainhops order, applies to every level.
    _small_pyramid().save(path, chunks=(2, 2, 2))

    group = abczarr.open(path, mode="r")
    assert tuple(group["0"].chunks) == (2, 2, 2)
    assert tuple(group["1"].chunks) == (2, 2, 2)


# Vector component axis


# ---- the vector component axis ---------------------------------------


def _displacement_field() -> OmeZarrImage:
    from brainhops.datamodel.axes import DisplacementAxis

    axes = [
        SpaceAxis(name="x"),
        SpaceAxis(name="y"),
        SpaceAxis(name="z"),
        DisplacementAxis(name="v"),
    ]
    data = np.zeros((2, 3, 4, 3), dtype="float32")
    # Each component encodes its spatial axis.
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
        SpaceAxis(name="x"),
        SpaceAxis(name="y"),
        SpaceAxis(name="z"),
        DisplacementAxis(name="v"),
    ]
    stored = _axisorder.permute(axes, _axisorder.to_storage(axes))
    # The component axis leads, where a channel would be.
    assert [a.name for a in stored] == ["v", "z", "y", "x"]


def test_field_components_are_not_reordered_by_the_axis_permutation(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "field.zarr")
    _displacement_field().save(path, version="0.6")

    stored = np.asarray(abczarr.open(path, mode="r")["0"][...])
    # The axes are transposed but the component values are not reordered.
    np.testing.assert_allclose(stored[:, 0, 0, 0], [10.0, 20.0, 30.0])

    back = OmeZarrImage.from_store(path)
    read = np.asarray(back.images[0].data)
    # The field round-trips unchanged.
    np.testing.assert_allclose(read[0, 0, 0, :], [10.0, 20.0, 30.0])
    assert np.array_equal(
        read, np.asarray(_displacement_field().images[0].data)
    )


# OME version option


# ---- the OME version option ------------------------------------------


def test_write_version_defaults_to_the_newest_stable(tmp_path: Path) -> None:
    path = str(tmp_path / "default.zarr")
    _small_pyramid().save(path)

    # Only the newest version carries every placement, so it is the default.
    node = abczarr.open(path, mode="r")
    assert node.ome.version == "0.6"


def test_write_version_can_be_requested(tmp_path: Path) -> None:
    path = str(tmp_path / "explicit.zarr")
    _small_pyramid().save(path, version="0.6")

    node = abczarr.open(path, mode="r")
    assert node.ome.version == "0.6"


def test_write_version_falls_back_to_the_source_version(
    tmp_path: Path,
) -> None:
    source = str(tmp_path / "source.zarr")
    # A non-default version proves that the source version is carried.
    from brainhops.io.images.zarr._ome import DEFAULT_WRITE_VERSION

    assert DEFAULT_WRITE_VERSION != "0.5"
    _small_pyramid().save(source, version="0.5")

    # A 0.5 store is written back as 0.5 without restating the version.
    back = OmeZarrImage.from_store(source)
    target = str(tmp_path / "target.zarr")
    back.save(target)

    assert abczarr.open(target, mode="r").ome.version == "0.5"


# Axes derived from OME metadata


# ---- axes are derived from the OME metadata --------------------------


def test_read_pyramid_derives_axes_from_ome(tmp_path: Path) -> None:
    axes = [
        SpaceAxis(name="x"),
        SpaceAxis(name="y"),
        SpaceAxis(name="z"),
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
    # A read pyramid takes its axes from `ome`.
    assert back.axes is None
    assert back.ome is not None

    # Saving again keeps the stored order.
    target = str(tmp_path / "resaved.zarr")
    back.save(target)
    assert _stored_axis_names(target) == ["t", "c", "z", "y", "x"]


# Native transformation mapping


# ---- native transformation mapping -----------------------------------


def _authored_pyramid(
    tmp_path: Path,
    transform: dict,
    axes: tx.List[dict],
    shape: tuple,
    arrays: tx.Optional[dict] = None,
) -> str:
    """Write a 0.6 group whose one level carries `transform`.

    `arrays` names extra arrays, such as the field of a displacement transform.
    """
    from abczarr.ome import v0_6 as v6

    path = str(tmp_path / "authored.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones(shape, "float32"))
    for name, array in (arrays or {}).items():
        # An array, or an (array, dimension_names) pair for a field array.
        if isinstance(array, tuple):
            data, names = array
            group.create_array(name, data=data, dimension_names=names)
        else:
            group.create_array(name, data=array)
    group.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [{"name": "phys", "axes": axes}],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                dict(
                                    transform,
                                    input={"path": "0"},
                                    output={"name": "phys"},
                                )
                            ],
                        }
                    ],
                }
            ],
        }
    )
    return path


_SPACE3 = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
_SPACE2 = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def test_reader_maps_a_rotation_to_a_rotation(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import Rotation

    path = _authored_pyramid(
        tmp_path,
        {"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]},
        _SPACE2,
        (5, 4),
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Rotation)
    # The stored (y, x) rotation is reordered to (x, y).
    np.testing.assert_allclose(
        np.asarray(geometry.matrix), [[0.0, 1.0], [-1.0, 0.0]]
    )


def test_reader_maps_an_affine_to_an_affine(tmp_path: Path) -> None:
    path = _authored_pyramid(
        tmp_path,
        {"type": "affine", "affine": [[2.0, 0.0, 5.0], [0.0, 3.0, 6.0]]},
        _SPACE2,
        (5, 4),
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Affine)
    np.testing.assert_allclose(
        np.asarray(geometry.matrix), [[3.0, 0.0, 6.0], [0.0, 2.0, 5.0]]
    )


def test_reader_maps_a_sequence_to_a_sequence(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import (
        Scaling,
        Sequence,
        Translation,
    )

    path = _authored_pyramid(
        tmp_path,
        {
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": [3.0, 2.0, 1.0]},
                {"type": "translation", "translation": [30.0, 20.0, 10.0]},
            ],
        },
        _SPACE3,
        (6, 5, 4),
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Sequence)
    assert isinstance(geometry.transformations[0], Scaling)
    assert isinstance(geometry.transformations[1], Translation)


def test_reader_maps_a_map_axis_to_a_permutation(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import Permutation

    path = _authored_pyramid(
        tmp_path, {"type": "mapAxis", "mapAxis": [1, 0]}, _SPACE2, (5, 4)
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Permutation)
    np.testing.assert_array_equal(np.asarray(geometry.permutation), [1, 0])


def test_reader_maps_a_projecting_map_axis_to_a_projection() -> None:
    # A mapAxis over a subset of the input axes is a Projection; tested on
    # the mapping, since such a level is not wired through the reader.
    from abczarr.ome.v0_6.transformations import CoordinateTransformation

    from brainhops.datamodel.transformations import Projection
    from brainhops.io.images.zarr import _ome

    transform = CoordinateTransformation.from_json(
        {"type": "mapAxis", "mapAxis": [0, 2]}
    )
    projection = _ome._map_transform(transform, perm=[0, 1, 2], ndim=3)
    assert isinstance(projection, Projection)
    np.testing.assert_array_equal(np.asarray(projection.dropped), [1])
    np.testing.assert_array_equal(np.asarray(projection.created), [])


# Displacement and coordinate fields


# ---- displacement and coordinate fields ------------------------------


def _field_array() -> np.ndarray:
    # The field is stored as (component, z, y, x), with each component
    # constant.
    field = np.zeros((3, 6, 5, 4), dtype="float32")
    field[0] = 100.0
    field[1] = 200.0
    field[2] = 300.0
    return field


def _typed_axes(spec: tx.Sequence[tx.Tuple[str, str]]) -> tx.List[dict]:
    return [{"name": name, "type": type_} for name, type_ in spec]


def _authored_field_pyramid(
    tmp_path: Path,
    field: np.ndarray,
    field_axes: tx.Sequence[tx.Tuple[str, str]],
) -> str:
    """Write a pyramid placed by a displacement field node with typed axes
    and no dimension_names, as real 0.6 files are.
    """
    from abczarr.ome import v0_6 as v6

    path = str(tmp_path / "field_ome.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones((6, 5, 4), "float32"))
    node = group.create_array("disp", data=field)
    node.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {"name": "field", "axes": _typed_axes(field_axes)}
                    ],
                    "datasets": [
                        {
                            "path": ".",
                            "coordinateTransformations": [
                                {
                                    "type": "identity",
                                    "input": {"path": "."},
                                    "output": {"name": "field"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    group.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [{"name": "phys", "axes": _SPACE3}],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "displacements",
                                    "path": "disp",
                                    "input": {"path": "0"},
                                    "output": {"name": "phys"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    return path


def test_reader_reads_a_field_from_its_own_typed_ome(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import DisplacementField

    # The component axis is stored last and found by its type.
    field = np.moveaxis(_field_array(), 0, -1)  # (z, y, x, d)
    path = _authored_field_pyramid(
        tmp_path,
        field,
        [
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
            ("d", "displacement"),
        ],
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, DisplacementField)
    result = np.asarray(geometry.field)
    # Read as (x, y, z, component), without reordering the values.
    assert result.shape == (4, 5, 6, 3)
    np.testing.assert_allclose(result[0, 0, 0, :], [100.0, 200.0, 300.0])


def test_reader_reads_a_displacement_field(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import DisplacementField

    # As a fallback, a bare array whose dimension_names name the axes is read.
    path = _authored_pyramid(
        tmp_path,
        {"type": "displacements", "path": "disp"},
        _SPACE3,
        (6, 5, 4),
        arrays={"disp": (_field_array(), ("d", "z", "y", "x"))},
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, DisplacementField)
    field = np.asarray(geometry.field)
    # The component axis moves to the end.
    assert field.shape == (4, 5, 6, 3)
    # The component values are not reordered.
    np.testing.assert_allclose(field[0, 0, 0, :], [100.0, 200.0, 300.0])


def test_reader_reads_a_field_by_its_axis_names_not_position(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.transformations import DisplacementField

    # The component axis is stored last and found by dimension_names.
    field = np.moveaxis(_field_array(), 0, -1)  # (z, y, x, d)
    path = _authored_pyramid(
        tmp_path,
        {"type": "displacements", "path": "disp"},
        _SPACE3,
        (6, 5, 4),
        arrays={"disp": (field, ("z", "y", "x", "d"))},
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, DisplacementField)
    result = np.asarray(geometry.field)
    assert result.shape == (4, 5, 6, 3)
    np.testing.assert_allclose(result[0, 0, 0, :], [100.0, 200.0, 300.0])


def test_reader_refuses_a_field_that_does_not_name_its_axes(
    tmp_path: Path,
) -> None:
    # Without dimension_names, the reader refuses to guess.
    path = _authored_pyramid(
        tmp_path,
        {"type": "displacements", "path": "disp"},
        _SPACE3,
        (6, 5, 4),
        arrays={"disp": _field_array()},
    )
    with pytest.raises(OmeImageError):
        OmeZarrImage.from_store(path)


def test_reader_reads_a_coordinate_field(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import CoordinatesField

    path = _authored_pyramid(
        tmp_path,
        {"type": "coordinates", "path": "coord"},
        _SPACE3,
        (6, 5, 4),
        arrays={"coord": (_field_array(), ("d", "z", "y", "x"))},
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, CoordinatesField)
    assert np.asarray(geometry.field).shape == (4, 5, 6, 3)


def test_reader_reads_an_affine_surrounded_field(tmp_path: Path) -> None:
    from brainhops.datamodel.transformations import (
        DisplacementField,
        Scaling,
        Sequence,
    )

    path = _authored_pyramid(
        tmp_path,
        {
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                {"type": "displacements", "path": "disp"},
            ],
        },
        _SPACE3,
        (6, 5, 4),
        arrays={"disp": (_field_array(), ("d", "z", "y", "x"))},
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Sequence)
    assert isinstance(geometry.transformations[0], Scaling)
    assert isinstance(geometry.transformations[1], DisplacementField)


def test_reader_maps_a_multi_field_sequence_without_banning(
    tmp_path: Path,
) -> None:
    # The image reader maps any OME composition faithfully; the affine
    # constraint on a field's placement is enforced by the field object.
    from brainhops.datamodel.transformations import (
        DisplacementField,
        Sequence,
    )

    path = _authored_pyramid(
        tmp_path,
        {
            "type": "sequence",
            "transformations": [
                {"type": "displacements", "path": "disp"},
                {"type": "displacements", "path": "disp"},
            ],
        },
        _SPACE3,
        (6, 5, 4),
        arrays={"disp": (_field_array(), ("d", "z", "y", "x"))},
    )
    geometry = OmeZarrImage.from_store(path).images[0].transformation
    assert isinstance(geometry, Sequence)
    assert all(
        isinstance(part, DisplacementField)
        for part in geometry.transformations
    )
