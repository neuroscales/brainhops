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
from bagof.magic import replace

import brainhops.io.images as images
from brainhops.datamodel.axes import (
    ChannelAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import Affine, Translation
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
    image = SingleScaleImage(np.zeros((3, 3, 3), "float32"))
    pyramid = OmeZarrImage(
        images=[image],
        axes=_spatial_axes(),
        transformations=[Affine(matrix=matrix)],
    )
    path = str(tmp_path / "sheared.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.transformation
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
    image = SingleScaleImage(np.zeros((3, 4, 5), "float32"))
    pyramid = OmeZarrImage(
        images=[image],
        axes=_spatial_axes(),
        transformations=[Rotation(matrix=linear)],
    )
    path = str(tmp_path / "rot.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.transformation
    assert isinstance(geometry, Rotation)
    np.testing.assert_allclose(
        _world_matrix(geometry), _world_matrix(pyramid.transformation)
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
    image = SingleScaleImage(np.zeros((3, 4, 5), "float32"))
    pyramid = OmeZarrImage(
        images=[image], axes=_spatial_axes(), transformations=[placement]
    )
    path = str(tmp_path / "seq.zarr")

    pyramid.save(path)
    back = images.load(path)

    geometry = back.transformation
    assert isinstance(geometry, Sequence)
    kinds = [type(part).__name__ for part in geometry.transformations]
    assert kinds == ["Rotation", "Translation"]
    np.testing.assert_allclose(
        _world_matrix(geometry), _world_matrix(pyramid.transformation)
    )


# ---- the 0.6 dataset rule on write -----------------------------------


def _rotation() -> tx.Any:
    from brainhops.datamodel.transformations import Rotation

    theta = 0.4
    return Rotation(
        matrix=[
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _dataset_entries(path: str) -> tx.List[dict]:
    import json

    with open(Path(path) / "zarr.json") as stream:
        ome = json.load(stream)["attributes"]["ome"]
    return [
        dataset["coordinateTransformations"][0]
        for dataset in ome["multiscales"][0]["datasets"]
    ]


def test_writer_moves_the_rotation_of_a_single_level_to_the_pyramid(
    tmp_path: Path,
) -> None:
    # A 0.6 dataset only scales and translates, so the rotation of the only
    # level becomes the placement of the pyramid, which is exact.
    rotation = _rotation()
    image = SingleScaleImage(
        np.zeros((3, 4, 5), "float32"), transformations=[rotation]
    )
    pyramid = OmeZarrImage(images=[image], axes=_spatial_axes())
    path = str(tmp_path / "moved.zarr")

    pyramid.save(path)
    back = images.load(path)

    assert [entry["type"] for entry in _dataset_entries(path)] == ["scale"]
    np.testing.assert_allclose(
        _world_matrix(back.images[0].transformation), np.eye(3, 4)
    )
    np.testing.assert_allclose(
        _world_matrix(back.transformation), _world_matrix(rotation)
    )


@pytest.mark.parametrize("levels", [1, 2])
def test_writer_refuses_a_rotated_level_it_cannot_move(
    tmp_path: Path, levels: int
) -> None:
    # Splitting the geometry of the levels into a diagonal part per level and
    # a shared placement is not done yet (#447), so a rotated level of a
    # pyramid of several levels, or of a pyramid that has its own
    # placement, is refused.
    data = np.zeros((4, 4, 4), "float32")
    rotated = SingleScaleImage(data, transformations=[_rotation()])
    if levels == 2:
        coarse = SingleScaleImage(
            data[::2, ::2, ::2],
            transformations=[_diag_affine([2.0, 2.0, 2.0], [0.5] * 3)],
        )
        pyramid = OmeZarrImage(images=[rotated, coarse], axes=_spatial_axes())
    else:
        pyramid = OmeZarrImage(
            images=[rotated],
            axes=_spatial_axes(),
            transformations=[Translation(translation=[1.0, 2.0, 3.0])],
        )

    with pytest.raises(WriterError, match="#447"):
        pyramid.save(str(tmp_path / "refused.zarr"))


def test_writer_writes_a_diagonal_level_as_a_scale_then_a_translation(
    tmp_path: Path,
) -> None:
    # A translation followed by a scale only scales and translates each
    # axis, so it is written in the order that the prose of 0.6 asks for.
    from brainhops.datamodel.transformations import Scaling, Sequence

    data = np.zeros((4, 4, 4), "float32")
    shifted = Sequence(
        [
            Translation(translation=[1.0, 2.0, 3.0]),
            Scaling(scale=[2.0, 3.0, 4.0]),
        ]
    )
    fine = SingleScaleImage(data, transformations=[shifted])
    coarse = SingleScaleImage(
        data[::2, ::2, ::2],
        transformations=[_diag_affine([4.0, 6.0, 8.0], [3.0, 7.0, 13.0])],
    )
    pyramid = OmeZarrImage(images=[fine, coarse], axes=_spatial_axes())
    path = str(tmp_path / "diagonal.zarr")

    pyramid.save(path)
    back = images.load(path)

    first = _dataset_entries(path)[0]
    kinds = [step["type"] for step in first["transformations"]]
    assert kinds == ["scale", "translation"]
    np.testing.assert_allclose(
        _world_matrix(back.images[0].transformation), _world_matrix(shifted)
    )


# ---- flips on write ----------------------------------------------------


def _scales_written(path: str) -> tx.List[tx.List[float]]:
    # Every OME scale of the metadata, in the levels and in the placements.
    import json

    with open(Path(path) / "zarr.json") as stream:
        ome = json.load(stream)["attributes"]["ome"]
    found = []  # type: tx.List[tx.List[float]]

    def walk(entry: dict) -> None:
        if entry.get("type") == "scale":
            found.append(entry["scale"])
        for child in entry.get("transformations", []):
            walk(child)

    for multiscale in ome["multiscales"]:
        for dataset in multiscale["datasets"]:
            for entry in dataset["coordinateTransformations"]:
                walk(entry)
        for entry in multiscale.get("coordinateTransformations", []):
            walk(entry)
    return found


def _flipped_pyramid(placements: tx.Optional[list] = None) -> OmeZarrImage:
    data = np.zeros((4, 4, 4), "float32")
    fine = SingleScaleImage(
        data, transformations=[_diag_affine([-1.0, 2.0, 3.0], [5, 6, 7])]
    )
    coarse = SingleScaleImage(
        data[::2, ::2, ::2],
        transformations=[_diag_affine([-2.0, 4.0, 6.0], [4.5, 7, 8])],
    )
    return OmeZarrImage(
        images=[fine, coarse],
        axes=_spatial_axes(),
        transformations=placements,
    )


def _world_of_level(pyramid: tx.Any, index: int) -> np.ndarray:
    from brainhops.datamodel.transformations import Sequence

    level = pyramid.images[index].transformation
    if not pyramid.transformations:
        return _world_matrix(level)
    return _world_matrix(Sequence([level, pyramid.transformation]))


@pytest.mark.parametrize("placed", [False, True])
def test_writer_moves_a_shared_flip_to_the_placement(
    tmp_path: Path, placed: bool
) -> None:
    # The flip that every level shares becomes a linear placement, written
    # as an affine, and each level keeps a positive scale (memo item 135).
    placements = [Translation([1.0, 2.0, 3.0])] if placed else None
    pyramid = _flipped_pyramid(placements)
    path = str(tmp_path / "flipped.zarr")

    pyramid.save(path)
    back = images.load(path)

    assert all(value > 0 for scale in _scales_written(path) for value in scale)
    for index in range(2):
        np.testing.assert_allclose(
            _world_of_level(back, index), _world_of_level(pyramid, index)
        )


def test_writer_writes_a_flipped_placement_as_an_affine(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.transformations import Scaling

    pyramid = OmeZarrImage(
        images=[SingleScaleImage(np.zeros((3, 4, 5), "float32"))],
        axes=_spatial_axes(),
        transformations=[Scaling([-1.0, 1.0, 2.0])],
    )
    path = str(tmp_path / "flipped.zarr")

    pyramid.save(path)
    back = images.load(path)

    assert all(value > 0 for scale in _scales_written(path) for value in scale)
    np.testing.assert_allclose(
        _world_matrix(back.transformation),
        _world_matrix(pyramid.transformation),
    )


def test_writer_refuses_levels_that_flip_different_axes(
    tmp_path: Path,
) -> None:
    data = np.zeros((4, 4, 4), "float32")
    fine = SingleScaleImage(
        data, transformations=[_diag_affine([-1.0, 2.0, 3.0], [0, 0, 0])]
    )
    coarse = SingleScaleImage(
        data[::2, ::2, ::2],
        transformations=[_diag_affine([2.0, -4.0, 6.0], [0, 0, 0])],
    )
    pyramid = OmeZarrImage(images=[fine, coarse], axes=_spatial_axes())

    with pytest.raises(WriterError, match="#447"):
        pyramid.save(str(tmp_path / "refused.zarr"))


def test_writer_refuses_a_flip_in_a_version_without_affines(
    tmp_path: Path,
) -> None:
    # Versions 0.4 and 0.5 only hold scales and translations, with no slot
    # for the linear placement of a flip.
    with pytest.raises(WriterError, match="flip"):
        _flipped_pyramid().save(str(tmp_path / "old.zarr"), version="0.5")


# ---- systems of other groups ------------------------------------------


def _parent_and_labels(tmp_path: Path) -> tx.Any:
    # Two pyramids whose systems have the same OME names: one in the root
    # group, one in its labels group.
    path = str(tmp_path / "parent.zarr")
    data = np.zeros((3, 4, 5), "float32")
    OmeZarrImage(images=[SingleScaleImage(data)], axes=_spatial_axes()).save(
        path
    )
    root = abczarr.open_group(path, mode="a")
    cells = root.create_group("labels").create_group("cells")
    labels = OmeZarrImage(
        images=[SingleScaleImage(data.astype("uint8"))], axes=_spatial_axes()
    )
    labels.to_node(cells)
    return root


def test_reader_names_the_systems_of_a_group_after_its_path(
    tmp_path: Path,
) -> None:
    root = _parent_and_labels(tmp_path)

    parent = OmeZarrImage.from_node(root)
    labels = OmeZarrImage.from_node(root["labels/cells"], group="labels/cells")

    parent_space = parent.images[0].transformation.output
    labels_space = labels.images[0].transformation.output
    assert parent_space.name == "physical"
    assert labels_space.name == "labels/cells//physical"
    assert parent_space != labels_space


def test_reader_of_a_group_without_its_path_keeps_the_ome_names(
    tmp_path: Path,
) -> None:
    # Without the path, which only the reader of the root knows, the labels
    # are read as a root group, and their system cannot be told apart from
    # the one of the parent (memo item 132).
    root = _parent_and_labels(tmp_path)

    parent = OmeZarrImage.from_node(root)
    labels = OmeZarrImage.from_node(root["labels/cells"])

    assert (
        labels.images[0].transformation.output
        == parent.images[0].transformation.output
    )


def test_writer_writes_the_ome_names_of_the_systems_of_its_group(
    tmp_path: Path,
) -> None:
    # A pyramid read from a labels group is written under the OME names of
    # its systems.
    root = _parent_and_labels(tmp_path)
    labels = OmeZarrImage.from_node(root["labels/cells"], group="labels/cells")
    world = labels.images[0].transformation.output
    placed = replace(
        labels,
        transformations=[Translation([1.0, 2.0, 3.0], output=world)],
    )
    path = str(tmp_path / "copy.zarr")

    placed.save(path)

    back = OmeZarrImage.from_store(path)
    assert back.transformation.output.name == "physical"


def test_writer_refuses_a_world_space_of_another_group(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.systems import CoordinateSystem

    other = CoordinateSystem(name="labels/cells//physical")
    pyramid = OmeZarrImage(
        images=[SingleScaleImage(np.zeros((3, 4, 5), "float32"))],
        axes=_spatial_axes(),
        transformations=[Translation([1.0, 2.0, 3.0], output=other)],
    )

    with pytest.raises(WriterError, match="labels/cells"):
        pyramid.save(str(tmp_path / "other.zarr"))


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


@pytest.mark.parametrize("kind", ["array", "pyramid"])
def test_from_filename_reads_a_store_as_load_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # A store is a directory, so `from_filename` must not open it as a
    # file. The dispatcher calls `from_filename` by name, and the path is
    # relative, as in the report of the bug.
    monkeypatch.chdir(tmp_path)
    if kind == "array":
        cls = ZarrImage
        ZarrImage(data=np.arange(6, dtype="float32").reshape(2, 3)).save(
            "z.zarr"
        )
    else:
        cls = OmeZarrImage
        _small_pyramid().save("z.zarr")
    expected = cls.load("z.zarr")

    assert cls.sniff_filename("z.zarr") > 0
    for back in (
        cls.from_filename("z.zarr"),
        cls.from_filename(Path("z.zarr")),
        images.ImageFormat.from_filename("z.zarr"),
    ):
        assert type(back) is cls
        if kind == "array":
            assert np.array_equal(
                np.asarray(back.data), np.asarray(expected.data)
            )
        else:
            assert back.nscales == expected.nscales
            for level in range(back.nscales):
                assert np.array_equal(
                    np.asarray(back.images[level].data),
                    np.asarray(expected.images[level].data),
                )


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


def test_opening_a_pyramid_does_not_read_its_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brainhops.io.common.zarr._data import ZarrProxy

    path = str(tmp_path / "lazy.zarr")
    _small_pyramid().save(path)
    reads = []
    read = ZarrProxy._read

    def recording(self, index):  # noqa: ANN001, ANN202
        values = read(self, index)
        reads.append(values.size)
        return values

    monkeypatch.setattr(ZarrProxy, "_read", recording)
    back = OmeZarrImage.from_store(path)

    # No level is read yet: the data of each level is a lazy array.
    for level in back.images:
        assert not isinstance(level.data, np.ndarray)
    assert reads == []
    # Reading one level reads only that level.
    _ = np.asarray(back.images[0].data)
    assert reads == [back.images[0].data.size]


def _placed_pyramid(path: str) -> OmeZarrImage:
    # A pyramid read from a store whose metadata holds two levels and one
    # placement.
    pyramid = _small_pyramid()
    pyramid.transformations = [Translation([5.0, 6.0, 7.0])]
    pyramid.save(path)
    return OmeZarrImage.from_store(path)


def test_a_read_pyramid_stores_no_levels_or_placements(
    tmp_path: Path,
) -> None:
    back = _placed_pyramid(str(tmp_path / "placed.zarr"))
    # Nothing is stored, so the levels and placements are read from the
    # node.
    assert back._images is None and back._transformations is None
    assert back.nscales == 2
    assert len(back.transformations) == 1
    # Assigning `None` reads them from the node again.
    back.images = None
    back.transformations = None
    assert back.nscales == 2
    assert len(back.transformations) == 1


def test_an_empty_list_of_levels_or_placements_is_used_as_it_is(
    tmp_path: Path,
) -> None:
    back = _placed_pyramid(str(tmp_path / "placed.zarr"))
    back.images = []
    back.transformations = []
    assert back.images == [] and back.nscales == 0
    assert back.transformations == []
    # The constructor keeps an empty list as well.
    built = OmeZarrImage(images=[], transformations=[], node=back.node)
    assert built.images == [] and built.transformations == []


def test_a_copy_of_a_read_pyramid_stays_lazy(tmp_path: Path) -> None:
    back = _placed_pyramid(str(tmp_path / "placed.zarr"))
    copy = replace(back)
    # The copy stores nothing either, and parses the metadata only when
    # its levels are first read.
    assert copy._images is None and copy._transformations is None
    assert not hasattr(copy, "_cache__layout")
    assert copy.nscales == 2
    assert len(copy.transformations) == 1
    # An empty list is copied as it is.
    back.transformations = []
    assert replace(back).transformations == []
    assert OmeZarrImage.from_instance(back).transformations == []


def test_a_copy_of_a_read_level_stays_lazy(tmp_path: Path) -> None:
    path = str(tmp_path / "lazy.zarr")
    _small_pyramid().save(path)
    level = OmeZarrImage.from_store(path).images[0]
    copy = replace(level)
    # The copy keeps the lazy array, which permutes the axes, and the
    # record, and reads no array.
    assert copy._data is None
    assert copy.rawdata is level.rawdata
    assert copy.metadata is level.metadata
    assert getattr(level, "_cache_data", None) is None
    assert np.array_equal(np.asarray(copy.data), np.asarray(level.data))
    # An assigned array is copied as it is.
    level.data = np.zeros(level.data.shape)
    assert replace(level).data is level.data


def test_a_copy_of_a_read_array_stays_lazy(tmp_path: Path) -> None:
    path = str(tmp_path / "plain.zarr")
    data = np.arange(24.0).reshape(2, 3, 4)
    ZarrImage(data).save(path)
    image = ZarrImage.from_store(path)
    copy = replace(image)
    assert copy._data is None
    assert getattr(image, "_cache_data", None) is None
    assert np.array_equal(np.asarray(copy.data), data)


def test_a_copy_from_a_data_model_pyramid_keeps_its_levels() -> None:
    model = MultiScaleImage(
        images=list(_small_pyramid().images),
        transformations=[Translation([5.0, 6.0, 7.0])],
    )
    pyramid = OmeZarrImage.from_instance(model)
    assert pyramid.images == model.images
    assert pyramid.transformations == model.transformations


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
    inner_transform: tx.Optional[dict] = None,
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
    transformations = (
        {
            "coordinateTransformations": [
                dict(
                    transform,
                    input={"name": "phys"},
                    output={"name": "model"},
                )
            ]
        }
        if transform
        else {}
    )
    inner_transform = inner_transform or {"type": "identity"}
    group.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {"name": "phys", "axes": axes},
                        {"name": "model", "axes": axes},
                    ],
                    **transformations,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                dict(
                                    inner_transform,
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
    geometry = OmeZarrImage.from_store(path).transformation
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
    geometry = OmeZarrImage.from_store(path).transformation
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
        {},
        _SPACE3,
        (6, 5, 4),
        inner_transform={
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": [3.0, 2.0, 1.0]},
                {"type": "translation", "translation": [30.0, 20.0, 10.0]},
            ],
        },
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
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, Permutation)
    np.testing.assert_array_equal(np.asarray(geometry.permutation), [1, 0])


def test_reader_refuses_a_map_axis_over_a_subset_of_the_axes() -> None:
    # A subset of the input axes was never a valid mapAxis, and the drop is
    # spelled projectAxis (spec/design/compute-api-and-open-systems.md,
    # section A.8); tested on the mapping, since such a level is not wired
    # through the reader. abczarr 0.2.3 already refuses it when it parses
    # it, with a value error.
    from abczarr.ome.v0_6.transformations import CoordinateTransformation

    from brainhops.io.images.zarr import _ome

    with pytest.raises((OmeImageError, ValueError)):
        transform = CoordinateTransformation.from_json(
            {"type": "mapAxis", "mapAxis": [0, 2]}
        )
        _ome._map_transform(transform, [0, 1, 2], [0, 1])


def test_reader_maps_a_project_axis_to_a_projection() -> None:
    # A projectAxis that drops an input axis is a Projection; tested on the
    # mapping, since such a level is not wired through the reader.
    from abczarr.ome.v0_6.transformations import CoordinateTransformation

    from brainhops.datamodel.transformations import Projection
    from brainhops.io.images.zarr import _ome

    transform = CoordinateTransformation.from_json(
        {"type": "projectAxis", "droppedInputs": [1]}
    )
    projection = _ome._map_transform(transform, [0, 1, 2], [0, 1])
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
                    "coordinateSystems": [
                        {"name": "phys", "axes": _SPACE3},
                        {"name": "model", "axes": _SPACE3},
                    ],
                    "coordinateTransformations": [
                        {
                            "type": "displacements",
                            "path": "disp",
                            "input": {"name": "phys"},
                            "output": {"name": "model"},
                        }
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "identity",
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
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, DisplacementField)
    result = np.asarray(geometry.field)
    # Read as (x, y, z, component). Component `i` belongs to the stored
    # output axis `i` (ngff-spec 0.6, "Vector component mapping"), so the
    # stored (z, y, x) components are reordered into (x, y, z).
    assert result.shape == (4, 5, 6, 3)
    np.testing.assert_allclose(result[0, 0, 0, :], [300.0, 200.0, 100.0])


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
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, DisplacementField)
    field = np.asarray(geometry.field)
    # The component axis moves to the end.
    assert field.shape == (4, 5, 6, 3)
    # The components follow the output axes into the (x, y, z) order.
    np.testing.assert_allclose(field[0, 0, 0, :], [300.0, 200.0, 100.0])


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
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, DisplacementField)
    result = np.asarray(geometry.field)
    assert result.shape == (4, 5, 6, 3)
    np.testing.assert_allclose(result[0, 0, 0, :], [300.0, 200.0, 100.0])


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
    from brainhops.datamodel.transformations import CoordinateField

    path = _authored_pyramid(
        tmp_path,
        {"type": "coordinates", "path": "coord"},
        _SPACE3,
        (6, 5, 4),
        arrays={"coord": (_field_array(), ("d", "z", "y", "x"))},
    )
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, CoordinateField)
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
    geometry = OmeZarrImage.from_store(path).transformation
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
    geometry = OmeZarrImage.from_store(path).transformation
    assert isinstance(geometry, Sequence)
    assert all(
        isinstance(part, DisplacementField)
        for part in geometry.transformations
    )


# ---- named coordinate systems of OME-NGFF 0.6 ------------------------


_UM = "micrometer"


def _space(names: str, unit: tx.Optional[str] = _UM) -> tx.List[dict]:
    axes = []
    for name in names:
        axis = {"name": name, "type": "space"}
        if unit is not None:
            axis["unit"] = unit
        axes.append(axis)
    return axes


def _named_pyramid(
    tmp_path: Path,
    systems: tx.List[dict],
    commons: tx.List[dict],
    arrays: tx.Optional[tx.Dict[str, np.ndarray]] = None,
    scale: tx.Sequence[float] = (1.0, 1.0, 1.0),
) -> str:
    """Write a 0.6 pyramid of one (z, y, x) level onto `intrinsic`, with
    other named systems and the common transformations that reach them.
    """
    from abczarr.ome import v0_6 as v6

    path = str(tmp_path / "named.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.ones((6, 5, 4), "float32"))
    for name, array in (arrays or {}).items():
        group.create_array(name, data=array)
    block = {
        "coordinateSystems": [{"name": "intrinsic", "axes": _space("zyx")}]
        + systems,
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": list(scale),
                        "input": {"path": "0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        "coordinateTransformations": commons,
    }
    group.ome = v6.OME.from_json({"version": "0.6", "multiscales": [block]})
    return path


@pytest.mark.parametrize("kind", ["scale", "translation"])
@pytest.mark.parametrize("nested", [False, True])
def test_reader_refuses_a_scale_or_translation_named_by_path(
    tmp_path: Path, kind: str, nested: bool
) -> None:
    # OME-NGFF 0.6 requires the values of a scale or a translation inline
    # (memo item 134), wherever the transformation sits.
    by_path = {"type": kind, "path": "params"}
    if nested:
        by_path = {
            "type": "sequence",
            "transformations": [{"type": "identity"}, by_path],
        }
    common = dict(
        by_path, input={"name": "intrinsic"}, output={"name": "moved"}
    )
    path = _named_pyramid(
        tmp_path,
        [{"name": "moved", "axes": _space("zyx")}],
        [common],
        arrays={"params": np.asarray([1.0, 2.0, 3.0])},
    )
    with pytest.raises(OmeImageError, match="inline"):
        OmeZarrImage.from_store(path)


def test_reader_reads_an_affine_named_by_path(tmp_path: Path) -> None:
    # An affine or a rotation may still name an array of parameters.
    matrix = np.eye(3, 4)
    matrix[:, 3] = [1.0, 2.0, 3.0]
    common = {
        "type": "affine",
        "path": "params",
        "input": {"name": "intrinsic"},
        "output": {"name": "moved"},
    }
    path = _named_pyramid(
        tmp_path,
        [{"name": "moved", "axes": _space("zyx")}],
        [common],
        arrays={"params": matrix},
    )
    placement = OmeZarrImage.from_store(path).transformation
    # The stored (z, y, x) translation is reordered to (x, y, z).
    np.testing.assert_allclose(_world_matrix(placement)[:, 3], [3.0, 2.0, 1.0])


def test_reader_names_each_system_with_its_own_axes_and_units(
    tmp_path: Path,
) -> None:
    # The world system lists its axes as (x, y, z) and in millimeters, and
    # the mapAxis only spells out that reordering.
    path = _named_pyramid(
        tmp_path,
        [{"name": "world", "axes": _space("xyz", "millimeter")}],
        [
            {
                "type": "mapAxis",
                "mapAxis": [2, 1, 0],
                "input": {"name": "intrinsic"},
                "output": {"name": "world"},
            }
        ],
    )
    pyramid = OmeZarrImage.from_store(path)
    (placement,) = pyramid.transformations
    assert placement.input.name == "intrinsic"
    assert placement.output.name == "world"
    assert [a.name for a in placement.input.axes] == ["x", "y", "z"]
    assert [str(a.unit) for a in placement.input.axes] == [_UM] * 3
    assert [a.name for a in placement.output.axes] == ["x", "y", "z"]
    assert [str(a.unit) for a in placement.output.axes] == ["millimeter"] * 3
    np.testing.assert_allclose(_world_matrix(placement), np.eye(3, 4))


def test_reader_keeps_a_system_with_fewer_axes(tmp_path: Path) -> None:
    # The plane system is reached by dropping z, so it has two axes.
    path = _named_pyramid(
        tmp_path,
        [{"name": "plane", "axes": _space("yx")}],
        [
            {
                "type": "projectAxis",
                "droppedInputs": [0],
                "input": {"name": "intrinsic"},
                "output": {"name": "plane"},
            }
        ],
    )
    (placement,) = OmeZarrImage.from_store(path).transformations
    assert [a.name for a in placement.output.axes] == ["x", "y"]


def test_reader_inverts_a_transformation_into_the_intrinsic_space(
    tmp_path: Path,
) -> None:
    path = _named_pyramid(
        tmp_path,
        [{"name": "world", "axes": _space("zyx")}],
        [
            {
                "type": "scale",
                "scale": [2.0, 4.0, 8.0],
                "input": {"name": "world"},
                "output": {"name": "intrinsic"},
            }
        ],
    )
    (placement,) = OmeZarrImage.from_store(path).transformations
    assert placement.output.name == "world"
    np.testing.assert_allclose(
        np.diag(_world_matrix(placement)[:, :3]), [1 / 8, 1 / 4, 1 / 2]
    )


def test_reader_reads_parameters_stored_in_an_array(tmp_path: Path) -> None:
    path = _named_pyramid(
        tmp_path,
        [{"name": "world", "axes": _space("zyx")}],
        [
            {
                "type": "affine",
                "path": "params/affine",
                "input": {"name": "intrinsic"},
                "output": {"name": "world"},
            }
        ],
        arrays={
            "params/affine": np.asarray(
                [[3.0, 0, 0, 30], [0, 2.0, 0, 20], [0, 0, 1.0, 10]]
            )
        },
    )
    (placement,) = OmeZarrImage.from_store(path).transformations
    np.testing.assert_allclose(
        _world_matrix(placement),
        [[1.0, 0, 0, 10], [0, 2.0, 0, 20], [0, 0, 3.0, 30]],
    )


def test_writer_refuses_a_placement_that_is_not_affine(
    tmp_path: Path,
) -> None:
    path = _named_pyramid(
        tmp_path,
        [{"name": "plane", "axes": _space("yx")}],
        [
            {
                "type": "projectAxis",
                "droppedInputs": [0],
                "input": {"name": "intrinsic"},
                "output": {"name": "plane"},
            }
        ],
    )
    pyramid = OmeZarrImage.from_store(path)
    with pytest.raises(WriterError):
        pyramid.save(str(tmp_path / "resaved.zarr"))


def test_writer_writes_an_inverted_placement_forward(tmp_path: Path) -> None:
    path = _named_pyramid(
        tmp_path,
        [{"name": "world", "axes": _space("zyx")}],
        [
            {
                "type": "scale",
                "scale": [2.0, 4.0, 8.0],
                "input": {"name": "world"},
                "output": {"name": "intrinsic"},
            }
        ],
    )
    target = str(tmp_path / "resaved.zarr")
    OmeZarrImage.from_store(path).save(target)
    block = abczarr.open(target, mode="r").ome.to_json()["multiscales"][0]
    (common,) = block["coordinateTransformations"]
    assert common["output"] == {"name": "world"}
    np.testing.assert_allclose(common["scale"], [0.5, 0.25, 0.125])


def _displacement_group(
    group: tx.Any,
    path: str,
    vector: tx.List[float],
    kind: str = "displacement",
) -> None:
    """Write a 0.6 field group with a constant vector.

    The vector axis comes before the spatial axes, as OME-NGFF 0.6
    requires, and the field is sampled every two units. `kind` is the type
    of the vector axis, `"displacement"` or `"coordinate"`.
    """
    from abczarr.ome import v0_6 as v6

    field = np.zeros((len(vector), 3, 3, 2), dtype="float32")
    for component, value in enumerate(vector):
        field[component] = value
    node = group.create_group(path)
    node.create_array("0", data=field)
    axes = [{"name": "d", "type": kind, "discrete": True}]
    node.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {"name": "intrinsic", "axes": axes + _space("zyx")}
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [1.0, 2.0, 2.0, 2.0],
                                    "input": {"path": "0"},
                                    "output": {"name": "intrinsic"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_reader_reads_a_displacement_field_group_of_version_0_6(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.transformations import (
        CoordinateField,
        Permutation,
    )

    path = _named_pyramid(
        tmp_path,
        [{"name": "warped", "axes": _space("zyx")}],
        [
            {
                "type": "displacements",
                "path": "fields/disp",
                "input": {"name": "intrinsic"},
                "output": {"name": "warped"},
            }
        ],
    )
    # The stored components are (dz, dy, dx).
    _displacement_group(
        abczarr.open_group(path, mode="r+"), "fields/disp", [1.0, 2.0, 3.0]
    )
    (placement,) = OmeZarrImage.from_store(path).transformations
    first, field, last = placement.transformations
    assert isinstance(first, Permutation)
    assert isinstance(last, Permutation)
    # A point is carried by (dx, dy, dz) = (3, 2, 1), in brainhops order.
    points = np.asarray([[[[0.5, 1.0, 1.5]]]])
    moved = placement(CoordinateField(field=points), compute=True)
    np.testing.assert_allclose(
        np.asarray(moved.field)[0, 0, 0], [3.5, 3.0, 2.5]
    )


# ---- base Zarr writer ------------------------------------------------


def _base_writer_class() -> tx.Type[tx.Any]:
    from brainhops.io.common.zarr._parsers import ZarrReaderWriter

    class _Writer(ZarrReaderWriter):
        pass

    return _Writer


def test_the_base_writer_holds_no_state() -> None:
    # The adapter adds methods and no field, so it takes no argument.
    with pytest.raises(TypeError):
        _base_writer_class()(node=None)
    assert vars(_base_writer_class()()) == {}


def test_the_base_writer_leaves_writing_to_the_format(
    tmp_path: Path,
) -> None:
    writer = _base_writer_class()()
    with pytest.raises(NotImplementedError):
        writer.to_node(None)
    with pytest.raises(NotImplementedError):
        writer.to_store(str(tmp_path / "out.zarr"))
    assert not (tmp_path / "out.zarr").exists()


def _field_pyramid(
    tmp_path: Path,
    kind: str,
    vector: tx.List[float],
    interpolation: tx.Optional[str] = None,
) -> str:
    """Write a pyramid placed in `warped` by a 0.6 field group."""
    common = {
        "type": kind + "s",
        "path": "fields/f",
        "input": {"name": "intrinsic"},
        "output": {"name": "warped"},
    }
    if interpolation is not None:
        common["interpolation"] = interpolation
    path = _named_pyramid(
        tmp_path, [{"name": "warped", "axes": _space("zyx")}], [common]
    )
    group = abczarr.open_group(path, mode="r+")
    _displacement_group(group, "fields/f", vector, kind)
    return path


@pytest.mark.parametrize(
    ("interpolation", "degree"),
    [("nearest", 0), ("linear", 1), ("bspline-cubic", 3), (None, 1)],
)
def test_reader_reads_the_interpolation_of_a_field_group(
    tmp_path: Path, interpolation: tx.Optional[str], degree: int
) -> None:
    from brainhops.datamodel.transformations import DisplacementField

    path = _field_pyramid(
        tmp_path, "displacement", [1.0, 2.0, 3.0], interpolation
    )
    (placement,) = OmeZarrImage.from_store(path).transformations
    field = placement.transformations[1]
    (vectors,) = [
        part
        for part in field.transformations
        if isinstance(part, DisplacementField)
    ]
    assert int(vectors.degree) == degree


def test_reader_refuses_an_unknown_interpolation(tmp_path: Path) -> None:
    path = _field_pyramid(
        tmp_path, "displacement", [1.0, 2.0, 3.0], "windowed-sinc"
    )
    with pytest.raises(OmeImageError, match="windowed-sinc"):
        OmeZarrImage.from_store(path)


def test_reader_refuses_coordinates_of_another_length(
    tmp_path: Path,
) -> None:
    # The field has two components, but the output system has three axes.
    path = _field_pyramid(tmp_path, "coordinate", [1.0, 2.0])
    with pytest.raises(OmeImageError, match="components"):
        OmeZarrImage.from_store(path)


def test_the_base_writer_refuses_streams_and_bytes() -> None:
    from io import BytesIO

    writer = _base_writer_class()()
    with pytest.raises(WriterError):
        writer.to_fileobj(BytesIO())
    with pytest.raises(WriterError):
        writer.to_bytes()


# ---- saving an image built in memory ---------------------------------


def test_save_writes_a_single_scale_image_as_ome_zarr(
    tmp_path: Path,
) -> None:
    from brainhops import io

    data = np.arange(60, dtype="float32").reshape(3, 4, 5)
    affine = _diag_affine([2, 3, 4], [1, 2, 3])
    image = SingleScaleImage(data=data, transformations=[affine])
    path = str(tmp_path / "img.ome.zarr")

    io.save(image, path)
    back = images.load(path)

    assert isinstance(back, OmeZarrImage)
    assert back.nscales == 1
    assert np.array_equal(np.asarray(back.images[0].data), data)
    np.testing.assert_allclose(
        _world_matrix(back.images[0].transformation), _world_matrix(affine)
    )


def test_save_writes_a_multiscale_image_as_ome_zarr(tmp_path: Path) -> None:
    from brainhops import io

    pyramid = _small_pyramid()
    image = MultiScaleImage(images=list(pyramid.images))
    path = str(tmp_path / "img.ome.zarr")

    io.save(image, path)

    assert images.load(path).nscales == 2


def test_save_refuses_an_image_with_a_geometry_in_plain_zarr(
    tmp_path: Path,
) -> None:
    from brainhops import io

    image = SingleScaleImage(
        data=np.zeros((3, 4, 5), "float32"),
        transformations=[_diag_affine([2, 2, 2], [0, 0, 0])],
    )

    with pytest.raises(WriterError):
        io.save(image, str(tmp_path / "img.zarr"))


# ---- axes written in full --------------------------------------------


def _written_systems(path: str) -> tx.List[tx.Dict[str, tx.Any]]:
    attrs = dict(abczarr.open(path, mode="r").attrs)
    return attrs["ome"]["multiscales"][0]["coordinateSystems"]


def test_writer_takes_the_units_of_the_world_of_the_geometry(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.systems import RASmm

    affine = replace(_diag_affine([2, 2, 2], [0, 0, 0]), output=RASmm())
    image = SingleScaleImage(
        np.zeros((3, 4, 5), "float32"), transformations=[affine]
    )
    path = str(tmp_path / "img.ome.zarr")

    OmeZarrImage(images=[image]).save(path)

    for system in _written_systems(path):
        assert system["axes"] == [
            {"name": name, "type": "space", "unit": "millimeter"}
            for name in ("z", "y", "x")
        ]


def test_writer_leaves_out_a_unit_that_is_not_known(tmp_path: Path) -> None:
    image = SingleScaleImage(
        np.zeros((3, 4, 5), "float32"),
        transformations=[_diag_affine([2, 2, 2], [0, 0, 0])],
    )
    path = str(tmp_path / "img.ome.zarr")

    OmeZarrImage(images=[image]).save(path)

    for system in _written_systems(path):
        assert system["axes"] == [
            {"name": name, "type": "space"} for name in ("z", "y", "x")
        ]


def test_writer_writes_the_unit_and_discreteness_of_its_axes(
    tmp_path: Path,
) -> None:
    axes = [
        SpaceAxis(name="x", unit="micrometer"),
        SpaceAxis(name="y", unit="micrometer"),
        SpaceAxis(name="z", unit="micrometer"),
        ChannelAxis(name="c", discrete=True),
    ]
    image = SingleScaleImage(
        np.zeros((3, 4, 5, 2), "float32"),
        transformations=[_diag_affine([2, 2, 2, 1], [0, 0, 0, 0])],
    )
    path = str(tmp_path / "img.ome.zarr")

    OmeZarrImage(images=[image], axes=axes).save(path)

    written = _written_systems(path)[0]["axes"]
    assert written[0] == {"name": "c", "type": "channel", "discrete": True}
    assert written[1:] == [
        {"name": name, "type": "space", "unit": "micrometer"}
        for name in ("z", "y", "x")
    ]


# ---- checking the metadata before it is written ----------------------


def _ome(
    systems: tx.List[str],
    placement: tx.Dict[str, tx.Any],
    commons: tx.Sequence[tx.Dict[str, tx.Any]] = (),
) -> tx.Any:
    from abczarr.ome import v0_6

    axes = [{"name": name, "type": "space"} for name in ("z", "y", "x")]
    block = {
        "coordinateSystems": [{"name": s, "axes": axes} for s in systems],
        "datasets": [{"path": "0", "coordinateTransformations": [placement]}],
    }  # type: tx.Dict[str, tx.Any]
    if commons:
        block["coordinateTransformations"] = list(commons)
    return v0_6.OME.from_json({"version": "0.6", "multiscales": [block]})


_SCALE = {
    "type": "scale",
    "scale": [1.0, 1.0, 1.0],
    "input": {"path": "0"},
    "output": {"name": "physical"},
}


def test_check_accepts_the_metadata_of_the_writer() -> None:
    from brainhops.io.common.zarr._ome_rules import check_ome

    check_ome(_ome(["physical"], _SCALE))


def test_check_refuses_a_dataset_placed_by_a_translation_then_a_scale() -> (
    None
):
    from brainhops.io.common.zarr._ome_rules import check_ome

    sequence = {
        "type": "sequence",
        "input": {"path": "0"},
        "output": {"name": "physical"},
        "transformations": [
            {"type": "translation", "translation": [1.0, 2.0, 3.0]},
            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
        ],
    }
    with pytest.raises(WriterError, match="dataset"):
        check_ome(_ome(["physical"], sequence))


def test_check_refuses_two_systems_of_the_same_name() -> None:
    from brainhops.io.common.zarr._ome_rules import check_ome

    with pytest.raises(WriterError, match="several coordinate systems"):
        check_ome(_ome(["physical", "physical"], _SCALE))


def test_check_refuses_an_output_that_names_no_system() -> None:
    from brainhops.io.common.zarr._ome_rules import check_ome

    placement = dict(_SCALE, output={"name": "elsewhere"})
    with pytest.raises(WriterError, match="names neither"):
        check_ome(_ome(["physical"], placement))


def test_check_refuses_a_common_input_that_names_no_system() -> None:
    from brainhops.io.common.zarr._ome_rules import check_ome

    common = {
        "type": "identity",
        "input": {"name": "nowhere"},
        "output": {"name": "physical"},
    }
    with pytest.raises(WriterError, match="names neither"):
        check_ome(_ome(["physical"], _SCALE, [common]))


def test_writer_checks_the_metadata_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A writer that would break a rule refuses to write.
    from brainhops.io.images.zarr import _ome

    def build(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
        return _ome_with_two_physical_systems()

    monkeypatch.setattr(_ome, "build_ome", build)
    image = SingleScaleImage(np.zeros((3, 4, 5), "float32"))
    with pytest.raises(WriterError, match="several coordinate systems"):
        OmeZarrImage(images=[image]).save(str(tmp_path / "img.ome.zarr"))


def _ome_with_two_physical_systems() -> tx.Any:
    return _ome(["physical", "physical"], _SCALE)


# ---- names of the axes that have none --------------------------------


def _written_axes(tmp_path: Path, axes: tx.List[tx.Any], ndim: int) -> tx.Any:
    shape = (2,) * ndim
    image = SingleScaleImage(np.zeros(shape, "float32"))
    path = str(tmp_path / "img.ome.zarr")
    OmeZarrImage(images=[image], axes=axes).save(path)
    return _written_systems(path)[0]["axes"]


def test_writer_gives_open_axes_the_conventional_names(
    tmp_path: Path,
) -> None:
    axes = _written_axes(tmp_path, [*_spatial_axes(), ...], 5)

    assert axes[:2] == [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
    ]
    assert [axis["name"] for axis in axes[2:]] == ["z", "y", "x"]


def test_writer_skips_a_conventional_name_that_is_taken(
    tmp_path: Path,
) -> None:
    given = [*_spatial_axes(), TimeAxis(name="t"), ...]
    axes = _written_axes(tmp_path, given, 5)

    names = [axis["name"] for axis in axes]
    assert sorted(names) == ["c", "t", "x", "y", "z"]
    assert {"name": "c", "type": "channel"} in axes


def test_axes_beyond_the_conventions_are_named_by_position() -> None:
    # The schema of OME-NGFF 0.6 holds at most five axes, so the fallback
    # is checked on the helper that names the axes.
    from brainhops.io.images.zarr._ome import name_undescribed_axes

    axes = [{"name": ""} for _ in range(3)] + [
        {"name": name, "type": "space"} for name in ("z", "y", "x")
    ]
    name_undescribed_axes(axes)

    assert axes[:3] == [
        {"name": "dim0"},
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
    ]
