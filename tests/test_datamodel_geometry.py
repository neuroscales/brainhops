"""Tests of `Geometry`, which pairs a grid with a voxel-to-world transform.

Flattening and computing must keep the pair, and must never rebuild a
geometry with more than its two elements.
"""

import numpy as np
import pytest

from brainhops.backends import backend
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.geometry import Geometry, _index2transform
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import (
    CoordinateSystem,
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    Sequence,
)


def _grid() -> CartesianField:
    return CartesianField(shape=(2, 3, 4))


def _affine(scale: float = 2.0) -> Affine:
    return Affine(matrix=np.diag([scale, scale, scale, 1.0])[:-1])


def test_geometry_with_nested_sequence_computes_to_two_element_geometry() -> (
    None
):
    inner = Sequence(transformations=[_affine(2.0), _affine(0.5)])
    geom = Geometry((_grid(), inner))

    computed = geom.compute()

    # The result is a (grid, transformation) pair, with the transformation
    #
    # simplified.
    assert isinstance(computed, Geometry)
    assert len(computed.transformations) == 2
    grid, transformation = computed.transformations
    assert isinstance(grid, CartesianField)
    assert grid.shape == (2, 3, 4)
    assert not isinstance(transformation, Sequence)


def test_geometry_flatten_preserves_the_pair_and_the_grid() -> None:
    # A doubly nested transformation still flattens into the single
    #
    # transformation slot.
    inner = Sequence(
        transformations=[Sequence(transformations=[_affine(), _affine()])]
    )
    geom = Geometry((_grid(), inner))

    flat = geom._flattened()

    assert isinstance(flat, Geometry)
    assert len(flat.transformations) == 2
    assert isinstance(flat.transformations[0], CartesianField)
    assert flat.transformations[0].shape == (2, 3, 4)


def test_geometry_flatten_propagates_endpoints() -> None:
    # Flattening moves the input of the geometry onto the grid and its output
    #
    # onto the transformation.
    voxel = VoxelCoordinateSystem()
    ras = RASCoordinateSystem()
    nested = Sequence(transformations=[_affine()])
    geom = Geometry((_grid(), nested), input=voxel, output=ras)

    flat = geom._flattened()

    assert flat.grid.input is voxel
    assert flat.transformation.output is ras


def test_composing_a_sequence_onto_a_geometry_keeps_a_geometry() -> None:
    # `sequence @ geometry` gives a geometry whose transformation is a nested
    #
    # sequence, which `compute` must simplify while keeping the pair.
    grid = CartesianField(
        shape=(2, 3, 4),
        input=VoxelCoordinateSystem(),
        output=VoxelCoordinateSystem(),
    )
    voxel_to_world = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
        input=VoxelCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    geom = Geometry((grid, Sequence(transformations=[voxel_to_world])))
    world_to_voxel = Sequence(transformations=[voxel_to_world.inverse()])

    composed = world_to_voxel @ geom

    assert isinstance(composed, Geometry)
    computed = composed.compute()
    assert isinstance(computed, Geometry)
    assert len(computed.transformations) == 2
    assert isinstance(computed.transformations[0], CartesianField)


def test_nesting_a_geometry_in_a_sequence_computes_to_a_field() -> None:
    # A geometry inside a sequence is flattened, and its grid is sampled
    #
    # rather than folded away.
    grid = CartesianField(shape=(2, 3, 4))
    geom = Geometry((grid, _affine(2.0)))

    result = Sequence(transformations=[geom, _affine(0.5)]).compute()

    assert result.field is not None
    assert result.field.shape == (2, 3, 4, 3)
    # Scaling by 2 and then by 1/2 gives back the grid coordinates.
    np.testing.assert_allclose(result.field, np.asarray(grid.field))


# ----------------------------------------------------------------------
#   SUB-ARRAY INDEXING (_index2transform)
# ----------------------------------------------------------------------


def test_index2transform_dropped_axis_is_a_pure_translation() -> None:
    # An integer index drops an axis, which becomes a constant offset in its
    #
    # row, without the diagonal 1 of the identity.
    transform, shape = _index2transform(
        (slice(None), 2, slice(None)), (4, 5, 6)
    )
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(np.asarray(transform.matrix), expected)
    assert shape == (4, 6)


def test_index2transform_inserted_axis_has_no_spurious_diagonal() -> None:
    # A None index inserts an axis, and later rows must not get a stray
    #
    # diagonal 1 in the inserted column.
    transform, shape = _index2transform(
        (slice(None), None, slice(None), slice(None)), (4, 5, 6)
    )
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(np.asarray(transform.matrix), expected)
    assert shape == (4, 1, 5, 6)


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize(
    "index",
    [
        (slice(None), 2, slice(None)),
        (1, slice(None), slice(None)),
        (slice(None), None, slice(None), slice(None)),
        (slice(None, None, -1), 3, slice(1, 5)),
    ],
)
def test_sub_geometry_reslice_reproduces_numpy_indexing(
    index: tuple, degree: int
) -> None:
    # With an identity affine, the result depends only on the affine of the
    #
    # index, so an error there would read the wrong voxels.
    system = CoordinateSystem(
        name="voxel",
        axes=[SpaceAxis(name=name, unit="index") for name in "xyz"],
    )
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    img = SingleScaleImage(
        data=data,
        transformations=[
            Affine(matrix=np.eye(3, 4), input=system, output=system)
        ],
    )
    geometry = img.geometry[index]
    with backend("numpy"):
        got = img.reslice(geometry, degree=degree, bound="reflect").data
    assert np.array_equal(np.asarray(got), data[index])
