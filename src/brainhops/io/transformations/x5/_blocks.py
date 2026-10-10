"""Transformations that X5 nodes decode into and encode from.

The rules are described in [`brainhops.io.transformations.x5`][].
"""

__all__ = [
    "X5BSplineField",
    "X5CoordinatesField",
    "X5DisplacementField",
    "node_to_transformation",
    "transformation_to_nodes",
]

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core import affines as _affines
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserNotImplementedError,
    UnrepresentableTransformationError,
)
from brainhops.io.common.hdf5 import DelayedH5Array
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

from ._raw import X5Domain, X5Node

_NDIM = 3
"""The number of spatial dimensions, which is 3 because X5 worlds are RAS."""

DISPLACEMENTS = ("displacements", "displacement", "deltas", "relative")
"""The `Representation` values of relative displacement fields."""

COORDINATES = (
    "deformations",
    "deformation",
    "coordinates",
    "absolute",
)
"""The `Representation` values of absolute coordinate fields."""

COEFFICIENTS = (None, "coefficients", "coefficient")
"""The `Representation` values of B-spline coefficients."""

_DENSE_SUBTYPES = (None, "densefield", "dense")
"""The `SubType` values of dense fields sampled on the `Domain`."""

_BSPLINE_SUBTYPES = ("bspline", "b-spline")
"""The `SubType` values of B-spline fields."""

_BSPLINE_DEGREE = 3
"""The degree of X5 B-splines, which is 3 because nitransforms only
evaluates cubic B-splines."""

_KINDS = ("space",) * _NDIM + ("vector",)


# ----------------------------------------------------------------------
#   FIELDS
# ----------------------------------------------------------------------


class _X5RASDisplacements(_xforms.ImmutableSequence):
    """A RAS displacement field on a voxel grid, modelled as a chain."""

    degree: tx.ClassVar[int] = 1
    """The degree of the spline that interpolates the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition outside the field of view."""

    store: tx.ClassVar[StoreEnum] = StoreEnum.values
    """Whether the field holds spline coefficients or values."""

    @classmethod
    def from_ras(cls, vectors: ArrayProtocol, vox2ras: np.ndarray) -> tx.Self:
        """Build the chain from RAS displacements and the grid's affine."""
        return cls(
            transformations=ras_displacement_chain(
                vectors,
                vox2ras,
                degree=cls.degree,
                bound=cls.bound,
                store=cls.store,
            )
        )

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The map from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in voxel units."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The map from the field's voxels to RAS world coordinates."""
        return self.transformations[2]


class X5DisplacementField(_X5RASDisplacements):
    """A nonlinear X5 transform that holds relative displacements.

    Each sample is the displacement, in RAS millimetres, of the world point at
    the centre of its voxel, on the grid placed by `Domain/Mapping`.

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |
    """


class X5BSplineField(_X5RASDisplacements):
    """A nonlinear X5 transform of cubic B-spline coefficients.

    Each coefficient is a displacement in RAS millimetres at a knot of a
    regular grid, placed by the node's `AdditionalParameters`. The transform
    maps `x` to `x + sum_k c_k B3(i(x) - k)`, where `i(x)` are the knot-grid
    coordinates of `x` and `B3` is the centred cubic B-spline.

    | Slot           | Transformation                                   |
    | -------------- | ------------------------------------------------ |
    | `ras2voxel`    | RAS world coordinates to the knot grid           |
    | `displacement` | the coefficients, in knot-grid units            |
    | `voxel2ras`    | the knot grid back to RAS world                  |
    """

    degree: tx.ClassVar[int] = _BSPLINE_DEGREE
    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.zeros
    store: tx.ClassVar[StoreEnum] = StoreEnum.coefficients


class X5CoordinatesField(_xforms.ImmutableSequence):
    """A nonlinear X5 transform that holds absolute coordinates.

    Each sample holds the RAS coordinates, in millimetres, to which the world
    point at the centre of its voxel is mapped, on the grid placed by
    `Domain/Mapping`.

    | Slot          | Transformation                              |
    | ------------- | ------------------------------------------- |
    | `ras2voxel`   | RAS world coordinates to the field's voxels |
    | `coordinates` | the field of RAS coordinates                |
    """

    @classmethod
    def from_ras(
        cls, coordinates: ArrayProtocol, vox2ras: np.ndarray
    ) -> tx.Self:
        """Build the chain from RAS coordinates and the grid's affine."""
        vox2ras = np.asarray(vox2ras, dtype=np.float64)
        compact = vox2ras[:-1]
        return cls(
            (
                RASToVoxel(matrix=_affines.inv(compact)),
                RASCoordinatesField(field=coordinates),
            )
        )

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The map from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def coordinates(self) -> _xforms.Transformation:
        """The field of RAS coordinates, sampled on the field's voxels."""
        return self.transformations[1]


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------
# A node is decoded from the arrays of the record without changing them,
# and the arrays that a transformation receives are read-only. The writer
# writes the nodes of the record, not the decoded transformations, so an
# edit in place would be lost, and it would also change the record, which
# other objects may share. A new chain is assigned instead.


def _read(stored: tx.Any) -> ArrayProtocol:
    """Return an array of a node, read from the file if it is a proxy.

    A [`DelayedH5Array`][brainhops.io.common.hdf5.DelayedH5Array] is read
    with the array backend, which loads it with NumPy and keeps it lazy
    with Dask. A NumPy array is returned as a read-only view.
    """
    if isinstance(stored, DelayedH5Array):
        array = get_array_backend().asarray(stored)
    else:
        array = get_array_backend(stored).asarray(stored)
    if isinstance(array, np.ndarray):
        array = array.view()
        array.flags.writeable = False
    return array


def node_to_transformation(node: X5Node) -> _xforms.Transformation:
    """Decode an X5 node into a transformation from RAS to RAS.

    Raises
    ------
    ParserNotImplementedError
        If the node is valid but cannot be represented, as is the case for
        a `composite` transform, a stack of affines, or a field that is not
        on a regular 3-D grid.
    ParserContentError
        If the node is malformed or of an unknown `Type`.
    """
    if node.type == "linear":
        return _decode_linear(node)
    if node.type == "nonlinear":
        return _decode_nonlinear(node)
    if node.type == "composite":
        raise ParserNotImplementedError(
            "The X5 draft lists a 'composite' transform type, but does "
            "not specify how its parts are stored, and neither "
            "nitransforms nor fslpy writes one. Chains of transforms are "
            "read from /TransformChain instead."
        )
    raise ParserContentError(f"Unknown X5 transform type: {node.type!r}.")


def _decode_linear(node: X5Node) -> _xforms.Affine:
    matrix = np.asarray(_read(node.transform), dtype=np.float64)
    if int(node.array_length) != 1 or matrix.ndim != 2:
        raise ParserNotImplementedError(
            f"This X5 node stacks {node.array_length} affines (one per "
            f"volume of a series), with shape {matrix.shape}. brainhops "
            f"has no transformation that holds a stack of transformations."
        )
    if matrix.shape != (_NDIM + 1, _NDIM + 1):
        raise ParserNotImplementedError(
            f"Only 3-D linear X5 transforms, stored as 4x4 matrices, are "
            f"supported, not a matrix of shape {matrix.shape}."
        )
    if not np.allclose(matrix[-1], [0, 0, 0, 1]):
        raise ParserContentError(
            f"The last row of a linear X5 transform must be [0, 0, 0, 1], "
            f"not {matrix[-1].tolist()}."
        )
    return _xforms.Affine(
        matrix=matrix[:-1],
        input=_systems.RASmm(),
        output=_systems.RASmm(),
    )


def _decode_nonlinear(node: X5Node) -> _xforms.Transformation:
    subtype = node.subtype.lower() if node.subtype else node.subtype
    if subtype in _BSPLINE_SUBTYPES:
        return _decode_bspline(node)
    if subtype not in _DENSE_SUBTYPES:
        raise ParserNotImplementedError(
            f"X5 nonlinear transforms of subtype {node.subtype!r} are not "
            f"supported: only dense fields ('densefield') and B-splines "
            f"('bspline') are."
        )
    domain = node.domain
    if domain is None or domain.mapping is None:
        raise ParserContentError(
            "A nonlinear X5 transform must have a Domain with a Mapping."
        )
    if not domain.grid:
        raise ParserNotImplementedError(
            "Only nonlinear X5 transforms sampled on a regular grid "
            "(Domain/Grid = 1) are supported."
        )
    if domain.coordinates not in (None, "cartesian"):
        raise ParserNotImplementedError(
            f"Only cartesian X5 domains are supported, not "
            f"{domain.coordinates!r}."
        )
    vox2ras = np.asarray(_read(domain.mapping), dtype=np.float64)
    if vox2ras.shape != (_NDIM + 1, _NDIM + 1):
        raise ParserNotImplementedError(
            f"Only 3-D X5 domains, mapped by a 4x4 affine, are supported, "
            f"not one mapped by an affine of shape {vox2ras.shape}."
        )
    field = _vector_last(node)
    shape = tuple(int(s) for s in field.shape)
    if len(shape) != _NDIM + 1 or shape[-1] != _NDIM:
        raise ParserContentError(
            f"A 3-D X5 field holds one 3-vector per voxel, not an array "
            f"of shape {shape}."
        )
    if domain.size and tuple(domain.size)[:_NDIM] != shape[:-1]:
        raise ParserContentError(
            f"The X5 field has shape {shape[:-1]}, and its Domain says "
            f"{tuple(domain.size)}."
        )
    representation = (node.representation or "").lower()
    if representation in DISPLACEMENTS:
        return _read_only_field(X5DisplacementField.from_ras(field, vox2ras))
    if representation in COORDINATES:
        return X5CoordinatesField.from_ras(field, vox2ras)
    raise ParserContentError(
        f"A nonlinear X5 transform must say whether it stores "
        f"displacements or coordinates, and its Representation is "
        f"{node.representation!r}."
    )


def _decode_bspline(node: X5Node) -> X5BSplineField:
    """Decode a B-spline node as nitransforms writes it.

    The `Domain`, which nitransforms only uses to sample the field on the
    reference grid, is ignored.
    """
    representation = node.representation
    if representation is not None:
        representation = representation.lower()
    if representation not in COEFFICIENTS:
        raise ParserNotImplementedError(
            f"An X5 B-spline stores 'coefficients', and this one says its "
            f"Representation is {node.representation!r}."
        )
    if node.additional_parameters is None:
        raise ParserContentError(
            "An X5 B-spline must store the voxel-to-RAS affine of its "
            "grid of knots as AdditionalParameters."
        )
    knots = np.asarray(_read(node.additional_parameters), dtype=np.float64)
    if knots.shape != (_NDIM + 1, _NDIM + 1):
        raise ParserNotImplementedError(
            f"Only 3-D X5 B-splines, whose knots are placed by a 4x4 "
            f"affine, are supported, not one placed by an array of shape "
            f"{knots.shape}."
        )
    if not np.allclose(knots[-1], [0, 0, 0, 1]):
        raise ParserContentError(
            f"The last row of the affine of an X5 B-spline's knots must be "
            f"[0, 0, 0, 1], not {knots[-1].tolist()}."
        )
    field = _vector_last(node)
    shape = tuple(int(s) for s in field.shape)
    if len(shape) != _NDIM + 1 or shape[-1] != _NDIM:
        raise ParserContentError(
            f"A 3-D X5 B-spline holds one 3-vector per knot, not an array "
            f"of shape {shape}."
        )
    return _read_only_field(X5BSplineField.from_ras(field, knots))


def _read_only_field(xform: _X5RASDisplacements) -> _X5RASDisplacements:
    """Make the vectors that a field computed from its node read-only.

    The vectors in voxel units are a new array rather than a view of the
    record, but the writer writes the node, so an edit in place would
    still be lost.
    """
    vectors = xform.displacement.data
    if isinstance(vectors, np.ndarray):
        vectors.flags.writeable = False
    return xform


def _field_to_model(
    stored: tx.Any, kinds: tx.Optional[tx.Sequence[str]]
) -> ArrayProtocol:
    """Return a field with its `vector` axis last, from the stored array.

    `kinds` is the `DimensionKinds` dataset of the node, which says what
    each axis of the stored array holds. When it places the vector axis
    elsewhere, that axis is moved last. The stored array is read with
    `_read`.
    """
    field = _read(stored)
    if kinds and len(kinds) == field.ndim and "vector" in kinds:
        axis = list(kinds).index("vector")
        if axis != field.ndim - 1:
            field = get_array_backend(field).moveaxis(field, axis, -1)
    return field


def _field_to_disk(field: ArrayProtocol) -> ArrayProtocol:
    """Return the array that a node stores for a field.

    The writer declares the vector axis last in `DimensionKinds`, so the
    field, whose vector axis is last, is stored as it is. With these
    kinds, `_field_to_model` returns the field unchanged.
    """
    return field


def _vector_last(node: X5Node) -> ArrayProtocol:
    """Return the node's field with its `vector` axis moved last."""
    return _field_to_model(node.transform, node.dimension_kinds)


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def _name(system: tx.Optional[_systems.CoordinateSystem]) -> str:
    if name := getattr(system, "name", None):
        return name
    if system is not None:
        return type(system).__name__
    return "an unspecified system"


def _check_ras(xform: _xforms.Transformation, what: str) -> None:
    src, dst = xform.input, xform.output
    if isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm):
        return
    raise UnrepresentableTransformationError(
        f"X5 stores transformations from RAS to RAS world coordinates, "
        f"and this {what} maps {_name(src)} to {_name(dst)}. Set its "
        f"input and output to RASmm to say that it is one."
    )


def transformation_to_nodes(
    xform: _xforms.Transformation,
) -> tx.List[X5Node]:
    """Encode a transformation into X5 nodes, in the order they are applied.

    The first rule that applies is used:

    - A RAS displacement chain, which is an affine followed by a
      displacement field and another affine, becomes a node of
      displacements. Examples are an [`X5DisplacementField`][] and a NIfTI
      `DISPVECT` field.
    - The same chain whose field stores coefficients, such as an
      [`X5BSplineField`][], becomes a `bspline` node. Coefficients of another
      degree or boundary condition are refitted to cubic coefficients with a
      zero boundary.
    - A RAS coordinate chain, which is an affine followed by a coordinate
      field, becomes a node of deformations. Examples are an
      [`X5CoordinatesField`][] and an SPM `y_` field.
    - A transformation that converts to an affine becomes a linear node.
    - Any other sequence becomes the nodes of its elements.

    Raises
    ------
    UnrepresentableTransformationError
        If no rule applies, or if the transformation does not map RAS to RAS.
    """
    chain = _chain(xform)
    if chain is not None:
        if len(chain) == 3 and isinstance(chain[1], _xforms.DisplacementField):
            if chain[1].store is StoreEnum.coefficients:
                return [_encode_bspline(xform, chain)]
            return [_encode_displacements(xform, chain)]
        if len(chain) == 2 and isinstance(chain[1], _xforms.CoordinatesField):
            return [_encode_coordinates(xform, chain)]
    try:
        affine = xform.to(_xforms.Affine)
    except Exception as error:  # noqa: BLE001
        if chain is None:
            raise UnrepresentableTransformationError(
                f"X5 cannot encode a {type(xform).__name__}: it stores "
                f"affines, dense fields of displacements or coordinates, "
                f"and cubic B-splines of displacements."
            ) from error
        return [node for item in chain for node in _nodes(item)]
    return [_encode_affine(xform, affine)]


def _nodes(xform: _xforms.Transformation) -> tx.List[X5Node]:
    return transformation_to_nodes(xform)


def _chain(
    xform: _xforms.Transformation,
) -> tx.Optional[tx.Tuple[_xforms.Transformation, ...]]:
    if isinstance(xform, _xforms.Sequence):
        return tuple(xform.transformations or ())
    return None


def _encode_affine(
    xform: _xforms.Transformation, affine: _xforms.Affine
) -> X5Node:
    _check_ras(xform, "affine")
    matrix = affine.homogeneous_matrix
    matrix = np.eye(_NDIM + 1) if matrix is None else np.asarray(matrix)
    if matrix.shape != (_NDIM + 1, _NDIM + 1):
        raise UnrepresentableTransformationError(
            f"X5 stores 3-D affines, and this one has shape "
            f"{matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
        )
    return X5Node(
        type="linear",
        subtype="affine",
        representation="matrix",
        transform=np.asarray(matrix, dtype=np.float64),
        dimension_kinds=_KINDS,
    )


def _encode_displacements(
    xform: _xforms.Transformation, chain: tx.Sequence
) -> X5Node:
    _check_ras(xform, "displacement field")
    vox2ras, vectors = split_ras_displacement_chain(
        chain, "An X5 displacement field", ndim=_NDIM
    )
    return _field_node(vectors, vox2ras, "displacements")


def _encode_bspline(
    xform: _xforms.Transformation, chain: tx.Sequence
) -> X5Node:
    _check_ras(xform, "B-spline field")
    # Fields that are not cubic with a zero boundary are refitted.
    vox2ras, coefficients = split_ras_displacement_chain(
        chain,
        "An X5 B-spline",
        ndim=_NDIM,
        store=StoreEnum.coefficients,
        degree=_BSPLINE_DEGREE,
        bound=BoundaryCondition.zeros,
    )
    node = _field_node(coefficients, vox2ras, "coefficients")
    # nitransforms requires a Domain (the reference grid) even though the
    # transform ignores it, so the knot grid is written as one.
    return replace(
        node,
        subtype="bspline",
        additional_parameters=np.asarray(vox2ras, dtype=np.float64),
    )


def _encode_coordinates(
    xform: _xforms.Transformation, chain: tx.Sequence
) -> X5Node:
    _check_ras(xform, "coordinates field")
    what = "An X5 coordinates field"
    ras2vox = homogeneous_matrix(chain[0], what, ndim=_NDIM)
    field = chain[1].to(store=StoreEnum.values)
    if field.data is None:
        raise UnrepresentableTransformationError(
            "This field has no coordinates, so there is nothing to write."
        )
    coordinates = field.data
    shape = tuple(int(s) for s in coordinates.shape)
    if len(shape) == _NDIM + 2 and shape[_NDIM] == 1:
        # Drop the singleton axis of the NIfTI layout (X, Y, Z, 1, 3).
        coordinates = coordinates[:, :, :, 0]
    return _field_node(coordinates, np.linalg.inv(ras2vox), "deformations")


def _field_node(
    field: ArrayProtocol, vox2ras: np.ndarray, representation: str
) -> X5Node:
    shape = tuple(int(s) for s in field.shape)
    if len(shape) != _NDIM + 1 or shape[-1] != _NDIM:
        raise UnrepresentableTransformationError(
            f"X5 stores 3-D fields, with one 3-vector per voxel, not an "
            f"array of shape {shape}."
        )
    return X5Node(
        type="nonlinear",
        subtype="densefield",
        representation=representation,
        transform=_field_to_disk(field),
        dimension_kinds=_KINDS,
        domain=X5Domain(
            grid=True,
            size=shape[:-1],
            mapping=np.asarray(vox2ras, dtype=np.float64),
            coordinates="cartesian",
        ),
    )
