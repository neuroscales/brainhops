"""
The transformations an X5 node decodes into, and the nodes they encode
back into.

The rules, and the sources they were checked against, are described in
the package docstring, [`brainhops.io.transformations.x5`][].
"""

__all__ = [
    "X5BSplineField",
    "X5CoordinatesField",
    "X5DisplacementField",
    "node_to_transformation",
    "transformation_to_nodes",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core import affines as _affines
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum

# io
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserNotImplementedError,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

# locals
from ._struct import X5Domain, X5Node

_NDIM = 3
"""The number of spatial dimensions supported: X5 world space is RAS."""

DISPLACEMENTS = ("displacements", "displacement", "deltas", "relative")
"""`Representation` values of a field of relative displacements."""

COORDINATES = (
    "deformations",
    "deformation",
    "coordinates",
    "absolute",
)
"""`Representation` values of a field of absolute coordinates."""

COEFFICIENTS = (None, "coefficients", "coefficient")
"""`Representation` values of a `bspline` field of coefficients."""

_DENSE_SUBTYPES = (None, "densefield", "dense")
"""`SubType` values of a dense field sampled on the domain."""

_BSPLINE_SUBTYPES = ("bspline", "b-spline")
"""`SubType` values of a field of B-spline coefficients."""

_BSPLINE_DEGREE = 3
"""The degree of an X5 B-spline: nitransforms evaluates only cubics."""

_KINDS = ("space",) * _NDIM + ("vector",)


# ----------------------------------------------------------------------
#   FIELDS
# ----------------------------------------------------------------------


class _X5RASDisplacements(_xforms.ImmutableSequence):
    """A field of RAS displacements on a voxel grid, as a chain."""

    degree: tx.ClassVar[int] = 1
    """The spline degree used to interpolate the field."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """The boundary condition used outside of the field of view."""

    store: tx.ClassVar[StoreEnum] = StoreEnum.values
    """Whether the field holds spline coefficients rather than values."""

    @classmethod
    def from_ras(cls, vectors: ArrayProtocol, vox2ras: np.ndarray) -> tx.Self:
        """Build the field from RAS vectors and their grid."""
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
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def displacement(self) -> _xforms.Transformation:
        """The displacement field, in the voxel units of its grid."""
        return self.transformations[1]

    @property
    def voxel2ras(self) -> _xforms.Transformation:
        """The affine from the field's voxels back to RAS world."""
        return self.transformations[2]


class X5DisplacementField(_X5RASDisplacements):
    """
    A `nonlinear` X5 transform that stores relative displacements.

    Each sample of the field holds the displacement, in RAS millimetres,
    of the world point at its centre: the field maps RAS to RAS as
    `x -> x + u(x)`. The `Domain/Mapping` of the node is the voxel-to-RAS
    affine of its grid.

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |
    """


class X5BSplineField(_X5RASDisplacements):
    """
    A `nonlinear` X5 transform of `SubType` `bspline`.

    The field holds, at each knot of a regular grid, the cubic B-spline
    coefficients of a displacement in RAS millimetres; the grid's
    voxel-to-RAS affine is the node's `AdditionalParameters` (its
    `Domain` is the reference grid, which the transform does not
    depend on). The displacement of a point `x` is
    `u(x) = sum_k c_k B3(i(x) - k)`, where `i(x)` are the coordinates of
    `x` in the knot grid, `B3` is the tensor-product centred cubic
    B-spline and coefficients beyond the grid are zero; the field maps
    RAS to RAS as `x -> x + u(x)`.

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
    """
    A `nonlinear` X5 transform that stores absolute coordinates.

    Each sample of the field holds the RAS coordinates, in millimetres,
    that the world point at its centre maps to. The `Domain/Mapping` of
    the node is the voxel-to-RAS affine of its grid.

    | Slot          | Transformation                              |
    | ------------- | ------------------------------------------- |
    | `ras2voxel`   | RAS world coordinates to the field's voxels |
    | `coordinates` | the field of RAS coordinates                |
    """

    @classmethod
    def from_ras(
        cls, coordinates: ArrayProtocol, vox2ras: np.ndarray
    ) -> tx.Self:
        """Build the field from RAS coordinates and their grid."""
        vox2ras = np.asarray(vox2ras, dtype=np.float64)
        compact = vox2ras[:-1]
        return cls((
            RASToVoxel(matrix=_affines.inv(compact)),
            RASCoordinatesField(field=coordinates),
        ))

    @property
    def ras2voxel(self) -> _xforms.Transformation:
        """The affine from RAS world coordinates to the field's voxels."""
        return self.transformations[0]

    @property
    def coordinates(self) -> _xforms.Transformation:
        """The field of RAS coordinates, defined on the field's voxels."""
        return self.transformations[1]


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def node_to_transformation(node: X5Node) -> _xforms.Transformation:
    """
    The transformation an X5 node encodes, from RAS to RAS.

    Raises
    ------
    ParserNotImplementedError
        If the node is valid but cannot be represented: a `composite`
        node, a stack of transforms (`ArrayLength > 1`), or a field
        that is not sampled on a regular 3-D grid.
    ParserContentError
        If the node is malformed.
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
    matrix = np.asarray(node.transform, dtype=np.float64)
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
    vox2ras = np.asarray(domain.mapping, dtype=np.float64)
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
        return X5DisplacementField.from_ras(field, vox2ras)
    if representation in COORDINATES:
        return X5CoordinatesField.from_ras(field, vox2ras)
    raise ParserContentError(
        f"A nonlinear X5 transform must say whether it stores "
        f"displacements or coordinates, and its Representation is "
        f"{node.representation!r}."
    )


def _decode_bspline(node: X5Node) -> X5BSplineField:
    """
    A field of B-spline coefficients, as nitransforms writes and reads it.

    `BSplineFieldTransform.to_x5` stores the coefficients, `(X, Y, Z, 3)`
    in RAS millimetres, as `Transform`, and the voxel-to-RAS affine of
    their grid as `AdditionalParameters`; `from_x5` reads them back as a
    NIfTI image of coefficients with that affine. The `Domain` is the
    reference grid it is sampled on by `to_field`, which `map` ignores.
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
    knots = np.asarray(node.additional_parameters, dtype=np.float64)
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
    return X5BSplineField.from_ras(field, knots)


def _vector_last(node: X5Node) -> ArrayProtocol:
    """The field, with the axis that `DimensionKinds` calls "vector"
    moved last."""
    field = node.transform
    backend = get_array_backend(field)
    field = backend.asarray(field)
    kinds = node.dimension_kinds
    if kinds and len(kinds) == field.ndim and "vector" in kinds:
        axis = list(kinds).index("vector")
        if axis != field.ndim - 1:
            field = backend.moveaxis(field, axis, -1)
    return field


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
    """
    The X5 nodes that encode a transformation, in the order they apply.

    - A field of RAS displacements (an [`X5DisplacementField`][], a
      NIfTI `DISPVECT` field, or any chain of an affine, a
      `DisplacementField` and an affine) is one `nonlinear` node that
      stores `displacements`.
    - The same chain whose field holds spline coefficients, such as an
      [`X5BSplineField`][], is one `nonlinear` `bspline`
      node that stores `coefficients`: the field's `store` flag selects
      which of the two a displacement field is written as. X5 stores
      cubic coefficients with a zero boundary, so coefficients of
      another degree or boundary are refitted to those.
    - A field of RAS coordinates (an [`X5CoordinatesField`][], an SPM
      `y_` field, or any chain of an affine and a `CoordinatesField`) is
      one `nonlinear` node that stores `deformations`.
    - Anything that converts to an `Affine` is one `linear` node.
    - Any other sequence is the nodes of its elements, in order.

    Raises
    ------
    UnrepresentableTransformationError
        If a transformation is none of these, or does not map RAS to RAS.
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
    # X5 stores cubic coefficients with nothing beyond the grid of knots
    # (a zero boundary). A field stored that way is written as it is; any
    # other is refitted to it.
    vox2ras, coefficients = split_ras_displacement_chain(
        chain,
        "An X5 B-spline",
        ndim=_NDIM,
        store=StoreEnum.coefficients,
        degree=_BSPLINE_DEGREE,
        bound=BoundaryCondition.zeros,
    )
    node = _field_node(coefficients, vox2ras, "coefficients")
    # nitransforms reads the affine of the knots from AdditionalParameters,
    # and the Domain as the grid of its reference image, which it requires
    # but does not use to map points. Without a reference, the knot grid
    # is the Domain.
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
    # X5 stores sampled coordinates.
    field = chain[1].to(store=StoreEnum.values)
    if field.data is None:
        raise UnrepresentableTransformationError(
            "This field has no coordinates, so there is nothing to write."
        )
    coordinates = field.data
    shape = tuple(int(s) for s in coordinates.shape)
    if len(shape) == _NDIM + 2 and shape[_NDIM] == 1:
        # The NIfTI layout, (X, Y, Z, 1, 3), keeps a singleton axis.
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
        transform=field,
        dimension_kinds=_KINDS,
        domain=X5Domain(
            grid=True,
            size=shape[:-1],
            mapping=np.asarray(vox2ras, dtype=np.float64),
            coordinates="cartesian",
        ),
    )
