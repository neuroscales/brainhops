__all__ = [
    "Transformation",
    "CoordinatesField",
    "CartesianField",
    "DisplacementField",
    "MultiscaleCoordinatesField",
    "MultiscaleDisplacementField",
    "Affine",
    "Linear",
    "Rotation",
    "Permutation",
    "Scaling",
    "Translation",
    "Identity",
    "Bijection",
    "Inverse",
    "SubspaceTransformation",
    "Projection",
    "Sequence",
    "PlacedOmeField",
    "ome_vector_axis",
    "is_identity",
    "is_translation",
    "is_scale",
    "is_permutation",
    "is_rotation",
    "is_linear",
    "multiscale_level_for",
    "place_ome_field",
    "resolve_multiscale_level",
    "OmePlacementError",
    "ConversionError",
    "LossyConversionError",
    "CompositionError",
    "AdaptationError",
]
# stdlib
import itertools
import types as _t
from collections.abc import MutableSequence
from functools import partial
from numbers import Integral, Real

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.typing import (
    ArrayProtocol,
    npmatrix,
    npvector,
    safe_get_origin,
)

# ext
from brainhops._ext.invfield import inverse as inverse_disp

# backends
from brainhops.backends import get_array_backend

# locals
from . import hierarchy
from .base import DataModelBase
from .enums import BoundaryCondition, InterpolationOrder
from .systems import CoordinateSystem

# `types.UnionType` (the runtime type of a PEP 604 `X | Y` union) only
# exists on Python 3.10+. It is `None` on older interpreters.
_UnionType = getattr(_t, "UnionType", None)

if False:
    # This is an idea to implement a pipe-like syntax `a |p> b`.
    # It's not very pythonic!

    class _Pipe:
        def __init__(
            self,
            input: tx.Optional["Transformation"] = None,
            side: tx.Optional[tx.Literal["L", "R"]] = None,
        ) -> None:
            self.input = input
            self.side = side

        def __ror__(
            self, other: "Transformation"
        ) -> tx.Union["_Pipe", "Transformation"]:
            if self.side == "R":
                return self.input(other)
            elif self.side is None:
                return _Pipe(input=other, side="L")
            else:
                raise SyntaxError("Invalid pipe syntax")

        def __gt__(
            self, other: "Transformation"
        ) -> tx.Union["_Pipe", "Transformation"]:
            if self.side == "L":
                return other(self.input)
            elif self.side is None:
                return _Pipe(input=other, side="R")
            else:
                raise SyntaxError("Invalid pipe syntax")

    p = _Pipe()


# ----------------------------------------------------------------------
#    BASE CLASS
# ----------------------------------------------------------------------


@hierarchy.Transformation.register
class Transformation(DataModelBase, reverse=True):
    """
    A transformation between coordinate systems.

    It maps coordinates from an input coordinate system to an output
    coordinate system. Transformations can be applied and/or composed
    using different syntaxes:

    1. Functional: `t(x)` applies the transform `t` to coordinates `x`,
       and `t2(t1)` composes the transform `t1` with the transform `t2`
       (i.e., applies `t1` first, then `t2`).

    2. Matrix-like: `t @ x` applies the transform `t` to coordinates `x`,
       and `t2 @ t1` composes the transform `t1` with the transform `t2`.

       This abuses the matrix multiplication operator `@` because linear
       and affine transformations can be represented as matrices, and are
       typically applied to coordinates using matrix multiplication, with
       the input space "on the right" and the output space "on the left".

    !!! warning "Direction of transformation"
        This mapping direction is the opposite of the direction that is
        typically used to transform images. For example, a transformation
        that deforms an image from space A to space B, will actually
        map coordinates from space B to space A. In our model, this
        transformation would be represented as `Transform(input=B, output=A)`.
    """

    parameter_names: tx.Annotated[
        tx.ClassVar[tx.Union[str, tx.Tuple[str, ...]]],
        tx.Doc("The attributes that parameterize the transformation."),
    ] = ()

    input: tx.Annotated[
        tx.Optional[CoordinateSystem],
        tx.Doc(
            """
            The input coordinate system of the transformation.
            If not specified, it can be inferred from the context (e.g.,
            from the coordinate system of the image being transformed).
            """
        ),
    ] = None

    output: tx.Annotated[
        tx.Optional[CoordinateSystem],
        tx.Doc(
            """
            The output coordinate system of the transformation.
            If not specified, it can be inferred from the context (e.g.,
            from the coordinate system of the image being transformed).
            """
        ),
    ] = None

    def compute(self, simplify: bool = False) -> tx.Self:
        """
        Compute the transformation, if it is not already fully defined.
        """
        # We will overload `compute()` in `Sequence`, so here we can
        # assume that `self` is not a `Sequence`.
        # For non sequence transformations, this function simplifies
        # to the simplest compatible kind, whose compatibility can be
        # detected with (almost) no overhead. For example, if the
        # parameter of a transformation is set to `None`, the
        # transformation is treated as an identity transformation.
        CHECKS = [
            (is_identity, Identity),
            (is_translation, Translation),
            (is_scale, Scaling),
            (is_permutation, Permutation),
            (is_rotation, Rotation),
            (is_linear, Linear),
        ]
        for check, cls in CHECKS:
            if check(self, compute=simplify):
                return self.to(cls)
        return self

    def to(
        self,
        cls: tx.Optional[tx.Type[tx.Self]] = None,
        *,
        lossy: bool = False,
        **kwargs,
    ) -> tx.Self:
        """
        Convert this transformation to a different type.

        Parameters
        ----------
        cls : type, optional
            The type to convert to. If `None`, keep the current type.
        lossy : bool, default=False
            Whether to allow lossy conversions.
        **kwargs : dict
            Attributes to override in the converted transform.
            This allows transformations to be modified within their type.
            For example, a [`DisplacementField`][] can be converted from
            a field of values to a field of spline coefficients by
            setting `coeff=True` in `kwargs`.

        Returns
        -------
        Transformation
            The converted transformation.
        """
        cls = cls or type(self)
        if lossy:
            try:
                return _to(self, cls, **kwargs)
            except LossyConversionError as e:
                return e.result
        else:
            return _to(self, cls, **kwargs)

    def inverse(self) -> tx.Self:
        """Return the inverse of this transformation.

        The inverse can also be obtained using the `__invert__` operator:
        `T.inverse()` is equivalent to `~T`.
        """
        raise NotImplementedError

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[True]
    ) -> "CoordinatesField":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "CoordinatesField", compute: tx.Literal[False]
    ) -> "Sequence":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.

        This variant does not compute the transformed coordinates, but
        instead returns an object that holds the original coordinates
        and the transform, and can be computed later when needed.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[True]
    ) -> "Transformation":
        """
        Compose this transform with another transform.

        The resulting transform will have the input coordinate system of
        the other transform and the output coordinate system of this
        transform.

        If the output coordinate system of the other transform does not
        match the input coordinate system of this transform, it will
        try to guess an intermediate adapter transform, and raise an
        error if it cannot find one.
        """
        ...

    @tx.overload
    def __call__(
        self, x: "Transformation", compute: tx.Literal[False]
    ) -> "Sequence":
        """
        Compose this transform with another transform, without computing
        the resulting transform, but instead returning a sequence of the
        two transformations that can be computed later when needed.
        """
        ...

    def __call__(self, x, compute: bool = False) -> "Transformation":
        if isinstance(x, Transformation):
            x = Sequence([x, self])
        else:
            raise TypeError(f"Unsupported type for transformation: {type(x)}")
        if compute:
            x = x.compute(mode=None)
        return x

    def __matmul__(self, other: "Transformation") -> "Transformation":
        return self(other)

    def __or__(self, other: "Transformation") -> "Transformation":
        return other(self)

    def __invert__(self) -> tx.Self:
        return self.inverse()


# ----------------------------------------------------------------------
#    CONCRETE CLASSES
# ----------------------------------------------------------------------


class CoordinatesField(Transformation):
    """
    A field of coordinates defined on a regular grid.

    The input space corresponds to the regular grid on which the
    coordinates are defined.
    """

    parameter_names: tx.ClassVar[str] = "field"

    field: tx.Annotated[
        tx.Optional[ArrayProtocol],
        tx.Doc("An array of shape `(*shape, ndim)`."),
    ] = None

    order: tx.Annotated[
        InterpolationOrder, tx.Doc("The spline interpolation order")
    ] = 1

    bound: tx.Annotated[
        tx.Union[BoundaryCondition, float],
        tx.Doc(
            """
            The boundary condition used to deal with coordinates outside
            of the field of view. If a float is given, it is treated as
            a constant value.
            """
        ),
    ] = BoundaryCondition.nearest

    coeff: tx.Annotated[
        bool,
        tx.Doc(
            """
            If `True`, the field is treated as a field of spline coefficients,
            rather than a field if values to interpolate.
            """
        ),
    ] = False

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.field is None:
            return cls(input=self.output, output=self.input)
        raise NotImplementedError
        # TODO:
        # If the input and output systems are the same, we can use the
        # displacement field's inverse (disp = coord - meshgrid).
        # Otherwise, I am not sure we can easily compute an inverse,
        # since it'll depend on the "shape" (and "orientation") of
        # the output space. However, we could introduced a delayed
        # `InverseCoordinatesField` class, that computes the inverse
        # on demand during interpolation (as the output shape will then
        # be known).


class CartesianField(CoordinatesField):
    """
    An identity transform over a regular grid of coordinates.

    Both the input and output spaces correspond to the underlying grid.

    Its `field` attribute is fully defined by the shape of the grid,
    and is generated on demand when accessed.
    """

    shape: tx.Annotated[
        tx.Optional[tx.Tuple[int, ...]], tx.Doc("The shape of the grid.")
    ] = None

    # `field` is computed on demand from `shape` by the property below,
    # so it is not a stored, constructor-taken field here. Declaring it a
    # `ClassVar` overrides the inherited init-field from `CoordinatesField`
    # and keeps `field` out of `__init__`, `fields()` and `replace()`,
    # while the property keeps serving reads.
    field: tx.ClassVar[tx.Optional[ArrayProtocol]]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        if self.shape is None:
            return None
        if getattr(self, "_field", None) is None:
            ab = get_array_backend()
            self._field = ab.stack(
                ab.meshgrid(
                    *[ab.arange(s) for s in self.shape], indexing="ij"
                ),
                -1,
            )
        return self._field

    @field.setter
    def field(self, value: None) -> None:
        if value is not None:
            raise ValueError(
                "Cannot set field of CartesianField to a non-None value."
            )

    def inverse(self) -> tx.Self:
        # Inverse is itself, with switched input and output.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)


class DisplacementField(Transformation):
    """
    A field of displacements defined on a regular grid.

    Both the input and output spaces correspond to the underlying grid.
    """

    parameter_names: tx.ClassVar[str] = "field"

    field: tx.Annotated[
        tx.Optional[ArrayProtocol],
        tx.Doc(
            "An array of shape `(*shape, ndim)`, where `len(shape) == ndim`"
        ),
    ] = None

    order: tx.Annotated[
        InterpolationOrder, tx.Doc("The spline interpolation order")
    ] = 1

    bound: tx.Annotated[
        tx.Union[BoundaryCondition, float],
        tx.Doc(
            """
            The boundary condition used to deal with coordinates outside
            of the field of view. If a float is given, it is treated as
            a constant value.
            """
        ),
    ] = BoundaryCondition.nearest

    coeff: tx.Annotated[
        bool,
        tx.Doc(
            """
            If `True`, the field is treated as a field of spline coefficients,
            rather than a field if values to interpolate.
            """
        ),
    ] = False

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.field is None:
            return cls(input=self.output, output=self.input)
        return cls(
            field=inverse_disp(self.field),
            input=self.output,
            output=self.input,
        )


class MultiscaleCoordinatesField(CoordinatesField):
    """A field of coordinates represented at several resolutions.

    The levels form a resolution pyramid, ordered from finest to
    coarsest. Each level is an array of coordinates defined on its own
    grid. One level is active at a time, and the field behaves exactly
    like a single-scale [`CoordinatesField`][] built from that level.
    The finest level is active by default.

    Each level also carries a transformation that maps the coordinates
    of that level's grid to the coordinates of the finest grid. The
    transformation of the finest level is the identity. These
    transformations are stored rather than inferred, because pyramid
    builders differ in the sub-pixel alignment between levels.

    A `MultiscaleCoordinatesField` is a `CoordinatesField`, so it
    composes with other transformations exactly as its active level
    would. Composition therefore collapses the pyramid to the active
    level.
    """

    levels: tx.Annotated[
        tx.Optional[tx.List[ArrayProtocol]],
        tx.Doc(
            """
            The coordinate array of each resolution level, ordered from
            finest to coarsest. Each array has shape `(*shape, ndim)`.
            """
        ),
    ] = None

    level_transforms: tx.Annotated[
        tx.Optional[tx.List[Transformation]],
        tx.Doc(
            """
            The transformation that maps each level's grid to the finest
            level's grid, ordered from finest to coarsest. The first
            entry is the identity. Each entry is a scaling and a shift.
            """
        ),
    ] = None

    level: tx.Annotated[
        int,
        tx.Doc(
            """
            The index of the active resolution level. Level `0` is the
            finest level, and is the default.
            """
        ),
    ] = 0

    # `field` is served on demand from the active level by the property
    # below, so it is not a stored, constructor-taken field here.
    # Declaring it a `ClassVar` overrides the inherited init-field from
    # `CoordinatesField` and keeps `field` out of `__init__`, `fields()`,
    # `replace()` and `from_instance()`, so a rebuild never freezes one
    # level's array while the active level changes.
    field: tx.ClassVar[tx.Optional[ArrayProtocol]]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The coordinate array of the active resolution level."""
        return _multiscale_field(self)

    @field.setter
    def field(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is not None:
            raise ValueError(
                "The field of a multiscale field is determined by its "
                "levels and cannot be set directly."
            )

    def at_level(self, level: int) -> tx.Self:
        """Return this field with a different resolution level active.

        Level `0` is the finest level. The returned field shares the
        same levels and level transformations, and differs only in which
        level is active.
        """
        return self.to(level=int(level))

    def select_level(self, target: tx.Any) -> tx.Self:
        """Return this field with the resolution-matched level active.

        The level whose grid resolution is closest to `target` is
        activated. The argument names the target resolution as a factor
        relative to the finest grid. See
        [`multiscale_level_for`][brainhops.datamodel.transformations.multiscale_level_for].
        """
        return self.at_level(multiscale_level_for(self, target))

    @property
    def level_scales(self) -> tx.List[tx.Optional[ArrayProtocol]]:
        """The per-axis downsampling factor of each level.

        The factor is expressed relative to the finest grid. The finest
        level, and any level without a stored cross-level transformation,
        is reported as `None`, which stands for a factor of one on every
        axis.
        """
        return _multiscale_level_scales(self)


class MultiscaleDisplacementField(DisplacementField):
    """A field of displacements represented at several resolutions.

    The levels form a resolution pyramid, ordered from finest to
    coarsest. Each level is an array of displacements defined on its own
    grid. One level is active at a time, and the field behaves exactly
    like a single-scale [`DisplacementField`][] built from that level.
    The finest level is active by default.

    Each level also carries a transformation that maps the coordinates
    of that level's grid to the coordinates of the finest grid. The
    transformation of the finest level is the identity. These
    transformations are stored rather than inferred, because pyramid
    builders differ in the sub-pixel alignment between levels.

    A `MultiscaleDisplacementField` is a `DisplacementField`, so it
    composes with other transformations exactly as its active level
    would. Composition therefore collapses the pyramid to the active
    level.
    """

    levels: tx.Annotated[
        tx.Optional[tx.List[ArrayProtocol]],
        tx.Doc(
            """
            The displacement array of each resolution level, ordered from
            finest to coarsest. Each array has shape `(*shape, ndim)`.
            """
        ),
    ] = None

    level_transforms: tx.Annotated[
        tx.Optional[tx.List[Transformation]],
        tx.Doc(
            """
            The transformation that maps each level's grid to the finest
            level's grid, ordered from finest to coarsest. The first
            entry is the identity. Each entry is a scaling and a shift.
            """
        ),
    ] = None

    level: tx.Annotated[
        int,
        tx.Doc(
            """
            The index of the active resolution level. Level `0` is the
            finest level, and is the default.
            """
        ),
    ] = 0

    # See the note on `MultiscaleCoordinatesField.field`. Declaring
    # `field` a `ClassVar` keeps it out of the constructor so a rebuild
    # follows the active level rather than freezing one level's array.
    field: tx.ClassVar[tx.Optional[ArrayProtocol]]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The displacement array of the active resolution level."""
        return _multiscale_field(self)

    @field.setter
    def field(self, value: tx.Optional[ArrayProtocol]) -> None:
        if value is not None:
            raise ValueError(
                "The field of a multiscale field is determined by its "
                "levels and cannot be set directly."
            )

    def inverse(self) -> tx.Self:
        cls = type(self)
        if not self.levels:
            return cls(input=self.output, output=self.input)
        return cls(
            levels=[inverse_disp(level) for level in self.levels],
            level_transforms=self.level_transforms,
            level=self.level,
            input=self.output,
            output=self.input,
        )

    def at_level(self, level: int) -> tx.Self:
        """Return this field with a different resolution level active.

        Level `0` is the finest level. The returned field shares the
        same levels and level transformations, and differs only in which
        level is active.
        """
        return self.to(level=int(level))

    def select_level(self, target: tx.Any) -> tx.Self:
        """Return this field with the resolution-matched level active.

        The level whose grid resolution is closest to `target` is
        activated. The argument names the target resolution as a factor
        relative to the finest grid. See
        [`multiscale_level_for`][brainhops.datamodel.transformations.multiscale_level_for].
        """
        return self.at_level(multiscale_level_for(self, target))

    @property
    def level_scales(self) -> tx.List[tx.Optional[ArrayProtocol]]:
        """The per-axis downsampling factor of each level.

        The factor is expressed relative to the finest grid. The finest
        level, and any level without a stored cross-level transformation,
        is reported as `None`, which stands for a factor of one on every
        axis.
        """
        return _multiscale_level_scales(self)


@hierarchy.AffineTransformation.register
class Affine(Transformation):
    """An affine transformation."""

    parameter_names: tx.ClassVar[str] = "matrix"

    matrix: tx.Annotated[
        tx.Optional[npmatrix[Real]],
        tx.Doc(
            """
            A matrix of shape `(No, Ni + 1)`, where `Ni` is the number of
            input dimensions and `No` is the number of output dimensions.
            The last column of the matrix corresponds to the translation
            component of the affine transformation.
            If `None`, the matrix is treated as an identity transformation.
            """
        ),
    ] = None

    @property
    def homogeneous_matrix(self) -> ArrayProtocol:
        """
        The homogeneous matrix of the affine transformation, of shape
        `(No + 1, Ni + 1)`. The last row of the homogeneous matrix is
        `[0, 0, ..., 1]`.
        """
        if self.matrix is None:
            return None
        ab = get_array_backend(self.matrix)
        No, NiPlus1 = self.matrix.shape
        homogeneous_matrix = ab.zeros((No + 1, NiPlus1))
        homogeneous_matrix[:-1, :-1] = self.matrix[:, :-1]
        homogeneous_matrix[:-1, -1:] = self.matrix[:, -1:]
        homogeneous_matrix[-1, -1] = 1
        return homogeneous_matrix

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.matrix is None:
            return cls(input=self.output, output=self.input)
        ab = get_array_backend(self.matrix)
        return cls(
            matrix=ab.linalg.inv(self.homogeneous_matrix)[:-1],
            input=self.output,
            output=self.input,
        )


@hierarchy.LinearTransformation.register
class Linear(Transformation):
    """A linear transformation."""

    parameter_names: tx.ClassVar[str] = "matrix"

    matrix: tx.Annotated[
        tx.Optional[npmatrix[Real]],
        tx.Doc(
            """
            A matrix of shape `(No, Ni)`, where `Ni` is the number of
            input dimensions and `No` is the number of output dimensions.
            If `None`, the matrix is treated as an identity transformation.
            """
        ),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.matrix is None:
            return cls(input=self.output, output=self.input)
        ab = get_array_backend(self.matrix)
        return cls(
            matrix=ab.linalg.inv(self.matrix),
            input=self.output,
            output=self.input,
        )


@hierarchy.SpecialOrthogonalTransformation.register
class Rotation(Linear):
    """An orthogonal transformation with determinant 1, i.e., a rotation."""

    # TODO: Implement Rotation subclasses that use other representations
    # (e.g., quaternions, Euler angles, etc.)

    matrix: tx.Annotated[
        tx.Optional[npmatrix[Real]],
        tx.Doc(
            """
            A matrix of shape `(No, Ni)`, where `Ni` is the number of
            input dimensions and `No` is the number of output dimensions.
            This matrix MUST have a determinant of 1.
            If `None`, the matrix is treated as an identity transformation.
            """
        ),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.matrix is None:
            return cls(input=self.output, output=self.input)
        return cls(matrix=self.matrix.T, input=self.output, output=self.input)


@hierarchy.Permutation.register
class Permutation(Transformation):
    """A permutation of axes."""

    parameter_names: tx.ClassVar[str] = "permutation"

    permutation: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc(
            """
            A vector of shape `(N,)`, where `N` is the number of axes
            to permute. The element at index `i` indicates the input
            dimension that corresponds to the output dimension `i`.
            If `None`, the permutation is treated as an identity
            transformation.
            """
        ),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.permutation is None:
            return cls(input=self.output, output=self.input)
        inverse_permutation = [0] * len(self.permutation)
        for i, p in enumerate(self.permutation):
            inverse_permutation[p] = i
        return cls(
            permutation=inverse_permutation,
            input=self.output,
            output=self.input,
        )


@hierarchy.DiagonalTransformation.register
class Scaling(Transformation):
    """A scaling of axes."""

    parameter_names: tx.ClassVar[str] = "scale"

    scale: tx.Annotated[
        tx.Optional[npvector[Real]],
        tx.Doc(
            """
            A vector of shape `(N,)`, where `N` is the number of dimensions
            to scale. If `None`, the scaling is treated as an identity
            transformation.
            """
        ),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.scale is None:
            return cls(input=self.output, output=self.input)
        return cls(
            scale=1.0 / self.scale, input=self.output, output=self.input
        )


@hierarchy.Translation.register
class Translation(Transformation):
    """A translation."""

    parameter_names: tx.ClassVar[str] = "translation"

    translation: tx.Annotated[
        tx.Optional[npvector[Real]],
        tx.Doc(
            """
            A vector of shape `(N,)`, where `N` is the number of dimensions
            to translate. If `None`, the translation is treated as an identity
            transformation.
            """
        ),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.translation is None:
            return cls(input=self.output, output=self.input)
        return cls(
            translation=-self.translation, input=self.output, output=self.input
        )


@hierarchy.IdentityTransformation.register
class Identity(Transformation):
    """An identity transformation.

    If the `input` and `output` coordinate systems are different, it maps
    the input axes to the output axes, while preserving their orders.
    """

    def inverse(self) -> tx.Self:
        cls = type(self)
        return cls(input=self.output, output=self.input)


# ----------------------------------------------------------------------
#    META TRANSFORMATIONS
# ----------------------------------------------------------------------


@hierarchy.BijectiveTransformation.register
class Bijection(Transformation):
    """
    A transformation whose inverse is explicitly defined.
    """

    forward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The forward transformation.")
    ] = None

    backward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The backward transformation.")
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        return cls(
            forward=self.backward,
            backward=self.forward,
            input=self.output,
            output=self.input,
        )

    @property
    def guess_input(self) -> tx.Optional[CoordinateSystem]:
        if self.input is not None:
            return self.input
        if self.forward is not None:
            return self.forward.input
        if self.backward is not None:
            return self.backward.output
        return None

    @property
    def guess_output(self) -> tx.Optional[CoordinateSystem]:
        if self.output is not None:
            return self.output
        if self.forward is not None:
            return self.forward.output
        if self.backward is not None:
            return self.backward.input


class Inverse(Transformation):
    """
    The inverse of a transformation.

    This is a delayed transformation that computes the inverse of the
    original transformation on demand when applied or computed.
    """

    transformation: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The transformation to invert.")
    ] = None

    def compute(self, simplify: bool = False) -> Transformation:
        if self.transformation is None:
            return (
                Identity(input=self.input, output=self.output)
                if simplify
                else self
            )
        return self.transformation.inverse().compute(simplify=simplify)

    def inverse(self) -> Transformation:
        return self.transformation.to(
            input=self.guess_output, output=self.guess_input
        )

    @property
    def guess_input(self) -> tx.Optional[CoordinateSystem]:
        if self.input is not None:
            return self.input
        if self.transformation is not None:
            return self.transformation.output
        return None

    @property
    def guess_output(self) -> tx.Optional[CoordinateSystem]:
        if self.output is not None:
            return self.output
        if self.transformation is not None:
            return self.transformation.input
        return None


class SubspaceTransformation(Transformation):
    """
    A transformation that is applied to a subset of the input and output axes.
    """

    transformation: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The transformation to apply.")
    ] = None

    input_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the input coordinate system to transform."),
    ] = None

    output_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the output coordinate system to transform."),
    ] = None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.transformations is None:
            return cls(input=self.output, output=self.input)
        return cls(
            transformations=[t.inverse() for t in self.transformations],
            input=self.output,
            output=self.input,
            input_axes=self.output_axes,
            output_axes=self.input_axes,
        )


class Projection(Transformation):
    """
    A transformation that acts as a projection from a higher dimensional
    space to a lower-dimensional space by removing one or more axes.

    Or its inverse (i.e., an embedding) that adds one or more axes to a
    lower-dimensional space.
    """

    dropped: npvector[Integral]
    created: npvector[Integral]

    def inverse(self) -> tx.Self:
        cls = type(self)
        return cls(
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )


# ----------------------------------------------------------------------
#    SEQUENCE
# ----------------------------------------------------------------------

# typing
_ModeCls = tx.Union[str, tx.Type[hierarchy.Transformation]]
_ModeLike = tx.Union[tx.Tuple[_ModeCls, tx.Optional[int]], _ModeCls, int]
ModeLike = tx.Union[_ModeLike, tx.Iterable[_ModeLike]]


class Sequence(MutableSequence, Transformation):
    """A sequence of transformations.

    !!! note
        Transformations in a sequence are listed in the order in which
        they are applied. It reads as the opposite order to function
        composition (or matrix multiplication), which may be confusing.

        * `Sequence([t1, t2, t3])(x)` is equivalent to `t3(t2(t1(x)))`.
        * `Sequence([t1, t2, t3]) @ x` is equivalent to `t3 @ t2 @ t1 @ x`.
    """

    parameter_names: tx.ClassVar[str] = "transformations"

    transformations: tx.Annotated[
        tx.Optional[tx.List[Transformation]],
        tx.Doc(
            """
            A list of transformations, in the order in which they are
            applied to an input coordinate system.
            """
        ),
    ] = None

    def guess_input(self) -> tx.Optional[CoordinateSystem]:
        if self.input is not None:
            return self.input
        if self.transformations or []:
            return self.transformations[0].input
        return None

    def guess_output(self) -> tx.Optional[CoordinateSystem]:
        if self.output is not None:
            return self.output
        if self.transformations or []:
            return self.transformations[-1].output
        return None

    def inverse(self) -> tx.Self:
        cls = type(self)
        if self.transformations is None:
            return cls(input=self.output, output=self.input)
        return cls(
            transformations=[
                t.inverse() for t in reversed(self.transformations)
            ],
            input=self.output,
            output=self.input,
        )

    def compute(self, mode: tx.Optional[ModeLike] = None) -> Transformation:
        """
        Compute the resulting transform of the sequence of transformations.

        If all transformations in the sequence are affine-like transformations,
        `compute()` returns an affine-like transform.

        If the first (= rightmost) transform in the sequence is a
        coordinate field, `compute()` returns a coordinate field.

        If the first (= rightmost) transform in the sequence is an
        affine-like transform, and the sequence contains at least one
        non-affine-like transform, `compute()` returns a sequence of two
        transformations:
        1. the composition of all affine-like transformations that appear
           before the first non-affine-like transform in the sequence, and
        2. the composition of all transformations in the sequence, starting
           from the first non-affine-like transform in the sequence.

        Parameters
        ----------
        mode : [list of] str, optional
            Types of transformations to compute.
            * If `None` (default): compute all transformations in the sequence.
            * If the name of a transformation type: compute only consecutive
              sequences of transformations that match the specified type.
        """
        mode = _ensure_proper_modes(mode)
        if not mode:
            return self  # No-op
        return _compute_sequence(self, mode=mode)

    def _flattened(self) -> tx.Self:
        # Flatten nested sequences into a single sequence, and propagate
        # the sequence's own input and output onto its first and last
        # transformations. Subclasses that must keep a fixed shape (such
        # as `Geometry`) override this.
        if self.transformations is None:
            return self
        inp, out = self.input, self.output
        flattened = []
        for i, t in enumerate(self.transformations):
            if i == 0 and t.input is None and inp is not None:
                t = t.to(input=inp)
            elif i == len(self) - 1 and t.output is None and out is not None:
                t = t.to(output=out)
            if isinstance(t, Sequence):
                # A `Geometry` child contributes its grid followed by its
                # transformation, which may itself still be a sequence. That
                # leaves one level of nesting in the splice, which is fine:
                # `_compute_sequence` re-flattens on the next pass.
                flattened.extend(t._flattened().transformations or [])
            else:
                flattened.append(t)
        return replace(self, transformations=flattened)

    # --- sequence API ---

    def __len__(self) -> int:
        return len(self.transformations or [])

    def __getitem__(self, index: tx.Union[int, slice]) -> Transformation:
        return self.transformations[index]

    def __setitem__(
        self, index: tx.Union[int, slice], value: Transformation
    ) -> None:
        self.transformations[index] = value

    def __delitem__(self, index: tx.Union[int, slice]) -> None:
        del self.transformations[index]

    def __iter__(self) -> tx.Iterator[Transformation]:
        return iter(self.transformations or [])

    def insert(self, index: int, value: Transformation) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.insert(index, value)

    def append(self, value: Transformation) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.append(value)

    def extend(self, values: tx.List[Transformation]) -> None:
        if self.transformations is None:
            self.transformations = []
        self.transformations.extend(values)

    def clear(self) -> None:
        if self.transformations:
            self.transformations.clear()

    def pop(self, index: int = -1) -> Transformation:
        if self.transformations is None:
            raise IndexError("pop from empty sequence")
        return self.transformations.pop(index)

    def remove(self, value: Transformation) -> None:
        if self.transformations is None:
            raise ValueError("remove from empty sequence")
        self.transformations.remove(value)


# ----------------------------------------------------------------------
#    MULTISCALE FIELDS
# ----------------------------------------------------------------------


class OmePlacementError(ValueError):
    """Raised when an OME field cannot be read as a placed field.

    A displacement field that is placed by a non-linear transformation,
    and a field whose axes mix the `displacement` and `coordinate`
    types, are both refused with this error.
    """


def _multiscale_field(field: Transformation) -> tx.Optional[ArrayProtocol]:
    # The array of the active resolution level. It is served on demand
    # from `levels` rather than stored, so a rebuild that changes the
    # active level serves the new level's array.
    levels = field.levels
    if not levels:
        return None
    return levels[field.level]


def _multiscale_level_scales(field: Transformation) -> tx.List[ArrayProtocol]:
    # The per-axis downsampling factor of each level, read from the
    # linear part of each level transformation. The finest level, and any
    # level without a stored transformation, is a factor of one.
    levels = field.levels or []
    transforms = field.level_transforms or []
    scales = []
    for i in range(len(levels)):
        transform = transforms[i] if i < len(transforms) else None
        if transform is None or is_identity(transform):
            scales.append(None)
        else:
            scales.append(_affine_scale(transform))
    return scales


def _affine_scale(xform: Transformation) -> ArrayProtocol:
    # The per-axis scale of a transformation, taken as the Euclidean norm
    # of each column of the linear part of its affine matrix. This is
    # invariant to rotation, so a rotated placement reports its true
    # voxel size on each axis.
    matrix = _linear_part(xform)
    return (matrix**2).sum(axis=0) ** 0.5


def _linear_part(xform: Transformation) -> npmatrix:
    # The (ndim, ndim) linear part of a transformation, as an array. A
    # transformation that does not reduce to an affine with a defined
    # matrix has no linear part and is refused.
    affine = xform.compute().to(Affine)
    if not isinstance(affine, Affine) or affine.matrix is None:
        raise OmePlacementError(
            "The placement of an OME field must be a defined affine "
            "transformation, but this one has no matrix."
        )
    return affine.matrix[:, :-1]


def multiscale_level_for(field: Transformation, target: tx.Any) -> int:
    """Return the index of the level that best matches a target grid.

    The `field` is a multiscale field. The `target` names the resolution
    to match, as a factor relative to the finest grid. A factor of one
    matches the finest level, and a factor of two matches a level whose
    grid is sampled half as densely. The target may be a single number,
    a per-axis sequence of numbers, or a transformation whose linear part
    supplies the per-axis factor.

    The level whose downsampling factor is closest to the target is
    returned, measured in logarithmic scale so that the level below and
    the level above the target are weighed evenly. When several levels
    are equally close, the finer of them is returned.
    """
    levels = field.levels or []
    if len(levels) <= 1:
        return 0
    ab = get_array_backend()
    if isinstance(target, Transformation):
        target_scale = _affine_scale(target)
    else:
        target_scale = ab.asarray(target, dtype=float)
    target_scale = ab.reshape(target_scale, (-1,))
    log_target = ab.log(ab.abs(target_scale)).mean()
    scales = _multiscale_level_scales(field)
    best_index, best_distance = 0, None
    for index, scale in enumerate(scales):
        if scale is None:
            log_scale = 0.0
        else:
            log_scale = float(ab.log(ab.abs(ab.asarray(scale))).mean())
        distance = abs(log_scale - float(log_target))
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def _ome_axis_kind(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Tuple[str, tx.Optional[int]]:
    # Classify the axes of an OME field by the `type` of each axis, and
    # report which axis holds the vector components. Per the OME-NGFF
    # coordinate-field specification, the axes list every input `space`
    # axis together with exactly one `displacement` or `coordinate` axis,
    # which carries the vector components. The result is a pair of the
    # kind and the index of that vector axis.
    #
    # The kind is one of "coordinate", "displacement", "mixed", or
    # "malformed". Exactly one displacement axis and no coordinate axis is
    # a displacement field. Exactly one coordinate axis and no
    # displacement axis is a coordinate field. Both present, or more than
    # one of either, is a mixed field. Neither present is malformed. When
    # no axes are given at all, the field is read as a coordinate field
    # with its vector axis last by convention.
    if not axes:
        return "coordinate", None
    displacement = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "displacement"
    ]
    coordinate = [
        i
        for i, a in enumerate(axes)
        if getattr(a, "type", None) == "coordinate"
    ]
    if len(displacement) == 1 and not coordinate:
        return "displacement", displacement[0]
    if len(coordinate) == 1 and not displacement:
        return "coordinate", coordinate[0]
    if displacement or coordinate:
        return "mixed", None
    return "malformed", None


class PlacedOmeField(Sequence):
    """An OME field placed in world space by the sandwich construction.

    The sequence is `Sequence([~xform_0, field, xform_0])`, where
    `xform_0` is the finest level's voxel-to-world transformation. A
    `PlacedOmeField` behaves as that sequence, and additionally keeps
    `xform_0` on the `placement` attribute. That placement records the
    finest grid, so a later reslice can match the target resolution to a
    level and rebuild the sandwich without recovering the placement from
    the sequence by position.
    """

    placement: tx.Annotated[
        tx.Optional[Transformation],
        tx.Doc(
            """
            The finest level's voxel-to-world transformation, `xform_0`.
            It is kept so a reslice can select a resolution level from the
            grid it is resliced onto.
            """
        ),
    ] = None


def place_ome_field(
    field: Transformation,
    placement: Transformation,
    *,
    level: tx.Optional[int] = None,
) -> "PlacedOmeField":
    """Place an OME field in world space with the sandwich construction.

    The `field` is a voxel-to-voxel field, either single-scale or
    multiscale. The `placement` is the finest level's voxel-to-world
    transformation, called `xform_0`. The result is the sequence

        Sequence([~xform_0, field, xform_0])

    which reads right to left as `xform_0 @ field @ ~xform_0`. It maps
    world coordinates to world coordinates, while the field itself works
    in voxel coordinates.

    This mirrors the way [`SPMCoordinatesField`][] is assembled as a
    sequence of a world-to-voxel affine and a field. The world-to-voxel
    affine on the right converts an incoming world coordinate to the
    field's voxel grid, the field is applied there, and the voxel-to-world
    affine on the left converts the result back to world coordinates.

    When `level` names a resolution level of a multiscale `field`, that
    level is activated. In every case the affines of the sandwich are
    built for the field's active level, so a coarse level is sampled on
    its own grid. When `level` is `None`, the field's current active
    level is used. The returned [`PlacedOmeField`][] keeps `xform_0`, the
    finest level's placement, so a reslice can select a level later.
    """
    if level is not None and getattr(field, "levels", None):
        field = field.to(level=int(level))
    placement = _dimensioned_placement(placement, field)
    level_place = _level_placement(field, placement)
    return PlacedOmeField(
        transformations=[level_place.inverse(), field, level_place],
        placement=placement,
        input=placement.output,
        output=placement.output,
    )


def _dimensioned_placement(
    placement: Transformation, field: Transformation
) -> Transformation:
    # A placement must supply a linear part for the voxel-to-voxel
    # normalization. An undimensioned identity or scaling supplies none,
    # so its size is taken from the field's array, which is
    # `(*grid_shape, ndim)`, and an identity placement of that size is
    # used in its place.
    try:
        _linear_part(placement)
        return placement
    except OmePlacementError:
        pass
    array = getattr(field, "field", None)
    if array is None:
        raise OmePlacementError(
            "This OME field has no placement of a defined size, and its "
            "array is empty, so the number of dimensions cannot be "
            "determined. Give the placement a defined affine "
            "transformation."
        )
    ndim = int(array.shape[-1])
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=placement.input,
        output=placement.output,
    )


def _level_placement(
    field: Transformation, placement: Transformation
) -> Transformation:
    # The voxel-to-world transformation of the active level, obtained by
    # composing the finest level's placement with the transformation that
    # maps the active level's grid to the finest grid.
    transforms = getattr(field, "level_transforms", None) or []
    level = getattr(field, "level", 0)
    if level < len(transforms) and transforms[level] is not None:
        cross = transforms[level]
        if not is_identity(cross):
            return (placement @ cross).compute()
    return placement


def normalize_ome_displacement(
    displacement: ArrayProtocol, placement: Transformation
) -> ArrayProtocol:
    """Convert a world-space displacement field to voxel-to-voxel form.

    An OME displacement field stores each displacement vector in world
    units. Inside the sandwich the field must work in voxel units, so
    each displacement is mapped through the inverse of the linear part of
    the placement. With `L` the linear part of `xform_0`, the voxel-space
    field is `displacement @ inverse(L).T`.

    This rescaling has no effect when the placement is an isotropic
    scaling by one, so a field on a one-millimetre isotropic grid reads
    the same whether or not it is applied. On an anisotropic or rotated
    grid the rescaling is essential.
    """
    matrix = _linear_part(placement)
    ab = get_array_backend(matrix)
    inverse_linear = ab.linalg.inv(matrix)
    displacement = ab.asarray(displacement)
    return displacement @ inverse_linear.T


def normalize_ome_coordinates(
    coordinates: ArrayProtocol, placement: Transformation
) -> ArrayProtocol:
    """Convert a world-space coordinate field to voxel-to-voxel form.

    An OME coordinate field stores each coordinate in world units. Inside
    the sandwich the field must produce voxel coordinates, so each
    coordinate is mapped through the inverse of the placement. With
    `xform_0` the placement, the voxel-space field is
    `inverse(xform_0)(coordinates)`.

    This mapping has no effect when the placement is the identity, so a
    field on a one-millimetre isotropic grid at the origin reads the same
    whether or not it is applied. On an anisotropic, shifted, or rotated
    grid the mapping is essential.
    """
    affine = placement.compute().to(Affine)
    inverse = affine.inverse()
    matrix = inverse.matrix
    ab = get_array_backend(matrix)
    coordinates = ab.asarray(coordinates)
    return coordinates @ matrix[:, :-1].T + matrix[:, -1]


def check_ome_axes(axes: tx.Optional[tx.Sequence[tx.Any]]) -> str:
    """Return the kind of an OME field, or refuse an unreadable one.

    The axes are classified by the `type` of each axis. A field carries
    every input `space` axis together with exactly one vector axis. A
    single `displacement` axis makes a displacement field, and a single
    `coordinate` axis makes a coordinate field. A field whose axes mix
    the two types, or that repeats either type, is refused. A field whose
    axes name no vector axis at all is refused as malformed. Both
    refusals raise an [`OmePlacementError`][].
    """
    kind, _ = _ome_axis_kind(axes)
    if kind == "mixed":
        raise OmePlacementError(
            "This OME field mixes displacement axes and coordinate axes "
            "in a single field, which cannot be read as one field. Store "
            "the displacement axes and the coordinate axes as separate "
            "fields."
        )
    if kind == "malformed":
        raise OmePlacementError(
            "This OME field names no displacement axis and no coordinate "
            "axis, so its vector components cannot be identified. A field "
            "must carry exactly one axis of type displacement or "
            "coordinate."
        )
    return kind


def ome_vector_axis(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Optional[int]:
    """Return the index of the axis that holds an OME field's vectors.

    The vector axis is the single `displacement` or `coordinate` axis of
    the field. The result is `None` when no axes are given, which reads
    the field with its vector components on the last array axis.
    """
    check_ome_axes(axes)
    _, index = _ome_axis_kind(axes)
    return index


def _multiscale_element(
    transformation: Transformation,
) -> tx.Optional[Transformation]:
    # The multiscale field carried by a transformation, or `None`. A field
    # is recognized by its levels, and is searched for through any nesting
    # of sequences rather than by position, so a composed or wrapped
    # sandwich still finds it.
    if getattr(transformation, "levels", None):
        return transformation
    if isinstance(transformation, Sequence):
        for part in transformation.transformations or []:
            found = _multiscale_element(part)
            if found is not None:
                return found
    return None


def resolve_multiscale_level(
    transformation: Transformation, target: Transformation
) -> Transformation:
    """Activate the resolution-matched level of a placed multiscale field.

    The `transformation` is a voxel-to-world transformation that may
    contain a multiscale field placed by
    [`place_ome_field`][brainhops.datamodel.transformations.place_ome_field].
    The `target` is the voxel-to-world transformation of the grid onto
    which an image is being resliced. The level whose world resolution is
    closest to the target grid is activated, and the sandwich is rebuilt
    with that level and its placement.

    Level selection needs the field's placement, so the target resolution
    can be expressed relative to the finest grid. The placement is read
    from the `placement` attribute that a placed field carries, not
    recovered from the sequence by position. A transformation that carries
    no multiscale field, and a bare multiscale field with no known
    placement, are both returned unchanged. A bare field is left on its
    finest level, which reslices correctly at any target resolution.
    """
    field = _multiscale_element(transformation)
    if field is None:
        return transformation
    placement = getattr(transformation, "placement", None)
    if placement is None:
        return transformation
    target_scale = _affine_scale(target)
    finest_scale = _affine_scale(placement)
    relative = target_scale / finest_scale
    level = multiscale_level_for(field, relative)
    return place_ome_field(field, placement, level=level)


def check_ome_displacement_placement(placement: Transformation) -> None:
    """Confirm that a displacement field's placement is affine.

    A displacement field is placed by mapping its world-unit vectors
    through the linear part of the placement, so the placement must reduce
    to an affine transformation. An identity placement, and a
    [`CartesianField`][] placement, are pure regrids that need no
    rescaling and are accepted. Any other placement is reduced to an
    [`Affine`][]. A placement that cannot be reduced, such as another
    displacement field, is refused with an [`OmePlacementError`][].
    """
    if isinstance(placement, CartesianField) or is_identity(placement):
        return
    try:
        affine = placement.compute().to(Affine)
    except ConversionError as error:
        raise OmePlacementError(
            "This OME displacement field is placed by a transformation "
            "that cannot be reduced to an affine, so the displacements "
            "cannot be rescaled to voxel units. Only an affine placement "
            "is supported for a displacement field."
        ) from error
    if not isinstance(affine, Affine) or affine.matrix is None:
        raise OmePlacementError(
            "This OME displacement field is placed by a transformation "
            "that cannot be reduced to an affine, so the displacements "
            "cannot be rescaled to voxel units. Only an affine placement "
            "is supported for a displacement field."
        )


# ----------------------------------------------------------------------
#    KIND CHECKS
# ----------------------------------------------------------------------


def is_identity(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is the identity.

    A transformation is recognized as the identity when its parameters
    are unset, or when it is an instance of [`Identity`][]. When
    `compute` is true, the parameters of a transformation such as
    [`Translation`][], [`Scaling`][], [`Permutation`][], [`Linear`][],
    [`Affine`][], or [`DisplacementField`][] are also inspected, so that
    a transformation whose parameters happen to encode the identity is
    recognized as such even though it is not stored as one.

    A [`CartesianField`][] is a regular grid of coordinates, which is the
    identity map over its grid by construction. When `compute` is true, a
    [`CartesianField`][] is therefore recognized as the identity. When
    `compute` is false, a [`CartesianField`][] that carries a grid is not
    recognized as the identity, because its `field` parameter is set. A
    [`CartesianField`][] with no grid has an unset `field` parameter and
    is recognized as the identity by the parameter check under either
    value of `compute`.

    Recognizing a grid as the identity does not mean a grid may be dropped
    on sight. A grid also defines the sampling domain onto which data is
    resampled. The decision to factor a grid away is made by the sequence
    simplifier, which only does so for a grid that sits strictly between
    two other transformations.
    """
    parameter_names = getattr(xform, "parameter_names", ())
    if isinstance(parameter_names, str):
        parameter_names = (parameter_names,)
    if all(getattr(xform, param) is None for param in parameter_names):
        return True
    if isinstance(xform, Identity):
        return True
    if not compute:
        return False
    if isinstance(xform, Translation):
        return (xform.translation == 0).all()
    if isinstance(xform, Scaling):
        return (xform.scale == 1).all()
    if isinstance(xform, Permutation):
        ndim = len(xform.permutation)
        return (xform.permutation == list(range(ndim))).all()
    if isinstance(xform, Linear):
        ndim = xform.matrix.shape[0]
        ab = get_array_backend(xform.matrix)
        return (xform.matrix == ab.eye(ndim)).all()
    if isinstance(xform, Affine):
        ndim = xform.matrix.shape[0] - 1
        ab = get_array_backend(xform.matrix)
        return (xform.matrix == ab.eye(ndim + 1)[:-1]).all()
    if isinstance(xform, DisplacementField):
        return (xform.field == 0).all()
    if isinstance(xform, CartesianField):
        return True
    return False


def is_translation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure translation.

    A transformation is recognized as a translation when it is an
    instance of [`Translation`][], or when [`is_identity`][] recognizes
    it as the identity, which is itself a translation by zero. When
    `compute` is true, the matrix of an [`Affine`][] transformation is
    also inspected for a linear part equal to the identity.
    """
    if isinstance(xform, Translation):
        return True
    if compute and isinstance(xform, Affine) and xform.matrix is not None:
        return (xform.matrix[:, :-1] == 0).all()
    return is_identity(xform, compute=compute)


def is_scale(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure scaling.

    A transformation is recognized as a scaling when it is an instance
    of [`Scaling`][], or when [`is_identity`][] recognizes it as the
    identity, which is itself a scaling by one. When `compute` is true,
    the matrix of a [`Linear`][] or [`Affine`][] transformation is also
    inspected for a diagonal structure.
    """
    if isinstance(xform, Scaling):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        ndim = matrix.shape[0]
        ab = get_array_backend(matrix)
        return not (matrix * (1 - ab.eye(ndim))).any()
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_scale(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_permutation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure permutation of axes.

    A transformation is recognized as a permutation when it is an
    instance of [`Permutation`][], or when [`is_identity`][] recognizes
    it as the identity, which is itself a trivial permutation. When
    `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for a binary, one-per-row and
    one-per-column structure.
    """
    if isinstance(xform, Permutation):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        ab = get_array_backend(matrix)
        is_binary = ab.isin(matrix, [0, 1]).all()
        is_perm = matrix.sum(axis=0) == 1 and matrix.sum(axis=1) == 1
        return is_binary and is_perm
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_permutation(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_rotation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure rotation.

    A transformation is recognized as a rotation when it is an instance
    of [`Rotation`][], or when [`is_identity`][] recognizes it as the
    identity, which is itself a rotation by zero. When `compute` is
    true, the matrix of a [`Linear`][] or [`Affine`][] transformation is
    also inspected for orthogonality and a positive determinant.
    """
    if isinstance(xform, Rotation):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        ndim = matrix.shape[0]
        ab = get_array_backend(matrix)
        is_orthogonal = (matrix @ matrix.T == ab.eye(ndim)).all()
        is_posdef = ab.linalg.det(matrix) > 0
        return is_orthogonal and is_posdef
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_rotation(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_linear(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is linear, without a translation.

    A transformation is recognized as linear when it is an instance of
    [`Linear`][], or when [`is_identity`][] recognizes it as the
    identity, which is itself linear. When `compute` is true, the matrix
    of an [`Affine`][] transformation is also inspected for a zero
    translation component.
    """
    if isinstance(xform, Linear):
        return True
    if compute and isinstance(xform, Affine) and xform.matrix is not None:
        matrix = xform.matrix
        no_translation = (matrix[:, -1] == 0).all()
        return no_translation
    return is_identity(xform, compute=compute)


# ----------------------------------------------------------------------
#    SEQUENCE COMPUTATION
# ----------------------------------------------------------------------


_IDENTITIES = {"identity"}
_TRANSLATIONS = {*_IDENTITIES, "translation"}
_SCALES = {*_IDENTITIES, "scale"}
_PERMUTATIONS = {*_IDENTITIES, "permutation"}
_LINEARS = {*_SCALES, *_PERMUTATIONS, "linear"}
_AFFINES = {*_LINEARS, *_TRANSLATIONS, "affine"}
_NONLINEARS = {*_AFFINES, "displacements", "coordinates"}
_XFORMHIERARCHY = {
    "identity": _IDENTITIES,
    "translation": _TRANSLATIONS,
    "scale": _SCALES,
    "permutation": _PERMUTATIONS,
    "linear": _LINEARS,
    "affine": _AFFINES,
    "nonlinear": _NONLINEARS,
}


def _is_flat(self: Sequence) -> bool:
    # Check if the sequence is flat (does not contain any nested sequences).
    if self.transformations is None:
        return True
    return all(not isinstance(t, Sequence) for t in self.transformations)


_ModePair = tx.Tuple[tx.Type[hierarchy.Transformation], tx.Optional[int]]


def _matches_mode(t: Transformation, mode: _ModePair) -> bool:
    # FIXME
    #   In many transforms, the ndim can be guessed from the content
    #   of the xform, even if the input/output spaces are not set
    #   (eg. the shape of the matri or the field).
    #
    #   The current implementation is a stricter bound.
    cls, ndim = mode
    if not isinstance(t, cls):
        return False
    if ndim is None:
        return True
    if t.input is None or t.output is None:
        return False
    if t.input.axes is None or t.output.axes is None:
        return False
    return len(t.input.axes) == ndim and len(t.output.axes) == ndim


def _mode_children(mode: _ModePair) -> list:
    children = []
    seen = set()
    cls, ndim = mode
    for child in cls.__subclasses__():
        if (child, ndim) not in seen:
            seen.add((child, ndim))
            children.append((child, ndim))
    return children


def _is_proper_mode(mode: ModeLike) -> bool:
    """
    A proper mode is a tuple (type, ndim),
    where type is a subclass of `hierarchy.Transformation`
    and ndim is an int or None.
    """
    if not isinstance(mode, tuple):
        return False
    if len(mode) != 2:
        return False
    if not isinstance(mode[0], type):
        return False
    if not isinstance(mode[1], (int, type(None))):
        return False
    return True


def _ensure_proper_modes(mode: ModeLike) -> tx.List[_ModePair]:
    """
    Convert any (list of) mode-like input into a list of (type, ndim) pairs.

    !!! note "An empty list of modes yields a no-op"
    """

    if mode is None:
        # Default case
        mode = [hierarchy.Transformation]
    elif isinstance(mode, (str, int, type)):
        # We know that these are single modes -> wrap them already
        mode = [mode]
    elif _is_proper_mode(mode):
        # Already a proper mode -> wrap it
        mode = [mode]

    # Convert each element to a proper mode = a (type, ndim) pair
    return [
        hierarchy.parseType(m) if not _is_proper_mode(m) else m for m in mode
    ]


def _drop_interior_grids(seq: Sequence) -> Sequence:
    """Remove every strictly interior grid from a sequence.

    A [`CartesianField`][] is the identity map over its grid. A grid that
    sits strictly between two other transformations is therefore
    redundant. The transformation before it and the transformation after
    it overwrite its coordinates, so `A @ grid @ B` computes the same
    field as `A @ B`. Each such interior grid is removed, which lets the
    two neighbours compose directly.

    The first element, the last element, and a standalone element are left
    in place. A grid in one of those positions defines the sampling domain
    onto which data is resampled, and removing it would lose that domain.
    A sequence of one or two transformations has no interior, so it is
    returned unchanged.

    Only a [`CartesianField`][] is removed, because only a grid is the
    identity map by construction. A general field of coordinates or a
    field of displacements carries its own values, which the neighbours do
    not reproduce, so such a field is kept wherever it appears.
    """
    xforms = seq.transformations or []
    if len(xforms) <= 2:
        return seq
    kept = [xforms[0]]
    for elem in xforms[1:-1]:
        if isinstance(elem, CartesianField) and is_identity(
            elem, compute=True
        ):
            continue
        kept.append(elem)
    kept.append(xforms[-1])
    if len(kept) == len(xforms):
        return seq
    return replace(seq, transformations=kept)


def _compute_sequence(
    seq: Sequence,
    mode: tx.List[_ModePair],
    memo: tx.Optional[tx.Set[_ModePair]] = None,
) -> Transformation:
    # For now, let's make simple assumptions:
    #
    # * coordinate systems are compatible
    #   - there is the same number of axes in the input and output spaces.
    #   - the axes of the input and output spaces match.
    #   - there is no need to check them (they may not even be defined)
    #
    # * we can safely "forget" the coordinate systems of the sequence
    #   object (i.e., they are also contained in the nested transformations)
    #
    # * we optimize by recursively finding the subclasses of all the modes
    #   specified. This allows us to combine similar transformations first
    #   before combining vauge transformations. For example say the user
    #   lists Affine as the mode. This will then recursively call this function
    #   with all subclasses then all subclasses of subclasses etc. in a depth
    #   first search fashion. This means if there are translations that are
    #   next to each other in the sequence it will combine the translations
    #   before combining any of the affines.

    # --- Flatten sequence
    if not _is_flat(seq):
        seq = seq._flattened()

    # --- Check if nothing to do
    if len(seq.transformations or []) < 1:
        return seq

    # --- If we are called from the public method, `mode`` is a `list`
    # > Recuerse with a memo
    if memo is None:
        # Factor away any strictly interior grid before composing. An
        # interior `CartesianField` is the identity map over its grid, and
        # its neighbours overwrite those coordinates, so it is redundant
        # and is removed. The first and last elements define the sampling
        # domain and are left in place.
        seq = _drop_interior_grids(seq)
        memo = set()
        for submode in mode:
            seq = _compute_sequence(seq, submode, memo=memo)
        return seq

    # --- Otherwise, `mode` is a single mode
    # > if the mode has been seen, return the sequence as is
    if mode in memo:
        return seq

    # --- Else compute all children of the mode
    children = _mode_children(mode)
    for child in children:
        seq = _compute_sequence(seq, child, memo=memo)
        if not isinstance(seq, Sequence):
            # NOTE(YB): why this?????
            return seq

    # Mark that we've been through this mode
    # !!! DO NOT RETURN HERE
    memo.add(mode)

    # --- Compose transformations that belong to this mode in order

    # Ensure a list of transformations (and make a copy so we can pop it)
    inputs = list(getattr(seq, "transformations", [seq]))
    outputs = []

    # Compose any consecutive sequence that matches the mode
    while inputs:
        item = inputs.pop(0)
        if _matches_mode(item, mode):
            while inputs and _matches_mode(inputs[0], mode):
                # NOTE: we compose to the left ! (see sequence definition)
                next_input = inputs.pop(0)
                try:
                    item = _compose(next_input, item)
                except CompositionError:
                    # NOTE(YB):
                    # When does this happen? When we don't know how to adapt?
                    outputs.append(item)
                    item = next_input
        outputs.append(item)

    # Return
    if len(outputs) == 1:
        return outputs[0]
    return Sequence(transformations=outputs)


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _distance(t1: type, t2: type, oriented: bool = True) -> int:
    """Compute the distance between two types in the class hierarchy."""
    # TODO: handle type hints (Union, Any)
    if t1 == t2:
        return 0
    if issubclass(t1, t2):
        return 1 + min(map(partial(_distance, t2=t2), t1.__bases__))
    if issubclass(t2, t1) and not oriented:
        return 1 + min(map(partial(_distance, t2=t1), t2.__bases__))
    return float("inf")


def _get_ndim(
    t: Transformation, default: tx.Optional[int] = None
) -> tx.Optional[int]:
    if t.input and t.input.axes is not None:
        return len(t.input.axes)
    if t.output and t.output.axes is not None:
        return len(t.output.axes)
    return default


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------
_CONVERTERS = {}
_CONVERTERS_FASTMAP = {}


class ConversionError(TypeError):
    """Raised when a transformation cannot be converted to another type."""


class LossyConversionError(ConversionError):
    """Raised when a conversion would discard information.

    This error is raised by [`Transformation.to`][] when a conversion is
    only possible at the cost of losing information, and the conversion
    was not explicitly allowed to be lossy. The `result` attribute holds
    the transformation that the conversion would have produced.
    """

    def __init__(
        self,
        *args,
        result: tx.Optional[Transformation] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.result = result


@tx.overload
def _converter(
    Ti: tx.Type[Transformation], To: tx.Type[Transformation]
) -> tx.Callable:
    """Return a decorator to register a converter of between types."""
    ...


@tx.overload
def _converter(func: tx.Callable) -> tx.Callable:
    """Register a converter of between types (based on hints)."""
    ...


def _converter(*args, **kwargs) -> tx.Callable:
    """Decorator to register a converter of between transformation types."""
    if len(args) == 2:
        Ti, To = args
        return partial(_converter, Ti=Ti, To=To)
    func = args[0]
    if kwargs:
        types = kwargs["Ti"], kwargs["To"]
    else:
        types = tuple(tx.get_type_hints(func).values())
    _CONVERTERS[types] = func
    _CONVERTERS_FASTMAP.clear()
    return func


def _to(
    x: Transformation, cls: tx.Type[Transformation], **kwargs
) -> Transformation:
    """Convert a transform to a different type."""
    # TODO: implement using the CONVERTERS map,
    #       similarly to _compose() and _COMPOSERS.
    t1, t2 = type(x), cls
    if (t1, t2) in _CONVERTERS_FASTMAP:
        func = _CONVERTERS_FASTMAP[(t1, t2)]
        return func(x, **kwargs)
    best_distance, best_func = float("inf"), None
    for (T1, T2), FUNC in _CONVERTERS.items():
        if not isinstance(T1, type):
            T1 = type(T1)
        if not isinstance(T2, type):
            T2 = type(T2)
        distance = _distance(t1, T1) + _distance(t2, T2)
        if distance < best_distance:
            best_distance, best_func = distance, FUNC
    if best_distance < float("inf"):
        _CONVERTERS_FASTMAP[(t1, t2)] = best_func
        return best_func(x, **kwargs)
    raise ConversionError(f"No converter found for: {t1} -> {t2}")


# ----------------------------------------------------------------------
#   COMPOSITIONS
# ----------------------------------------------------------------------
_COMPOSERS = {}
_COMPOSERS_FASTMAP = {}


class CompositionError(TypeError):
    """Raised when two transformations cannot be composed."""


def _composer(func: tx.Callable) -> tx.Callable:
    """
    Decorator to register a function as a composer of two transformations.
    """
    types = tuple(tx.get_type_hints(func).values())[:2]
    _COMPOSERS[types] = func
    _COMPOSERS_FASTMAP.clear()
    return func


def _compose(x1: Transformation, x2: Transformation) -> Transformation:
    """
    Dispatch the composition of two transformations to the appropriate
    composer function.
    """
    t1, t2 = type(x1), type(x2)
    if (t1, t2) in _COMPOSERS_FASTMAP:
        func = _COMPOSERS_FASTMAP[(t1, t2)]
        return func(x1, x2)
    best_distance, best_func = float("inf"), None
    for (T1, T2), FUNC in _COMPOSERS.items():
        origin1 = safe_get_origin(T1)
        if origin1 is tx.Union or (
            _UnionType is not None and origin1 is _UnionType
        ):
            T1s = tx.get_args(T1)
        else:
            T1s = (T1,)
        origin2 = safe_get_origin(T2)
        if origin2 is tx.Union or (
            _UnionType is not None and origin2 is _UnionType
        ):
            T2s = tx.get_args(T2)
        else:
            T2s = (T2,)
        for T1, T2 in itertools.product(T1s, T2s):
            distance = _distance(t1, T1) + _distance(t2, T2)
            if distance < best_distance:
                best_distance, best_func = distance, FUNC
    if best_distance < float("inf"):
        _COMPOSERS_FASTMAP[(t1, t2)] = best_func
        return best_func(x1, x2)
    raise CompositionError(f"No composer found for types: {t1}, {t2}")


# ----------------------------------------------------------------------
#   ADAPTORS
# ----------------------------------------------------------------------
_ADAPTORS = {}
_ADAPTORS_FASTMAP = {}


class AdaptationError(TypeError):
    """Raised when no transformation adapts one coordinate system to
    another."""


def _adaptor(func: tx.Callable) -> tx.Callable:
    """
    Decorator to register a function as an adaptor between two
    coordinate systems.
    """
    types = tuple(tx.get_type_hints(func).values())[:2]
    _ADAPTORS[types] = func
    _ADAPTORS_FASTMAP.clear()
    return func


def _adapt(s1: CoordinateSystem, s2: CoordinateSystem) -> Transformation:
    """
    Dispatch the adaptation between two coordinate systems to the appropriate
    adaptor function.
    """
    if (s1, s2) in _ADAPTORS_FASTMAP:
        func = _ADAPTORS_FASTMAP[(s1, s2)]
        return func(s1, s2)
    best_distance, best_func = float("inf"), None
    for (S1, S2), FUNC in _ADAPTORS.items():
        distance = _distance(s1, S1) + _distance(s2, S2)
        if distance < best_distance:
            best_distance, best_func = distance, FUNC
    if best_distance < float("inf"):
        _ADAPTORS_FASTMAP[(s1, s2)] = best_func
        return best_func(s1, s2)
    raise AdaptationError(f"No adaptor found for types: {s1}, {s2}")
