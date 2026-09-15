__all__ = [
    "Transformation",
    "CoordinatesField",
    "CartesianField",
    "DisplacementField",
    "ImmutableSequenceMixin",
    "Multiscale",
    "MultiscaleField",
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
    "is_identity",
    "is_translation",
    "is_scale",
    "is_permutation",
    "is_rotation",
    "is_linear",
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
from brainhops._core.affines import axis_scales
from brainhops._core.affines import inv as _affine_inv
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
#    MULTISCALE
# ----------------------------------------------------------------------


class ImmutableSequenceMixin:
    """A [`Sequence`][] whose contents cannot be edited in place.

    A subclass mixes this in ahead of the mutable sequence base to block
    item assignment, deletion, and insertion. Every in-place edit raises
    `TypeError`.

    The data model regenerates the subscript hooks on every class it
    builds, so a subclass restates `__setitem__` and `__delitem__` in its
    own body, bound to the versions defined here.
    """

    def _refuse_in_place_edit(self, *args: tx.Any) -> tx.NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be edited in place.")

    __setitem__ = _refuse_in_place_edit
    __delitem__ = _refuse_in_place_edit
    insert = _refuse_in_place_edit


class Multiscale(DataModelBase):
    """A pyramid of resolution scales, ordered from finest to coarsest.

    This mixin gives a transformation a list of scales and the operations
    that select one of them. The scales are ordered from finest to
    coarsest, so the first scale is the highest resolution one.

    The mixin does not say what a scale is. A subclass supplies the scales
    and, through the `_level_resolution` hook, the physical grid size of
    each one. With those, `_nearest_level` picks the scale whose
    resolution is closest to a target grid.
    """

    scales: tx.Annotated[
        tx.Optional[tx.List[tx.Any]],
        tx.Doc("The resolution scales, ordered from finest to coarsest."),
    ] = None

    @property
    def nscales(self) -> int:
        """The number of resolution scales."""
        return len(self.scales or [])

    @property
    def _finest(self) -> tx.Any:
        # The finest resolution scale, or `None` when there are no scales.
        scales = self.scales or []
        return scales[0] if scales else None

    def to_singlescale(self, index: int = 0) -> tx.Any:
        """Return the resolution scale at a given index.

        Index `0` is the finest scale. The scale is returned as it is
        stored, so for a field it is the plain transformation of that
        scale rather than the multiscale field.
        """
        return (self.scales or [])[int(index)]

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # The physical grid size of a level, as a per-axis vector in the
        # input units of the multiscale. `None` means the resolution of
        # the level is unknown. A subclass overrides this hook.
        return None

    def _nearest_level(self, voxel2world: "Transformation") -> int:
        # The index of the level whose resolution is closest to the grid
        # described by `voxel2world`. The comparison is made in
        # logarithmic scale, so the level above and the level below the
        # target are weighed evenly, and the finer level wins a tie. When
        # the resolution of any level, or of the target, is unknown, the
        # finest scale is returned.
        scales = self.scales or []
        if len(scales) <= 1:
            return 0
        resolutions = [self._level_resolution(i) for i in range(len(scales))]
        if any(resolution is None for resolution in resolutions):
            return 0
        target = axis_scales(_affine_matrix(voxel2world))
        if target is None:
            return 0
        ab = get_array_backend()
        log_target = float(ab.log(ab.abs(ab.asarray(target))).mean())
        best_index, best_distance = 0, None
        for index, resolution in enumerate(resolutions):
            log_resolution = float(
                ab.log(ab.abs(ab.asarray(resolution))).mean()
            )
            distance = abs(log_resolution - log_target)
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        return best_index


class MultiscaleField(Multiscale, ImmutableSequenceMixin, Sequence):
    """A field of coordinates or displacements at several resolutions.

    Each scale is a [`Sequence`][] that maps the multiscale's input space
    to its output space, sampled on that scale's grid. A coordinate scale
    is a two-element sequence of a world-to-voxel affine and a field of
    coordinates. A displacement scale is a three-element sequence of a
    world-to-voxel affine, a field of displacements in voxel units, and
    the voxel-to-world affine. The container treats a scale as a plain
    sequence, so the same class carries both kinds.

    A `MultiscaleField` behaves as its finest scale. It composes with
    other transformations exactly as the finest scale would, and reduces
    to the finest scale when it is computed. The finest scale is the one
    used unless a scale is selected with `to_singlescale`.

    The scales replace this class as the unit that is edited. A scale is
    selected with `to_singlescale`, and the returned sequence is edited in
    place. The container itself does not support item assignment,
    insertion, or deletion.
    """

    # The subscript hooks are regenerated on every class the data model
    # builds, so the immutable ones from `ImmutableSequenceMixin` are restated
    # here to keep them. `insert` is not regenerated and is inherited.
    __setitem__ = ImmutableSequenceMixin.__setitem__
    __delitem__ = ImmutableSequenceMixin.__delitem__

    scales: tx.Annotated[
        tx.Optional[tx.List[Sequence]],
        tx.Doc(
            "The resolution scales, ordered from finest to coarsest. Each "
            "scale is a sequence that maps the input space to the output "
            "space, sampled on that scale's grid."
        ),
    ] = None

    # `transformations` is served on demand from the finest scale rather
    # than stored, so it is not a constructor-taken field here. Declaring
    # it a `ClassVar` overrides the inherited init-field from `Sequence`
    # and keeps it out of `__init__`, `fields()` and `replace()`, while
    # the property keeps the container reading as the finest scale.
    transformations: tx.ClassVar[tx.Optional[tx.List[Transformation]]]

    @property
    def transformations(self) -> tx.Optional[tx.List[Transformation]]:
        """The transformations of the finest scale."""
        finest = self._finest
        return finest.transformations if finest is not None else None

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[Transformation]]
    ) -> None:
        if value is not None:
            raise TypeError(
                "The elements of a multiscale field are determined by its "
                "scales. Select a scale with to_singlescale() and edit that "
                "sequence."
            )

    def _as_sequence(self) -> Sequence:
        # The finest scale as a plain sequence, carrying the container's
        # input and output. `compute` and `_flattened` go through this, so
        # they operate on a real init-field sequence rather than on the
        # container, whose `transformations` is derived and cannot be
        # rebuilt by `replace`.
        finest = self._finest
        transformations = (
            None if finest is None else list(finest.transformations or [])
        )
        return Sequence(
            transformations=transformations,
            input=self.input,
            output=self.output,
        )

    def compute(self, mode: tx.Optional[ModeLike] = None) -> Transformation:
        """Compute the field as a plain transformation.

        The finest scale is composed and returned. The result is an
        ordinary transformation, with no pyramid, so it computes exactly
        as the finest scale would on its own.
        """
        return self._as_sequence().compute(mode)

    def _flattened(self) -> Sequence:
        # Flatten through the finest scale. A multiscale field spliced
        # into a surrounding sequence contributes the elements of its
        # finest scale.
        return self._as_sequence()._flattened()

    def inverse(self) -> tx.Self:
        return replace(
            self,
            scales=[scale.inverse() for scale in (self.scales or [])],
            input=self.output,
            output=self.input,
        )

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # The physical grid size of a scale, read from the scale's leading
        # world-to-voxel affine. The inverse of that affine is the scale's
        # voxel-to-world transformation, and the norm of each of its
        # columns is the voxel size along one axis. A scale whose leading
        # element is not a defined affine has an unknown resolution.
        scales = self.scales or []
        if not 0 <= index < len(scales):
            return None
        scale = scales[index]
        if not len(scale):
            return None
        matrix = _affine_matrix(scale[0])
        if matrix is None:
            return None
        return axis_scales(_affine_inv(matrix))


def _affine_matrix(xform: Transformation) -> tx.Optional[npmatrix]:
    # The compact affine matrix of a transformation, or `None`. A
    # transformation that does not reduce to an affine with a defined
    # matrix has no matrix, and is reported as `None` rather than refused.
    try:
        affine = xform.compute().to(Affine)
    except ConversionError:
        return None
    if not isinstance(affine, Affine) or affine.matrix is None:
        return None
    return affine.matrix


def _at_resolution(
    transformation: Transformation, voxel2world: Transformation
) -> Transformation:
    # Select, inside a transformation, the scale of every multiscale field
    # whose resolution matches a target grid. The `voxel2world` is the
    # voxel-to-world transformation of the grid onto which an image is
    # resliced. Each multiscale field is replaced by its resolution-
    # matched scale, walking through any nesting of sequences. A
    # transformation that carries no multiscale field is returned
    # unchanged.
    if isinstance(transformation, Multiscale):
        return transformation.to_singlescale(
            transformation._nearest_level(voxel2world)
        )
    if isinstance(transformation, Sequence):
        parts = transformation.transformations or []
        resolved = [_at_resolution(part, voxel2world) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(transformation, transformations=resolved)
        return transformation
    return transformation


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
