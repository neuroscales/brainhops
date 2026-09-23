__all__ = [
    "ConcreteTransformation",
    "CoordinatesField",
    "CartesianField",
    "DisplacementField",
    "Affine",
    "Linear",
    "Rotation",
    "Permutation",
    "Scaling",
    "Translation",
    "Identity",
]

# stdlib
from numbers import Integral, Real

# dependencies
import typing_extensions as tx
from bagof.magic import fields

# core
from brainhops._core.typing import ArrayProtocol, npmatrix, npvector

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel import hierarchy
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# transformations
from . import registries
from .base import Transformation
from .meta import SubspaceTransformation
from .modes import ModeLike, _ensure_proper_modes, _mode_admits

# typing
if tx.TYPE_CHECKING:
    from .inverse import Inverse


def _pins_endpoints(cls: type) -> bool:
    # Whether `cls` pins its endpoints to fixed coordinate systems
    # instead of leaving them open. `Magic` clears the class body, so a
    # class-level default is not visible in `cls.__dict__`; it survives
    # on the built field, either as a per-instance factory (`build`) --
    # which is how a mutable default such as a coordinate system is
    # stored -- or as a plain non-`None` `default`. A class that leaves
    # its endpoints open inherits `Transformation`'s `= None` default
    # and builds nothing, so it reads as `False` here. The check is
    # inherited along with the fields, so a subclass of a pinned class
    # is pinned too.
    for field in fields(cls):
        if field.name in ("_input", "_output"):
            if field.build or field.default is not None:
                return True
    return False


def _unpinned_base(cls: type) -> type:
    # The nearest ancestor of `cls` that does not pin its endpoints, or
    # `cls` itself when `cls` does not pin them either. The walk stops
    # short of `ConcreteTransformation`, which parameterizes nothing, so
    # a class with no unpinned concrete ancestor is returned unchanged.
    for base in cls.__mro__:
        if base is ConcreteTransformation:
            break
        if not issubclass(base, ConcreteTransformation):
            continue
        if not _pins_endpoints(base):
            return base
    return cls


class _LazyInverseMixin:
    # The typed inverse that represents the inverse of this type. Each
    # `Inverse` subclass names the forward type it inverts in its
    # `_inverseof`, and hands that type this back-pointer as it is
    # created, so the pairing is declared once and read back by plain
    # attribute lookup. Inheritance then serves a refinement for free: a
    # reader's `LPSToVoxel(Affine)` finds `InverseAffine` on `Affine`, and
    # the most derived base wins, since a `MyRotation(Rotation)` finds
    # `Rotation`'s `InverseRotation` before `Affine`'s. A forward type
    # that has no typed inverse -- or that opts out of its base's by
    # setting this back to `None` in its own body -- raises instead.
    _inverse_type: tx.ClassVar[tx.Optional[tx.Type["Inverse"]]] = None

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        # The shared `inverse()` of every forward type that defers its
        # inversion to a type-transparent `Inverse` wrapper. A transformation
        # with an unset parameter has nothing to invert, so its inverse is the
        # plain endpoint-swapped transform. Otherwise the wrapper for this type
        # is built, holding the transform as its `forward` and materializing
        # the inverse only when its parameter is read or it is computed.
        cls = type(self)
        param = cls.parameter_names
        if getattr(self, param) is None:
            # STOPGAP. Swapping the endpoints of a class that pins them
            # -- `VoxelToLPS` and friends declare `_input`/`_output` as
            # class-level defaults -- produces the right endpoints under
            # a type whose name states the opposite direction, which
            # misleads anything that dispatches on the type. Until the
            # inversion rework decides whether such a class may be
            # endpoint-swapped at all (and, if so, how it names the
            # result), the swap is carried by the nearest ancestor that
            # pins nothing, so the type claims no direction it does not
            # have. A class that pins nothing is its own such ancestor
            # and is built exactly as before. Delete this together with
            # `_pins_endpoints`/`_unpinned_base` once that decision is
            # made.
            return _unpinned_base(cls)(input=self.output, output=self.input)
        wrapper = cls._inverse_type
        if wrapper is None:
            raise TypeError(f"{cls.__name__} has no typed inverse.")
        obj = wrapper(forward=self, input=self.output, output=self.input)
        if compute:
            obj = obj.compute(**kwargs)
        return obj


class ConcreteTransformation(Transformation):
    """Base class for concrete transformations that hold a parameter."""

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: bool = False,
    ) -> tx.Self:
        """
        Compute the transformation, downcasting it to the cheapest
        compatible kind.

        A concrete transformation holds a parameter, so it simplifies to
        the simplest compatible kind, whose compatibility can be detected
        with (almost) no overhead. For example, a transformation whose
        parameter is set to `None` is treated as an identity.

        Parameters
        ----------
        mode : [list of] str or type, optional
            Which kinds of transformations to materialize. `None` (the
            default) admits every kind. If a `mode` is given and this leaf
            is not admitted by it, the leaf is returned unchanged.
        simplify : bool, default=False
            Run the numeric kind-checks that downcast the transformation
            to the cheapest compatible type.
        """
        # A leaf that the requested mode does not admit is left untouched.
        # This mirrors how the sequence simplifier only composes
        # transformations that match the mode. The checks and the concrete
        # types both live in this module, so nothing is imported in the
        # body.
        if mode is not None and not _mode_admits(
            self, _ensure_proper_modes(mode)
        ):
            return self
        checks = [
            (is_identity, Identity),
            (is_translation, Translation),
            (is_scale, Scaling),
            (is_permutation, Permutation),
            (is_rotation, Rotation),
            (is_linear, Linear),
        ]
        for check, cls in checks:
            if check(self, compute=simplify):
                return self.to(cls)
        return self


class CoordinatesField(_LazyInverseMixin, ConcreteTransformation):
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
    ] = InterpolationOrder.linear

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

    def inverse(self) -> tx.Self:
        # Inverse is itself, with switched input and output.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)


class DisplacementField(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.AffineTransformation.register
class Affine(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.LinearTransformation.register
class Linear(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.Permutation.register
class Permutation(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.DiagonalTransformation.register
class Scaling(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.Translation.register
class Translation(_LazyInverseMixin, ConcreteTransformation):
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


@hierarchy.IdentityTransformation.register
class Identity(ConcreteTransformation):
    """An identity transformation.

    If the `input` and `output` coordinate systems are different, it maps
    the input axes to the output axes, while preserving their orders.
    """

    def inverse(self) -> tx.Self:
        cls = type(self)
        return cls(input=self.output, output=self.input)


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

    # --- Meta transformations -----------------------------------------

    # An inverse is the identity exactly when the transform it inverts
    # is, so the answer is read from `forward`. This reads neither the
    # check with `compute=False` nor the one with `compute=True` into
    # materializing the (possibly unmaterializable) inverse of a field.
    # An inverse of nothing is itself the identity.
    if isinstance(xform, registries.INVERSE):
        forward = xform.forward
        if forward is None:
            return True
        return is_identity(forward, compute=compute)

    # --- Generic check ------------------------------------------------

    # If all parameters are None -> identity.
    parameter_names = getattr(xform, "parameter_names", ())
    if isinstance(parameter_names, str):
        parameter_names = (parameter_names,)
    if all(getattr(xform, param) is None for param in parameter_names):
        return True

    # --- Typed check --------------------------------------------------

    if isinstance(xform, hierarchy.IdentityTransformation):
        return True

    if not compute:
        return False

    # A subspace transform is the identity when the transform it wraps
    # is itself the identity and it reads the same axes it writes. A
    # subspace that reorders axes is not the identity even when its
    # inner transform is, so a differing pair of axis vectors keeps it
    # non-identity.
    if isinstance(xform, SubspaceTransformation):
        inner = xform.transformation
        if inner is not None and not is_identity(inner, compute=compute):
            return False
        input_axes = xform.input_axes
        output_axes = xform.output_axes
        if input_axes is None or output_axes is None:
            return True
        return list(input_axes) == list(output_axes)

    # --- Compute concrete types ---------------------------------------

    if isinstance(xform, Translation):
        return (xform.translation == 0).all()
    if isinstance(xform, Scaling):
        return (xform.scale == 1).all()
    if isinstance(xform, Permutation):
        ndim = len(xform.permutation)
        return (xform.permutation == list(range(ndim))).all()
    if isinstance(xform, Linear):
        rows, cols = xform.matrix.shape
        if rows != cols:
            # A transform between spaces of different dimension is never
            # the identity, and its matrix cannot be compared to a square
            # identity matrix.
            return False
        ab = get_array_backend(xform.matrix)
        return (xform.matrix == ab.eye(rows)).all()
    if isinstance(xform, Affine):
        rows, cols = xform.matrix.shape
        if cols != rows + 1:
            # An affine whose input and output have different dimensions is
            # never the identity. Its matrix is `(No, Ni + 1)`, so the
            # identity requires `No == Ni`.
            return False
        ab = get_array_backend(xform.matrix)
        return (xform.matrix == ab.eye(rows + 1)[:-1]).all()
    if isinstance(xform, DisplacementField):
        return (xform.field == 0).all()
    if isinstance(xform, CartesianField):
        return True
    return False


def is_translation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure translation.

    A transformation is recognized as a translation when it is an
    instance of [`hierarchy.Translation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a translation by zero.

    When `compute` is true, the matrix of an [`Affine`][] transformation
    is also inspected for a linear part equal to the identity.
    """
    if isinstance(xform, hierarchy.Translation):
        return True
    if compute and isinstance(xform, Affine) and xform.matrix is not None:
        return (xform.matrix[:, :-1] == 0).all()
    return is_identity(xform, compute=compute)


def is_scale(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure scaling.

    A transformation is recognized as a scaling when it is an instance
    of [`hierarchy.DiagonalTransformation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a scaling by one.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for a diagonal structure.
    """
    if isinstance(xform, hierarchy.DiagonalTransformation):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        rows, cols = matrix.shape
        if rows != cols:
            # A scaling maps a space onto itself, so a non-square matrix
            # (different input and output dimension) is never a scaling.
            return False
        ab = get_array_backend(matrix)
        return not (matrix * (1 - ab.eye(rows))).any()
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_scale(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_permutation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure permutation of axes.

    A transformation is recognized as a permutation when it is an
    instance of [`hierarchy.Permutation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a trivial permutation.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for a binary, one-per-row and
    one-per-column structure.
    """
    if isinstance(xform, hierarchy.Permutation):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        rows, cols = matrix.shape
        if rows != cols:
            # A permutation reorders the axes of one space, so a non-square
            # matrix is never a permutation.
            return False
        ab = get_array_backend(matrix)
        is_binary = bool(ab.isin(matrix, [0, 1]).all())
        # Reduce the row/column sums to a single truth value before the
        # `and`: comparing whole arrays with `and` raises on their
        # ambiguous truth value.
        is_perm = bool(
            (matrix.sum(0) == 1).all() and (matrix.sum(1) == 1).all()
        )
        return is_binary and is_perm
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_permutation(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_rotation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure rotation.

    A transformation is recognized as a rotation when it is an instance
    of [`hierarchy.SpecialOrthogonalTransformation`][], or when
    [`is_identity`][] recognizes it as the identity, which is itself a
    rotation by zero.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for orthogonality and a positive
    determinant.
    """
    if isinstance(xform, hierarchy.SpecialOrthogonalTransformation):
        return True
    if compute and isinstance(xform, Linear) and xform.matrix is not None:
        matrix = xform.matrix
        rows, cols = matrix.shape
        if rows != cols:
            # A rotation is orthogonal, hence square; a non-square matrix
            # is never a rotation, and its determinant is undefined.
            return False
        ab = get_array_backend(matrix)
        is_orthogonal = bool((matrix @ matrix.T == ab.eye(rows)).all())
        is_posdef = bool(ab.linalg.det(matrix) > 0)
        return is_orthogonal and is_posdef
    if isinstance(xform, Affine) and xform.matrix is not None:
        return is_linear(xform, compute=compute) and is_rotation(
            xform.to(Linear), compute=compute
        )
    return is_identity(xform, compute=compute)


def is_linear(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is linear, without a translation.

    A transformation is recognized as linear when it is an instance of
    [`hierarchy.LinearTransformation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself linear.

    When `compute` is true, the matrix of an [`Affine`][] transformation
    is also inspected for a zero translation component.
    """
    if isinstance(xform, hierarchy.LinearTransformation):
        return True
    if compute and isinstance(xform, Affine) and xform.matrix is not None:
        matrix = xform.matrix
        no_translation = (matrix[:, -1] == 0).all()
        return no_translation
    return is_identity(xform, compute=compute)


def is_affine(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is affine.

    A transformation is recognized as affine when it is an instance of
    [`hierarchy.AffineTransformation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself affine.
    """
    if isinstance(xform, hierarchy.AffineTransformation):
        return True
    return is_identity(xform, compute=compute)
