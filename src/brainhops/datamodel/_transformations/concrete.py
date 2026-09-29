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

# core
from brainhops._core.typing import ArrayProtocol, Derived, npmatrix, npvector

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# transformations
from . import registries
from .base import Transformation
from .check import is_kind
from .modes import ModeLike
from .simplify import SimplifyLike
from .simplify import simplify as _simplify


class ConcreteTransformation(Transformation):
    """Base class for concrete transformations that hold a parameter."""

    # Some transformation types come in pairs that map the same two spaces
    # in opposite directions -- `VoxelToLPS` and `LPSToVoxel` pin their
    # endpoints, and each one's name states a direction. The inverse of such
    # a type is not itself: it is the other half of the pair. The pairing is
    # declared once, on either half, by naming the other in `_reverseof`;
    # the hook below resolves it both ways into `_reverse_type`, which is
    # what `inverse()` reads.
    #
    # This is not the same relation as the wrapper that stands for an
    # inversion not yet carried out. That wrapper is an `Inverse`, picked
    # polymorphically from the transform handed to it, and it wears no
    # direction of its own; `_reverse_type` is a concrete forward type that
    # maps the opposite direction, so it is what an inversion resolves *to*.
    # A type with no pair -- the overwhelming majority -- leaves this `None`
    # and inverts to itself.
    _reverseof: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None
    _reverse_type: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        # A type names its opposite half in `_reverseof`, and that single
        # declaration pairs the two both ways: the declaring class is
        # pointed at the type it names, and that type is pointed back.
        # Declaring it on the second half of the pair to be defined is
        # therefore enough, which is also the only place it can be declared
        # -- the first half cannot name a class that does not exist yet.
        super().__init_subclass__(**kwargs)
        # Read from `cls.__dict__`, never `getattr`: only a class that
        # declares `_reverseof` in its own body claims a pair. An inherited
        # one would let a refinement such as `class MyLPSToVoxel(LPSToVoxel)`
        # silently steal `VoxelToLPS`'s half of the pairing, and would also
        # let the throwaway stand-in classes `Magic` builds while reading the
        # MRO -- which reach this hook too -- do the same. A refinement
        # instead inherits the pairing of its base, and reverses to that
        # base's opposite half unless it declares a `_reverseof` of its own.
        other = cls.__dict__.get("_reverseof")
        if other is not None:
            cls._reverse_type = other
            other._reverse_type = cls

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
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
        mode : [list of] name or type, optional
            Ignored on a leaf. `mode` gates which kinds *compose*, and a
            leaf has nothing to compose; its downcast is gated only by
            `simplify` (simplification is decoupled from the compose mode).
        simplify : simplify policy, default="analytic"
            How hard this leaf may be looked at. The resolved
            [`SimplifyPolicy`][brainhops.datamodel.enums.SimplifyPolicy]
            decides whether the kind-checks run structure-only (`analytic`)
            or read values (`numeric`), or are skipped entirely (`none`).
        factor : bool, default=False
            Whether to factor this leaf into its axis-group normal form. A
            leaf factors by wrapping itself in a one-element sequence, so a
            diagonal affine (say) splits into its per-axis blocks. Off by
            default.
        """
        if factor:
            # A leaf asked to factor is handed to the sequence engine as a
            # one-element sequence, which runs the factor pass. The engine
            # never passes `factor` back to a leaf's `compute`, so there is
            # no recursion.
            return registries.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        return _simplify(self, policy=simplify)

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        # The shared `inverse()` of every forward type that defers its
        # inversion to a type-transparent `Inverse` wrapper. The wrapper
        # holds this transform as its `forward` and materializes the
        # inverse only when its parameter is read or it is computed, so a
        # transform placed next to its own inverse cancels for free.
        if self._is_unparameterized():
            # Nothing to invert: an unset parameter reads as the identity,
            # whose inverse is itself with the endpoints swapped. Wrapping
            # it would only defer a computation that does not exist.
            reverse = type(self)._reverse_type
            if reverse is not None:
                # ... except for a paired type, whose swapped direction is
                # the other half of the pair, not itself.
                return reverse(input=self.output, output=self.input)
            return self.to(input=self.output, output=self.input)
        obj = registries.INVERSE(self)
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def _is_unparameterized(self) -> bool:
        # Whether every parameter this transform is defined by is unset.
        return all(
            getattr(self, name, None) is None for name in self.data_fields
        )


class TransformationField(ConcreteTransformation):
    """
    Base class for dense transformation fields (displacements or coordinates)
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("field",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "order", "bound", "coeff"

    # --- attributes ---------------------------------------------------

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


class DisplacementField(TransformationField):
    """
    A field of displacements defined on a regular grid.

    Both the input and output spaces correspond to the underlying grid.
    """


class CoordinatesField(TransformationField):
    """
    A field of coordinates defined on a regular grid.

    The input space corresponds to the regular grid on which the
    coordinates are defined.
    """


class CartesianField(CoordinatesField):
    """
    An identity transform over a regular grid of coordinates.

    Both the input and output spaces correspond to the underlying grid.

    Its `field` attribute is fully defined by the shape of the grid,
    and is generated on demand when accessed.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("shape",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("field",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "order", "bound", "coeff"

    # --- attributes ---------------------------------------------------

    shape: tx.Annotated[
        tx.Optional[tx.Tuple[int, ...]], tx.Doc("The shape of the grid.")
    ] = None

    # --- derived attributes -------------------------------------------
    # Mark them as `ClassVar` to keep them out of `__init__`.

    field: Derived[tx.Optional[ArrayProtocol]]

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

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # A grid is the identity map over its own coordinates, so its
        # inverse is itself with the endpoints switched. There is nothing
        # to defer, so `compute` changes nothing.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)


@kinds.Affine.register
class Affine(ConcreteTransformation):
    """An affine transformation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("homogeneous_matrix",)

    # --- attributes ---------------------------------------------------

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

    # --- derived attributes -------------------------------------------

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


@kinds.Linear.register
class Linear(ConcreteTransformation):
    """A linear transformation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    # --- attributes ---------------------------------------------------

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


@kinds.SpecialOrthogonal.register
class Rotation(Linear):
    """An orthogonal transformation with determinant 1, i.e., a rotation."""

    # TODO: Implement Rotation subclasses that use other representations
    # (e.g., quaternions, Euler angles, etc.)

    # --- attributes ---------------------------------------------------

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


@kinds.Permutation.register
class Permutation(ConcreteTransformation):
    """A permutation of axes."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("permutation",)

    # --- attributes ---------------------------------------------------

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


@kinds.Diagonal.register
class Scaling(ConcreteTransformation):
    """A scaling of axes."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    # --- attributes ---------------------------------------------------

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


@kinds.Translation.register
class Translation(ConcreteTransformation):
    """A translation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("translation",)

    # --- attributes ---------------------------------------------------

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


@kinds.Identity.register
class Identity(ConcreteTransformation):
    """An identity transformation.

    If the `input` and `output` coordinate systems are different, it maps
    the input axes to the output axes, while preserving their orders.
    """

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
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
    recognized as the identity, because its `shape` parameter is set. A
    [`CartesianField`][] with no grid has an unset `shape` parameter and
    is recognized as the identity by the parameter check under either
    value of `compute`. The `shape` is read, never the derived `field`, so
    the meshgrid is not built by a structural check.

    Recognizing a grid as the identity does not mean a grid may be dropped
    on sight. A grid also defines the sampling domain onto which data is
    resampled. The decision to factor a grid away is made by the sequence
    simplifier, which only does so for a grid that sits strictly between
    two other transformations.
    """
    return is_kind(xform, kinds.Identity, compute)


def is_translation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure translation.

    A transformation is recognized as a translation when it is an
    instance of [`kinds.Translation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a translation by zero.

    When `compute` is true, the matrix of an [`Affine`][] transformation
    is also inspected for a linear part equal to the identity.
    """
    return is_kind(xform, kinds.Translation, compute)


def is_scaling(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure scaling.

    A transformation is recognized as a scaling when it is an instance
    of [`kinds.Diagonal`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a scaling by one.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for a diagonal structure.
    """
    return is_kind(xform, kinds.Diagonal, compute)


def is_permutation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure permutation of axes.

    A transformation is recognized as a permutation when it is an
    instance of [`kinds.Permutation`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself a trivial permutation.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for a binary, one-per-row and
    one-per-column structure.
    """
    return is_kind(xform, kinds.Permutation, compute)


def is_rotation(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is a pure rotation.

    A transformation is recognized as a rotation when it is an instance
    of [`kinds.SpecialOrthogonal`][], or when
    [`is_identity`][] recognizes it as the identity, which is itself a
    rotation by zero.

    When `compute` is true, the matrix of a [`Linear`][] or [`Affine`][]
    transformation is also inspected for orthogonality and a positive
    determinant.
    """
    return is_kind(xform, kinds.SpecialOrthogonal, compute)


def is_linear(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is linear, without a translation.

    A transformation is recognized as linear when it is an instance of
    [`kinds.Linear`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself linear.

    When `compute` is true, the matrix of an [`Affine`][] transformation
    is also inspected for a zero translation component.
    """
    return is_kind(xform, kinds.Linear, compute)


def is_affine(xform: Transformation, /, compute: bool = False) -> bool:
    """Return whether a transformation is affine.

    A transformation is recognized as affine when it is an instance of
    [`kinds.Affine`][], or when [`is_identity`][]
    recognizes it as the identity, which is itself affine.
    """
    return is_kind(xform, kinds.Affine, compute)
