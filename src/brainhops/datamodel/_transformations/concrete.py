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
from bagof.magic import InitVar, KwOnly, NoPolymorphError

# core
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.properties import lazyproperty, smartproperty
from brainhops._core.typing import (
    ArrayProtocol,
    Deactivated,
    npmatrix,
    npvector,
)

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# transformations
from . import registries
from .base import Transformation
from .check import is_kind
from .modes import ModeLike
from .registries import INVERSE_CACHE
from .simplify import SimplifyLike
from .simplify import simplify as _simplify
from .utils import require_endomorphism


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

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # The shared `sqrt()` of every forward type whose square root is a
        # typed lazy wrapper. The front door builds the wrapper of this
        # transform's family, and refuses a family that has none; the
        # wrapper computes its parameter only when it is read.
        require_endomorphism(self, "square root")
        if self._is_unparameterized():
            # An unset parameter reads as the identity, which is its own
            # square root. There is nothing to defer.
            obj = self
        else:
            try:
                obj = registries.SQRT(self)
            except NoPolymorphError:
                raise NotImplementedError(
                    f"The square root of a {type(self).__name__} is not "
                    "implemented."
                ) from None
        return obj.compute(**kwargs) if compute else obj

    def _is_unparameterized(self) -> bool:
        # Whether every parameter this transform is defined by is unset.
        return all(
            getattr(self, name, None) is None for name in self.data_fields
        )


class TransformationField(ConcreteTransformation):
    """
    Base class for dense transformation fields (displacements or coordinates)

    The field is stored in `data`, either as values or, when `coeff` is
    true, as the coefficients of the spline of degree `degree` that
    interpolates them. Its `field` view is always the values.

    The `field=` keyword is the map, as values, and the flags say how it
    is stored: `DisplacementField(field=u, degree=3, coeff=True)` holds
    the cubic coefficients of `u` in `data`, the same `data` as
    `DisplacementField(field=u, degree=3).to(coeff=True)`. To store an
    array as it is already encoded, pass it as `data=`.

    The `field` view is decoded once and cached. Assigning `data` or a
    flag (`t.coeff = True`) clears it, so the next read reflects the
    change. Unlike `.to(coeff=True)`, which re-encodes, assigning a flag
    reinterprets the stored array.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "degree", "bound", "coeff"
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("field",)

    # --- attributes ---------------------------------------------------
    # `data` and the flags are stored under private names, and exposed
    # through properties whose setters clear the views they key (see
    # `_forget_views`). The constructor still takes `data=`, `degree=`...,
    # as it takes `input=` for the private `_input`.

    _data: tx.Annotated[
        tx.Optional[ArrayProtocol],
        tx.Doc(
            """
            The stored field, an array of shape `(*shape, ndim)`: its
            values, or their spline coefficients when `coeff` is true.
            Read the map through `field`, which is always the values.
            """
        ),
    ] = None

    _degree: tx.Annotated[InterpolationOrder, tx.Doc("The spline degree")] = (
        InterpolationOrder.linear
    )

    _bound: tx.Annotated[
        tx.Union[BoundaryCondition, float],
        tx.Doc(
            """
            The boundary condition used to deal with coordinates outside
            of the field of view. If a float is given, it is treated as
            a constant value.
            """
        ),
    ] = BoundaryCondition.nearest

    _coeff: tx.Annotated[
        bool,
        tx.Doc(
            """
            If `True`, `data` holds the spline coefficients of the field,
            rather than its values. The map, read through `field`, is the
            same either way.
            """
        ),
    ] = False

    _field: tx.Annotated[
        InitVar[tx.Optional[ArrayProtocol]],
        tx.Doc(
            """
            The field, as values: a convenience for `data`. It is stored
            as the flags say, so with `coeff=True` its spline
            coefficients are what `data` holds. It cannot be combined
            with `data=`.
            """
        ),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `field=` is the map, as values: it fills `data`, encoded as the
        # flags say.
        field = arguments.get("field")
        if field is not None:
            _refuse_data_too(self, arguments, "field")
            self.data = _encode(field, self.coeff, self.degree, self.bound)

    # --- stored attributes, which key the views -----------------------

    def _forget_views(self) -> None:
        # The cached views (`field`, and the `data` a grid derives), and
        # the inverse cached on this transform, are computed from `data`
        # and the flags: a new value of either clears them.
        for name in ("_cache_field", "_cache_data", INVERSE_CACHE):
            self.__dict__.pop(name, None)

    def _set_data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value
        self._forget_views()

    def _set_degree(self, value: InterpolationOrder) -> None:
        self._degree = value
        self._forget_views()

    def _set_bound(self, value: tx.Union[BoundaryCondition, float]) -> None:
        self._bound = value
        self._forget_views()

    def _set_coeff(self, value: bool) -> None:
        self._coeff = value
        self._forget_views()

    data = smartproperty("data", _set_data)
    degree = smartproperty("degree", _set_degree)
    bound = smartproperty("bound", _set_bound)
    coeff = smartproperty("coeff", _set_coeff)

    # --- views --------------------------------------------------------

    @lazyproperty
    def field(self) -> tx.Optional[ArrayProtocol]:
        """
        The field, as values: an array of shape `(*shape, ndim)`.

        It is `data` itself when `coeff` is false, and `data` decoded
        from spline coefficients (once, then cached) when it is true.
        """
        return _decode(self.data, self.coeff, self.degree, self.bound)


class DisplacementField(TransformationField):
    """
    A field of displacements defined on a regular grid.

    Both the input and output spaces correspond to the underlying grid.
    """

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        if not self._is_unparameterized():
            raise NotImplementedError(
                "The square root of a displacement field is implemented only "
                "for a stationary velocity field (log=True), whose square "
                "root halves its velocity. A general field has no principal "
                "square root that brainhops computes."
            )
        return super().sqrt(compute, **kwargs)


class CoordinatesField(TransformationField):
    """
    A field of coordinates defined on a regular grid.

    The input space corresponds to the regular grid on which the
    coordinates are defined.
    """

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        raise NotImplementedError(
            "The square root of a coordinates field is not implemented. A "
            "map to take the square root of is stored as a "
            "StationaryVelocityField."
        )


class CartesianField(CoordinatesField):
    """
    An identity transform over a regular grid of coordinates.

    Both the input and output spaces correspond to the underlying grid.

    It stores the `shape` of the grid rather than an array. Its `field`
    (the coordinates of the grid points) and its `data` (the same
    coordinates, encoded under the flags) are generated on demand, and
    cached until `shape` or a flag is assigned.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("shape",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "field")
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "degree", "bound", "coeff"

    # --- attributes ---------------------------------------------------

    _shape: tx.Annotated[
        tx.Optional[tx.Tuple[int, ...]], tx.Doc("The shape of the grid.")
    ] = None

    def _set_shape(self, value: tx.Optional[tx.Tuple[int, ...]]) -> None:
        self._shape = value
        self._forget_views()

    shape = smartproperty("shape", _set_shape)

    # --- derived attributes -------------------------------------------
    # Mark them as `ClassVar` to keep them out of `__init__`. The grid is
    # fully defined by its shape, so neither `data=` nor `field=` is
    # taken.

    _data: Deactivated[tx.Optional[ArrayProtocol]]
    _field: Deactivated[tx.Optional[ArrayProtocol]]

    @lazyproperty
    def field(self) -> tx.Optional[ArrayProtocol]:
        """
        The coordinates of the grid points, of shape `(*shape, ndim)`.

        They are real coordinates, so they are built in the backend's
        default floating dtype (`float64` with NumPy), and so are their
        spline coefficients in `data`.
        """
        if self.shape is None:
            return None
        ab = get_array_backend()
        grid = [ab.arange(s, dtype=float) for s in self.shape]
        return ab.stack(ab.meshgrid(*grid, indexing="ij"), -1)

    @lazyproperty
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The coordinates of the grid points, encoded under the flags."""
        return _encode(self.field, self.coeff, self.degree, self.bound)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # A grid is the identity map over its own coordinates, so its
        # inverse is itself with the endpoints switched. There is nothing
        # to defer, so `compute` changes nothing.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)

    # A grid is the identity map, which its square and its square root
    # are too. It is returned as is, rather than as an `Identity`, because
    # it is also the sampling domain.

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        return _fixed_point(self, "square", compute, kwargs)

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        return _fixed_point(self, "square root", compute, kwargs)


@kinds.Affine
class Affine(ConcreteTransformation):
    """An affine transformation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "matrix",
        "homogeneous_matrix",
    )

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
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

    _matrix: tx.Annotated[
        InitVar[tx.Optional[npmatrix[Real]]],
        tx.Doc("The matrix: a convenience for `data`."),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `matrix=` is the map: it fills `data`.
        matrix = arguments.get("matrix")
        if matrix is not None:
            _refuse_data_too(self, arguments, "matrix")
            self.data = matrix

    # --- views --------------------------------------------------------

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """
        The affine matrix, of shape `(No, Ni + 1)`, whose last column is
        the translation component.
        """
        return self.data

    @property
    def homogeneous_matrix(self) -> ArrayProtocol:
        """
        The homogeneous matrix of the affine transformation, of shape
        `(No + 1, Ni + 1)`. The last row of the homogeneous matrix is
        `[0, 0, ..., 1]`.
        """
        matrix = self.matrix
        if matrix is None:
            return None
        ab = get_array_backend(matrix)
        No, NiPlus1 = matrix.shape
        homogeneous_matrix = ab.zeros((No + 1, NiPlus1))
        homogeneous_matrix[:-1, :-1] = matrix[:, :-1]
        homogeneous_matrix[:-1, -1:] = matrix[:, -1:]
        homogeneous_matrix[-1, -1] = 1
        return homogeneous_matrix


@kinds.Linear
class Linear(ConcreteTransformation):
    """A linear transformation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
        tx.Optional[npmatrix[Real]],
        tx.Doc(
            """
            A matrix of shape `(No, Ni)`, where `Ni` is the number of
            input dimensions and `No` is the number of output dimensions.
            If `None`, the matrix is treated as an identity transformation.
            """
        ),
    ] = None

    _matrix: tx.Annotated[
        InitVar[tx.Optional[npmatrix[Real]]],
        tx.Doc("The matrix: a convenience for `data`."),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `matrix=` is the map: it fills `data`.
        matrix = arguments.get("matrix")
        if matrix is not None:
            _refuse_data_too(self, arguments, "matrix")
            self.data = matrix

    # --- views --------------------------------------------------------

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """The matrix, of shape `(No, Ni)`."""
        return self.data


@kinds.SpecialOrthogonal
class Rotation(Linear):
    """An orthogonal transformation with determinant 1, i.e., a rotation."""

    # TODO: Implement Rotation subclasses that use other representations
    # (e.g., quaternions, Euler angles, etc.)

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
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


@kinds.Permutation
class Permutation(ConcreteTransformation):
    """A permutation of axes."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("permutation",)

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
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

    _permutation: tx.Annotated[
        InitVar[tx.Optional[npvector[Integral]]],
        tx.Doc("The permutation: a convenience for `data`."),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `permutation=` is the map: it fills `data`.
        permutation = arguments.get("permutation")
        if permutation is not None:
            _refuse_data_too(self, arguments, "permutation")
            self.data = permutation

    # --- views --------------------------------------------------------

    @property
    def permutation(self) -> tx.Optional[ArrayProtocol]:
        """The permutation vector, of shape `(N,)`."""
        return self.data


@kinds.Diagonal
class Scaling(ConcreteTransformation):
    """A scaling of axes."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
        tx.Optional[npvector[Real]],
        tx.Doc(
            """
            A vector of shape `(N,)`, where `N` is the number of dimensions
            to scale. If `None`, the scaling is treated as an identity
            transformation.
            """
        ),
    ] = None

    _scale: tx.Annotated[
        InitVar[tx.Optional[npvector[Real]]],
        tx.Doc("The scaling factors: a convenience for `data`."),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `scale=` is the map: it fills `data`.
        scale = arguments.get("scale")
        if scale is not None:
            _refuse_data_too(self, arguments, "scale")
            self.data = scale

    # --- views --------------------------------------------------------

    @property
    def scale(self) -> tx.Optional[ArrayProtocol]:
        """The scaling factors, of shape `(N,)`."""
        return self.data


@kinds.Translation
class Translation(ConcreteTransformation):
    """A translation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("translation",)

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
        tx.Optional[npvector[Real]],
        tx.Doc(
            """
            A vector of shape `(N,)`, where `N` is the number of dimensions
            to translate. If `None`, the translation is treated as an identity
            transformation.
            """
        ),
    ] = None

    _translation: tx.Annotated[
        InitVar[tx.Optional[npvector[Real]]],
        tx.Doc("The translation vector: a convenience for `data`."),
        KwOnly(),
    ] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # `translation=` is the map: it fills `data`.
        translation = arguments.get("translation")
        if translation is not None:
            _refuse_data_too(self, arguments, "translation")
            self.data = translation

    # --- views --------------------------------------------------------

    @property
    def translation(self) -> tx.Optional[ArrayProtocol]:
        """The translation vector, of shape `(N,)`."""
        return self.data


@kinds.Identity
class Identity(ConcreteTransformation):
    """An identity transformation.

    If the `input` and `output` coordinate systems are different, it maps
    the input axes to the output axes, while preserving their orders.

    It has no parameter: its `data` is always `None`, and is not a
    constructor argument.
    """

    # --- views --------------------------------------------------------

    @property
    def data(self) -> None:
        """Always `None`: the identity has no parameter to store."""
        return None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        cls = type(self)
        return cls(input=self.output, output=self.input)

    # The identity is its own square and its own square root.

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        return _fixed_point(self, "square", compute, kwargs)

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        return _fixed_point(self, "square root", compute, kwargs)


def _fixed_point(
    t: Transformation,
    operator: str,
    compute: bool,
    kwargs: tx.Dict[str, tx.Any],
) -> Transformation:
    # The result of an operator that leaves `t` unchanged.
    require_endomorphism(t, operator)
    return t.compute(**kwargs) if compute else t


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


# ----------------------------------------------------------------------
#    HELPERS
# ----------------------------------------------------------------------


def _refuse_data_too(
    xform: Transformation, arguments: tx.Any, keyword: str
) -> None:
    # A convenience keyword (`field=`, `matrix=`, ...) is the map, as
    # values, and fills `data`, so it cannot come with `data=` as well.
    # `replace()` reaches here too: it carries `data` over.
    if arguments.get("data") is None:
        return
    raise TypeError(
        f"{type(xform).__name__}() got both data= and {keyword}=: "
        f"{keyword}= is the map, as values, and fills data, so it cannot "
        f"be combined with data=, which replace() also passes on. To "
        f"change the map of an existing transformation, use "
        f"t.to({keyword}=...), which encodes it under the flags of t; to "
        f"store an array as it is encoded, pass data= alone."
    )


def _encode(
    values: tx.Optional[ArrayProtocol],
    coeff: bool,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    # A field, given as values, as the `data` that stores it: the values
    # themselves, or their spline coefficients when `coeff` is true.
    if values is None or not coeff:
        return values
    values = _prefilter_dtype(values)
    return value2coeff_field(values, degree=degree, bound=bound)


def _decode(
    data: tx.Optional[ArrayProtocol],
    coeff: bool,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    # The values of a field, from the `data` that stores it.
    if data is None or not coeff:
        return data
    return coeff2value_field(data, degree=degree, bound=bound)


def _prefilter_dtype(values: ArrayProtocol) -> ArrayProtocol:
    # The coefficients of integer or boolean values are not integers, and
    # fitting them in an integer array would truncate them: such an array
    # is fitted in `float32`, on its own backend. A floating array keeps
    # its dtype.
    if values.dtype.kind in "biu":
        return values.astype("float32")
    return values
