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
from bagof.magic import InitVar, KwOnly

# core
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.typing import (
    ArrayProtocol,
    Deactivated,
    Derived,
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

    # --- encoding -----------------------------------------------------
    #
    # Every concrete family stores its parameter as one array, `data`,
    # and says what that array holds with the encoding flags listed in
    # `metadata_fields` (a field's `coeff`, `degree` and `bound`). What a
    # consumer reads is a named, read-only *view* -- `field`, `matrix`,
    # `scale`, ... -- which is always the map, as values, whatever the
    # flags. Another encoding is reached by conversion, never by a view:
    # spline coefficients are `t.to(coeff=True).data`.
    #
    # The view's name doubles as a constructor keyword, a convenience
    # that means "the map, as values", while the flags say how it is
    # stored: `Affine(matrix=m)` is `Affine(data=m)`, and
    # `DisplacementField(field=u, coeff=True, degree=3)` stores the
    # cubic coefficients of `u`, exactly as
    # `DisplacementField(field=u, degree=3).to(coeff=True)` does. Each
    # family declares the keyword as an `InitVar` under the view's name
    # with a leading underscore (so that the constructor takes `field=`
    # while the class keeps its `field` view), and `__post_init__` below
    # encodes it into `data` with `_encode_view`, which `.to(field=...)`
    # shares.

    _view: tx.ClassVar[tx.Optional[str]] = None
    """The name of the view that reads the map, and of its keyword."""

    _values_flags: tx.ClassVar[tx.Mapping[str, tx.Any]] = {}
    """The flags under which `data` holds the map as values."""

    def __post_init__(self, arguments: tx.Any) -> None:
        # The convenience keyword is the map, as values, encoded into
        # `data` under the flags. It fills `data`, so it cannot come with
        # `data=` as well.
        view = type(self)._view
        values = arguments.get(view) if view is not None else None
        if values is None:
            return
        name = type(self).__name__
        if arguments.get("data") is not None:
            # `replace()` reaches here too: it carries `data` over, so a
            # convenience keyword passed through it always meets one.
            raise TypeError(
                f"{name}() got both data= and {view}=: {view}= is the "
                f"map, as values, and fills data, so it cannot be "
                f"combined with data=, which replace() also passes on. "
                f"To change the map of an existing transformation, use "
                f"t.to({view}=...), which encodes it under the flags of "
                f"t; to store an array as it is encoded, pass data= alone."
            )
        self.data = type(self)._encode_view(values, **self._flags())

    def _flags(self) -> tx.Dict[str, tx.Any]:
        # The encoding flags of `data`, by name.
        return {name: getattr(self, name) for name in self.metadata_fields}

    @classmethod
    def _holds_values(cls, flags: tx.Mapping[str, tx.Any]) -> bool:
        # Whether `data` holds the map as values under these flags.
        return all(
            flags.get(flag, value) == value
            for flag, value in cls._values_flags.items()
        )

    @classmethod
    def _encode_view(cls, values: tx.Any, **flags) -> tx.Any:
        """The `data` of the map given as values to its keyword.

        This is the one path from a convenience keyword (`field=`,
        `matrix=`, ...) to `data`, shared by the constructor and by
        `.to(...)`: the values are encoded under `flags`.
        """
        return None if values is None else cls._encode(values, **flags)

    @classmethod
    def _encode(cls, values: tx.Any, **flags) -> tx.Any:
        """Encode the map, given as values, into `data` under `flags`."""
        return values

    @classmethod
    def _decode(cls, data: tx.Any, **flags) -> tx.Any:
        """Decode `data`, encoded under `flags`, into the map as values."""
        return data

    def _values(self) -> tx.Any:
        # The map, as values: what every view returns. Data that already
        # holds values is returned as is. Anything else is decoded once,
        # and the result is cached on the instance against the array and
        # the flags it was decoded from, so that a view stays cheap to
        # read and never serves a decode of a `data` or of flags that
        # have since been replaced. (Like the inverse cache, it assumes
        # the array is not edited in place.)
        data = self.data
        if data is None:
            return None
        flags = self._flags()
        if self._holds_values(flags):
            return data
        cached = self.__dict__.get("_decoded")
        if cached is None or cached[0] is not data or cached[1] != flags:
            cached = (data, flags, type(self)._decode(data, **flags))
            self.__dict__["_decoded"] = cached
        return cached[2]

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

    The field is stored in `data`, either as values or, when `coeff` is
    true, as the coefficients of the spline of degree `degree` that
    interpolates them. Its `field` view is always the values.

    The `field=` keyword is the map, as values, and the flags say how it
    is stored: `DisplacementField(field=u, degree=3, coeff=True)` holds
    the cubic coefficients of `u` in `data`, the same `data` as
    `DisplacementField(field=u, degree=3).to(coeff=True)`. To store an
    array as it is already encoded, pass it as `data=`.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "degree", "bound", "coeff"
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("field",)
    _view: tx.ClassVar[str] = "field"
    _values_flags: tx.ClassVar[tx.Mapping[str, tx.Any]] = {"coeff": False}

    # --- attributes ---------------------------------------------------

    data: tx.Annotated[
        tx.Optional[ArrayProtocol],
        tx.Doc(
            """
            The stored field, an array of shape `(*shape, ndim)`: its
            values, or their spline coefficients when `coeff` is true.
            Read the map through `field`, which is always the values.
            """
        ),
    ] = None

    degree: tx.Annotated[InterpolationOrder, tx.Doc("The spline degree")] = (
        InterpolationOrder.linear
    )

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

    # --- views --------------------------------------------------------

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """
        The field, as values: an array of shape `(*shape, ndim)`.

        It is `data` itself when `coeff` is false, and `data` decoded
        from spline coefficients (once, then cached) when it is true.
        """
        return self._values()

    # --- encoding -----------------------------------------------------

    @classmethod
    def _encode(
        cls,
        values: ArrayProtocol,
        *,
        coeff: bool = False,
        degree: InterpolationOrder = InterpolationOrder.linear,
        bound: tx.Union[BoundaryCondition, float] = BoundaryCondition.nearest,
    ) -> ArrayProtocol:
        if not coeff:
            return values
        values = _prefilter_dtype(values)
        return value2coeff_field(values, degree=degree, bound=bound)

    @classmethod
    def _decode(
        cls,
        data: ArrayProtocol,
        *,
        coeff: bool = False,
        degree: InterpolationOrder = InterpolationOrder.linear,
        bound: tx.Union[BoundaryCondition, float] = BoundaryCondition.nearest,
    ) -> ArrayProtocol:
        if not coeff:
            return data
        return coeff2value_field(data, degree=degree, bound=bound)


def _prefilter_dtype(values: ArrayProtocol) -> ArrayProtocol:
    """The array the spline prefilter is run on, in a floating dtype.

    The coefficients of integer or boolean values (a grid, say) are not
    integers, and fitting them in an integer array would truncate them:
    such an array is cast to `float32`, on its own backend. A floating
    array keeps its dtype.
    """
    if values.dtype.kind in "biu":
        return values.astype("float32")
    return values


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

    It stores the `shape` of the grid rather than an array. Its `field`
    (the coordinates of the grid points) and its `data` (the same
    coordinates, encoded under the flags) are generated on demand.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("shape",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "field")
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = "degree", "bound", "coeff"

    # --- attributes ---------------------------------------------------

    shape: tx.Annotated[
        tx.Optional[tx.Tuple[int, ...]], tx.Doc("The shape of the grid.")
    ] = None

    # --- derived attributes -------------------------------------------
    # Mark them as `ClassVar` to keep them out of `__init__`. The grid is
    # fully defined by its shape, so neither `data=` nor `field=` is
    # taken.

    data: Derived[tx.Optional[ArrayProtocol]]
    _field: Deactivated[None]

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The coordinates of the grid points, of shape `(*shape, ndim)`."""
        if self.shape is None:
            return None
        if getattr(self, "_grid", None) is None:
            ab = get_array_backend()
            self._grid = ab.stack(
                ab.meshgrid(
                    *[ab.arange(s) for s in self.shape], indexing="ij"
                ),
                -1,
            )
        return self._grid

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The coordinates of the grid points, encoded under the flags."""
        field = self.field
        if field is None:
            return None
        flags = self._flags()
        if self._holds_values(flags):
            return field
        cached = self.__dict__.get("_encoded")
        if cached is None or cached[0] != flags:
            cached = (flags, type(self)._encode(field, **flags))
            self.__dict__["_encoded"] = cached
        return cached[1]

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # A grid is the identity map over its own coordinates, so its
        # inverse is itself with the endpoints switched. There is nothing
        # to defer, so `compute` changes nothing.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)


@kinds.Affine
class Affine(ConcreteTransformation):
    """An affine transformation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "matrix",
        "homogeneous_matrix",
    )
    _view: tx.ClassVar[str] = "matrix"

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

    # --- views --------------------------------------------------------

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """
        The affine matrix, of shape `(No, Ni + 1)`, whose last column is
        the translation component.
        """
        return self._values()

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
    _view: tx.ClassVar[str] = "matrix"

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

    # --- views --------------------------------------------------------

    @property
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        """The matrix, of shape `(No, Ni)`."""
        return self._values()


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
    _view: tx.ClassVar[str] = "permutation"

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

    # --- views --------------------------------------------------------

    @property
    def permutation(self) -> tx.Optional[ArrayProtocol]:
        """The permutation vector, of shape `(N,)`."""
        return self._values()


@kinds.Diagonal
class Scaling(ConcreteTransformation):
    """A scaling of axes."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)
    _view: tx.ClassVar[str] = "scale"

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

    # --- views --------------------------------------------------------

    @property
    def scale(self) -> tx.Optional[ArrayProtocol]:
        """The scaling factors, of shape `(N,)`."""
        return self._values()


@kinds.Translation
class Translation(ConcreteTransformation):
    """A translation."""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("data",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("translation",)
    _view: tx.ClassVar[str] = "translation"

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

    # --- views --------------------------------------------------------

    @property
    def translation(self) -> tx.Optional[ArrayProtocol]:
        """The translation vector, of shape `(N,)`."""
        return self._values()


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
