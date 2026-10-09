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
from bagof.magic import InitVar, NotKwOnly

# core
from brainhops._core.affines import inv as affine_inv
from brainhops._core.affines import sqrtm as affine_sqrtm
from brainhops._core.affines import to_compact, to_homogeneous
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.linalg import inv as invm
from brainhops._core.linalg import require_positive, sqrtm
from brainhops._core.magic import field_default
from brainhops._core.properties import (
    InvalidatorInAttribute,
    lazyproperty,
    smartproperty,
)
from brainhops._core.typing import (
    ArrayLike,
    ArrayProtocol,
    Deactivated,
    npmatrix,
    npvector,
)
from brainhops._ext.invfield import inverse as inverse_disp

# api
from brainhops.backends import backend, get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import (
    BoundaryCondition,
    InterpolationOrder,
    StoreEnum,
)
from brainhops.datamodel.systems import CoordinateSystem

# transformations
from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import require_endomorphism
from .modes import ModeLike

# typing
_TypeReference = tx.ClassVar[tx.Optional[tx.Type[Transformation]]]
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]

# --- helpers ----------------------------------------------------------

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""The views an assignment clears: those a class lists in `derived_fields`."""


def _invalidating_property(*args, **kwargs) -> property:
    """
    A [`smartproperty`][] that automatically invalidates the cached
    value of properties listed in the `derived_fields` class attribute.
    """
    kwargs.setdefault("invalidates", _FORGET_VIEWS)
    return smartproperty(*args, **kwargs)


def _alias(name: str, source: str, fset: bool = True) -> property:
    """
    A [`property`][] that simply redirects to/from another attribute.

    `source` may be a path -- `"forward.degree"` -- which reads through
    the attributes it names in turn. Such an alias is read-only whatever
    `fset` says: it names an attribute of another transformation, and a
    wrapper reports what it wraps rather than writing into it.
    """
    path = source.split(".")

    def fget(self: tx.Any) -> tx.Any:
        value = self
        for attr in path:
            value = getattr(value, attr)
        return value

    fget.__name__ = name

    if fset and len(path) == 1:

        def fset(self: tx.Any, value: tx.Any) -> None:
            setattr(self, source, value)

        fset.__name__ = name
    else:
        fset = None

    return property(fget, fset, doc=f"Alias for `{source}`.")


# --- API --------------------------------------------------------------


class ConcreteTransformation(Transformation):
    """Base class for concrete transformations that hold a parameter."""

    _reverseof: _TypeReference = None
    """
    Concrete type of the resolved (= computed) inverse of a transformation,
    returned by `inverse(compute=True)`. An example is `VoxelToLPS`, whose
    inverse is of type `LPSToVoxel`.

    It is only necessary to set it on one half of a pair (the one defined
    second, which is the only one that can name the other), and the
    `__init_subclass__` hook below points the other half back.

    `None` indicates that the inverse is of the same type (default).
    """

    map_field: tx.ClassVar[tx.Optional[str]] = None
    """
    The view that is the *map* of this family: `matrix`, `scale`,
    `translation`, `permutation`, `field`. It is declared on the class
    that roots the family, and `from_instance` copies through it.

    `None` for a class with no map to copy: the identity, which stores
    nothing, and a grid, whose map is generated from its `shape`.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # A type names its opposite half in `_reverseof`, and that single
        # declaration pairs the two both ways: the class that declares it
        # already points at the type it names, and that type is pointed
        # back here.
        #
        # Read from `cls.__dict__`, never `getattr`: only a class that
        # declares `_reverseof` in its own body claims a pair. An
        # inherited one would let a refinement such as
        # `class MyLPSToVoxel(LPSToVoxel)` silently steal `VoxelToLPS`'s
        # half of the pairing, and would let the throwaway stand-in
        # classes `Magic` builds while reading the MRO -- which reach this
        # hook too -- do the same. A refinement instead inherits the
        # pairing of its base, and reverses to that base's opposite half
        # unless it declares a `_reverseof` of its own.
        if (other := cls.__dict__.get("_reverseof")) is not None:
            other._reverseof = cls

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
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        return _simplify(self, policy=simplify)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance from an instance of a similar class.

        See [`DataModelBase.from_instance`][]. The *map* is copied, not
        the stored array: a tangent (`log=True`) copied into a class that
        holds the map is exponentiated, an `Affine` copied into a tangent
        has its logarithm taken, and a velocity copied into a plain field
        is integrated. The map is read through the family's
        [`map_field`][], so a lazy wrapper is read through the view it
        derives rather than off a `data` it does not store.

        The encoding travels with it: the metadata both classes share --
        a field's `degree`, `bound` and `store`, a velocity's `steps` --
        is copied over, so the copy holds the same array the same way.
        `log` is the exception, since it is what the map may have to
        cross.
        """
        view = cls.map_field
        if view is not None and _same_family(cls, other):
            shared = set(cls.metadata_fields) & set(other.metadata_fields)
            for name in shared - {"log"}:
                kwargs.setdefault(name, getattr(other, name))
            _copy_map(other, cls, view, kwargs)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        # The shared `inverse()` of every forward type that defers its
        # inversion to a type-transparent `Inverse` wrapper. The wrapper
        # holds this transform as its `forward` and materializes the
        # inverse only when its parameter is read or it is computed, so a
        # transform placed next to its own inverse cancels for free.
        if is_identity(self):
            # No point being lazy ...
            if (reverse := type(self)._reverseof) is not None:
                # ... except for a paired type, whose swapped direction is
                # the other half of the pair, not itself.
                return reverse(input=self.output, output=self.input)
            return self.to(input=self.output, output=self.input)
        # Otherwise, defer to the `Inverse` base class, which will
        # dispatch to the correct lazy class polymorphically.
        obj = nocycles.OPERATORS["inverse"](self)
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # The shared `sqrt()` of every forward type whose square root is a
        # typed lazy wrapper, which computes its parameter only when it is
        # read. Every concrete family has one, or a `sqrt()` of its own.
        require_endomorphism(self, "square root")
        if is_identity(self):
            # Identity.sqrt() == Identity
            obj = self
        else:
            obj = nocycles.OPERATORS["sqrt"](self)
        return obj.compute(**kwargs) if compute else obj


class TransformationField(ConcreteTransformation):
    """
    Base class for dense transformation fields (displacements or coordinates)

    The field is stored in `data`, either as values or, when
    `stores="coefficients"`, as the coefficients of the spline of degree
    `degree` that interpolates them. Its `field` view is always the values.

    The `field=` keyword is the map, as values, and the flags say how it
    is stored: `DisplacementField(field=u, degree=3, stores="coefficients")`
    holds the cubic coefficients of `u` in `data`, the same `data` as
    `DisplacementField(field=u, degree=3).to(stores="coefficients")`.
    To store an array as it is already encoded, pass it as `data=` or
    as `coefficients=`.

    The `field` view is decoded once and cached. Assigning `data` or a
    flag (`t.stores = "coefficients"`) clears it, so the next read
    reflects the change. Unlike `.to(stores="coefficients")`, which
    re-encodes, assigning a flag reinterprets the stored array.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "field"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = "degree", "bound", "store"
    derived_fields: _FieldNames = "field", "values", "coefficients", "_inverse"

    # --- attributes ---------------------------------------------------
    # `data` and the flags are stored under private names, and exposed
    # through properties that invalidate cached values when set.
    # The constructor still takes `data=`, `degree=`..., etc.

    _data: NotKwOnly[tx.Optional[ArrayProtocol]] = None
    """
    The stored field, an array of shape `(*shape, ndim)`:
    its values, or their spline coefficients when `store="coefficients"`.
    Read the map through `field`, which is always the values.
    """

    _degree: InterpolationOrder = InterpolationOrder.linear
    "The spline degree"

    _bound: tx.Union[BoundaryCondition, float] = BoundaryCondition.nearest
    """
    The boundary condition used to deal with coordinates outside of the
    field of view. If a float is given, it is treated as a constant value.
    """

    _store: tx.Optional[StoreEnum] = None
    """
    If `"coefficients"`, `data` holds the spline coefficients of the
    field, rather than its values.

    If not set, it is assumed to be `"coefficients"` if data is set via
    the `coefficients=` keyword, and `"values"` if it is set via
    `data=`, `values=` or `field=` keywords.

    The map, read through `field` or `values`, is the same either way.
    """

    _field: InitVar[tx.Optional[ArrayProtocol]] = None
    """
    The field, as values: a convenience for `data=...`.

    If `store="coefficients"`, the field is converted to spline
    coefficients before being stored in `data`.
    """

    _values: InitVar[tx.Optional[ArrayProtocol]] = None
    """
    The field, as values: a convenience for `data=...`.

    If `store="coefficients"`, the field is converted to spline
    coefficients before being stored in `data`.
    """

    _coefficients: InitVar[tx.Optional[ArrayProtocol]] = None
    """
    The field, as spline coefficients: a convenience for `data=...`.

    If `store="values"`, the field is evaluated before being stored
    in `data`.
    """

    # --- [meta]data views ---------------------------------------------

    data = _invalidating_property("data")
    degree = _invalidating_property("degree")
    bound = _invalidating_property("bound")

    @_invalidating_property
    def store(self) -> StoreEnum:
        """
        What `data` holds: the field's values, or their spline
        coefficients.

        Unset, it reads as `"values"`: every spelling but `coefficients=`
        hands over the map as values, and `coefficients=` sets the flag
        when the constructor was not given one.
        """
        return StoreEnum.values

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            field: ArrayLike,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            values: ArrayLike,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            coefficients: ArrayLike,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "coefficients",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    def _set_derived_fields(self, arguments: tx.Any) -> None:
        # `coefficients=` says what the array it hands over is, so it sets
        # the flag when the constructor was not given one:
        # `DisplacementField(coefficients=c)` stores coefficients. Every
        # other spelling is the map as values, which an unset flag means.
        if (
            arguments.get("store") is None
            and arguments.get("coefficients") is not None
        ):
            self.store = StoreEnum.coefficients
        super()._set_derived_fields(arguments)

    # --- derived views ------------------------------------------------

    @lazyproperty
    def values(self) -> tx.Optional[ArrayProtocol]:
        """
        The field, as values: an array of shape `(*shape, ndim)`.

        It is `data` itself when `store` is `"values"`, and `data`
        decoded from spline coefficients (once, then cached) when `data`
        holds those.
        """
        return _data2values(self.data, self.store, self.degree, self.bound)

    @values.setter
    def values(self, value: tx.Optional[ArrayProtocol]) -> None:
        # Fit the spline coefficients if the field stores them. The write
        # goes through `data`, which clears every view keyed on it ...
        self.data = _values2data(value, self.store, self.degree, self.bound)
        # ... and only then is the input cached, rather than recomputed.
        self._cache_values = value

    field = _alias("field", "values")

    @lazyproperty
    def coefficients(self) -> tx.Optional[ArrayProtocol]:
        """
        The field, as spline coefficients: an array of shape
        `(*shape, ndim)`.

        It is `data` itself when `store` is `"coefficients"`, and `data`
        fitted to a spline of degree `degree` (once, then cached) when
        `data` holds the values.
        """
        return _data2coeffs(self.data, self.store, self.degree, self.bound)

    @coefficients.setter
    def coefficients(self, coeffs: tx.Optional[ArrayProtocol]) -> None:
        # Evaluate the spline if the field stores values. The write goes
        # through `data`, which clears every view keyed on it ...
        self.data = _coeffs2data(coeffs, self.store, self.degree, self.bound)
        # ... and only then is the input cached, rather than recomputed.
        self._cache_coefficients = coeffs

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        """
        The parameter of the inverse field, in this field's own
        encoding: an array of shape `(*shape, ndim)`.

        Each family derives it and caches it here, which is where its
        lazy inverse reads it from -- so an inversion is run once, and a
        field whose `data` or flags are assigned clears it.
        """
        raise NotImplementedError


class DisplacementField(TransformationField, polymorphic=True):
    """
    A field of displacements defined on a regular grid.

    Both the input and output spaces correspond to the underlying grid.

    The `log` flag says which function `data` describes: the displacement
    itself, or, when `log` is true, the stationary velocity whose flow at
    time one is the map. `log=True` builds a
    [`StationaryVelocityField`][], whose `field` view integrates that
    velocity; a plain `DisplacementField` always holds displacements.
    """

    # --- class attributes ---------------------------------------------

    _base_metadata_fields: _FieldNames = TransformationField.metadata_fields
    metadata_fields: _FieldNames = (
        *_base_metadata_fields,
        "log",
    )

    # --- attributes ---------------------------------------------------

    _log: bool = False
    """
    If `True`, `data` holds the stationary velocity whose flow at time
    one is the map (its tangent about the identity), rather than its
    displacement: the field is a [`StationaryVelocityField`][].
    """

    log = _invalidating_property("log", fset=False)

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            log: bool = False,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            field: ArrayLike,
            log: bool = False,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            values: ArrayLike,
            log: bool = False,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            coefficients: ArrayLike,
            log: bool = False,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "coefficients",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    def __post_init__(self, arguments: tx.Any) -> None:
        _refuse_log(self, arguments.get("log"), "StationaryVelocityField")
        super().__post_init__(arguments)

    # --- derived views ------------------------------------------------

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        """
        The materialized inverse of this field, cached.
        """
        if self.data is None:
            return None
        ifield = inverse_disp(self.field)
        return _values2data(ifield, self.store, self.degree, self.bound)

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        if not is_identity(self):
            raise NotImplementedError(
                "The square root of a displacement field is implemented "
                "only for a stationary velocity field (log=True), whose "
                "square root halves its velocity. A general field has no "
                "principal square root that brainhops computes."
            )
        return super().sqrt(compute, **kwargs)


class CoordinatesField(TransformationField):
    """
    A field of coordinates defined on a regular grid.

    The input space corresponds to the regular grid on which the
    coordinates are defined.
    """

    # --- derived ------------------------------------------------------

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        ifield = _inv_coords(self.field)
        return _values2data(ifield, self.store, self.degree, self.bound)

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        raise NotImplementedError(
            "The square root of a coordinates field is not implemented. "
            "A map to take the square root of is stored as a "
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

    # A grid generates its map from its `shape`, which is copied as the
    # plain field it is: there is no map to re-encode.
    map_field: tx.ClassVar[tx.Optional[str]] = None

    _base_derivied_fields: _FieldNames = CoordinatesField.derived_fields
    derived_fields: _FieldNames = (*_base_derivied_fields, "data")
    data_fields: _FieldNames = ("shape",)

    # --- attributes ---------------------------------------------------

    _shape: NotKwOnly[tx.Optional[tx.Tuple[int, ...]]] = None
    "The shape of the grid."

    # Mark derived fields as `ClassVar` to keep them out of `__init__`.
    # The grid is fully defined by its shape, so neither `data=` nor
    # `field=` is taken.

    _data: Deactivated[tx.Optional[ArrayProtocol]]
    _field: Deactivated[tx.Optional[ArrayProtocol]]
    _values: Deactivated[tx.Optional[ArrayProtocol]]
    _coefficients: Deactivated[tx.Optional[ArrayProtocol]]

    # --- attribute views ----------------------------------------------

    shape = _invalidating_property("shape")

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            shape: tx.Sequence[int],
            *,
            degree: tx.Union[InterpolationOrder, int, str] = "linear",
            bound: tx.Union[BoundaryCondition, float, str] = "nearest",
            store: tx.Union[StoreEnum, str] = "values",
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    # --- derived attributes -------------------------------------------

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

    values = _alias("values", "field", fset=False)

    @lazyproperty
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The coordinates of the grid points, encoded under the flags."""
        return _values2data(self.field, self.store, self.degree, self.bound)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # A grid is the identity map over its own coordinates, so its
        # inverse is itself with the endpoints switched. There is nothing
        # to defer, so `compute` changes nothing.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)

    # A grid is the identity map, which its square and its square root
    # are too. It is returned as is, rather than as an `Identity`,
    # because it is also the sampling domain.

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square")
        return self.compute(**kwargs) if compute else self

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square root")
        return self.compute(**kwargs) if compute else self


@kinds.Affine
class Affine(ConcreteTransformation, polymorphic=True):
    """
    An affine transformation.

    `data` holds its `(No, Ni + 1)` matrix. `log=True` builds an
    [`AffineExponential`][], whose `data` is the tangent of the map about
    the identity instead.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "matrix"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = ("log",)
    derived_fields: _FieldNames = (
        "matrix",
        "homogeneous_matrix",
        "_sqrt",
        "_inverse",
    )

    # --- attributes ---------------------------------------------------
    # `data` and `log` are stored under private names, and exposed through
    # properties whose setters clear the views they key, as a field's are.

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """
    A matrix of shape `(No, Ni + 1)`, where `Ni` is the number of
    input dimensions and `No` is the number of output dimensions.
    The last column of the matrix corresponds to the translation
    component of the affine transformation.

    If `None`, the matrix is treated as an identity transformation.
    """

    _log: bool = False
    """
    If `True`, `data` holds the tangent of the map about the identity,
    rather than its matrix: the transformation is an [`AffineExponential`][].
    """

    _matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The matrix, with shape (No, Ni+1). An alias for `data`."

    _homogeneous_matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The homogeneous form of the matrix, with shape (No+1, Ni+1)."

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            matrix: ArrayLike,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            homogeneous_matrix: ArrayLike,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    def __post_init__(self, arguments: tx.Any) -> None:
        _refuse_log(self, arguments.get("log"), "AffineExponential")
        super().__post_init__(arguments)

    # --- views --------------------------------------------------------

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)
    matrix = _alias("matrix", "data")

    @lazyproperty
    def homogeneous_matrix(self) -> tx.Optional[ArrayProtocol]:
        matrix = self.matrix
        return to_homogeneous(matrix) if matrix is not None else None

    @homogeneous_matrix.setter
    def homogeneous_matrix(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.matrix = to_compact(value) if value is not None else None

    @lazyproperty
    def _sqrt(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        what = f"The square root of this {type(self).__name__}"
        return affine_sqrtm(self.matrix, what)

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        return affine_inv(self.matrix)


@kinds.Linear
class Linear(ConcreteTransformation, polymorphic=True):
    """
    A linear transformation.

    `data` holds its `(No, Ni)` matrix. `log=True` builds a
    [`LinearExponential`][], whose `data` is the tangent of the map about
    the identity instead.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "matrix"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = ("log",)
    derived_fields: _FieldNames = "matrix", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """
    A matrix of shape `(No, Ni)`, where `Ni` is the number of input
    dimensions and `No` is the number of output dimensions.

    If `None`, the matrix is treated as an identity transformation.
    """

    _log: bool = False
    """
    If `True`, `data` holds the tangent of the map about the identity,
    rather than its matrix: the transformation is a [`LinearExponential`][]
    (or a [`RotationExponential`][]).
    """

    _matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    "The matrix: a convenience for `data`."

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            matrix: ArrayLike,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    def __post_init__(self, arguments: tx.Any) -> None:
        classname = type(self).__name__
        classname = {
            "Linear": "LinearExponential",
            "Rotation": "RotationExponential",
        }.get(classname, classname)
        _refuse_log(self, arguments.get("log"), classname)
        super().__post_init__(arguments)

    # --- views --------------------------------------------------------

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)
    matrix = _alias("matrix", "data")

    @lazyproperty
    def _sqrt(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        what = f"The square root of this {type(self).__name__}"
        return sqrtm(self.matrix, what)

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        return invm(self.matrix)


@kinds.SpecialOrthogonal
class Rotation(Linear):
    """
    An orthogonal transformation with determinant 1, i.e., a rotation.

    `data` holds its `(N, N)` matrix. `log=True` builds a
    [`RotationExponential`][], whose `data` is the tangent of the map about
    the identity instead.
    """

    # TODO: Implement Rotation subclasses that use other representations
    # (e.g., quaternions, Euler angles, etc.)

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """
    A matrix of shape `(No, Ni)`, where `Ni` is the number of input
    dimensions and `No` is the number of output dimensions.

    This matrix MUST have a determinant of 1.

    If `None`, the matrix is treated as an identity transformation.
    """

    # --- views --------------------------------------------------------

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        return self.matrix.T


@kinds.Permutation
class Permutation(ConcreteTransformation):
    """A permutation of axes."""

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "permutation"

    data_fields: _FieldNames = ("data",)
    derived_fields: _FieldNames = "permutation", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """
    A vector of shape `(N,)`, where `N` is the number of axes to permute.
    The element at index `i` indicates the input dimension that corresponds
    to the output dimension `i`.

    If `None`, the permutation is treated as an identit transformation.
    """

    _permutation: InitVar[tx.Optional[npvector[Integral]]] = None
    """The permutation: a convenience for `data`."""

    # --- views --------------------------------------------------------

    data = _invalidating_property("data")
    permutation = _alias("permutation", "data")

    @lazyproperty
    def _inverse(self) -> tx.Optional[npvector[Integral]]:
        perm = self.permutation
        if perm is None:
            return None
        backend = get_array_backend(perm)
        inverse_permutation = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse_permutation[p] = i
        return backend.asarray(inverse_permutation, dtype=perm.dtype)

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            permutation: ArrayLike,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    # --- methods ------------------------------------------------------

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        if not is_identity(self):
            raise NotImplementedError(
                "The square root of a permutation is not implemented: no "
                "reordering of axes is the half of another. The "
                "three-cycle, for one, is a third of a turn, so its "
                "square root is a sixth of one -- a rotation, which "
                "permutes no axes. Read the permutation as the linear "
                "transform it stands for if that is what you want: "
                "p.to(Linear).sqrt()."
            )
        return super().sqrt(compute, **kwargs)


@kinds.Diagonal
class Scaling(ConcreteTransformation, polymorphic=True):
    """
    A scaling of axes.

    `data` holds its scaling factors. `log=True` builds a
    [`ScalingExponential`][], whose `data` is their logarithm instead.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "scale"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = ("log",)
    derived_fields: _FieldNames = "scale", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Real]]] = None
    """
    A vector of shape `(N,)`, where `N` is the number of dimensions to scale.

    If `None`, the scaling is treated as an identity transformation.
    """

    _log: bool = False
    """
    If `True`, `data` holds the logarithm of the scaling factors,
    the tangent of the map about the identity: the transformation
    is a [`ScalingExponential`][].
    """

    _scale: InitVar[tx.Optional[npvector[Real]]] = None
    """The scaling factors: a convenience for `data`."""

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            scale: ArrayLike,
            log: bool = False,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

    def __post_init__(self, arguments: tx.Any) -> None:
        _refuse_log(self, arguments.get("log"), "ScalingExponential")
        super().__post_init__(arguments)

    # --- views --------------------------------------------------------

    data = _invalidating_property("data")
    log = _invalidating_property("log", fset=False)
    scale = _alias("scale", "data")

    @lazyproperty
    def _sqrt(self) -> tx.Optional[ArrayProtocol]:
        if self.scale is None:
            return None
        what = f"The square root of this {type(self).__name__}"
        require_positive(self.scale, what)
        return get_array_backend(self.scale).sqrt(self.scale)

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.scale is None:
            return None
        return get_array_backend(self.scale).reciprocal(self.scale)


@kinds.Translation
class Translation(ConcreteTransformation):
    """A translation."""

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "translation"

    data_fields: _FieldNames = ("data",)
    derived_fields: _FieldNames = "translation", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Real]]] = None
    """
    A vector of shape `(N,)`, where `N` is the number of dimensions
    to translate. If `None`, the translation is treated as an identity
    transformation.
    """

    _translation: InitVar[tx.Optional[npvector[Real]]] = None
    """The translation vector: a convenience for `data`."""

    # --- views --------------------------------------------------------

    data = _invalidating_property("data")
    translation = _alias("translation", "data")

    @lazyproperty
    def _sqrt(self) -> tx.Optional[ArrayProtocol]:
        if self.translation is None:
            return None
        return 0.5 * self.translation

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.translation is None:
            return None
        return -self.translation

    # --- overloads ----------------------------------------------------

    if tx.TYPE_CHECKING:

        @tx.overload
        def __init__(
            self,
            data: ArrayLike,
            *,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...

        @tx.overload
        def __init__(
            self,
            *,
            translation: ArrayLike,
            input: tx.Optional[CoordinateSystem] = None,
            output: tx.Optional[CoordinateSystem] = None,
        ) -> None: ...


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

    @property
    def _sqrt(self) -> None:
        return None

    @property
    def _inverse(self) -> None:
        return None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        cls = type(self)
        return cls(input=self.output, output=self.input)

    # The identity is its own square and its own square root.

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square")
        return self.compute(**kwargs) if compute else self

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square root")
        return self.compute(**kwargs) if compute else self


# ----------------------------------------------------------------------
#    KIND CHECKS
# ----------------------------------------------------------------------
# External methods kept for backward comp for now.

is_identity = Transformation.is_identity
is_translation = Transformation.is_translation
is_scaling = Transformation.is_scaling
is_permutation = Transformation.is_permutation
is_rotation = Transformation.is_rotation
is_linear = Transformation.is_linear
is_affine = Transformation.is_affine


# ----------------------------------------------------------------------
#    HELPERS
# ----------------------------------------------------------------------


def _same_family(cls: tx.Type[Transformation], other: tx.Any) -> bool:
    """
    Whether `other` belongs to the family whose map `cls` copies.

    The family is rooted at the class that declares `map_field`, so an
    `Affine` and a `Linear` are different families although both name
    `matrix`: their matrices are not the same shape.
    """
    root = next(b for b in cls.__mro__ if "map_field" in b.__dict__)
    return isinstance(other, root)


def _copy_map(
    other: Transformation,
    cls: tx.Type[Transformation],
    view: str,
    kwargs: tx.Dict[str, tx.Any],
) -> None:
    """
    Fill `kwargs` with the map of `other`, encoded as `cls` holds it.

    `view` names the map of the family (`matrix`, `scale`): a class that
    holds the map stores it as it is, and one that holds a tangent stores
    its logarithm. When the two agree, the stored array carries over
    untouched -- a tangent copied into a tangent is not exponentiated and
    then logged again.
    """
    out_tangent = bool(kwargs.get("log", _is_tangent(cls)))
    if out_tangent == bool(getattr(other, "log", False)):
        kwargs.setdefault("data", other.data)
    elif out_tangent:
        # The tangent of the map: the view is the map, and the class that
        # takes it stores its logarithm.
        kwargs.setdefault(view, getattr(other, view))
    else:
        # The map itself, which the view of a tangent already is.
        kwargs.setdefault("data", getattr(other, view))
    if "log" in cls.metadata_fields:
        kwargs.setdefault("log", out_tangent)


def _is_tangent(cls: tx.Type[Transformation]) -> bool:
    """
    Whether a class holds the tangent of the map, rather than the map.

    A tangent class is the one `log=True` selects, and it is declared as
    the default of its `log` field -- `on={"_log": True}`. That default
    is read off the field table: a default a subclass overrides is not
    bound as a class attribute, so `cls._log` would answer with the
    value the base class wrote (`False`) for every one of them.
    """
    return bool(field_default(cls, "_log", False))


def _data2values(
    data: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    """
    Convert `data` to `values`.
    * If `store == "values"`, return `data` as is.
    * If `store == "coefficients"`, decode the spline coefficients.
    """
    if data is None or store == StoreEnum.VALUES:
        return data
    return coeff2value_field(data, degree=degree, bound=bound)


def _values2data(
    values: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    """
    Convert `values` to `data`.
    * If `store == "values"`, return `values` as is.
    * If `store == "coefficients"`, return the spline coefficients.
    """
    if values is None or store == StoreEnum.VALUES:
        return values
    return value2coeff_field(values, degree=degree, bound=bound)


def _data2coeffs(
    data: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    """
    Convert `data` to the spline coefficients of the field.
    * If `store == "coefficients"`, return `data` as is.
    * If `store == "values"`, fit the spline to them.
    """
    if data is None or store == StoreEnum.COEFFICIENTS:
        return data
    return value2coeff_field(data, degree=degree, bound=bound)


def _coeffs2data(
    coeffs: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    """
    Convert the spline coefficients of a field to `data`.
    * If `store == "coefficients"`, return `coeffs` as is.
    * If `store == "values"`, evaluate the spline at the grid points.
    """
    if coeffs is None or store == StoreEnum.COEFFS:
        return coeffs
    return coeff2value_field(coeffs, degree=degree, bound=bound)


def _inv_coords(coords: ArrayProtocol) -> ArrayProtocol:
    """
    A coordinate field (as values) is the identity grid plus a
    displacement, so it is inverted by inverting that displacement and
    adding the grid back.

    This reads the coordinates as living in the units of their own grid.
    """
    # FIXME
    #   A better approach is to regress out the affine transformation
    #   from the field, such that the normalised field is close to
    #   being in voxel space. The objective would be
    #   ``||aff @ field - idgrid||_F``.

    # Compute the identity coordinates mapping.
    # The grid is built on the backend the field lives on,
    # rather than on whichever backend happens to be selected.
    ab = get_array_backend(coords)
    with backend(ab):
        idgrid = CartesianField(shape=coords.shape[:-1]).field

    # Invert the coordinates field via the equivalent displacement field.
    return inverse_disp(coords - idgrid) + idgrid


def _refuse_log(
    xform: Transformation, log: tx.Optional[bool], tangent: str
) -> None:
    """
    `log=True` builds the tangent subclass (`tangent`). An instance of
    any other class reads its `data` as the map, so it refuses the flag,
    whether it is passed to a class that `log=True` does not select (an
    io subclass, say) or assigned in place, which cannot change the
    class of an instance.

    The tangent subclass runs this too, through the `__post_init__` it
    inherits, and its own `log` is true: the check is for the classes
    that hold the map.
    """
    if not log or _is_tangent(type(xform)):
        return
    raise TypeError(
        f"A {type(xform).__name__} holds the map, not its tangent about the "
        f"identity, so its log flag cannot be set: log=True builds a "
        f"{tangent}. To convert a transformation to its tangent, use "
        f"t.to(log=True)."
    )
