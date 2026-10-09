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

from numbers import Integral, Real

import typing_extensions as tx
from bagof.magic import InitVar, NotKwOnly

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
from brainhops.backends import backend, get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.enums import (
    BoundaryCondition,
    InterpolationOrder,
    StoreEnum,
)
from brainhops.datamodel.systems import CoordinateSystem

from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import require_endomorphism
from .modes import ModeLike

_TypeReference = tx.ClassVar[tx.Optional[tx.Type[Transformation]]]
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]

# --- helpers ----------------------------------------------------------

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the cached views listed in `derived_fields`."""


def _invalidating_property(*args, **kwargs) -> property:
    """Build a [`smartproperty`][] whose assignment clears the cached views."""
    kwargs.setdefault("invalidates", _FORGET_VIEWS)
    return smartproperty(*args, **kwargs)


def _alias(name: str, source: str, fset: bool = True) -> property:
    """Build a property that reads, and possibly writes, another attribute.

    `source` may be a dotted path, such as `"forward.degree"`. Such an alias is
    read-only whatever `fset` says, because a wrapper reports what it wraps and
    never writes into it.
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
    """Base class of the transformations that hold their own parameters.

    A concrete transformation, such as an affine matrix or a displacement
    field, stores the numbers that define its map, whereas a sequence is built
    from other transformations. Computing a concrete transformation converts it
    to the cheapest type that still represents it. Its inverse and its square
    root are lazy wrappers, which compute their parameters only when these
    parameters are read.
    """

    _reverseof: _TypeReference = None
    """Concrete type of the resolved inverse, if it differs from this type.

    A type whose name states a direction, such as `VoxelToLPS`, has the other
    type of its pair, `LPSToVoxel`, as the type of its inverse. Only the type
    defined second sets this attribute, and `__init_subclass__` then sets it on
    the first type so that the pair points both ways.
    """

    map_field: tx.ClassVar[tx.Optional[str]] = None
    """Name of the view that gives the map itself, such as `matrix`.

    A family is a class that declares this attribute, together with its
    subclasses, such as `Affine` and its tangent class `AffineExponential`.
    The view gives the map whatever the form in which `data` stores it, and
    `from_instance` copies the map from one class of the family to another
    through the view. The attribute is `None` for the identity and for a grid,
    whose map is generated from its shape.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Read from `cls.__dict__`, not with `getattr`, so that only a class
        # declaring `_reverseof` in its own body claims a pair. An inherited
        # value would let a refinement such as `MyLPSToVoxel(LPSToVoxel)`, or a
        # stand-in class that Magic builds while reading the MRO, take over
        # the pairing with `VoxelToLPS`.
        if (other := cls.__dict__.get("_reverseof")) is not None:
            other._reverseof = cls

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        """Convert the transformation to the cheapest type that represents it.

        Parameters
        ----------
        mode : ModeLike, default=True
            Kinds of transformation that may be composed, used only when
            `factor` is true. A concrete transformation has nothing to compose
            on its own, so otherwise only `simplify` decides whether it is
            converted.
        simplify : SimplifyLike, default="analytic"
            How much effort to spend on simplification. `"analytic"` decides
            from the type alone, `"numeric"` also reads the parameter values,
            and `"none"` skips simplification.
        factor : bool, default=False
            Whether to split the transformation into independent factors, one
            for each group of axes that transform together. To do so, the
            transformation is computed as a sequence of one element. A
            diagonal affine transformation, for example, splits into one
            factor per axis.

        Returns
        -------
        Transformation
            The simplified transformation.
        """
        if factor:
            # The sequence performs the factoring and never passes `factor`
            # back to the `compute()` of its elements, so this call does not
            # recurse.
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        return _simplify(self, policy=simplify)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Create an instance from an instance of a similar class.

        When both classes belong to the same family, the map is copied through
        the [`map_field`][] view rather than as the stored array. A tangent
        (`log=True`) copied into the plain class of its family is therefore
        exponentiated, and an affine transformation copied into a tangent class
        is stored as its logarithm. The metadata that both classes share, such
        as the flags of a field, is copied too, except `log`, because the two
        classes may store the map in different forms. Everything else is as in
        [`DataModelBase.from_instance`][brainhops.datamodel.base.DataModelBase.from_instance].
        """
        view = cls.map_field
        if view is not None and _same_family(cls, other):
            shared = set(cls.metadata_fields) & set(other.metadata_fields)
            for name in shared - {"log"}:
                kwargs.setdefault(name, getattr(other, name))
            _copy_map(other, cls, view, kwargs)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        # The inverse is an `Inverse` wrapper that is also an instance of the
        # family of this transformation. The wrapper holds this transformation
        # as its `forward` and is evaluated only when a parameter is read or
        # computed, so a transformation composed with its inverse can cancel
        # out without any arithmetic.
        if is_identity(self):
            # An identity gains nothing from being lazy.
            if (reverse := type(self)._reverseof) is not None:
                # A paired type inverts to the other half of its pair, which
                # states the swapped direction.
                return reverse(input=self.output, output=self.input)
            return self.to(input=self.output, output=self.input)
        # The `Inverse` operator selects the lazy wrapper class of this family.
        obj = nocycles.OPERATORS["inverse"](self)
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        # The square root is a lazy wrapper that is also an instance of the
        # family, and it computes its parameter when the parameter is read. A
        # family without such a wrapper overrides `sqrt()`.
        require_endomorphism(self, "square root")
        if is_identity(self):
            obj = self
        else:
            obj = nocycles.OPERATORS["sqrt"](self)
        return obj.compute(**kwargs) if compute else obj


class TransformationField(ConcreteTransformation):
    """Base class of the dense fields of displacements or coordinates.

    The field is stored in `data`, either as its values or, when `store` is
    `"coefficients"`, as the coefficients of a spline of degree `degree`. The
    `field` view and the `field=` keyword always give the values, and the flags
    only say how the values are stored in `data`.
    `DisplacementField(field=u, degree=3, store="coefficients")` therefore
    holds the cubic coefficients of `u`, the same data as
    `DisplacementField(field=u, degree=3).to(store="coefficients")`. An array
    that is already encoded is passed as `data=` or `coefficients=`.

    The `field` view is decoded once and cached until `data` or a flag is
    assigned. Whereas `.to(store="coefficients")` re-encodes the field,
    assigning a flag (`t.store = "coefficients"`) reinterprets the stored array
    without changing it.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "field"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = "degree", "bound", "store"
    derived_fields: _FieldNames = "field", "values", "coefficients", "_inverse"

    # --- attributes ---------------------------------------------------
    # The data and the flags are stored under private names, behind properties
    # whose assignment clears the cached views. The constructor still takes
    # `data=`, `degree=`, and so on.

    _data: NotKwOnly[tx.Optional[ArrayProtocol]] = None
    """Stored field, of shape `(*shape, ndim)`.

    It holds the values, or the spline coefficients when `store` is
    `"coefficients"`. The map is read through `field`.
    """

    _degree: InterpolationOrder = InterpolationOrder.linear
    """Degree of the spline that interpolates the field."""

    _bound: tx.Union[BoundaryCondition, float] = BoundaryCondition.nearest
    """Boundary condition for coordinates outside the field of view.

    A float is a constant fill value.
    """

    _store: tx.Optional[StoreEnum] = None
    """Whether `data` holds spline coefficients or values (see `store`)."""

    _field: InitVar[tx.Optional[ArrayProtocol]] = None
    """Field as values, accepted in place of `data`.

    It is encoded according to `store` before it is stored.
    """

    _values: InitVar[tx.Optional[ArrayProtocol]] = None
    """Same as `field`."""

    _coefficients: InitVar[tx.Optional[ArrayProtocol]] = None
    """Field as spline coefficients, accepted in place of `data`.

    It sets `store` to `"coefficients"` unless the flag is given. With
    `store="values"`, the spline is evaluated before the field is stored.
    """

    # --- [meta]data views ---------------------------------------------

    data = _invalidating_property("data")
    degree = _invalidating_property("degree")
    bound = _invalidating_property("bound")

    @_invalidating_property
    def store(self) -> StoreEnum:
        """Whether `data` holds the values or the spline coefficients.

        An unset flag reads as `"values"`, because every constructor argument
        except `coefficients=` gives the map as values. When the constructor
        receives `coefficients=` without a flag, it sets the flag to
        `"coefficients"`.
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
        # `coefficients=` declares that the array holds coefficients, so it
        # sets the flag when the constructor was given none.
        if (
            arguments.get("store") is None
            and arguments.get("coefficients") is not None
        ):
            self.store = StoreEnum.coefficients
        super()._set_derived_fields(arguments)

    # --- derived views ------------------------------------------------

    @lazyproperty
    def values(self) -> tx.Optional[ArrayProtocol]:
        """Field as values, of shape `(*shape, ndim)`.

        It is `data` itself when `store` is `"values"`; otherwise it is decoded
        from the coefficients once and cached.
        """
        return _data2values(self.data, self.store, self.degree, self.bound)

    @values.setter
    def values(self, value: tx.Optional[ArrayProtocol]) -> None:
        # Fit coefficients if the field stores them. Assigning `data` clears
        # every view derived from it,
        self.data = _values2data(value, self.store, self.degree, self.bound)
        # so the input can be cached only afterwards.
        self._cache_values = value

    field = _alias("field", "values")

    @lazyproperty
    def coefficients(self) -> tx.Optional[ArrayProtocol]:
        """Field as spline coefficients, of shape `(*shape, ndim)`.

        It is `data` itself when `store` is `"coefficients"`; otherwise it is
        fitted to a spline of degree `degree` once and cached.
        """
        return _data2coeffs(self.data, self.store, self.degree, self.bound)

    @coefficients.setter
    def coefficients(self, coeffs: tx.Optional[ArrayProtocol]) -> None:
        # Evaluate the spline if the field stores values. Assigning `data`
        # clears every view derived from it,
        self.data = _coeffs2data(coeffs, self.store, self.degree, self.bound)
        # so the input can be cached only afterwards.
        self._cache_coefficients = coeffs

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        """Parameter of the inverse field, in the encoding of this field.

        Each family derives the parameter here and caches it until `data` or a
        flag is assigned, so the lazy inverse runs the inversion only once.
        """
        raise NotImplementedError


class DisplacementField(TransformationField, polymorphic=True):
    """Displacement field on a regular grid, which maps the grid to itself.

    The `log` flag says what `data` describes. By default, `data` holds a
    displacement; when `log=True`, it holds a stationary velocity whose flow
    at time one is the map.
    `DisplacementField(..., log=True)` builds a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField],
    whose `field` view integrates the velocity, so a plain displacement field
    always holds displacements.
    """

    # --- class attributes ---------------------------------------------

    _base_metadata_fields: _FieldNames = TransformationField.metadata_fields
    metadata_fields: _FieldNames = (
        *_base_metadata_fields,
        "log",
    )

    # --- attributes ---------------------------------------------------

    _log: bool = False
    """Whether `data` holds a stationary velocity rather than a displacement.

    The velocity is the tangent of the map about the identity, and a field that
    holds one is a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField].
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
        """Parameter of the inverse field, cached.

        The values are inverted by treating the grid as a mesh of small affine
        cells, and the result is re-encoded according to the `store`, `degree`
        and `bound` flags of this field.
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
    """Coordinates field on a regular grid, whose input space is the grid."""

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
    """Identity over a regular grid of coordinates.

    A grid stores only its `shape`. The coordinates of its points (`field`) and
    their encoding (`data`) are generated on demand and cached until the shape
    or a flag is assigned.
    """

    # --- class attributes ---------------------------------------------

    # The map is generated from the shape, so there is no map to copy or
    # re-encode.
    map_field: tx.ClassVar[tx.Optional[str]] = None

    _base_derivied_fields: _FieldNames = CoordinatesField.derived_fields
    derived_fields: _FieldNames = (*_base_derivied_fields, "data")
    data_fields: _FieldNames = ("shape",)

    # --- attributes ---------------------------------------------------

    _shape: NotKwOnly[tx.Optional[tx.Tuple[int, ...]]] = None
    """Shape of the grid."""

    # These annotations remove the inherited fields from `__init__`, because
    # a grid is fully defined by its shape and takes neither `data=` nor
    # `field=`.

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
        """Coordinates of the grid points, of shape `(*shape, ndim)`.

        The coordinates are floating point (`dtype=float`), and so are the
        spline coefficients that `data` may hold.
        """
        if self.shape is None:
            return None
        ab = get_array_backend()
        grid = [ab.arange(s, dtype=float) for s in self.shape]
        return ab.stack(ab.meshgrid(*grid, indexing="ij"), -1)

    values = _alias("values", "field", fset=False)

    @lazyproperty
    def data(self) -> tx.Optional[ArrayProtocol]:
        """Coordinates of the grid points, encoded according to the flags."""
        return _values2data(self.field, self.store, self.degree, self.bound)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # A grid is the identity over its own coordinates, so its inverse is
        # itself with the endpoints swapped, and there is nothing to defer.
        cls = type(self)
        return cls(shape=self.shape, input=self.output, output=self.input)

    # A grid is the identity, so it is its own square and square root. It is
    # returned as it is, not as an `Identity`, because it is also a sampling
    # domain.

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square")
        return self.compute(**kwargs) if compute else self

    def sqrt(self, compute: bool = False, **kwargs) -> Transformation:
        require_endomorphism(self, "square root")
        return self.compute(**kwargs) if compute else self


@kinds.Affine
class Affine(ConcreteTransformation, polymorphic=True):
    """Affine transformation.

    `data` is a matrix of shape `(No, Ni+1)`, whose last column is the
    translation. `Affine(..., log=True)` builds an
    [`AffineExponential`][brainhops.datamodel.transformations.AffineExponential],
    whose `data` is the tangent of the map about the identity.
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
    # `data` and `log` are stored under private names, behind properties whose
    # assignment clears the views derived from them, as for a field.

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """Matrix of shape `(No, Ni+1)`, whose last column is the translation.

    It maps `Ni` input axes to `No` output axes. `None` stands for the
    identity.
    """

    _log: bool = False
    """Whether `data` is the tangent of the map rather than its matrix.

    A transformation that holds the tangent is an
    [`AffineExponential`][brainhops.datamodel.transformations.AffineExponential].
    """

    _matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Matrix of shape `(No, Ni+1)`, accepted in place of `data`."""

    _homogeneous_matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Homogeneous matrix of shape `(No+1, Ni+1)`, accepted for `data`."""

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
    """Linear transformation.

    `data` is a matrix of shape `(No, Ni)`. `Linear(..., log=True)` builds a
    [`LinearExponential`][brainhops.datamodel.transformations.LinearExponential],
    whose `data` is the tangent of the map about the identity.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "matrix"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = ("log",)
    derived_fields: _FieldNames = "matrix", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """Matrix of shape `(No, Ni)`. `None` stands for the identity."""

    _log: bool = False
    """Whether `data` is the tangent of the map rather than its matrix.

    A transformation that holds the tangent is a
    [`LinearExponential`][brainhops.datamodel.transformations.LinearExponential]
    or a
    [`RotationExponential`][brainhops.datamodel.transformations.RotationExponential].
    """

    _matrix: InitVar[tx.Optional[npmatrix[Real]]] = None
    """Matrix, accepted in place of `data`."""

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
    """Rotation, that is, an orthogonal linear map with determinant one.

    `data` is a matrix of shape `(N, N)`. `Rotation(..., log=True)` builds a
    [`RotationExponential`][brainhops.datamodel.transformations.RotationExponential],
    whose `data` is the tangent of the map about the identity.
    """

    # TODO: implement rotations in other representations (quaternions, Euler
    # angles, ...).

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npmatrix[Real]]] = None
    """Matrix of shape `(N, N)`, with determinant one.

    `None` stands for the identity.
    """

    # --- views --------------------------------------------------------

    @lazyproperty
    def _inverse(self) -> tx.Optional[ArrayProtocol]:
        if self.matrix is None:
            return None
        return self.matrix.T


@kinds.Permutation
class Permutation(ConcreteTransformation):
    """Permutation of the axes."""

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "permutation"

    data_fields: _FieldNames = ("data",)
    derived_fields: _FieldNames = "permutation", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """Vector of shape `(N,)` giving the input axis of each output axis.

    Element `i` is the input axis that feeds output axis `i`. `None` stands for
    the identity.
    """

    _permutation: InitVar[tx.Optional[npvector[Integral]]] = None
    """Permutation vector, accepted in place of `data`."""

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
    """Scaling of the axes.

    `data` holds one scale factor per axis. `Scaling(..., log=True)` builds a
    [`ScalingExponential`][brainhops.datamodel.transformations.ScalingExponential],
    whose `data` holds the logarithms of the factors.
    """

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "scale"

    data_fields: _FieldNames = ("data",)
    metadata_fields: _FieldNames = ("log",)
    derived_fields: _FieldNames = "scale", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Real]]] = None
    """Vector of shape `(N,)` of scale factors, one per axis.

    `None` stands for the identity.
    """

    _log: bool = False
    """Whether `data` holds the logarithms of the scale factors.

    The logarithms are the tangent of the map about the identity, and a
    transformation that holds them is a
    [`ScalingExponential`][brainhops.datamodel.transformations.ScalingExponential].
    """

    _scale: InitVar[tx.Optional[npvector[Real]]] = None
    """Scale factors, accepted in place of `data`."""

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
    """Translation by a vector."""

    # --- class attributes ---------------------------------------------

    map_field: tx.ClassVar[str] = "translation"

    data_fields: _FieldNames = ("data",)
    derived_fields: _FieldNames = "translation", "_sqrt", "_inverse"

    # --- attributes ---------------------------------------------------

    _data: NotKwOnly[tx.Optional[npvector[Real]]] = None
    """Vector of shape `(N,)` to translate by.

    `None` stands for the identity.
    """

    _translation: InitVar[tx.Optional[npvector[Real]]] = None
    """Translation vector, accepted in place of `data`."""

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
    """Identity transformation.

    When the input and output systems differ, the identity maps the input axes
    to the output axes in order. The identity has no parameter, so its `data`
    is always `None` and is not a constructor argument.
    """

    # --- views --------------------------------------------------------

    @property
    def data(self) -> None:
        """Always `None`, since the identity stores nothing."""
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
# Module-level aliases of the `Transformation.is_*` methods, kept for backward
# compatibility.

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
    """Return whether `other` belongs to the same family as `cls`.

    The family of `cls` is the closest base class that declares `map_field`,
    together with its subclasses. `Affine` and `Linear` are therefore
    different families, although both name `matrix` as their view, because
    their matrices have different shapes.
    """
    root = next(b for b in cls.__mro__ if "map_field" in b.__dict__)
    return isinstance(other, root)


def _copy_map(
    other: Transformation,
    cls: tx.Type[Transformation],
    view: str,
    kwargs: tx.Dict[str, tx.Any],
) -> None:
    """Add the map of `other` to `kwargs`, in the form that `cls` stores.

    `view` is the name of the view that gives the map, such as `matrix`. When
    both classes hold the map, or both hold the tangent, the stored array is
    carried over untouched, so that a tangent is not exponentiated and then
    turned back into a tangent with a logarithm.
    """
    out_tangent = bool(kwargs.get("log", _is_tangent(cls)))
    if out_tangent == bool(getattr(other, "log", False)):
        kwargs.setdefault("data", other.data)
    elif out_tangent:
        # The output class holds the tangent and `other` holds the map. The
        # map is passed through the view, and the output class stores its
        # logarithm.
        kwargs.setdefault(view, getattr(other, view))
    else:
        # The output class holds the map, which the view of a tangent
        # already gives.
        kwargs.setdefault("data", getattr(other, view))
    if "log" in cls.metadata_fields:
        kwargs.setdefault("log", out_tangent)


def _is_tangent(cls: tx.Type[Transformation]) -> bool:
    """Return whether `cls` holds the tangent of its map.

    A tangent class declares `True` as the default of its `_log` field. The
    default is read from the field table, because a default overridden in a
    subclass is not bound as a class attribute.
    """
    return bool(field_default(cls, "_log", False))


def _data2values(
    data: tx.Optional[ArrayProtocol],
    store: StoreEnum,
    degree: InterpolationOrder,
    bound: tx.Union[BoundaryCondition, float],
) -> tx.Optional[ArrayProtocol]:
    """Convert stored data to values.

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
    """Convert values to stored data.

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
    """Convert stored data to spline coefficients.

    * If `store == "coefficients"`, return `data` as is.
    * If `store == "values"`, fit a spline to the values.
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
    """Convert spline coefficients to stored data.

    * If `store == "coefficients"`, return `coeffs` as is.
    * If `store == "values"`, evaluate the spline at the grid points.
    """
    if coeffs is None or store == StoreEnum.COEFFS:
        return coeffs
    return coeff2value_field(coeffs, degree=degree, bound=bound)


def _inv_coords(coords: ArrayProtocol) -> ArrayProtocol:
    """Invert a field of coordinates given as values.

    A coordinate field is an identity grid plus a displacement, so the
    displacement is inverted and the grid is added back. The coordinates are
    assumed to be in units of their own grid.
    """
    # TODO: regress the affine part out of the field first, so that the
    # normalized field is close to voxel space (minimize ||aff @ field -
    # idgrid||_F).

    # Build the identity grid on the backend of the field, not on the selected
    # one.
    ab = get_array_backend(coords)
    with backend(ab):
        idgrid = CartesianField(shape=coords.shape[:-1]).field

    return inverse_disp(coords - idgrid) + idgrid


def _refuse_log(
    xform: Transformation, log: tx.Optional[bool], tangent: str
) -> None:
    """Refuse the `log` flag on a class that does not hold a tangent.

    `log=True` builds the tangent subclass of a family. Every other class, such
    as a reader subclass, reads `data` as the map and refuses the flag.

    Raises
    ------
    TypeError
        If `log` is true and the class does not hold a tangent.
    """
    if not log or _is_tangent(type(xform)):
        return
    raise TypeError(
        f"A {type(xform).__name__} holds the map, not its tangent about the "
        f"identity, so its log flag cannot be set: log=True builds a "
        f"{tangent}. To convert a transformation to its tangent, use "
        f"t.to(log=True)."
    )
