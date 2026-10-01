# stdlib
from functools import wraps
from numbers import Integral, Real

# dependencies
import typing_extensions as tx

# core
from brainhops._core.affines import inv as inverse_affine
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol, Derived, npmatrix, npvector
from brainhops._ext.invfield import inverse as inverse_disp

# api
from brainhops.backends import backend, get_array_backend
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.datamodel.systems import _is_unknown

# internals
from .base import Transformation
from .concrete import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from .modes import ModeLike, mode_admits, normalize_modes
from .registries import INVERSE_CACHE, register_inverse
from .simplify import SimplifyLike
from .simplify import simplify as _simplify

# typing
if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)


def _invcache(func: tx.Callable) -> property:
    """A parameter of an inverse, derived on demand and cached.

    The materialized value is cached on the **forward** transform, under
    `registries.INVERSE_CACHE`, rather than on the wrapper. A wrapper
    rebuilt by `replace` or `.to(...)` keeps the same forward transform, so
    the cache survives the rebuild and an expensive inversion is not run
    twice.

    The cache assumes the forward transform is not mutated in place after
    it is wrapped: a forward transform whose parameter is replaced by
    editing the same object would keep serving the stale inverse.
    """
    name = func.__name__

    @property
    @wraps(func)
    def parameter(self: "Inverse") -> tx.Any:
        forward = self.forward
        if forward is None:
            return None
        cache = getattr(forward, INVERSE_CACHE, None)
        if cache is None:
            cache = {}
            setattr(forward, INVERSE_CACHE, cache)
        if name not in cache:
            cache[name] = func(self)
        return cache[name]

    return parameter


@register_inverse
class Inverse(Transformation, tx.Generic[TRANSFORMATION], polymorphic=True):
    """The inverse of a transformation, resolved on demand.

    An `Inverse` holds a forward transformation and represents its
    inverse. The inverse is not computed when the wrapper is built. It is
    computed only when the wrapper is applied, computed, or converted to a
    concrete type. Placed next to its forward transformation in a
    [`Sequence`][], the two cancel to the identity, and no inverse is ever
    computed.

    Constructing `Inverse(forward=t)` represents the inverse of any
    transformation `t`. Each family of transformations also has its own
    typed inverse, such as [`InverseAffine`][] or
    [`InverseDisplacementField`][], which a transformation returns from its
    `inverse()` method. A typed inverse remains an instance of the family
    it inverts, so composition and the kind checks treat it exactly like a
    forward transformation of that family.
    """

    # --- class attributes ---------------------------------------------

    # The forward transformation type a typed inverse inverts. It is unset
    # on the generic `Inverse` front-door and set on each typed subclass,
    # which is what the materialization below rebuilds. Which wrapper a
    # given transform gets is decided polymorphically, from the `on={...}`
    # predicates below, so nothing is registered anywhere.
    _inverseof: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Transformation],  # TRANSFORMATION
        tx.Doc("The forward transformation whose inverse this represents."),
    ] = None

    # --- properties ---------------------------------------------------

    # An inverse maps the forward's output back to its input, so it knows
    # its own endpoints without being told: they are the forward's,
    # swapped. An endpoint declared on the wrapper still wins -- that is
    # what `smartproperty` does -- so an explicit override is honoured.

    @smartproperty(fallback_when=_is_unknown)
    def input(self) -> tx.Optional["CoordinateSystem"]:
        return self.forward.output if self.forward is not None else None

    @smartproperty(fallback_when=_is_unknown)
    def output(self) -> tx.Optional["CoordinateSystem"]:
        return self.forward.input if self.forward is not None else None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False) -> Transformation:
        """Return the forward transformation, with the endpoints restored.

        The inverse of an inverse is the original forward transformation.
        An endpoint edit made on the wrapper is carried onto it.
        """
        forward = self.forward
        if forward is None:
            return Identity(input=self.input, output=self.output)
        new_input = forward.input if _is_unknown(self.output) else self.output
        new_output = forward.output if _is_unknown(self.input) else self.input
        if new_input is not forward.input or new_output is not forward.output:
            forward = forward.to(input=new_input, output=new_output)
        if compute:
            forward = forward.compute()
        return forward

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Resolve the inverse, if the mode admits it.

        Materializing an inverse is the one thing this wrapper exists to
        put off, so it happens here and nowhere else: never in a
        simplifier, which may only rewrite for free. The compose `mode` is
        the gate -- a mode that does not admit this wrapper leaves it lazy,
        so an adjacent pair can still cancel in a sequence -- and
        `simplify` then still downcasts what it wraps. `factor` is forwarded
        to the materialized result.
        """
        modes = normalize_modes(mode)
        if mode_admits(self, modes):
            return self._materialize().compute(
                mode, simplify=simplify, factor=factor
            )
        return _simplify(self, policy=simplify)

    def to(
        self,
        cls: tx.Optional[tx.Type[Transformation]] = None,
        *,
        lossy: bool = False,
        **kwargs,
    ) -> Transformation:
        if cls is None or cls is type(self):
            # An endpoint or metadata edit keeps the inverse unresolved,
            # reusing the forward transform and its cached materialization.
            return super().to(cls, **kwargs)
        # A conversion to another type, including the forward type,
        # materializes the concrete inverse first, then converts onward.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    # --- helpers ------------------------------------------------------

    def _materialize(self) -> Transformation:
        # The concrete inverse. A typed inverse builds a plain instance of
        # the forward type holding the inverted parameter, with the
        # wrapper's (swapped) endpoints -- or, when the forward type is
        # half of a pair that names the direction it maps, an instance of
        # the other half. The generic front-door defers to the forward
        # transform's own inverse, which resolves to the typed inverse of
        # that family.
        forward = self.forward
        inverseof = type(self)._inverseof
        if inverseof is None:
            # Not a concrete inverse, but no forward to resolve to identity.
            if forward is None:
                return Identity(input=self.input, output=self.output)

            # Compute an explicit inverse, and edit its spaces.
            resolved = forward.inverse()
            edits = {}
            if not _is_unknown(self.input):
                edits["input"] = self.input
            if not _is_unknown(self.output):
                edits["output"] = self.output
            return resolved.to(**edits)

        if forward is None:
            # No forward, but we know the concrete inverse type
            return inverseof(input=self.input, output=self.output)
        changes = {
            "input": self.input,
            "output": self.output,
            **{name: getattr(self, name) for name in inverseof.data_fields},
        }
        # This is the path that matters for a type that names a direction:
        # the wrapper wears no direction of its own, but what it materializes
        # into does, and rebuilding the forward type around an inverted
        # parameter would state the direction backwards. The other half of
        # the pair states it right.
        reverse = getattr(type(forward), "_reverse_type", None)
        if reverse is not None:
            return reverse.from_instance(forward, **changes)
        return forward.to(**changes)


class InverseTranslation(
    Inverse[Translation],
    Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """The inverse of a [`Translation`][], resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Translation
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("translation",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Translation],
        tx.Doc("The translation whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    translation: Derived[tx.Optional[npvector[Real]]]

    @_invcache
    def translation(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.translation is None:
            return None
        return -self.forward.translation


class InverseScaling(
    Inverse,
    Scaling,  # [Scaling]
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The inverse of a [`Scaling`][], resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Scaling
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    scale: Derived[tx.Optional[npvector[Real]]]

    @_invcache
    def scale(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.scale is None:
            return None
        return 1.0 / self.forward.scale


class InversePermutation(
    Inverse,
    Permutation,  # [Permutation]
    on={"forward": partial(isinstance, PLACEHOLDER, Permutation)},
):
    """The inverse of a [`Permutation`][], resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Permutation
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("permutation",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Permutation],
        tx.Doc("The permutation whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    permutation: Derived[tx.Optional[npvector[Integral]]]

    @_invcache
    def permutation(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.permutation is None:
            return None
        backend = get_array_backend(self.forward.permutation)
        forward = self.forward.permutation
        inverse_permutation = backend.zeros(forward.shape, dtype=forward.dtype)
        for i, p in enumerate(forward):
            inverse_permutation[p] = i
        return inverse_permutation


class InverseRotation(
    Inverse[Rotation],
    Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # A `Rotation` is a `Linear`, so `Inverse(forward=rotation)` matches
    # `InverseLinear` just as well. The more specific wrapper wins: it
    # inverts by transposing rather than by solving a linear system.
    priority=1,
):
    """The inverse of a [`Rotation`][], resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Rotation
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Rotation],
        tx.Doc("The rotation whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_invcache
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.matrix is None:
            return None
        # A rotation is orthogonal, so its inverse is its transpose.
        return self.forward.matrix.T


class InverseLinear(
    Inverse[Linear],
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, Linear)},
):
    """The inverse of a [`Linear`][] transformation, resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Linear
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Linear],
        tx.Doc("The linear transformation whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_invcache
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.matrix is None:
            return None
        ab = get_array_backend(self.forward.matrix)
        return ab.linalg.inv(self.forward.matrix)


class InverseAffine(
    Inverse[Affine],
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The inverse of an [`Affine`][] transformation, resolved on demand."""

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = Affine
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    # --- attributes ---------------------------------------------------
    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_invcache
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        if self.forward.matrix is None:
            return None
        return inverse_affine(self.forward.matrix)


class InverseDisplacementField(
    Inverse[DisplacementField],
    DisplacementField,
    on={"forward": partial(isinstance, PLACEHOLDER, DisplacementField)},
):
    """The inverse of a [`DisplacementField`][], resolved on demand.

    The wrapper reports the `order`, `bound` and `coeff` of the forward
    field, and the inverse field it materializes preserves them. A field
    of spline coefficients is inverted by re-fitting, and its inverse is
    itself a field of coefficients.
    """

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = DisplacementField
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = ()
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "field",
        "order",
        "bound",
        "coeff",
    )

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[DisplacementField],
        tx.Doc("The displacement field whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    field: Derived[tx.Optional[ArrayProtocol]]
    order: Derived[InterpolationOrder]
    bound: Derived[tx.Union[BoundaryCondition, float]]
    coeff: Derived[bool]

    @_invcache
    def field(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward
        return _inv_disp(
            forward.field,
            coeff=forward.coeff,
            order=forward.order,
            bound=forward.bound,
        )

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff


class InverseCoordinatesField(
    Inverse[CoordinatesField],
    CoordinatesField,
    on={"forward": partial(isinstance, PLACEHOLDER, CoordinatesField)},
):
    """The inverse of a [`CoordinatesField`][], resolved on demand.

    The wrapper reports the `order`, `bound` and `coeff` of the forward
    field, and the inverse field it materializes preserves them. A field of
    spline coefficients is inverted by re-fitting, and its inverse is
    itself a field of coefficients.

    Placed next to the field it inverts in a [`Sequence`][], the two cancel
    and nothing is computed. That is the cheap path, and the one worth
    reaching for: a coordinate field has no closed-form inverse, so
    materializing this wrapper runs a mesh inversion.

    !!! warning "The coordinates must live on the grid they are sampled on"
        A coordinate field is inverted by reading it as the identity grid
        plus a displacement, inverting that displacement, and adding the
        grid back. The mesh inversion therefore assumes the coordinates are
        expressed in the units of the grid they are sampled on -- voxels,
        in practice. A field whose coordinates are in world units is
        inverted as though they were voxel coordinates, and the result,
        while well defined, falls outside the output lattice and is of
        little use. Compose the world-to-voxel affine into the field first.
    """

    # --- class attributes ---------------------------------------------

    _inverseof: tx.ClassVar[tx.Type[Transformation]] = CoordinatesField
    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = ()
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "field",
        "order",
        "bound",
        "coeff",
    )

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[CoordinatesField],
        tx.Doc("The coordinate field whose inverse this represents."),
    ] = None

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

    field: Derived[tx.Optional[ArrayProtocol]]
    order: Derived[InterpolationOrder]
    bound: Derived[tx.Union[BoundaryCondition, float]]
    coeff: Derived[bool]

    @_invcache
    def field(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward
        return _inv_coords(
            forward.field,
            coeff=forward.coeff,
            order=forward.order,
            bound=forward.bound,
        )

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _inv_disp(
    disp: ArrayProtocol, coeff: bool = False, **options
) -> ArrayProtocol:
    if disp is None:
        return None
    if not coeff:
        return inverse_disp(disp)
    # A coefficient field is inverted by re-fitting: the coefficients
    # are read out as values, the value field is inverted, and the
    # result is fitted back to coefficients.
    disp = coeff2value_field(disp, **options)
    disp = inverse_disp(disp)
    return value2coeff_field(disp, **options)


def _inv_coords(
    coords: ArrayProtocol, coeff: bool = False, **options
) -> ArrayProtocol:
    if coords is None:
        return None

    # A coordinate field is the identity grid plus a displacement, so
    # it is inverted by inverting that displacement and adding the grid
    # back. See the class docstring: this reads the coordinates as
    # living in the units of their own grid.
    #
    # FIXME
    #   A better approach is to regress out the affine transformation
    #   from the field, such that the normalised field is close to
    #   being in voxel space. The objective would be
    #   ``||aff @ field - idgrid||_F``.

    if coeff:
        # The coefficients are read out as values first: the arithmetic
        # below is on coordinates, not on spline coefficients.
        coords = coeff2value_field(coords, **options)

    # Compute the identity coordinates mapping.
    # The grid is built on the backend the field lives on,
    # rather than on whichever backend happens to be selected.
    ab = get_array_backend(coords)
    with backend(ab):
        idgrid = CartesianField(shape=coords.shape[:-1]).field

    # Invert the coordinates field via the equivalent displacement field.
    inverse_coords = inverse_disp(coords - idgrid) + idgrid

    if coeff:
        # Convert back to coefficients, so that the inverse uses the
        # same parameters as the forward field.
        inverse_coords = value2coeff_field(inverse_coords, **options)

    return inverse_coords
