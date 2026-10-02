"""The unary operators of a transformation, and their lazy results.

A transformation `T` that maps a space to itself has a square, a principal
square root, an exponential and a principal logarithm, alongside the
inverse that [`Inverse`][] already represents:

* `T.inverse()` is the map `U` such that `U @ T` is the identity;
* `T.square()` is `T @ T`;
* `T.sqrt()` is the principal `S` such that `S @ S` is `T`, the half-map;
* `T.exp()` is the flow at time one of the velocity `x -> T(x) - x`;
* `T.log()` is `x -> x + v(x)`, where `v` is the principal velocity whose
  flow at time one is `T`.

Reading a transformation as a velocity
--------------------------------------
The square and the square root are functions of the *map*: they do not
depend on how `T` is stored. The exponential and the logarithm are not:
they relate a map to a velocity, so a transformation has to be read as a
velocity first. brainhops reads it as its **displacement**,
`v(x) = T(x) - x`, for every kind of transformation:

* a [`DisplacementField`][] `u` is read as the stationary velocity field
  `u`, which is the usual way to store one, so `u.exp()` integrates it by
  scaling and squaring;
* an affine `x -> M x + t` is read as the linear velocity
  `v(x) = (M - I) x + t`, whose flow at time one is the matrix exponential
  of the homogeneous generator `[[M - I, t], [0, 0]]`. Its logarithm is
  `x -> (I + L) x + l`, where `[[L, l], [0, 0]]` is the principal matrix
  logarithm of the homogeneous matrix of `T`.

That reading does not depend on the coordinates, which is what keeps the
operators consistent with everything else: the identity and the zero
field both have the zero velocity, so the exponential and the logarithm
both fix the identity; a translation is its own exponential and its own
logarithm; an affine has the same exponential as the field that samples
it; and the exponential of `P^-1 @ T @ P` is `P^-1 @ exp(T) @ P`, so a
velocity stored in voxels, between a world-to-voxel affine and its
inverse, exponentiates in world space. It is *not* the matrix exponential
of the stored matrix, which would send the identity to `e I`.

Laziness
--------
As with [`Inverse`][], each operator returns a wrapper that holds the
transformation it acts on as its `forward`, and computes its parameter
only when it is read, applied, converted or computed. A typed wrapper is
an instance of the family its result belongs to -- [`Sqrt`][] of a
[`Rotation`][] is a `Rotation` -- so composition, conversion and the kind
checks treat it like any transformation of that family, and the kind
checks and the simplifiers reason from its `forward` rather than
materialize it. Square is the exception: `T.square()` is the sequence
`[T, T]`, which is lazy already.

The front doors [`Sqrt`][], [`Exp`][] and [`Log`][] build only the typed
wrappers registered below; anything else is refused when it is
constructed. The methods (`T.sqrt()`, ...) are the general entry point:
they also handle the transformations that need no wrapper (an identity, a
translation's exponential), the containers (a [`Sequence`][], a subspace
transformation) and the refusals.
"""

__all__ = [
    "Operation",
    "Sqrt",
    "Exp",
    "Log",
    "UNARY_OPERATORS",
]

# stdlib
import math
from functools import wraps
from numbers import Real
from operator import methodcaller
from types import MappingProxyType

# dependencies
import numpy as np
import scipy.linalg
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol, Derived, npmatrix, npvector

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.datamodel.systems import CoordinateSystem

# internals
from .base import Transformation
from .compose import compose
from .concrete import (
    Affine,
    DisplacementField,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from .errors import DomainError
from .inverse import InverseDisplacementField, _invcache
from .modes import ModeLike, mode_admits, normalize_modes
from .registries import register_exp, register_log, register_sqrt
from .simplify import SimplifyLike
from .simplify import simplify as _simplify
from .utils import with_endpoints

# ======================================================================
#
#                              B A S E
#
# ======================================================================

_CACHE = "_operation_param_cache"
"""
The attribute of the *forward* transform that caches the parameters its
operators materialize, for the reason `registries.INVERSE_CACHE` gives: a
wrapper rebuilt by `replace` or `.to(...)` keeps its forward, so the
expensive part is not computed twice. The key names the wrapper class, the
parameter and the wrapper's own metadata (the number of squaring steps of
a field exponential), so two operators of one forward never collide. The
cache assumes the forward is not mutated in place after it is wrapped.
"""


class Operation(Transformation):
    """An operator applied to a transformation, resolved on demand.

    The base of [`Sqrt`][], [`Exp`][] and [`Log`][]. It holds the
    transformation the operator acts on as its `forward`. Every operator
    here maps a space to itself, so the wrapper's endpoints are the
    forward's. The result is computed only when the wrapper is applied,
    computed, or converted to another type.
    """

    # --- class attributes ---------------------------------------------

    # The plain type a typed wrapper materializes into; unset on the front
    # doors, which only ever build a typed wrapper.
    _result: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

    # The name of the method that applies this operator, which is how a
    # wrapper whose forward has been simplified is rebuilt.
    _method: tx.ClassVar[str] = ""

    data_fields: tx.ClassVar[tx.Tuple[str]] = ("forward",)

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Transformation],
        tx.Doc("The transformation the operator acts on."),
    ] = None

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        return self.forward.input if self.forward is not None else None

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        return self.forward.output if self.forward is not None else None

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Resolve the operator, if the mode admits it.

        As for [`Inverse.compute`][], the compose `mode` is the gate: a mode
        that does not admit the wrapper leaves it unresolved, and
        `simplify` then still simplifies what it wraps. `factor` is
        forwarded to the materialized result.
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
            # An edit of the wrapper's own fields -- its endpoints, its
            # forward, its metadata -- keeps the operator unresolved, and
            # the forward keeps its cache.
            own = {"input", "output", "forward", *type(self).metadata_fields}
            if own.issuperset(kwargs):
                return replace(self, **kwargs) if kwargs else self
            cls = None
        # Anything else edits the result, which is materialized first.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    # --- helpers ------------------------------------------------------

    def _materialize(self) -> Transformation:
        # A plain instance of the result type, holding the derived
        # parameters and the wrapper's endpoints.
        cls = type(self)._result
        names = (*cls.data_fields, *cls.metadata_fields)
        params = {name: getattr(self, name) for name in names}
        return cls(input=self.input, output=self.output, **params)

    def _what(self) -> str:
        # "The square root of this Rotation", for error messages.
        name = {"sqrt": "square root", "exp": "exponential"}.get(
            self._method, "logarithm"
        )
        return f"The {name} of this {type(self.forward).__name__}"


def _cached(func: tx.Callable) -> property:
    """A parameter of an operator, derived on demand and cached."""
    name = func.__name__

    @property
    @wraps(func)
    def parameter(self: Operation) -> tx.Any:
        forward = self.forward
        if forward is None:
            return None
        cache = getattr(forward, _CACHE, None)
        if cache is None:
            cache = {}
            setattr(forward, _CACHE, cache)
        metadata = tuple(getattr(self, m) for m in type(self).metadata_fields)
        key = (type(self), name, *metadata)
        if key not in cache:
            cache[key] = func(self)
        return cache[key]

    return parameter


def _computed(
    t: Transformation, compute: bool, kwargs: tx.Dict[str, tx.Any]
) -> Transformation:
    return t.compute(**kwargs) if compute else t


@register_sqrt
class Sqrt(Operation, polymorphic="strict"):
    """The principal square root of a transformation, resolved on demand.

    The square root `S` of `T` is the transformation with `S @ S == T` --
    the half-transformation. Among the many square roots a map may have,
    the principal one is the one whose linear part has its eigenvalues in
    the open right half-plane. It exists, and is real, when the linear part
    of `T` has no eigenvalue on the closed negative real axis; otherwise the
    wrapper raises [`DomainError`][] when it is resolved. A field has a
    square root here only when it is an exponential, `exp(v)`, whose square
    root is `exp(v / 2)`.

    Build it with [`Transformation.sqrt`][]. Constructing `Sqrt(forward=t)`
    builds the typed wrapper of `t`'s family, and refuses a family that has
    none.
    """

    _method: tx.ClassVar[str] = "sqrt"

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation: `sqrt(T)` squared is `T`."""
        return _computed(with_endpoints(self.forward, self), compute, kwargs)


@register_exp
class Exp(Operation, polymorphic="strict"):
    """The exponential of a transformation, resolved on demand.

    The exponential of `T` is the flow at time one of the stationary
    velocity field `v(x) = T(x) - x`, its displacement (see the module
    documentation). For an affine `x -> M x + t`, it is the matrix
    exponential of `[[M - I, t], [0, 0]]`. For a [`DisplacementField`][],
    the field is integrated by scaling and squaring: it is divided by
    `2**steps` and composed with itself `steps` times, with the existing
    field composition and interpolation (the field's own `order` and
    `bound`). The displacements are in the voxels of the field's own grid,
    as for any [`DisplacementField`][], and `steps` is chosen by default so
    that the first step moves no point by more than an eighth of a voxel.

    Build it with [`Transformation.exp`][]. Constructing `Exp(forward=t)`
    builds the typed wrapper of `t`'s family, and refuses a family that has
    none.
    """

    _method: tx.ClassVar[str] = "exp"


@register_log
class Log(Operation, polymorphic="strict"):
    """The principal logarithm of a transformation, resolved on demand.

    The logarithm of `T` is `x -> x + v(x)`, where `v` is the principal
    stationary velocity whose flow at time one is `T`, so that
    `T.log().exp()` is `T` (see the module documentation). For an affine,
    `v` is read from the principal matrix logarithm of its homogeneous
    matrix, which exists, and is real, when the linear part has no
    eigenvalue on the closed negative real axis; otherwise the wrapper
    raises [`DomainError`][] when it is resolved.

    Build it with [`Transformation.log`][]. Constructing `Log(forward=t)`
    builds the typed wrapper of `t`'s family, and refuses a family that has
    none.
    """

    _method: tx.ClassVar[str] = "log"

    def exp(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation: `exp(log(T))` is `T`."""
        return _computed(with_endpoints(self.forward, self), compute, kwargs)


# ======================================================================
#
#                  T Y P E D   W R A P P E R S :   S Q R T
#
# ======================================================================


class SqrtTranslation(
    Sqrt,
    Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """The square root of a [`Translation`][]: half the translation."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Translation
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("translation",)

    forward: tx.Annotated[
        tx.Optional[Translation],
        tx.Doc("The translation whose square root this represents."),
    ] = None

    translation: Derived[tx.Optional[npvector[Real]]]

    @_cached
    def translation(self) -> tx.Optional[ArrayProtocol]:
        return self.forward.translation / 2


class SqrtScaling(
    Sqrt,
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The square root of a [`Scaling`][], for positive scales only."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Scaling
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose square root this represents."),
    ] = None

    scale: Derived[tx.Optional[npvector[Real]]]

    @_cached
    def scale(self) -> tx.Optional[ArrayProtocol]:
        scale = self.forward.scale
        _require_positive(scale, self._what())
        return get_array_backend(scale).sqrt(scale)


class SqrtRotation(
    Sqrt,
    Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # A `Rotation` is a `Linear`, so `SqrtLinear` matches it too. The
    # square root of a rotation is a rotation, which this wrapper keeps.
    priority=1,
):
    """The principal square root of a [`Rotation`][]: half the rotation.

    It is not defined for a rotation by a half turn in some plane, which
    has two square roots and no principal one.
    """

    _result: tx.ClassVar[tx.Type[Transformation]] = Rotation
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[Rotation],
        tx.Doc("The rotation whose square root this represents."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _linear_sqrt(self.forward.matrix, self._what())


class SqrtLinear(
    Sqrt,
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, (Linear, Permutation))},
):
    """The principal square root of a [`Linear`][] or [`Permutation`][]."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Linear
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[tx.Union[Linear, Permutation]],
        tx.Doc("The linear transformation whose square root this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _linear_sqrt(_linear_matrix(self.forward), self._what())


class SqrtAffine(
    Sqrt,
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The principal square root of an [`Affine`][] transformation."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Affine
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose square root this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _affine_sqrt(self.forward.matrix, self._what())


# ======================================================================
#
#                  T Y P E D   W R A P P E R S :   E X P
#
# ======================================================================


class ExpScaling(
    Exp,
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The exponential of a [`Scaling`][] `s`: the scaling `exp(s - 1)`."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Scaling
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose exponential this represents."),
    ] = None

    scale: Derived[tx.Optional[npvector[Real]]]

    @_cached
    def scale(self) -> tx.Optional[ArrayProtocol]:
        scale = self.forward.scale
        return get_array_backend(scale).exp(scale - 1)


class ExpLinear(
    Exp,
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, (Linear, Permutation))},
):
    """The exponential of a [`Linear`][] `M`: the matrix `expm(M - I)`."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Linear
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[tx.Union[Linear, Permutation]],
        tx.Doc("The linear transformation whose exponential this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _linear_exp(_linear_matrix(self.forward), self._what())


class ExpAffine(
    Exp,
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The exponential of an [`Affine`][] `x -> M x + t`.

    It is the matrix exponential of the generator `[[M - I, t], [0, 0]]`.
    """

    _result: tx.ClassVar[tx.Type[Transformation]] = Affine
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose exponential this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _affine_exp(self.forward.matrix, self._what())


class ExpDisplacementField(
    Exp,
    DisplacementField,
    on={"forward": partial(isinstance, PLACEHOLDER, DisplacementField)},
):
    """The exponential of a stationary velocity field, by scaling and squaring.

    The `forward` field is read as a stationary velocity field, in the
    voxels of its own grid, and integrated to the displacement field of its
    flow at time one. The wrapper reports the `order`, `bound` and `coeff`
    of the velocity, and the field it materializes uses them: they are the
    interpolation and boundary condition every squaring step samples the
    field with. A velocity of spline coefficients is integrated in values,
    and its exponential is a field of coefficients again.

    Its inverse is `exp(-v)`, integrated the same way rather than by
    inverting the field; its square root is `exp(v / 2)`; and its logarithm
    is `v`.
    """

    _result: tx.ClassVar[tx.Type[Transformation]] = DisplacementField
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = ("steps",)
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "field",
        "order",
        "bound",
        "coeff",
    )

    forward: tx.Annotated[
        tx.Optional[DisplacementField],
        tx.Doc("The stationary velocity field to integrate."),
    ] = None

    steps: tx.Annotated[
        tx.Optional[int],
        tx.Doc(
            """
            The number of squaring steps. By default, the smallest number
            for which the first step, `v / 2**steps`, moves no point by
            more than an eighth of a voxel.
            """
        ),
    ] = None

    field: Derived[tx.Optional[ArrayProtocol]]
    order: Derived[InterpolationOrder]
    bound: Derived[tx.Union[BoundaryCondition, float]]
    coeff: Derived[bool]

    @_cached
    def field(self) -> tx.Optional[ArrayProtocol]:
        return _exp_velocity(self.forward, self.steps)

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff

    def log(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the velocity: the logarithm of `exp(v)` is `v`.

        A field has no principal logarithm that brainhops computes, so `v`
        is the logarithm of the exponential it was built as.
        """
        return _computed(with_endpoints(self.forward, self), compute, kwargs)


class SqrtDisplacementField(
    Sqrt,
    DisplacementField,
    on={"forward": partial(isinstance, PLACEHOLDER, ExpDisplacementField)},
):
    """The square root of the exponential of a velocity: `exp(v / 2)`.

    It is integrated with the steps of `exp(v)` but one, so that squaring
    it reproduces the field `exp(v)` materializes.
    """

    _result: tx.ClassVar[tx.Type[Transformation]] = DisplacementField
    metadata_fields: tx.ClassVar[tx.Tuple[str]] = ()
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "field",
        "order",
        "bound",
        "coeff",
    )

    forward: tx.Annotated[
        tx.Optional[ExpDisplacementField],
        tx.Doc("The exponential whose square root this represents."),
    ] = None

    field: Derived[tx.Optional[ArrayProtocol]]
    order: Derived[InterpolationOrder]
    bound: Derived[tx.Union[BoundaryCondition, float]]
    coeff: Derived[bool]

    @_cached
    def field(self) -> tx.Optional[ArrayProtocol]:
        exp = self.forward
        return _exp_velocity(exp.forward, exp.steps, skip=1)

    @property
    def order(self) -> InterpolationOrder:
        return self.forward.order

    @property
    def bound(self) -> tx.Union[BoundaryCondition, float]:
        return self.forward.bound

    @property
    def coeff(self) -> bool:
        return self.forward.coeff


class _InverseExpDisplacementField(
    InverseDisplacementField,
    on={"forward": partial(isinstance, PLACEHOLDER, ExpDisplacementField)},
):
    # The inverse of `exp(v)` is `exp(-v)`. Picked over the generic field
    # inverse because it is the deeper subclass, it integrates `-v` instead
    # of inverting the materialized field, which is both exact in the limit
    # and cheaper. Still an `Inverse` whose `forward` is the exponential,
    # so the two cancel next to each other without computing anything.

    forward: tx.Annotated[
        tx.Optional[ExpDisplacementField],
        tx.Doc("The exponential whose inverse this represents."),
    ] = None

    field: Derived[tx.Optional[ArrayProtocol]]

    @_invcache
    def field(self) -> tx.Optional[ArrayProtocol]:
        exp = self.forward
        return _exp_velocity(exp.forward, exp.steps, negate=True)

    def _materialize(self) -> Transformation:
        # A plain field. The generic route rebuilds the forward's own type,
        # which here would resolve the exponential for nothing.
        return DisplacementField(
            field=self.field,
            order=self.order,
            bound=self.bound,
            coeff=self.coeff,
            input=self.input,
            output=self.output,
        )


# ======================================================================
#
#                  T Y P E D   W R A P P E R S :   L O G
#
# ======================================================================


class LogScaling(
    Log,
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The logarithm of a [`Scaling`][] `s > 0`: the scaling `1 + log(s)`."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Scaling
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("scale",)

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose logarithm this represents."),
    ] = None

    scale: Derived[tx.Optional[npvector[Real]]]

    @_cached
    def scale(self) -> tx.Optional[ArrayProtocol]:
        scale = self.forward.scale
        _require_positive(scale, self._what())
        return 1 + get_array_backend(scale).log(scale)


class LogLinear(
    Log,
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, (Linear, Permutation))},
):
    """The logarithm of a [`Linear`][] `M`: the matrix `I + logm(M)`."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Linear
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[tx.Union[Linear, Permutation]],
        tx.Doc("The linear transformation whose logarithm this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _linear_log(_linear_matrix(self.forward), self._what())


class LogAffine(
    Log,
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The logarithm of an [`Affine`][] transformation.

    With `[[L, l], [0, 0]]` the principal matrix logarithm of its
    homogeneous matrix, it is the affine `x -> (I + L) x + l`.
    """

    _result: tx.ClassVar[tx.Type[Transformation]] = Affine
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("matrix",)

    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose logarithm this is."),
    ] = None

    matrix: Derived[tx.Optional[npmatrix[Real]]]

    @_cached
    def matrix(self) -> tx.Optional[ArrayProtocol]:
        return _affine_log(self.forward.matrix, self._what())


# ======================================================================
#
#                          O P E R A T O R S
#
# ======================================================================

UNARY_OPERATORS: tx.Mapping[
    str, tx.Callable[[Transformation], Transformation]
] = MappingProxyType(
    {
        name: methodcaller(name)
        for name in ("inverse", "square", "sqrt", "exp", "log")
    }
)
"""
The unary operators of a transformation, by name.

Each value takes a transformation and returns the result of the method of
the same name, with its defaults: a lazy result where there is one. It is
the table an expression parser looks names up in, so that the operators it
accepts are exactly the ones a transformation implements.
"""


# ======================================================================
#
#                          N U M E R I C S
#
# ======================================================================

_RTOL = float(np.sqrt(np.finfo(np.float64).eps))
"""
The relative tolerance below which an eigenvalue counts as zero, or as
real, and an imaginary part as rounding error. Near the boundary of their
domain the principal square root and logarithm are ill-conditioned, so a
matrix that is within rounding of it is refused rather than answered.
"""

_FIRST_STEP = 0.125
"""
The largest displacement, in voxels, of the first step of scaling and
squaring. Arsigny et al. (2006) bound it by half a voxel; a smaller step
keeps the first-order step `id + v / 2**steps` accurate for velocities
whose gradient is large, at the price of two or three more squarings.
"""

_MAX_STEPS = 64
"""A velocity that would need more squarings than this is refused."""


def _to_host(array: ArrayProtocol) -> np.ndarray:
    # A small matrix or vector, as a float64 numpy array. The matrix
    # functions run on the host, whatever backend the parameter lives on.
    backend = get_array_backend(array)
    if backend is not np and backend.__name__.startswith("cupy"):
        array = array.get()
    return np.asarray(array, dtype=np.float64)


def _like(result: np.ndarray, like: ArrayProtocol) -> ArrayProtocol:
    # Return a host result on the backend, and in the floating dtype, of
    # the parameter it was computed from.
    dtype = np.dtype(like.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return get_array_backend(like).asarray(result.astype(dtype))


def _require_positive(scale: ArrayProtocol, what: str) -> None:
    backend = get_array_backend(scale)
    if bool(backend.any(scale <= 0)):
        raise DomainError(
            f"{what} is not defined: its scales must all be positive, but "
            f"they are {_to_host(scale).tolist()}."
        )


def _require_square(matrix: np.ndarray, what: str) -> None:
    rows, cols = matrix.shape
    if rows != cols:
        raise DomainError(
            f"{what} is not defined: it maps {cols} axes to {rows} axes, "
            "not a space to itself."
        )


def _require_principal(linear: np.ndarray, what: str) -> None:
    # The principal square root and logarithm of a real matrix exist, and
    # are real, exactly when it has no eigenvalue on the closed negative
    # real axis (Higham, Functions of Matrices, Thms 1.29 and 1.31).
    if not np.isfinite(linear).all():
        raise DomainError(f"{what} is not defined: it is not finite.")
    eigenvalues = np.linalg.eigvals(linear)
    size = np.abs(eigenvalues)
    scale = float(size.max(initial=0.0))
    if scale == 0.0 or (size <= _RTOL * scale).any():
        raise DomainError(
            f"{what} is not defined: its linear part is singular, so it has "
            "no principal square root or logarithm."
        )
    negative = (eigenvalues.real < 0) & (
        np.abs(eigenvalues.imag) <= _RTOL * size
    )
    if negative.any():
        values = sorted(float(v) for v in eigenvalues.real[negative])
        raise DomainError(
            f"{what} is not defined: its linear part has the eigenvalue(s) "
            f"{values} on the negative real axis, so it has no real "
            "principal square root or logarithm. A reflection, or a "
            "rotation by a half turn, is such a transformation."
        )


def _real(result: np.ndarray, what: str) -> np.ndarray:
    # The real part of a matrix function whose exact value is real, after
    # checking that the imaginary part is only rounding error.
    if np.iscomplexobj(result):
        scale = max(1.0, float(np.abs(result).max(initial=0.0)))
        if float(np.abs(result.imag).max(initial=0.0)) > _RTOL * scale:
            raise DomainError(f"{what} is not defined: it is not real.")
        result = result.real
    return result


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    # The square homogeneous matrix of a compact `(N, N + 1)` affine.
    n = matrix.shape[0]
    full = np.zeros((n + 1, n + 1))
    full[:n] = matrix
    full[n, n] = 1
    return full


def _linear_matrix(t: Transformation) -> ArrayProtocol:
    return t.matrix if isinstance(t, Linear) else t.to(Linear).matrix


def _linear_sqrt(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    _require_square(host, what)
    _require_principal(host, what)
    return _like(_real(scipy.linalg.sqrtm(host), what), matrix)


def _linear_exp(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    _require_square(host, what)
    generator = host - np.eye(len(host))
    return _like(scipy.linalg.expm(generator), matrix)


def _linear_log(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    _require_square(host, what)
    _require_principal(host, what)
    log = _real(scipy.linalg.logm(host), what)
    return _like(log + np.eye(len(host)), matrix)


def _affine_sqrt(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    n = host.shape[0]
    _require_square(host[:, :-1], what)
    _require_principal(host[:, :-1], what)
    # The homogeneous matrix adds the eigenvalue 1, whose square root is 1,
    # so its principal square root keeps the last row `[0, ..., 0, 1]`.
    root = _real(scipy.linalg.sqrtm(_homogeneous(host)), what)
    return _like(root[:n], matrix)


def _affine_exp(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    n = host.shape[0]
    _require_square(host[:, :-1], what)
    # The generator of the flow of `x -> (M - I) x + t`.
    generator = np.zeros((n + 1, n + 1))
    generator[:n] = host
    generator[:n, :n] -= np.eye(n)
    return _like(scipy.linalg.expm(generator)[:n], matrix)


def _affine_log(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    host = _to_host(matrix)
    n = host.shape[0]
    _require_square(host[:, :-1], what)
    _require_principal(host[:, :-1], what)
    # The principal logarithm of a homogeneous matrix has the last row
    # `[0, ..., 0]`; the velocity it describes is the compact block.
    log = _real(scipy.linalg.logm(_homogeneous(host)), what)[:n]
    log[:, :n] += np.eye(n)
    return _like(log, matrix)


def _squaring_steps(velocity: ArrayProtocol) -> int:
    # The smallest number of squarings for which the first step moves no
    # point by more than `_FIRST_STEP` voxels.
    backend = get_array_backend(velocity)
    norm = float(backend.max(backend.sqrt((velocity**2).sum(axis=-1))))
    if not math.isfinite(norm):
        raise DomainError(
            "The exponential of this velocity field is not defined: it is "
            "not finite."
        )
    if norm <= _FIRST_STEP:
        return 0
    steps = int(math.ceil(math.log2(norm / _FIRST_STEP)))
    if steps > _MAX_STEPS:
        raise DomainError(
            "The exponential of this velocity field is not computed: its "
            f"largest displacement, {norm} voxels, would need more than "
            f"{_MAX_STEPS} squaring steps."
        )
    return steps


def _exp_velocity(
    velocity: DisplacementField,
    steps: tx.Optional[int] = None,
    *,
    skip: int = 0,
    negate: bool = False,
) -> tx.Optional[ArrayProtocol]:
    """Integrate a stationary velocity field by scaling and squaring.

    Returns the displacement field of the flow at time one of `velocity`
    (or of its negation), as an array in the velocity's encoding (values
    or spline coefficients). With `skip`, that many of the last squarings
    are left out, which integrates `velocity / 2**skip`.
    """
    field = velocity.field
    if field is None:
        return None
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
    ):
        raise ValueError(
            f"The number of squaring steps must be a non-negative integer, "
            f"not {steps!r}."
        )
    order, bound, coeff = velocity.order, velocity.bound, velocity.coeff
    if coeff:
        field = coeff2value_field(field, order=order, bound=bound)
    if negate:
        field = -field
    if steps is None:
        steps = _squaring_steps(field)
    steps = max(steps, skip)
    # Scaling: a small enough velocity is its own flow, to first order.
    step = DisplacementField(
        field=field * (0.5**steps), order=order, bound=bound, coeff=False
    )
    # Squaring: the flow at time `2 s` is the flow at time `s`, twice.
    for _ in range(steps - skip):
        step = compose(step, step)
    result = step.field
    if coeff:
        result = value2coeff_field(result, order=order, bound=bound)
    return result
