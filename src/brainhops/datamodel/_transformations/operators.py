"""The unary operators of a transformation, and their lazy results.

A transformation `T` that maps a space to itself has a square and a
principal square root, alongside the inverse that [`Inverse`][] already
represents:

* `T.inverse()` is the map `U` such that `U @ T` is the identity;
* `T.square()` is `T @ T`;
* `T.sqrt()` is the principal `S` such that `S @ S` is `T`, the half-map.

The square and the square root are functions of the *map*: they do not
depend on how `T` is stored. A transformation stored as its tangent about
the identity (`log=True`, such as a [`StationaryVelocityField`][] or an
[`AffineExponential`][]) takes them exactly, in the tangent: its square
doubles the tangent and its square root halves it, so neither needs a
wrapper.

Laziness
--------
As with [`Inverse`][], the square root of any other transformation is a
wrapper that holds the transformation it acts on as its `forward`, and
computes its parameter only when it is read, applied, converted or
computed. A typed wrapper is an instance of the family its result belongs
to -- [`Sqrt`][] of a [`Rotation`][] is a `Rotation` -- so composition,
conversion and the kind checks treat it like any transformation of that
family, and the kind checks and the simplifiers reason from its `forward`
rather than materialize it. Square is the exception: `T.square()` is the
sequence `[T, T]`, which is lazy already.

The front door [`Sqrt`][] builds only the typed wrappers registered below;
anything else is refused when it is constructed. The method `T.sqrt()` is
the general entry point: it also handles the transformations that need no
wrapper (an identity, a tangent), the containers (a [`Sequence`][], a
subspace transformation) and the refusals.
"""

__all__ = [
    "Operation",
    "Sqrt",
    "UNARY_OPERATORS",
]

# stdlib
from functools import wraps
from numbers import Real
from operator import methodcaller
from types import MappingProxyType

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.properties import smartproperty
from brainhops._core.typing import (
    ArrayProtocol,
    Deactivated,
    Derived,
    npmatrix,
    npvector,
)

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel.systems import CoordinateSystem

# internals
from .base import Transformation
from .concrete import (
    Affine,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from .matfuncs import affine_sqrtm, require_positive, sqrtm
from .modes import ModeLike, mode_admits, normalize_modes
from .registries import OPERATION_CACHE, register_sqrt
from .simplify import SimplifyLike
from .simplify import simplify as _simplify
from .utils import with_endpoints

# ======================================================================
#
#                              B A S E
#
# ======================================================================


class Operation(Transformation):
    """An operator applied to a transformation, resolved on demand.

    The base of [`Sqrt`][]. It holds the transformation the operator acts
    on as its `forward`. Every operator here maps a space to itself, so the
    wrapper's endpoints are the forward's. The result is computed only when
    the wrapper is applied, computed, or converted to another type.
    """

    # --- class attributes ---------------------------------------------

    # The plain type a typed wrapper materializes into; unset on the front
    # door, which only ever builds a typed wrapper.
    _result: tx.ClassVar[tx.Optional[tx.Type[Transformation]]] = None

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
            # forward -- keeps the operator unresolved, and the forward
            # keeps its cache.
            if {"input", "output", "forward"}.issuperset(kwargs):
                return replace(self, **kwargs) if kwargs else self
            cls = None
        # Anything else edits the result, which is materialized first.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    # --- helpers ------------------------------------------------------

    def _materialize(self) -> Transformation:
        # A plain instance of the result type, holding the derived
        # parameter and the wrapper's endpoints.
        cls = type(self)._result
        return cls(data=self.data, input=self.input, output=self.output)


def _cached(func: tx.Callable) -> property:
    """A parameter of an operator, derived on demand and cached.

    As for an inverse (see `registries.INVERSE_CACHE`), the value is cached
    on the *forward* transform, under `registries.OPERATION_CACHE`, so a
    wrapper rebuilt by `replace` or `.to(...)` does not compute it twice.
    The key names the wrapper class and the parameter, so two operators of
    one forward never collide. The forward clears it when its `data` or a
    flag is assigned.
    """
    name = func.__name__

    @property
    @wraps(func)
    def parameter(self: Operation) -> tx.Any:
        forward = self.forward
        if forward is None:
            return None
        cache = getattr(forward, OPERATION_CACHE, None)
        if cache is None:
            cache = {}
            setattr(forward, OPERATION_CACHE, cache)
        key = (type(self), name)
        if key not in cache:
            cache[key] = func(self)
        return cache[key]

    return parameter


def _computed(
    t: Transformation, compute: bool, kwargs: tx.Dict[str, tx.Any]
) -> Transformation:
    return t.compute(**kwargs) if compute else t


def _sqrt_of(op: Operation) -> str:
    # "The square root of this Rotation", for error messages.
    return f"The square root of this {type(op.forward).__name__}"


@register_sqrt
class Sqrt(Operation, polymorphic="strict"):
    """The principal square root of a transformation, resolved on demand.

    The square root `S` of `T` is the transformation with `S @ S == T` --
    the half-transformation. Among the many square roots a map may have,
    the principal one is the one whose linear part has its eigenvalues in
    the open right half-plane. It exists, and is real, when the linear part
    of `T` has no eigenvalue on the closed negative real axis; otherwise the
    wrapper raises [`DomainError`][] when it is resolved. A field has a
    square root here only when it is stored as its velocity (a
    [`StationaryVelocityField`][]), whose square root halves the velocity
    and needs no wrapper.

    Build it with [`Transformation.sqrt`][]. Constructing `Sqrt(forward=t)`
    builds the typed wrapper of `t`'s family, and refuses a family that has
    none.
    """

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation: `sqrt(T)` squared is `T`."""
        return _computed(with_endpoints(self.forward, self), compute, kwargs)


# ======================================================================
#
#                  T Y P E D   W R A P P E R S
#
# ======================================================================

# Each typed square root derives its `data` from its forward transform, and
# reads its views (`translation`, `matrix`, ...) off that `data` exactly as
# a forward transform does. The convenience keyword its family takes
# (`translation=`, `matrix=`, ...) is deactivated: a wrapper is built from
# its forward only.


class SqrtTranslation(
    Sqrt,
    Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """The square root of a [`Translation`][]: half the translation."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Translation
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "translation")

    forward: tx.Annotated[
        tx.Optional[Translation],
        tx.Doc("The translation whose square root this represents."),
    ] = None

    data: Derived[tx.Optional[npvector[Real]]]
    _translation: Deactivated[None]

    @_cached
    def data(self) -> tx.Optional[ArrayProtocol]:
        return self.forward.translation / 2


class SqrtScaling(
    Sqrt,
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The square root of a [`Scaling`][], for positive scales only."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Scaling
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "scale")

    forward: tx.Annotated[
        tx.Optional[Scaling],
        tx.Doc("The scaling whose square root this represents."),
    ] = None

    data: Derived[tx.Optional[npvector[Real]]]
    _scale: Deactivated[None]

    @_cached
    def data(self) -> tx.Optional[ArrayProtocol]:
        scale = self.forward.scale
        require_positive(scale, _sqrt_of(self))
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
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "matrix")

    forward: tx.Annotated[
        tx.Optional[Rotation],
        tx.Doc("The rotation whose square root this represents."),
    ] = None

    data: Derived[tx.Optional[npmatrix[Real]]]
    _matrix: Deactivated[None]

    @_cached
    def data(self) -> tx.Optional[ArrayProtocol]:
        return sqrtm(self.forward.matrix, _sqrt_of(self))


class SqrtLinear(
    Sqrt,
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, (Linear, Permutation))},
):
    """The principal square root of a [`Linear`][] or [`Permutation`][]."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Linear
    derived_fields: tx.ClassVar[tx.Tuple[str]] = ("data", "matrix")

    forward: tx.Annotated[
        tx.Optional[tx.Union[Linear, Permutation]],
        tx.Doc("The linear transformation whose square root this is."),
    ] = None

    data: Derived[tx.Optional[npmatrix[Real]]]
    _matrix: Deactivated[None]

    @_cached
    def data(self) -> tx.Optional[ArrayProtocol]:
        forward = self.forward
        if not isinstance(forward, Linear):
            forward = forward.to(Linear)
        return sqrtm(forward.matrix, _sqrt_of(self))


class SqrtAffine(
    Sqrt,
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The principal square root of an [`Affine`][] transformation."""

    _result: tx.ClassVar[tx.Type[Transformation]] = Affine
    derived_fields: tx.ClassVar[tx.Tuple[str]] = (
        "data",
        "matrix",
        "homogeneous_matrix",
    )

    forward: tx.Annotated[
        tx.Optional[Affine],
        tx.Doc("The affine transformation whose square root this is."),
    ] = None

    data: Derived[tx.Optional[npmatrix[Real]]]
    _matrix: Deactivated[None]

    @_cached
    def data(self) -> tx.Optional[ArrayProtocol]:
        return affine_sqrtm(self.forward.matrix, _sqrt_of(self))


# ======================================================================
#
#                          O P E R A T O R S
#
# ======================================================================

UNARY_OPERATORS: tx.Mapping[
    str, tx.Callable[[Transformation], Transformation]
] = MappingProxyType(
    {name: methodcaller(name) for name in ("inverse", "square", "sqrt")}
)
"""
The unary operators of a transformation, by name.

Each value takes a transformation and returns the result of the method of
the same name, with its defaults: a lazy result where there is one. It is
the table an expression parser looks names up in, so that the operators it
accepts are exactly the ones a transformation implements.
"""
