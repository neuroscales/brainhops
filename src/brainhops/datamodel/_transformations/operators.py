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
The inverse and the square root of a transformation are both wrappers
that hold the transformation they act on as their `forward` and compute
their parameter only when it is read, applied, converted or computed.
[`Operation`][] is what the two share, and [`Inverse`][] (in `inverse`)
and [`Sqrt`][] are its two subclasses: an inversion is the one that
*reverses* the direction, which is the only thing the base has to be told
(`_reverses`). A typed wrapper is an instance of the family its result
belongs to -- [`Sqrt`][] of a [`Rotation`][] is a `Rotation` -- so
composition, conversion and the kind checks treat it like any
transformation of that family, and the kind checks and the simplifiers
reason from its `forward` rather than materialize it. Square is the
exception: `T.square()` is the sequence `[T, T]`, which is lazy already.

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
from numbers import Real
from operator import methodcaller
from types import MappingProxyType

# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly, fields

# core
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.properties import smartproperty
from brainhops._core.typing import Derived, npmatrix, npvector

from . import nocycles

# internals
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .concrete import (
    Affine,
    Identity,
    Linear,
    Rotation,
    Scaling,
    Translation,
    _alias,
)
from .modes import ModeLike, mode_admits, normalize_modes
from .nocycles import register_operator

# typing
if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem

_TypeReference = tx.ClassVar[tx.Optional[tx.Type[Transformation]]]
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]
_OptionalMatrix: tx.TypeAlias = tx.Optional[npmatrix[Real]]
_OptionalVector: tx.TypeAlias = tx.Optional[npvector[Real]]


# ======================================================================
#
#                              B A S E
#
# ======================================================================


class Operation(Transformation):
    """A unary operator applied to a transformation, resolved on demand.

    The base of [`Inverse`][] and [`Sqrt`][]. It holds the transformation
    the operator acts on as its `forward`, and derives its own parameter
    from it only when that parameter is read, applied, converted or
    computed.

    A subclass says three things about its operator: the name of the
    `Transformation` method that applies it (`_operator`), the family its
    result belongs to (`_resultof`), and whether it reverses the direction
    the forward maps (`_reverses`). Everything else below follows from
    those.
    """

    # --- class attributes ---------------------------------------------

    _operator: tx.ClassVar[tx.Optional[str]] = None
    """
    The name of the `Transformation` method that applies this operator:
    `"inverse"`, `"sqrt"`. The front door derives no parameter of its
    own, so it hands the forward to that method, which picks the typed
    wrapper of the forward's own family.
    """

    _resultof: _TypeReference = None
    """
    The family the result belongs to, which the typed subclass declares:
    `SqrtRotation._resultof` is `Rotation`, and `InverseAffine._resultof`
    is `Affine`. It is the type `_materialize` builds, and the one whose
    `data_fields` name the parameter to carry over.

    `None` on a front door, which only ever builds a typed wrapper.
    """

    _reverses: tx.ClassVar[bool] = False
    """
    Whether the operator maps the forward's output back to its input.

    An inversion does, so its endpoints are the forward's swapped, and
    what it resolves to keeps the forward's own class -- or swaps to the
    other half of a pair that names a direction (`_reverseof`). Every
    other operator here maps a space to itself.
    """

    data_fields: _FieldNames = ("forward",)

    # --- attributes ---------------------------------------------------

    forward: NotKwOnly[tx.Optional[Transformation]] = None
    "The transformation the operator acts on."

    # --- properties ---------------------------------------------------

    # The wrapper knows its own endpoints without being told: they are the
    # forward's, swapped when the operator reverses the direction. An
    # endpoint declared on the wrapper still wins -- that is what
    # `smartproperty` does -- so an explicit override is honoured.

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        forward = self.forward
        if forward is None:
            return None
        return forward.output if self._reverses else forward.input

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        forward = self.forward
        if forward is None:
            return None
        return forward.input if self._reverses else forward.output

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Resolve the operator, if the mode admits it.

        Resolving is the one thing a lazy wrapper exists to put off, so it
        happens here and nowhere else -- never in a simplifier, which may
        only rewrite for free. The compose `mode` is the gate: a mode that
        does not admit this wrapper leaves it lazy, so an adjacent pair
        can still cancel in a sequence, and `simplify` then still
        simplifies what it wraps. `factor` is forwarded to the result.
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
        """Convert this wrapper to a different type or encoding.

        See [`Transformation.to`][]. An edit a lazy wrapper can absorb
        leaves it lazy, and only a conversion to another type resolves it:

        * the wrapper's own fields -- its endpoints, its forward -- are
          edited on the wrapper, which keeps the forward and its cache;
        * anything else is an edit of the *forward*, and the operator
          stays lazy around the edited one: the square root of a
          re-encoded field is the re-encoded square root, and likewise
          for an inverse. A new encoding can change the class of the
          forward (`log=True` builds a tangent subclass), so the wrapper
          is chosen again for it, with the endpoints this one declares;
        * the map itself cannot be set: the wrapper derives it, so what
          its family lists in `derived_fields` (`data`, and the views
          that spell it) is refused and points at the forward;
        * `cls` is a conversion, which resolves the operator first.
        """
        if cls is None or cls is type(self):
            derived = [k for k in kwargs if k in type(self).derived_fields]
            if derived:
                raise TypeError(
                    f"{type(self).__name__}.to() got {derived[0]}=, but "
                    f"the map of a lazy {type(self)._operator} is derived "
                    f"from its forward; change the forward instead: "
                    f"t.forward.to({derived[0]}=...)."
                )
            own = {f.public_name for f in fields(type(self)) if f.init}
            own.add("error")
            others = {k: kwargs.pop(k) for k in list(kwargs) if k not in own}
            if others:
                front = nocycles.OPERATORS[type(self)._operator]
                obj = front(
                    forward=self.forward.to(**others),
                    input=self._input,
                    output=self._output,
                )
                return obj.to(**kwargs) if kwargs else obj
            return super().to(cls, **kwargs)
        # A conversion to another type, including the family's own,
        # resolves the operator first, then converts onward.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    # --- helpers ------------------------------------------------------

    def _undo(self, compute: bool = False, **kwargs) -> Transformation:
        """
        The forward transformation, with the endpoints restored.

        Both operators here are involutions on the wrapper: the inverse of
        an inverse, and the square of a square root, are the
        transformation wrapped. An endpoint declared on the wrapper is
        carried onto the forward -- swapped when the operator reverses the
        direction -- and the forward is handed back as the *same object*
        when the wrapper declares none, so a wrapper placed next to what
        it undoes still cancels by identity.
        """
        forward = self.forward
        if forward is None:
            return Identity(input=self.input, output=self.output)
        first, second = (
            ("output", "input") if self._reverses else ("input", "output")
        )
        edits = {}
        if self._input is not None:
            edits[first] = self._input
        if self._output is not None:
            edits[second] = self._output
        obj = forward.to(**edits) if edits else forward
        return obj.compute(**kwargs) if compute else obj

    def _materialize(self) -> Transformation:
        """
        The operator applied, concretely.

        A typed wrapper builds an instance of the family its result
        belongs to, holding the parameter it derives and its own
        endpoints. A front door derives nothing of its own: it hands the
        forward to the method that applies the operator, which picks the
        typed wrapper of the forward's family, so `compute` resolves in
        two hops rather than one.
        """
        forward = self.forward
        family = type(self)._resultof
        if forward is None:
            # Nothing to operate on: an unset transformation is the
            # identity, which is its own inverse and its own square root.
            return (family or Identity)(input=self.input, output=self.output)
        if family is None:
            return self._resolve_through_forward()
        changes = {
            "input": self.input,
            "output": self.output,
            **{name: getattr(self, name) for name in family.data_fields},
        }
        if not self._reverses:
            # A new map of the family, built as the plain type: the square
            # root of a `VoxelToLPS` is half of it, and maps voxels to no
            # space that type names.
            return family(**changes)
        # An inversion stays in the direction business, so it keeps
        # whatever the forward holds besides its parameter -- the flags of
        # a field, the `image` and `header` of a reader -- by rebuilding
        # from the forward.
        #
        # `changes` was read off this wrapper, which derives its parameter
        # without resolving what it wraps; the rebuild cannot be, because
        # a lazy forward does not *hold* anything yet. So a wrapper around
        # a wrapper resolves the inner one first -- which is what
        # `_materialize` is for, and both are being resolved anyway.
        if isinstance(forward, Operation):
            forward = forward._materialize()
        # A type whose *name* states the direction it maps does not invert
        # to itself: the other half of its pair states the swapped
        # direction right.
        reverseof = getattr(type(forward), "_reverseof", None)
        if reverseof is not None:
            return reverseof.from_instance(forward, **changes)
        return forward.to(**changes)

    def _resolve_through_forward(self) -> Transformation:
        # The front door's path: the forward's own operator method picks
        # the typed wrapper for its family, and the endpoints this wrapper
        # declares are carried onto it.
        resolved = getattr(self.forward, type(self)._operator)()
        edits = {}
        if self.input:
            edits["input"] = self.input
        if self.output:
            edits["output"] = self.output
        return resolved.to(**edits) if edits else resolved


@register_operator("sqrt")
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

    _operator: tx.ClassVar[str] = "sqrt"

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation: `sqrt(T)` squared is `T`.

        An endpoint edit made on the wrapper is carried onto it. With
        `compute`, the other keywords are passed on to `compute()`.
        """
        return self._undo(compute, **kwargs)


# ======================================================================
#
#                  T Y P E D   W R A P P E R S
#
# ======================================================================


class ConcreteSqrtMixin:
    """The parameter of a typed square root, read off its forward.

    Every concrete family derives the parameter of its own square root
    under `_sqrt` -- halved, square-rooted, `sqrtm` -- and caches it
    there, in its own encoding. A wrapper holds no array: it reports that
    one. The forward clears the cache when its `data` or a flag is
    assigned, so a rebuilt wrapper never recomputes a square root and an
    edited forward never serves a stale one.

    The views of the family (`translation`, `matrix`, ...) read that
    `data` exactly as a forward transform does, and the convenience
    keyword it takes (`translation=`, `matrix=`, ...) has no way in: a
    wrapper is built from its forward only.
    """

    data = _alias("data", "forward._sqrt", fset=False)


class SqrtTranslation(
    ConcreteSqrtMixin, Sqrt, Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """The square root of a [`Translation`][]: half the translation."""

    _resultof: _TypeReference = Translation

    derived_fields: _FieldNames = ("data", "translation")

    forward: NotKwOnly[tx.Optional[Translation]] = None
    "The translation whose square root this represents."

    _data: Derived[_OptionalVector]
    _translation: Derived[_OptionalVector]


class SqrtScaling(
    ConcreteSqrtMixin, Sqrt, Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """The square root of a [`Scaling`][], for positive scales only."""

    _resultof: _TypeReference = Scaling

    derived_fields: _FieldNames = ("data", "scale")

    forward: NotKwOnly[tx.Optional[Scaling]] = None
    "The scaling whose square root this represents."

    _data: Derived[_OptionalVector]
    _scale: Derived[_OptionalVector]


class SqrtRotation(
    ConcreteSqrtMixin, Sqrt, Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # A `Rotation` is a `Linear`, so `SqrtLinear` matches it too. The
    # square root of a rotation is a rotation, which this wrapper keeps.
    priority=1,
):
    """The principal square root of a [`Rotation`][]: half the rotation.

    It is not defined for a rotation by a half turn in some plane, which
    has two square roots and no principal one.
    """

    _resultof: _TypeReference = Rotation

    derived_fields: _FieldNames = ("data", "matrix")

    forward: NotKwOnly[tx.Optional[Rotation]] = None
    "The rotation whose square root this represents."

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]


class SqrtLinear(
    ConcreteSqrtMixin, Sqrt, Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, Linear)},
):
    """The principal square root of a [`Linear`][] transformation."""

    _resultof: _TypeReference = Linear

    derived_fields: _FieldNames = ("data", "matrix")

    forward: NotKwOnly[tx.Optional[Linear]] = None
    "The linear transformation whose square root this is."

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]


class SqrtAffine(
    ConcreteSqrtMixin, Sqrt, Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """The principal square root of an [`Affine`][] transformation."""

    _resultof: _TypeReference = Affine

    derived_fields: _FieldNames = ("data", "matrix", "homogeneous_matrix")

    forward: NotKwOnly[tx.Optional[Affine]] = None
    "The affine transformation whose square root this is."

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]


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
