"""Unary operators on transformations and their lazy results.

A transformation `T` that maps a space onto itself has an inverse `U` (`U @ T`
is the identity), a square `T @ T` and a principal square root `S`
(`S @ S == T`, half of the map). A transformation stored in its tangent space,
such as a
[`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField]
or an
[`AffineExponential`][brainhops.datamodel.transformations.AffineExponential],
takes the square and the square root exactly in that space, by doubling or
halving its tangent.

Elsewhere, inversion and the square root are lazy. An
[`Inverse`][brainhops.datamodel.transformations.Inverse] or a [`Sqrt`][] holds
`T` as its `forward` and computes its own parameter only when the parameter is
read or the wrapper is applied, converted or computed. Each typed wrapper is an
instance of its result family (the square root of a [`Rotation`][] is a
[`Rotation`][]), so composition, conversion and simplification treat it like
any other member of that family. The square needs no wrapper, because
`T.square()` is the [`Sequence`][brainhops.datamodel.transformations.Sequence]
`[T, T]`, which is already lazy.
"""

__all__ = [
    "Operation",
    "Sqrt",
    "UNARY_OPERATORS",
]

from numbers import Real
from operator import methodcaller
from types import MappingProxyType

import typing_extensions as tx
from bagof.magic import NotKwOnly, fields

from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.properties import smartproperty
from brainhops._core.typing import Derived, npmatrix, npvector

from . import nocycles
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
    """A unary operator applied to a transformation and resolved on demand.

    An operation holds its target as `forward` and derives its own parameter
    only when the parameter is needed. It is the base class of
    [`Inverse`][brainhops.datamodel.transformations.Inverse] and [`Sqrt`][]. A
    subclass declares the name of the operator, the family of its result, and
    whether the operator reverses the direction of the map.
    """

    _operator: tx.ClassVar[tx.Optional[str]] = None
    """Name of the [`Transformation`][] method that applies the operator.

    The front door passes the forward to this method, which picks the typed
    wrapper for the family of the forward.
    """

    _resultof: _TypeReference = None
    """Family of the result, set by each typed subclass.

    `_materialize` builds an instance of this family, carrying over the
    parameters named by its `data_fields`. The front door leaves it as `None`.
    """

    _reverses: tx.ClassVar[bool] = False
    """Whether the operator maps the output of the forward back to its input.

    Only inversion does so; its endpoints are those of the forward, swapped.
    """

    data_fields: _FieldNames = ("forward",)

    forward: NotKwOnly[tx.Optional[Transformation]] = None
    """Transformation that the operator acts on."""

    # Endpoints are inferred from the forward, swapped if the operator reverses
    # direction; an endpoint declared on the wrapper takes precedence.

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

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> Transformation:
        """Resolve the operator if the mode admits it.

        Resolution happens only here and never in the simplifier, which applies
        only rewrites that cost nothing. When `mode` does not admit the
        operator, the wrapper stays lazy so that an adjacent pair in a sequence
        can still cancel, and it is only simplified according to `simplify`.
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
        """Convert the wrapper to another type or encoding.

        Edits that a lazy wrapper can absorb keep it lazy (see
        [`Transformation.to`][brainhops.datamodel.transformations.Transformation.to]
        for the general behaviour):

        - The endpoints and the forward are edited on the wrapper itself.
        - Any other keyword edits the forward, and the operator stays lazy
          around the result. Because the new encoding may change the class of
          the forward, the wrapper is chosen again, with the endpoints declared
          on this one.
        - The map itself cannot be set: passing a derived field of the family
          raises a `TypeError` that points to editing the forward instead.
        - A target class `cls` resolves the operator before converting the
          result.
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
        # Converting to any other type, even the family's own, resolves the
        # operator first.
        return self._materialize().to(cls, lossy=lossy, **kwargs)

    def _undo(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward with the endpoints of the wrapper restored.

        The inverse of an inverse, and the square of a square root, is the
        wrapped transformation. Endpoints declared on the wrapper are carried
        onto the forward (swapped if the operator reverses direction). Without
        such endpoints, the forward itself is returned, so that adjacent
        wrappers cancel by identity.
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
        """Apply the operator concretely.

        A typed wrapper builds its result family from the derived parameter and
        its own endpoints. The front door delegates to the operator method of
        the forward, which picks a typed wrapper.
        """
        forward = self.forward
        family = type(self)._resultof
        if forward is None:
            # An unset transformation is the identity, which is its own inverse
            # and its own square root.
            return (family or Identity)(input=self.input, output=self.output)
        if family is None:
            return self._resolve_through_forward()
        changes = {
            "input": self.input,
            "output": self.output,
            **{name: getattr(self, name) for name in family.data_fields},
        }
        if not self._reverses:
            # Build the plain family type: the root of a `VoxelToLPS` maps
            # voxels to no space that the type names.
            return family(**changes)
        # Rebuilding from the forward keeps its extra state, such as the flags
        # of a field. The rebuild needs a resolved forward, so a nested wrapper
        # is resolved first.
        if isinstance(forward, Operation):
            forward = forward._materialize()
        # A type whose name states a direction inverts to the other half of its
        # pair.
        reverseof = getattr(type(forward), "_reverseof", None)
        if reverseof is not None:
            return reverseof.from_instance(forward, **changes)
        return forward.to(**changes)

    def _resolve_through_forward(self) -> Transformation:
        # The forward picks the typed wrapper; carry the endpoints over.
        resolved = getattr(self.forward, type(self)._operator)()
        edits = {}
        if self.input:
            edits["input"] = self.input
        if self.output:
            edits["output"] = self.output
        return resolved.to(**edits) if edits else resolved


@register_operator("sqrt")
class Sqrt(Operation, polymorphic="strict"):
    """Lazy principal square root of a transformation.

    The square root `S` of `T` satisfies `S @ S == T`. The principal root is
    the one whose linear part has all its eigenvalues in the open right
    half-plane. It exists when the linear part of `T` has no eigenvalue on the
    closed negative real axis; otherwise, resolving the root raises a
    [`DomainError`][brainhops.errors.DomainError]. A field has a square root
    only when it is stored as a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.StationaryVelocityField],
    which halves its velocity without a wrapper.

    Square roots are normally built with
    [`Transformation.sqrt`][brainhops.datamodel.transformations.Transformation.sqrt].
    `Sqrt(forward=t)` builds the typed wrapper for the family of `t` and
    refuses a family that has none.
    """

    _operator: tx.ClassVar[str] = "sqrt"

    def square(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward, since the square of `sqrt(T)` is `T`.

        Endpoints edited on the wrapper are carried onto the forward. With
        `compute`, the other keywords are passed to `compute()`.
        """
        return self._undo(compute, **kwargs)


# ======================================================================
#
#                  T Y P E D   W R A P P E R S
#
# ======================================================================


class ConcreteSqrtMixin:
    """Mixin that reads the parameter of a typed square root off its forward.

    Each concrete family derives the parameter of its square root under `_sqrt`
    and caches it, so the wrapper holds no array of its own. The forward clears
    that cache whenever its data or a flag is assigned. Because the wrapper is
    built from its forward only, the convenience keywords of the family, such
    as `matrix=`, are not accepted.
    """

    data = _alias("data", "forward._sqrt", fset=False)


class SqrtTranslation(
    ConcreteSqrtMixin,
    Sqrt,
    Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """Square root of a [`Translation`][], which is half the translation."""

    _resultof: _TypeReference = Translation

    derived_fields: _FieldNames = ("data", "translation")

    forward: NotKwOnly[tx.Optional[Translation]] = None
    """Translation whose square root this is."""

    _data: Derived[_OptionalVector]
    _translation: Derived[_OptionalVector]


class SqrtScaling(
    ConcreteSqrtMixin,
    Sqrt,
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """Square root of a [`Scaling`][], defined for positive scales."""

    _resultof: _TypeReference = Scaling

    derived_fields: _FieldNames = ("data", "scale")

    forward: NotKwOnly[tx.Optional[Scaling]] = None
    """Scaling whose square root this is."""

    _data: Derived[_OptionalVector]
    _scale: Derived[_OptionalVector]


class SqrtRotation(
    ConcreteSqrtMixin,
    Sqrt,
    Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # `SqrtLinear` also matches a rotation; the priority keeps the root a
    # rotation.
    priority=1,
):
    """Principal square root of a [`Rotation`][], which is half the rotation.

    The root is undefined for a half turn in any plane, which has two square
    roots and no principal one.
    """

    _resultof: _TypeReference = Rotation

    derived_fields: _FieldNames = ("data", "matrix")

    forward: NotKwOnly[tx.Optional[Rotation]] = None
    """Rotation whose square root this is."""

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]


class SqrtLinear(
    ConcreteSqrtMixin,
    Sqrt,
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, Linear)},
):
    """Principal square root of a [`Linear`][] transformation."""

    _resultof: _TypeReference = Linear

    derived_fields: _FieldNames = ("data", "matrix")

    forward: NotKwOnly[tx.Optional[Linear]] = None
    """Linear transformation whose square root this is."""

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]


class SqrtAffine(
    ConcreteSqrtMixin,
    Sqrt,
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """Principal square root of an [`Affine`][] transformation."""

    _resultof: _TypeReference = Affine

    derived_fields: _FieldNames = ("data", "matrix", "homogeneous_matrix")

    forward: NotKwOnly[tx.Optional[Affine]] = None
    """Affine transformation whose square root this is."""

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
"""Unary operators by name: `"inverse"`, `"square"` and `"sqrt"`.

Each callable calls the method of the same name with its defaults. The
expression parser looks names up in this table, so it accepts exactly the
operators that a transformation implements.
"""
