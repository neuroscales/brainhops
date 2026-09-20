# stdlib
from numbers import Integral

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.properties import smartproperty
from brainhops._core.typing import npvector

# datamodel
from brainhops.datamodel import hierarchy
from brainhops.datamodel.axes import Axis

# internals
from . import registries
from .base import Transformation
from .modes import (
    _NONINVERTIBLE_OF,
    ModeLike,
    SimplifyLike,
    SimplifyPolicy,
    _lift_targets,
    _lower_modes,
    _lower_simplify,
    _mode_admits,
    _permute_targets,
    _reindex_is_even,
    _resolve_simplify,
    is_member,
    register_kind,
)

# typing
if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem


class MetaTransformation(Transformation):
    """
    A transformation that is defined in terms of another transformation.

    This is a base class for transformations that are defined in terms of
    other transformations, such as `SubspaceTransformation` and
    `Bijection`. It is not meant to be instantiated directly.
    """

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        # A bare meta transformation has nothing to simplify on its own, so
        # -- once mode-gated -- it is returned unchanged. Subclasses
        # (`SubspaceTransformation`, `Projection`, `Bijection`) override this.
        if factor:
            return registries.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        if mode is not None and not _mode_admits(self, _lower_modes(mode)):
            return self
        return self


class SubspaceTransformation(MetaTransformation):
    """
    A transformation that is applied to a subset of the input and output axes.

    The transformation acts on the axes named by `input_axes` and
    `output_axes`, and leaves every other axis unchanged. The
    dimensionality of the space is preserved. An axis that is not named
    passes through as the identity. This lifts a transformation defined
    over a few axes, such as a spatial transformation over `(x, y, z)`,
    into a larger space, such as `(x, y, z, t)`, where it acts on the
    spatial axes and leaves time untouched.
    """

    parameter_names: tx.ClassVar[str] = "transformation"

    # --- attributes ---------------------------------------------------

    transformation: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The transformation to apply.")
    ] = None

    input_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the input coordinate system to transform."),
    ] = None

    output_axes: tx.Annotated[
        tx.Optional[npvector[Integral]],
        tx.Doc("The axes of the output coordinate system to transform."),
    ] = None

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "input", None)
        return _subsystem(system, self.input_axes, full=self._input)

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        system = getattr(self.transformation, "output", None)
        return _subsystem(system, self.output_axes, full=self._output)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        if self.transformation is None:
            return replace(
                self,
                input=self._output,
                output=self._input,
                input_axes=self.output_axes,
                output_axes=self.input_axes,
            )
        kwargs.setdefault("compute", compute)
        return replace(
            self,
            transformation=self.transformation.inverse(**kwargs),
            input=self._output,
            output=self._input,
            input_axes=self.output_axes,
            output_axes=self.input_axes,
        )

    def _is_member(self, node: type, policy: SimplifyPolicy) -> bool:
        # A subspace is `P ∘ blockdiag(inner, I)`: a lift of `inner` into the
        # full space, optionally composed with a coordinate permutation `P`
        # when it reindexes axes. Its membership in `node` follows from the
        # inner's membership in the lift (and permutation) targets of `node`.
        inner = self.transformation
        same = (self.input_axes is None and self.output_axes is None) or (
            self.input_axes is not None
            and self.output_axes is not None
            and list(self.input_axes) == list(self.output_axes)
        )
        if same:
            return any(
                _inner_member(inner, m, policy) for m in _lift_targets(node)
            )
        even = _reindex_is_even(self.input_axes, self.output_axes)
        return any(
            _inner_member(inner, m, policy)
            for target in _permute_targets(node, even)
            for m in _lift_targets(target)
        )

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        from .concrete import Identity

        if factor:
            return registries.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        if mode is not None and not _mode_admits(self, _lower_modes(mode)):
            return self
        table = _lower_simplify(simplify)
        policy = _resolve_simplify(self, table)
        if policy is SimplifyPolicy.none:
            # Untouched; the inner is not visited.
            return self
        same = (self.input_axes is None and self.output_axes is None) or (
            self.input_axes is not None
            and self.output_axes is not None
            and list(self.input_axes) == list(self.output_axes)
        )
        inner = self.transformation
        if inner is None:
            # A subspace of nothing: the identity on the same axes, a pure
            # reindex otherwise (kept as is).
            if same:
                return Identity(input=self.input, output=self.output)
            return self
        # A simplify pass may only downcast leaves -- never compose, never
        # materialize -- and it does not carry the outer `mode` into the
        # inner (the mode gate is the run loop's job). `_simplify_inner`
        # enforces this (a lazy inverse stays lazy under analytic; an inner
        # sequence is downcast element-wise, never composed).
        inner2 = _simplify_inner(inner, table)
        if inner2 is inner:
            return self
        if same and isinstance(inner2, hierarchy.IdentityTransformation):
            return Identity(input=self.input, output=self.output)
        return replace(self, transformation=inner2)


class Projection(MetaTransformation):
    """
    A transformation that acts as a projection from a higher dimensional
    space to a lower-dimensional space by removing one or more axes.

    Or its inverse (i.e., an embedding) that adds one or more axes to a
    lower-dimensional space.
    """

    # --- attributes ---------------------------------------------------

    dropped: npvector[Integral] = ()
    created: npvector[Integral] = ()

    # --- methods ------------------------------------------------------

    def inverse(self) -> tx.Self:
        return replace(
            self,
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )

    def _is_member(self, node: type, policy: SimplifyPolicy) -> bool:
        # Establish injectivity/surjectivity/identity from the axis lists.
        # A projection that only drops axes is surjective; one that only
        # creates axes is injective; one that does both establishes nothing
        # beyond `Transformation`. (Establishing `Linear`/`Affine` is left to
        # a follow-up converter.)
        dropped = 0 if self.dropped is None else len(self.dropped)
        created = 0 if self.created is None else len(self.created)
        if not dropped and not created:
            return issubclass(hierarchy.IdentityTransformation, node)
        if dropped and not created:
            return issubclass(hierarchy.SurjectiveTransformation, node)
        if created and not dropped:
            return issubclass(hierarchy.InjectiveTransformation, node)
        return False

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        from .concrete import Identity

        if factor:
            return registries.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        if mode is not None and not _mode_admits(self, _lower_modes(mode)):
            return self
        policy = _resolve_simplify(self, _lower_simplify(simplify))
        dropped = 0 if self.dropped is None else len(self.dropped)
        created = 0 if self.created is None else len(self.created)
        if policy is not SimplifyPolicy.none and not dropped and not created:
            return Identity(input=self.input, output=self.output)
        return self


@hierarchy.BijectiveTransformation.register
class Bijection(Transformation):
    """
    A transformation whose inverse is explicitly defined.
    """

    # --- attributes ---------------------------------------------------

    forward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The forward transformation.")
    ] = None

    backward: tx.Annotated[
        tx.Optional[Transformation], tx.Doc("The backward transformation.")
    ] = None

    @smartproperty
    def input(self) -> tx.Optional["CoordinateSystem"]:
        if self.forward is not None and self.forward.input is not None:
            return self.forward.input
        if self.backward is not None and self.backward.output is not None:
            return self.backward.output
        return None

    @smartproperty
    def output(self) -> tx.Optional["CoordinateSystem"]:
        if self.forward is not None and self.forward.output is not None:
            return self.forward.output
        if self.backward is not None and self.backward.input is not None:
            return self.backward.input
        return None

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        obj = replace(
            self,
            forward=self.backward,
            backward=self.forward,
            input=self._output,
            output=self._input,
        )
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def _is_member(self, node: type, policy: SimplifyPolicy) -> bool:
        # Delegate to the forward map (or the inverse of the backward when no
        # forward is given). A `Bijection` is declared bijective, so a node
        # that only its invertible variant establishes (e.g. `Affine` when
        # the forward is a bare affine) is relaxed to the non-invertible set.
        from .inverse import Inverse

        f = self.forward
        if f is None and self.backward is not None:
            f = Inverse(forward=self.backward)
        if f is None:
            return False
        if is_member(f, node, policy):
            return True
        relaxed = _NONINVERTIBLE_OF.get(node)
        return relaxed is not None and is_member(f, relaxed, policy)

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        if mode is not None and not _mode_admits(self, _lower_modes(mode)):
            return self
        table = _lower_simplify(simplify)
        none = _resolve_simplify(self, table) is SimplifyPolicy.none
        if not factor and none:
            return self
        if factor:
            # A `Bijection` is a container over its two sides; it forwards
            # `factor` to both, each of which factors independently.
            forward, backward = self.forward, self.backward
            if forward is not None:
                forward = forward.compute(mode, simplify=table, factor=True)
            if backward is not None:
                backward = backward.compute(mode, simplify=table, factor=True)
        else:
            # Downcast the forward/backward leaves only -- never compose or
            # materialize (a lazy inverse forward/backward stays lazy under
            # analytic). `_simplify_inner` enforces this.
            forward = _simplify_inner(self.forward, table)
            backward = _simplify_inner(self.backward, table)
        if forward is self.forward and backward is self.backward:
            # Unchanged: keep object identity so an adjacent `Inverse` of
            # this bijection still cancels.
            return self
        return replace(self, forward=forward, backward=backward)


def _simplify_inner(
    inner: tx.Optional[Transformation], table: tx.Any
) -> tx.Optional[Transformation]:
    """Apply a simplify table to a wrapper's inner, downcasting leaves only.

    A simplify pass may never compose or materialize. This routes the inner
    through the sequence leaf-pass, which leaves a lazy `Inverse` lazy under
    an analytic policy (materializing only under numeric, and only when it
    can), downcasts each element of an inner `Sequence` without composing it,
    and downcasts any other inner in place. The outer `mode` is deliberately
    not forwarded -- gating which kinds compose is the run loop's job, not a
    leaf downcast's.
    """
    if inner is None:
        return inner
    from .sequence import _simplify_result

    return _simplify_result(inner, table)


def _inner_member(
    inner: tx.Optional[Transformation], m: type, policy: SimplifyPolicy
) -> bool:
    # Membership of a subspace's inner in node `m`. A `None` inner is the
    # identity (`blockdiag(I, I) = I`), which is a member of `m` exactly when
    # `m` contains the identity node.
    from brainhops.datamodel import hierarchy as _h

    if inner is None:
        return issubclass(_h.IdentityTransformation, m)
    return is_member(inner, m, policy)


def _subsystem(
    system: tx.Optional["CoordinateSystem"] = None,
    index: tx.Optional[tx.Sequence[Integral]] = None,
    full: tx.Optional["CoordinateSystem"] = None,
) -> tx.Optional["CoordinateSystem"]:
    """Build the full-space system a subspace transform presents.

    `system` is the inner transform's own (subspace) coordinate system,
    which names one axis per acted-on dimension. `index` gives the
    positions those axes occupy in the full space.

    When a declared endpoint (`full`) is available, it is returned as is.
    Otherwise a full-space system is reconstructed: the inner system's
    axis `j` is placed at position `index[j]`, and any position no inner
    axis lands on is filled with a placeholder [`Axis`][]. This spans
    `max(index) + 1` axes, so an endpoint-less subspace whose axes exceed
    the inner system's length no longer indexes past its end.
    """
    if full is not None:
        return full
    if index is None:
        return system
    axes = getattr(system, "axes", None)
    if axes is None:
        return system
    index = list(index)
    n = max(index) + 1 if index else 0
    new_axes = [Axis() for _ in range(n)]
    for j, i in enumerate(index):
        if j < len(axes):
            new_axes[i] = axes[j]
    return replace(
        system,
        axes=new_axes,
        name=f"subspace({system.name})" if system.name else None,
    )


# The wrappers are addressed as class kinds (matched by `isinstance`): they
# have no hierarchy node, since their membership depends on their contents.
register_kind("meta", MetaTransformation)
register_kind("subspace", SubspaceTransformation)
register_kind("projection", Projection)
