# stdlib
from numbers import Integral

# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

# core
from brainhops._core.properties import smartproperty
from brainhops._core.typing import npvector

# datamodel
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError

# internals
from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import axis_list, require_endomorphism
from .modes import ModeLike

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)


class MetaTransformation(Transformation):
    """
    A transformation that is defined in terms of another transformation.

    This is a base class for transformations that are defined in terms of
    other transformations, such as `SubspaceTransformation` and
    `Bijection`. It is not meant to be instantiated directly.
    """

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        if factor:
            # A wrapper asked to factor is handed to the sequence engine as a
            # one-element sequence, which runs the factor pass. The engine
            # never passes `factor` back to a leaf's `compute`, so there is
            # no recursion.
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        # A meta transformation holds no parameter of its own to fuse, so
        # computing it is simplifying it: the registered simplifier for its
        # type recurses into what it wraps, gated by `simplify`. `mode`
        # gates which kinds *compose*, and there is nothing here to compose.
        return _simplify(self, policy=simplify)


class SubspaceTransformation(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """
    A transformation that is applied to a subset of the input and output axes.

    The transformation acts on the axes named by `input_axes` and
    `output_axes`, and leaves every other axis unchanged. The
    dimensionality of the space is preserved. An axis that is not named
    passes through as the identity. This embeds a transformation defined
    over a few axes, such as a spatial transformation over `(x, y, z)`,
    into a larger space, such as `(x, y, z, t)`, where it acts on the
    spatial axes and leaves time untouched.

    Generic in the wrapped transformation type:
    `SubspaceTransformation[TRANSFORMATION]` embeds a `TRANSFORMATION`. Its
    membership is decided by a checker registered in `checkers` that recurses
    into the wrapped transform.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = (
        "transformation",
        "input_axes",
        "output_axes",
    )

    # --- attributes ---------------------------------------------------

    transformation: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """The transformation to apply."""

    input_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    "The axes of the input coordinate system to transform."

    output_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """The axes of the output coordinate system to transform."""

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        system = getattr(self.transformation, "input", None)
        return _subsystem(system, self.input_axes, full=self._input)

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        system = getattr(self.transformation, "output", None)
        return _subsystem(system, self.output_axes, full=self._output)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        if self.transformation is None:
            return self.to(
                input=self._output,
                output=self._input,
                input_axes=self.output_axes,
                output_axes=self.input_axes,
            )
        kwargs.setdefault("compute", compute)
        return self.to(
            transformation=self.transformation.inverse(**kwargs),
            input=self._output,
            output=self._input,
            input_axes=self.output_axes,
            output_axes=self.input_axes,
        )

    def sqrt(self, compute: bool = False, **kwargs) -> tx.Self:
        # A subspace transformation that reads and writes the same axes is
        # `blockdiag(inner, I)`. Its principal square root is
        # `blockdiag(sqrt(inner), I)`, so the square root acts on the inner
        # transformation alone and the subspace keeps its axes.
        require_endomorphism(self, "square root")
        if not _same_axes(self):
            raise NotImplementedError(
                "The square root of a subspace transformation that "
                "reindexes its axes is not implemented."
            )
        obj = self
        if self.transformation is not None:
            obj = self.to(transformation=self.transformation.sqrt())
        return obj.compute(**kwargs) if compute else obj


class Projection(MetaTransformation):
    """
    A transformation that acts as a projection from a higher dimensional
    space to a lower-dimensional space by removing one or more axes.

    Or its inverse (i.e., an embedding) that adds one or more axes to a
    lower-dimensional space.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "dropped", "created"

    # --- attributes ---------------------------------------------------

    dropped: NotKwOnly[npvector[Integral]] = ()
    created: NotKwOnly[npvector[Integral]] = ()

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # Swapping what is dropped and what is created is already the
        # exact inverse, so there is nothing to defer and `compute`
        # changes nothing.
        return self.to(
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )


@kinds.Bijection
class Bijection(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """
    A transformation whose inverse is explicitly defined.

    Generic in the forward transformation type: `Bijection[TRANSFORMATION]`
    wraps a `TRANSFORMATION`. Declared bijective; a checker registered in
    `checkers` refines its membership from the forward map.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "forward", "backward"

    # --- attributes ---------------------------------------------------

    forward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """The forward transformation."""

    backward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """The backward transformation."""

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        if self.forward is not None and self.forward.input is not None:
            return self.forward.input
        if self.backward is not None and self.backward.output is not None:
            return self.backward.output
        return None

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        if self.forward is not None and self.forward.output is not None:
            return self.forward.output
        if self.backward is not None and self.backward.input is not None:
            return self.backward.input
        return None

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        if not factor:
            return super().compute(mode, simplify=simplify)
        # A `Bijection` is a container over its two sides; it forwards
        # `factor` to both, each of which factors independently.
        forward, backward = self.forward, self.backward
        if forward is not None:
            forward = forward.compute(mode, simplify=simplify, factor=True)
        if backward is not None:
            backward = backward.compute(mode, simplify=simplify, factor=True)
        if forward is self.forward and backward is self.backward:
            # Unchanged: keep object identity so an adjacent `Inverse` of
            # this bijection still cancels.
            return self
        return self.to(forward=forward, backward=backward)

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        obj = self.to(
            forward=self.backward,
            backward=self.forward,
            input=self._output,
            output=self._input,
        )
        if compute:
            obj = obj.compute(**kwargs)
        return obj

    def sqrt(self, compute: bool = False, **kwargs) -> tx.Self:
        # The principal square root of an inverse is the inverse of the
        # principal square root, so both directions are kept.
        require_endomorphism(self, "square root")
        forward, backward = self.forward, self.backward
        obj = self.to(
            forward=None if forward is None else forward.sqrt(),
            backward=None if backward is None else backward.sqrt(),
        )
        return obj.compute(**kwargs) if compute else obj


def _same_axes(t: SubspaceTransformation) -> bool:
    # Whether a subspace reads and writes the same axes, in the same
    # order -- i.e. whether it embeds its inner transform without also
    # reindexing the coordinates.
    if t.input_axes is None and t.output_axes is None:
        return True
    if t.input_axes is None or t.output_axes is None:
        return False
    return list(t.input_axes) == list(t.output_axes)


def _subsystem(
    system: tx.Optional[CoordinateSystem] = None,
    index: tx.Optional[tx.Sequence[Integral]] = None,
    full: tx.Optional[CoordinateSystem] = None,
) -> tx.Optional[CoordinateSystem]:
    """Build the full-space system a subspace transform presents.

    `system` is the inner transform's own (subspace) coordinate system,
    which names one axis per acted-on dimension. `index` gives the
    positions those axes occupy in the full space.

    When a declared endpoint (`full`) is available, it is returned as is.
    Otherwise the full-space system is derived by
    [`CoordinateSystem.embed`][brainhops.datamodel.systems.CoordinateSystem.embed]:
    the inner system's axis `j` sits at position `index[j]`, every other
    position before the last one holds an unknown [`Axis`][], and the
    system ends with `...`. The positions are known, but the number of
    axes of the full space is not, so the derived system is open. An
    inner system that states no axis has nothing to embed, and is
    returned as is.
    """
    if full is not None:
        return full
    if index is None or system is None:
        return system
    if list(system.axes) == [...]:
        return system
    embedded = system.embed([int(i) for i in index])
    name = f"subspace({system.name})" if system.name else None
    return replace(embedded, name=name)


def _close_subspace(
    t: SubspaceTransformation,
    n_in: tx.Optional[int] = None,
    n_out: tx.Optional[int] = None,
) -> SubspaceTransformation:
    """Give a subspace transform closed full-space systems, when known.

    A subspace transform whose systems are missing or open does not know
    how many axes its full space has. A neighbour in a composition may
    know it: the space between two transforms is one space, so the
    number of axes a neighbour states for it is the number of axes on
    that side of `t`. `n_in` and `n_out` are such counts, for the input
    and the output side of `t`.

    The count on one side gives the count on the other, because a
    subspace transform passes every axis it does not act on through:
    both sides have as many pass-through axes. A count `t` states itself
    is never overridden.

    The open systems are closed by
    [`CoordinateSystem.expand`][brainhops.datamodel.systems.CoordinateSystem.expand]
    and declared on the returned transform. `t` is returned unchanged when
    its systems are already closed, or when no count is known. A count
    that cannot hold the axes `t` acts on, or the axes its systems state,
    is refused with a [`CompositionError`][].
    """
    in_axes = axis_list(t.input_axes)
    out_axes = axis_list(t.output_axes)
    own_in, own_out = (
        get_axes(t.input).ndim,
        get_axes(t.output).ndim,
    )
    n_in = own_in if own_in is not None else n_in
    n_out = own_out if own_out is not None else n_out
    if n_in is None and n_out is not None:
        n_in = n_out - len(out_axes) + len(in_axes)
    if n_out is None and n_in is not None:
        n_out = n_in - len(in_axes) + len(out_axes)
    if n_in is None or n_out is None:
        return t
    for count, axes in ((n_in, in_axes), (n_out, out_axes)):
        if any(not 0 <= a < count for a in axes):
            raise CompositionError(
                f"A subspace transform that acts on the axes {axes} cannot "
                f"act in the space of {count} axes its neighbour states."
            )
    # A missing system expands as `CoordinateSystem()`, which says nothing.
    changes = {}
    try:
        if own_in is None:
            system = CoordinateSystem() if t.input is None else t.input
            changes["input"] = system.expand(n_in)
        if own_out is None:
            system = CoordinateSystem() if t.output is None else t.output
            changes["output"] = system.expand(n_out)
    except ValueError as error:
        raise CompositionError(
            f"The systems of a subspace transform do not fit the space its "
            f"neighbour states: {error}"
        ) from error
    return t.to(**changes) if changes else t
