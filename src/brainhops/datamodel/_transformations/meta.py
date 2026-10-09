from numbers import Integral

import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

from brainhops._core.properties import smartproperty
from brainhops._core.typing import npvector
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError

from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import axis_list, require_endomorphism
from .modes import ModeLike

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)


class MetaTransformation(Transformation):
    """Base class of the transformations defined in terms of others.

    The subclasses are [`SubspaceTransformation`][], [`Projection`][] and
    [`Bijection`][]. The class is not meant to be instantiated directly.
    """

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        if factor:
            # The sequence engine runs the factor pass and never passes
            # `factor` back to the `compute` of a leaf, so there is no
            # recursion.
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        # A meta transformation has no parameter of its own to fuse, so
        # computing it is simplifying it: the registered simplifier recurses
        # into what it wraps. `mode` gates composition, and there is nothing
        # here to compose.
        return _simplify(self, policy=simplify)


class SubspaceTransformation(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """Transformation applied to a subset of the axes.

    The wrapped transformation acts on `input_axes` and `output_axes`, and the
    other axes pass through unchanged, so the dimensionality is preserved. A
    spatial transformation of `(x, y, z)`, for example, can be embedded into
    `(x, y, z, t)` and leave time untouched. The class is generic:
    `SubspaceTransformation[T]` embeds a `T`. Its kind is decided by a checker
    that recurses into the wrapped transformation.
    """

    data_fields: tx.ClassVar[tx.Tuple[str]] = (
        "transformation",
        "input_axes",
        "output_axes",
    )

    transformation: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Transformation to apply."""

    input_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """Axes of the input coordinate system to transform."""

    output_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """Axes of the output coordinate system to transform."""

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        system = getattr(self.transformation, "input", None)
        return _subsystem(system, self.input_axes, full=self._input)

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        system = getattr(self.transformation, "output", None)
        return _subsystem(system, self.output_axes, full=self._output)

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
        # A subspace with the same axes on both sides is blockdiag(inner, I),
        # whose principal root is blockdiag(sqrt(inner), I), so the root acts
        # on the inner transformation alone and keeps its axes.
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
    """Projection that removes axes, or the embedding that adds them.

    Removing axes maps a space to one of lower dimension; the inverse, an
    embedding, adds them back.
    """

    data_fields: tx.ClassVar[tx.Tuple[str]] = "dropped", "created"

    dropped: NotKwOnly[npvector[Integral]] = ()
    created: NotKwOnly[npvector[Integral]] = ()

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        # Swapping the dropped and created axes is the exact inverse, so there
        # is nothing to defer.
        return self.to(
            dropped=self.created,
            created=self.dropped,
            input=self.output,
            output=self.input,
        )


@kinds.Bijection
class Bijection(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """Transformation whose inverse is defined explicitly.

    The class is generic: `Bijection[T]` holds two transformations of type `T`.
    It is declared bijective, and a checker refines its kind from the forward
    map.
    """

    data_fields: tx.ClassVar[tx.Tuple[str]] = "forward", "backward"

    forward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Forward transformation."""

    backward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Backward transformation."""

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

    def compute(
        self,
        mode: ModeLike = True,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        if not factor:
            return super().compute(mode, simplify=simplify)
        # A bijection is a container of two sides, so factoring is forwarded to
        # both, which factor independently.
        forward, backward = self.forward, self.backward
        if forward is not None:
            forward = forward.compute(mode, simplify=simplify, factor=True)
        if backward is not None:
            backward = backward.compute(mode, simplify=simplify, factor=True)
        if forward is self.forward and backward is self.backward:
            # Keep the object unchanged, so that an adjacent inverse of this
            # bijection still cancels.
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
        # The principal root of an inverse is the inverse of the principal
        # root, so both directions are kept.
        require_endomorphism(self, "square root")
        forward, backward = self.forward, self.backward
        obj = self.to(
            forward=None if forward is None else forward.sqrt(),
            backward=None if backward is None else backward.sqrt(),
        )
        return obj.compute(**kwargs) if compute else obj


def _same_axes(t: SubspaceTransformation) -> bool:
    # Whether the subspace reads and writes the same axes in the same order,
    # that is, embeds the inner transformation without reindexing.
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
    """Build the full-space system that a subspace transformation presents.

    `system` is the system of the inner transformation, with one axis per
    dimension it acts on, and `index` holds the positions of those axes in the
    full space. A declared endpoint `full` is returned when there is one.
    Otherwise, the system is derived with
    [`CoordinateSystem.embed`][brainhops.datamodel.systems.CoordinateSystem.embed]:
    inner axis `j` sits at `index[j]`, the other positions hold unknown axes,
    and the system ends with `...`, because the positions are known but the
    total count is not. The result is named `subspace(<name>)`. An inner system
    that states no axis is returned as it is.
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
    """Give a subspace transformation closed full-space systems, when possible.

    An unset or open system does not know how many axes the full space has, but
    a neighbour in a composition may know, since the space between two
    transformations is one space. `n_in` and `n_out` are the counts for the
    input and output sides of `t`, and one side gives the other, because the
    pass-through axes are the same on both sides. A count that `t` states
    itself is never overridden. The open systems are closed with
    [`CoordinateSystem.expand`][brainhops.datamodel.systems.CoordinateSystem.expand]
    and declared on the returned transformation; `t` is returned unchanged if
    no count is known.

    Raises
    ------
    CompositionError
        If a count cannot hold the axes that `t` acts on, or the axes that its
        systems state.
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
    # A missing system expands from a `CoordinateSystem()`, which states
    # nothing.
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
