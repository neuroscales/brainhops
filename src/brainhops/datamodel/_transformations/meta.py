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
            # The sequence performs the factoring and never passes `factor`
            # back to the `compute()` of its elements, so this call does not
            # recurse.
            return nocycles.SEQUENCE([self]).compute(
                mode, simplify=simplify, factor=True
            )
        # A meta transformation has no parameter of its own to combine, so
        # computing it amounts to simplifying it, and the registered
        # simplifier recurses into the transformations that it wraps. `mode`
        # only controls composition, and there is nothing here to compose.
        return _simplify(self, policy=simplify)


class SubspaceTransformation(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """Transformation applied to a subset of the axes.

    The wrapped transformation acts on `input_axes` and `output_axes`, and the
    other axes pass through unchanged, so the dimensionality is preserved. A
    spatial transformation of `(x, y, z)`, for example, can be embedded into
    `(x, y, z, t)` and leave time untouched. The class is generic:
    `SubspaceTransformation[T]` embeds a `T`. Whether a subspace
    transformation is of a given kind, such as a rotation, is decided by
    examining the wrapped transformation and the axes it acts on.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = (
        "transformation",
        "input_axes",
        "output_axes",
    )

    # --- attributes ---------------------------------------------------

    transformation: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Transformation to apply."""

    input_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """Axes of the input coordinate system to transform."""

    output_axes: NotKwOnly[tx.Optional[npvector[Integral]]] = None
    """Axes of the output coordinate system to transform."""

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

    A projection removes the axes listed in `dropped` and therefore maps a
    space to one of lower dimension. An embedding adds the axes listed in
    `created`. Each is the inverse of the other, and the inverse is obtained
    by swapping the two lists.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "dropped", "created"

    # --- attributes ---------------------------------------------------

    dropped: NotKwOnly[npvector[Integral]] = ()
    created: NotKwOnly[npvector[Integral]] = ()

    # --- methods ------------------------------------------------------

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
    The class is registered as a bijection. Its more specific kind, such as
    affine, is decided from the forward transformation, or from the inverse of
    the backward transformation when no forward one is given.
    """

    # --- class attributes ---------------------------------------------

    data_fields: tx.ClassVar[tx.Tuple[str]] = "forward", "backward"

    # --- attributes ---------------------------------------------------

    forward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Forward transformation."""

    backward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """Backward transformation."""

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
    # Return whether the subspace reads and writes the same axes in the same
    # order, that is, whether it embeds the inner transformation without
    # reindexing its axes.
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
    """Build the system of the full space on one side of a subspace.

    `system` is the system of the inner transformation, with one axis per
    dimension it acts on, and `index` holds the positions of those axes in the
    full space. A declared system `full` is returned when there is one.
    Otherwise, the system is derived with
    [`CoordinateSystem.embed`][brainhops.datamodel.systems.CoordinateSystem.embed].
    Axis `j` of the inner system is placed at position `index[j]`, the other
    positions hold unknown axes, and the system ends with `...`, because the
    positions are known but the total number of axes is not. The result is
    named `subspace(<name>)`. An inner system that states no axis is returned
    as it is.
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
    """Fix the number of axes in the systems of a subspace, when it is known.

    A system that is unset, or open because it ends with `...`, does not say
    how many axes the full space has. A neighbour in a composition may say so,
    since the space between two transformations is a single space. `n_in` and
    `n_out` are the numbers of axes on the input and output sides of `t`. The
    number on one side gives the number on the other, because the axes that
    pass through unchanged are the same on both sides. A number that `t`
    states itself is never overridden. The open systems are closed with
    [`CoordinateSystem.expand`][brainhops.datamodel.systems.CoordinateSystem.expand]
    and declared on the returned transformation. If no number is known, `t`
    is returned unchanged.

    Raises
    ------
    CompositionError
        If a number of axes is too small to hold the axes that `t` acts on,
        or the axes that its systems state.
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
    # A missing system is expanded from an empty `CoordinateSystem()`, which
    # states no axis.
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
