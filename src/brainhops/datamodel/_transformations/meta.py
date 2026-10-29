import math
from numbers import Integral, Real

import typing_extensions as tx
from bagof.magic import NotKwOnly, replace

from brainhops._core._internal_api import npvector, smartproperty
from brainhops.datamodel import kinds
from brainhops.datamodel._backends import _MetaBackendMixin
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import CompositionError

from . import nocycles
from .base import Transformation
from .compute.simplify import SimplifyLike
from .compute.simplify import simplify as _simplify
from .compute.utils import (
    axis_list,
    compute_depth,
    refuse_free_axes,
    require_endomorphism,
)
from .modes import ModeLike, normalize_modes

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]


class MetaTransformation(_MetaBackendMixin, Transformation):
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
        with compute_depth() as depth:
            result = _simplify(self, policy=simplify)
        # Only a composition removes the free axes that nothing reads, so
        # the result is checked when something may be composed.
        if depth == 0 and normalize_modes(mode):
            refuse_free_axes(result)
        return result


class SubspaceTransformation(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """Transformation applied to a subset of the axes.

    The wrapped transformation acts on `input_axes` and `output_axes`, and the
    other axes pass through unchanged, so the dimensionality is preserved. A
    spatial transformation of `(x, y, z)`, for example, can be embedded into
    `(x, y, z, t)` and leave time untouched. The class is generic:
    `SubspaceTransformation[T]` embeds a `T`. Whether a subspace
    transformation is of a given kind, such as a rotation, is decided by
    examining the wrapped transformation and the axes it acts on.

    Parameters
    ----------
    transformation : Transformation, optional
        Transformation to apply.
    input_axes : array_like of int, optional
        Axes of the input coordinate system to transform.
    output_axes : array_like of int, optional
        Axes of the output coordinate system to transform.

    Other Parameters
    ----------------
    input : CoordinateSystem, optional
        Input coordinate system of the full space.
        If not given, it is derived from the input system of the wrapped
        transformation.
    output : CoordinateSystem, optional
        Output coordinate system of the full space.
        If not given, it is derived from the output system of the wrapped
        transformation.
    """

    # --- class attributes ---------------------------------------------

    meta_fields: _FieldNames = ("transformation",)
    data_fields: _FieldNames = "transformation", "input_axes", "output_axes"

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
    `created`, whose coordinates are zero unless `at` gives other values.
    The axes that are kept stay in the same order, so a projection is an
    affine map whose matrix selects the kept axes and has a row of zeros
    for each created axis, with the value of that axis as its
    translation. It is linear when every created axis is at zero.

    A created axis whose value is NaN is free: it stands for an input
    that the result must not depend on, as when a transformation is
    restricted to some of its axes. In the matrix products of `compute`,
    an exact zero coefficient cancels the NaN, so that a free axis that
    no output reads disappears, while an output that reads it gets a NaN
    translation. A `compute` that composes raises when its result still
    depends on a free axis.

    The positions in `dropped` refer to the input axes and the positions
    in `created` to the output axes, so a projection does not need to know
    how many axes the space has. When one of its systems is closed, the
    system on the other side is derived from it. When neither system is
    known, a neighbour in a composition gives the number of axes.

    An embedding followed by the projection that drops the same axes is
    the identity, but a projection followed by the embedding that creates
    the same axes sets these axes to zero. A projection that drops axes
    therefore has no inverse, and the inverse of an embedding is the
    projection that drops its axes, which undoes the embedding only on
    the points that the embedding reaches.

    Parameters
    ----------
    dropped : array_like of int, optional
        Axes of the input coordinate system to remove.
    created : array_like of int, optional
        Axes of the output coordinate system to add.

    Other Parameters
    ----------------
    at : array_like of float, optional
        Value of each created axis, in the order of `created`. All
        values are zero when it is omitted, as in the `projectAxis` of
        OME-NGFF, and the projection then stores these zeros. NaN makes
        an axis free.
    input : CoordinateSystem, optional
        Input coordinate system of the full space. If not given, it is
        derived from the output system when that system is closed.
    output : CoordinateSystem, optional
        Output coordinate system of the full space. If not given, it is
        derived from the input system when that system is closed.
    """

    # --- class attributes ---------------------------------------------

    data_fields: _FieldNames = "dropped", "created", "at"

    # --- attributes ---------------------------------------------------

    dropped: NotKwOnly[npvector[Integral]] = ()
    """Axes of the input coordinate system to remove."""

    created: NotKwOnly[npvector[Integral]] = ()
    """Axes of the output coordinate system to add."""

    at: npvector[Real] = ()
    """Value of each created axis, which is NaN when the axis is free."""

    def __post_init__(self, arguments: tx.Any) -> None:
        values = [] if self.at is None else list(self.at)
        created = axis_list(self.created)
        if values and len(values) != len(created):
            raise ValueError(
                f"A projection needs one value in `at` per created axis, "
                f"but it creates {len(created)} and `at` holds "
                f"{len(values)}."
            )
        if not values:
            # An omitted `at` is stored as one zero per created axis, so
            # that the code that reads `at` needs no special case.
            self.at = [0.0] * len(created)
        super().__post_init__(arguments)

    # --- properties ---------------------------------------------------

    @smartproperty
    def input(self) -> tx.Optional[CoordinateSystem]:
        if self._input is not None:
            return self._input
        return _projected_system(self._output, self.created, self.dropped)

    @smartproperty
    def output(self) -> tx.Optional[CoordinateSystem]:
        if self._output is not None:
            return self._output
        return _projected_system(self._input, self.dropped, self.created)

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> tx.Self:
        """Return the projection that drops the axes this one creates.

        Raises
        ------
        ValueError
            If the projection drops axes, because the coordinates along
            these axes are lost.
        """
        if axis_list(self.dropped):
            raise ValueError(
                "A projection that drops axes has no inverse, because the "
                "coordinates along the dropped axes are lost. The embedding "
                "that creates these axes again at zero is only a right "
                "inverse."
            )
        return self.to(
            dropped=self.created,
            created=self.dropped,
            at=(),
            input=self._output,
            output=self._input,
        )


@kinds.Bijection
class Bijection(MetaTransformation, tx.Generic[TRANSFORMATION]):
    """Transformation whose inverse is defined explicitly.

    The class is generic: `Bijection[T]` holds two transformations of type `T`.
    The class is registered as a bijection. Its more specific kind, such as
    affine, is decided from the forward transformation, or from the inverse of
    the backward transformation when no forward one is given.

    Parameters
    ----------
    forward : Transformation, optional
        Forward transformation.
    backward : Transformation, optional
        Backward transformation.

    Other Parameters
    ----------------
    input : CoordinateSystem, optional
        Input coordinate system of the full space.
        If not given, it is derived from the input system of the wrapped
        transformation(s).
    output : CoordinateSystem, optional
        Output coordinate system of the full space.
        If not given, it is derived from the input system of the wrapped
        transformation(s).
    """

    # --- class attributes ---------------------------------------------

    meta_fields: _FieldNames = "forward", "backward"
    data_fields: _FieldNames = "forward", "backward"

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
    if list(get_axes(system)) == [...]:
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


def _projected_system(
    system: tx.Optional[CoordinateSystem],
    removed: tx.Any,
    added: tx.Any,
) -> tx.Optional[CoordinateSystem]:
    """Return the system on the other side of a projection.

    The axes of `system` at the positions `removed` are removed, and
    unknown axes are added at the positions `added` of the result. The
    other axes keep their order. `None` is returned when `system` is
    missing or open, because the number of axes on the other side is then
    unknown, and when a position does not fit.
    """
    axes = get_axes(system)
    if system is None or axes.ndim is None:
        return None
    removed, added = axis_list(removed), axis_list(added)
    ndim = axes.ndim - len(removed) + len(added)
    if any(not 0 <= i < axes.ndim for i in removed) or any(
        not 0 <= i < ndim for i in added
    ):
        return None
    kept = [i for i in range(axes.ndim) if i not in removed]
    positions = [i for i in range(ndim) if i not in added]
    if len(kept) != len(positions):
        return None
    return system.restrict(kept).embed(positions, ndim=ndim)


def _close_projection(
    t: Projection,
    n_in: tx.Optional[int] = None,
    n_out: tx.Optional[int] = None,
) -> Projection:
    """Fix the number of axes in the systems of a projection, when known.

    A neighbour in a composition may say how many axes the space between
    them has, as for [`_close_subspace`][]. The number on one side gives
    the number on the other, because a projection removes `len(dropped)`
    axes and adds `len(created)` axes. A number that `t` states itself is
    never overridden. If no number is known, `t` is returned unchanged.

    Raises
    ------
    CompositionError
        If the numbers of axes do not fit the axes that `t` drops and
        creates.
    """
    dropped, created = axis_list(t.dropped), axis_list(t.created)
    own_in, own_out = get_axes(t.input).ndim, get_axes(t.output).ndim
    n_in = own_in if own_in is not None else n_in
    n_out = own_out if own_out is not None else n_out
    if n_in is None and n_out is not None:
        n_in = n_out - len(created) + len(dropped)
    if n_out is None and n_in is not None:
        n_out = n_in - len(dropped) + len(created)
    if n_in is None or n_out is None:
        return t
    if (
        n_out != n_in - len(dropped) + len(created)
        or any(not 0 <= i < n_in for i in dropped)
        or any(not 0 <= i < n_out for i in created)
    ):
        raise CompositionError(
            f"A projection that drops the axes {dropped} and creates the "
            f"axes {created} cannot map {n_in} axes onto {n_out} axes, as "
            f"its neighbour states."
        )
    # An open system that is stored is closed. When no system is stored,
    # only the input system is set, and the output system is derived from
    # it, so that it keeps the axes that the input system describes.
    changes = {}
    try:
        if own_in is None and t._input is not None:
            changes["input"] = t._input.expand(n_in)
        if own_out is None and t._output is not None:
            changes["output"] = t._output.expand(n_out)
        if t._input is None and t._output is None:
            changes["input"] = CoordinateSystem().expand(n_in)
    except ValueError as error:
        raise CompositionError(
            f"The systems of a projection do not fit the space its "
            f"neighbour states: {error}"
        ) from error
    return t.to(**changes) if changes else t


def _has_free_axes(t: Projection) -> bool:
    """Return whether a projection creates an axis whose value is NaN."""
    return any(math.isnan(value) for value in t.at)
