"""Reduction of the axes that a chain of transformations does not need.

A chain often computes axes that a later element throws away, or adds an
axis whose value is fixed. Restricting a transformation to some of its
axes, for example, wraps it between a projection that creates the
removed inputs and a projection that drops the removed outputs, both of
them instances of
[`Projection`][brainhops.datamodel.transformations.Projection].
The reduction of this module rewrites such a chain, without composing
anything, so that every element only handles the axes that are needed.
[`simplify`][brainhops.datamodel._transformations.compute.simplify.simplify]
runs it on every chain, and the factor pass uses it to find the two ends
of its normal form.

The reduction reads the dependency pattern of each element, which says
which of its outputs read which of its inputs. It is a boolean matrix
with one row per output and one column per input. A projection is an
element like any other: the axes that it creates are outputs that read
no input, and the axes that it drops are inputs that no output reads.
Two passes read the patterns.

1. The backward pass starts from the outputs at the end of the chain,
   which are all needed. At each element, going towards the input, an
   input is needed when a needed output reads it. Each element is cut
   down to its needed outputs and to the inputs that they read, which is
   a closed block, and an input that is not needed is dropped as early
   as possible.
2. The forward pass follows each created axis, and its value, from the
   element that creates it towards the output. A created axis is carried
   through the elements that pass it through unchanged, such as a
   permutation, an identity or a subspace that does not act on it. An
   output that an element creates at a value, such as a constant row of
   a matrix under the numeric policy, becomes a created axis too. When
   an element reads a created axis, whether its value is a number or
   NaN, the axis is created again just before that element, and a
   composing `compute` folds the value into the element by a matrix
   product. The reduction itself computes nothing: it only copies the
   values of created axes.

Between two elements, and at the two ends of the chain, a projection
drops the axes that the next element does not read and creates the axes
that it reads but that the previous element did not produce. The
reduction drops an axis only when the patterns prove that nothing reads
it, and it never raises.

The knowledge about each type is registered on two dispatched functions.
A pattern reader, registered with [`pattern_reader`][] and called as
`f(t, policy)`, returns the pattern of `t` under a simplification
policy, or [`UNREADABLE`][] when it cannot tell. An element whose
pattern cannot be read counts as fully coupled. A block restrictor,
registered with [`block_restrictor`][] and called as
`f(t, inputs, outputs, policy)`, cuts `t` down to a closed block, which
means that the outputs in `outputs` read only inputs in `inputs`. It
returns None to decline. An element whose restrictor declines keeps all
its axes, so that a drop happens after it and a creation stays before
it. The built-in rules live in the `reducers` module.
"""

__all__ = [
    "pattern_reader",
    "block_restrictor",
    "read_pattern",
    "restrict_block",
    "counts_of",
    "stage_counts",
    "may_reduce",
    "reduce_chain",
]


import numpy as np
import typing_extensions as tx
from bagof.dispatchers import Function
from bagof.magic import replace

from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.errors import (
    CompositionError,
)

from .utils import (
    UNREADABLE,
    _preserves_count,
    axis_counts,
    axis_list,
    boundary_disagrees,
    systems_disagree,
)

if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem

    from ..base import Transformation
    from .simplify import SimplifyTable


_read_pattern: Function = Function("read_pattern")
"""Dispatched function on which every pattern reader is registered."""

_restrict_block: Function = Function("restrict_block")
"""Dispatched function on which every block restrictor is registered."""

# ======================================================================
#
#                            D I S P A T C H
#
# ======================================================================


def pattern_reader(func: tx.Callable) -> tx.Callable:
    """Register a pattern reader for the type of its first parameter.

    A pattern reader takes `(t, policy)`, where `policy` is a
    [`SimplifyTable`][brainhops.datamodel._transformations.compute.simplify.SimplifyTable],
    and returns the dependency pattern of `t` as a boolean array of
    shape `(n_out, n_in)`, or `UNREADABLE`.
    """
    _read_pattern.register(func)
    return func


def block_restrictor(func: tx.Callable) -> tx.Callable:
    """Register a block restrictor for the type of its first parameter.

    A block restrictor takes `(t, inputs, outputs, policy)`, where
    `inputs` and `outputs` are two sorted lists of axis positions, and
    returns the transformation from the axes `inputs` to the axes
    `outputs`, or None to decline.
    """
    _restrict_block.register(func)
    return func


def read_pattern(t: "Transformation", policy: "SimplifyTable") -> tx.Any:
    """Return the dependency pattern of a transformation.

    Parameters
    ----------
    t : Transformation
        Transformation whose numbers of axes are stated, by its
        parameters or by closed coordinate systems.
    policy : SimplifyTable
        Policy that says how closely `t` may be read. The analytic
        policy reads only the type and the structure of `t`, and the
        numeric policy also reads the exact zeros of a matrix.

    Returns
    -------
    numpy.ndarray or object
        Boolean array of shape `(n_out, n_in)` whose entry `(i, j)` is
        true when output `i` may read input `j`, or `UNREADABLE` when
        the pattern cannot be read. A true entry does not prove a
        dependence, but a false entry proves its absence.
    """
    try:
        return _read_pattern(t, policy=policy)
    except Exception:
        # A type without a rule, or a rule that fails, counts as fully
        # coupled, because the reduction never raises.
        return UNREADABLE


def restrict_block(
    t: "Transformation",
    inputs: tx.Sequence[int],
    outputs: tx.Sequence[int],
    policy: "SimplifyTable",
) -> tx.Optional["Transformation"]:
    """Cut a transformation down to a closed block of its axes.

    The block is closed when the outputs in `outputs` read only inputs
    in `inputs`, which the caller guarantees. The result keeps the type
    of `t` when that type can represent the block, so that a cut
    [`Scaling`][brainhops.datamodel.transformations.Scaling] is a
    smaller `Scaling`.

    Parameters
    ----------
    t : Transformation
        Transformation whose numbers of axes are stated.
    inputs : sequence of int
        Sorted positions of the inputs of the block.
    outputs : sequence of int
        Sorted positions of the outputs of the block.
    policy : SimplifyTable
        Policy that says how closely `t` may be read. Under the analytic
        policy, a lazy inverse is never computed, so a rule declines a
        block that it could only cut from the computed inverse.

    Returns
    -------
    Transformation or None
        Transformation from the axes `inputs` to the axes `outputs`, an
        [`Identity`][brainhops.datamodel.transformations.Identity] when
        the block is the identity, or None when the rule declines.
    """
    try:
        return _restrict_block(t, list(inputs), list(outputs), policy)
    except Exception:
        # A type without a rule, or a rule that fails on an unexpected
        # block, declines, because the reduction never raises.
        return None


# ======================================================================
#
#                              C O U N T S
#
# ======================================================================


def _delta(t: "Transformation") -> tx.Optional[int]:
    # Return how many axes `t` adds, which may be negative, or None when
    # that number is unknown. A projection and a subspace know it even
    # when they do not know their numbers of axes.
    from ..concrete import Identity
    from ..derived import DerivedTransformation
    from ..inverse import Inverse
    from ..meta import Projection, SubspaceTransformation
    from ..sequence import Sequence

    if isinstance(t, Projection):
        return len(axis_list(t.created)) - len(axis_list(t.dropped))
    if isinstance(t, SubspaceTransformation):
        if t.input_axes is None or t.output_axes is None:
            return None
        return len(axis_list(t.output_axes)) - len(axis_list(t.input_axes))
    if isinstance(t, Inverse):
        if t.forward is None:
            return 0
        delta = _delta(t.forward)
        return None if delta is None else -delta
    if isinstance(t, DerivedTransformation):
        return 0
    if isinstance(t, Identity):
        return 0
    if isinstance(t, Sequence):
        total = 0
        for member in t.transformations or []:
            delta = _delta(member)
            if delta is None:
                return None
            total += delta
        return total
    ni, no = axis_counts(t)
    if ni is None or no is None:
        return 0 if _preserves_count(t) else None
    return no - ni


def counts_of(t: "Transformation") -> tx.Tuple[tx.Optional[int], ...]:
    """Return the numbers of input and output axes that `t` states.

    This is [`axis_counts`][], except for a projection, whose counts are
    read from the systems that it stores. A projection derives the
    system on its other side, which costs more than the reduction needs.
    """
    from ..meta import Projection

    if not isinstance(t, Projection):
        return axis_counts(t)
    ni = get_axes(t._input).ndim
    no = get_axes(t._output).ndim
    delta = len(axis_list(t.created)) - len(axis_list(t.dropped))
    if ni is None and no is not None:
        ni = no - delta
    if no is None and ni is not None:
        no = ni + delta
    return ni, no


def unnamed_system(n: int) -> "CoordinateSystem":
    """Return a new closed system of `n` axes that states nothing else.

    A new system is built at each call, because a system can be edited
    in place.
    """
    from brainhops.datamodel.systems import CoordinateSystem

    return CoordinateSystem().expand(n)


def stage_counts(
    leaves: tx.Sequence["Transformation"],
    start: tx.Optional[int] = None,
) -> tx.Optional[tx.List[int]]:
    """Return the number of axes before and after each chain element.

    An element that does not state its own numbers of axes takes them
    from its neighbours, because the space between two elements is a
    single space.

    Parameters
    ----------
    leaves : sequence of Transformation
        Elements of the chain, in the order in which they are applied.
    start : int, optional
        Number of axes before the first element, when the caller knows
        it.

    Returns
    -------
    list of int or None
        List of `len(leaves) + 1` numbers, in which entry `k` is the
        number of axes between element `k - 1` and element `k`. None is
        returned when a number is unknown or when two elements disagree.
    """
    n = len(leaves)
    counts: tx.List[tx.Optional[int]] = [None] * (n + 1)
    counts[0] = start
    facts = [(counts_of(t), _delta(t)) for t in leaves]

    def settle(index: int, value: int) -> bool:
        # Record a count, and tell whether it is new. A count that
        # contradicts the recorded one is an error of the chain.
        if counts[index] is None:
            counts[index] = value
            return True
        if counts[index] != value:
            raise ValueError
        return False

    try:
        changed = True
        while changed:
            changed = False
            for k, ((ni, no), delta) in enumerate(facts):
                if ni is not None:
                    changed |= settle(k, ni)
                if no is not None:
                    changed |= settle(k + 1, no)
                if delta is None:
                    continue
                if counts[k] is not None:
                    changed |= settle(k + 1, counts[k] + delta)
                if counts[k + 1] is not None:
                    changed |= settle(k, counts[k + 1] - delta)
    except ValueError:
        return None
    if any(c is None or c < 0 for c in counts):
        return None
    return tx.cast(tx.List[int], counts)


def close_counts(
    t: "Transformation", n_in: int, n_out: int
) -> tx.Optional["Transformation"]:
    """Return `t` with closed systems when it does not state its counts.

    A projection and a subspace take the numbers of axes of the chain
    through their own rules, and None is returned when the numbers do
    not fit them. Another transformation that does not state its
    numbers, such as an affine without a matrix, receives unnamed closed
    systems. A transformation that states its numbers is returned as the
    same object.
    """
    from ..meta import (
        Projection,
        SubspaceTransformation,
        _close_projection,
        _close_subspace,
    )

    try:
        if isinstance(t, Projection):
            return _close_projection(t, n_in, n_out)
        if isinstance(t, SubspaceTransformation):
            return _close_subspace(t, n_in, n_out)
    except CompositionError:
        return None
    own_in, own_out = counts_of(t)
    if own_in is not None and own_out is not None:
        return t
    edits = {}
    if own_in is None:
        edits["input"] = unnamed_system(n_in)
    if own_out is None:
        edits["output"] = unnamed_system(n_out)
    try:
        return t.to(**edits)
    except Exception:
        return t


# ======================================================================
#
#                             T R I G G E R
#
# ======================================================================


def may_reduce(
    leaves: tx.Sequence["Transformation"], policy: "SimplifyTable"
) -> bool:
    """Tell cheaply whether the reduction may find axes to remove.

    The reduction may find something only when an element has different
    numbers of input and output axes or, under the numeric policy, when
    the linear part of a matrix transformation has a zero row or a zero
    column. The test reads no pattern, because
    [`simplify`][brainhops.datamodel._transformations.compute.simplify.simplify]
    runs on every partial product of a composition.
    """
    for t in leaves:
        if _changes_axes(t):
            return True
        if policy.resolve(t) is SimplifyPolicy.numeric and _has_zero_line(t):
            return True
    return False


def _changes_axes(t: "Transformation") -> bool:
    # Tell whether `t` has different numbers of input and output axes,
    # or wraps a transformation that does.
    from ..meta import Projection, SubspaceTransformation

    if isinstance(t, Projection):
        return bool(axis_list(t.dropped) or axis_list(t.created))
    if isinstance(t, SubspaceTransformation):
        if t.input_axes is not None and t.output_axes is not None:
            if len(t.input_axes) != len(t.output_axes):
                return True
        inner = t.transformation
        return inner is not None and _changes_axes(inner)
    ni, no = counts_of(t)
    return ni is not None and no is not None and ni != no


def _has_zero_line(t: "Transformation") -> bool:
    # Tell whether the linear part of a matrix transformation has a row
    # or a column of exact zeros. A lazy inverse is not read, because
    # the inverse of an invertible matrix has no zero line.
    from ..concrete import Affine, Linear, Scaling
    from ..inverse import Inverse
    from ..meta import SubspaceTransformation

    if isinstance(t, Inverse):
        return False
    if isinstance(t, SubspaceTransformation):
        inner = t.transformation
        return inner is not None and _has_zero_line(inner)
    if isinstance(t, Scaling):
        return t.scale is not None and bool(np.any(np.asarray(t.scale) == 0))
    if isinstance(t, (Affine, Linear)):
        if t.matrix is None:
            return False
        matrix = np.asarray(t.matrix)
        linear = matrix[:, :-1] if isinstance(t, Affine) else matrix
        zero = linear == 0
        return bool(zero.all(axis=1).any() or zero.all(axis=0).any())
    return False


# ======================================================================
#
#                            R E D U C T I O N
#
# ======================================================================

# Markers of the backward pass for an element that keeps all its axes
# and for an element whose outputs are all unused.
_KEEP = object()
_REMOVED = object()


class _Abort(Exception):
    # Raised inside the reduction when the chain is not in a state that
    # the passes can handle. The reduction then leaves the chain alone.
    pass


def reduce_chain(
    leaves: tx.Sequence["Transformation"],
    policy: "SimplifyTable",
    *,
    ends: bool = False,
) -> tx.Optional[tx.List["Transformation"]]:
    """Remove the axes that a chain does not need, without composing.

    The two passes are described in the module documentation. An
    element that is not cut is kept as the same object, so that it
    still cancels with its own lazy inverse. Nothing raises: a chain
    that the passes cannot handle is left alone.

    Parameters
    ----------
    leaves : sequence of Transformation
        Flat chain, in the order in which the elements are applied.
    policy : SimplifyTable
        Policy that says how closely each element may be read. An
        element that the policy leaves alone keeps all its axes.
    ends : bool, default=False
        Whether the result may be longer than the chain. Simplification
        never lengthens a chain, so by default a reduction that would
        add elements, for example the projection that drops an input
        that the first element does not read, is not applied. The factor
        pass sets `ends` to obtain the two ends of its normal form.

    Returns
    -------
    list of Transformation or None
        The reduced chain, or None when nothing is removed.
    """
    try:
        return _reduce(list(leaves), policy, ends)
    except Exception:
        # A rule that fails on an unexpected block is a refusal, because
        # the reduction never raises.
        return None


def _reduce(
    leaves: tx.List["Transformation"], policy: "SimplifyTable", ends: bool
) -> tx.Optional[tx.List["Transformation"]]:
    if not leaves:
        return None
    if any(boundary_disagrees(a, b) for a, b in zip(leaves, leaves[1:])):
        # Reconciling the systems of two neighbours is the job of
        # composition, which inserts a bridge before it simplifies.
        return None
    counts = stage_counts(leaves)
    if counts is None:
        return None
    elements = []
    for k, t in enumerate(leaves):
        closed = close_counts(t, counts[k], counts[k + 1])
        if closed is None:
            return None
        elements.append(closed)
    try:
        chain = _Reduction(leaves, elements, counts, policy).run()
    except _Abort:
        return None
    if _same_chain(chain, leaves):
        return None
    if not ends and len(chain) > len(leaves):
        return None
    return chain


class _Reduction:
    # One run of the two passes over a chain. `elements` holds the
    # leaves with closed systems where a leaf did not state its numbers
    # of axes, and `counts[k]` is the number of axes before element k.

    def __init__(
        self,
        leaves: tx.List["Transformation"],
        elements: tx.List["Transformation"],
        counts: tx.List[int],
        policy: "SimplifyTable",
    ) -> None:
        self.leaves = leaves
        self.elements = elements
        self.counts = counts
        self.policy = policy
        self.n = len(leaves)
        self.full = [list(range(c)) for c in counts]
        self.frozen = [
            policy.resolve(t) is SimplifyPolicy.none for t in leaves
        ]
        self.patterns = [self._pattern(k) for k in range(self.n)]
        self.systems = [self._stage_system(k) for k in range(self.n + 1)]
        self.blocks: tx.List[tx.Tuple[tx.List[int], tx.List[int]]] = []
        self.cuts: tx.List[tx.Any] = []

    # --- reading ------------------------------------------------------

    def _pattern(self, k: int) -> np.ndarray:
        # Read the pattern of element k, which is full when it cannot be
        # read or does not have the shape of its stage.
        shape = (self.counts[k + 1], self.counts[k])
        if not self.frozen[k]:
            pattern = read_pattern(self.elements[k], self.policy)
            if pattern is not UNREADABLE:
                pattern = np.asarray(pattern, dtype=bool)
                if pattern.shape == shape:
                    return pattern
        return np.ones(shape, dtype=bool)

    def _restrict(
        self, k: int, inputs: tx.List[int], outputs: tx.List[int]
    ) -> tx.Optional["Transformation"]:
        return restrict_block(self.elements[k], inputs, outputs, self.policy)

    def _stage_system(self, k: int) -> tx.Optional["CoordinateSystem"]:
        # Return the closed system of the space before element k, read
        # from either neighbour, or None when neither states one. Only
        # the systems that the leaves store are read, because the
        # systems that a projection derives cost more to build.
        candidates = []
        if k > 0:
            candidates.append(getattr(self.leaves[k - 1], "_output", None))
        if k < self.n:
            candidates.append(getattr(self.leaves[k], "_input", None))
        for system in candidates:
            if get_axes(system).ndim == self.counts[k]:
                return system
        return None

    def _subsystem(
        self, k: int, axes: tx.List[int]
    ) -> tx.Optional["CoordinateSystem"]:
        # Return the system of stage k restricted to some of its axes.
        system = self.systems[k]
        if system is None:
            return None
        return system.restrict(axes)

    # --- passes -------------------------------------------------------

    def run(self) -> tx.List["Transformation"]:
        self._backward()
        return self._finish(self._forward())

    def _backward(self) -> None:
        # Find, from the end of the chain, the outputs that each element
        # must produce and the inputs that they read, and cut each
        # element down to that block.
        blocks: tx.List[tx.Any] = [None] * self.n
        cuts: tx.List[tx.Any] = [None] * self.n
        need = self.full[self.n]
        for k in reversed(range(self.n)):
            outputs = need
            if not outputs and not self.frozen[k]:
                # None of the outputs is needed, so the element is
                # removed when its rule accepts the empty block. A rule
                # that declines, such as that of a grid, keeps the
                # element.
                if self._restrict(k, [], []) is not None:
                    blocks[k], cuts[k] = ([], []), _REMOVED
                    continue
            pattern = self.patterns[k]
            inputs = np.nonzero(pattern[outputs].any(axis=0))[0].tolist()
            constant = not pattern[outputs].any(axis=1).all()
            whole = (
                outputs == self.full[k + 1]
                and inputs == self.full[k]
                and not constant
            )
            cut: tx.Any = _KEEP
            if whole and not self.frozen[k]:
                # The element is kept whole by this pass, but the
                # forward pass may still cut it, when a created axis
                # passes through it.
                cut = self.leaves[k]
            elif not self.frozen[k]:
                piece = self._restrict(k, inputs, outputs)
                if piece is not None:
                    cut = piece
            if cut is _KEEP:
                outputs, inputs = self.full[k + 1], self.full[k]
            blocks[k], cuts[k] = (inputs, outputs), cut
            need = inputs
        self.blocks, self.cuts = blocks, cuts

    def _forward(self) -> tx.List[tx.Optional["Transformation"]]:
        # Follow the created axes from the start of the chain, and emit
        # each piece together with the projection that feeds it.
        out: tx.List[tx.Optional[Transformation]] = []
        present = self.full[0]
        created: tx.Dict[int, float] = {}
        for k in range(self.n):
            cut = self.cuts[k]
            if cut is _REMOVED:
                out.append(self._glue(k, present, [], created))
                present, created = [], {}
                continue
            if cut is _KEEP:
                inputs, outputs = self.blocks[k]
                piece, carried = self.leaves[k], {}
            else:
                inputs, piece, outputs, carried = self._cut(k, created)
            out.append(self._glue(k, present, inputs, created))
            if piece is not None:
                out.append(self._with_systems(k, piece, inputs, outputs))
            present, created = outputs, carried
        out.append(self._glue(self.n, present, self.full[self.n], created))
        return out

    def _cut(
        self, k: int, created: tx.Dict[int, float]
    ) -> tx.Tuple[
        tx.List[int],
        tx.Optional["Transformation"],
        tx.List[int],
        tx.Dict[int, float],
    ]:
        # Cut element k for the forward pass. An output that the element
        # creates, or that passes a created axis through unchanged, is
        # created after the element instead. Every other output keeps
        # the inputs that it reads, so that a created axis that it reads
        # is created again just before the element. Return the inputs
        # that the piece consumes, the piece, the outputs that it
        # produces and the values of the axes created after it.
        inputs, outputs = self.blocks[k]
        pattern = self.patterns[k]
        reads = {o: np.nonzero(pattern[o])[0].tolist() for o in outputs}
        carried: tx.Dict[int, float] = {}
        for o in outputs:
            value = self._carried(k, reads[o], o, created)
            if value is not None:
                carried[o] = value
        kept = [o for o in outputs if o not in carried]
        if not kept:
            return [], None, [], carried
        read = sorted({i for o in kept for i in reads[o]})
        piece = self.cuts[k]
        if (kept, read) != (outputs, inputs):
            piece = self._restrict(k, read, kept)
            if piece is None:
                # The smaller block is declined, so the block of the
                # backward pass is used, and nothing is carried.
                piece, kept, read = self.cuts[k], outputs, inputs
                carried = {}
        return read, piece, kept, carried

    def _carried(
        self,
        k: int,
        sources: tx.List[int],
        output: int,
        created: tx.Dict[int, float],
    ) -> tx.Optional[float]:
        # Return the value of an output of element k that is a created
        # axis, or None when it is not. The output is a created axis
        # when it reads no input and the element creates it at a value,
        # or when it reads a single created axis and passes it through
        # unchanged. Values are only copied, never computed.
        from ..concrete import Identity
        from ..meta import Projection

        if not sources:
            piece = self._restrict(k, [], [output])
            if (
                isinstance(piece, Projection)
                and not axis_list(piece.dropped)
                and len(piece.at) == 1
            ):
                return float(piece.at[0])
            return None
        if len(sources) != 1 or sources[0] not in created:
            return None
        piece = self._restrict(k, sources, [output])
        if isinstance(piece, Identity) and not _changes_system(piece):
            return created[sources[0]]
        return None

    def _glue(
        self,
        k: int,
        present: tx.List[int],
        wanted: tx.List[int],
        created: tx.Dict[int, float],
    ) -> tx.Optional["Transformation"]:
        # Return the projection, at stage k, that drops the present axes
        # that are not wanted and creates the wanted axes that are not
        # present, or None when there is nothing to do.
        from ..meta import Projection

        if present == wanted:
            return None
        dropped = [p for p, a in enumerate(present) if a not in wanted]
        missing = [q for q, a in enumerate(wanted) if a not in present]
        if any(wanted[q] not in created for q in missing):
            raise _Abort
        values = [created[wanted[q]] for q in missing]
        system_in = self._subsystem(k, present)
        if system_in is None:
            system_in = unnamed_system(len(present))
        return Projection(
            dropped=dropped,
            created=missing,
            at=values,
            input=system_in,
            output=self._subsystem(k, wanted),
        )

    def _with_systems(
        self,
        k: int,
        piece: "Transformation",
        inputs: tx.List[int],
        outputs: tx.List[int],
    ) -> "Transformation":
        # Give a piece the systems of the element that it is cut from,
        # restricted to its axes, or else those of its stages, so that
        # it keeps what these systems say and agrees with its
        # neighbours. A piece takes a system that disagrees with its
        # own, or one that says more than the number of axes when it has
        # none. An identity always takes them, so that an identity that
        # changes the system keeps that change. A piece that needs no
        # system is kept as the same object, and so is a lazy inverse.
        from ..concrete import Identity
        from ..inverse import Inverse

        if piece is self.leaves[k] or isinstance(piece, Inverse):
            return piece
        identity = isinstance(piece, Identity)
        edits = {}
        sides = (
            ("input", "_input", k, inputs),
            ("output", "_output", k + 1, outputs),
        )
        for name, stored, stage, axes in sides:
            own = getattr(self.leaves[k], stored, None)
            if get_axes(own).ndim == self.counts[stage]:
                system = own.restrict(axes)
            else:
                system = self._subsystem(stage, axes)
            if system is None:
                continue
            current = getattr(piece, name)
            if (
                identity
                or systems_disagree(current, system)
                or (current is None and _informative(system))
            ):
                edits[name] = system
        if not edits:
            return piece
        if identity:
            return Identity(**edits)
        return replace(piece, **edits)

    def _finish(
        self, out: tx.List[tx.Optional["Transformation"]]
    ) -> tx.List["Transformation"]:
        # Remove the identity pieces that do not change the system, and
        # merge consecutive projections.
        from ..concrete import Identity
        from ..meta import Projection
        from .simplifiers import _merge_projections

        frozen = {id(t) for t, f in zip(self.leaves, self.frozen) if f}
        chain: tx.List[Transformation] = []
        for t in out:
            if t is None:
                continue
            if (
                isinstance(t, Identity)
                and id(t) not in frozen
                and not _changes_system(t)
            ):
                continue
            previous = chain[-1] if chain else None
            if (
                isinstance(t, Projection)
                and isinstance(previous, Projection)
                and id(t) not in frozen
                and id(previous) not in frozen
            ):
                merged = _merge_projections(previous, t)
                if merged is not None:
                    chain.pop()
                    dropped, created, at = merged
                    if dropped or created:
                        chain.append(
                            Projection(
                                dropped=dropped,
                                created=created,
                                at=at,
                                input=previous.input,
                                output=t.output,
                            )
                        )
                    continue
            chain.append(t)
        return chain


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================


def _informative(system: "CoordinateSystem") -> bool:
    # Tell whether a closed system says more than its number of axes.
    return system != unnamed_system(len(get_axes(system)))


def _changes_system(t: "Transformation") -> bool:
    # Tell whether an identity maps one closed system to another, in
    # which case it is kept, because it renames the axes.
    return (
        t.input is not None
        and t.output is not None
        and systems_disagree(t.input, t.output)
    )


def _same_chain(
    chain: tx.List["Transformation"], leaves: tx.List["Transformation"]
) -> bool:
    # Tell whether the reduction left the chain as it was, in which case
    # the original objects are kept. A projection that the reduction
    # rebuilt is the same when it drops and creates the same axes at the
    # same values.
    from ..meta import Projection

    if len(chain) != len(leaves):
        return False
    for new, old in zip(chain, leaves):
        if new is old:
            continue
        if not (isinstance(new, Projection) and isinstance(old, Projection)):
            return False
        if axis_list(new.dropped) != axis_list(old.dropped):
            return False
        if axis_list(new.created) != axis_list(old.created):
            return False
        if not np.array_equal(
            np.asarray(new.at, dtype=float),
            np.asarray(old.at, dtype=float),
            equal_nan=True,
        ):
            return False
    return True
