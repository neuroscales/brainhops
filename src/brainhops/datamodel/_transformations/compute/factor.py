"""Factoring of a chain into its axis-group normal form.

A chain often acts on groups of axes that never interact, as when each
axis is scaled separately and two axes are then swapped. The factor pass
finds these independent groups and rewrites the chain so that each group
is handled by a transformation of its own. The rewritten chain, called
the normal form, consists of an optional leading grid, an optional
projection that drops the inputs that nothing reads, one axis-preserving
factor per group of axes that transform together, an optional
permutation, and an optional projection that creates the outputs that
read no input:

```text
NF = [grid]? . [delta]? . F_1 . F_2 . ... . F_m . [Pi_perm]? . [kappa]?
```

- The grid is the leading [`CartesianField`][] of the chain, and it is
  kept as the same object.
- `delta` is a [`Projection`][] that drops the inputs of the chain that
  no output reads. It is omitted when every input is read.
- Each factor `F_i = Subspace(inner_i, A_i, A_i)` acts on the sorted axis
  positions `A_i` of its group, called its work positions. The work
  positions of different factors are disjoint, and the factors are ordered
  by their smallest position. The inner transformation of a factor is the
  part of the chain that concerns its group, composed according to the
  `compute` argument of [`factor_sequence`][]. A factor whose inner
  transformation is the identity is dropped.
- `Pi_perm` is a [`Permutation`][] that moves the work axes of each group to
  its data axes, that is, to the positions that the group occupies in the
  output of the chain. It is omitted when it is the identity.
- `kappa` is a [`Projection`][] that creates the outputs of the chain
  that read no input, at their constant values. It is omitted when every
  output reads an input.

The two projections are found by the pruning of the `prune` module,
which removes the axes that a chain does not need and which may, for
this pass only, add a projection at each end of the chain. A chain whose
two ends have different numbers of axes is therefore factored when the
projections account for the difference. Transformations are never
composed across groups, so the coordinates of independent axes are never
tiled into a grid over all of them. Only chains whose part between the
two projections has as many input axes as output axes, and whose groups
each have as many input axes as output axes, are factored. Intermediate
stages may still pass through a wider space. Other chains are returned
unchanged.

The reslicing code in the `separable` module reads the normal form to
decide how to sample each group, so this module is the only place where
axis groups are determined. Each element is cut into one piece per group
by the block restrictors of the `pruners` module. The cut keeps a lazy
inverse lazy when a block of its forward transformation gives it, and an
element that lies entirely within one group is kept as the same object.
As a result, a pair such as `Sub(warp) . Sub(warp^-1)` appears as
`[warp, warp^-1]` inside its group, and the two cancel before any
composer runs.
"""

__all__ = ["factor_sequence", "PatternCache"]

from dataclasses import dataclass, field

import numpy as np
import typing_extensions as tx

from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.errors import (
    CompositionError,
    ConversionError,
    RestrictionError,
)

from ..base import Transformation
from ..concrete import CartesianField, Identity, Permutation
from ..meta import Projection, SubspaceTransformation
from .prune import (
    close_counts,
    counts_of,
    may_prune,
    prune_chain,
    read_pattern,
    restrict_block,
    stage_counts,
)
from .simplify import ANALYTIC_FLOOR, SimplifyTable
from .utils import UNREADABLE, axis_list

if tx.TYPE_CHECKING:
    from ..modes import ModeLike
    from ..sequence import Sequence
    from .simplify import SimplifyLike


# ======================================================================
#
#                              P U B L I C
#
# ======================================================================


def factor_sequence(
    seq: "Sequence",
    policy: "SimplifyLike" = "analytic",
    *,
    compute: "ModeLike" = False,
    cache: tx.Optional["PatternCache"] = None,
) -> Transformation:
    """Rewrite a sequence into its axis-group normal form.

    This function splits a chain into groups of axes that transform
    independently, so that each group can be composed and resampled on
    its own. To find the groups, the chain is read as a graph. The nodes
    of the graph are the axes before and after each element, and an edge
    links an output axis of an element to every input axis that it
    depends on. The connected components of the graph are the groups of
    axes that transform together. Each element is then restricted to the
    groups that it touches, and the pieces of each group are composed,
    separately from the other groups, into one factor, as described in
    the module documentation. Before that, the axes that the chain does
    not need are removed, which leaves a projection that drops inputs at
    the start of the chain and a projection that creates outputs at its
    end. These two projections are the ends of the normal form, and the
    groups are found between them.

    A call performs one round of the pass. `Sequence.compute(factor=True)`
    runs such rounds between simplification and composition, until a round
    leaves the chain unchanged.

    Parameters
    ----------
    seq : Sequence
        Chain to factor. Nested sequences are flattened, and a leading
        [`CartesianField`][] is the sampling grid, kept as is.
    policy : SimplifyLike, default="analytic"
        How closely the elements may be read, with the values and the
        meaning of the policy of
        [`simplify`][brainhops.datamodel._transformations.compute.simplify.simplify].
        It decides the groups, the two ends and the simplification of
        the pieces of each group. Under the analytic policy, a matrix
        couples all its axes, and under the numeric policy, its exact
        zeros are read. A policy that leaves every transformation alone
        reads the structure only, as the analytic policy does.
    compute : ModeLike, default=False
        Kinds of transformation that may be composed inside a group, with
        the values accepted by the `mode` argument of
        [`Sequence.compute`][]. The groups never depend on it: they are
        found before it is read, and nothing is ever composed across them.
        The default, `False`, composes nothing, so each factor holds the
        sub-chain of its group, whereas `True` leaves each factor holding a
        single transformation.
    cache : PatternCache, optional
        Cache of dependency patterns, shared across the rounds of one
        computation. A fresh cache is used if it is omitted.

    Returns
    -------
    Transformation
        `seq` itself when the chain is already in normal form, which
        makes the pass idempotent, or when the chain does not factor. A
        chain does not factor when, for example, it forms a single group
        and has neither end, the part between its two ends differs in
        dimension, or one of its elements cannot be read or restricted.
        Otherwise, the result is a sequence
        `[grid?, delta?, F_1, ..., F_m, Pi_perm?, kappa?]` with the
        endpoints of `seq`, or an [`Identity`][] when nothing remains.
        An element that lies entirely within one group is kept as the
        same object, so that a transformation still cancels with its
        lazy inverse, and no lazy inverse is materialized.

    Examples
    --------
    A diagonal scaling followed by an axis swap gives one factor per axis
    and a trailing permutation. Each factor holds the sub-chain of its own
    axis, because nothing is composed by default.

    ```pycon
    >>> import numpy as np
    >>> from brainhops.datamodel.transformations import (
    ...     CartesianField, Permutation, Scaling, Sequence,
    ... )
    >>> seq = Sequence([
    ...     CartesianField(shape=(4, 5, 6)),
    ...     Scaling(scale=np.array([2.0, 3.0, 1.0])),
    ...     Permutation(permutation=np.array([1, 0, 2])),
    ... ])
    >>> nf = factor_sequence(seq)
    >>> grid, *body = nf.transformations
    >>> [f.input_axes.tolist() for f in body[:-1]]
    [[0], [1], [2]]
    >>> type(body[-1]).__name__
    'Permutation'
    ```

    With `compute=True`, each group is composed into a single
    transformation, and, under the numeric policy, the factor of the
    last axis, whose scale is one, is dropped because its group composes
    to the identity.

    ```pycon
    >>> nf = factor_sequence(seq, compute=True, policy="numeric")
    >>> grid, *body = nf.transformations
    >>> [type(t).__name__ for t in body]
    ['SubspaceTransformation', 'SubspaceTransformation', 'Permutation']
    >>> [f.input_axes.tolist() for f in body[:2]]
    [[0], [1]]
    ```
    """
    from ..sequence import _unnest

    if cache is None:
        cache = PatternCache()
    table = SimplifyTable.from_like(policy)
    if table.is_noop():
        table = ANALYTIC_FLOOR

    xforms = _unnest(seq.transformations)
    if not xforms:
        return seq

    grid: tx.Optional[CartesianField] = None
    body = xforms
    if isinstance(xforms[0], CartesianField):
        grid = xforms[0]
        body = xforms[1:]
    if not body:
        return seq
    if any(isinstance(t, CartesianField) for t in body):
        # Grids inside the chain are removed before this pass runs, so a
        # grid that remains here is not supported.
        return seq

    n_in = _chain_ndim_in(grid, body)
    if n_in is None:
        return seq

    head, middle, tail = _split_ends(body, table)
    if head is not None:
        n_in -= len(axis_list(head.dropped))

    stages = _build_stages(middle, n_in, cache, table)
    if stages is None:
        return seq

    groups = _partition(stages, n_in)
    if groups is None:
        return seq
    if len(groups) <= 1 and head is None and tail is None:
        return seq

    try:
        factors = _build_factors(groups, stages, compute, table)
    except RestrictionError:
        # When an element cannot be cut into group pieces safely, the chain
        # is left unfactored rather than losing one of the pieces.
        return seq
    if factors is None:
        return seq
    perm = _build_perm(groups, n_in)
    if head is not None and tail is not None and not factors and not perm:
        # Two ends with nothing between them merge into one projection
        # when the chain is simplified, so they are not a normal form.
        return seq

    if _same_structure(body, head, factors, perm, tail):
        # The fixpoint loop stops when a round leaves every element as the
        # same object, so returning `seq` itself lets the loop end.
        return seq

    elements: tx.List[Transformation] = []
    if grid is not None:
        elements.append(grid)
    if head is not None:
        elements.append(head)
    elements.extend(factors)
    if perm is not None:
        elements.append(perm)
    if tail is not None:
        elements.append(tail)
    if not elements:
        return Identity(input=seq.input, output=seq.output)
    return seq.to(transformations=elements)


@dataclass
class PatternCache:
    """Cache of the dependency patterns of chain elements.

    A dependency pattern records which input axes each output axis of an
    element depends on. It is a boolean `(n_out, n_in)` matrix whose entry
    `(i, j)` is true when output axis `i` depends on input axis `j`. The
    cache looks patterns up by the `id` of the element. Each cached element
    is also kept in `keepalive`, so that its `id` cannot be reused by another
    object while the pattern is cached. A cache is meant to last for one
    computation.
    """

    patterns: tx.Dict[int, np.ndarray] = field(default_factory=dict)
    keepalive: tx.List[Transformation] = field(default_factory=list)

    def pattern(
        self,
        element: Transformation,
        closed: Transformation,
        policy: SimplifyTable,
    ) -> np.ndarray:
        """Return the pattern of `element`, read from `closed`.

        `closed` is `element` with closed systems. An element whose
        pattern cannot be read links all its axes.
        """
        key = id(element)
        if key in self.patterns:
            return self.patterns[key]
        n_in, n_out = counts_of(closed)
        pattern = read_pattern(closed, policy)
        if pattern is UNREADABLE or np.shape(pattern) != (n_out, n_in):
            pattern = np.ones((n_out, n_in), dtype=bool)
        pattern = np.asarray(pattern, dtype=bool)
        self.patterns[key] = pattern
        self.keepalive.append(element)
        return pattern


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================

# ----------------------------------------------------------------------
#   DATA STRUCTURES
# ----------------------------------------------------------------------


@dataclass
class _Stage:
    # A stage is one element of the chain together with its dependency
    # pattern, whose entry (i, j) is true when output axis i depends on
    # input axis j.
    # `closed` is the element with closed systems, which the restrictors
    # receive.
    element: Transformation
    closed: Transformation
    pattern: np.ndarray


@dataclass
class _Group:
    # A group holds axes that transform together, and it is found as a
    # connected component of the graph whose nodes are (stage, axis) pairs.
    # `axes` lists the sorted work positions that the factor acts on.
    # `grid_axes` (G) and `data_axes` (D) list the positions of the group in
    # the input and in the output of the chain. `per_stage` maps the index
    # of each stage, where 0 is the chain input, to the axes of the group.
    axes: tx.List[int]
    grid_axes: tx.List[int]
    data_axes: tx.List[int]
    per_stage: tx.Dict[int, tx.List[int]] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# ----------------------------------------------------------------------
#   DEPENDENCY PATTERNS
# ----------------------------------------------------------------------
# The patterns are read by the pattern readers of the `pruners` module,
# and an element whose pattern cannot be read links all its axes. The
# partition follows the policy: under the analytic policy a matrix
# couples all its axes, and under the numeric policy its exact zeros are
# read. For the lazy inverse of a matrix transformation, the numeric
# policy reads the coefficients of the inverse, which the forward
# transformation computes and caches while the inverse stays lazy.
# Otherwise, the reader returns the closure of the forward pattern, and
# no coefficient is read. Grouping only needs to know which axes are
# linked, whatever the direction of the link, and the closure links
# every pair of axes that the inverse links.


# ----------------------------------------------------------------------
#   PARTITION
# ----------------------------------------------------------------------


def _chain_ndim_in(
    grid: tx.Optional[CartesianField], body: tx.List[Transformation]
) -> tx.Optional[int]:
    if grid is not None and grid.shape is not None:
        return len(grid.shape)
    counts = stage_counts(body)
    return None if counts is None else counts[0]


def _build_stages(
    body: tx.List[Transformation],
    n_in: int,
    cache: PatternCache,
    policy: tx.Optional[SimplifyTable] = None,
) -> tx.Optional[tx.List[_Stage]]:
    # Intermediate stages may change dimension, for example by embedding
    # into a wider space and projecting back, but the two ends of the
    # part of the chain that is grouped may not.
    policy = ANALYTIC_FLOOR if policy is None else policy
    counts = stage_counts(body, start=n_in)
    if counts is None or counts[-1] != n_in:
        return None
    stages: tx.List[_Stage] = []
    for k, element in enumerate(body):
        closed = close_counts(element, counts[k], counts[k + 1])
        if closed is None:
            return None
        pattern = cache.pattern(element, closed, policy)
        stages.append(_Stage(element=element, closed=closed, pattern=pattern))
    return stages


def _stage_dims(stages: tx.List[_Stage], n_axes: int) -> tx.List[int]:
    return [n_axes] + [s.pattern.shape[0] for s in stages]


def _partition(
    stages: tx.List[_Stage], n_axes: int
) -> tx.Optional[tx.List[_Group]]:
    # Find the groups as the connected components of a graph. Its nodes are
    # (stage, axis) pairs, and each true entry of a stage pattern links an
    # output axis of that stage to an input axis of the previous stage.
    dims = _stage_dims(stages, n_axes)
    offsets = np.cumsum([0] + dims).tolist()
    uf = _UnionFind(offsets[-1])

    def node(stage: int, axis: int) -> int:
        return offsets[stage] + axis

    for stage, s in enumerate(stages, start=1):
        for i, j in zip(*np.nonzero(s.pattern)):
            uf.union(node(stage, int(i)), node(stage - 1, int(j)))

    roots: tx.Dict[int, tx.Dict[int, tx.List[int]]] = {}
    for stage, dim in enumerate(dims):
        for axis in range(dim):
            root = uf.find(node(stage, axis))
            roots.setdefault(root, {}).setdefault(stage, []).append(axis)

    last = len(dims) - 1
    groups: tx.List[_Group] = []
    for comp in roots.values():
        grid_axes = sorted(comp.get(0, []))
        data_axes = sorted(comp.get(last, []))
        if not grid_axes or not data_axes:
            # A group that has no axes at the start or at the end of the
            # chain, such as an axis that is created and later discarded,
            # cannot become a factor.
            return None
        if len(grid_axes) != len(data_axes):
            return None
        groups.append(
            _Group(
                axes=grid_axes,
                grid_axes=grid_axes,
                data_axes=data_axes,
                per_stage={s: sorted(a) for s, a in comp.items()},
            )
        )

    covered_grid = sorted(a for g in groups for a in g.grid_axes)
    covered_data = sorted(a for g in groups for a in g.data_axes)
    if covered_grid != list(range(n_axes)):
        return None
    if covered_data != list(range(n_axes)):
        return None
    groups.sort(key=lambda g: g.axes[0])
    return groups


# ----------------------------------------------------------------------
#   NORMAL FORM
# ----------------------------------------------------------------------


def _restrict_group(
    group: _Group,
    stages: tx.List[_Stage],
    policy: tx.Optional[SimplifyTable] = None,
) -> tx.List[Transformation]:
    # Cut one piece out of each element that the group touches, in chain
    # order. A group is decoupled from the other groups, so its block of
    # each element is closed. An element that lies entirely within the
    # group is kept as the same object, and identity pieces are omitted.
    policy = ANALYTIC_FLOOR if policy is None else policy
    sub: tx.List[Transformation] = []
    for stage, s in enumerate(stages, start=1):
        rows = group.per_stage.get(stage, [])
        cols = group.per_stage.get(stage - 1, [])
        if not rows or not cols:
            continue
        no, ni = s.pattern.shape
        if rows == list(range(no)) and cols == list(range(ni)):
            piece = s.element
        else:
            piece = restrict_block(s.closed, cols, rows, policy)
            if piece is None:
                raise RestrictionError(
                    f"Cannot cut a {type(s.element).__name__} into the "
                    f"groups of axes of the chain"
                )
        if not isinstance(piece, Identity):
            sub.append(piece)
    return sub


def _inner_is_identity(
    inner: tx.Optional[Transformation], policy: SimplifyTable
) -> bool:
    # An inner transformation is checked by value only when the policy
    # reads its values, so that, under the numeric policy, a piece that
    # turns out to be the identity, such as Scaling([1]), is dropped. A
    # field and a chain are checked by structure, because checking them
    # by value would sample the field or compose the chain.
    from ..sequence import Sequence, _interpolates

    if inner is None:
        return True
    if _interpolates(inner) or isinstance(inner, Sequence):
        return inner.is_identity(compute=False)
    numeric = policy.resolve(inner) is SimplifyPolicy.numeric
    return inner.is_identity(compute=numeric)


def _build_factors(
    groups: tx.List[_Group],
    stages: tx.List[_Stage],
    compute: "ModeLike",
    policy: SimplifyTable,
) -> tx.Optional[tx.List[SubspaceTransformation]]:
    from ..sequence import Sequence

    factors: tx.List[SubspaceTransformation] = []
    for group in groups:
        sub = _restrict_group(group, stages, policy)
        if not sub:
            continue
        # The pieces are composed under the kinds that the caller admits
        # and simplified under the policy, so a restricted
        # transformation and its inverse cancel before anything else is
        # composed.
        try:
            inner = Sequence(transformations=sub).compute(
                compute, simplify=policy, factor=False
            )
        except (ConversionError, CompositionError):
            # Factoring is an optimization, so returning the chain
            # unchanged is always sound. Restricted pieces can fail to
            # compose even where the whole chain composes, for example
            # when a subspace piece with an unknown axis count sits next
            # to an affine piece.
            return None
        if isinstance(inner, Identity) or _inner_is_identity(inner, policy):
            continue
        axes = np.asarray(group.axes, dtype=int)
        factors.append(
            SubspaceTransformation(
                transformation=inner,
                input_axes=axes,
                output_axes=axes,
            )
        )
    return factors


def _build_perm(
    groups: tx.List[_Group], n_axes: int
) -> tx.Optional[Permutation]:
    # Build the permutation in which output data axis D[j] reads work axis
    # A[j], or return None when that permutation is the identity.
    perm = list(range(n_axes))
    identity = True
    for group in groups:
        for a, d in zip(group.axes, group.data_axes):
            perm[d] = a
            if a != d:
                identity = False
    if identity:
        return None
    return Permutation(permutation=np.asarray(perm, dtype=int))


def _same_structure(
    body: tx.List[Transformation],
    head: tx.Optional[Projection],
    factors: tx.List[SubspaceTransformation],
    perm: tx.Optional[Permutation],
    tail: tx.Optional[Projection],
) -> bool:
    # Decide whether the chain already has the structure of the normal form.
    # Only the axis lists, the permutation and the ends are compared,
    # because the inner transformations are assumed to be stable across
    # rounds.
    expected: tx.List[Transformation] = []
    if head is not None:
        expected.append(head)
    expected.extend(factors)
    if perm is not None:
        expected.append(perm)
    if tail is not None:
        expected.append(tail)
    if len(body) != len(expected):
        return False
    for got, want in zip(body, expected):
        if isinstance(want, Projection):
            if not _same_projection(got, want):
                return False
        elif isinstance(want, Permutation):
            if not isinstance(got, Permutation) or got.permutation is None:
                return False
            if list(got.permutation) != list(want.permutation):
                return False
        else:
            if not isinstance(got, SubspaceTransformation):
                return False
            if axis_list(got.input_axes) != axis_list(want.input_axes):
                return False
            if axis_list(got.output_axes) != axis_list(want.output_axes):
                return False
    return True


def _same_projection(got: Transformation, want: Projection) -> bool:
    # Two projections are the same end when they drop and create the
    # same axes at the same values.
    if not isinstance(got, Projection):
        return False
    if axis_list(got.dropped) != axis_list(want.dropped):
        return False
    if axis_list(got.created) != axis_list(want.created):
        return False
    return bool(
        np.array_equal(
            np.asarray(got.at, dtype=float),
            np.asarray(want.at, dtype=float),
            equal_nan=True,
        )
    )


# ----------------------------------------------------------------------
#   ENDS
# ----------------------------------------------------------------------


def _split_ends(
    body: tx.List[Transformation], policy: SimplifyTable
) -> tx.Tuple[
    tx.Optional[Projection],
    tx.List[Transformation],
    tx.Optional[Projection],
]:
    # Prune the chain, allowing it to grow by its two ends, and return
    # the projection that drops the unread inputs at its start, the
    # elements in between and the projection that creates the constant
    # outputs at its end. The pairs that cancel at no cost collapse
    # first, so that a lazy inverse is not cut, and maybe
    # computed, before it meets its forward transformation.
    from .simplifiers import _settle

    chain = list(body)
    if may_prune(chain, policy):
        chain = _settle(chain, policy)
        pruned = prune_chain(chain, policy, ends=True)
        if pruned is not None:
            chain = pruned
    head = None
    if chain and _only_drops(chain[0]):
        head = chain.pop(0)
    tail = None
    if chain and _only_creates(chain[-1]):
        tail = chain.pop()
    return head, chain, tail


def _only_drops(t: Transformation) -> bool:
    return (
        isinstance(t, Projection)
        and bool(axis_list(t.dropped))
        and not axis_list(t.created)
    )


def _only_creates(t: Transformation) -> bool:
    return (
        isinstance(t, Projection)
        and bool(axis_list(t.created))
        and not axis_list(t.dropped)
    )
