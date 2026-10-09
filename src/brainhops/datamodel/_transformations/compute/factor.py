"""Factoring of a chain into its axis-group normal form.

A chain often acts on groups of axes that never interact, as when each axis
is scaled separately and two axes are then swapped. The factor pass finds
these independent groups and rewrites the chain so that each group is
handled by a transformation of its own. The rewritten chain, called the
normal form, consists of an optional leading grid, one axis-preserving
factor per group of axes that transform together, and an optional trailing
permutation:

```text
NF = [grid]? . F_1 . F_2 . ... . F_m . [Pi_perm]?
```

- The grid is the leading [`CartesianField`][] of the chain, and it is
  kept as the same object.
- Each factor `F_i = Subspace(inner_i, A_i, A_i)` acts on the sorted axis
  positions `A_i` of its group, called its work positions. The work
  positions of different factors are disjoint, and the factors are ordered
  by their smallest position. The inner transformation of a factor is the
  part of the chain that concerns its group, composed according to the `mode`
  argument of [`factor_sequence`][]. A factor whose inner transformation
  is the identity is dropped.
- `Pi_perm` is a [`Permutation`][] that moves the work axes of each group to
  its data axes, that is, to the positions that the group occupies in the
  output of the chain. It is omitted when it is the identity.

Transformations are never composed across groups, so the coordinates of
independent axes are never tiled into a grid over all of them. Only chains
whose two ends have the same number of axes, and whose groups each have as
many input axes as output axes, are factored. Intermediate stages may still
pass through a wider space. Other chains are returned unchanged.

The reslicing code in the `separable` module reads the normal form to decide
how to sample each group, so this module is the only place where axis groups
are determined. Each element is cut into one piece per group by the
`restrict` function. Restriction keeps a lazy inverse lazy, and
an element that lies entirely within one group is kept as the same object.
As a result, a pair such as `Sub(warp) . Sub(warp^-1)` appears as
`[warp, warp^-1]` inside its group, and the two cancel before any composer
runs.
"""

__all__ = ["factor_sequence", "PatternCache"]

from dataclasses import dataclass, field

import numpy as np
import typing_extensions as tx

from brainhops.errors import (
    CompositionError,
    ConversionError,
    RestrictionError,
)

from ..base import Transformation
from ..concrete import CartesianField, Identity, Permutation
from ..meta import SubspaceTransformation
from .restrict import restrict
from .utils import UNREADABLE, affine_matrix, axis_list

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
    mode: "ModeLike" = True,
    *,
    simplify: "SimplifyLike" = "analytic",
    cache: tx.Optional["PatternCache"] = None,
) -> Transformation:
    """Rewrite a sequence into its axis-group normal form.

    This function splits a chain into groups of axes that transform
    independently, so that each group can be composed and resampled on its
    own. To find the groups, the chain is read as a graph. The nodes of the
    graph are the axes before and after each element, and an edge links an
    output axis of an element to every input axis that it depends on. The
    connected components of the graph are the groups of axes that transform
    together. Each element is then restricted to the groups that it touches,
    and the pieces of each group are composed, separately from the other
    groups, into one factor, as described in the module documentation.

    A call performs one round of the pass. `Sequence.compute(factor=True)`
    runs such rounds between simplification and composition, until a round
    leaves the chain unchanged.

    Parameters
    ----------
    seq : Sequence
        Chain to factor. Nested sequences are flattened, and a leading
        [`CartesianField`][] is the sampling grid, kept as is.
    mode : ModeLike, default=True
        Kinds of transformation allowed to compose inside a group, as in
        [`Sequence.compute`][].
    simplify : SimplifyLike, default="analytic"
        Simplification policy for composing the pieces of each group.
    cache : PatternCache, optional
        Cache of dependency patterns, shared across the rounds of one
        computation. A fresh cache is used if it is omitted.

    Returns
    -------
    Transformation
        `seq` itself when the chain is already in normal form, which makes
        the pass idempotent, or when the chain does not factor. A chain does
        not factor when, for example, it forms a single group, its two ends
        differ in dimension, or one of its elements cannot be read or
        restricted. Otherwise, the result is a sequence
        `[grid?, F_1, ..., F_m, Pi_perm?]` with the endpoints of `seq`, or an
        [`Identity`][] when nothing remains. An element that lies entirely
        within one group is kept as the same object, so that a transformation
        still cancels with its lazy inverse, and no lazy inverse is
        materialized.

    Examples
    --------
    A diagonal scaling followed by an axis swap gives one factor per scaled
    axis and a trailing permutation. The last axis has a unit scale, so its
    factor is dropped.

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
    >>> [type(t).__name__ for t in body]
    ['SubspaceTransformation', 'SubspaceTransformation', 'Permutation']
    >>> [f.input_axes.tolist() for f in body[:2]]
    [[0], [1]]
    ```
    """
    from ..sequence import _unnest

    if cache is None:
        cache = PatternCache()

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

    stages = _build_stages(body, n_in, cache)
    if stages is None:
        return seq

    groups = _partition(stages, n_in)
    if groups is None:
        return seq
    if len(groups) <= 1:
        return seq

    try:
        factors = _build_factors(groups, stages, mode, simplify)
    except RestrictionError:
        # When an element cannot be cut into group pieces safely, the chain
        # is left unfactored rather than losing one of the pieces.
        return seq
    if factors is None:
        return seq
    perm = _build_perm(groups, n_in)

    if _same_structure(body, factors, perm):
        # The fixpoint loop stops when a round leaves every element as the
        # same object, so returning `seq` itself lets the loop end.
        return seq

    elements: tx.List[Transformation] = []
    if grid is not None:
        elements.append(grid)
    elements.extend(factors)
    if perm is not None:
        elements.append(perm)
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

    patterns: tx.Dict[int, tx.Optional[np.ndarray]] = field(
        default_factory=dict
    )
    keepalive: tx.List[Transformation] = field(default_factory=list)

    def pattern(
        self, element: Transformation, ni: int, no: int
    ) -> tx.Optional[np.ndarray]:
        """Return the pattern of `element`, or None if it is unreadable."""
        key = id(element)
        if key in self.patterns:
            return self.patterns[key]
        pat = _read_pattern(element, ni, no)
        self.patterns[key] = pat
        self.keepalive.append(element)
        return pat


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
    element: Transformation
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
# Reading a pattern never materializes a lazy inverse, because the pattern of
# an inverse is the transpose of the pattern of its forward transformation. An
# element whose affine matrix cannot be read (UNREADABLE) leaves the chain
# unfactored, and it is never mistaken for the identity.


def _element_ndim(
    element: Transformation, ndim: int
) -> tx.Optional[tx.Tuple[int, int]]:
    # Return the input and output dimensionality of an element, or None if
    # the element cannot be read.
    from ..inverse import Inverse
    from ..sequence import _interpolates

    if isinstance(element, (SubspaceTransformation, Identity)):
        return ndim, ndim
    if isinstance(element, Inverse):
        forward = element.forward
        if forward is None:
            return ndim, ndim
        dims = _element_ndim(forward, ndim)
        if dims is None:
            return None
        ni, no = dims
        return no, ni
    if _interpolates(element):
        return ndim, ndim
    matrix = affine_matrix(element)
    if matrix is UNREADABLE:
        return None
    if matrix is None:
        return ndim, ndim
    no, cols = matrix.shape
    return cols - 1, no


def _read_pattern(
    element: Transformation, ni: int, no: int
) -> tx.Optional[np.ndarray]:
    from ..inverse import Inverse
    from ..sequence import Sequence, _interpolates

    if isinstance(element, Sequence):
        # The pattern of a nested sub-chain, such as the partially composed
        # inner transformation of a subspace, is the product of the patterns
        # of its members along the chain.
        acc = np.eye(ni, dtype=bool)
        ndim = ni
        for member in element.transformations or []:
            dims = _element_ndim(member, ndim)
            if dims is None or dims[0] != ndim:
                return None
            part = _read_pattern(member, *dims)
            if part is None or part.shape != dims[::-1]:
                return None
            acc = (part.astype(int) @ acc.astype(int)) > 0
            ndim = dims[1]
        return acc if acc.shape == (no, ni) else None
    if isinstance(element, SubspaceTransformation):
        if ni != no:
            return None
        return _subspace_pattern(element, ni)
    if isinstance(element, Inverse):
        forward = element.forward
        if forward is None:
            return np.eye(ni, dtype=bool) if ni == no else None
        fpat = _read_pattern(forward, no, ni)
        if fpat is None or fpat.shape != (ni, no):
            return None
        return fpat.T
    if _interpolates(element):
        return np.ones((no, ni), dtype=bool) if ni == no else None
    if isinstance(element, Identity):
        return np.eye(ni, dtype=bool) if ni == no else None
    return _read_pattern_affine(element, ni, no)


def _subspace_pattern(
    element: SubspaceTransformation, ndim: int
) -> tx.Optional[np.ndarray]:
    from ..sequence import _interpolates

    inner = element.transformation
    in_axes = axis_list(element.input_axes)
    out_axes = axis_list(element.output_axes)
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            return None
        return _read_pattern_affine(element, ndim, ndim)
    ki, ko = len(in_axes), len(out_axes)
    if inner is None and ki == ko:
        # A subspace without an inner transformation is the identity on its
        # axes, so input_axes[k] feeds output_axes[k] and the other axes pass
        # through in order. When the input and output axes differ, the
        # subspace is a reindexing, which every reader interprets the same
        # way. The affine converter and both subspace composers have agreed
        # on this interpretation since #110.
        inner_dep = np.eye(ko, dtype=bool)
    elif interpolates:
        inner_dep = np.ones((ko, ki), dtype=bool)
    else:
        inner_dep = None
        if ki == ko:
            candidate = _read_pattern(inner, ki, ko)
            if candidate is not None and candidate.shape == (ko, ki):
                inner_dep = candidate
        if inner_dep is None:
            return _read_pattern_affine(element, ndim, ndim)
    dep = np.zeros((ndim, ndim), dtype=bool)
    for a, o in enumerate(out_axes):
        for b, i in enumerate(in_axes):
            if inner_dep[a, b]:
                dep[o, i] = True
    pass_in = [i for i in range(ndim) if i not in set(in_axes)]
    pass_out = [o for o in range(ndim) if o not in set(out_axes)]
    for o, i in zip(pass_out, pass_in):
        dep[o, i] = True
    return dep


def _read_pattern_affine(
    element: Transformation, ni: int, no: int
) -> tx.Optional[np.ndarray]:
    matrix = affine_matrix(element)
    if matrix is UNREADABLE:
        return None
    if matrix is None:
        return np.eye(ni, dtype=bool) if ni == no else None
    if matrix.shape[0] != no or matrix.shape[1] - 1 != ni:
        return None
    return matrix[:, :-1] != 0


# ----------------------------------------------------------------------
#   PARTITION
# ----------------------------------------------------------------------


def _chain_ndim_in(
    grid: tx.Optional[CartesianField], body: tx.List[Transformation]
) -> tx.Optional[int]:
    if grid is not None and grid.shape is not None:
        return len(grid.shape)
    for element in body:
        dims = _element_ndim(element, -1)
        if dims is not None and dims[0] >= 0:
            return dims[0]
    return None


def _build_stages(
    body: tx.List[Transformation], n_in: int, cache: PatternCache
) -> tx.Optional[tx.List[_Stage]]:
    # Intermediate stages may change dimension, for example by embedding into
    # a wider space and projecting back, but the two ends of the chain may
    # not.
    stages: tx.List[_Stage] = []
    ndim = n_in
    for element in body:
        dims = _element_ndim(element, ndim)
        if dims is None or dims[0] != ndim:
            return None
        ni, no = dims
        pat = cache.pattern(element, ni, no)
        if pat is None or pat.shape != (no, ni):
            return None
        stages.append(_Stage(element=element, pattern=pat))
        ndim = no
    if ndim != n_in:
        # Created or dropped axes are not supported yet.
        return None
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
    group: _Group, stages: tx.List[_Stage]
) -> tx.List[Transformation]:
    # Cut one piece out of each element that the group touches, in chain
    # order. Identity pieces, for which `restrict` returns None, are omitted.
    sub: tx.List[Transformation] = []
    for stage, s in enumerate(stages, start=1):
        rows = group.per_stage.get(stage, [])
        cols = group.per_stage.get(stage - 1, [])
        if not rows or not cols:
            continue
        no, ni = s.pattern.shape
        piece = restrict(s.element, rows, cols, ni, no)
        if piece is not None:
            sub.append(piece)
    return sub


def _inner_is_identity(inner: tx.Optional[Transformation]) -> bool:
    # An inner transformation that does not interpolate is checked by value,
    # so a restriction that turns out to be the identity, such as
    # Scaling([1]), is dropped. A field is checked without computing its
    # values, so it is never dropped because of them.
    from ..sequence import _interpolates

    if inner is None:
        return True
    if _interpolates(inner):
        return inner.is_identity(compute=False)
    return inner.is_identity(compute=True)


def _build_factors(
    groups: tx.List[_Group],
    stages: tx.List[_Stage],
    mode: "ModeLike",
    simplify: "SimplifyLike",
) -> tx.Optional[tx.List[SubspaceTransformation]]:
    from ..sequence import Sequence

    factors: tx.List[SubspaceTransformation] = []
    for group in groups:
        sub = _restrict_group(group, stages)
        if not sub:
            continue
        # The pieces are composed under the mode and the simplification
        # policy of the caller, so a restricted transformation and its
        # inverse cancel before anything else is composed.
        try:
            inner = Sequence(transformations=sub).compute(
                mode, simplify=simplify, factor=False
            )
        except (ConversionError, CompositionError):
            # Factoring is an optimization, so returning the chain unchanged is
            # always sound. Restricted pieces can fail to compose even where
            # the whole chain composes, for example when a subspace piece
            # with an unknown axis count sits next to an affine piece.
            return None
        if _inner_is_identity(inner):
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
    factors: tx.List[SubspaceTransformation],
    perm: tx.Optional[Permutation],
) -> bool:
    # Decide whether the chain already has the structure of the normal form.
    # Only the axis lists and the permutation are compared, because the inner
    # transformations are assumed to be stable across rounds.
    expected: tx.List[Transformation] = list(factors)
    if perm is not None:
        expected.append(perm)
    if len(body) != len(expected):
        return False
    for got, want in zip(body, expected):
        if isinstance(want, Permutation):
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
