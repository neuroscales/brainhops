"""Factoring of a chain into its axis-group normal form.

The factor pass rewrites a chain into an optional leading grid, one
axis-preserving factor per group of axes that transform together, and an
optional trailing permutation:

```text
NF = [grid]? . F_1 . F_2 . ... . F_m . [Pi_perm]?
```

- The grid is the leading [`CartesianField`][], kept as the same object.
- Each factor `F_i = Subspace(inner_i, A_i, A_i)` acts on sorted work
  positions `A_i`, which are pairwise disjoint and ordered by their smallest
  element. Its inner transformation is the restricted sub-chain of the
  group, composed under `mode`. A factor whose inner transformation is the
  identity is dropped.
- `Pi_perm` is a [`Permutation`][] that scatters the work axes of each group
  to its data axes. It is omitted when it is the identity.

Nothing is composed across groups, so the coordinates of independent axes
are never tiled. Only dimension-preserving chains whose groups are all
square are factored, although intermediate stages may pass through a wider
space. Other chains are returned unchanged.

The reslice executor in the `separable` module reads the normal form to
decide how to sample each group, so this module is the single place where
axis groups are determined. Elements are cut into group pieces by the
dispatched `restrict` operation, which keeps a lazy inverse lazy and keeps
an element held whole as the same object. A restricted pair
`Sub(warp) . Sub(warp^-1)` therefore meets as `[warp, warp^-1]` inside its
group and cancels before any composer runs.
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

    The chain is read as a graph whose nodes are the axes at every stage and
    whose edges are the dependencies of each element. Its connected components
    are the groups of axes that transform together. Each element is restricted
    to the groups it touches, and the pieces of each group are composed alone
    into one factor, as described in the module documentation. A call performs
    one round of the pass, which `Sequence.compute(factor=True)` runs between
    simplification and composition until a fixpoint is reached.

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
        Memo of dependency patterns, shared across the rounds of one
        computation. A fresh cache is used if it is omitted.

    Returns
    -------
    Transformation
        `seq` itself when the chain does not factor (a single group, ends that
        differ in dimension, an element that cannot be read or restricted) or
        is already in normal form, which makes the pass idempotent. Otherwise,
        a sequence `[grid?, F_1, ..., F_m, Pi_perm?]` with the endpoints of
        `seq`, or an [`Identity`][] when nothing remains. Elements held whole
        by a group stay the same objects, so that a transformation still
        cancels with its lazy inverse, and no lazy inverse is materialized.

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
        # Interior grids are dropped before this pass, so a grid here is
        # unsupported.
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
        # An element that cannot be soundly cut into group pieces leaves the
        # chain unfactored, instead of losing a piece.
        return seq
    if factors is None:
        return seq
    perm = _build_perm(groups, n_in)

    if _same_structure(body, factors, perm):
        # Returning the same objects lets the identity-based exit test of the
        # fixpoint loop converge.
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
    """Memo of element dependency patterns.

    A pattern is a boolean `(n_out, n_in)` matrix whose entry `(i, j)` is true
    when output axis `i` depends on input axis `j`. Patterns are keyed by the
    `id` of the element, which is kept in `keepalive` so that the `id` is not
    reused while it is cached. A cache lives for one computation.
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
    element: Transformation
    pattern: np.ndarray  # (N_out, N_in): output i depends on input j


@dataclass
class _Group:
    # Connected component of the (stage, work axis) graph.
    axes: tx.List[int]  # sorted work positions acted on
    grid_axes: tx.List[int]  # G: positions in the chain input
    data_axes: tx.List[int]  # D: positions in the chain output
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
# Reading never materializes a lazy inverse, whose pattern is the transpose of
# the pattern of its forward transformation. An element without an affine
# reading (UNREADABLE) leaves the chain unfactored and is never read as the
# identity.


def _element_ndim(
    element: Transformation, ndim: int
) -> tx.Optional[tx.Tuple[int, int]]:
    # (input, output) dimensionality, or None if unreadable.
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
        # A nested sub-chain, such as a partially composed subspace inner, has
        # the product of the member patterns along the chain.
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
        # No inner transformation is the identity on the acted axes:
        # input_axes[k] feeds output_axes[k], the others pass through in order.
        # Differing axes make it a reindex, which all readers agree on.
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
    # Intermediate stages may change dimension (embedding then projection), but
    # the ends may not.
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
    # Components of the graph whose nodes are (stage, work axis) pairs and
    # whose edges are the true entries of the stage patterns.
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
            # A source or sink group is not a per-axis step.
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
    # One piece per touched element, in chain order; identity pieces are
    # omitted.
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
    # A non-interpolating inner is checked by value, so restrictions that land
    # on the identity, such as Scaling([1]), are dropped. A field is never
    # dropped by value.
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
        # The caller's mode and policy apply, so a restricted transformation
        # and inverse pair cancels first.
        try:
            inner = Sequence(transformations=sub).compute(
                mode, simplify=simplify, factor=False
            )
        except (ConversionError, CompositionError):
            # Factoring is an optimization, so returning the chain unchanged is
            # always sound. Restricted pieces can fail to compose where the
            # whole chain does not, for example a subspace piece without an
            # axis count next to an affine piece.
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
    # Output data axis D[j] reads work axis A[j].
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
    # Only the axis lists and the permutation are compared; inner
    # transformations are trusted to be stable across rounds.
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
