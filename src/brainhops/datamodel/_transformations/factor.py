"""Factor a transformation chain into its axis-group normal form.

The factor pass rewrites a sequence into a *normal form* (NF): a leading
grid (when present), one axis-preserving factor per group of axes that
transform together, and a trailing reindex permutation. Each factor acts on
a group of work axes and leaves every other axis untouched, so nothing is
ever composed across groups; the coordinates of independent axes are never
tiled together.

    NF = [grid]? . F_1 . F_2 . ... . F_m . [Pi_perm]?

* ``grid`` -- the leading ``CartesianField``, kept as the same object.
* ``F_i = Subspace(inner_i, A_i, A_i)`` -- an axis-preserving factor over
  the group's sorted work positions ``A_i`` (pairwise disjoint, ordered by
  ``min(A_i)``). ``inner_i`` is the group's restricted sub-chain composed
  under ``mode``. A factor whose inner is the identity is dropped, so a pure
  permutation chain reduces to ``[grid, Pi_perm]``.
* ``Pi_perm`` -- a ``Permutation`` that scatters each group's work axes to
  its data axes; omitted when it is the identity.

Scope. Only *dimension-preserving* chains, whose groups are all square
(``|G| == |D|``), are factored. An intermediate stage may have more or fewer
axes than the chain's ends (an embedding into a wider space, a warp there,
and a projection back), in which case a group's inner passes through that
wider space. A chain whose ends differ (a rectangular affine, a projection,
a ``None``-index broadcast), or that has a mixed / rank-deficient group, is
left unfactored and returned unchanged. The embedding / drop projections
(``E`` / ``Pi_drop``) of the full normal form are a follow-up.

The reslice executor (the `separable` module) reads this normal form to
decide how to sample the data of each group, so it is the one place where
the axis groups of a transformation are worked out.

The pass is *idempotent* and *identity-preserving*: a sequence already in
normal form is returned unchanged (same objects), and a leaf the pass does
not split keeps its identity, so the adjacent-inverse cancellation (the
pair simplifiers, which link a transform to its lazy inverse by object
identity) still sees it.

Reading a chain never materializes a lazy inverse: an `Inverse` is read as
the transpose of its forward's dependency pattern. How each element is cut
into the pieces of each group is the dispatched `restrict` operation (see
the `restrict` module), which keeps each piece's type, restricts a lazy
inverse by restricting its forward and re-inverting lazily, and keeps an
element a group holds whole as the same object. Composition inside a group
goes through the ordinary [`Sequence.compute`][], so a restricted
``Sub(warp) . Sub(warp^-1)`` pair meets as ``[warp, warp^-1]`` and is
collapsed by the cancellation pair simplifier before any composer runs.
"""

# stdlib
from dataclasses import dataclass, field

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from .base import Transformation
from .concrete import CartesianField, Identity, Permutation, is_identity
from .meta import SubspaceTransformation
from .restrict import restrict
from .utils import UNREADABLE, affine_matrix, axis_list

# typing
if tx.TYPE_CHECKING:
    from .modes import Family
    from .sequence import Sequence
    from .simplify import SimplifyTable

# ----------------------------------------------------------------------
#   DATA STRUCTURES
# ----------------------------------------------------------------------


@dataclass
class _Stage:
    # One chain element read in the work view.
    element: Transformation
    pattern: np.ndarray  # (N_out, N_in) bool: output i depends on input j


@dataclass
class _Group:
    # A connected component of the (stage, work-axis) graph.
    axes: tx.List[int]  # sorted work positions the factor acts on (A = G)
    grid_axes: tx.List[int]  # G (chain-input positions)
    data_axes: tx.List[int]  # D (chain-output positions), paired with `axes`
    per_stage: tx.Dict[int, tx.List[int]] = field(default_factory=dict)


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


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


def _element_ndim(
    element: Transformation, ndim: int
) -> tx.Optional[tx.Tuple[int, int]]:
    # The (input, output) dimensionality of a chain element, or `None` when
    # it cannot be read. A subspace, a raw field and an identity preserve the
    # full dimensionality; an affine-ish element is read from its matrix
    # shape; a lazy inverse is read from its forward, transposed, so it is
    # never materialized.
    from .inverse import Inverse
    from .sequence import _interpolates

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


def _pattern(
    element: Transformation,
    ni: int,
    no: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    # The element's `(no, ni)` boolean dependency pattern in the work view,
    # memoized by `id`. The keepalive list pins the element so its `id` is
    # not reused while the cache holds its pattern.
    key = id(element)
    if key in dep_cache:
        return dep_cache[key]
    pat = _read_pattern(element, ni, no, dep_cache, dep_keepalive)
    dep_cache[key] = pat
    dep_keepalive.append(element)
    return pat


def _read_pattern(
    element: Transformation,
    ni: int,
    no: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    from .inverse import Inverse
    from .sequence import Sequence, _interpolates

    if isinstance(element, Sequence):
        # A nested sub-chain (e.g. a subspace inner that composed only
        # partially under a restrictive mode): the pattern is the product of
        # its members' patterns along the chain.
        acc = np.eye(ni, dtype=bool)
        ndim = ni
        for member in element.transformations or []:
            dims = _element_ndim(member, ndim)
            if dims is None or dims[0] != ndim:
                return None
            part = _read_pattern(member, *dims, dep_cache, dep_keepalive)
            if part is None or part.shape != dims[::-1]:
                return None
            acc = (part.astype(int) @ acc.astype(int)) > 0
            ndim = dims[1]
        return acc if acc.shape == (no, ni) else None
    if isinstance(element, SubspaceTransformation):
        if ni != no:
            return None
        return _subspace_pattern(element, ni, dep_cache, dep_keepalive)
    if isinstance(element, Inverse):
        forward = element.forward
        if forward is None:
            return np.eye(ni, dtype=bool) if ni == no else None
        # The pattern of a lazy inverse is the transpose of the forward's,
        # which never materializes it and gives the same coupling components.
        fpat = _read_pattern(forward, no, ni, dep_cache, dep_keepalive)
        if fpat is None or fpat.shape != (ni, no):
            return None
        return fpat.T
    if _interpolates(element):
        # A raw field couples every axis to every axis.
        return np.ones((no, ni), dtype=bool) if ni == no else None
    if isinstance(element, Identity):
        return np.eye(ni, dtype=bool) if ni == no else None
    return _read_pattern_affine(element, ni, no)


def _subspace_pattern(
    element: SubspaceTransformation,
    ndim: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    from .sequence import _interpolates

    inner = element.transformation
    in_axes = axis_list(element.input_axes)
    out_axes = axis_list(element.output_axes)
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            # An interpolating subspace naming no axes cannot be read.
            return None
        # A non-interpolating subspace carries its full affine embedding.
        return _read_pattern_affine(element, ndim, ndim)
    ki, ko = len(in_axes), len(out_axes)
    if interpolates:
        # An interpolating inner couples every acted-on axis to every other.
        inner_dep = np.ones((ko, ki), dtype=bool)
    else:
        inner_dep = None
        if ki == ko:
            candidate = _read_pattern(inner, ki, ko, dep_cache, dep_keepalive)
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


def _build_stages(
    body: tx.List[Transformation],
    n_in: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[tx.List[_Stage]]:
    # Read every element's dependency pattern in the work view. Returns
    # `None` when the chain's ends differ in dimension or an element cannot
    # be read, so the caller leaves the sequence unfactored. An element may
    # change the dimension in between (an embedding, then a projection).
    stages: tx.List[_Stage] = []
    ndim = n_in
    for element in body:
        dims = _element_ndim(element, ndim)
        if dims is None or dims[0] != ndim:
            return None
        ni, no = dims
        pat = _pattern(element, ni, no, dep_cache, dep_keepalive)
        if pat is None or pat.shape != (no, ni):
            return None
        stages.append(_Stage(element=element, pattern=pat))
        ndim = no
    if ndim != n_in:
        # Not dimension-preserving: created / dropped axes are left to a
        # follow-up (no E / Pi_drop is emitted yet).
        return None
    return stages


def _stage_dims(stages: tx.List[_Stage], n_axes: int) -> tx.List[int]:
    # The axis count at every stage: stage 0 is the chain's input, and
    # stage `s` is the output of element `s`.
    return [n_axes] + [s.pattern.shape[0] for s in stages]


def _partition(
    stages: tx.List[_Stage], n_axes: int
) -> tx.Optional[tx.List[_Group]]:
    # Connected components of the graph whose nodes are `(stage, work-axis)`
    # and whose edges are the true entries of each stage pattern. Returns
    # `None` on any non-square group (mixed / source / sink), so the chain is
    # left unfactored.
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
            # A group touching only the grid or only the data (source / sink)
            # is not a per-axis step (yet).
            return None
        if len(grid_axes) != len(data_axes):
            return None  # mixed / rank-deficient -> unfactored
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
#   RESTRICTION (see the `restrict` module)
# ----------------------------------------------------------------------


def _restrict_group(
    group: _Group, stages: tx.List[_Stage]
) -> tx.List[Transformation]:
    # The group's pieces, one per element it touches, in chain order. Each
    # element is restricted to the group's axes at its two stages by the
    # dispatched `restrict` operation; an identity piece is left out.
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


# ----------------------------------------------------------------------
#   NORMAL FORM
# ----------------------------------------------------------------------


def _inner_is_identity(inner: tx.Optional[Transformation]) -> bool:
    # Whether a group's composed inner is the identity, so its factor is
    # dropped. A non-interpolating inner is checked at value level
    # (`compute=True`), so a restriction that lands on the identity -- a
    # single-axis permutation `Permutation([0])`, a unit `Scaling([1])`, an
    # identity affine sub-block -- is recognized and dropped. An interpolating
    # inner (a field) is only ever checked structurally, so a field factor is
    # never dropped by reading its values.
    from .sequence import _interpolates

    if inner is None:
        return True
    if _interpolates(inner):
        return is_identity(inner, compute=False)
    return is_identity(inner, compute=True)


def _build_factors(
    groups: tx.List[_Group],
    stages: tx.List[_Stage],
    modes: tx.List["Family"],
    policy: "SimplifyTable",
) -> tx.List[SubspaceTransformation]:
    from .sequence import Sequence

    factors: tx.List[SubspaceTransformation] = []
    for group in groups:
        sub = _restrict_group(group, stages)
        if not sub:
            continue
        # The group's restricted sub-chain is computed like any other
        # sequence, under the caller's own mode and simplify policy, so the
        # pieces inside a group compose exactly as far as `mode` admits --
        # and a restricted transform/inverse pair cancels first, for free.
        inner = Sequence(transformations=sub).compute(
            modes, simplify=policy, factor=False
        )
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
    # `Pi_perm` scatters each group's work axes `A` to its data axes `D`:
    # output data axis `D[j]` reads work axis `A[j]`. Omitted when identity.
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
    # Whether the current body already has the normal form's structure (same
    # factor axes, same reindex), so it is returned unchanged (idempotence /
    # identity preservation). Inners are trusted to be stable across
    # iterations (they were composed when first built), so only axis lists
    # and the permutation are compared.
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


def _factor(
    seq: "Sequence",
    modes: tx.List["Family"],
    policy: "SimplifyTable",
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> Transformation:
    """The factor pass. Returns `seq` (same object) when nothing factors.

    `dep_cache` memoizes each leaf's dependency pattern by `id`, and
    `dep_keepalive` pins the leaves it holds so an `id` is never reused while
    the cache holds it; the caller owns both for the length of one `compute`.
    """
    from .sequence import _unnest

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
        # An interior or trailing grid is not our shape (interior grids are
        # dropped before this pass); leave it unfactored.
        return seq

    n_in = _chain_ndim_in(grid, body)
    if n_in is None:
        return seq

    stages = _build_stages(body, n_in, dep_cache, dep_keepalive)
    if stages is None:
        return seq  # not dimension-preserving / unreadable

    groups = _partition(stages, n_in)
    if groups is None:
        return seq  # a mixed / source / sink group -> unfactored
    if len(groups) <= 1:
        return seq  # trivial: nothing separable -> unchanged, same objects

    factors = _build_factors(groups, stages, modes, policy)
    perm = _build_perm(groups, n_in)

    if _same_structure(body, factors, perm):
        # Already in normal form for this partition: keep the same objects so
        # the fixpoint loop's identity-based exit test converges.
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


def _chain_ndim_in(
    grid: tx.Optional[CartesianField], body: tx.List[Transformation]
) -> tx.Optional[int]:
    # The chain's input dimensionality: the grid's axis count when a grid
    # leads, else read from the first element that states a dimension.
    if grid is not None and grid.shape is not None:
        return len(grid.shape)
    for element in body:
        dims = _element_ndim(element, -1)
        if dims is not None and dims[0] >= 0:
            return dims[0]
    return None
