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

Scope of PR-D (this module). Only *dimension-preserving* chains, whose
groups are all square (``|G| == |D|``), are factored. A chain that creates
or drops axes (a rectangular affine, a projection, a ``None``-index
broadcast), or that has a mixed / rank-deficient group, is left unfactored
and returned unchanged -- the same conservative fallback the reslice
detector uses today. The embedding / drop projections (``E`` / ``Pi_drop``)
of the full normal form are a follow-up.

The pass is *idempotent* and *identity-preserving*: a sequence already in
normal form is returned unchanged (same objects), and a leaf the pass does
not split keeps its identity, so the adjacent-inverse cancel link and the
driver's simplify id-cache stay valid.
"""

# stdlib
from dataclasses import dataclass, field

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from .base import Transformation
from .concrete import (
    Affine,
    CartesianField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
    is_identity,
)
from .errors import ConversionError
from .meta import Projection, SubspaceTransformation

# ----------------------------------------------------------------------
#   DATA STRUCTURES
# ----------------------------------------------------------------------


@dataclass
class _Stage:
    # One chain element read in the (dimension-preserving) work view.
    element: Transformation
    pattern: np.ndarray  # (N, N) bool: output axis i depends on input axis j


@dataclass
class _Group:
    # A connected component of the (stage, work-axis) graph.
    axes: tx.List[int]  # sorted work positions the factor acts on (A = G)
    grid_axes: tx.List[int]  # G (chain-input positions)
    data_axes: tx.List[int]  # D (chain-output positions), paired with `axes`
    per_stage: tx.Dict[int, tx.List[int]] = field(default_factory=dict)


# Returned by `_restrict_simple` for a piece that restricts to the identity.
_DROP = object()


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _axis_list(axes: tx.Optional[tx.Any]) -> tx.List[int]:
    # A plain list of integer axis indices. An axis vector may be a numpy
    # array (ambiguous truth value), so it is tested against `None`.
    if axes is None:
        return []
    return [int(a) for a in axes]


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
    matrix = _affine_matrix(element)
    if matrix is None:
        return ndim, ndim
    no, cols = matrix.shape
    return cols - 1, no


def _affine_matrix(element: Transformation) -> tx.Optional[np.ndarray]:
    # The affine matrix of an affine-ish element, or `None` when the element
    # is matrix-less (an identity) or cannot be converted. Never called on an
    # `Inverse` (its `.matrix` would materialize the inverse).
    try:
        affine = element.to(Affine)
    except ConversionError:
        return None
    matrix = affine.matrix
    return None if matrix is None else np.asarray(matrix)


def _pattern(
    element: Transformation,
    ndim: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    # The element's `(ndim, ndim)` boolean dependency pattern in the work
    # view, memoized by `id`. The keepalive list pins the element so its `id`
    # is not reused while the cache holds its pattern.
    key = id(element)
    if key in dep_cache:
        return dep_cache[key]
    pat = _read_pattern(element, ndim, dep_cache, dep_keepalive)
    dep_cache[key] = pat
    dep_keepalive.append(element)
    return pat


def _read_pattern(
    element: Transformation,
    ndim: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    from .inverse import Inverse
    from .sequence import Sequence, _interpolates

    if isinstance(element, Sequence):
        # A nested sub-chain (e.g. a subspace inner that composed only
        # partially under a restrictive mode): the pattern is the product of
        # its members' patterns along the chain.
        acc = np.eye(ndim, dtype=bool)
        for member in element.transformations or []:
            part = _read_pattern(member, ndim, dep_cache, dep_keepalive)
            if part is None or part.shape != (ndim, ndim):
                return None
            acc = (part.astype(int) @ acc.astype(int)) > 0
        return acc
    if isinstance(element, SubspaceTransformation):
        return _subspace_pattern(element, ndim, dep_cache, dep_keepalive)
    if isinstance(element, Inverse):
        forward = element.forward
        if forward is None:
            return np.eye(ndim, dtype=bool)
        # The pattern of a lazy inverse is the transpose of the forward's,
        # which never materializes it and gives the same coupling components.
        fpat = _read_pattern(forward, ndim, dep_cache, dep_keepalive)
        if fpat is None or fpat.shape != (ndim, ndim):
            return None
        return fpat.T
    if _interpolates(element):
        # A raw field couples every axis to every axis.
        return np.ones((ndim, ndim), dtype=bool)
    if isinstance(element, Identity):
        return np.eye(ndim, dtype=bool)
    matrix = _affine_matrix(element)
    if matrix is None:
        return np.eye(ndim, dtype=bool)
    if matrix.shape[0] != ndim or matrix.shape[1] - 1 != ndim:
        return None
    return matrix[:, :-1] != 0


def _subspace_pattern(
    element: SubspaceTransformation,
    ndim: int,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> tx.Optional[np.ndarray]:
    from .sequence import _interpolates

    inner = element.transformation
    in_axes = _axis_list(element.input_axes)
    out_axes = _axis_list(element.output_axes)
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            # An interpolating subspace naming no axes cannot be read.
            return None
        # A non-interpolating subspace carries its full affine embedding.
        return _read_pattern_affine(element, ndim)
    ki, ko = len(in_axes), len(out_axes)
    if interpolates:
        # An interpolating inner couples every acted-on axis to every other.
        inner_dep = np.ones((ko, ki), dtype=bool)
    else:
        inner_dep = None
        if ki == ko:
            candidate = _read_pattern(inner, ki, dep_cache, dep_keepalive)
            if candidate is not None and candidate.shape == (ko, ki):
                inner_dep = candidate
        if inner_dep is None:
            return _read_pattern_affine(element, ndim)
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
    element: Transformation, ndim: int
) -> tx.Optional[np.ndarray]:
    matrix = _affine_matrix(element)
    if matrix is None:
        return np.eye(ndim, dtype=bool)
    if matrix.shape[0] != ndim or matrix.shape[1] - 1 != ndim:
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
    # `None` when the chain is not dimension-preserving-square or an element
    # cannot be read, so the caller leaves the sequence unfactored.
    stages: tx.List[_Stage] = []
    ndim = n_in
    for element in body:
        dims = _element_ndim(element, ndim)
        if dims is None:
            return None
        ni, no = dims
        if ni != ndim or no != ndim:
            # Not dimension-preserving: created / dropped axes are left to a
            # follow-up (PR-D emits no E / Pi_drop).
            return None
        pat = _pattern(element, ndim, dep_cache, dep_keepalive)
        if pat is None or pat.shape != (ndim, ndim):
            return None
        stages.append(_Stage(element=element, pattern=pat))
        ndim = no
    return stages


def _partition(
    stages: tx.List[_Stage], n_axes: int
) -> tx.Optional[tx.List[_Group]]:
    # Connected components of the graph whose nodes are `(stage, work-axis)`
    # and whose edges are the true entries of each stage pattern. Returns
    # `None` on any non-square group (mixed / source / sink), so the chain is
    # left unfactored.
    num_stages = len(stages) + 1
    uf = _UnionFind(num_stages * n_axes)

    def node(stage: int, axis: int) -> int:
        return stage * n_axes + axis

    for stage, s in enumerate(stages, start=1):
        pat = s.pattern
        for i in range(n_axes):
            for j in range(n_axes):
                if pat[i, j]:
                    uf.union(node(stage, i), node(stage - 1, j))

    roots: tx.Dict[int, tx.Dict[int, tx.List[int]]] = {}
    for stage in range(num_stages):
        for axis in range(n_axes):
            root = uf.find(node(stage, axis))
            roots.setdefault(root, {}).setdefault(stage, []).append(axis)

    last = num_stages - 1
    groups: tx.List[_Group] = []
    for comp in roots.values():
        grid_axes = sorted(comp.get(0, []))
        data_axes = sorted(comp.get(last, []))
        if not grid_axes or not data_axes:
            # A group touching only the grid or only the data (source / sink)
            # is not a per-axis step in PR-D.
            return None
        if len(grid_axes) != len(data_axes):
            return None  # mixed / rank-deficient -> unfactored (JC-6)
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
#   RESTRICTION (type-preserving)
# ----------------------------------------------------------------------


def _restrict_simple(
    element: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Any:
    # Restrict a `Scaling` / `Translation` / `Permutation` to a group's axes,
    # keeping its type. `_DROP` marks an identity restriction; `None` falls
    # back to the affine sub-block.
    if isinstance(element, Scaling):
        if element.scale is None:
            return _DROP
        return Scaling(scale=np.asarray(element.scale)[cols])
    if isinstance(element, Translation):
        if element.translation is None:
            return _DROP
        return Translation(translation=np.asarray(element.translation)[rows])
    if isinstance(element, Permutation):
        if element.permutation is None:
            return _DROP
        permutation = np.asarray(element.permutation)
        col_pos = {c: j for j, c in enumerate(cols)}
        new_perm = []
        for r in rows:
            source = int(permutation[r])
            if source not in col_pos:
                return None
            new_perm.append(col_pos[source])
        return Permutation(permutation=np.asarray(new_perm, dtype=int))
    return None


def _restrict_via_affine(
    element: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Transformation]:
    matrix = _affine_matrix(element)
    if matrix is None:
        return None
    n_in = matrix.shape[1] - 1
    return Affine(matrix=matrix[np.ix_(rows, list(cols) + [n_in])])


def _restrict_subspace(
    element: SubspaceTransformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Transformation]:
    from .sequence import _interpolates

    in_axes = _axis_list(element.input_axes)
    out_axes = _axis_list(element.output_axes)
    inner = element.transformation
    interpolates = _interpolates(inner)
    if not in_axes or not out_axes:
        if interpolates:
            return element
        return _restrict_via_affine(element, rows, cols)
    acted_in = [a for a in in_axes if a in cols]
    if not acted_in:
        return None  # the group is pass-through for this subspace
    if list(in_axes) == list(cols) and list(out_axes) == list(rows):
        # The group covers exactly the acted axes -> return the inner itself,
        # preserving object identity so `Sub(warp) . Sub(warp^-1)` cancels via
        # `_cancels` inside the group without materializing a field.
        return inner
    if interpolates:
        # An interpolating inner is all-ones, so it never partially overlaps
        # a group; re-wrap defensively over the group's local axes.
        new_in = [cols.index(a) for a in in_axes if a in cols]
        new_out = [rows.index(a) for a in out_axes if a in rows]
        return SubspaceTransformation(
            transformation=inner,
            input_axes=np.asarray(new_in, dtype=int),
            output_axes=np.asarray(new_out, dtype=int),
        )
    if inner is None:
        return None
    # Partial overlap of a non-interpolating (hence separable) inner: recurse
    # into the inner over the group's *local* axes, which keeps the cheaper
    # type (a diagonal `Sub(Scaling, ...)` restricts to a `Scaling`, not a
    # rank-deficient affine sub-block that would need the subspace's endpoints
    # to embed).
    local_in = [j for j, a in enumerate(in_axes) if a in cols]
    local_out = [j for j, a in enumerate(out_axes) if a in rows]
    return _restrict_element(inner, local_out, local_in)


def _restrict_inverse(
    element: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Transformation]:
    forward = element.forward
    if forward is None:
        return None
    dims = _element_ndim(forward, len(rows))
    full = dims is not None and dims == (len(cols), len(rows))
    if full and sorted(rows) == sorted(cols):
        # The group covers the whole element: return the inverse object
        # itself, so it stays lazy and cancels by identity.
        return element
    # A block-diagonal inverse: restrict the forward with swapped rows/cols
    # and re-invert, which stays lazy.
    inner = _restrict_element(forward, cols, rows)
    if inner is None:
        return None
    return inner.inverse()


def _restrict_element(
    element: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Optional[Transformation]:
    # The piece a group contributes for one element, restricted to the
    # group's axes at that element's stages. `None` marks an identity piece.
    from .inverse import Inverse
    from .sequence import _interpolates

    if isinstance(element, (Identity, Projection)):
        return None
    if isinstance(element, SubspaceTransformation):
        return _restrict_subspace(element, rows, cols)
    if isinstance(element, Inverse):
        return _restrict_inverse(element, rows, cols)
    if _interpolates(element):
        # A raw field couples the whole space, so it is its own single group
        # and is spliced whole (this is reached only defensively).
        return element
    simple = _restrict_simple(element, rows, cols)
    if simple is _DROP:
        return None
    if simple is not None:
        return simple
    return _restrict_via_affine(element, rows, cols)


def _restrict_group(
    group: _Group, stages: tx.List[_Stage]
) -> tx.List[Transformation]:
    sub: tx.List[Transformation] = []
    for stage, s in enumerate(stages, start=1):
        rows = group.per_stage.get(stage, [])
        cols = group.per_stage.get(stage - 1, [])
        if not rows or not cols:
            continue
        piece = _restrict_element(s.element, rows, cols)
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
    mode: tx.Any,
    table: tx.Any,
) -> tx.List[SubspaceTransformation]:
    from .sequence import Sequence

    factors: tx.List[SubspaceTransformation] = []
    for group in groups:
        sub = _restrict_group(group, stages)
        if not sub:
            continue
        inner = Sequence(transformations=sub).compute(
            mode, simplify=table, factor=False
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
            if _axis_list(got.input_axes) != _axis_list(want.input_axes):
                return False
            if _axis_list(got.output_axes) != _axis_list(want.output_axes):
                return False
    return True


def _factor(
    seq: "tx.Any",
    mode: tx.Any,
    table: tx.Any,
    dep_cache: tx.Dict[int, tx.Optional[np.ndarray]],
    dep_keepalive: tx.List[Transformation],
) -> "tx.Any":
    """The factor pass. Returns `seq` (same object) when nothing factors."""
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
        return seq  # a mixed / source / sink group -> unfactored (JC-6)
    if len(groups) <= 1:
        return seq  # trivial: nothing separable -> unchanged, same objects

    factors = _build_factors(groups, stages, mode, table)
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
    return replace(seq, transformations=elements)


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


# Silence linters about the imported-but-conditionally-used names that keep
# the restriction/pattern readers self-documenting.
_ = (Linear, Rotation)
