"""Reslice an image by exploiting the separability of its transformation.

A reslice applies a voxel-to-voxel transformation to image data by
sampling the data at transformed coordinates. When the transformation
couples every output axis to every input axis, the whole coordinates
field must be built and the data sampled with an N-dimensional pull.
Many transformations do not couple every axis (for example: translations,
scalings, permutations, or transformations that act on a subset of the axes).

This module factors such a transformation into independent steps, one per
group of axes that transform together, and applies each step on its own.
A group that needs no interpolation, such as a flip or an integer shift,
is applied as a gather or a view. A group that is a low-dimensional affine
is applied as a precomputed weight matrix. Only the irreducible coupled
group keeps the N-dimensional pull, and it runs over the fewest axes.

The result is numerically identical to the monolithic reslice within the
interpolation tolerance, and bit-identical when nothing is separable, in
which case the whole transformation is one group and the monolithic pull
is used unchanged.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.bsplines import pull, pull_axes, spline_matrix

# api
from brainhops.backends import get_array_backend
from brainhops.datamodel import hierarchy

# locals
from .base import Transformation
from .concrete import (
    Affine,
    CartesianField,
    Identity,
    Permutation,
    Scaling,
    Translation,
)
from .errors import CompositionError, ConversionError
from .meta import SubspaceTransformation
from .sequence import Sequence, _interpolates


class _Unknown(Exception):
    """Raised when the detector meets a chain element it cannot read.

    The factoring cannot proceed for a transformation it does not
    understand, and the caller falls back to the monolithic pull.
    """


# ----------------------------------------------------------------------
#   DETECTOR
# ----------------------------------------------------------------------


def _axis_list(axes: tx.Optional[tx.Any]) -> tx.List[int]:
    # A plain list of integer axis indices. An axis vector may be a numpy
    # array, whose truth value is ambiguous, so it is tested against `None`
    # rather than for emptiness.
    if axes is None:
        return []
    return [int(a) for a in axes]


def _affine_dependency(
    element: Transformation, ndim_in: int, ndim_out: int
) -> np.ndarray:
    # The dependency read from an element's affine matrix. A matrix-less
    # affine is the identity, and an element that cannot be reduced to an
    # affine is unreadable. The translation column carries no coupling and
    # is dropped, so an entry is kept only where the linear part is nonzero.
    try:
        affine = element.to(Affine)
    except ConversionError as error:
        raise _Unknown() from error
    matrix = affine.matrix
    if matrix is None:
        if ndim_in != ndim_out:
            raise _Unknown()
        return np.eye(ndim_in, dtype=bool)
    return np.asarray(matrix)[:, :-1] != 0


def _subspace_dependency(
    element: SubspaceTransformation, ndim_in: int, ndim_out: int
) -> np.ndarray:
    """
    The dependency of a transform that acts on a subset of the axes.
    The behaviour depends on whether the wrapped transform interpolates.

    A non-interpolating inner is read by recursing through the same
    machinery, so its coupling among the acted-on axes is the dependency
    of the inner transform itself. A subspace wrapping a diagonal linear,
    a nested non-interpolating subspace, or a shear on a sub-block then
    couples only what the inner transform actually mixes, rather than
    every acted-on axis to every other. `_restrict` reproduces such a
    group exactly through the affine sub-block embedding.

    An interpolating inner instead couples every acted-on output axis to
    every acted-on input axis. Its finer partition is not recovered here:
    `_restrict` would re-wrap the whole inner over the group's fewer axes,
    and the inner's own axis indices would then point at the wrong axes.
    The all-ones over-approximation keeps every acted-on axis in one group,
    which reslices correctly and matches what a full-axis spatial warp
    emits.

    In either case the remaining axes pass through in order, paired the same
    way as the subspace-to-affine reduction pairs them.
    """
    inner = element.transformation
    interpolates = _interpolates(inner)
    in_axes = _axis_list(element.input_axes)
    out_axes = _axis_list(element.output_axes)
    if not in_axes or not out_axes:
        if interpolates:
            # An interpolating subspace transform that names no axes cannot
            # be read. Treating it as pass-through would drop the warp and
            # return the data unwarped, so the whole reslice falls back to
            # the monolithic pull, which then reports the malformed input.
            raise _Unknown()
        # A non-interpolating subspace that names no axes carries its full
        # affine embedding, which names the axes it touches through its
        # matrix.
        return _affine_dependency(element, ndim_in, ndim_out)
    ki, ko = len(in_axes), len(out_axes)
    if interpolates:
        # An interpolating inner, whether a raw field or a warp on a subset
        # of the axes, couples every acted-on axis to every other. This is a
        # safe over-approximation, which only ever misses an optimization and
        # never splits an axis that should stay coupled.
        inner_dep = np.ones((ko, ki), dtype=bool)
    else:
        inner_dep = None
        if ki == ko:
            try:
                candidate = _transform_dependency(inner, ki, ko)
            except Exception:
                # A wrapped transform whose dependency cannot be read is
                # read from the affine embedding below, never under-coupled.
                candidate = None
            if candidate is not None and candidate.shape == (ko, ki):
                inner_dep = np.asarray(candidate, dtype=bool)
        if inner_dep is None:
            # A non-interpolating inner that could not be recursed is read
            # from the subspace's full affine embedding instead.
            return _affine_dependency(element, ndim_in, ndim_out)
    dep = np.zeros((ndim_out, ndim_in), dtype=bool)
    for a, o in enumerate(out_axes):
        for b, i in enumerate(in_axes):
            if inner_dep[a, b]:
                dep[o, i] = True
    pass_in = [i for i in range(ndim_in) if i not in set(in_axes)]
    pass_out = [o for o in range(ndim_out) if o not in set(out_axes)]
    for o, i in zip(pass_out, pass_in):
        dep[o, i] = True
    return dep


def _dependency(
    element: Transformation, ndim_in: int, ndim_out: int
) -> np.ndarray:
    """Return the dependency matrix of one chain element.

    The result is a boolean matrix of shape `(ndim_out, ndim_in)`. Its
    entry `[i, j]` is true when output coordinate `i` depends on input
    coordinate `j`. A translation never couples axes, so the translation
    column of an affine is dropped. A shear couples axes through its
    off-diagonal entries, which are kept.
    """
    if isinstance(element, SubspaceTransformation):
        return _subspace_dependency(element, ndim_in, ndim_out)
    if isinstance(element, Identity):
        if ndim_in != ndim_out:
            raise _Unknown()
        return np.eye(ndim_in, dtype=bool)
    if _interpolates(element):
        # A raw field, not wrapped in a subspace, couples every axis to
        # every axis. Nothing about it is separable.
        return np.ones((ndim_out, ndim_in), dtype=bool)
    return _affine_dependency(element, ndim_in, ndim_out)


def _element_dims(element: Transformation, ndim_in: int) -> tx.Tuple[int, int]:
    # The input and output dimensionality of a chain element. The input
    # dimensionality is known from the previous stage. A transform that
    # states a matrix or an axis count is read directly, and one that
    # states neither is assumed to preserve dimensionality.
    if isinstance(element, SubspaceTransformation):
        system = element.input if element.input is not None else element.output
        if system is not None and system.axes is not None:
            return len(system.axes), len(system.axes)
        return ndim_in, ndim_in
    if not _interpolates(element) and not isinstance(element, Identity):
        try:
            matrix = element.to(Affine).matrix
        except ConversionError:
            return ndim_in, ndim_in
        if matrix is not None:
            return int(np.asarray(matrix).shape[1]) - 1, int(
                np.asarray(matrix).shape[0]
            )
    return ndim_in, ndim_in


def _build_deps(
    els: tx.List[Transformation], n_grid: int
) -> tx.Tuple[tx.List[np.ndarray], tx.List[int]]:
    # Build the per-stage dependency matrices of a transformation chain.
    #
    # For each element `T_s` of `els`, this returns its dependency matrix
    # `Dep(T_s)` -- a boolean matrix whose entry `[i, j]` is true when the
    # element's output axis `i` depends on its input axis `j`. It also
    # returns the axis count at every stage, where stage 0 is the sampling
    # grid of `n_grid` axes and stage `s` is the output of element `s`.
    # Together the matrices and dimensions describe how coupling flows
    # along the chain, which `_components` walks to partition the axes.
    #
    # A chain whose stages do not line up, so that one element's input
    # dimensionality does not match the previous stage, is unreadable.
    deps: tx.List[np.ndarray] = []
    stage_dims = [n_grid]
    current = n_grid
    for element in els:
        ndim_in, ndim_out = _element_dims(element, current)
        if ndim_in != current:
            raise _Unknown()
        deps.append(_dependency(element, ndim_in, ndim_out))
        current = ndim_out
        stage_dims.append(current)
    return deps, stage_dims


def _transform_dependency(
    transform: Transformation, ndim_in: int, ndim_out: int
) -> np.ndarray:
    # The full boolean dependency of a possibly-composite transform, of
    # shape `(ndim_out, ndim_in)`. A sequence composes the dependency of its
    # elements along the chain, so coupling introduced by any stage reaches
    # the output. Any other transform is read directly by `_dependency`.
    # This lets the dependency of a subspace's wrapped transform be computed
    # by the same machinery that reads the outer chain.
    if isinstance(transform, Sequence):
        els = transform.transformations or []
        deps, _stage_dims = _build_deps(els, ndim_in)
        acc = np.eye(ndim_in, dtype=bool)
        for dep in deps:
            acc = np.dot(dep.astype(int), acc.astype(int)) > 0
        return acc
    return _dependency(transform, ndim_in, ndim_out)


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


def _components(
    deps: tx.List[np.ndarray], n_grid: int, stage_dims: tx.List[int]
) -> tx.Optional[tx.List[dict]]:
    # Partition the axes into groups that transform together. A group is a
    # connected component of the graph whose nodes are the axes at every
    # stage and whose edges are the true entries of each stage's dependency
    # matrix. Each group is returned with its grid axes, its data axes, and
    # its axis set at every stage. A group that is not square, or a data
    # axis reached by no grid axis, makes the transformation unseparable
    # and returns `None`.
    offsets = [0]
    for dim in stage_dims:
        offsets.append(offsets[-1] + dim)
    total = offsets[-1]
    uf = _UnionFind(total)

    def node(stage: int, axis: int) -> int:
        return offsets[stage] + axis

    for stage, dep in enumerate(deps, start=1):
        rows, cols = dep.shape
        for i in range(rows):
            for j in range(cols):
                if dep[i, j]:
                    uf.union(node(stage, i), node(stage - 1, j))

    last = len(stage_dims) - 1
    roots: tx.Dict[int, dict] = {}
    for stage, dim in enumerate(stage_dims):
        for axis in range(dim):
            root = uf.find(node(stage, axis))
            comp = roots.setdefault(root, {"stages": {}})
            comp["stages"].setdefault(stage, []).append(axis)

    components: tx.List[dict] = []
    for comp in roots.values():
        stages = comp["stages"]
        grid_axes = sorted(stages.get(0, []))
        data_axes = sorted(stages.get(last, []))
        if not grid_axes or not data_axes:
            # A group that touches only the grid or only the data cannot be
            # inverted as a step, so the transformation is not separable.
            return None
        if len(grid_axes) != len(data_axes):
            # A rectangular group is not an invertible per-axis step.
            return None
        comp["G"] = grid_axes
        comp["D"] = data_axes
        comp["stages"] = {s: sorted(a) for s, a in stages.items()}
        components.append(comp)

    covered_grid = sorted(a for comp in components for a in comp["G"])
    covered_data = sorted(a for comp in components for a in comp["D"])
    if covered_grid != list(range(n_grid)):
        return None
    if covered_data != list(range(stage_dims[-1])):
        return None
    return components


# ----------------------------------------------------------------------
#   FACTORING
# ----------------------------------------------------------------------


# Returned by `_restrict_simple` for an element that restricts to the
# identity over the group's axes, so the caller drops it from the
# sub-sequence.
_DROP = object()


def _restrict_simple(
    element: Transformation, rows: tx.List[int], cols: tx.List[int]
) -> tx.Any:
    # Restrict a non-interpolating element to a group's axes, keeping its
    # type. A `Scaling`, `Translation`, or `Permutation` carries the cheaper,
    # truer type into the sub-sequence rather than a general affine sub-block.
    # A restriction that is the identity returns `_DROP`. An element this
    # function does not handle, or a permutation that maps outside the group,
    # returns `None`, and the caller falls back to the affine sub-block.
    if isinstance(element, Scaling):
        if element.scale is None:
            return _DROP
        scale = np.asarray(element.scale)
        return Scaling(scale=scale[cols])
    if isinstance(element, Translation):
        if element.translation is None:
            return _DROP
        translation = np.asarray(element.translation)
        return Translation(translation=translation[rows])
    if isinstance(element, Permutation):
        if element.permutation is None:
            return _DROP
        permutation = np.asarray(element.permutation)
        col_pos = {c: j for j, c in enumerate(cols)}
        new_perm = []
        for r in rows:
            source = int(permutation[r])
            if source not in col_pos:
                # The permutation sends this output axis to an input axis
                # outside the group, so the restriction is not a permutation
                # of the group's axes and the affine sub-block is used.
                return None
            new_perm.append(col_pos[source])
        return Permutation(permutation=np.asarray(new_perm, dtype=int))
    return None


def _restrict(
    comp: dict, els: tx.List[Transformation], shape: tx.Tuple[int, ...]
) -> tx.List[Transformation]:
    # The sub-sequence that a group contributes, restricted to the group's
    # axes at every stage. It begins with the group's own sampling grid,
    # and each non-grid element is cut down to the rows and columns the
    # group occupies at that element's stages. An identity element is
    # dropped.
    grid_shape = tuple(shape[g] for g in comp["G"])
    sub: tx.List[Transformation] = [CartesianField(shape=grid_shape)]
    for stage, element in enumerate(els, start=1):
        rows = comp["stages"].get(stage, [])
        cols = comp["stages"].get(stage - 1, [])
        if isinstance(element, Identity):
            continue
        if isinstance(element, SubspaceTransformation) and _interpolates(
            element.transformation
        ):
            in_axes = _axis_list(element.input_axes)
            out_axes = _axis_list(element.output_axes)
            inner = element.transformation
            acted = [a for a in in_axes if a in cols]
            if not acted:
                # The group is entirely pass-through for this subspace, so
                # the subspace is the identity here and is dropped.
                continue
            if list(in_axes) == list(cols) and list(out_axes) == list(rows):
                # The wrapped transform acts on exactly the group's axes,
                # so it is spliced in directly rather than re-wrapped.
                sub.append(inner)
            else:
                new_in = [cols.index(a) for a in in_axes if a in cols]
                new_out = [rows.index(a) for a in out_axes if a in rows]
                sub.append(
                    SubspaceTransformation(
                        transformation=inner,
                        input_axes=np.asarray(new_in, dtype=int),
                        output_axes=np.asarray(new_out, dtype=int),
                    )
                )
        elif not _interpolates(element):
            simple = _restrict_simple(element, rows, cols)
            if simple is _DROP:
                # The restriction is the identity, the same reading
                # `_dependency` gives it, and contributes nothing here.
                continue
            if simple is not None:
                sub.append(simple)
                continue
            matrix = element.to(Affine).matrix
            if matrix is None:
                # A matrix-less affine is the identity, the same reading
                # `_dependency` gives it, and contributes nothing here.
                continue
            matrix = np.asarray(matrix)
            n_in = matrix.shape[1] - 1
            sub_matrix = matrix[np.ix_(rows, list(cols) + [n_in])]
            sub.append(Affine(matrix=sub_matrix))
        else:
            # A raw field spanning the whole space would be a single group
            # and would already have fallen back, so this is not reached in
            # practice. It is spliced whole for safety.
            sub.append(element)
    return sub


def _restricted_affine(sub: tx.List[Transformation]) -> tx.Optional[Affine]:
    # The single affine that the non-grid part of a sub-sequence composes
    # to, or `None` when the non-grid part is empty (the identity).
    nongrid = sub[1:]
    if not nongrid:
        return None
    composed = Sequence(transformations=nongrid).compute(
        mode=hierarchy.AffineTransformation
    )
    return composed.to(Affine)


# The largest weight matrix a one-dimensional interpolating step builds
# before it falls back to the batched pull. The weight matrix has
# `n_out * n_in` elements, so a long axis makes it enormous. Above this
# many elements the batched pull is used instead, which samples the axis
# without materializing the matrix. The two paths give the same result, so
# the threshold trades memory for the matrix's speed and nothing else. Four
# million elements is about 32 MiB at float64, small enough to always hold.
_MAX_WEIGHT_MATRIX_ELEMENTS = 4_000_000


def _classify(
    comp: dict,
    els: tx.List[Transformation],
    shape: tx.Tuple[int, ...],
    data_shape: tx.Tuple[int, ...],
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> dict:
    # Build one executable step for a group. The step is classified as a
    # gather (no interpolation), a weight matrix (a one-dimensional
    # affine), or a batched pull (a coupled or higher-dimensional group).
    grid_axes = comp["G"]
    data_axes = comp["D"]
    k = len(data_axes)
    sub = _restrict(comp, els, shape)
    composed = Sequence(transformations=sub).compute(
        mode=hierarchy.AffineTransformation
    )
    interpolating = _interpolates(composed)

    n_in = int(np.prod([data_shape[d] for d in data_axes]))
    n_out = int(np.prod([shape[g] for g in grid_axes]))

    kind = "c"
    scale = shift = None
    unit = False
    discrete = _discrete_axis(comp, grid_system, data_system)[0]
    if not interpolating and k == 1:
        affine = _restricted_affine(sub)
        if affine is None:
            scale, shift = 1.0, 0.0
        else:
            matrix = np.asarray(affine.matrix)
            scale, shift = float(matrix[0, 0]), float(matrix[0, 1])
        # A scale of exactly unit magnitude with an integer shift maps each
        # output sample onto one input sample, so no value is interpolated.
        # Such a step is a whole-sample map.
        integer_shift = abs(shift - round(shift)) < 1e-9
        unit = abs(abs(scale) - 1.0) == 0.0 and integer_shift
        # The gather returns the input sample itself. That matches the
        # monolithic pull only when the pull returns the same sample. With
        # spline coefficients at order two or above the pull returns the
        # reconstruction of the coefficients, not the raw coefficient, and
        # scipy's `reflect` prefilter is not exactly interpolating above
        # order one. For a continuous axis those cases take the weight-matrix
        # path instead, which reproduces the reconstruction exactly.
        gather = order <= 1 or (not coeff and bound != "reflect")
        # A discrete axis holds no value between its samples, so a
        # whole-sample map along it is always an exact gather. It must never
        # be interpolated or blended across, even at order two and above
        # where the monolithic pull would mix its samples.
        if unit and (gather or discrete):
            kind = "a"
        else:
            kind = "b"

    _check_discrete(comp, unit, grid_system, data_system)

    step = {
        "D": data_axes,
        "G": grid_axes,
        "k": k,
        "kind": kind,
        "n_in": n_in,
        "n_out": n_out,
    }
    if kind == "a":
        step["run"] = _make_gather(
            data_axes[0], grid_axes[0], scale, shift, bound, data_shape, shape
        )
    elif kind == "b":
        if n_in * n_out <= _MAX_WEIGHT_MATRIX_ELEMENTS:
            step["run"] = _make_matrix(
                data_axes[0],
                grid_axes[0],
                scale,
                shift,
                order,
                bound,
                coeff,
                data_shape,
                shape,
            )
        else:
            # The weight matrix would be too large to hold, so the axis is
            # sampled with the batched pull the coupled class uses. The
            # result is the same as the weight-matrix path.
            step["run"] = _make_pull(
                sub, data_axes, grid_axes, order, bound, coeff
            )
    else:
        step["run"] = _make_pull(
            sub, data_axes, grid_axes, order, bound, coeff
        )
    return step


def _discrete_axis(
    comp: dict,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> tx.Tuple[bool, tx.Optional[str]]:
    # Whether this group touches a discrete axis, and that axis's name. A
    # discrete axis, such as a channel or a labelled time axis, holds no
    # value between its samples.
    discrete = False
    name: tx.Optional[str] = None
    for axis_index, system in (
        (comp["G"], grid_system),
        (comp["D"], data_system),
    ):
        axes = getattr(system, "axes", None)
        if axes is None:
            continue
        for a in axis_index:
            if a < len(axes) and getattr(axes[a], "discrete", None):
                discrete = True
                name = getattr(axes[a], "name", None) or getattr(
                    axes[a], "type", None
                )
    return discrete, name


def _check_discrete(
    comp: dict,
    unit: bool,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> None:
    # A discrete axis holds no value between its samples, so it can only be
    # permuted, flipped, or shifted by whole samples. Such an axis must be
    # its own group and its step must be a whole-sample map. Any other case
    # is refused.
    discrete, name = _discrete_axis(comp, grid_system, data_system)
    if not discrete:
        return
    if not unit or len(comp["D"]) != 1:
        raise CompositionError(
            f"Cannot apply an interpolating transform along the discrete "
            f"axis {name!r}. A discrete axis holds no value between its "
            f"samples, so it can only be permuted, flipped, or shifted by "
            f"whole samples."
        )
    # A whole-sample map along a discrete axis is exact: it moves whole
    # samples without reading any value between them, so it is always a
    # gather and is allowed. A step that genuinely resamples the axis is
    # refused above by the `not unit` check.


# ----------------------------------------------------------------------
#   EXECUTORS
# ----------------------------------------------------------------------


def _is_unit_progression(idx: np.ndarray) -> bool:
    # Whether an index array is an arithmetic progression of step +1 or -1,
    # so a gather along it can be expressed as a basic slice (a view).
    if idx.size < 1:
        return False
    if idx.size == 1:
        return True
    steps = np.diff(idx)
    return bool((steps == 1).all() or (steps == -1).all())


def _slice_axis(arr: tx.Any, axis: int, idx: np.ndarray) -> tx.Any:
    # A basic slice along one axis, given an index array known to be a unit
    # progression. This returns a view rather than a copy.
    start = int(idx[0])
    if idx.size == 1 or int(idx[1]) - int(idx[0]) == 1:
        stop = int(idx[-1]) + 1
        step = 1
    else:
        stop = int(idx[-1]) - 1
        step = -1
        if stop < 0:
            stop = None
    index = [slice(None)] * arr.ndim
    index[axis] = slice(start, stop, step)
    return arr[tuple(index)]


def _make_gather(
    d: int,
    g: int,
    scale: float,
    shift: float,
    bound: tx.Union[str, float],
    data_shape: tx.Tuple[int, ...],
    shape: tx.Tuple[int, ...],
) -> tx.Callable:
    n_in = data_shape[d]
    n_out = shape[g]
    bound0 = bound if isinstance(bound, str) else 0.0

    def run(arr: tx.Any, labels: tx.List[tx.Any]) -> tx.Tuple[tx.Any, list]:
        ab = get_array_backend(arr)
        axis = labels.index(("data", d))
        coords = np.arange(n_out) * scale + shift
        # Order-0 weights turn each output coordinate into the single input
        # sample it lands on. A row that sums to zero fell outside the grid
        # under a constant boundary, and takes the fill value.
        # FOLLOW-UP: `spline_matrix` builds an `n_in x n_in` identity to
        # read off these indices, which costs O(n_in**2) memory for a long
        # axis. Direct index arithmetic per boundary mode would avoid it,
        # but it must reproduce scipy's boundary indices exactly.
        weights = np.asarray(spline_matrix(n_in, coords, 0, bound0, True))
        rowsum = weights.sum(1)
        idx = weights.argmax(1)
        inbounds = rowsum > 0
        if bool(inbounds.all()) and _is_unit_progression(idx):
            result = _slice_axis(arr, axis, idx)
        else:
            result = ab.take(arr, ab.asarray(idx), axis=axis)
            if not bool(inbounds.all()):
                cval = 0.0 if isinstance(bound, str) else float(bound)
                mask_shape = [1] * result.ndim
                mask_shape[axis] = n_out
                mask = ab.asarray(inbounds).reshape(tuple(mask_shape))
                result = ab.where(mask, result, cval)
        labels = list(labels)
        labels[axis] = ("grid", g)
        return result, labels

    return run


def _make_matrix(
    d: int,
    g: int,
    scale: float,
    shift: float,
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
    data_shape: tx.Tuple[int, ...],
    shape: tx.Tuple[int, ...],
) -> tx.Callable:
    n_in = data_shape[d]
    n_out = shape[g]

    def run(arr: tx.Any, labels: tx.List[tx.Any]) -> tx.Tuple[tx.Any, list]:
        ab = get_array_backend(arr)
        axis = labels.index(("data", d))
        coords = ab.arange(n_out) * scale + shift
        weights = spline_matrix(n_in, coords, order, bound, coeff)
        out = ab.moveaxis(
            ab.tensordot(arr, weights, axes=([axis], [1])), -1, axis
        )
        if not isinstance(bound, str):
            # A constant boundary is not linear, so its fill is added
            # separately through the partition of unity of the weights.
            correction = float(bound) * (1.0 - np.asarray(weights).sum(1))
            corr_shape = [1] * out.ndim
            corr_shape[axis] = n_out
            out = out + ab.asarray(correction).reshape(tuple(corr_shape))
        labels = list(labels)
        labels[axis] = ("grid", g)
        return out, labels

    return run


def _make_pull(
    sub: tx.List[Transformation],
    data_axes: tx.List[int],
    grid_axes: tx.List[int],
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Callable:
    def run(arr: tx.Any, labels: tx.List[tx.Any]) -> tx.Tuple[tx.Any, list]:
        coords = Sequence(transformations=sub).compute().field
        positions = [labels.index(("data", d)) for d in data_axes]
        moved = pull_axes(arr, coords, positions, order, bound, coeff)
        used = set(positions)
        remaining = [labels[i] for i in range(len(labels)) if i not in used]
        labels = remaining + [("grid", g) for g in grid_axes]
        return moved, labels

    return run


# ----------------------------------------------------------------------
#   ORDERING
# ----------------------------------------------------------------------


def _order_steps(
    steps: tx.List[dict],
    data_shape: tx.Tuple[int, ...],
    shape: tx.Tuple[int, ...],
    order: int,
    coeff: bool,
) -> tx.List[dict]:
    # Order the steps so the intermediate arrays stay small. The steps act
    # on disjoint axes and commute, so ordering changes only the cost. A
    # step's ratio `r` is its output size over its input size, and its
    # weight `w` estimates its per-element work. Steps that shrink the
    # array run first, cheapest per unit shrunk; steps that keep the size
    # run next, with the free views first; steps that grow the array run
    # last, smallest growth first.
    for step in steps:
        k = step["k"]
        kind = step["kind"]
        n_in = step["n_in"]
        n_out = step["n_out"]
        r = (n_out / n_in) if n_in else 1.0
        if kind == "a":
            c = 1.0
        elif kind == "b":
            # The matmul touches every input sample once per output sample,
            # so its per-element cost times the ratio is the output size.
            c = float(n_in)
        else:
            c = float((order + 1) ** k)
        a = k * order if (order > 1 and not coeff) else 0
        step["_r"] = r
        step["_w"] = a + c * r

    shrink = [s for s in steps if s["_r"] < 1]
    unit = [s for s in steps if s["_r"] == 1]
    expand = [s for s in steps if s["_r"] > 1]

    shrink.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    unit.sort(key=lambda s: 0 if s["kind"] == "a" else 1)
    # For an expanding step `1 - r` is negative, so a smaller expansion
    # gives a more negative key and sorts first.
    expand.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    return shrink + unit + expand


# ----------------------------------------------------------------------
#   ENTRY POINT
# ----------------------------------------------------------------------


def pull_separable(
    data: tx.Any,
    seq: Transformation,
    *,
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Any:
    """Reslice `data` through `seq`, exploiting separable axes.

    The transformation `seq` maps the coordinates of the output grid to
    the coordinates of `data`. Its leading element is the sampling grid, a
    [`CartesianField`][brainhops.datamodel.transformations.CartesianField]
    whose shape is the shape of the output grid and whose output system is
    the coordinate system of that grid. The system in which `seq` leaves
    its coordinates is the coordinate system of the data. The output shape
    and both coordinate systems are read from `seq`, so they are not passed
    separately.

    `seq` is factored into independent steps, one per group of axes that
    transform together, and each step is applied on its own. A group that
    needs no interpolation is a gather or a view, a one-dimensional affine
    group is a weight matrix, and the irreducible coupled group keeps the
    N-dimensional pull over the fewest axes.

    The result equals the monolithic reslice within the interpolation
    tolerance. When the transformation does not factor, the whole of it is
    one group and the monolithic pull is used, so the result is then
    bit-identical to the monolithic reslice.

    Parameters
    ----------
    data : array-like
        The image data to reslice.
    seq : Transformation
        The transformation from the output grid to the data coordinates,
        before it is computed.
    order : {0..5}
        The interpolation order.
    bound : str or float
        The boundary condition, as accepted by
        [`pull`][brainhops._core.bsplines.pull].
    coeff : bool
        Whether the data already contains spline coefficients.

    Returns
    -------
    array-like
        The resliced data, of the shape of the output grid.
    """
    ab = get_array_backend(data)
    opt = dict(order=order, bound=bound, coeff=coeff)

    def fallback() -> tx.Any:
        return pull(data, seq.compute().field, **opt)

    try:
        part = seq.compute(mode=hierarchy.AffineTransformation)
    except Exception:
        return fallback()

    if not isinstance(part, Sequence):
        return fallback()
    elements = part.transformations or []
    if not elements or not isinstance(elements[0], CartesianField):
        return fallback()
    grid = elements[0]
    # The shape and the grid coordinate system are carried by the leading
    # grid of `seq`, and the data coordinate system is the system in which
    # `seq` leaves its coordinates. The systems are read only to identify
    # discrete axes.
    shape = grid.shape
    if shape is None:
        return fallback()
    grid_system = grid.output
    data_system = part.output
    els = list(elements[1:])
    n_grid = len(shape)

    try:
        deps, stage_dims = _build_deps(els, n_grid)
    except _Unknown:
        return fallback()
    if stage_dims[-1] != data.ndim:
        return fallback()

    components = _components(deps, n_grid, stage_dims)
    if components is None:
        return fallback()
    if len(components) == 1:
        return fallback()

    steps = [
        _classify(
            comp,
            els,
            shape,
            data.shape,
            order,
            bound,
            coeff,
            grid_system,
            data_system,
        )
        for comp in components
    ]
    steps = _order_steps(steps, data.shape, shape, order, coeff)

    # The pipeline runs in a floating working dtype so an integer input is
    # not rounded between steps. The weight-matrix step already produces a
    # float64 result, and the gather and pull steps preserve the dtype they
    # are given, so an integer input is lifted to float once here.
    out_dtype = np.dtype(data.dtype)
    floating = np.issubdtype(out_dtype, np.floating)
    arr = data if floating else data.astype(float)
    labels: tx.List[tx.Any] = [("data", i) for i in range(data.ndim)]
    for step in steps:
        arr, labels = step["run"](arr, labels)

    permutation = [labels.index(("grid", g)) for g in range(n_grid)]
    arr = ab.transpose(arr, permutation)
    # The assembled array is cast back to the input dtype exactly once, at
    # the end. The monolithic pull returns the input dtype and reproduces
    # scipy's `map_coordinates`, so an integer or boolean output is finished
    # the same way scipy's `CASE_INTERP_OUT_INT` does.
    if arr.dtype != out_dtype:
        if out_dtype.kind == "b":
            # scipy truncates toward zero for a boolean output, so a
            # fractional value such as 0.6 becomes False.
            arr = ab.trunc(arr).astype(out_dtype)
        elif np.issubdtype(out_dtype, np.integer):
            # scipy rounds an integer output half away from zero, then
            # clips it to the dtype range so an overshoot saturates
            # instead of wrapping. The order is round, clip, cast.
            rounded = ab.trunc(ab.where(arr > 0, arr + 0.5, arr - 0.5))
            info = np.iinfo(out_dtype)
            arr = ab.clip(rounded, info.min, info.max).astype(out_dtype)
        else:
            arr = arr.astype(out_dtype)
    return arr
