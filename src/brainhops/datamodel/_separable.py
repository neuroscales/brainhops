"""Reslice an image by exploiting the separability of its transformation.

A reslice applies a voxel-to-voxel transformation to image data by
sampling the data at transformed coordinates. When the transformation
couples every output axis to every input axis, the whole coordinates
field must be built and the data sampled with an N-dimensional pull.
Many transformations do not couple every axis. A time axis that is only
rescaled, an axis that is flipped or permuted, and a spatial warp that
leaves time alone each act on a group of axes independently of the rest.

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

from __future__ import annotations

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.bsplines import pull, pull_axes, spline_matrix

# backends
from brainhops.backends import get_array_backend

# locals
from . import hierarchy
from .transformations import (
    Affine,
    CartesianField,
    CompositionError,
    ConversionError,
    Identity,
    Sequence,
    SubspaceTransformation,
    Transformation,
    _interpolates,
)


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
    if isinstance(element, SubspaceTransformation) and _interpolates(
        element.transformation
    ):
        # An interpolating transform over a subset of axes couples every
        # one of its acted-on axes to every other. The remaining axes pass
        # through in order, paired the same way as the subspace-to-affine
        # reduction pairs them.
        dep = np.zeros((ndim_out, ndim_in), dtype=bool)
        in_axes = _axis_list(element.input_axes)
        out_axes = _axis_list(element.output_axes)
        for o in out_axes:
            for i in in_axes:
                dep[o, i] = True
        pass_in = [i for i in range(ndim_in) if i not in set(in_axes)]
        pass_out = [o for o in range(ndim_out) if o not in set(out_axes)]
        for o, i in zip(pass_out, pass_in):
            dep[o, i] = True
        return dep
    if isinstance(element, Identity):
        if ndim_in != ndim_out:
            raise _Unknown()
        return np.eye(ndim_in, dtype=bool)
    if _interpolates(element):
        # A raw field, not wrapped in a subspace, couples every axis to
        # every axis. Nothing about it is separable.
        return np.ones((ndim_out, ndim_in), dtype=bool)
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
    # The dependency matrix of every non-grid element, and the axis count
    # at every stage. Stage 0 is the grid, and stage s is the output of
    # element s. A chain whose stages do not line up is unreadable.
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
            matrix = np.asarray(element.to(Affine).matrix)
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
    if not interpolating and k == 1:
        affine = _restricted_affine(sub)
        if affine is None:
            scale, shift = 1.0, 0.0
        else:
            matrix = np.asarray(affine.matrix)
            scale, shift = float(matrix[0, 0]), float(matrix[0, 1])
        # A scale of exactly unit magnitude with an integer shift maps each
        # output sample onto one input sample, so no value is interpolated.
        if abs(abs(scale) - 1.0) == 0.0 and abs(shift - round(shift)) < 1e-9:
            kind = "a"
        else:
            kind = "b"

    _check_discrete(comp, kind, order, grid_system, data_system)

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
        step["run"] = _make_pull(
            sub, data_axes, grid_axes, order, bound, coeff
        )
    return step


def _check_discrete(
    comp: dict,
    kind: str,
    order: int,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> None:
    # A discrete axis, such as a channel or a labelled time axis, holds no
    # value between its samples, so it can only be permuted, flipped, or
    # shifted by whole samples. Such an axis must be its own group and be
    # applied as a gather. Any other case is refused.
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
    if not discrete:
        return
    if kind != "a" or len(comp["D"]) != 1:
        raise CompositionError(
            f"Cannot apply an interpolating transform along the discrete "
            f"axis {name!r}. A discrete axis holds no value between its "
            f"samples, so it can only be permuted, flipped, or shifted by "
            f"whole samples."
        )
    if order == 0:
        # A discrete axis with order 0 is refused pending a maintainer
        # decision. An order-0 gather along a discrete axis is exact and
        # could be allowed, but is raised here to keep the behaviour
        # explicit for now.
        raise CompositionError(
            f"Cannot reslice the discrete axis {name!r} with order 0. Use a "
            f"higher order, or reslice without touching this axis."
        )


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
    shape: tx.Tuple[int, ...],
    grid_system: tx.Optional[tx.Any] = None,
    data_system: tx.Optional[tx.Any] = None,
) -> tx.Any:
    """Reslice `data` through `seq`, exploiting separable axes.

    The transformation `seq` maps the coordinates of the output grid to
    the coordinates of `data`. It is factored into independent steps, one
    per group of axes that transform together, and each step is applied on
    its own. A group that needs no interpolation is a gather or a view, a
    one-dimensional affine group is a weight matrix, and the irreducible
    coupled group keeps the N-dimensional pull over the fewest axes.

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
    shape : tuple of int
        The shape of the output grid.
    grid_system : CoordinateSystem, optional
        The coordinate system of the output grid. It is read only to
        identify discrete axes.
    data_system : CoordinateSystem, optional
        The coordinate system of the data. It is read only to identify
        discrete axes.

    Returns
    -------
    array-like
        The resliced data, of shape `shape`.
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

    arr = data
    labels: tx.List[tx.Any] = [("data", i) for i in range(data.ndim)]
    for step in steps:
        arr, labels = step["run"](arr, labels)

    permutation = [labels.index(("grid", g)) for g in range(n_grid)]
    return ab.transpose(arr, permutation)
