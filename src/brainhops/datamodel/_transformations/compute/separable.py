# TODO: rename this module to separable_pull.

"""Reslicing of an image through the axis-group normal form of its transform.

Reslicing samples data at transformed coordinates. A transformation that
couples every output axis to every input axis requires a full coordinates
field and an N-d pull, but translations, scalings, permutations and many
other transformations do not couple everything. The grouping of the axes is
read from the normal form `[grid, F_1, ..., F_m, Pi_perm?]` returned by
`Sequence.compute(factor=True)`, in which each factor acts on its own group
of grid axes and the trailing permutation sends the groups to the data
axes.

This module executes one step per group. A group that needs no
interpolation, such as a flip or an integer shift, is applied as a gather or
a view. A one-dimensional affine group is applied with a precomputed weight
matrix, and any other group is sampled by an N-d pull over its own axes. The
result equals that of the monolithic reslice, a single N-d pull over all
axes, within interpolation tolerance. It is bit-identical when nothing is
separable, because the monolithic pull is then used unchanged.
"""

import numpy as np
import typing_extensions as tx

from brainhops._core.bsplines import pull, pull_axes, spline_matrix
from brainhops.backends import copy_array, get_array_backend, may_share_memory
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import CompositionError, ConversionError

from ..base import Transformation
from ..concrete import Affine, CartesianField, Permutation
from ..meta import SubspaceTransformation
from ..sequence import Sequence, _interpolates
from .utils import axis_list

# ======================================================================
#
#                              P U B L I C
#
# ======================================================================


def pull_separable(
    data: tx.Any,
    seq: Transformation,
    *,
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
    copy: bool = False,
) -> tx.Any:
    """Reslice data through a transformation, exploiting separable axes.

    `seq` maps output grid coordinates to data coordinates. Its leading element
    is the sampling grid, a
    [`CartesianField`][brainhops.datamodel.transformations.CartesianField]
    that gives the output shape and the coordinate system of the grid, while
    the system in which `seq` leaves the coordinates is that of the data. The
    transformation is computed into its axis-group normal form, with affine
    composition allowed, and each group is applied on its own.

    Parameters
    ----------
    data : Any
        Image to reslice.
    seq : Transformation
        Uncomputed transformation from output grid coordinates to data
        coordinates.
    degree : int
        Spline degree, from 0 to 5.
    bound : str or float
        Boundary condition, as accepted by
        [`pull`][brainhops._core.bsplines.pull].
    coeff : bool
        Whether `data` already holds spline coefficients.
    copy : bool, default=False
        Whether the result must be a fresh array. When `copy` is False, as with
        `torch.Tensor.to`, a reslice made only of gathers (a flip, a
        permutation or a unit-step slice) may return a view of `data`. No copy
        is made when the result is already fresh, nor on the immutable dask
        backend.

    Returns
    -------
    array-like
        Resliced data, with the shape of the output grid.
    """
    opt = dict(degree=degree, bound=bound, coeff=coeff)
    steps = _plan(tuple(data.shape), seq, **opt)
    if steps is None:
        # The monolithic pull writes into an array that it allocates.
        return _fresh(pull(data, seq.compute().field, **opt), data, copy, True)

    # The steps work in floating point so that an integer input is not
    # rounded between them. The gather and pull steps preserve the dtype that
    # they are given, so an integer input is converted to float once here.
    ab = get_array_backend(data)
    out_dtype = np.dtype(data.dtype)
    floating = np.issubdtype(out_dtype, np.floating)
    arr = data if floating else data.astype(float)
    # Only the gather steps and the transpose can return a view of `data`,
    # so `fresh` records whether the result is known to be a new array.
    fresh = not floating or any(step["kind"] != "gather" for step in steps)
    labels: tx.List[tx.Any] = [("data", i) for i in range(data.ndim)]
    for step in steps:
        arr, labels = step["run"](arr, labels)

    permutation = [labels.index(("grid", g)) for g in range(data.ndim)]
    arr = ab.transpose(arr, permutation)
    # The monolithic pull reproduces scipy's map_coordinates, so an integer
    # or boolean output is converted back in the same way as in scipy.
    if arr.dtype != out_dtype:
        fresh = True
        if out_dtype.kind == "b":
            # scipy truncates toward zero for a boolean output, so a value
            # such as 0.6 becomes False.
            arr = ab.trunc(arr).astype(out_dtype)
        elif np.issubdtype(out_dtype, np.integer):
            # scipy rounds an integer output half away from zero, then clips
            # it to the dtype range, so that an overshoot saturates instead
            # of wrapping.
            rounded = ab.trunc(ab.where(arr > 0, arr + 0.5, arr - 0.5))
            info = np.iinfo(out_dtype)
            arr = ab.clip(rounded, info.min, info.max).astype(out_dtype)
        else:
            arr = arr.astype(out_dtype)
    return _fresh(arr, data, copy, fresh)


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================

# ----------------------------------------------------------------------
#   READING THE NORMAL FORM
# ----------------------------------------------------------------------


def _groups(
    nf: Transformation, ndim: int
) -> tx.Optional[tx.Tuple[CartesianField, tx.List[dict]]]:
    # Read the sampling grid and the axis groups of a normal form, or return
    # None when `nf` is not a normal form over `ndim` axes. A group holds its
    # grid axes G, the data axes D that are paired with them, and the inner
    # transformation from G to D. The factor pass drops identity factors, so
    # a grid axis that no factor names forms its own group without an inner
    # transformation.
    if isinstance(nf, CartesianField):
        grid, body = nf, []
    elif isinstance(nf, Sequence) and nf.transformations:
        grid, *body = nf.transformations
    else:
        return None
    if not isinstance(grid, CartesianField):
        return None
    if grid.shape is None or len(grid.shape) != ndim:
        return None
    perm = list(range(ndim))
    if body and isinstance(body[-1], Permutation):
        permutation = body.pop().permutation
        if permutation is not None:
            perm = axis_list(permutation)
    if sorted(perm) != list(range(ndim)):
        return None
    # The permutation sends grid axis perm[d] to data axis d.
    data_axis = {a: d for d, a in enumerate(perm)}
    free = set(range(ndim))
    groups: tx.List[dict] = []
    for factor in body:
        if not isinstance(factor, SubspaceTransformation):
            return None
        axes = axis_list(factor.input_axes)
        if not axes or axes != axis_list(factor.output_axes):
            return None
        if len(set(axes)) != len(axes) or not free.issuperset(axes):
            # Overlapping factors are not independent steps.
            return None
        free.difference_update(axes)
        groups.append(
            {
                "G": axes,
                "D": [data_axis[a] for a in axes],
                "inner": factor.transformation,
            }
        )
    for a in free:
        groups.append({"G": [a], "D": [data_axis[a]], "inner": None})
    # The steps are planned in grid-axis order. As a result, the order in
    # which two steps of equal cost run, and therefore the rounding, does not
    # depend on which factors the factor pass dropped.
    groups.sort(key=lambda group: min(group["G"]))
    return grid, groups


def _line(inner: tx.Optional[Transformation]) -> tx.Optional[tuple]:
    # Return the scale and shift of a one-dimensional inner transformation.
    # None means that the inner transformation has no one-dimensional affine
    # form, for example because the pieces of its group could not be
    # composed, and the group is then sampled with a pull.
    if inner is None:
        return 1.0, 0.0
    try:
        matrix = inner.to(Affine).matrix
    except ConversionError:
        return None
    if matrix is None:
        return 1.0, 0.0
    matrix = np.asarray(matrix)
    if matrix.shape != (1, 2):
        return None
    return float(matrix[0, 0]), float(matrix[0, 1])


# ----------------------------------------------------------------------
#   PLANNING
# ----------------------------------------------------------------------


# This constant is the largest number of elements, n_out * n_in, that the
# weight matrix of a one-dimensional step may have. A larger matrix is not
# built, and the axis is sampled by a batched pull instead, which gives the
# same result without materializing the matrix. Four million elements take
# about 32 MiB at float64.
_MAX_WEIGHT_MATRIX_ELEMENTS = 4_000_000


def _classify(
    group: dict,
    shape: tx.Tuple[int, ...],
    data_shape: tx.Tuple[int, ...],
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> dict:
    # Build one executable step for a group. The step is a gather when no
    # value is interpolated, a weight matrix for a one-dimensional affine
    # group, and a batched pull for a coupled or interpolating group.
    grid_axes = group["G"]
    data_axes = group["D"]
    inner = group["inner"]
    k = len(data_axes)

    n_in = int(np.prod([data_shape[d] for d in data_axes]))
    n_out = int(np.prod([shape[g] for g in grid_axes]))

    kind = "pull"
    scale = shift = None
    unit = False
    discrete = _discrete_axis(group, grid_system, data_system)[0]
    line = None
    if k == 1 and not _interpolates(inner):
        line = _line(inner)
    if line is not None:
        scale, shift = line
        # A scale of exactly unit magnitude with an integer shift maps each
        # output sample to one input sample, so no interpolation is needed.
        integer_shift = abs(shift - round(shift)) < 1e-9
        unit = abs(abs(scale) - 1.0) == 0.0 and integer_shift
        # A gather returns the input sample itself, which matches the
        # monolithic pull only when the pull returns the same sample. The
        # pull returns a reconstruction instead when the data holds spline
        # coefficients at degree 2 or more, and scipy's `reflect` prefilter
        # is not exactly interpolating above degree 1. A continuous axis in
        # those cases uses the weight matrix, which reproduces the pull.
        gather = degree <= 1 or (not coeff and bound != "reflect")
        # A discrete axis holds no value between samples, so a whole-sample map
        # along it is always an exact gather and is never blended.
        kind = "gather" if unit and (gather or discrete) else "matrix"

    _check_discrete(group, unit, grid_system, data_system)

    step = {
        "D": data_axes,
        "G": grid_axes,
        "k": k,
        "kind": kind,
        "n_in": n_in,
        "n_out": n_out,
    }
    if kind == "gather":
        step["run"] = _make_gather(
            data_axes[0], grid_axes[0], scale, shift, bound, data_shape, shape
        )
    elif kind == "matrix" and n_in * n_out <= _MAX_WEIGHT_MATRIX_ELEMENTS:
        step["run"] = _make_matrix(
            data_axes[0],
            grid_axes[0],
            scale,
            shift,
            degree,
            bound,
            coeff,
            data_shape,
            shape,
        )
    else:
        # A coupled group, or a weight matrix that is too large, gives a
        # batched pull.
        step["run"] = _make_pull(
            inner, data_axes, grid_axes, shape, degree, bound, coeff
        )
    return step


def _discrete_axis(
    group: dict,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> tx.Tuple[bool, tx.Optional[str]]:
    # Decide whether the group touches a discrete axis, such as a channel
    # axis, and return the name of that axis.
    discrete = False
    name: tx.Optional[str] = None
    for axis_index, system in (
        (group["G"], grid_system),
        (group["D"], data_system),
    ):
        # Positional access reads the axes that an open system states, and
        # an axis that the system does not state is treated as continuous.
        axes = get_axes(system)
        for a in axis_index:
            if axes.ndim is not None and a >= axes.ndim:
                continue
            axis = axes.at(a)
            if getattr(axis, "discrete", None):
                discrete = True
                name = getattr(axis, "name", None) or getattr(
                    axis, "type", None
                )
    return discrete, name


def _check_discrete(
    group: dict,
    unit: bool,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> None:
    # A discrete axis may only be permuted, flipped or shifted by whole
    # samples, so it must form its own group with a gather step.
    discrete, name = _discrete_axis(group, grid_system, data_system)
    if discrete and (not unit or len(group["D"]) != 1):
        raise CompositionError(
            f"Cannot apply an interpolating transform along the discrete "
            f"axis {name!r}. A discrete axis holds no value between its "
            f"samples, so it can only be permuted, flipped, or shifted by "
            f"whole samples."
        )


def _order_steps(
    steps: tx.List[dict], degree: int, coeff: bool
) -> tx.List[dict]:
    # The steps act on disjoint axes and commute, so they are ordered to keep
    # the intermediate arrays small. The ratio r of a step is its output size
    # over its input size, and its weight w estimates the work per element.
    # Steps that shrink the array run first, and steps that keep its size run
    # next, with the free views first. Steps that grow the array run last.
    for step in steps:
        k = step["k"]
        kind = step["kind"]
        n_in = step["n_in"]
        n_out = step["n_out"]
        r = (n_out / n_in) if n_in else 1.0
        if kind == "gather":
            c = 1.0
        elif kind == "matrix":
            c = float(n_in)
        else:
            c = float((degree + 1) ** k)
        a = k * degree if (degree > 1 and not coeff) else 0
        step["_r"] = r
        step["_w"] = a + c * r

    shrink = [s for s in steps if s["_r"] < 1]
    unit = [s for s in steps if s["_r"] == 1]
    expand = [s for s in steps if s["_r"] > 1]

    shrink.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    unit.sort(key=lambda s: 0 if s["kind"] == "gather" else 1)
    # For a growing step 1 - r is negative, so a smaller growth gives a more
    # negative key and sorts first.
    expand.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    return shrink + unit + expand


def _plan(
    data_shape: tx.Tuple[int, ...],
    seq: Transformation,
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Optional[tx.List[dict]]:
    # Return the ordered steps that reslice data of shape `data_shape`
    # through `seq`, or None when `seq` does not factor into at least two
    # groups, in which case the monolithic pull is used. A transformation
    # that cannot be computed is also left to the monolithic pull, which
    # reports the malformed input.
    try:
        nf = seq.compute(mode=kinds.Affine, factor=True)
    except Exception:
        return None
    read = _groups(nf, len(data_shape))
    if read is None:
        return None
    grid, groups = read
    if len(groups) == 1:
        return None
    # The systems are read only to identify discrete axes.
    steps = [
        _classify(
            group,
            tuple(grid.shape),
            tuple(data_shape),
            degree,
            bound,
            coeff,
            grid.output,
            seq.output,
        )
        for group in groups
    ]
    return _order_steps(steps, degree, coeff)


# ----------------------------------------------------------------------
#   EXECUTORS
# ----------------------------------------------------------------------


def _is_unit_progression(idx: np.ndarray) -> bool:
    if idx.size < 1:
        return False
    if idx.size == 1:
        return True
    steps = np.diff(idx)
    return bool((steps == 1).all() or (steps == -1).all())


def _slice_axis(arr: tx.Any, axis: int, idx: np.ndarray) -> tx.Any:
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
        # Degree-0 weights map each output coordinate to the input sample
        # that it lands on. A row that sums to zero lies outside the grid
        # and is filled with the boundary constant, or with zero when the
        # boundary is given by name.
        # TODO: spline_matrix builds an n_in x n_in identity here, which
        # takes O(n_in**2) memory. Index arithmetic per boundary mode would
        # avoid that cost, but it must reproduce scipy's boundary indices
        # exactly.
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
    degree: int,
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
        weights = spline_matrix(n_in, coords, degree, bound, coeff)
        out = ab.moveaxis(
            ab.tensordot(arr, weights, axes=([axis], [1])), -1, axis
        )
        if not isinstance(bound, str):
            # A constant boundary is not linear, so the fill value is added
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
    inner: tx.Optional[Transformation],
    data_axes: tx.List[int],
    grid_axes: tx.List[int],
    shape: tx.Tuple[int, ...],
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Callable:
    chain: tx.List[Transformation] = [
        CartesianField(shape=tuple(shape[g] for g in grid_axes))
    ]
    if inner is not None:
        chain.append(inner)

    def run(arr: tx.Any, labels: tx.List[tx.Any]) -> tx.Tuple[tx.Any, list]:
        coords = Sequence(transformations=chain).compute().field
        positions = [labels.index(("data", d)) for d in data_axes]
        moved = pull_axes(arr, coords, positions, degree, bound, coeff)
        used = set(positions)
        remaining = [labels[i] for i in range(len(labels)) if i not in used]
        labels = remaining + [("grid", g) for g in grid_axes]
        return moved, labels

    return run


# ----------------------------------------------------------------------
#   COPY
# ----------------------------------------------------------------------


def _fresh(arr: tx.Any, data: tx.Any, copy: bool, fresh: bool) -> tx.Any:
    """Return `arr`, copied if `copy` is set and `arr` may alias `data`.

    No copy is made when `fresh` states that `arr` is new. Otherwise the
    backend is asked whether the arrays may share memory, and a backend that
    cannot tell leads to a copy.
    """
    if not copy or fresh or may_share_memory(arr, data) is False:
        return arr
    return copy_array(arr)
