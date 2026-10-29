# TODO: rename this module to separable_pull.

"""Reslicing of an image through the axis-group normal form of its transform.

Reslicing samples data at transformed coordinates. A transformation that
couples every output axis to every input axis requires a full
coordinates field and an N-d pull, but translations, scalings,
permutations and many other transformations do not couple everything.
The grouping of the axes is read from the normal form
`[grid, F_1, ..., F_m, Pi_perm?]` returned by
`Sequence.compute(factor=True)` under the numeric policy, which reads the
exact zeros of the matrices. In the
normal form, each factor acts on its own group of grid axes, and the
trailing permutation sends the groups to the data axes.

This module executes one step per group. A group that maps each output
sample onto an input sample, such as a flip, an integer shift or an
integer stride, is applied as a gather or a view when no value needs to
be interpolated. A one-dimensional affine group is sampled by a
one-dimensional pull along its axis, in which the other axes are batch
axes, and any other group is sampled by an N-d pull over its own axes.
The prefilter is separable, so each pull only prefilters its own axes.
The result equals that of the monolithic reslice, a single N-d pull
over all axes, to rounding errors. It is bit-identical when nothing is
separable, because the monolithic pull is then used unchanged.
"""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike

# core
from brainhops._core.backends import (
    array_backend,
    backend,
    copy_array,
    may_share_memory,
    resolve_backend,
    to_backend_of,
)
from brainhops._core.bsplines import (
    cast_output,
    clip_mask,
    interpol_boundary,
    normalize_clip,
    pull,
    pull_axes,
    sampling_dtype,
)
from brainhops._core.typing import ArrayProtocol, Clip
from brainhops._ext import interpol

# api
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import CompositionError, ConversionError

# transformations
from ..base import Transformation
from ..concrete import Affine, CartesianField, Permutation
from ..meta import SubspaceTransformation
from ..sequence import Sequence, _interpolates
from .utils import axis_list

# typing
_AxisLabels = tx.List[tx.Tuple[tx.Literal["data", "grid"], int]]
_StepResult = tx.Tuple[ArrayProtocol, _AxisLabels, tx.Optional[ArrayProtocol]]
_StepRunner = tx.Callable[[ArrayProtocol, _AxisLabels], _StepResult]

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
    clip: Clip = None,
    dtype: tx.Optional[DTypeLike] = None,
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

    clip : float, pair, sequence of pairs, or None, default=None
        Zero samples outside the closed voxel margin, as in [`pull`][].
        Each independent group builds its own mask, without a full field.

    dtype : DTypeLike, optional
        Output dtype. None preserves the input dtype. An explicit dtype
        promotes input and output dtypes to select working precision,
        as in [`pull`][], then uses the existing rounding and saturation
        rules for integer output.

    Returns
    -------
    array-like
        Resliced data, with the shape of the output grid.
    """
    clip = normalize_clip(clip, data.ndim)
    opt = dict(degree=degree, bound=bound, coeff=coeff, clip=clip)
    steps = _plan(tuple(data.shape), seq, **opt)
    if steps is None:
        # The monolithic pull writes into an array that it allocates.
        coords = _coordinates_like(seq, data)
        return _fresh(
            pull(data, coords, dtype=dtype, **opt),
            data,
            copy,
            True,
        )

    # The steps work in floating point so that an integer input is not
    # rounded between them. The gather and pull steps preserve the dtype that
    # they are given, so an integer input is converted to float once here.
    xp = array_backend(data)
    inp_dtype = np.dtype(data.dtype)
    out_dtype = np.dtype(inp_dtype if dtype is None else dtype)
    tmp_dtype = sampling_dtype(inp_dtype, dtype)
    arr = data.astype(tmp_dtype, copy=False)
    # Only the gather steps and the transpose can return a view of `data`,
    # so `fresh` records whether the result is known to be a new array.
    fresh = tmp_dtype != inp_dtype or any(
        step["kind"] != "gather" for step in steps
    )
    labels: _AxisLabels = [("data", i) for i in range(data.ndim)]
    masks = []
    for step in steps:
        arr, labels, mask = step["run"](arr, labels)
        if mask is not None:
            masks.append((mask, step["G"]))

    permutation = [labels.index(("grid", g)) for g in range(data.ndim)]
    arr = xp.transpose(arr, permutation)
    if masks:
        arr = _apply_clip_masks(arr, masks)
        fresh = True
    # An integer or boolean output is converted back with the same rules
    # as in the monolithic pull.
    if arr.dtype != out_dtype:
        fresh = True
        arr = cast_output(arr, out_dtype)
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


_SYMMETRIC_BOUNDS = (
    "mirror",
    "reflect",
    "grid-wrap",
    "antimirror",
    "antireflect",
)
"""The boundaries of `interpol` under which a spline extends its values.

Under these boundaries, the extension of the coefficients is symmetric
or antisymmetric about each edge, and the spline values at the integers
beyond the edges are the same extension of the values inside. A gather
along the boundary therefore gives the samples of the monolithic pull
at every integer coordinate. The replicate and the zero boundaries
extend the coefficients instead, and a spline of degree 2 or more then
takes other values than the gathered ones beyond the edges.
"""


def _classify(
    group: dict,
    shape: tx.Tuple[int, ...],
    data_shape: tx.Tuple[int, ...],
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
    clip: Clip = None,
) -> dict:
    # Build one executable step for a group. The step is a gather when no
    # value is interpolated, a one-dimensional pull for any other
    # one-dimensional affine group, and an N-d pull for a coupled group.
    grid_axes = group["G"]
    data_axes = group["D"]
    clip = [clip[d] for d in data_axes] if clip is not None else None
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
        # An integer scale with an integer shift maps each output sample
        # onto one input sample, so that no value lies between samples.
        integer_shift = abs(shift - round(shift)) < 1e-9
        whole = scale != 0 and scale == round(scale) and integer_shift
        unit = whole and abs(scale) == 1.0
        kind = "pull1d"
        if whole:
            scale, shift = int(round(scale)), int(round(shift))
            coords = np.arange(n_out) * scale + shift
            inside = n_out == 0 or (
                coords.min() >= 0 and coords.max() <= n_in - 1
            )
            # A gather returns the input samples themselves. The pull
            # returns the same samples when the data holds values, since
            # the prefilter is exact, or when the degree is 0 or 1. Beyond
            # the edges, this only holds for the boundaries that extend
            # the values (see `_SYMMETRIC_BOUNDS`). The pull returns a
            # reconstruction instead when the data holds coefficients at
            # degree 2 or more.
            name = interpol_boundary(bound)[0]
            same = degree <= 1 or (
                not coeff and (inside or name in _SYMMETRIC_BOUNDS)
            )
            # A discrete axis holds no value between samples, so a
            # whole-sample map along it is always a gather and is never
            # blended.
            if same or discrete:
                kind = "gather"

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
        step["view"], step["run"] = _make_gather(
            data_axes[0],
            grid_axes[0],
            scale,
            shift,
            bound,
            data_shape,
            shape,
            clip,
        )
    elif kind == "pull1d":
        step["run"] = _make_pull1d(
            data_axes[0],
            grid_axes[0],
            scale,
            shift,
            degree,
            bound,
            coeff,
            data_shape,
            shape,
            clip,
        )
    else:
        step["run"] = _make_pull(
            inner,
            data_axes,
            grid_axes,
            shape,
            degree,
            bound,
            coeff,
            data_shape,
            clip,
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
    # over its input size, and its weight w estimates the work per input
    # element. A view costs nothing, and a gather copies each output
    # element once. A pull over k axes adds (degree + 1) ** k taps per
    # output element, and its prefilter, when the data holds values at
    # degree 2 or more, makes about `degree` passes per axis over the
    # input. Steps that shrink the array run first, and steps that keep
    # its size run next, from the cheapest to the most expensive. Steps
    # that grow the array run last.
    for step in steps:
        k = step["k"]
        n_in = step["n_in"]
        n_out = step["n_out"]
        r = (n_out / n_in) if n_in else 1.0
        if step["kind"] == "gather":
            a = 0.0
            c = 0.0 if step["view"] else 1.0
        else:
            a = float(k * degree) if (degree > 1 and not coeff) else 0.0
            c = float((degree + 1) ** k)
        step["_r"] = r
        step["_w"] = a + c * r

    shrink = [s for s in steps if s["_r"] < 1]
    unit = [s for s in steps if s["_r"] == 1]
    expand = [s for s in steps if s["_r"] > 1]

    shrink.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    unit.sort(key=lambda s: s["_w"])
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
    clip: Clip = None,
) -> tx.Optional[tx.List[dict]]:
    # Return the ordered steps that reslice data of shape `data_shape`
    # through `seq`, or None when `seq` does not factor into at least two
    # groups, in which case the monolithic pull is used. A transformation
    # that cannot be computed is also left to the monolithic pull, which
    # reports the malformed input.
    clip = normalize_clip(clip, len(data_shape))
    try:
        # Resampling reads every value anyway, so the groups are found
        # from the exact zeros of the matrices.
        nf = seq.compute(mode=kinds.Affine, factor=True, simplify="numeric")
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
            clip,
        )
        for group in groups
    ]
    return _order_steps(steps, degree, coeff)


# ----------------------------------------------------------------------
#   EXECUTORS
# ----------------------------------------------------------------------


def _as_slice(index: np.ndarray) -> tx.Optional[slice]:
    # Return the slice that selects these indices, or None when they do
    # not form an arithmetic progression with a nonzero step.
    if index.size == 0:
        return None
    start = int(index[0])
    if index.size == 1:
        return slice(start, start + 1)
    steps = np.diff(index)
    step = int(steps[0])
    if step == 0 or not bool((steps == step).all()):
        return None
    stop = int(index[-1]) + step
    return slice(start, stop if stop >= 0 else None, step)


def _make_gather(
    d: int,
    g: int,
    scale: int,
    shift: int,
    bound: tx.Union[str, float],
    data_shape: tx.Tuple[int, ...],
    shape: tx.Tuple[int, ...],
    clip: Clip,
) -> tx.Tuple[bool, _StepRunner]:
    # Return whether the gather is a view of its input, and the runner.
    n_in = data_shape[d]
    n_out = shape[g]
    name, cval = interpol_boundary(bound)
    coords = np.arange(n_out, dtype=np.float64) * scale + shift
    # The boundary folds each coordinate onto a sample of the input, and
    # may flip its sign or give zero. Sampling the ramp 1, 2, ..., n_in
    # with degree 0 at the coordinates, which are integers, returns the
    # sample number with that sign, or zero for a sample beyond the edge
    # of the zero boundary, or on the zero of `antimirror`.
    ramp = np.arange(1, n_in + 1, dtype=np.float64)
    signed = np.rint(interpol.pull(ramp, coords[:, None], 0, name))
    signed = signed.astype(np.intp)
    sign = np.sign(signed)
    outside = sign == 0
    index = np.where(outside, 0, np.abs(signed) - 1)
    view = _as_slice(index) if bool((sign == 1).all()) else None

    def run(arr: ArrayProtocol, labels: _AxisLabels) -> _StepResult:
        xp = array_backend(arr)
        axis = labels.index(("data", d))
        shape = [1] * arr.ndim
        shape[axis] = n_out
        if view is not None:
            selection = [slice(None)] * arr.ndim
            selection[axis] = view
            result = arr[tuple(selection)]
        else:
            result = xp.take(arr, xp.asarray(index), axis=axis)
            if bool((sign < 0).any()):
                factor = np.where(outside, 1, sign).astype(arr.dtype)
                factor = to_backend_of(factor.reshape(shape), result)
                result = result * factor
            if bool(outside.any()):
                mask = to_backend_of(outside.reshape(shape), result)
                result = xp.where(mask, cval, result)
        labels = list(labels)
        labels[axis] = ("grid", g)
        mask = clip_mask(coords[..., None], (n_in,), clip)
        return result, labels, mask

    return view is not None, run


def _make_pull1d(
    d: int,
    g: int,
    scale: float,
    shift: float,
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
    data_shape: tx.Tuple[int, ...],
    shape: tx.Tuple[int, ...],
    clip: Clip,
) -> _StepRunner:
    n_in = data_shape[d]
    n_out = shape[g]
    # The coordinates are a small NumPy array, which the pull moves to
    # the device of the data.
    coords = (np.arange(n_out, dtype=np.float64) * scale + shift)[:, None]

    def run(arr: ArrayProtocol, labels: _AxisLabels) -> _StepResult:
        axis = labels.index(("data", d))
        moved = pull_axes(arr, coords, [axis], degree, bound, coeff)
        remaining = [label for i, label in enumerate(labels) if i != axis]
        labels = remaining + [("grid", g)]
        mask = clip_mask(coords, (n_in,), clip)
        return moved, labels, mask

    return run


def _coordinates_like(seq: Sequence, array: ArrayProtocol) -> ArrayProtocol:
    """Return the coordinates at which a reslice samples an array.

    The coordinates are the field of a sequence that starts with the grid
    of the output. A grid creates its coordinates with the backend
    setting in effect, so the field is computed under the setting that
    names the backend of the array, including its device. The coordinates
    are then created where the array is, rather than with NumPy and moved
    afterwards.
    """
    with backend(resolve_backend(array, "native")):
        return seq.compute().field


def _make_pull(
    inner: tx.Optional[Transformation],
    data_axes: tx.List[int],
    grid_axes: tx.List[int],
    shape: tx.Tuple[int, ...],
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
    data_shape: tx.Tuple[int, ...],
    clip: Clip,
) -> _StepRunner:
    chain: tx.List[Transformation] = [
        CartesianField(shape=tuple(shape[g] for g in grid_axes))
    ]
    if inner is not None:
        chain.append(inner)

    def run(arr: ArrayProtocol, labels: _AxisLabels) -> _StepResult:
        coords = _coordinates_like(Sequence(transformations=chain), arr)
        positions = [labels.index(("data", d)) for d in data_axes]
        moved = pull_axes(arr, coords, positions, degree, bound, coeff)
        used = set(positions)
        remaining = [labels[i] for i in range(len(labels)) if i not in used]
        labels = remaining + [("grid", g) for g in grid_axes]
        mask = clip_mask(coords, [data_shape[d] for d in data_axes], clip)
        return moved, labels, mask

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


def _apply_clip_masks(
    arr: ArrayProtocol, masks: tx.List[tx.Tuple[ArrayProtocol, tx.List[int]]]
) -> ArrayProtocol:
    """Zero the union of group cutoffs after every interpolation step.

    Delaying the masks prevents a later nonzero constant boundary from
    bringing clipped samples back. Masks stay lazy and only span the axes
    of their independent group.
    """
    xp = array_backend(arr)
    for mask, axes in masks:
        order = sorted(range(len(axes)), key=lambda index: axes[index])
        mask = xp.transpose(xp.asarray(to_backend_of(mask, arr)), order)
        shape = [1] * arr.ndim
        for index, axis in enumerate(sorted(axes)):
            shape[axis] = mask.shape[index]
        arr = xp.where(mask.reshape(tuple(shape)), arr, xp.zeros_like(arr))
    return arr
