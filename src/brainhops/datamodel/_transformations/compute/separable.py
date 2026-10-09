# FIXME: module should be renamed to separable_pull

"""Reslice an image by executing the normal form of its transformation.

A reslice applies a voxel-to-voxel transformation to image data by
sampling the data at transformed coordinates. When the transformation
couples every output axis to every input axis, the whole coordinates
field must be built and the data sampled with an N-dimensional pull.
Many transformations do not couple every axis (for example: translations,
scalings, permutations, or transformations that act on a subset of the axes).

Which axes transform together is not decided here. It is read from the
axis-group normal form that `Sequence.compute(factor=True)` returns (see
the `factor` module):

    [grid, F_1, ..., F_m, Pi_perm?]

Each factor `F_i` is a subspace that acts on its own group of grid axes,
and the trailing permutation sends each group to its data axes. This
module only executes that normal form, one step per group. A group that
needs no interpolation, such as a flip or an integer shift, is applied as
a gather or a view. A group that is a one-dimensional affine is applied as
a precomputed weight matrix. Any other group keeps the N-dimensional pull,
over its own axes only.

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
from brainhops.backends import copy_array, get_array_backend, may_share_memory
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import CompositionError, ConversionError

# locals
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
    """Reslice `data` through `seq`, exploiting separable axes.

    The transformation `seq` maps the coordinates of the output grid to
    the coordinates of `data`. Its leading element is the sampling grid, a
    [`CartesianField`][brainhops.datamodel.transformations.CartesianField]
    whose shape is the shape of the output grid and whose output system is
    the coordinate system of that grid. The system in which `seq` leaves
    its coordinates is the coordinate system of the data. The output shape
    and both coordinate systems are read from `seq`, so they are not passed
    separately.

    `seq` is computed into its axis-group normal form
    (`compute(factor=True)`), and each group is applied on its own. A group
    that needs no interpolation is a gather or a view, a one-dimensional
    affine group is a weight matrix, and any other group keeps the
    N-dimensional pull over its own axes.

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
    degree : {0..5}
        The spline degree.
    bound : str or float
        The boundary condition, as accepted by
        [`pull`][brainhops._core.bsplines.pull].
    coeff : bool
        Whether the data already contains spline coefficients.
    copy : bool
        Whether the result must be a fresh array. As with
        `torch.Tensor.to`, when `False` the result may share memory with
        `data`: a reslice that only gathers (a flip, a permutation, or a
        unit-step slice, such as a reslice onto the data's own grid) can
        return a view of it. When `True` the result never shares memory
        with `data`. No copy is made when the result is already fresh, nor
        on the dask backend: a dask array is immutable, and writing into
        the result rebinds its own graph, never that of `data`, so a lazy
        result is returned as is. (Its computed value may still be a view
        of the numpy array a gather-only graph was built from, which is
        outside what `copy` promises.)

    Returns
    -------
    array-like
        The resliced data, of the shape of the output grid.
    """
    opt = dict(degree=degree, bound=bound, coeff=coeff)
    steps = _plan(tuple(data.shape), seq, **opt)
    if steps is None:
        # The monolithic pull writes into an array it allocates, so its
        # result is always fresh.
        return _fresh(pull(data, seq.compute().field, **opt), data, copy, True)

    # The pipeline runs in a floating working dtype so an integer input is
    # not rounded between steps. The weight-matrix step already produces a
    # float64 result, and the gather and pull steps preserve the dtype they
    # are given, so an integer input is lifted to float once here.
    ab = get_array_backend(data)
    out_dtype = np.dtype(data.dtype)
    floating = np.issubdtype(out_dtype, np.floating)
    arr = data if floating else data.astype(float)
    # Whether `arr` is known to be a new array. A lift to float, a step
    # that interpolates, and the final cast each allocate one. Only the
    # gather steps and the transpose can return a view of `data`.
    fresh = not floating or any(step["kind"] != "gather" for step in steps)
    labels: tx.List[tx.Any] = [("data", i) for i in range(data.ndim)]
    for step in steps:
        arr, labels = step["run"](arr, labels)

    permutation = [labels.index(("grid", g)) for g in range(data.ndim)]
    arr = ab.transpose(arr, permutation)
    # The assembled array is cast back to the input dtype exactly once, at
    # the end. The monolithic pull returns the input dtype and reproduces
    # scipy's `map_coordinates`, so an integer or boolean output is finished
    # the same way scipy's `CASE_INTERP_OUT_INT` does.
    if arr.dtype != out_dtype:
        fresh = True
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
    # The sampling grid of a normal form and its axis groups, or `None`
    # when `nf` is not a normal form over `ndim` axes. A group holds its
    # grid axes `G`, its data axes `D` (paired with `G`), and its `inner`,
    # the transformation from the first to the second. The factor pass
    # drops a factor whose inner is the identity, so a grid axis that no
    # factor names is a group of its own, with no inner.
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
    # The permutation sends grid axis `perm[d]` to data axis `d`.
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
    # The steps are planned in grid-axis order, so the order in which two
    # steps of equal cost run (and thus the rounding) does not depend on
    # which groups the factor pass dropped.
    groups.sort(key=lambda group: min(group["G"]))
    return grid, groups


def _line(inner: tx.Optional[Transformation]) -> tx.Optional[tuple]:
    # The scale and shift of a one-dimensional non-interpolating inner, or
    # `None` when it has no affine reading (a group whose pieces could not
    # be composed), in which case the group is pulled.
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


# The largest weight matrix a one-dimensional interpolating step builds
# before it falls back to the batched pull. The weight matrix has
# `n_out * n_in` elements, so a long axis makes it enormous. Above this
# many elements the batched pull is used instead, which samples the axis
# without materializing the matrix. The two paths give the same result, so
# the threshold trades memory for the matrix's speed and nothing else. Four
# million elements is about 32 MiB at float64, small enough to always hold.
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
    # Build one executable step for a group. The step is classified as a
    # gather (no interpolation), a weight matrix (a one-dimensional
    # affine), or a batched pull (a coupled or interpolating group).
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
        # output sample onto one input sample, so no value is interpolated.
        # Such a step is a whole-sample map.
        integer_shift = abs(shift - round(shift)) < 1e-9
        unit = abs(abs(scale) - 1.0) == 0.0 and integer_shift
        # The gather returns the input sample itself. That matches the
        # monolithic pull only when the pull returns the same sample. With
        # spline coefficients at degree two or above the pull returns the
        # reconstruction of the coefficients, not the raw coefficient, and
        # scipy's `reflect` prefilter is not exactly interpolating above
        # degree one. For a continuous axis those cases take the weight-matrix
        # path instead, which reproduces the reconstruction exactly.
        gather = degree <= 1 or (not coeff and bound != "reflect")
        # A discrete axis holds no value between its samples, so a
        # whole-sample map along it is always an exact gather. It must never
        # be interpolated or blended across, even at degree two and above
        # where the monolithic pull would mix its samples.
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
        # A coupled group, or a weight matrix too large to hold, which is
        # sampled with the batched pull instead. The result is the same as
        # the weight-matrix path.
        step["run"] = _make_pull(
            inner, data_axes, grid_axes, shape, degree, bound, coeff
        )
    return step


def _discrete_axis(
    group: dict,
    grid_system: tx.Optional[tx.Any],
    data_system: tx.Optional[tx.Any],
) -> tx.Tuple[bool, tx.Optional[str]]:
    # Whether this group touches a discrete axis, and that axis's name. A
    # discrete axis, such as a channel or a labelled time axis, holds no
    # value between its samples.
    discrete = False
    name: tx.Optional[str] = None
    for axis_index, system in (
        (group["G"], grid_system),
        (group["D"], data_system),
    ):
        # Positional access reads the axes an open system states, and an
        # unknown axis, which is not discrete, anywhere else.
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
    # A discrete axis holds no value between its samples, so it can only be
    # permuted, flipped, or shifted by whole samples. Such an axis must be
    # its own group and its step must be a whole-sample map, which is
    # always a gather. Any other case is refused.
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
        if kind == "gather":
            c = 1.0
        elif kind == "matrix":
            # The matmul touches every input sample once per output sample,
            # so its per-element cost times the ratio is the output size.
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
    # For an expanding step `1 - r` is negative, so a smaller expansion
    # gives a more negative key and sorts first.
    expand.sort(key=lambda s: s["_w"] / (1 - s["_r"]))
    return shrink + unit + expand


def _plan(
    data_shape: tx.Tuple[int, ...],
    seq: Transformation,
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Optional[tx.List[dict]]:
    # The ordered steps that reslice data of shape `data_shape` through
    # `seq`, or `None` when `seq` does not factor into at least two groups,
    # in which case the monolithic pull is used. A transformation that
    # cannot be computed is left to the monolithic pull, which then reports
    # the malformed input.
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
    # The grid's system and the system in which `seq` leaves its
    # coordinates are read only to identify discrete axes.
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
        # Degree-0 weights turn each output coordinate into the single input
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
    inner: tx.Optional[Transformation],
    data_axes: tx.List[int],
    grid_axes: tx.List[int],
    shape: tx.Tuple[int, ...],
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> tx.Callable:
    # The group's coordinates are those of its inner over its own grid
    # axes, so the field is built over the group's axes only.
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
    """Return `arr`, copied if `copy` is set and it may alias `data`.

    `fresh` says the caller knows `arr` is a new array, so no copy is
    needed. Otherwise the backend is asked whether the two may share
    memory, and an array of a backend that cannot tell is copied
    regardless. A dask result never shares memory with its input, so it is
    never copied.
    """
    if not copy or fresh or may_share_memory(arr, data) is False:
        return arr
    return copy_array(arr)
