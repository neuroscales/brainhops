"""Coarser levels of a pyramid, built from its finest level.

A pyramid stores the same image or field at several resolutions, so
that a viewer or an algorithm that only needs a coarse view reads less
data. This module holds the arithmetic with which the multiscale classes
of the data model build their coarser levels, in their method
`with_scales`.

A level is downsampled from a finer level by an integer factor `f` along
each axis, and a factor of 1 leaves the axis as it is. How the voxels of
the coarse level are laid over those of the finer level is the
alignment, `align`, and [`level_map`][] returns the shape, the actual
scale `s` and the shift `t` of each axis, so that the coarse voxel `i`
lies at the fine voxel `s * i + t`:

- `"exact"`, the default, crops an axis of `n` voxels to the multiple of
  `f` below it, and keeps the factor exactly: the coarse level has
  `n // f` voxels, and its voxel `i` covers the fine voxels `f * i` to
  `f * i + f - 1`, so that `s = f` and `t = (f - 1) / 2`. This is the
  rule that OME-NGFF 0.6 states for a level downsampled by a factor `f`
  (ngff-spec tag `0.6` at 89da1da, `index.md`, section "multiscales
  metadata"). The windows of `f` voxels do not overlap, so a coarse
  voxel reads only the fine voxels of its own window.
- `"edges"` keeps the outer edges of the field of view: the coarse level
  has `ceil(n / f)` voxels, the actual scale is `s = n / ceil(n / f)`,
  and `t = (s - 1) / 2`, so that the fine voxel of the coarse voxel `i`
  is `(i + 1 / 2) * s - 1 / 2`.
- `"centers"` keeps the centres of the first and the last voxels: the
  coarse level has `ceil((n - 1) / f) + 1` voxels, the actual scale is
  `s = (n - 1) / (ceil((n - 1) / f))`, and `t = 0`. An axis of one voxel
  is left as it is.

With `"edges"` and `"centers"`, the finer level is smoothed by the
kernel of the mode, sized from the actual scale, and then pulled at the
positions of the coarse voxels with the B-splines of
[`brainhops._ext.interpol`][], through
[`pull_axes`][brainhops._core.bsplines.pull_axes]: of degree 1, which
interpolates linearly, after smoothing, and of degree 0, the nearest
voxel, in the mode `"nearest"`. The interpolation is separable, one
axis after the other, which gives the same values as an interpolation
over every axis at once, because a level only scales and shifts each
axis. The axis that shrinks the most goes first, so that the next
passes work on a smaller array, and no grid of coordinates is built
over every axis. These maps compose: a level
downsampled by `g` from a level that was itself downsampled by `f` lies
where a level downsampled by `f * g` from the finest level lies, and has
as many voxels, so a level is computed from the level before it whenever
its factors are multiples of those of that level. The mode `"nearest"`
is the exception, because a voxel picked from a picked voxel is not
always the voxel nearest to the coarse voxel, so it always picks from
the finest level.

How the value of a coarse voxel is computed is the mode, one of
[`MODES`][]:

- `"mean"`, the default, averages the voxels that the coarse voxel
  covers: a window of `f` voxels with `"exact"`, and a window as wide as
  the actual scale, centred at each voxel, before the pull, otherwise.
- `"binomial"` and `"gaussian"` apply a filter whose width follows one
  principle: a voxel has an intrinsic Gaussian profile whose full width
  at half maximum (FWHM) is one voxel, and a level downsampled by a
  scale `s` must end with a FWHM of `s` fine voxels. Gaussian widths add
  in quadrature, so the filter has a FWHM of `sqrt(s ** 2 - 1)` fine
  voxels, which is a standard deviation
  `sigma = sqrt(s ** 2 - 1) / (2 * sqrt(2 * log(2)))`, about 0.74 for a
  scale of 2. `"gaussian"` samples that Gaussian at the voxels, out to 3
  standard deviations, and normalizes it to a sum of 1. `"binomial"` is
  the row `k` of Pascal's triangle divided by its sum, whose variance is
  `k / 4`, for the order `k` that brings that variance closest to
  `sigma ** 2` among the orders that centre the filter where it must
  lie. With `"exact"` and a factor of 2, it is the filter
  `(1, 3, 3, 1) / 8`, centred between two voxels, and before a pull, it
  is `(1, 2, 1) / 4` for a scale of 2.
- `"nearest"` takes the fine voxel nearest to the position of the coarse
  voxel, and computes nothing. With `"exact"`, it is the voxel
  `f * i + (f - 1) // 2`, which is half a fine voxel from the centre of
  the coarse voxel when `f` is even, the usual error of a nearest
  neighbour. It suits labels, and a field, which it samples at some of
  its nodes without changing their values.
- `"mode"`, `"vote_binomial"` and `"vote_gaussian"`, the modes of
  [`VOTE_MODES`][], vote among labels, whose values are not averaged.
  The indicator of a label is 1 at the voxels of that label and 0
  elsewhere. A vote smooths the indicator of each label with the box
  kernel of `"mean"`, the binomial kernel or the Gaussian kernel, pulls
  it by linear interpolation under `"edges"` and `"centers"`, and
  keeps, at each coarse voxel, the label of the highest score. When
  labels tie, the vote keeps the smallest label value. A level keeps
  the labels and their type.
- `"max"` and `"min"` keep the largest or the smallest value of the
  window. These two modes reduce the windows of `"exact"`, and the other
  alignments refuse them.

A mode is described by a [`Mode`][], which pairs a kernel with a
reduction, and says whether the mode votes.

[`vote`][] explains how the loop over the labels works: it only visits
the labels of the block that it is given, so a chunk of a Dask array
visits its own labels, and it never holds one indicator per label at
once. With the box kernel and the alignment `"exact"`, the score of a
label is the fraction of the window that the label covers, so `"mode"`
keeps the value that occurs most often in the window. `"mode"` computes
that value by counting the equal voxels of each window instead of
looping over the labels, which gives the same labels, ties included,
and costs less for the small windows of a level computed from the level
before it. The binomial and the Gaussian kernels weight the voxels near
the coarse voxel more than those further away, and reach beyond its
window, which gives smoother boundaries between the labels.

A vote keeps a label only where that label wins. A structure thinner
than half a coarse voxel, such as a line one voxel wide downsampled by
2, therefore disappears when a single other label surrounds it, while
`"nearest"` keeps the voxels of the structure that it happens to pick.
Where several labels mix around the structure, the structure holds the
highest score, and a vote keeps it along its whole length, while
`"nearest"` keeps pieces of it.

A vote is computed from the level before it, as the other modes except
`"nearest"` are, so a coarse level votes among the labels that the
level before it kept, and not among the voxels of the finest level.
This changes some voxels along the boundaries between labels. On a test
parcellation, `"vote_gaussian"` changes 5 % of the coarse voxels for a
factor of 4 and 10 % for a factor of 8, while the fraction of the fine
voxels that keep their label changes by less than one percentage point.
A vote from the finest level would cost more, and a chunk of a Dask
level would read a region of the finest level that grows as `f ** d`
for `d` downsampled axes.

A filter may read voxels beyond the edges of the finer level, which are
extended as `edge` says. With `"repeat"`, for an image, an added voxel
repeats the voxel at the edge. With `"linear"`, for a field, an added
voxel continues the line through the voxels at the edge, by a point
reflection through the voxel at the edge, so that a field that is
affine, such as the coordinates of an identity, stays exact up to the
edges. A vote refuses `"linear"`, which would add labels that the
array does not hold.

The values of a field may depend on the grid of the field itself, which
[`rescale_vectors`][] handles after any mode.

The arithmetic is written with the operations of the array package, so
an array on a GPU stays on the GPU. A NumPy or a CuPy array is computed
as a whole, one pass after the other. A Dask array stays lazy, and each
chunk of a level is one task of its graph: the task copies the region
of the finer level that the chunk reads, with the voxels around it that
the filter and the interpolation read, from the chunks of the finer
level that it overlaps, and runs the same passes on that region with
the package of the chunks. The passes of a level are therefore not
separate layers of the graph, which Dask could not fuse, and a level
has as many tasks as chunks, which have the shape of the chunks of the
finest level. The interpolation of [`brainhops._ext.interpol`][] does
not smooth, so the filters of the modes are applied before it, and the
`ndimage` packages of SciPy and CuPy, which have no version for Dask,
are not used.

When the factors are not given, [`default_factors`][] chooses them
step by step. At each step, each axis that may be downsampled is
downsampled by its factor, unless it has `min_size` voxels or fewer, in
which case it stays as it is from then on, or unless its voxels are more
than `sqrt(f)` times larger than the smallest voxels of the axes that
can still be downsampled. An axis of 40 voxels therefore becomes an axis
of 20 voxels when `min_size` is 32. The steps stop when no axis can be
downsampled. The second rule makes anisotropic voxels roughly isotropic
before the coarse axes are downsampled, because downsampling an axis
whose voxels are larger than that would move it further, in a ratio of
sizes, from the voxel size of the other axes.
"""

__all__ = [
    "ALIGNMENTS",
    "BLOCK_MODES",
    "DEFAULT_ALIGN",
    "DEFAULT_MODE",
    "FIELD_MODES",
    "MIN_SIZE",
    "MODES",
    "Mode",
    "VOTE_MODES",
    "check_align",
    "check_mode",
    "default_factors",
    "downsample",
    "level_geometry",
    "level_map",
    "rescale_vectors",
    "vote",
]

# stdlib
import itertools
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internal
from .backends import array_backend, is_dask
from .bsplines import pull_axes

MIN_SIZE = 32
"""The size at or below which the rules stop downsampling an axis."""

MODES = (
    "mean",
    "binomial",
    "gaussian",
    "nearest",
    "mode",
    "vote_binomial",
    "vote_gaussian",
    "max",
    "min",
)
"""The ways in which a coarse voxel is computed from the finer level.

| Mode              | Value of a coarse voxel                | Suits          |
| ----------------- | -------------------------------------- | -------------- |
| `"mean"`          | the mean of the voxels that it covers  | images, fields |
| `"binomial"`      | a binomial filter around its position  | images, fields |
| `"gaussian"`      | a Gaussian filter around its position  | images, fields |
| `"nearest"`       | the fine voxel nearest to its position | labels, fields |
| `"mode"`          | the label that covers most of it       | labels         |
| `"vote_binomial"` | the label with the best binomial score | labels         |
| `"vote_gaussian"` | the label with the best Gaussian score | labels         |
| `"max"`           | the largest value of its window        | images, masks  |
| `"min"`           | the smallest value of its window       | images, masks  |

A mode that averages is named after its reduction (`"mean"`) or its
kernel (`"binomial"`, `"gaussian"`). The votes are named after the
reduction and then the kernel, except the vote with the box kernel,
which keeps the name of the statistic that it computes, `"mode"`.
"""

DEFAULT_MODE = "mean"
"""The mode with which a pyramid is built by default."""

FIELD_MODES = ("mean", "binomial", "gaussian", "nearest")
"""The modes that apply to a field of displacements or coordinates.

The other modes compare the vectors of a window component by component,
or vote among them, which does not give a vector of the field.
"""

VOTE_MODES = ("mode", "vote_binomial", "vote_gaussian")
"""The modes that vote among labels, which work with every alignment."""

BLOCK_MODES = ("max", "min")
"""The modes that only reduce the windows of the alignment `"exact"`."""

ALIGNMENTS = ("exact", "edges", "centers")
"""How the voxels of a coarse level lie over those of the finer one."""

DEFAULT_ALIGN = "exact"
"""The alignment with which a pyramid is built by default."""

EDGES = ("repeat", "linear")
"""The ways in which the voxels are extended beyond the edges."""

_FWHM = 2 * math.sqrt(2 * math.log(2))
"""The ratio of the FWHM of a Gaussian to its standard deviation."""

_TIE = 1e-9
"""The difference of scores below which a vote counts a tie."""


class Mode(tx.NamedTuple):
    """How a mode computes a coarse voxel from the finer level.

    A mode filters the voxels with a kernel, or reduces the voxels of a
    window of the alignment `"exact"`, or takes the fine voxel nearest
    to the coarse voxel when it has neither. `"mean"` has both: it
    reduces the windows of `"exact"`, and filters before a pull. A mode
    that votes runs its kernel, and the pull of the alignment, on the
    indicator of each label, and keeps the label of the highest score.
    `"mode"` votes and has a reduction too, which counts the values of
    each window of `"exact"` and gives the labels of a vote with the
    box kernel at a lower cost.
    """

    kernel: tx.Optional[tx.Callable[[float, float], tx.Tuple[int, np.ndarray]]]
    """The filter of a scale `s`, centred at an offset `c`, or `None`.

    It returns the offset of its first tap from a voxel, and the weights
    of its taps, which are centred at that voxel plus `c`.
    """

    reduction: tx.Optional[tx.Callable[..., ArrayProtocol]]
    """The reduction of each window of `"exact"`, or `None`.

    It is called on an array whose axis `k` is split into the axes
    `2 * k` and `2 * k + 1`, the second of which holds the window, with
    the axes of the windows given as `axis`, as `dask.array.coarsen`
    calls it.
    """

    averages: bool
    """Whether the mode averages the values, which an integer rounds."""

    votes: bool = False
    """Whether the mode votes among the labels with its kernel.

    The vote goes through [`vote`][]. A reduction, when the mode has
    one, replaces the vote for the windows of `"exact"`, and must give
    the same labels.
    """


def check_mode(
    mode: str, field: bool = False, align: str = DEFAULT_ALIGN
) -> str:
    """Return a mode of [`MODES`][], or refuse it.

    Parameters
    ----------
    mode : str
        The mode.
    field : bool, default=False
        Whether the pyramid is a field of displacements or coordinates,
        which only takes the modes of [`FIELD_MODES`][].
    align : str, default="exact"
        The alignment, since the modes of [`BLOCK_MODES`][] only reduce
        the windows of `"exact"`.

    Returns
    -------
    str
        The mode.

    Raises
    ------
    ValueError
        If the mode is unknown, if it does not apply to a field, or if
        it does not apply to the alignment.
    """
    if mode not in MODES:
        known = ", ".join(repr(name) for name in MODES)
        raise ValueError(
            f"A pyramid is built with one of the modes {known}, not {mode!r}."
        )
    if field and mode not in FIELD_MODES:
        known = ", ".join(repr(name) for name in FIELD_MODES)
        raise ValueError(
            f"The mode {mode!r} compares the vectors of a field component "
            "by component, or votes among them, which does not give a "
            f"vector of the field. A field is downsampled with {known}."
        )
    if mode in BLOCK_MODES and check_align(align) != "exact":
        raise ValueError(
            f"The mode {mode!r} reduces the windows of `f` voxels that the "
            "alignment 'exact' gives a coarse voxel, while the alignment "
            f"{align!r} pulls the coarse voxels by interpolation. Use "
            "align='exact' with this mode."
        )
    return mode


def check_align(align: str) -> str:
    """Return an alignment of [`ALIGNMENTS`][], or refuse it.

    Parameters
    ----------
    align : str
        The alignment.

    Returns
    -------
    str
        The alignment.

    Raises
    ------
    ValueError
        If the alignment is unknown.
    """
    if align not in ALIGNMENTS:
        known = ", ".join(repr(name) for name in ALIGNMENTS)
        raise ValueError(
            f"The voxels of a coarse level are aligned by one of {known}, "
            f"not by {align!r}."
        )
    return align


def level_map(
    align: str, length: int, factor: int
) -> tx.Tuple[int, float, float]:
    """Return how a level lies over the finer level along one axis.

    Parameters
    ----------
    align : str
        The alignment of [`ALIGNMENTS`][].
    length : int
        The number of voxels of the finer level along the axis.
    factor : int
        The factor of the axis.

    Returns
    -------
    count : int
        The number of voxels of the level along the axis.
    scale : float
        The actual scale `s` of the axis.
    shift : float
        The shift `t` of the axis, so that the voxel `i` of the level
        lies at the voxel `s * i + t` of the finer level.
    """
    length, factor = int(length), int(factor)
    check_align(align)
    if factor == 1:
        return length, 1.0, 0.0
    if align == "exact":
        return length // factor, float(factor), (factor - 1) / 2
    if align == "edges":
        count = -(-length // factor)
        scale = length / count
        return count, scale, (scale - 1) / 2
    if length <= 1:
        return length, 1.0, 0.0
    count = -(-(length - 1) // factor) + 1
    return count, (length - 1) / (count - 1), 0.0


def level_geometry(
    align: str, shape: tx.Sequence[int], factors: tx.Sequence[int]
) -> tx.Tuple[tx.Tuple[int, ...], np.ndarray, np.ndarray]:
    """Return how a level lies over the finer level along each axis.

    Parameters
    ----------
    align : str
        The alignment of [`ALIGNMENTS`][].
    shape : sequence of int
        The shape of the finer level.
    factors : sequence of int
        The factor of each axis.

    Returns
    -------
    shape : tuple of int
        The shape of the level.
    scales : ndarray
        The actual scale of each axis.
    shifts : ndarray
        The shift of each axis, as [`level_map`][] returns them.
    """
    maps = [level_map(align, n, f) for n, f in zip(shape, factors)]
    return (
        tuple(count for count, _, _ in maps),
        np.array([scale for _, scale, _ in maps], dtype=float),
        np.array([shift for _, _, shift in maps], dtype=float),
    )


def default_factors(
    shape: tx.Sequence[int],
    sizes: tx.Sequence[float],
    step: tx.Sequence[int],
    nscales: tx.Optional[int] = None,
    min_size: int = MIN_SIZE,
    strict: bool = False,
    align: str = DEFAULT_ALIGN,
) -> tx.List[tx.Tuple[int, ...]]:
    """Return the factors of the coarser levels of a pyramid.

    The factors are chosen step by step, by the rules of the module
    docstring.

    Parameters
    ----------
    shape : sequence of int
        The number of voxels of the finest level along each axis.
    sizes : sequence of float
        The size of a voxel of the finest level along each axis, in the
        same unit for every axis that may be downsampled. A size that is
        not positive counts as 1.
    step : sequence of int
        The factor of each axis at each step, where 1 means that the
        axis is never downsampled.
    nscales : int, optional
        The number of levels, the finest level included. By default, the
        levels go on until no axis can be downsampled.
    min_size : int, default=32
        The number of voxels at or below which an axis is no longer
        downsampled.
    strict : bool, default=False
        Whether to refuse to build fewer levels than `nscales`, when the
        rules stop before.
    align : str, default="exact"
        The alignment of the levels, which decides their numbers of
        voxels.

    Returns
    -------
    list of tuple of int
        For each coarser level, the factor of each axis from the finest
        level.

    Raises
    ------
    ValueError
        If the lengths of `shape`, `sizes` and `step` differ, if a
        factor of `step` is not a positive integer, if `nscales` or
        `min_size` is smaller than 1, or if `strict` is true and the
        rules stop before `nscales` levels.
    """
    shape = [int(length) for length in shape]
    sizes = [float(size) if size > 0 else 1.0 for size in sizes]
    step = [int(factor) for factor in step]
    if not len(shape) == len(sizes) == len(step):
        raise ValueError(
            f"The pyramid has {len(shape)} axes, but {len(sizes)} voxel "
            f"sizes and {len(step)} factors."
        )
    if any(factor < 1 for factor in step):
        raise ValueError(
            f"A factor is a positive integer, which {step} is not."
        )
    if nscales is not None and nscales < 1:
        raise ValueError(f"A pyramid has at least one level, not {nscales}.")
    if min_size < 1:
        raise ValueError(
            f"An axis keeps at least one voxel, so the smallest size of a "
            f"level cannot be {min_size}."
        )
    current = [1] * len(shape)
    factors = []  # type: tx.List[tx.Tuple[int, ...]]
    while nscales is None or len(factors) + 1 < nscales:
        candidates = []  # type: tx.List[int]
        for axis, (length, factor) in enumerate(zip(shape, step)):
            if factor == 1 or length <= min_size:
                continue
            count, _, _ = level_map(align, length, factor)
            if 1 <= count < length:
                candidates.append(axis)
        if not candidates:
            break
        finest = min(sizes[axis] for axis in candidates)
        moved = [
            axis
            for axis in candidates
            # The tolerance keeps an axis whose ratio is exactly the
            # threshold, up to rounding.
            if sizes[axis] <= math.sqrt(step[axis]) * finest * (1 + 1e-9)
        ]
        for axis in moved:
            count, scale, _ = level_map(align, shape[axis], step[axis])
            current[axis] *= step[axis]
            shape[axis] = count
            sizes[axis] *= scale
        factors.append(tuple(current))
    if strict and nscales is not None and len(factors) + 1 < nscales:
        raise ValueError(
            f"A pyramid of {nscales} levels was asked for, but only "
            f"{len(factors) + 1} can be built before every axis that may be "
            f"downsampled has {min_size} voxels or fewer. Lower `min_size`, "
            "or give the factors of the levels."
        )
    return factors


def downsample(
    array: ArrayProtocol,
    factors: tx.Sequence[tx.Sequence[int]],
    mode: str = DEFAULT_MODE,
    edge: str = "repeat",
    align: str = DEFAULT_ALIGN,
    previous: tx.Optional[tx.Tuple[tx.Sequence[int], ArrayProtocol]] = None,
) -> tx.List[ArrayProtocol]:
    """Return the coarser levels of a pyramid, from its finest level.

    A level whose factors are multiples of those of the level before it
    is computed from that level, which costs less, and any other level
    from the finest level. The mode `"nearest"` always picks the voxels
    of the finest level, because a voxel picked from a coarser level is
    not always the voxel nearest to the coarse voxel.

    With `"mean"`, `"binomial"` or `"gaussian"`, a level of an integer
    or boolean type is computed in double precision and rounded back to
    its type, which it never leaves, since these modes average its
    values. The other modes keep the values and their type. A Dask
    array gives Dask arrays, which are not computed. Their chunks have
    the shape of the chunks of `array`, capped at the shape of each
    level, and each chunk is one task of the graph.

    Parameters
    ----------
    array : ArrayProtocol
        The finest level.
    factors : sequence of sequence of int
        For each coarser level, the factor of each axis of `array` from
        the finest level, which is 1 for an axis that is not
        downsampled.
    mode : str, default="mean"
        How a coarse voxel is computed, one of [`MODES`][].
    edge : {"repeat", "linear"}, default="repeat"
        How the voxels are extended beyond the edges, as the module
        docstring explains.
    align : {"exact", "edges", "centers"}, default="exact"
        How the voxels of a coarse level are laid over those of the
        finer level, as the module docstring explains.
    previous : tuple of (sequence of int, ArrayProtocol), optional
        The factors and the array of a coarser level that already
        exists, from which the first level of `factors` is computed when
        its factors are multiples of those of that level.

    Returns
    -------
    list of ArrayProtocol
        The coarser levels, in the order of `factors`.

    Raises
    ------
    ValueError
        If the mode, the edge or the alignment is unknown, if the mode
        does not apply to the alignment, if a vote is asked with the
        edge `"linear"`, or if a level does not give a positive integer
        factor to each axis or leaves no voxel.
    """
    check_mode(mode, align=align)
    if edge not in EDGES:
        raise ValueError(
            f"The voxels are extended beyond the edges by 'repeat' or "
            f"'linear', not by {edge!r}."
        )
    dtype = np.dtype(array.dtype)
    rule = _MODES[mode]
    if rule.votes and edge == "linear":
        raise ValueError(
            f"The mode {mode!r} votes among labels, which the edge 'linear' "
            "would extend by labels that the array does not hold. Use "
            "edge='repeat' with this mode."
        )
    exact = rule.averages and dtype.kind in "biu"
    xp = array_backend(array)

    def working(values: ArrayProtocol) -> ArrayProtocol:
        return values.astype(np.float64) if exact else values

    finest = working(array)
    chunks = None  # type: tx.Optional[tx.Tuple[int, ...]]
    if is_dask(array):
        chunks = tuple(int(max(sizes)) for sizes in array.chunks)
    ones = tuple([1] * array.ndim)
    picks = rule.kernel is None and rule.reduction is None
    if previous is None:
        previous = (ones, finest)
    else:
        previous = (tuple(int(f) for f in previous[0]), working(previous[1]))
    levels = []  # type: tx.List[ArrayProtocol]
    for level in factors:
        level = tuple(int(factor) for factor in level)
        if len(level) != array.ndim or any(f < 1 for f in level):
            raise ValueError(
                f"A level of an array of {array.ndim} axes takes one "
                f"positive integer factor per axis, not {list(level)}."
            )
        shape, _, _ = level_geometry(align, array.shape, level)
        if any(count < 1 for count in shape):
            raise ValueError(
                f"An array of shape {tuple(array.shape)} has no voxel left "
                f"once downsampled by {list(level)}."
            )
        before, source = previous
        if picks or any(f % b for f, b in zip(level, before)):
            # A voxel picked from a picked voxel is not the voxel
            # nearest to the coarse voxel, so "nearest" always picks
            # from the finest level, which costs no more.
            before, source = ones, finest
        relative = [f // b for f, b in zip(level, before)]
        # The chunks of a Dask level have the shape of the chunks of the
        # finest level, so that they do not shrink with each level.
        current = _reduce(source, relative, rule, edge, align, chunks)
        previous = (level, current)
        levels.append(xp.round(current).astype(dtype) if exact else current)
    return levels


def rescale_vectors(
    values: ArrayProtocol,
    scales: tx.Sequence[float],
    shifts: tx.Sequence[float],
    kind: tx.Optional[str],
    inverse: bool = False,
) -> ArrayProtocol:
    """Express the vectors of a field in the grid of a coarser level.

    A field whose vectors are given in voxels of its own grid holds
    other values on a coarser grid, since the voxel `i` of the finer
    grid is the voxel `(i - t) / s` of the coarser one, with the actual
    scale `s` and the shift `t` of [`level_map`][]. Coordinates in that
    grid are mapped by this affine, and displacements, which are
    differences of coordinates, by its linear part only, which divides
    them by `s`. Vectors given in the world, or in the grid of another
    image, do not depend on the grid of the field, so they are only
    averaged by the mode, and returned as they are here.

    Parameters
    ----------
    values : ArrayProtocol
        The vectors of the level, of shape `(*grid, ndim)`, after the
        mode averaged them.
    scales : sequence of float
        The actual scale of each axis of the grid, from the finest
        level.
    shifts : sequence of float
        The shift of each axis, from the finest level.
    kind : {"coordinates", "displacements"}, optional
        What the vectors are when they are given in voxels of the grid
        of the field, or `None` when they are not.
    inverse : bool, default=False
        Whether to express the vectors of the level in the grid of the
        finest level instead.

    Returns
    -------
    ArrayProtocol
        The vectors in the grid of the level.

    Raises
    ------
    ValueError
        If `kind` is unknown.
    """
    if kind is None:
        return values
    scales = np.asarray(scales, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    if kind == "displacements":
        return values * scales if inverse else values / scales
    if kind == "coordinates":
        if inverse:
            return values * scales + shifts
        return (values - shifts) / scales
    raise ValueError(
        "The vectors of a field in its own grid are 'coordinates' or "
        f"'displacements', not {kind!r}."
    )


def vote(
    labels: ArrayProtocol,
    smooth: tx.Callable[[ArrayProtocol], ArrayProtocol],
) -> ArrayProtocol:
    """Return the label of the highest score at each output voxel.

    A label map cannot be averaged, because the mean of two labels is
    a third label that means nothing. This function computes a label
    map at other positions, such as the voxels of a coarser level, by a
    vote: `smooth` turns the indicator of each label, which is 1 where
    the label is and 0 elsewhere, into a score at each output voxel,
    and the label of the highest score wins. A smoothing kernel weights
    the voxels near the output voxel more than those further away,
    which gives smoother boundaries between the labels than a vote that
    counts the voxels of a window.

    The labels are those that `labels` holds, so a chunk of a Dask array
    that calls this function on its own block only loops over the labels
    of that block. The vote keeps a running best score and best label,
    and never builds one indicator per label at once, so it holds two
    arrays of the output shape whatever the number of labels. Scores
    that differ by less than `1e-9` tie, so that the rounding of the
    smoothing does not decide a tie, and a tie goes to the smallest
    label, which is the first label that the loop visits.

    Parameters
    ----------
    labels : ArrayProtocol
        The labels, of any integer, boolean or floating type, as one
        array of the package that computes them.
    smooth : callable
        The function that maps the indicator of a label, an array of
        the shape of `labels` and of type `float64`, to its scores at
        the output voxels. It is called once per label, and it must
        give scores between about 0 and 1, as a kernel normalized to a
        sum of 1 and a linear interpolation do.

    Returns
    -------
    ArrayProtocol
        The label of the highest score at each output voxel, of the
        type of `labels`.
    """
    xp = array_backend(labels)
    best_label = None  # type: tx.Optional[ArrayProtocol]
    best_score = None  # type: tx.Optional[ArrayProtocol]
    # `unique` sorts the labels, so the first label of a tie, which the
    # strict comparison keeps, is the smallest one.
    for value in xp.unique(labels).tolist():
        score = smooth((labels == value).astype(np.float64))
        if best_score is None:
            best_score = score
            best_label = xp.full(score.shape, value, dtype=labels.dtype)
            continue
        better = score > best_score + _TIE
        best_label[better] = value
        best_score = xp.where(better, score, best_score)
    if best_label is None:
        shape = smooth(xp.zeros(labels.shape, dtype=np.float64)).shape
        return xp.zeros(shape, dtype=labels.dtype)
    return best_label


class _Step(tx.NamedTuple):
    # How one axis of a level is computed from the finer level.

    kind: str
    # "keep" leaves the axis as it is, "reduce" reduces the windows of
    # "exact" (jointly over the axes), "filter" filters them with taps,
    # "pick" picks a voxel of each window, and "pull" smooths and then
    # pulls the voxels by interpolation.
    length: int
    # The length of the axis of the finer level.
    count: int
    # The length of the axis of the level.
    factor: int
    scale: float
    shift: float
    first: int
    # The offset of the first tap, from the voxel `factor * i` for
    # "filter", and from each voxel for the smoothing of "pull".
    taps: tx.Optional[tx.Tuple[float, ...]]


def _steps(
    shape: tx.Sequence[int],
    factors: tx.Sequence[int],
    rule: Mode,
    align: str,
) -> tx.List[_Step]:
    # Return the step of each axis of a level.
    steps = []
    for length, factor in zip(shape, factors):
        length, factor = int(length), int(factor)
        count, scale, shift = level_map(align, length, factor)
        first, taps = 0, None  # type: tx.Tuple[int, tx.Any]
        if factor == 1 or (align != "exact" and count == length):
            kind = "keep"
        elif align != "exact":
            kind = "pull"
            if rule.kernel is not None:
                first, taps = rule.kernel(scale, 0.0)
        elif rule.reduction is not None:
            kind = "reduce"
        elif rule.kernel is not None:
            kind = "filter"
            first, taps = rule.kernel(float(factor), (factor - 1) / 2)
        else:
            kind = "pick"
        if taps is not None:
            taps = tuple(float(tap) for tap in taps)
        steps.append(
            _Step(kind, length, count, factor, scale, shift, first, taps)
        )
    return steps


def _span(step: _Step, start: int, stop: int) -> tx.Tuple[int, int]:
    # Return the voxels `[lo, hi)` of the finer level that the voxels
    # `[start, stop)` of the level read along one axis. They go beyond
    # the edges only where a filter reads there.
    f = step.factor
    if step.kind == "keep":
        return start, stop
    if step.kind == "reduce":
        return f * start, f * stop
    if step.kind == "pick":
        first = (f - 1) // 2
        return f * start + first, f * (stop - 1) + first + 1
    if step.kind == "filter":
        last = f * (stop - 1) + step.first + len(step.taps)
        return f * start + step.first, last
    low, high = _pulled(step, start, stop)
    if step.taps is None:
        return low, high
    return low + step.first, high - 1 + step.first + len(step.taps)


def _pulled(step: _Step, start: int, stop: int) -> tx.Tuple[int, int]:
    # Return the voxels `[low, high)` that the interpolation of the
    # voxels `[start, stop)` of the level reads: the voxels around the
    # positions of the first and the last of them, which lie between
    # the first and the last voxel of the axis.
    first = step.scale * start + step.shift
    last = step.scale * (stop - 1) + step.shift
    low = max(0, math.floor(first))
    high = min(step.length, math.floor(last) + 2)
    return low, high


def _region(
    span: tx.Tuple[int, int], length: int, edge: str
) -> tx.Tuple[int, int]:
    # Return the voxels of an axis that are read to build the voxels of
    # `span`: those of `span` within the axis, and the voxels that
    # `_pad` reads to extend the axis where `span` goes beyond its
    # edges.
    low, high = span
    start, stop = max(0, low), min(length, high)
    if low < 0:
        reach = -low + 1 if edge == "linear" else 1
        stop = max(stop, min(length, reach))
    if high > length:
        reach = high - length + 1 if edge == "linear" else 1
        start = min(start, max(0, length - reach))
    return start, stop


def _compute(
    region: ArrayProtocol,
    origin: tx.Sequence[int],
    steps: tx.Sequence[_Step],
    ranges: tx.Sequence[tx.Tuple[int, int]],
    rule: Mode,
    edge: str,
) -> ArrayProtocol:
    # Compute the voxels `ranges` of a level from a region of the finer
    # level whose first voxel is the voxel `origin`. The region holds
    # the voxels that `_region` gives. A whole array is one region, and
    # a chunk of a Dask array is computed from its own region, by this
    # same code.
    block = region
    spans = [_span(step, *bounds) for step, bounds in zip(steps, ranges)]
    for axis, (step, (low, high)) in enumerate(zip(steps, spans)):
        before = max(0, -low)
        after = max(0, high - step.length)
        block = _pad(block, axis, before, after, edge)
        start = int(origin[axis]) - before
        block = _part(block, axis, low - start, high - start)
    if any(step.kind == "reduce" for step in steps):
        factors = [
            step.factor if step.kind == "reduce" else 1 for step in steps
        ]
        return _coarsen(block, factors, rule.reduction)
    # The axis that shrinks the most goes first, which keeps the arrays
    # of the next passes small. The passes act on different axes, so
    # their order changes the result only by rounding.
    order = sorted(
        (axis for axis, step in enumerate(steps) if step.kind != "keep"),
        key=lambda axis: steps[axis].count / steps[axis].length,
    )

    def passes(values: ArrayProtocol) -> ArrayProtocol:
        for axis in order:
            step, (start, stop) = steps[axis], ranges[axis]
            values = _step(values, axis, step, start, stop, edge)
        return values

    # A vote runs the same passes on the indicator of each label of the
    # block, so a chunk of a Dask array only loops over its own labels.
    return vote(block, passes) if rule.votes else passes(block)


def _step(
    block: ArrayProtocol,
    axis: int,
    step: _Step,
    start: int,
    stop: int,
    edge: str,
) -> ArrayProtocol:
    # Compute the voxels `[start, stop)` of a level along one axis, from
    # the block that holds exactly the voxels of `_span` along it.
    count = stop - start
    if step.kind == "pick":
        return _part(
            block, axis, 0, step.factor * (count - 1) + 1, step.factor
        )
    if step.kind == "filter":
        return _filter(block, axis, step.factor, count, 0, step.taps, edge)
    # The pull smooths the voxels `[low, high)` with the kernel of the
    # mode, and interpolates them with the splines of `pull_axes`: of
    # degree 1, which is linear, after smoothing, and of degree 0, the
    # nearest voxel, in the mode "nearest". The coordinates are those of
    # the one axis, so that the other axes are batch axes, and no grid
    # of coordinates is built over every axis. The positions lie between
    # the first and the last voxel that the block holds, so the boundary
    # condition only guards against rounding.
    low, high = _pulled(step, start, stop)
    degree = 0
    if step.taps is not None:
        block = _filter(block, axis, 1, high - low, 0, step.taps, edge)
        degree = 1
    positions = step.scale * np.arange(start, stop) + step.shift - low
    xp = array_backend(block)
    pulled = pull_axes(
        block, positions[:, None], [axis], degree, "nearest", True
    )
    return xp.moveaxis(pulled, -1, axis)


def _reduce(
    array: ArrayProtocol,
    factors: tx.Sequence[int],
    rule: Mode,
    edge: str,
    align: str,
    chunks: tx.Optional[tx.Sequence[int]] = None,
) -> ArrayProtocol:
    # Downsample an array by one factor per axis, as a mode and an
    # alignment say. A Dask array is computed chunk by chunk, into
    # chunks of the shape `chunks`, capped at the shape of the level.
    steps = _steps(array.shape, factors, rule, align)
    if all(step.kind == "keep" for step in steps):
        return array
    if is_dask(array):
        return _reduce_dask(array, steps, rule, edge, chunks)
    ranges = [(0, step.count) for step in steps]
    regions = [
        _region(_span(step, *bounds), step.length, edge)
        for step, bounds in zip(steps, ranges)
    ]
    region = array[tuple(slice(start, stop) for start, stop in regions)]
    origin = [start for start, _ in regions]
    return _compute(region, origin, steps, ranges, rule, edge)


def _reduce_dask(
    array: ArrayProtocol,
    steps: tx.Sequence[_Step],
    rule: Mode,
    edge: str,
    chunks: tx.Optional[tx.Sequence[int]],
) -> ArrayProtocol:
    # Build a level of a Dask array with one task per chunk of the
    # level. The task copies the region of the finer level that the
    # chunk reads from the chunks of the finer level that it overlaps,
    # and runs `_compute` on it with the package of the chunks, so that
    # the passes of a level are not separate layers of the graph.
    import dask.array as da
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    if chunks is None:
        chunks = [max(sizes) for sizes in array.chunks]
    shape = tuple(step.count for step in steps)
    out_chunks = da.core.normalize_chunks(
        tuple(min(int(size), n) for size, n in zip(chunks, shape)), shape
    )
    name = "pyramid-level-" + tokenize(array, steps, rule, edge, out_chunks)
    in_edges = [np.cumsum((0,) + tuple(sizes)) for sizes in array.chunks]
    out_edges = [np.cumsum((0,) + tuple(sizes)) for sizes in out_chunks]
    layer = {}  # type: tx.Dict[tx.Any, tx.Any]
    for index in itertools.product(*[range(len(c)) for c in out_chunks]):
        ranges = [(int(e[i]), int(e[i + 1])) for e, i in zip(out_edges, index)]
        regions = [
            _region(_span(step, *bounds), step.length, edge)
            for step, bounds in zip(steps, ranges)
        ]
        axes = [
            _pieces(e, start, stop)
            for e, (start, stop) in zip(in_edges, regions)
        ]
        keys, pieces = [], []
        for combination in itertools.product(*axes):
            keys.append((array.name,) + tuple(c for c, _, _ in combination))
            pieces.append(
                (
                    tuple(s for _, s, _ in combination),
                    tuple(t for _, _, t in combination),
                )
            )
        region = tuple(stop - start for start, stop in regions)
        origin = [start for start, _ in regions]
        layer[(name,) + index] = (
            _chunk,
            keys,
            pieces,
            region,
            origin,
            list(steps),
            ranges,
            rule,
            edge,
        )
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[array])
    meta = da.utils.meta_from_array(array, len(shape))
    return da.Array(graph, name, out_chunks, meta=meta)


def _pieces(
    edges: np.ndarray, start: int, stop: int
) -> tx.List[tx.Tuple[int, slice, slice]]:
    # Return, for each chunk of an axis that the voxels `[start, stop)`
    # overlap, the index of the chunk, the slice of the chunk that they
    # cover, and the place of that slice among them.
    pieces = []
    first = int(np.searchsorted(edges, start, side="right")) - 1
    for chunk in range(first, len(edges) - 1):
        low, high = int(edges[chunk]), int(edges[chunk + 1])
        if low >= stop:
            break
        begin, end = max(start, low), min(stop, high)
        pieces.append(
            (
                chunk,
                slice(begin - low, end - low),
                slice(begin - start, end - start),
            )
        )
    return pieces


def _chunk(
    parts: tx.List[ArrayProtocol],
    pieces: tx.List[tx.Tuple[tx.Tuple[slice, ...], tx.Tuple[slice, ...]]],
    region: tx.Tuple[int, ...],
    origin: tx.Sequence[int],
    steps: tx.Sequence[_Step],
    ranges: tx.Sequence[tx.Tuple[int, int]],
    rule: Mode,
    edge: str,
) -> ArrayProtocol:
    # Compute one chunk of a level: copy its region of the finer level
    # from the chunks that it overlaps, and compute the chunk from it.
    if len(parts) == 1:
        block = parts[0][pieces[0][0]]
    else:
        xp = array_backend(parts[0])
        block = xp.empty(region, dtype=parts[0].dtype)
        for part, (source, target) in zip(parts, pieces):
            block[target] = part[source]
    return _compute(block, origin, steps, ranges, rule, edge)


def _filter(
    array: ArrayProtocol,
    axis: int,
    step: int,
    count: int,
    first: int,
    taps: tx.Sequence[float],
    edge: str,
) -> ArrayProtocol:
    # Filter one axis with the taps that read the voxels from
    # `step * i + first` on, for the output voxel `i`. The taps are
    # Python floats, so that the array keeps its type.
    length = int(array.shape[axis])
    before = max(0, -first)
    last = step * (count - 1) + first + len(taps) - 1
    after = max(0, last - (length - 1))
    padded = _pad(array, axis, before, after, edge)
    start = first + before
    total = None
    for offset, weight in enumerate(taps):
        begin = start + offset
        stop = begin + step * (count - 1) + 1
        part = _part(padded, axis, begin, stop, step) * float(weight)
        total = part if total is None else total + part
    return total


def _coarsen(
    array: ArrayProtocol,
    factors: tx.Sequence[int],
    reduction: tx.Callable,
) -> ArrayProtocol:
    # Reduce the windows of `factors` voxels that do not overlap, in an
    # array whose axes are multiples of their factors.
    shape = []  # type: tx.List[int]
    for length, factor in zip(array.shape, factors):
        shape += [int(length) // int(factor), int(factor)]
    windows = tuple(range(1, 2 * array.ndim, 2))
    return reduction(array.reshape(tuple(shape)), axis=windows)


def _pad(
    array: ArrayProtocol, axis: int, before: int, after: int, edge: str
) -> ArrayProtocol:
    # Add `before` voxels before the first voxel of an axis and `after`
    # voxels after its last voxel. With "repeat", an added voxel repeats
    # the voxel at the edge. With "linear", the voxel `-k` is
    # `2 * x[0] - x[k]` and the voxel `n - 1 + k` is
    # `2 * x[n - 1] - x[n - 1 - k]`, read at the nearest voxel when the
    # axis is too short.
    if not before and not after:
        return array
    xp = array_backend(array)
    length = int(array.shape[axis])
    parts = []
    if before:
        first = _part(array, axis, 0, 1)
        if edge == "linear":
            mirror = [min(k, length - 1) for k in range(before, 0, -1)]
            parts.append(2 * first - xp.take(array, mirror, axis=axis))
        else:
            parts.append(xp.take(array, [0] * before, axis=axis))
    parts.append(array)
    if after:
        last = _part(array, axis, length - 1, length)
        if edge == "linear":
            mirror = [max(length - 1 - k, 0) for k in range(1, after + 1)]
            parts.append(2 * last - xp.take(array, mirror, axis=axis))
        else:
            parts.append(xp.take(array, [length - 1] * after, axis=axis))
    return xp.concatenate(parts, axis=axis)


def _part(
    array: ArrayProtocol, axis: int, start: int, stop: int, step: int = 1
) -> ArrayProtocol:
    # Return the slice of an array along one axis, which keeps the axis.
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop, step)
    return array[tuple(index)]


def _mean(block: ArrayProtocol, axis: tx.Any = None) -> ArrayProtocol:
    return block.mean(axis=axis)


def _max(block: ArrayProtocol, axis: tx.Any = None) -> ArrayProtocol:
    return block.max(axis=axis)


def _min(block: ArrayProtocol, axis: tx.Any = None) -> ArrayProtocol:
    return block.min(axis=axis)


def _majority(block: ArrayProtocol, axis: tx.Any = None) -> ArrayProtocol:
    # Keep the most frequent value of each window, and the smallest of
    # the values that tie, as `vote` does. A voxel that does not win is
    # replaced by the largest value of its window, which no winner
    # exceeds, so that the minimum over the window is the smallest
    # winner.
    xp = array_backend(block)
    axis = tuple(axis)
    moved = xp.moveaxis(block, axis, tuple(range(-len(axis), 0)))
    flat = moved.reshape(moved.shape[: moved.ndim - len(axis)] + (-1,))
    counts = (flat[..., :, None] == flat[..., None, :]).sum(axis=-1)
    wins = counts == counts.max(axis=-1, keepdims=True)
    largest = flat.max(axis=-1, keepdims=True)
    return xp.where(wins, flat, largest).min(axis=-1)


def _sigma(scale: float) -> float:
    # The standard deviation of the filter that brings a FWHM of 1 voxel
    # to a FWHM of `scale` voxels, since Gaussian widths add in
    # quadrature.
    return math.sqrt(max(scale**2 - 1, 0.0)) / _FWHM


def _box(scale: float, centre: float) -> tx.Tuple[int, np.ndarray]:
    # The window as wide as `scale`, centred at `centre`: each voxel
    # counts for the part of it that the window covers.
    low, high = centre - scale / 2, centre + scale / 2
    first = math.floor(low + 0.5)
    last = math.ceil(high - 0.5)
    offsets = np.arange(first, last + 1)
    cover = np.minimum(offsets + 0.5, high) - np.maximum(offsets - 0.5, low)
    taps = np.clip(cover, 0.0, None)
    return first, taps / taps.sum()


def _binomial(scale: float, centre: float) -> tx.Tuple[int, np.ndarray]:
    # The row of Pascal's triangle whose variance, a quarter of its
    # order, is closest to the square of `_sigma`, among the orders that
    # centre it at `centre`, which is a voxel or the middle of two.
    variance = _sigma(scale) ** 2
    parity = int(round(2 * centre)) % 2
    orders = range(parity, int(4 * variance) + 4, 2)
    order = min(orders, key=lambda k: (abs(k / 4 - variance), -k))
    row = np.array([math.comb(order, k) for k in range(order + 1)], float)
    return int(round(centre - order / 2)), row / row.sum()


def _gaussian(scale: float, centre: float) -> tx.Tuple[int, np.ndarray]:
    # A Gaussian of standard deviation `_sigma`, sampled at the voxels
    # within 3 standard deviations of `centre`.
    sigma = _sigma(scale)
    if sigma == 0:
        return int(round(centre)), np.ones(1)
    first = math.ceil(centre - 3 * sigma)
    last = math.floor(centre + 3 * sigma)
    offsets = np.arange(first, last + 1) - centre
    taps = np.exp(-0.5 * (offsets / sigma) ** 2)
    return first, taps / taps.sum()


_MODES = {
    "mean": Mode(_box, _mean, averages=True),
    "binomial": Mode(_binomial, None, averages=True),
    "gaussian": Mode(_gaussian, None, averages=True),
    "nearest": Mode(None, None, averages=False),
    "mode": Mode(_box, _majority, averages=False, votes=True),
    "vote_binomial": Mode(_binomial, None, averages=False, votes=True),
    "vote_gaussian": Mode(_gaussian, None, averages=False, votes=True),
    "max": Mode(None, _max, averages=False),
    "min": Mode(None, _min, averages=False),
}
"""How each mode of [`MODES`][] computes a coarse voxel."""
