"""Write-side counterparts of the slicing utilities of `nibabel.fileslice`."""

from threading import Lock

import numpy as np
import typing_extensions as _tx
from nibabel.fileslice import (
    _NullLock,
    _positive_slice,
    canonical_slicers,
    fill_slicer,
    is_fancy,
    operator,
    predict_shape,
    read_segments,
    reduce,
    slicers2segments,
    threshold_heuristic,
)
from numpy.typing import ArrayLike

from brainhops._core.path import FileLike
from brainhops._ext.npfileobj.indexing import IndexLike


def full_heuristic(*args, **kwargs) -> _tx.Literal["full", "contiguous", None]:
    """Heuristic that never reads bytes that the slice does not need.

    The function calls `threshold_heuristic` with `skip_thresh=0`, so that
    every gap in memory is skipped rather than read and discarded. Despite
    the name of the function, it returns `None` for every slice, and it
    returns `"full"` only for an integer index on an axis of length 1.
    """
    return threshold_heuristic(*args, **kwargs, skip_thresh=0)


def write_segments(
    fileobj: FileLike,
    segments: list,
    dat: np.byte,
    lock: _tx.Optional[Lock] = None,
) -> None:
    """Write a byte array to segments of a file object.

    Parameters
    ----------
    fileobj : FileLike
        File object that implements `seek` and `write`.
    segments : list of (int, int)
        Absolute byte offset and number of bytes of each segment.
    dat : bytes
        Data, whose length is the sum of the segment lengths.
    lock : threading.Lock or lock-like, optional
        Lock that guards each pair of `seek` and `write` calls. Threads that
        share a file must share a lock. By default, no lock is used.

    Raises
    ------
    ValueError
        If fewer bytes than expected are written.
    """
    if lock is None:
        lock = _NullLock()

    if len(segments) == 0:
        return
    if len(segments) == 1:
        offset, length = segments[0]
        with lock:
            fileobj.seek(offset)
            nb_written = fileobj.write(dat)
        if nb_written != length:
            raise ValueError(
                f"Expected to write {length} bytes but wrote {nb_written}."
            )
        return
    # Several segments consume the data sequentially.
    dat_offset = 0
    for offset, length in segments:
        with lock:
            fileobj.seek(offset)
            nb_written = fileobj.write(dat[dat_offset : dat_offset + length])
        dat_offset += length
        if nb_written != length:
            raise ValueError(
                f"Expected to write {length} bytes but wrote {nb_written}."
            )


def writeslice(
    dat: ArrayLike,
    fileobj: FileLike,
    sliceobj: object,
    shape: _tx.Tuple[int],
    dtype: type,
    offset: int = 0,
    order: str = "C",
    heuristic: _tx.Optional[_tx.Callable] = threshold_heuristic,
    lock: _tx.Optional[Lock] = None,
) -> ArrayLike:
    """Write an array into a slice of an array stored in a file.

    The file holds an array of shape `shape`, type `dtype` and layout
    `order`, stored contiguously from byte `offset`, and `dat` is written
    into `array[sliceobj]`. Many short writes separated by seeks can be
    slow, so it is sometimes faster to read a larger block, modify it in
    memory and write it back. The `heuristic` decides between these two
    strategies. The default heuristic is the one used for reading, which
    may be suboptimal, because reading a block and writing it back costs
    more than reading a block and discarding part of it.

    Parameters
    ----------
    dat : ArrayLike
        Data to write.
    fileobj : FileLike
        Binary file object opened for reading and writing.
    sliceobj : object
        Any object that can index an array, as in `array[sliceobj]`.
    shape : tuple of int
        Shape of the full array.
    dtype : dtype-like
        Data type of the array.
    offset : int, default=0
        Byte offset of the array in the file.
    order : {"C", "F"}, default="C"
        Memory layout of the array.
    heuristic : callable, optional
        Function of a slice, an axis length and a stride that returns
        `"full"`, `"contiguous"` or `None`. See `threshold_heuristic`.
    lock : threading.Lock or lock-like, optional
        Lock that guards each `seek` call together with the read or write
        that follows it. Threads that share a file must share a lock. By
        default, no lock is used.

    Raises
    ------
    ValueError
        If `sliceobj` uses fancy indexing.
    """
    if is_fancy(sliceobj):
        raise ValueError("Cannot handle fancy indexing")
    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    pre_slicers, segments, sub_slicers, sub_shape = calc_slicedefs_write(
        sliceobj, shape, itemsize, offset, order, heuristic
    )
    dat = dat[pre_slicers]
    if not all(sub_slicer == slice(None) for sub_slicer in sub_slicers):
        # Read a larger block, patch it in memory and write it back, instead
        # of performing many small writes.
        n_bytes = reduce(operator.mul, sub_shape, 1) * itemsize
        bytes = read_segments(fileobj, segments, n_bytes, lock)
        block = np.ndarray(sub_shape, dtype, buffer=bytes, order=order)
        block[sub_slicers] = dat
        dat = block
    dat = dat.tobytes(order="C")
    write_segments(fileobj, segments, dat, lock)
    return


def calc_slicedefs_write(
    sliceobj: object,
    in_shape: _tx.Sequence[int],
    itemsize: int,
    offset: int,
    order: _tx.Literal["C", "F"],
    heuristic: _tx.Optional[_tx.Callable] = threshold_heuristic,
) -> _tx.Tuple[_tx.Tuple, _tx.Tuple, _tx.Tuple, _tx.Tuple]:
    """Compute the segments and slicers needed to write into a slice.

    The arguments are the same as in [`writeslice`][], with `in_shape` the
    shape of the full array and `itemsize` the size of an element in bytes.
    A `ValueError` is raised if `order` is neither `"C"` nor `"F"`.

    Returns
    -------
    pre_slicers : tuple
        Slicers applied to the data first, which remove new axes and make
        the strides positive.
    segments : tuple of (int, int)
        Byte offset and length of each chunk to write, and possibly to read
        first.
    sub_slicers : tuple
        Slicers that place the data into a larger chunk read from the file.
        If they are all full, nothing needs to be read.
    sub_shape : tuple of int
        Shape of the chunk described by the segments.
    """
    if order not in "CF":
        raise ValueError("order should be one of 'CF'")
    sliceobj = canonical_slicers(sliceobj, in_shape)
    # Work in Fortran order, with the fastest dimension first.
    if order == "C":
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    # The write slicers are never applied to the data, but they define the
    # bytes of the file that are touched.
    pre_slicers, write_slicers, sub_slicers = optimize_write_slicers(
        sliceobj, in_shape, itemsize, heuristic
    )
    segments = slicers2segments(write_slicers, in_shape, offset, itemsize)
    sub_shape = predict_shape(write_slicers, in_shape)
    if order == "C":
        sub_shape = sub_shape[::-1]
        sub_slicers = sub_slicers[::-1]
        pre_slicers = pre_slicers[::-1]
    return tuple(pre_slicers), tuple(segments), tuple(sub_slicers), sub_shape


def optimize_write_slicers(
    sliceobj: _tx.Tuple[IndexLike],
    in_shape: _tx.Sequence[int],
    itemsize: int,
    heuristic: _tx.Callable,
) -> _tx.Tuple[_tx.Tuple, _tx.Tuple, _tx.Tuple]:
    """Compute the slicers needed to write into a slice, axis by axis.

    `sliceobj` must be canonical, as returned by `canonical_slicers`, and
    `in_shape` is assumed to be in Fortran order. For an array in C order,
    `sliceobj` and `in_shape` must be reversed beforehand.

    Returns
    -------
    pre_slicers : tuple
        Slicers applied to the data before writing, which discard new axes
        and invert negative strides.
    write_slicers : tuple
        Slicers that describe the chunk that is written, and possibly read
        first.
    sub_slicers : tuple
        Slicers into the chunk described by `write_slicers`. If they are all
        full, the data is written directly. Otherwise, a larger chunk is read
        and the data is placed into it with `sub_slicers`.
    """
    pre_slicers = []
    sub_slicers = []
    write_slicers = []
    real_no = 0
    stride = itemsize
    all_full = True
    for slicer in sliceobj:
        if slicer is None:
            pre_slicers.append(0)
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_last = real_no == len(in_shape)
        pre_slicer, write_slicer, sub_slicer = optimize_write_slicer(
            slicer, dim_len, all_full, is_last, stride, heuristic
        )
        pre_slicers.append(pre_slicer)
        sub_slicers.append(sub_slicer)
        write_slicers.append(write_slicer)
        all_full = all_full and write_slicer == slice(None)
        stride *= dim_len
    return tuple(pre_slicers), tuple(write_slicers), tuple(sub_slicers)


def optimize_write_slicer(
    slicer: _tx.SupportsIndex,
    dim_len: int,
    all_full: bool,
    is_slowest: bool,
    stride: int,
    heuristic: _tx.Optional[_tx.Callable] = threshold_heuristic,
) -> _tx.Tuple[_tx.Union[slice, int], _tx.Union[slice, int], slice]:
    """Compute the slicers needed to write along a single axis.

    A contiguous slice has a step of 1 or -1, and a full slice is a
    contiguous slice that covers every element of the axis. The function
    decides whether the slicer is split into a write slicer and a sub-slicer,
    so that a single large read and write replaces many small writes. The
    `heuristic` takes this decision, and it is only consulted when all the
    faster axes are full (`all_full`). Otherwise, the slicer is only split
    into a pre-slicer and a write slicer, so that the write slicer has a
    positive step.

    Parameters
    ----------
    slicer : slice or int
        Index along the axis.
    dim_len : int
        Length of the axis.
    all_full : bool
        Whether the slicers of all the faster axes are full.
    is_slowest : bool
        Whether the axis is the slowest one in memory.
    stride : int
        Stride of the axis, in bytes.
    heuristic : callable, optional
        See `threshold_heuristic`.

    Returns
    -------
    pre_slicer : slice or int or None
        Slicer applied to the data before writing.
    write_slicer : slice or int
        Slicer that is written, or read and then written. Its step is
        positive.
    sub_slicer : slice
        Slicer that places the data into the larger block.
    """
    try:
        slicer = int(slicer)
    except TypeError:
        if slicer == slice(None):
            return slicer, slicer, slice(None)
        slicer = fill_slicer(slicer, dim_len)
        if slicer == slice(0, dim_len, 1):
            return slice(None), slice(None), slice(None)
        if slicer == slice(dim_len - 1, None, -1):
            return slice(None, None, -1), slice(None), slice(None)
        is_int = False
    else:
        if slicer < 0:
            slicer = dim_len + slicer
        is_int = True
    if all_full:
        action = heuristic(slicer, dim_len, stride)
        # A custom heuristic may return anything.
        if action not in ("full", "contiguous", None):
            raise ValueError(f"Unexpected return {action} from heuristic")
        if is_int and action == "contiguous":
            raise ValueError("int index cannot be contiguous")
        # On the slowest axis, a full read is downgraded to a contiguous
        # read, or to no read at all for an integer index.
        if is_slowest and action == "full":
            action = None if is_int else "contiguous"
        if action == "full":
            return slice(None), slice(None), slicer
        elif action == "contiguous":  # an int was rejected above
            # Slices with a step of 1 or -1 fall through to the default.
            step = slicer.step
            if step not in (-1, 1):
                if step < 0:
                    slicer = _positive_slice(slicer)
                return (
                    slice(None, None, -1 if step < 0 else 1),
                    slice(slicer.start, slicer.stop, 1),
                    slice(None, None, slicer.step),
                )
    # By default, the pre-slicer only makes the step positive.
    if is_int:
        return None, slicer, slice(None)
    if slicer.step > 0:
        return slice(None), slicer, slice(None)
    return slice(None, None, -1), _positive_slice(slicer), slice(None)
