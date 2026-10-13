"""Imaris voxels: the lazy array that reads one resolution level."""

# stdlib
import contextlib
import operator
from io import BytesIO

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# this format
from ._raw import ImarisLevel, ImarisRaw

_Source = tx.Union[str, bytes, h5py.File]
"""Where the voxels are read from: a local path, the bytes of a file, or
an HDF5 file that the caller opened."""


class _ImarisProxy:
    """A lazy array of the voxels of one resolution level of an Imaris file.

    An Imaris file stores each channel at each time point of a level as
    its own `Data` dataset, ordered z, y, x and padded to whole chunks.
    The proxy presents all of them as one array of shape
    `(channels, time points, z, y, x)`, cropped to the size of the image,
    which is the reverse of the order of the data of an image. It knows
    its shape and type from the record, and it reads the file only when
    its values are needed, opening the file each time unless the caller
    gave an open file. A selection reads only the datasets and the voxels
    that it covers, so that Dask reads one block at a time, and an empty
    selection reads nothing.
    """

    def __init__(self, source: _Source, record: ImarisRaw, level: int) -> None:
        self.source = source
        """The path of the file, its bytes, or the open file."""
        self.record = record
        """The record of the file, which gives the shape and the type."""
        self.level = level
        """The index of the resolution level that the proxy reads."""
        self._chunks: tx.Optional[tx.Tuple[int, ...]] = None

    @property
    def _level(self) -> ImarisLevel:
        return self.record.levels[self.level]

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape `(channels, time points, z, y, x)`, from the record."""
        x, y, z = self._level.size
        return (self.record.channels, self.record.timepoints, z, y, x)

    @property
    def ndim(self) -> int:
        """The number of dimensions, which is 5."""
        return 5

    @property
    def dtype(self) -> np.dtype:
        """The type of the voxels."""
        return self._level.dtype

    @property
    def chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of the chunks of the file, or `None` if it has none.

        A chunk covers one channel at one time point. Dask reads it to
        choose blocks that hold whole chunks. The file is opened once to
        read it, and the voxels are not read.
        """
        if self._chunks is None:
            with _opened(self.source) as f:
                chunks = f[_path(self.level, 0, 0)].chunks
            if chunks is None:
                return None
            self._chunks = (1, 1) + tuple(
                min(int(c), n) for c, n in zip(chunks, self.shape[2:])
            )
        return self._chunks

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        values = self[(slice(None),) * 5]
        return values if dtype is None else values.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        # The selection is checked, and an empty one, such as the one that
        # Dask reads to learn the type of an array, is answered without
        # reading the file.
        selected = np.broadcast_to(np.empty((), self.dtype), self.shape)[index]
        if selected.size == 0:
            return np.empty(selected.shape, self.dtype)
        parts = _basic_index(index, self.shape)
        if parts is None:
            # Other selections, such as arrays of indices, read the whole
            # level first.
            return np.asarray(self)[index]
        return self._read(parts)

    def _read(self, parts: tx.List[tx.Union[int, slice]]) -> np.ndarray:
        """Read a selection of integers and slices, one per axis."""
        channels = _positions(parts[0], self.shape[0])
        times = _positions(parts[1], self.shape[1])
        spatial = [_span(p, n) for p, n in zip(parts[2:], self.shape[2:])]
        read = tuple(slice(lo, hi) for lo, hi, _ in spatial)
        steps = tuple(step for _, _, step in spatial)
        shape = [len(channels), len(times)]
        shape += [len(range(lo, hi)[s]) for lo, hi, s in spatial]
        out = np.empty(shape, self.dtype)
        with _opened(self.source) as f:
            for i, channel in enumerate(channels):
                for j, time in enumerate(times):
                    block = f[_path(self.level, time, channel)][read]
                    out[i, j] = block[steps]
        keep = tuple(0 if isinstance(p, int) else slice(None) for p in parts)
        return out[keep]

    def __repr__(self) -> str:
        return (
            f"_ImarisProxy(level={self.level}, shape={self.shape}, "
            f"dtype={self.dtype})"
        )


def read_imaris_level(
    source: _Source, record: ImarisRaw, level: int
) -> _ImarisProxy:
    """Return a lazy array of the voxels of one level of an Imaris file.

    Parameters
    ----------
    source : str, bytes or h5py.File
        The local path of the file, the bytes of the file, or the file
        open for reading, which the caller keeps open as long as the
        array is used.
    record : ImarisRaw
        The record of the file.
    level : int
        The index of the resolution level, from 0 for the finest.

    Returns
    -------
    array
        The lazy array, of shape `(channels, time points, z, y, x)`.

    Raises
    ------
    IndexError
        If the file has no such level.
    """
    if not 0 <= level < len(record.levels):
        raise IndexError(
            f"The Imaris file has {len(record.levels)} resolution levels, "
            f"so it has no level {level}."
        )
    return _ImarisProxy(source, record, level)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


@contextlib.contextmanager
def _opened(source: _Source) -> tx.Iterator[h5py.File]:
    """Open the file of the voxels for the duration of a read.

    A path or bytes are opened and closed again, so that Dask can read
    from several threads, and a file that the caller opened is used as it
    is and stays open.
    """
    if isinstance(source, h5py.File):
        yield source
        return
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(bytes(source))
    with h5py.File(source, "r") as f:
        yield f


def _path(level: int, time: int, channel: int) -> str:
    """Return the path of the `Data` dataset of a level, time and channel."""
    return (
        f"/DataSet/ResolutionLevel {level}/TimePoint {time}/"
        f"Channel {channel}/Data"
    )


def _basic_index(
    index: tx.Any, shape: tx.Sequence[int]
) -> tx.Optional[tx.List[tx.Union[int, slice]]]:
    """Return one integer or slice per axis, or `None` for any other
    selection.

    An ellipsis is expanded and missing axes are selected whole. Negative
    integers are counted from the end. The selection is assumed to be
    valid, since the caller has checked it.
    """
    if not isinstance(index, tuple):
        index = (index,)
    if sum(item is Ellipsis for item in index) > 1:
        return None
    parts: tx.List[tx.Any] = []
    for item in index:
        if item is Ellipsis:
            parts.append(item)
        elif isinstance(item, slice):
            parts.append(item)
        elif isinstance(item, (bool, np.bool_)) or item is None:
            return None
        else:
            try:
                parts.append(operator.index(item))
            except TypeError:
                return None
    if Ellipsis in parts:
        at = parts.index(Ellipsis)
        fill = [slice(None)] * (len(shape) - len(parts) + 1)
        parts[at : at + 1] = fill
    parts += [slice(None)] * (len(shape) - len(parts))
    return [
        p + n if isinstance(p, int) and p < 0 else p
        for p, n in zip(parts, shape)
    ]


def _positions(part: tx.Union[int, slice], size: int) -> tx.List[int]:
    """Return the positions that an integer or a slice selects."""
    if isinstance(part, int):
        return [part]
    return list(range(*part.indices(size)))


def _span(part: tx.Union[int, slice], size: int) -> tx.Tuple[int, int, slice]:
    """Return the contiguous range to read for an integer or a slice, and
    the step that selects the result from that range.

    A slice with a negative step is read as the increasing range that
    holds it, and the step then runs from the end of that range.
    """
    if isinstance(part, int):
        return part, part + 1, slice(None)
    selected = range(*part.indices(size))
    low, high = min(selected[0], selected[-1]), max(selected[0], selected[-1])
    return low, high + 1, slice(None, None, selected.step)
