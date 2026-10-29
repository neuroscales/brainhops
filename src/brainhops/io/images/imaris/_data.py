"""Imaris voxels: the lazy array that reads one resolution level."""

# stdlib
import contextlib
from io import BytesIO

# dependencies
import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import NoInit, NoRepr

# internals
from brainhops._core.proxy import StoredProxy

# this format
from ._raw import ImarisLevel, ImarisRaw

_Source = tx.Union[str, bytes, h5py.File]
"""
Where the voxels are read from: a local path, the bytes of a file, or
an HDF5 file that the caller opened.
"""


class ImarisProxy(StoredProxy):
    """A lazy array of the voxels of one resolution level of an Imaris file.

    An Imaris file stores each channel at each time point of a level as
    its own `Data` dataset, ordered z, y, x and padded to whole chunks.
    The proxy presents all of them as one array of shape
    `(channels, time points, z, y, x)`, cropped to the size of the image,
    which is the reverse of the order of the data of an image. It knows
    its shape and type from the record, and it reads the file only when
    its values are needed, opening the file each time unless the caller
    gave an open file. Indexing the proxy returns another lazy array (see
    [`StoredProxy`][]), and converting that array reads only the datasets
    and the voxels that it covers, so that Dask reads one block at a
    time.
    """

    # --- magic attributes ----

    source: tx.Any
    """The path of the file, its bytes, or the open file."""

    record: NoRepr[ImarisRaw]
    """The record of the file, which gives the shape and the type."""

    level: int = 0
    """The index of the resolution level that the proxy reads."""

    _chunks: NoInit[NoRepr[tx.Optional[tx.Tuple[int, ...]]]] = None
    """The chunks of the level, once they were read."""

    # --- properties ----

    @property
    def _level(self) -> ImarisLevel:
        return self.record.levels[self.level]

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape `(channels, time points, z, y, x)`, from the record."""
        x, y, z = self._level.size
        return (self.record.channels, self.record.timepoints, z, y, x)

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

    # --- methods ----

    def _read(self, parts: tx.Tuple[tx.Union[int, slice], ...]) -> np.ndarray:
        """Read a box of integers and slices, one per axis."""
        shape = self.shape
        channels = _positions(parts[0], shape[0])
        times = _positions(parts[1], shape[1])
        spatial = [_span(p, n) for p, n in zip(parts[2:], shape[2:])]
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


def read_imaris_level(
    source: _Source, record: ImarisRaw, level: int
) -> ImarisProxy:
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
    return ImarisProxy(source, record, level)


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
