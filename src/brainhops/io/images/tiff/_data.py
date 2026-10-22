"""The lazy array of a level of a TIFF file."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.proxy import ArrayProxy
from brainhops.io.images.base import _utils_raster as raster

# this format
from . import _utils as backend
from ._raw import TiffRaw


class _TiffProxy(ArrayProxy):
    """A lazy array of the pixels of one level of a series of a TIFF file.

    The proxy holds the source of the file, which is a local file name or
    the bytes of the file, the record of the series, and the index of the
    level in the record. It reads the pixels each time its values are
    needed and keeps no file open. The values are presented in the order
    of brainhops, spatial axes x, y and z first, then time, channels and
    other axes, as a transposed view of the array that tifffile reads,
    which is laid out in the order of `TiffRaw.storage_axes`.

    How the pixels are read is decided by the reader options `mmap` and
    `lazy` that the proxy keeps, as
    [`read_series`][brainhops.io.images.tiff._utils.read_series] describes:
    a local, uncompressed and contiguous level is memory-mapped, a level
    may be read as a dask array, and any other level is read in full.
    Indexing the proxy returns another lazy array (see [`ArrayProxy`][]),
    which reads only its selection of a memory map or of a dask array.
    """

    def __init__(
        self,
        source: backend.TiffSource,
        record: TiffRaw,
        level: int,
        mmap: bool = True,
        lazy: tx.Optional[bool] = None,
    ) -> None:
        self.source = source
        """The file name or the bytes of the file."""
        self.record = record
        """The record of the series that the level belongs to."""
        self.level = int(level)
        """The index of the level in the record."""
        self.mmap = bool(mmap)
        """Whether a local, uncompressed, contiguous level is memory-mapped."""
        self.lazy = lazy
        """Whether the level is read as a dask array, or `None` for the
        array backend to decide."""

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the level in the order of brainhops."""
        stored = self.record.levels[self.level].shape
        perm = raster.canonical_permutation(self.record.storage_axes)
        return tuple(int(stored[p]) for p in perm)

    @property
    def dtype(self) -> np.dtype:
        """The type of the pixels."""
        return np.dtype(self.record.levels[self.level].dtype)

    def read(self) -> tx.Any:
        """Return the pixels as a view in the order of brainhops.

        The array is a transposed view of the memory map, of the dask array
        or of the NumPy array that tifffile gives, so a memory map and a
        dask array are not read here. A view of the proxy reads its
        selection and returns it as a NumPy array.

        Raises
        ------
        ParserContentError
            If tifffile cannot decode the level.
        """
        if self.is_view:
            return np.asarray(self)
        return self._level_array()

    def _level_array(self) -> tx.Any:
        """Return the whole level as tifffile gives it, transposed."""
        stored = backend.read_series(
            self.source,
            series=self.record.series,
            level=self.level,
            mmap=self.mmap,
            lazy=self.lazy,
        )
        data, _ = raster.to_canonical(stored, self.record.storage_axes)
        return data

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        return np.asanyarray(self._level_array()[index])

    def to_dask(
        self, *, device: tx.Union[str, int, None] = "cpu", **kwargs: tx.Any
    ) -> tx.Any:
        """Return a Dask array of the pixels, reading nothing.

        For the whole level, the array is the Dask array that tifffile
        gives, or a Dask array over the memory map or the decoded level,
        so that tifffile reads the level once rather than once per block.
        A view of the proxy reads its blocks as [`ArrayProxy`][] does.

        Parameters
        ----------
        device : str or int, default="cpu"
            The device for the chunks.
        **kwargs : Any
            Options of `dask.array.from_array`.

        Returns
        -------
        dask.array.Array
            The lazy array.
        """
        from brainhops import backends

        if self.is_view or backends.da is None:
            return super().to_dask(device=device, **kwargs)
        level = self._level_array()
        if not isinstance(level, backends.da.Array):
            kwargs.setdefault("chunks", self.chunks or "auto")
            level = backends.da.from_array(level, **kwargs)
        return backends.to_dask(level, device=device)

    def read_bytes(self) -> bytes:
        """Return the bytes of the whole file that the proxy reads.

        Raises
        ------
        OSError
            If the file cannot be read.
        """
        if self.source.content is not None:
            return self.source.content
        with open(self.source.filename, "rb") as file:
            return file.read()
