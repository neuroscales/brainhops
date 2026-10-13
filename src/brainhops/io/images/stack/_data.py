"""The pixels of an image stack: the lazy array that reads its files."""

# stdlib
import os
from numbers import Integral

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

# this format
from ._raw import StackMember, StackRaw


class _StackProxy:
    """A lazy array of the pixels of an image stack.

    The proxy holds the path of the folder and the record of the stack.
    It presents the files stacked along a new first axis, in the order of
    the record, so its shape is `(files, rows, columns[, channels])`, as
    `StackRaw.shape` gives it. Each file is read with the reader of its
    format, tifffile or Pillow, only when one of its pixels is selected,
    and no file is kept open. A selection along the first axis therefore
    reads only the files that it selects, which lets Dask read one file
    per chunk (see `chunks`).

    Bringing the axes of a file into the order `(rows, columns[,
    channels])` depends on the record, so it happens in the proxy, as the
    permuted layouts of MRtrix do.
    """

    def __init__(self, folder: str, record: StackRaw) -> None:
        self.folder = folder
        """The path of the folder that holds the files."""
        self.record = record
        """The record of the stack, which lists the files in order."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stacked pixels, read from the record."""
        return self.record.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the stacked pixels."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the pixels."""
        return self.record.dtype

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The shape of the block that one file fills: one plane."""
        return (1,) + self.shape[1:]

    def read_member(self, index: int) -> np.ndarray:
        """Return the pixels of one file, as `(rows, columns[, channels])`.

        Parameters
        ----------
        index : int
            The position of the file in the record.

        Returns
        -------
        numpy.ndarray
            The pixels of the file.

        Raises
        ------
        ParserContentError
            If the file cannot be read, or if it no longer has the shape or
            the type that the record says.
        """
        member = self.record.members[index]
        filename = os.path.join(self.folder, *member.name.split("/"))
        if not os.path.isfile(filename):
            raise ParserContentError(
                f"The file {member.name!r} of the stack in "
                f"{self.folder!r} no longer exists."
            )
        values = _read_file(filename, member)
        if tuple(values.shape) != member.shape or values.dtype != np.dtype(
            member.dtype
        ):
            raise ParserContentError(
                f"The file {member.name!r} holds an array of shape "
                f"{tuple(values.shape)} and type {values.dtype}, but the "
                f"stack was read with shape {member.shape} and type "
                f"{member.dtype}."
            )
        return values.transpose(member.plane_order)

    def read_members(self, indices: tx.Sequence[int]) -> np.ndarray:
        """Return the pixels of several files, stacked along a first axis.

        Parameters
        ----------
        indices : Sequence[int]
            The positions of the files in the record.

        Returns
        -------
        numpy.ndarray
            The pixels, of shape `(len(indices), rows, columns[,
            channels])`.
        """
        out = np.empty((len(indices),) + self.shape[1:], self.dtype)
        for k, index in enumerate(indices):
            out[k] = self.read_member(index)
        return out

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        values = self.read_members(range(self.shape[0]))
        return values if dtype is None else values.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        # An empty selection, such as the one that Dask reads to learn the
        # type of an array, is answered without reading the files.
        selected = np.broadcast_to(np.empty((), self.dtype), self.shape)[index]
        if selected.size == 0:
            return np.empty(selected.shape, self.dtype)
        index = index if isinstance(index, tuple) else (index,)
        first = index[0] if index else slice(None)
        rest = index[1:]
        if isinstance(first, Integral) and not isinstance(first, bool):
            position = range(self.shape[0])[first]
            return self.read_member(position)[rest]
        if isinstance(first, slice):
            positions = range(self.shape[0])[first]
            return self.read_members(positions)[(slice(None),) + rest]
        # Any other selection, such as an ellipsis or an array of indices,
        # reads every file.
        return np.asarray(self)[index]

    def __repr__(self) -> str:
        return f"_StackProxy(shape={self.shape}, dtype={self.dtype})"


def _read_file(filename: str, member: StackMember) -> np.ndarray:
    """Decode one file with the reader of its format.

    The readers are those of the single-file formats: `read_series` of the
    TIFF backend, without memory-mapping, and `read_pillow` of the Pillow
    backend. Their modules are imported here, because the TIFF package
    registers its formats when it is imported, which must happen only
    when tifffile is installed.
    """
    if member.format == "tiff":
        from brainhops.io.images.tiff import _utils as tiff

        return np.asarray(
            tiff.read_series(
                tiff.TiffSource(filename=filename), mmap=False, lazy=False
            )
        )
    from brainhops.io.images.pillow._utils import read_pillow

    return read_pillow(filename).array
