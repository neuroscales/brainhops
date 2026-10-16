"""The pixels of an image stack: the lazy array that reads its files."""

# stdlib
import math
import os

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.proxy import ArrayProxy
from brainhops.io.base.parsers import ParserContentError

# this format
from ._raw import StackMember, StackRaw


class _StackProxy(ArrayProxy):
    """A lazy array of the pixels of an image stack.

    The proxy holds the path of the folder and the record of the stack.
    It presents the slices stacked along a new first axis, in the order of
    their numbers, so its shape is `(slices, rows, columns[, channels])`,
    as `StackRaw.shape` gives it. Each file is read with the reader of its
    format, tifffile or Pillow, only when one of its pixels is selected,
    and no file is kept open. A selection along the first axis therefore
    reads only the files that it selects, which lets Dask read one file
    per chunk (see `chunks`). Indexing the proxy returns another lazy
    array (see [`ArrayProxy`][]), which reads the files when it is
    converted. A slice whose number no file carries is
    filled with `fill`, and reading it reads no file.

    Bringing the axes of a file into the order `(rows, columns[,
    channels])` depends on the record, so it happens in the proxy, as the
    permuted layouts of MRtrix do.
    """

    def __init__(
        self, folder: str, record: StackRaw, fill: tx.Any = None
    ) -> None:
        self.folder = folder
        """The path of the folder that holds the files."""
        self.record = record
        """The record of the stack, which lists the files in order."""
        self.fill = fill_value(fill, record.dtype)
        """The value of the pixels of a missing slice."""
        first = record.members[0].number if record.members else 0
        self._members = {
            member.number - first: index
            for index, member in enumerate(record.members)
        }

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stacked pixels, read from the record."""
        return tuple(self.record.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the pixels."""
        return self.record.dtype

    @property
    def base_chunks(self) -> tx.Tuple[int, ...]:
        """The shape of the block that one file fills: one plane."""
        return (1,) + self.base_shape[1:]

    def filenames(self) -> tx.List[str]:
        """Return the paths of the files that the proxy reads.

        Returns
        -------
        list[str]
            The path of each file of the record, in order.
        """
        return [
            os.path.join(self.folder, *member.name.split("/"))
            for member in self.record.members
        ]

    def read_member(self, position: int) -> np.ndarray:
        """Return the pixels of one slice, as `(rows, columns[, channels])`.

        Parameters
        ----------
        position : int
            The position of the slice along the stacking axis.

        Returns
        -------
        numpy.ndarray
            The pixels of the file of the slice, or the fill value when
            the slice has no file.

        Raises
        ------
        ParserContentError
            If the file cannot be read, or if it no longer has the shape or
            the type that the record says.
        """
        index = self._members.get(position)
        if index is None:
            return np.full(self.base_shape[1:], self.fill, self.dtype)
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

    def read_members(self, positions: tx.Sequence[int]) -> np.ndarray:
        """Return the pixels of several slices, stacked along a first axis.

        Parameters
        ----------
        positions : Sequence[int]
            The positions of the slices along the stacking axis.

        Returns
        -------
        numpy.ndarray
            The pixels, of shape `(len(positions), rows, columns[,
            channels])`.
        """
        out = np.empty((len(positions),) + self.base_shape[1:], self.dtype)
        for k, position in enumerate(positions):
            out[k] = self.read_member(position)
        return out

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        # Only the files of the slices that the first index selects are
        # read.
        first, rest = index[0], index[1:]
        if isinstance(first, int):
            return self.read_member(first)[rest]
        positions = range(self.base_shape[0])[first]
        return self.read_members(positions)[(slice(None),) + rest]


def fill_value(fill: tx.Any, dtype: np.dtype) -> tx.Any:
    """Return the value of the pixels of a missing slice, of a given type.

    By default, it is NaN for a floating or complex type, and zero for any
    other type, such as an integer type or `bool`.

    Raises
    ------
    ValueError
        If `fill` cannot be stored exactly in `dtype`, such as NaN in an
        integer type or 300 in `uint8`.
    """
    dtype = np.dtype(dtype)
    if fill is None:
        if dtype.kind in "fc":
            return dtype.type(math.nan)
        return dtype.type(0)
    try:
        with np.errstate(all="ignore"):
            value = np.asarray(fill).astype(dtype)
        same = value.shape == () and (
            value == np.asarray(fill)
            or (np.isnan(value) and np.isnan(np.asarray(fill)))
        )
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(
            f"fill must be a value that the type {dtype} of the stack "
            f"stores exactly, not {fill!r}."
        )
    return value[()]


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
