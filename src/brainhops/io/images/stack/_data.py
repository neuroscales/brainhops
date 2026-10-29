"""The pixels of an image stack: the lazy array that reads its files."""

# stdlib
import math
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops._core.properties import alias
from brainhops._core.proxy import StoredProxy
from brainhops._core.typing import Internal
from brainhops.io.base.parsers import ParserContentError

# this format
from ._raw import StackMember, StackRaw


class _StackProxy(StoredProxy):
    """A lazy array of the pixels of an image stack.

    The proxy holds the path of the folder and the record of the stack.
    It presents the slices stacked along a new first axis, in the order of
    their numbers, so its shape is `(slices, rows, columns[, channels])`,
    as `StackRaw.shape` gives it. Each file is read with the reader of its
    format, tifffile or Pillow, only when one of its pixels is selected,
    and no file is kept open. A selection along the first axis therefore
    reads only the files that it selects, which lets Dask read one file
    per chunk (see `chunks`). Indexing the proxy returns another lazy
    array (see [`StoredProxy`][]), which reads the files when it is
    converted. A slice whose number no file carries is
    filled with `fill`, and reading it reads no file.

    Bringing the axes of a file into the order `(rows, columns[,
    channels])` depends on the record, so it happens in the proxy, as the
    permuted layouts of MRtrix do.
    """

    # --- magic attributes ----

    folder: path.Path
    """The path of the folder that holds the files."""

    record: NoRepr[StackRaw]
    """"The record of the stack, which lists the files in order."""

    _fill: tx.Any = None
    """
    The value of the pixels of a missing slice, as it was given as `fill=`,
    or `None` for the default of the type.
    """

    _members: Internal[tx.Dict[int, int]]
    """The index in the record of the file of each slice."""

    def __post_init__(self) -> None:
        # A fill value that the type of the stack cannot hold is refused
        # before anything is read.
        fill_value(self._fill, self.record.dtype)
        # Each slice is found by its number, counted from the first file.
        members = self.record.members
        first = members[0].number if members else 0
        self._members = {
            member.number - first: index
            for index, member in enumerate(members)
        }

    # --- properties ----

    @property
    def fill(self) -> tx.Any:
        """The value of a missing pixel, in the type of the stack."""
        return fill_value(self._fill, self.record.dtype)

    shape = alias("record.shape", readonly=True)
    """The shape of the stacked pixels, read from the record."""

    dtype = alias("record.dtype", readonly=True)
    """The type of the pixels."""

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The shape of the block that one file fills: one plane."""
        return (1,) + self.shape[1:]

    # --- methods ----

    def to_dask(
        self, *, device: tx.Union[str, int, None] = "cpu", **kwargs: tx.Any
    ) -> tx.Any:
        """Return a Dask array with one chunk per slice, reading nothing.

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
        # `name=False` gives Dask a random name instead of a hash of the
        # proxy, which Dask computes only to recognize two arrays of the
        # same source (docstring of `dask.array.from_array`, dask
        # 2023.5.0), and which says nothing about the files.
        kwargs.setdefault("name", False)
        return super().to_dask(device=device, **kwargs)

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
            return np.full(self.shape[1:], self.fill, self.dtype)
        member = self.record.members[index]
        filename = os.path.join(self.folder, *member.name.split("/"))
        if not os.path.isfile(filename):
            raise ParserContentError(
                f"The file {member.name!r} of the stack in "
                f"{str(self.folder)!r} no longer exists."
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
        out = np.empty((len(positions),) + self.shape[1:], self.dtype)
        for k, position in enumerate(positions):
            out[k] = self.read_member(position)
        return out

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        # Only the files of the slices that the first index selects are
        # read.
        first, rest = index[0], index[1:]
        if isinstance(first, int):
            return self.read_member(first)[rest]
        positions = range(self.shape[0])[first]
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
