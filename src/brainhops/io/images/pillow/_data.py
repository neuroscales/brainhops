"""The pixels of a raster image: the lazy array that decodes them with
Pillow, and finding them."""

# stdlib
import contextlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core import path

# internals
from brainhops._core.proxy import ArrayProxy
from brainhops._core.streams import preserve_position
from brainhops.io.base._utils_files import open_path, same_file
from brainhops.io.base.parsers import ParserContentError, ParserExistsError

# this format
from ._raw import PillowRaw
from ._utils import read_pillow


class _PillowProxy(ArrayProxy):
    """A lazy array of the pixels of a frame of a raster file.

    The proxy holds where the file lies, which is a path or the stream
    that the caller passed, the record of the frame, and whether the
    colours of a palette image are looked up. It decodes the frame with
    Pillow each time its values are needed and keeps no file open, but a
    stream given by the caller must stay open while the values may be
    read. Pillow reads a stream from its start, whatever its position, so
    the file is the whole stream. The values are `(rows, columns)`, or
    `(rows, columns, samples)`, as Pillow decodes them, and their shape and
    type are known from the record without decoding anything. Indexing
    the proxy returns another lazy array (see [`ArrayProxy`][]), but
    Pillow decodes the whole frame for any read.

    The proxy also gives the bytes of the file, with [`read_stored`][], so
    that an image that was read and not changed is written back as the
    file that it was read from.
    """

    source: tx.Annotated[
        tx.Any, tx.Doc("The path of the file, or the stream that holds it.")
    ]
    record: NoRepr[
        PillowRaw,
        tx.Doc(
            "The record of the frame, which says what the frame decodes to."
        ),
    ]
    palette: tx.Annotated[
        bool, tx.Doc("Whether the colours of a palette image are looked up.")
    ]

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the decoded frame, read from the record."""
        return tuple(self.record.layout(self.palette)[0])

    @property
    def dtype(self) -> np.dtype:
        """The type of the decoded pixels, read from the record."""
        return self.record.layout(self.palette)[1]

    def read_stored(self) -> bytes:
        """Return the bytes of the file.

        The bytes are those of the whole file, whatever part of it a view
        of the proxy selects.

        Returns
        -------
        bytes
            The content of the file, from its first byte.
        """
        with _opened(self.source) as f:
            f.seek(0)
            return f.read()

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        with _opened(self.source) as f:
            frame = read_pillow(
                f, frame=self.record.frame, palette=self.palette
            )
        values = frame.array
        shape = self.base_shape
        if values.shape != shape or values.dtype != self.dtype:
            raise ParserContentError(
                f"Pillow decoded frame {self.record.frame} as an array of "
                f"shape {values.shape} and type {values.dtype}, but its "
                f"header describes shape {shape} and type {self.dtype}."
            )
        return values[index]


def read_pillow_image(
    file: tx.Any, frame: int = 0, palette: bool = True
) -> tx.Tuple[PillowRaw, _PillowProxy]:
    """Read the record and a lazy array of a frame of a raster file.

    The record is read from the header, and the pixels are not decoded.

    Parameters
    ----------
    file : path-like or file object
        The path of the file, or a stream open in binary mode, which the
        lazy array keeps.
    frame : int, default=0
        The frame to read. A negative index counts from the end.
    palette : bool, default=True
        Whether the lazy array looks up the colours of a palette image.

    Returns
    -------
    raw : PillowRaw
        The record.
    proxy : _PillowProxy
        A lazy array of the pixels of the frame.

    Raises
    ------
    ParserExistsError
        If the path does not exist.
    ParserContentError
        If Pillow cannot read the file.
    IndexError
        If the file has no such frame.
    """
    if _is_path(file):
        if isinstance(file, str):
            file = path.Path(file)
        if not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        with open_path(file) as f:
            record = PillowRaw.from_fileobj(f, frame=frame)
    else:
        record = PillowRaw.from_fileobj(file, frame=frame)
    return record, _PillowProxy(file, record, palette)


def reads_from(raw: tx.Any, filename: tx.Any) -> bool:
    """Tell whether a lazy array reads the file at a path.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an image.
    filename : path-like
        The path of a file that a writer writes.

    Returns
    -------
    bool
        Whether `raw` is a lazy array of that file.
    """
    if not isinstance(raw, _PillowProxy) or not _is_path(raw.source):
        return False
    return same_file(raw.source, filename)


@contextlib.contextmanager
def _opened(source: tx.Any) -> tx.Iterator[tx.BinaryIO]:
    """Open a path, or protect the position of a stream, for reading."""
    if _is_path(source):
        if isinstance(source, str):
            source = path.Path(source)
        with open_path(source) as f:
            yield f
    else:
        with preserve_position(source):
            yield source


def _is_path(file: tx.Any) -> bool:
    """Tell whether a source is a path rather than a stream."""
    return isinstance(file, (str, os.PathLike, path.PathLike))
