"""JPEG 2000 samples: the lazy array that decodes them with Pillow.

Pillow decodes a whole resolution level at once, with OpenJPEG. Its own
`reduce` rounds the size of a reduced level instead of taking its
ceiling, and the decoder then refuses a level whose size is not a
multiple of its factor, so a level is decoded here with the size that
the codestream defines (ITU-T T.800, B.5).
"""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from PIL import Image, ImageFile

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base._utils_files import is_local, same_file
from brainhops.io.base.parsers import ParserContentError, ParserExistsError
from brainhops.io.common.jpeg2000 import Jpeg2000Raw
from brainhops.io.common.jpeg2000._raw import read_record
from brainhops.io.images.pillow._utils import pillow_to_array


class _Jpeg2000Source:
    """Where a JPEG 2000 file lies, so that it can be decoded again.

    The source is the path of a local file, which is opened each time a
    level is decoded, or the bytes of any other file, held in memory. It
    keeps no file open.
    """

    def __init__(
        self,
        filename: tx.Optional[str] = None,
        content: tx.Optional[bytes] = None,
    ) -> None:
        if (filename is None) == (content is None):
            raise ValueError("Give either a filename or the content.")
        self.filename = filename
        """The path of a local file, or `None`."""
        self.content = content
        """The bytes of the file, or `None` for a local file."""

    def open(self) -> tx.BinaryIO:
        """Open the file in binary mode, as a context manager."""
        if self.filename is not None:
            return open(self.filename, "rb")
        return BytesIO(self.content)

    def read(self) -> bytes:
        """Return the bytes of the file."""
        if self.content is not None:
            return self.content
        with open(self.filename, "rb") as f:
            return f.read()

    def is_file(self, filename: tx.Any) -> bool:
        """Return whether the source is a local file of a given path."""
        return self.filename is not None and same_file(self.filename, filename)

    def __repr__(self) -> str:
        if self.filename is not None:
            return f"_Jpeg2000Source({self.filename!r})"
        return f"_Jpeg2000Source(<{len(self.content)} bytes>)"


class _Jpeg2000Proxy:
    """A lazy array of the samples of one resolution level.

    The proxy holds the source of the file, the record that the file was
    read with, and the level. It decodes the level with Pillow each time
    its values are needed, and keeps no file open. The samples are
    presented as Pillow decodes them, `(rows, columns)` or `(rows,
    columns, samples)`, with the values that the codestream holds: the
    scaling of Pillow to 8 or 16 bits, and its offset of signed samples,
    are undone.
    """

    def __init__(
        self, source: _Jpeg2000Source, record: Jpeg2000Raw, level: int
    ) -> None:
        self.source = source
        """Where the file lies."""
        self.record = record
        """The record that the file was read with, which gives the shape
        and the type of the samples."""
        self.level = level
        """The resolution level, where 0 is the full resolution."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the samples, read from the record."""
        return self.record.stored_shape(self.level)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the samples."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the samples, read from the record."""
        return self.record.dtype

    def read_samples(self) -> np.ndarray:
        """Decode the samples of the level.

        Returns
        -------
        numpy.ndarray
            The samples, C-ordered.

        Raises
        ------
        ParserContentError
            If Pillow cannot decode the level, would lose precision (a
            colour image of more than 8 bits per component), or decodes
            samples of another shape or type than the record says, or if
            a reduced level of an image whose area has an offset is asked
            for, which Pillow misplaces.
        PIL.Image.DecompressionBombError
            If the full-resolution image is larger than the limit of
            Pillow, even when a coarse level is asked for.
        """
        values = _decode_level(self.source, self.record, self.level)
        if values.shape != self.shape or values.dtype != self.dtype:
            raise ParserContentError(
                f"Pillow decoded level {self.level} of this JPEG 2000 file "
                f"as {values.dtype} samples of shape {values.shape}, but its "
                f"header says {self.dtype} samples of shape {self.shape}."
            )
        return values

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        values = self.read_samples()
        return values if dtype is None else values.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        # An empty selection, such as the one that Dask reads to learn the
        # type of an array, is answered without reading the file.
        selected = np.broadcast_to(np.empty((), self.dtype), self.shape)[index]
        if selected.size == 0:
            return np.empty(selected.shape, self.dtype)
        return np.asarray(self)[index]

    def __repr__(self) -> str:
        return (
            f"_Jpeg2000Proxy(level={self.level}, shape={self.shape}, "
            f"dtype={self.dtype})"
        )


def read_jpeg2000(file: tx.Any) -> tx.Tuple[Jpeg2000Raw, _Jpeg2000Source]:
    """Read the record of a JPEG 2000 file and where it lies.

    A local path is read for its header alone, and the samples are
    decoded from it by name when they are needed. Any other path, and a
    stream, are read into memory, and the position of a stream is
    restored. Bytes are kept as they are.

    Parameters
    ----------
    file : path-like or file object or bytes
        The path of the file, a stream open in binary mode, or the bytes
        of the file.

    Returns
    -------
    raw : Jpeg2000Raw
        The record.
    source : _Jpeg2000Source
        Where the file lies.

    Raises
    ------
    ParserExistsError
        If the path does not exist.
    ParserContentError
        If the content is neither a JP2 file nor a codestream.
    """
    if isinstance(file, (bytes, bytearray)):
        content = bytes(file)
        return read_record(content), _Jpeg2000Source(content=content)
    if isinstance(file, str):
        file = path.Path(file)
    if isinstance(file, os.PathLike):
        if not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        if is_local(file):
            filename = os.fspath(file)
            with open(filename, "rb") as f:
                record = Jpeg2000Raw.from_fileobj(f)
            return record, _Jpeg2000Source(filename=filename)
        with file.open("rb") as f:
            content = f.read()
        return read_record(content), _Jpeg2000Source(content=content)
    with preserve_position(file):
        content = file.read()
    return read_record(content), _Jpeg2000Source(content=content)


def _decode_level(
    source: _Jpeg2000Source, record: Jpeg2000Raw, level: int
) -> np.ndarray:
    """Decode one resolution level with Pillow.

    Raises
    ------
    ParserContentError
        If Pillow cannot decode the level, or would lose precision.
    PIL.Image.DecompressionBombError
        If the full-resolution image is larger than the limit of Pillow.
    """
    if level and any(record.image_offset):
        # Pillow places a reduced level at the full-resolution offset,
        # outside the image, and refuses it.
        raise ParserContentError(
            f"Pillow cannot decode a reduced level of a JPEG 2000 image "
            f"whose area has an offset on the reference grid "
            f"{record.image_offset}, so only level 0 can be read."
        )
    width, height = record.level_shape(level)
    with source.open() as f:
        try:
            im = Image.open(f, formats=["JPEG2000"])
        except Image.DecompressionBombError:
            raise
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot read this JPEG 2000 file: {e}"
            ) from e
        try:
            if level:
                # The `reduce` of Pillow rounds the size of the level,
                # which the decoder refuses when it is not the ceiling, so
                # the tile is given the size that the codestream defines.
                tile = im.tile[0]
                args = tile[3]
                new = (
                    tile[0],
                    (0, 0, width, height),
                    tile[2],
                    (args[0], int(level), 0) + tuple(args[3:]),
                )
                if hasattr(tile, "_replace"):
                    new = type(tile)(*new)
                im._size = (width, height)
                im.tile = [new]
                ImageFile.ImageFile.load(im)
            else:
                im.load()
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot decode level {level} of this JPEG 2000 "
                f"file: {e}"
            ) from e
        mode = im.mode
        array, _ = pillow_to_array(im)
    if mode in ("P", "PA"):
        return array
    return _undo_pillow_scaling(array, record)


def _undo_pillow_scaling(array: np.ndarray, record: Jpeg2000Raw) -> np.ndarray:
    """Return the samples that the codestream holds.

    Pillow decodes a component of `b` bits into its 8-bit or 16-bit mode
    as `(v + offset) << (bits - b)`, where `offset` is `2**(b - 1)` for a
    signed component and 0 otherwise. This function undoes it.

    Raises
    ------
    ParserContentError
        If the component has more bits than Pillow decodes.
    """
    bits = array.dtype.itemsize * 8
    depth = max(record.bit_depths)
    if not np.issubdtype(array.dtype, np.unsignedinteger):
        return array
    if depth > bits:
        raise ParserContentError(
            f"This JPEG 2000 image has {depth} bits per component, which "
            f"Pillow decodes to {bits} bits, so it cannot be read without "
            f"losing precision."
        )
    shift = bits - depth
    signed = any(record.signed)
    if not shift and not signed:
        return array
    if not signed:
        return array >> shift
    out = (array.astype(np.int32) >> shift) - 2 ** (depth - 1)
    return out.astype(np.int8 if depth <= 8 else np.int16)
