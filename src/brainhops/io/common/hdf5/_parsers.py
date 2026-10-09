"""The HDF5 parser and writer mixins."""

# stdlib
from io import BytesIO
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import path
from brainhops._core.streams import preserve_position

# io
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
)

H5Like = tx.Union[tx.BinaryIO, PathLike, str, h5py.File]
"""Anything an HDF5 format can be read from or written to."""


def _raise_or(
    error: tx.Union[bool, tx.Type[Exception]],
    default: tx.Type[Exception],
    message: str,
    cause: tx.Optional[BaseException] = None,
) -> float:
    """Raise `error`, or `default` if `error` is true, or return NO."""
    if error:
        if error is True:
            error = default
        raise error(message) from cause
    return Confidence.NO


class Hdf5Reader(BinaryFileReader):
    """
    A mixin that reads a format stored in HDF5.

    A concrete format implements the class methods `sniff_h5` and
    `from_h5`, which take an open `h5py.File`, and the mixin routes paths,
    streams and bytes to them. A path is handed to h5py by name, so that a
    lazily read dataset can reopen the file long after parsing.
    """

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is of this format."""
        raise NotImplementedError

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> tx.Self:
        """Build an object from an open HDF5 file."""
        raise NotImplementedError

    @classmethod
    def sniff_file(
        cls,
        file: H5Like,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a path, an open HDF5 file or a binary stream."""
        if isinstance(file, h5py.File):
            return cls.sniff_h5(file, error=error)
        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, error=error, **kwargs)
        return cls.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_filename(
        cls,
        filename: tx.Union[str, PathLike],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the HDF5 file at a path."""
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            return _raise_or(
                error, SnifferExistsError, f"No such file: {filename}"
            )
        try:
            is_hdf5 = h5py.is_hdf5(str(filename))
        except Exception:  # noqa: BLE001  (remote paths, ...)
            is_hdf5 = None
        if is_hdf5 is False:
            return _raise_or(
                error, SnifferContentError, f"Not an HDF5 file: {filename}"
            )
        if is_hdf5 is None:
            with filename.open("rb") as f:
                return cls.sniff_fileobj(f, error=error, **kwargs)
        with h5py.File(str(filename), "r") as f:
            return cls.sniff_h5(f, error=error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open, seekable binary stream."""
        with preserve_position(file):
            try:
                with h5py.File(file, "r") as f:
                    return cls.sniff_h5(f, error=error)
            except Exception as e:
                return _raise_or(
                    error,
                    SnifferContentError,
                    f"Content is not a {cls.__name__} HDF5 file",
                    e,
                )

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of an HDF5 file."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    @classmethod
    def from_file(
        cls,
        file: H5Like,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """
        Build an object from a path, a binary stream or an open HDF5 file.

        With `load`, large datasets are read into memory; otherwise they stay
        on disk. With `keep_open`, the file stays open after loading; otherwise
        it is closed, and a lazy dataset reopens it on each access.
        """
        if isinstance(file, h5py.File):
            return cls.from_h5(file, keep_open=keep_open, load=load, **kwargs)
        if isinstance(file, (str, path.PathLike)):
            return cls.from_filename(
                file, keep_open=keep_open, load=load, **kwargs
            )
        return cls.from_fileobj(file, keep_open=keep_open, load=load, **kwargs)

    @classmethod
    def from_filename(
        cls,
        filename: tx.Union[str, PathLike],
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """Build an object from the HDF5 file at a path."""
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        f = h5py.File(str(filename), "r")
        try:
            return cls.from_h5(f, keep_open=keep_open, load=load, **kwargs)
        finally:
            if not keep_open:
                f.close()

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """Build an object from an open, seekable binary stream."""
        with preserve_position(file):
            f = h5py.File(file, "r")
            try:
                return cls.from_h5(f, keep_open=keep_open, load=load, **kwargs)
            finally:
                if not keep_open:
                    f.close()


class Hdf5ReaderWriter(Hdf5Reader, BinaryFileWriter):
    """
    A mixin that reads and writes a format stored in HDF5.

    The format also implements `to_h5`, which fills a new, empty HDF5 file.
    """

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the object into an empty HDF5 file open for writing."""
        raise NotImplementedError

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        """
        Return a function that fills an HDF5 file with the object.

        The function is obtained before the file is opened, so that a format
        can refuse an unwritable object without truncating the file.
        """
        return lambda h5file: self.to_h5(h5file, **kwargs)

    def to_file(self, file: H5Like, **kwargs) -> None:
        """Write to a path, a binary stream or an open HDF5 file."""
        if isinstance(file, h5py.File):
            return self._h5_writer(**kwargs)(file)
        return super().to_file(file, **kwargs)

    def to_filename(self, filename: tx.Union[str, PathLike], **kwargs) -> None:
        """Write to the file at a path, replacing it."""
        writer = self._h5_writer(**kwargs)
        with h5py.File(str(filename), "w") as f:
            writer(f)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write to a binary stream open for writing."""
        file.write(self.to_bytes(**kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of an HDF5 file that encodes the object."""
        writer = self._h5_writer(**kwargs)
        buffer = BytesIO()
        with h5py.File(buffer, "w") as f:
            writer(f)
        return buffer.getvalue()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def read_string(value: tx.Any) -> tx.Optional[str]:
    """
    Decode an HDF5 string, whatever its storage.

    The value may be `str`, `bytes`, a NumPy scalar, or a one-element array
    or dataset wrapping one of these. `None` is returned unchanged.

    Raises
    ------
    ValueError
        If the value is an array of more than one element.
    """
    if value is None:
        return None
    if isinstance(value, h5py.Dataset):
        value = value[()]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Not a scalar string: shape {value.shape}")
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode()
    return str(value)
