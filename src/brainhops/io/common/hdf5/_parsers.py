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

# io
from brainhops._core._internal_api import preserve_position, replace_file
from brainhops.datamodel.metadata._internal_api import takes_error_policy
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
        """Return the confidence that an open HDF5 file is of this format.

        A concrete format implements this method.

        Parameters
        ----------
        h5file : h5py.File
            The file, open for reading.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not of this
            format. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.

        Returns
        -------
        float
            The confidence that the content is of this format.

        Raises
        ------
        NotImplementedError
            If the format does not implement it.
        """
        raise NotImplementedError

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> tx.Self:
        """Build an object from an open HDF5 file.

        A concrete format implements this method.

        Parameters
        ----------
        h5file : h5py.File
            The file, open for reading.
        **kwargs : Any
            The options of the format.

        Returns
        -------
        Self
            The object.

        Raises
        ------
        NotImplementedError
            If the format does not implement it.
        """
        raise NotImplementedError

    @classmethod
    def sniff_file(
        cls,
        file: H5Like,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a path, an open HDF5 file or a binary stream.

        Parameters
        ----------
        file : path-like, h5py.File or file object
            The content to test.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not of this
            format. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Options passed to [`sniff_filename`][] or
            [`sniff_fileobj`][].

        Returns
        -------
        float
            The confidence that the content is of this format.

        Raises
        ------
        SnifferError
            If the content is not of this format and `error` is `True`.
        """
        if isinstance(file, h5py.File):
            return cls.sniff_h5(file, error=error)
        file = path.as_path_or_stream(file)
        if isinstance(file, path.Path):
            return cls.sniff_filename(file, error=error, **kwargs)
        return cls.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_filename(
        cls,
        filename: tx.Union[str, PathLike],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the HDF5 file at a path.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not of this
            format. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Options passed to [`sniff_fileobj`][] when h5py cannot open
            the path by name.

        Returns
        -------
        float
            The confidence that the content is of this format.

        Raises
        ------
        SnifferExistsError
            If the path does not exist and `error` is `True`.
        SnifferContentError
            If the file is not of this format and `error` is `True`.
        """
        filename = path.Path(filename)
        if not filename.exists():
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
        """Score an open, seekable binary stream.

        The position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not of this
            format. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence that the content is of this format.

        Raises
        ------
        SnifferContentError
            If the stream is not of this format and `error` is `True`.
        """
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
    def from_file(cls, file: H5Like, **kwargs) -> tx.Self:
        """
        Build an object from a path, a binary stream or an open HDF5 file.

        The keyword arguments are the options of the format, which
        `from_h5` receives. A file that this method opens is closed once
        it is read, so a format that reads a dataset later opens the file
        again by its name.

        Parameters
        ----------
        file : path-like, h5py.File or file object
            The content to read.
        **kwargs : Any
            The options of the format.

        Returns
        -------
        Self
            The object.
        """
        if isinstance(file, h5py.File):
            return cls.from_h5(file, **kwargs)
        file = path.as_path_or_stream(file)
        if isinstance(file, path.Path):
            return cls.from_filename(file, **kwargs)
        return cls.from_fileobj(file, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: tx.Union[str, PathLike], **kwargs
    ) -> tx.Self:
        """Build an object from the HDF5 file at a path.

        The file is opened by name, so that a dataset that is read later
        can open the file again by the same name.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of the format.

        Returns
        -------
        Self
            The object.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        filename = path.Path(filename)
        if not filename.exists():
            raise ParserExistsError(f"No such file: {filename}")
        # h5py opens a file by its name, given as text.
        with h5py.File(str(filename), "r") as f:
            return cls.from_h5(f, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Build an object from an open, seekable binary stream.

        The position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        **kwargs : Any
            The options of the format.

        Returns
        -------
        Self
            The object.
        """
        with preserve_position(file), h5py.File(file, "r") as f:
            return cls.from_h5(f, **kwargs)


class Hdf5ReaderWriter(Hdf5Reader, BinaryFileWriter):
    """
    A mixin that reads and writes a format stored in HDF5.

    The format also implements `to_h5`, which fills a new, empty HDF5 file.
    """

    @takes_error_policy
    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the object into an empty HDF5 file open for writing.

        A concrete format implements this method.

        Parameters
        ----------
        h5file : h5py.File
            The file, empty and open for writing.
        **kwargs : Any
            The options of the writer.

        Raises
        ------
        NotImplementedError
            If the format does not implement it.
        """
        raise NotImplementedError

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        """
        Return a function that fills an HDF5 file with the object.

        The function is obtained before the file is opened, so that a format
        can refuse an unwritable object without truncating the file.
        """
        return lambda h5file: self.to_h5(h5file, **kwargs)

    @takes_error_policy
    def to_file(self, file: H5Like, **kwargs) -> None:
        """Write to a path, a binary stream or an open HDF5 file.

        Parameters
        ----------
        file : path-like, h5py.File or file object
            Where to write.
        **kwargs : Any
            The options of the writer.
        """
        if isinstance(file, h5py.File):
            return self._h5_writer(**kwargs)(file)
        return super().to_file(file, **kwargs)

    @takes_error_policy
    def to_filename(self, filename: tx.Union[str, PathLike], **kwargs) -> None:
        """Write to the file at a path, replacing it.

        A local file is written beside its target, which it then replaces
        at once. A lazy array that reads the old file, such as a field of
        an object that was loaded from that same file, therefore reads
        the old content while the new file is written, and an error
        while the content is written leaves the target as it was. Other
        paths are written in place.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of the writer.
        """
        writer = self._h5_writer(**kwargs)

        def write(name: str) -> None:
            with h5py.File(name, "w-") as f:
                writer(f)

        if path.Path(filename).is_local():
            replace_file(filename, write)
        else:
            with h5py.File(str(filename), "w") as f:
                writer(f)

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of an HDF5 file that encodes the object.

        Parameters
        ----------
        **kwargs : Any
            The options of the writer.

        Returns
        -------
        bytes
            The content of the file.
        """
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

    Parameters
    ----------
    value : Any
        The stored value.

    Returns
    -------
    str or None
        The decoded string.

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
