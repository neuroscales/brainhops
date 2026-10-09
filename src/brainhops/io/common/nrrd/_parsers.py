"""The NRRD parser."""

# stdlib
import os
from io import BytesIO

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    preserve_position,
)

# this format
from ._constants import _DATA_EXTENSIONS
from ._data import (
    encode_data,
    read_data,
)
from ._header import NrrdHeader

# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class NrrdReaderWriter(DataModelBase, BinaryFileReader, BinaryFileWriter):
    """Base class for objects stored as NRRD files.

    Concrete formats give the values a meaning and provide [`_nrrd_header`][]
    and [`_nrrd_data`][] for writing.
    """

    HINTS = ("nrrd",)

    _header: tx.Annotated[
        tx.Optional[NrrdHeader],
        tx.Doc(
            """
            The NRRD header this object was read from, kept so that the
            fields and key/value pairs the data model has no slot for
            (`measurement frame`, `DWMRI_gradient_0000`, ...) are written
            back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, as an array of the file's axes (fastest
            first, F order). A view of a memory-mapped file when the file
            allows one.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[NrrdHeader]:
        """The header that the object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[NrrdHeader]) -> None:
        self._header = value

    @classmethod
    def _from_header(
        cls, header: NrrdHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build an object from a header and its values; formats override this
        hook.
        """
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Read an object from a path or a file object."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return cls.from_filename(file, **kwargs)
        return super().from_file(file, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> tx.Self:
        """Read an object from a `.nrrd` or `.nhdr` path.

        Raw local data are memory-mapped unless `mmap` is false.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with _open_path(filename) as f:
            header, offset = NrrdHeader.from_fileobj(f)
        data = read_data(header, filename, offset, mmap=mmap)
        return cls._from_header(header, data, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read an object from a file object, without memory mapping.

        Detached data files are resolved against the name of the stream.
        """
        kwargs.pop("mmap", None)
        name = getattr(file, "name", None)
        with preserve_position(file):
            content = file.read()
        header, offset = NrrdHeader.from_fileobj(BytesIO(content))
        if header.data_files and isinstance(name, (str, os.PathLike)):
            source: tx.Any = path.Path(os.fsdecode(name))
        else:
            source = content
        data = read_data(header, source, offset, mmap=False)
        return cls._from_header(header, data, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Read an object from the bytes of a NRRD file with attached data."""
        kwargs.pop("mmap", None)
        content = bytes(content)
        header, offset = NrrdHeader.from_fileobj(BytesIO(content))
        data = read_data(header, content, offset, mmap=False)
        return cls._from_header(header, data, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds this kind of NRRD file.

        The parsed header is scored with [`_score_header`][].
        """
        base_error = None
        score = Confidence.NO
        try:
            with preserve_position(file):
                header, _ = NrrdHeader.from_fileobj(file)
                score = cls._score_header(header)
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Content is not a {cls.__name__}") from base_error
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold this kind of NRRD file."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: NrrdHeader) -> float:
        """Return how well a valid header matches this class.

        Concrete formats override this hook; the base implementation returns
        `Confidence.MAYBE`.
        """
        return Confidence.MAYBE

    def _nrrd_header(self, **kwargs) -> NrrdHeader:
        """Return the header to write; the base implementation raises
        `WriterError`.
        """
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def _nrrd_data(self, header: NrrdHeader) -> tx.Any:
        """Return the array to write; the base implementation raises
        `WriterError`.
        """
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a NRRD file with attached data."""
        kwargs.pop("data_file", None)
        header = self._nrrd_header(**kwargs)
        data = encode_data(header, self._nrrd_data(header))
        return header.to_text().encode("utf-8") + b"\n" + data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a NRRD file with attached data to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(
        self,
        filename: path.FilenameLike,
        data_file: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write the object to a path.

        The data are written to a separate file when the name ends with `.nhdr`
        or `data_file` is given.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        name = filename.name
        if not name.lower().endswith(".nhdr") and data_file is None:
            with filename.open("wb") as f:
                f.write(self.to_bytes(**kwargs))
            return
        header = self._nrrd_header(**kwargs)
        data = encode_data(header, self._nrrd_data(header))
        if data_file is None:
            base = (
                name[: -len(".nhdr")]
                if name.lower().endswith(".nhdr")
                else name
            )
            data_file = base + _DATA_EXTENSIONS[header.encoding]
        with _sibling(filename, data_file).open("wb") as f:
            f.write(data)
        header = NrrdHeader(
            version=header.version,
            fields=header.fields,
            keyvalue=header.keyvalue,
            data_files=(data_file,),
        )
        with filename.open("wb") as f:
            f.write(header.to_text().encode("utf-8"))

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the object to a path or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)
