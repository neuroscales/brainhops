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
    BinaryFileParserWriter,
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


class NrrdParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by a NRRD file.

    It reads and writes the container -- the header and the sample
    values -- for every NRRD-based format. What the values mean is for
    the concrete format to say, through `_nrrd_header` and `_nrrd_data`
    when writing.

    Reading a `raw` file from a local path memory-maps the values, so
    nothing but the header is read until they are indexed.
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
        """The NRRD header this object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[NrrdHeader]) -> None:
        self._header = value

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: NrrdHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build the object from a decoded header and its stored values."""
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build the object from a NRRD file (path or file object)."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return cls.from_filename(file, **kwargs)
        return super().from_file(file, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> tx.Self:
        """
        Build the object from the path of a `.nrrd` or `.nhdr` file.

        The values of a `raw` local data file are memory-mapped unless
        `mmap` is false. Detached data files are found relative to the
        header's directory.
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
        """
        Build the object from an open NRRD file object.

        Detached data files are resolved against the stream's `name`,
        when it has one.
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
        """Build the object from the bytes of an attached NRRD file."""
        kwargs.pop("mmap", None)
        content = bytes(content)
        header, offset = NrrdHeader.from_fileobj(BytesIO(content))
        data = read_data(header, content, offset, mmap=False)
        return cls._from_header(header, data, **kwargs)

    # --- sniffing -----------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a stream holds a NRRD
        file of its own kind."""
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
        """Score how confident the class is that bytes hold a NRRD file of
        its own kind."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: NrrdHeader) -> float:
        """
        How well a valid NRRD header matches *this* class.

        Called once the header has been parsed, so the answer is never
        "not NRRD". A concrete format overrides it.
        """
        return Confidence.MAYBE

    # --- writing ------------------------------------------------------

    def _nrrd_header(self, **kwargs) -> NrrdHeader:
        """The header to write. Each concrete format builds its own."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def _nrrd_data(self, header: NrrdHeader) -> tx.Any:
        """The array, in the header's axis order, to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to NRRD."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """The bytes of an attached NRRD file."""
        kwargs.pop("data_file", None)
        header = self._nrrd_header(**kwargs)
        data = encode_data(header, self._nrrd_data(header))
        return header.to_text().encode("utf-8") + b"\n" + data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write an attached NRRD file to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(
        self,
        filename: path.FilenameLike,
        data_file: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Write to a path: an attached file, or, for a `.nhdr` name (or when
        `data_file` is given), a detached header and its data file.

        The data file of a detached header is named after it, with an
        extension that says its encoding (`.raw`, `.raw.gz`, `.raw.bz2`,
        `.txt`, `.hex`), unless `data_file` names it (relative to the
        header's directory).
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
        """Write to a path (variant chosen by extension) or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)
