"""The MRtrix image parser."""

# stdlib
import gzip
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
    preserve_position,
)

# this format
from ._data import (
    _data_location,
    _is_gzip,
    _read_buffer,
    decode_data,
    encode_data,
)
from ._header import MrtrixHeader

# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class MrtrixParser(DataModelBase, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by an MRtrix image file.

    It reads and writes the container -- the header and the raw voxel
    values -- for every MRtrix-based format: an image, and later a warp.
    What the values mean is for the concrete format to say, through
    `_mrtrix_header` and `_mrtrix_data` when writing.

    Reading a `.mif` or a `.mih` from a local path memory-maps the data,
    so nothing but the header is read until the data are indexed. A
    `.mif.gz`, a file object or bytes are read into memory.
    """

    HINTS = ("mrtrix",)

    _header: tx.Annotated[
        tx.Optional[MrtrixHeader],
        tx.Doc(
            """
            The MRtrix header this object was read from, kept so that the
            keys the data model has no slot for (`dw_scheme`,
            `command_history`, ...) are written back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, as an array of the header's axes, before
            intensity scaling. A view of a memory-mapped file when the file
            allows one.
            """
        ),
    ] = None

    @property
    def header(self) -> tx.Optional[MrtrixHeader]:
        """The MRtrix header this object was read from, if any."""
        return getattr(self, "_header", None)

    @header.setter
    def header(self, value: tx.Optional[MrtrixHeader]) -> None:
        self._header = value

    def _scaled_data(self) -> tx.Optional[tx.Any]:
        """
        The stored values with the header's intensity scaling applied.

        Without scaling (or with the identity one), the stored values are
        returned as they are, so a memory map stays one. With scaling,
        the values are read and scaled, to at least single precision.
        """
        raw = getattr(self, "dataobj", None)
        if raw is None:
            return None
        header = self.header
        scaling = getattr(header, "scaling", None)
        if not scaling or tuple(scaling) == (0.0, 1.0):
            return raw
        offset, scale = scaling
        dtype = np.result_type(np.asarray(raw).dtype, np.float32)
        return offset + scale * np.asarray(raw, dtype=dtype)

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: MrtrixHeader, dataobj: tx.Any, **kwargs
    ) -> tx.Self:
        """Build the object from a decoded header and its stored values."""
        return cls(header=header, dataobj=dataobj, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build the object from an MRtrix file (path or file object)."""
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
        Build the object from the path of a `.mif`, `.mih` or `.mif.gz`.

        The data of an uncompressed local file are memory-mapped unless
        `mmap` is false. A `.mih` header is followed to the data file it
        names, relative to the header's directory.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not filename.exists():
            raise ParserExistsError(f"No such file: {filename}")

        if _is_gzip(filename):
            with _open_path(filename) as f:
                return cls.from_fileobj(f, **kwargs)

        with _open_path(filename) as f:
            header, _ = MrtrixHeader.from_fileobj(f)
        name, offset = _data_location(header)
        if name == ".":
            datafile = filename
        else:
            datafile = _sibling(filename, name)
            if not datafile.exists():
                raise ParserExistsError(
                    f"No such file: {datafile} (the data file named by the "
                    f"MRtrix header {filename})"
                )
        buffer = _read_buffer(datafile, offset, header.nbytes, mmap)
        return cls._from_header(header, decode_data(header, buffer), **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build the object from an open MRtrix file object, gzipped or not.

        The data of a single-file image are read from the stream. A
        header that names a separate data file is resolved against the
        stream's `name`, when it has one.
        """
        kwargs.pop("mmap", None)
        with preserve_position(file):
            stream = open_compressed(file)
            content = stream.read()
        return cls._from_content(
            content, origin=getattr(file, "name", None), **kwargs
        )

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the object from the bytes of a single-file MRtrix image
        (gzipped or not)."""
        kwargs.pop("mmap", None)
        content = bytes(content)
        if content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)
        return cls._from_content(content, origin=None, **kwargs)

    @classmethod
    def _from_content(
        cls, content: bytes, origin: tx.Any = None, **kwargs
    ) -> tx.Self:
        header, _ = MrtrixHeader.from_fileobj(BytesIO(content))
        name, offset = _data_location(header)
        if name == ".":
            buffer = memoryview(content)[offset : offset + header.nbytes]
        else:
            if not isinstance(origin, (str, bytes, os.PathLike)):
                raise ParserContentError(
                    f"The MRtrix header names a separate data file "
                    f"({name!r}), which cannot be found from a stream that "
                    f"has no file name. Read it from its path instead."
                )
            datafile = _sibling(path.Path(os.fsdecode(origin)), name)
            if not datafile.exists():
                raise ParserExistsError(f"No such file: {datafile}")
            buffer = _read_buffer(datafile, offset, header.nbytes, False)
        return cls._from_header(header, decode_data(header, buffer), **kwargs)

    # --- sniffing -----------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a stream holds an MRtrix
        image, gzipped or not."""
        score = Confidence.NO
        base_error = None
        try:
            with preserve_position(file):
                header, _ = MrtrixHeader.from_fileobj(open_compressed(file))
            score = cls._score_header(header)
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Content is not an MRtrix image") from base_error
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold an MRtrix
        image, gzipped or not."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        How well a valid MRtrix header matches *this* class.

        Called once the header has been parsed, so the answer is never
        "not MRtrix". A concrete format overrides it to tell its own kind
        of MRtrix file (an image, a warp) from the others.
        """
        return Confidence.MAYBE

    # --- writing ------------------------------------------------------

    def _mrtrix_header(self, **kwargs) -> MrtrixHeader:
        """The header to write. Each concrete format builds its own."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def _mrtrix_data(self) -> tx.Any:
        """The array of the header's axes to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to "
            f"MRtrix."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """The bytes of a single-file, uncompressed `.mif`."""
        header = self._mrtrix_header(**kwargs)
        data = encode_data(header, self._mrtrix_data_for(header))
        return header.embedded() + data

    def _mrtrix_data_for(self, header: MrtrixHeader) -> tx.Any:
        """The data to write, mapped through the header's scaling."""
        data = self._mrtrix_data()
        if header.scaling and tuple(header.scaling) != (0.0, 1.0):
            offset, scale = header.scaling
            data = (np.asarray(data) - offset) / scale
        return data

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a single-file, uncompressed `.mif` to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write to a path, in the variant its extension names.

        * `.mif`: header and data in one file;
        * `.mif.gz`: the same, gzip-compressed;
        * `.mih`: the header, and the data in a `.dat` file next to it
          (named after the header), which the header points to.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        name = str(filename)
        lower = name.lower()
        if lower.endswith(".mih"):
            header = self._mrtrix_header(**kwargs)
            data = encode_data(header, self._mrtrix_data_for(header))
            base = filename.name[: -len(".mih")]
            datafile = _sibling(filename, base + ".dat")
            with datafile.open("wb") as f:
                f.write(data)
            with filename.open("wb") as f:
                text = header.to_text(file=(base + ".dat", 0))
                f.write(text.encode("utf-8"))
            return
        content = self.to_bytes(**kwargs)
        if lower.endswith(".gz"):
            content = gzip.compress(content)
        with filename.open("wb") as f:
            f.write(content)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path (variant chosen by extension) or a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)
