"""The AFNI format base and dataset parser."""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
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
from ._constants import _BRIK_SUFFIXES
from ._data import (
    _brik_suffix,
    _read_brik,
    _write_brik,
    afni_dataset_files,
    scale_bricks,
)
from ._header import (
    AfniHeader,
    _looks_like_head,
    _read_head,
    _read_header_stream,
)

# ----------------------------------------------------------------------
#   FORMAT FAMILY
# ----------------------------------------------------------------------


class AfniFormat:
    """
    A format of the AFNI family, whatever it stores.

    It is the shared base of the AFNI image formats (BRIK/HEAD datasets)
    and transformation formats, and carries the `"afni"` hint they all
    answer to. Each format adds its own hints (`"brik"`, ...), which are
    then also reachable as `"afni.brik"`, ...
    """

    HINTS = ("afni",)


# ----------------------------------------------------------------------
#   PARSER
# ----------------------------------------------------------------------


class AfniParser(DataModelBase, AfniFormat, BinaryFileParserWriter):
    """
    Base class for objects that are encoded by an AFNI dataset
    (`.HEAD` + `.BRIK`).

    It reads and writes the container -- the header and the sub-bricks --
    for every AFNI dataset-based format: an image, and later a warp. What
    the values mean is for the concrete format to say, through
    `_from_header` when reading, and `_afni_header` and `_afni_data` when
    writing.

    Reading from a local path memory-maps an uncompressed BRIK, so
    nothing but the header is read until the data are indexed. A
    compressed BRIK is read into memory.
    """

    HINTS = ("brik",)

    _header: tx.Annotated[
        tx.Optional[AfniHeader],
        tx.Doc(
            """
            The AFNI header this object was read from, kept so that the
            attributes the data model has no slot for (`HISTORY_NOTE`,
            `BRICK_LABS`, `BRICK_STATAUX`, ...) are written back.
            """
        ),
    ] = None

    dataobj: tx.Annotated[
        tx.Optional[tx.Any],
        tx.Doc(
            """
            The stored values, before scaling by `BRICK_FLOAT_FACS`: a
            `(x, y, z, sub-brick)` array, a view of a memory-mapped BRIK
            when the file allows one.
            """
        ),
    ] = None

    header = smartproperty("header")

    def _scaled_data(self) -> tx.Optional[tx.Any]:
        """The stored values, scaled by `BRICK_FLOAT_FACS`."""
        raw = getattr(self, "dataobj", None)
        if raw is None or self.header is None:
            return raw
        return scale_bricks(self.header, raw)

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: AfniHeader, bricks: np.ndarray, **kwargs
    ) -> tx.Self:
        """Build the object from a header and its `(x, y, z, sub-brick)`
        stored values."""
        return cls(header=header, dataobj=bricks, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Build the object from an AFNI dataset (path or file object)."""
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
        Build the object from the path of a `.HEAD`, of a `.BRIK`
        (`.BRIK.gz`, `.BRIK.bz2`), or of the dataset without extension.

        An uncompressed local BRIK is memory-mapped unless `mmap` is
        false.
        """
        head, brik, _ = afni_dataset_files(filename)
        if not path.exists(head):
            raise ParserExistsError(f"No such file: {head}")
        header = _read_head(head).validate()
        bricks = _read_brik(header, brik, mmap)
        return cls._from_header(header, bricks, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """
        Build the object from an open `.HEAD` (or `.BRIK`) file object.

        The other file of the dataset is found from the stream's `name`,
        which it must therefore have.
        """
        kwargs.pop("mmap", None)
        name = getattr(file, "name", None)
        with preserve_position(file):
            content = file.read()
        if isinstance(content, str):
            content = content.encode("latin-1")
        if _looks_like_head(bytes(content[:256]).decode("latin-1")):
            header = AfniHeader.from_bytes(content).validate()
            if not isinstance(name, (str, bytes, os.PathLike)):
                raise ParserContentError(
                    "An AFNI header was given as a stream with no file "
                    "name, so its BRIK cannot be found. Read it from its "
                    "path instead."
                )
            _, brik, _ = afni_dataset_files(os.fsdecode(name))
            bricks = _read_brik(header, brik, mmap=False)
            return cls._from_header(header, bricks, **kwargs)
        if isinstance(name, (str, bytes, os.PathLike)):
            return cls.from_filename(os.fsdecode(name), mmap=False, **kwargs)
        raise ParserContentError(
            "This stream is not an AFNI header, and has no file name to "
            "find one from."
        )

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """An AFNI dataset is two files, so bytes alone cannot hold one."""
        raise ParserContentError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Read it from its path instead."
        )

    # --- sniffing -----------------------------------------------------

    @classmethod
    def _sniff_header(
        cls,
        read: tx.Callable[[], AfniHeader],
        error: tx.Union[bool, tx.Type[Exception]],
    ) -> float:
        base_error = None
        try:
            score = cls._score_header(read().validate())
        except Exception as e:  # noqa: BLE001
            base_error = e
            score = Confidence.NO
        if score:
            return score
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Content is not an AFNI dataset") from base_error
        return Confidence.NO

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a path names an AFNI
        dataset: its `.HEAD` is read, whichever of its files is named."""
        head, _, _ = afni_dataset_files(filename)

        def _read() -> AfniHeader:
            if not path.exists(head):
                raise ParserExistsError(f"No such file: {head}")
            with _open_path(head) as f:
                return _read_header_stream(f)

        return cls._sniff_header(_read, error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that a stream holds an AFNI
        header."""

        def _read() -> AfniHeader:
            with preserve_position(file):
                return _read_header_stream(file)

        return cls._sniff_header(_read, error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold an AFNI
        header."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _score_header(cls, header: AfniHeader) -> float:
        """
        How well a valid AFNI header matches *this* class.

        Called once the header has been parsed and validated, so the
        answer is never "not AFNI". A concrete format overrides it to tell
        its own kind of dataset (an image, a warp) from the others.
        """
        return Confidence.MAYBE

    # --- writing ------------------------------------------------------

    def _afni_header(self, **kwargs) -> AfniHeader:
        """The header to write. Each concrete format builds its own."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to AFNI."
        )

    def _afni_data(self) -> tx.Any:
        """The `(x, y, z[, sub-brick])` array to write."""
        raise WriterError(
            f"{type(self).__name__} does not know how to write itself to AFNI."
        )

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the dataset: its `.HEAD` and its `.BRIK`.

        The path may name the `.HEAD`, the `.BRIK` (`.BRIK.gz` or
        `.BRIK.bz2` to compress it), or the dataset without extension
        (`out+orig`). Another BRIK of the same dataset, compressed
        differently, is removed (as AFNI does), so that it cannot be read
        in place of the new one.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        head, _, stem = afni_dataset_files(filename)
        brik_ext = ".brik" if head.name.endswith(".head") else ".BRIK"
        suffix = ""
        if ".BRIK" in filename.name.upper():
            suffix = _brik_suffix(filename)
        brik = filename.parent / (stem + brik_ext + suffix)
        header = self._afni_header(filename=filename, **kwargs)
        _write_brik(header, self._afni_data(), brik)
        with head.open("wb") as f:
            f.write(header.to_text().encode("latin-1"))
        for other in _BRIK_SUFFIXES:
            stale = filename.parent / (stem + brik_ext + other)
            if other != suffix and path.exists(stale):
                local = _local_path(stale)
                if local is not None:
                    os.remove(local)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path; an AFNI dataset cannot be written to a
        stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return self.to_fileobj(file, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """An AFNI dataset is two files, which a stream cannot hold."""
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which a "
            "single stream cannot hold. Write it to a path instead."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """An AFNI dataset is two files, which bytes cannot hold."""
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Write it to a path instead."
        )
