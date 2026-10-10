"""The metadata of an AFNI dataset."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, BinaryFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._files import afni_dataset_files
from ._raw import AfniRaw


@register_format
class AfniMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of an AFNI dataset, which is its `.HEAD` file.

    The metadata of an AFNI dataset is the whole header, held as an
    [`AfniRaw`][] record. [`AfniImage`][brainhops.io.images.afni.AfniImage]
    uses that record as the base of the header that it writes. Reading
    the metadata reads the `.HEAD` file alone, whichever file of the
    dataset a path names, so the `.BRIK` file is neither read nor needed.
    The metadata keeps no open file.

    A dataset is read through the record: [`from_fileobj`][] and
    [`from_filename`][] read an [`AfniRaw`][], check that it describes a
    dataset, and wrap it. Writing the metadata writes the `.HEAD` file of
    the dataset that a path names.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".HEAD",
        ".BRIK",
        ".BRIK.gz",
        ".BRIK.bz2",
        ".head",
        ".brik",
        ".brik.gz",
        ".brik.bz2",
    )
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("afni", "brik")

    raw: NoRepr[tx.Optional[AfniRaw]] = None
    """The record of the dataset, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[AfniRaw]:
        """Return a copy of the record that the caller is free to change.

        Returns
        -------
        AfniRaw or None
            A copy of `raw`, or `None` when no record is held.
        """
        return None if self.raw is None else self.raw.copy()

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an AFNI header.

        The stream is tested with [`AfniRaw.sniff_fileobj`][].
        """
        return AfniRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names an AFNI dataset.

        The path is tested with [`AfniRaw.sniff_filename`][], which reads
        the `.HEAD` file of the dataset.
        """
        return AfniRaw.sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an AFNI header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "AfniMetadata":
        """Read the metadata from an open `.HEAD` or `.BRIK` file.

        The record is read with [`AfniRaw.from_fileobj`][] and validated.

        Raises
        ------
        ParserContentError
            If the stream holds no valid header, or is a BRIK without a
            file name.
        """
        return cls.from_raw(AfniRaw.from_fileobj(file, **kwargs).validate())

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "AfniMetadata":
        """Read the metadata from the path of a dataset.

        The record is read from the `.HEAD` file with
        [`AfniRaw.from_filename`][] and validated.

        Raises
        ------
        ParserExistsError
            If the `.HEAD` file does not exist.
        ParserContentError
            If the header is invalid.
        """
        record = AfniRaw.from_filename(filename, **kwargs)
        return cls.from_raw(record.validate())

    def to_bytes(self, **kwargs) -> bytes:
        """Return the content of the `.HEAD` file.

        Metadata without a record is written as an empty header.
        """
        if self.raw is None:
            return b""
        return self.raw.to_bytes(**kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the `.HEAD` file of the dataset that a path names.

        The path may name the `.HEAD` file, the `.BRIK` file or the bare
        dataset, and no BRIK is written.
        """
        head, _, _ = afni_dataset_files(filename)
        with head.open("wb") as file:
            file.write(self.to_bytes(**kwargs))
