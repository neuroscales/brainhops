"""The metadata of an N5 dataset or pyramid."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, Confidence
from brainhops.io.metadata import MetadataFormat

# this format
from ._data import read_node
from ._raw import N5Raw, no_options, score


@register_format
class N5Metadata(MetadataFormat, BinaryFileReader):
    """Metadata of an N5 dataset or pyramid, which is its attributes.

    The metadata of an N5 dataset is the content of its `attributes.json`
    file, held as an [`N5Raw`][] record. The metadata of a pyramid holds
    the attributes of its group and of each of its datasets. It is read
    from the directory of the node, from its `attributes.json` file, or
    from the content of that file, and reading it never reads a block.
    The metadata keeps no open file and no location. N5 containers are
    not written, so the metadata cannot be written either.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("n5",)

    raw: NoRepr[tx.Optional[N5Raw]] = None
    """The record of the node, or `None` for metadata without a record."""

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a name designates an N5 node.

        The name is the directory or the `attributes.json` file of a
        dataset or of a pyramid, and the attributes are read.

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of a node.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no dataset
            and no pyramid.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        try:
            read_node(filename)
        except Exception:  # noqa: BLE001
            return score(Confidence.NO, error)
        return Confidence.CERTAIN

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds dataset attributes.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no attributes
            of an N5 dataset.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        return N5Raw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold dataset attributes.

        Parameters
        ----------
        content : bytes
            The content of an `attributes.json` file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no attributes
            of an N5 dataset.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "N5Metadata":
        """Read the metadata of an N5 node from its directory or attributes.

        The attributes of a pyramid are read with those of its datasets
        `s0`, `s1`, ...

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of a dataset or of
            a pyramid.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        N5Metadata
            The metadata, which holds the record of the node.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If there is no node at that name.
        ParserContentError
            If the node is neither a dataset nor a pyramid, or its
            attributes break the specification.
        """
        no_options(kwargs)
        _, record = read_node(filename)
        return cls.from_raw(record)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "N5Metadata":
        """Read the metadata of an N5 node from a stream of its attributes.

        The record has no levels, since the datasets of a pyramid are
        other files.

        Parameters
        ----------
        file : BinaryIO
            The stream of the `attributes.json` file, open in binary mode.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        N5Metadata
            The metadata, which holds the record of the attributes.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream does not hold valid attributes.
        """
        return cls.from_raw(N5Raw.from_fileobj(file, **kwargs))
