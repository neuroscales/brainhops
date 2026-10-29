"""The metadata of an N5 dataset or pyramid."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base._internal_api import checked_score, no_options
from brainhops.io.base.parsers import Confidence
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._data import read_node
from ._raw import DECLINE_MESSAGE, OPTION_OWNER, N5Raw


@register_format
class N5Metadata(MetadataReader, MetadataFormat):
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
    RAW: tx.ClassVar[type] = N5Raw

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
            return checked_score(Confidence.NO, error, DECLINE_MESSAGE)
        return Confidence.CERTAIN

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
        no_options(kwargs, OPTION_OWNER)
        _, record = read_node(filename)
        return cls.from_raw(record)
