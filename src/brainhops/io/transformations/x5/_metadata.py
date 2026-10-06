"""
The metadata of X5 files: [`X5Metadata`][].

Its raw record (`raw`) is an `X5Raw`, which holds the root of the file
(`X5Header`) and the node the transformation was read from (`X5Node`).
Every vocabulary field is stored in the node's JSON `Metadata`
attribute, under its BIDS sidecar key (`Description`, `GeneratedBy`,
...) or, when BIDS has none, under its name in `CamelCase` (`History`,
`Moving`, `Fixed`, `InputSpace`, `OutputSpace`). The other keys of the
JSON object are `extra`. A JSON object stores anything, so nothing is
unsupported.

`Domain.Coordinates` is not decoded: nitransforms writes the kind of
coordinates there (`"cartesian"`), not the label of a space.
"""

__all__ = ["X5Metadata", "X5Raw", "metadata_index"]

# dependencies
import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel.metadata import ConversionReport

# internals
from brainhops.datamodel.metadata._sentinel import ALL
from brainhops.datamodel.metadata._vocabulary import VOCABULARY
from brainhops.io.base._base import register_format
from brainhops.io.base.hdf5 import Hdf5MetadataParser
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.metadata import FileBasedMetadata
from brainhops.io.metadata._json import (
    decode_object,
    encode_changes,
    encode_extra,
)

# locals
from ._struct import X5Header, X5Node, is_x5, read_x5

# The default of `X5Raw(node=...)`: a new, empty node. `None` means a
# chain of several nodes, so it cannot be the default. Above `X5Raw`,
# whose signature evaluates it.
_NEW = object()


class X5Raw:
    """
    The raw record of an X5 transformation: the root of the file and the
    node that the transformation was read from.

    A class of its own, rather than a pair, so that the type of the
    record names its format: a conversion gives a record back to the
    format whose class declares its type, and no other format declares
    this one.
    """

    __slots__ = ("header", "node")

    def __init__(
        self,
        header: tx.Optional[X5Header] = None,
        node: tx.Any = _NEW,
    ) -> None:
        """
        Parameters
        ----------
        header : X5Header, optional
            The root of the file. By default, an empty header.
        node : X5Node or None, optional
            The node of the transformation, or `None` for a chain of
            several nodes, whose own metadata stays in their nodes. By
            default, an empty node.
        """
        self.header = X5Header() if header is None else header
        self.node = X5Node() if node is _NEW else node

    def is_record_of(self, header: tx.Any, node: tx.Any) -> bool:
        """
        Whether this record holds exactly the given header and node.

        Parameters
        ----------
        header : X5Header
            A root of a file.
        node : X5Node or None
            A node.

        Returns
        -------
        bool
            Whether both are the objects this record holds.
        """
        return self.header is header and self.node is node

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, X5Raw):
            return NotImplemented
        return self.header == other.header and self.node == other.node

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"X5Raw(header=..., node={self.node is not None})"


@register_format
class X5Metadata(
    Hdf5MetadataParser,
    FileBasedMetadata[X5Raw],
    on={"format": "x5"},
    supports=ALL,
):
    """
    The metadata of an X5 transform node, stored in its JSON `Metadata`.

    Its raw record (`raw`) is an `X5Raw`: the root of the file and the
    node the transformation was read from (no node for a chain of
    several nodes, whose own metadata stays in their nodes). `node` and
    `header` are its two halves.
    `X5Metadata.load(path)` reads the metadata of the transformation that
    `X5Transform.load(path)` would read (`chain=` and `position=` select
    another one), without reading its arrays.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".x5",)
    HINTS = ("x5",)

    # --- reading the node of a file -----------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: tx.Any,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """
        Score how confident the class is that an open HDF5 file is an X5
        file.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type, optional
            Raise an error (this one, or `SnifferContentError` for `True`)
            instead of returning 0.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        if is_x5(h5file):
            return Confidence.CERTAIN
        if error:
            raise (SnifferContentError if error is True else error)(
                "HDF5 file is not an X5 file: no Format='X5'."
            )
        return Confidence.NO

    @classmethod
    def from_h5(
        cls,
        h5file: tx.Any,
        chain: tx.Optional[int] = None,
        position: tx.Optional[int] = None,
        **kwargs: tx.Any,
    ) -> tx.Self:
        """
        Read the metadata of the transformation of an open X5 file that
        `X5Transform` would read, without its arrays.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        chain : int, optional
            The chain of `/TransformChain` to read.
        position : int, optional
            The single transform of `/TransformGroup` to read.
        **kwargs
            Ignored.

        Returns
        -------
        X5Metadata
            The metadata of the node, with the root and the node as `raw`.

        Raises
        ------
        ParserContentError
            If the file has no such transform.
        """
        header, nodes = read_x5(h5file, load=False)
        try:
            index = metadata_index(header, chain, position)
            node = None if index is None else nodes[index]
        except IndexError:
            raise ParserContentError(
                f"This X5 file has no transform for chain={chain}, "
                f"position={position}."
            ) from None
        return cls.from_raw(X5Raw(header, node))

    @property
    def header(self) -> tx.Optional[X5Header]:
        """The root of the file (first half of the raw record)."""
        return None if self.raw is None else self.raw.header

    @property
    def node(self) -> tx.Optional[X5Node]:
        """The node (second half of the raw record)."""
        return None if self.raw is None else self.raw.node

    # --- hooks --------------------------------------------------------

    @classmethod
    def _decode_raw(
        cls, raw: tx.Any, *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        node = None if raw is None else raw.node
        json = getattr(node, "metadata", None)
        if not isinstance(json, dict):
            return {}
        values, extra = decode_object(json, VOCABULARY)
        return {**values, "extra": extra or None}

    def _encode_raw(
        self,
        raw: X5Raw,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> X5Raw:
        header, node = raw.header, raw.node
        if not changed:
            return raw
        if node is None:
            # A chain of several nodes: no node to write into.
            for name, value in changed.items():
                report.lost[name] = value
            return raw
        json = node.metadata
        if isinstance(json, str) and json:
            report.approximated["extra"] = (
                "the node's Metadata was not JSON, and is replaced"
            )
        json = dict(json) if isinstance(json, dict) else {}
        encode_changes(
            json,
            {k: v for k, v in changed.items() if k != "extra"},
            report=report,
        )
        if "extra" in changed:
            encode_extra(json, changed["extra"], report=report)
        if not json and node.metadata is None:
            return raw
        return X5Raw(header, replace(node, metadata=json))


def metadata_index(
    header: X5Header,
    chain: tx.Optional[int] = None,
    position: tx.Optional[int] = None,
) -> tx.Optional[int]:
    """
    The node whose metadata is the metadata of a transformation.

    Parameters
    ----------
    header : X5Header
        The root of the file.
    chain : int, optional
        The chain of `/TransformChain` the transformation is.
    position : int, optional
        The single transform of `/TransformGroup` the transformation is.

    Returns
    -------
    int or None
        The index of the node: the single node read, or `None` for a
        chain of several nodes, which has no metadata of its own
        (composition does not merge).
    """
    if position is not None:
        return int(position)
    if chain is not None:
        nodes = header.chains[chain]
    elif header.chains:
        nodes = header.chains[0]
    else:
        return 0
    return nodes[0] if len(nodes) == 1 else None
