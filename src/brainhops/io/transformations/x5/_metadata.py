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

__all__ = ["X5Metadata", "X5Raw"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr, replace

# internals
from brainhops.datamodel.metadata import (
    ALL,
    VOCABULARY,
    ConversionReport,
    FileBasedMetadata,
)
from brainhops.io.metadata._json import (
    decode_object,
    encode_changes,
    encode_extra,
)

# locals
from ._struct import X5Header, X5Node


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
        node: tx.Optional[X5Node] = None,
    ) -> None:
        """
        Parameters
        ----------
        header : X5Header, optional
            The root of the file. By default, an empty header.
        node : X5Node, optional
            The node of the transformation, or `None` for a chain of
            several nodes, whose own metadata stays in their nodes.
        """
        self.header = X5Header() if header is None else header
        self.node = node

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


class X5Metadata(FileBasedMetadata, on={"format": "x5"}, supports=ALL):
    """
    The metadata of an X5 transform node, stored in its JSON `Metadata`.

    `node` and `header` are the two halves of the raw record.
    """

    raw: tx.Annotated[
        tx.Optional[X5Raw],
        tx.Doc(
            """
            The root of the file and the node the transformation was
            read from (no node for a chain of several nodes, whose own
            metadata stays in their nodes).
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

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
    def _default_raw(cls) -> X5Raw:
        return X5Raw(X5Header(), X5Node())

    @classmethod
    def _decode(
        cls, raw: tx.Any, *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        node = None if raw is None else raw.node
        json = getattr(node, "metadata", None)
        if not isinstance(json, dict):
            return {}
        values, extra = decode_object(json, VOCABULARY)
        return {**values, "extra": extra or None}

    def _encode(
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
            json, {k: v for k, v in changed.items() if k != "extra"}
        )
        if "extra" in changed:
            encode_extra(json, changed["extra"], report=report)
        if not json and node.metadata is None:
            return raw
        return X5Raw(header, replace(node, metadata=json))
