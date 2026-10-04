"""
The metadata of X5 files: [`X5Metadata`][].

Its raw record (`raw`) is the pair `(X5Header, X5Node)`: the root of the
file and the node the transformation was read from. Every vocabulary
field is stored in the node's JSON `Metadata` attribute, under its BIDS
sidecar key (`Description`, `GeneratedBy`, ...) or, when BIDS has none,
under its name in `CamelCase` (`History`, `Moving`, `Fixed`,
`InputSpace`, `OutputSpace`). The other keys of the JSON object are
`extra`. A JSON object stores anything, so nothing is unsupported.

`Domain.Coordinates` is not decoded: nitransforms writes the kind of
coordinates there (`"cartesian"`), not the label of a space.
"""

__all__ = ["X5Metadata"]

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


class X5Metadata(FileBasedMetadata, on={"format": "x5"}, supports=ALL):
    """
    The metadata of an X5 transform node, stored in its JSON `Metadata`.

    `node` and `header` are the two halves of the raw record.
    """

    format: tx.Annotated[tx.Literal["x5"], tx.Doc("Always `'x5'`.")] = "x5"

    raw: tx.Annotated[
        tx.Optional[tx.Tuple[X5Header, tx.Optional[X5Node]]],
        tx.Doc(
            """
            `(header, node)`: the root of the file and the node the
            transformation was read from (`None` for a chain of several
            nodes, whose own metadata stays in their nodes).
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def header(self) -> tx.Optional[X5Header]:
        """The root of the file (first half of the raw record)."""
        return None if self.raw is None else self.raw[0]

    @property
    def node(self) -> tx.Optional[X5Node]:
        """The node (second half of the raw record)."""
        return None if self.raw is None else self.raw[1]

    # --- hooks --------------------------------------------------------

    @classmethod
    def _default_raw(cls) -> tx.Tuple[X5Header, X5Node]:
        return X5Header(), X5Node()

    @classmethod
    def _decode(
        cls, raw: tx.Any, *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        node = None if raw is None else raw[1]
        json = getattr(node, "metadata", None)
        if not isinstance(json, dict):
            return {}
        values, extra = decode_object(json, VOCABULARY)
        return {**values, "extra": extra or None}

    def _encode(
        self,
        raw: tx.Tuple[X5Header, tx.Optional[X5Node]],
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Tuple[X5Header, tx.Optional[X5Node]]:
        header, node = raw
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
        return header, replace(node, metadata=json)
