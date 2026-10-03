"""
The metadata of X5 files: [`X5Metadata`][].

Its record (`raw`) is the pair `(X5Header, X5Node)`: the root of the
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
    _VOCABULARY,
    ALL,
    ConversionReport,
    FormatMetadata,
)
from brainhops.io.metadata.bids import _from_json, _jsonable, _to_json
from brainhops.io.metadata.bids import sidecar_key as _sidecar_key

# locals
from ._struct import X5Header, X5Node

# JSON key -> vocabulary field.
_KEYS: tx.Dict[str, str] = {
    _sidecar_key(name): name for name in _VOCABULARY if name != "extra"
}
_NAMES: tx.Dict[str, str] = {name: key for key, name in _KEYS.items()}


class X5Metadata(FormatMetadata, on={"format": "x5"}, supports=ALL):
    """
    The metadata of an X5 transform node, stored in its JSON `Metadata`.

    `node` and `header` are the two halves of the record.
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
        """The root of the file (first half of the record)."""
        return None if self.raw is None else self.raw[0]

    @property
    def node(self) -> tx.Optional[X5Node]:
        """The node (second half of the record)."""
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
        out: tx.Dict[str, tx.Any] = {}
        extra: tx.Dict[str, tx.Any] = {}
        for key, value in json.items():
            name = _KEYS.get(key)
            if name is None:
                extra[key] = value
            else:
                out[name] = _from_json(name, value)
        out["extra"] = extra or None
        return out

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
        for name, value in changed.items():
            if name == "extra":
                for key, item in value.items():
                    if item is None:
                        json.pop(key, None)
                    else:
                        json[key] = _jsonable(item)
                continue
            key = _NAMES[name]
            if value is None:
                json.pop(key, None)
            else:
                json[key] = _to_json(name, value)
        if not json and node.metadata is None:
            return raw
        return header, replace(node, metadata=json)
