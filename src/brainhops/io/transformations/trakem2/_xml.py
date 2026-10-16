"""The XML of a TrakEM2 project: parsing it with the byte spans of its
elements and attributes.

The record of a project keeps the bytes of its XML file, so that an
untouched file is written back byte for byte, and a changed patch is
written by replacing the value of its `transform` attribute only. The
parser below is therefore built on expat, which reports where each
element starts in the bytes, rather than on `xml.etree`, which does not.
"""

# stdlib
import re
from xml.parsers import expat

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

_SNIFF_SIZE = 4096
"""Number of leading bytes that are read to recognize a project.

TrakEM2 writes a document type declaration before the root element, and
its internal subset can be longer than this. The name of the document
type comes first, so these bytes are enough to read it.
"""

_ATTRIBUTE = re.compile(rb"""([^\s=/>]+)\s*=\s*("[^"]*"|'[^']*')""")
"""An attribute of a start tag, with its name and its quoted value."""


class _Element:
    """An element of the XML tree, with where its start tag lies.

    The tree is built by [`parse_xml`][] and is private to the parsing of
    a record. `start` is the offset of the `<` of the start tag, and
    `stop` is the offset just after the `>` of the start tag. The names
    of the element and of its attributes are in lower case, as TrakEM2
    reads them (`TMLHandler.startElement` in `ini/trakem2/persistence/
    TMLHandler.java`, TrakEM2 commit cdd8b8f).
    """

    __slots__ = ("attrs", "children", "name", "start", "stop")

    def __init__(
        self, name: str, attrs: tx.Dict[str, str], start: int, stop: int
    ) -> None:
        self.name = name
        self.attrs = attrs
        self.start = start
        self.stop = stop
        self.children: tx.List[_Element] = []


class _Doctype(Exception):
    """Raised by the sniffer as soon as it meets the document type or the
    root element."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def document_name(head: bytes) -> tx.Optional[str]:
    """Return the name of the document type, or else of the root element.

    Only the leading bytes are parsed, up to the document type
    declaration or the start tag of the root element, whichever comes
    first.

    Parameters
    ----------
    head : bytes
        The first bytes of the document.

    Returns
    -------
    str or None
        The name in lower case, or `None` if the bytes are not the start
        of an XML document.
    """
    parser = expat.ParserCreate()

    def doctype(name: str, *args: tx.Any) -> None:
        raise _Doctype(name)

    def start(name: str, attrs: tx.Dict[str, str]) -> None:
        raise _Doctype(name)

    parser.StartDoctypeDeclHandler = doctype
    parser.StartElementHandler = start
    try:
        parser.Parse(bytes(head[:_SNIFF_SIZE]), False)
    except _Doctype as found:
        return found.name.lower()
    except (expat.ExpatError, ValueError):
        return None
    return None


def parse_xml(content: bytes) -> _Element:
    """Parse an XML document into a tree that knows its byte spans.

    Parameters
    ----------
    content : bytes
        The whole document.

    Returns
    -------
    _Element
        The root element.

    Raises
    ------
    ParserContentError
        If the content is not well-formed XML.
    """
    parser = expat.ParserCreate()
    stack: tx.List[_Element] = []
    roots: tx.List[_Element] = []

    def start(name: str, attrs: tx.Dict[str, str]) -> None:
        offset = parser.CurrentByteIndex
        element = _Element(
            name.lower(),
            {key.lower(): value for key, value in attrs.items()},
            offset,
            _tag_end(content, offset),
        )
        if stack:
            stack[-1].children.append(element)
        else:
            roots.append(element)
        stack.append(element)

    def end(name: str) -> None:
        stack.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    try:
        parser.Parse(bytes(content), True)
    except expat.ExpatError as error:
        raise ParserContentError(
            f"This XML file is malformed: {error}."
        ) from error
    return roots[0]


def attribute_span(
    content: bytes, element: _Element, name: str
) -> tx.Optional[tx.Tuple[int, int]]:
    """Return where the value of an attribute lies in the bytes.

    The span covers the value between its quotes, so that replacing it
    keeps the name, the quotes and everything around them.

    Parameters
    ----------
    content : bytes
        The whole document.
    element : _Element
        The element whose start tag holds the attribute.
    name : str
        The name of the attribute, in lower case.

    Returns
    -------
    tuple of int or None
        The offset of the first byte of the value and the offset just
        after its last byte, or `None` if the start tag has no such
        attribute.
    """
    tag = content[element.start : element.stop]
    for match in _ATTRIBUTE.finditer(tag):
        if match.group(1).decode("latin-1").lower() == name:
            begin = element.start + match.start(2) + 1
            return begin, element.start + match.end(2) - 1
    return None


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _tag_end(content: bytes, start: int) -> int:
    """Return the offset just after the `>` of the tag at `start`."""
    quote = None
    position = start
    while True:
        byte = content[position : position + 1]
        if not byte:
            return position
        if quote is not None:
            if byte == quote:
                quote = None
        elif byte in (b'"', b"'"):
            quote = byte
        elif byte == b">":
            return position + 1
        position += 1
