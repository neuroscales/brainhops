"""The XML of a BigDataViewer dataset: parsing it with the byte spans of
its elements, and writing the elements of a view registration.

The record of a dataset keeps the bytes of its XML file, so that an
untouched file is written back byte for byte, and a changed registration
is written by replacing the bytes of one `<ViewRegistration>` element
only. The parser below is therefore built on expat, which reports where
each element starts in the bytes, rather than on `xml.etree`, which does
not.
"""

# stdlib
from xml.parsers import expat

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError

_SNIFF_SIZE = 4096
"""Number of leading bytes that are enough to find the root element."""


class _Element:
    """An element of the XML tree, with where it lies in the bytes.

    The tree is built by [`parse_xml`][] and is private to the parsing of
    a record. `start` is the offset of the `<` of the start tag, and `end`
    is the offset just after the `>` of the end tag, or of the start tag
    of an empty element.
    """

    __slots__ = ("attrs", "children", "end", "name", "parts", "start")

    def __init__(self, name: str, attrs: tx.Dict[str, str], start: int):
        self.name = name
        self.attrs = attrs
        self.start = start
        self.end = start
        self.parts: tx.List[str] = []
        self.children: tx.List["_Element"] = []

    @property
    def text(self) -> str:
        """The text that the element holds, without surrounding spaces."""
        return "".join(self.parts).strip()

    def child(self, name: str) -> tx.Optional["_Element"]:
        """Return the first child of a given name, or `None`."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def all(self, name: str) -> tx.List["_Element"]:
        """Return the children of a given name, in order."""
        return [child for child in self.children if child.name == name]

    def child_text(self, name: str) -> tx.Optional[str]:
        """Return the text of the first child of a given name, or `None`."""
        child = self.child(name)
        return None if child is None else child.text


class _Root(Exception):
    """Raised by the sniffer as soon as it meets the root element."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def root_name(head: bytes) -> tx.Optional[str]:
    """Return the name of the root element of an XML document.

    Only the leading bytes are parsed, up to the start tag of the root
    element, which follows the XML declaration and any comments.

    Parameters
    ----------
    head : bytes
        The first bytes of the document.

    Returns
    -------
    str or None
        The name of the root element, or `None` if the bytes are not the
        start of an XML document.
    """
    parser = expat.ParserCreate()

    def start(name: str, attrs: tx.Dict[str, str]) -> None:
        raise _Root(name)

    parser.StartElementHandler = start
    try:
        parser.Parse(bytes(head[:_SNIFF_SIZE]), False)
    except _Root as root:
        return root.name
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
        element = _Element(name, attrs, parser.CurrentByteIndex)
        if stack:
            stack[-1].children.append(element)
        else:
            roots.append(element)
        stack.append(element)

    def end(name: str) -> None:
        # An empty element ends with its start tag. Otherwise expat stands
        # at the `<` of the end tag.
        element = stack.pop()
        stop = _tag_end(content, element.start)
        if content[stop - 2 : stop] != b"/>":
            stop = _tag_end(content, parser.CurrentByteIndex)
        element.end = stop

    def characters(data: str) -> None:
        if stack:
            stack[-1].parts.append(data)

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    try:
        parser.Parse(bytes(content), True)
    except expat.ExpatError as error:
        raise ParserContentError(f"This XML file is malformed: {error}.")
    return roots[0]


def newline_of(content: bytes) -> bytes:
    """Return the line separator of a document, CRLF or LF.

    JDOM, with which BigDataViewer writes its XML files, separates lines
    with CRLF unless told otherwise (`LineSeparator.DEFAULT` in
    `org/jdom2/output/LineSeparator.java` at jdom commit c6065f6).
    """
    return b"\r\n" if b"\r\n" in content else b"\n"


def indent_before(content: bytes, offset: int) -> bytes:
    """Return the spaces and tabs between a line start and an offset."""
    start = offset
    while start > 0 and content[start - 1 : start] in (b" ", b"\t"):
        start -= 1
    return content[start:offset]


def render_registration(
    timepoint: int,
    setup: int,
    transforms: tx.Sequence[bytes],
    indent: bytes,
    unit: bytes,
    newline: bytes,
) -> bytes:
    """Return the bytes of a `<ViewRegistration>` element.

    The element is laid out as JDOM lays it out with its pretty format,
    which BigDataViewer uses: one child per line, indented by `unit` more
    than the element. An element without transforms is written as an
    empty element, which BigDataViewer reads as the identity.

    Parameters
    ----------
    timepoint : int
        The id of the time point of the view.
    setup : int
        The id of the setup of the view.
    transforms : sequence of bytes
        The `<ViewTransform>` elements, in the order of the file, each
        already indented for its own lines after the first.
    indent : bytes
        The indentation of the element itself.
    unit : bytes
        One level of indentation.
    newline : bytes
        The line separator of the document.

    Returns
    -------
    bytes
        The element, from its `<` to its last `>`.
    """
    tag = b'<ViewRegistration timepoint="%d" setup="%d"' % (timepoint, setup)
    if not transforms:
        return tag + b" />"
    lines = [tag + b">"]
    lines += [indent + unit + element for element in transforms]
    lines.append(indent + b"</ViewRegistration>")
    return newline.join(lines)


def render_transform(
    name: tx.Optional[str],
    affine: tx.Sequence[float],
    indent: bytes,
    unit: bytes,
    newline: bytes,
) -> bytes:
    """Return the bytes of an affine `<ViewTransform>` element.

    The twelve values are written in the shortest form that reads back
    as the same double, which Java's `Double.parseDouble` also reads.

    Parameters
    ----------
    name : str, optional
        The text of the `<Name>` child, which is left out when `None`.
    affine : sequence of float
        The first three rows of the matrix, row after row.
    indent : bytes
        The indentation of the element itself.
    unit : bytes
        One level of indentation.
    newline : bytes
        The line separator of the document.

    Returns
    -------
    bytes
        The element, from its `<` to its last `>`.
    """
    lines = [b'<ViewTransform type="affine">']
    if name is not None:
        lines.append(indent + unit + b"<Name>" + _escape(name) + b"</Name>")
    values = " ".join(repr(float(v)) for v in affine).encode("ascii")
    lines.append(indent + unit + b"<affine>" + values + b"</affine>")
    lines.append(indent + b"</ViewTransform>")
    return newline.join(lines)


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


def _escape(text: str) -> bytes:
    """Escape text for an element and encode it as ASCII character
    references where needed, so that it fits any ASCII-based encoding."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.encode("ascii", "xmlcharrefreplace")
