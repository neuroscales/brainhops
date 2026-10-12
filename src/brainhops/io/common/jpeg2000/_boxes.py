"""The boxes of a JP2 file and the main header of a codestream.

A JP2 file is a sequence of boxes. Each box starts with its length and
its type, and the box `jp2c` holds the codestream, which is the coded
image. A codestream starts with a main header, made of marker segments,
which is followed by the tiles. This module walks these structures, and
rewrites the parts of them that the record of a file describes beside
the coded image: the resolution box, the XML and UUID boxes, and the
comments of the main header. The coded image is copied as it is.
"""

# stdlib
import math
import struct
from fractions import Fraction
from io import BytesIO

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import ParserContentError, WriterError

JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
"""The signature box that starts a JP2 file, and a JPX file."""

J2K_SIGNATURE = b"\xff\x4f\xff\x51"
"""The start of a codestream: the SOC marker, then the SIZ marker."""

KEPT_BOXES = (b"xml ", b"uuid", b"uinf")
"""The top-level boxes that the record keeps, such as GMLJP2 and XMP."""

SOT = 0x90
"""The marker that starts a tile-part, after the main header."""

SOD = 0x93
"""The marker that starts the data of a tile-part."""

SIZ = 0x51
"""The marker of the size of the image and of its components."""

COD = 0x52
"""The marker of the default coding style."""

COC = 0x53
"""The marker of the coding style of one component."""

COM = 0x64
"""The marker of a comment."""

_Box = tx.Tuple[bytes, int, int, tx.Optional[int]]


def walk_boxes(file: tx.IO, end: tx.Optional[int] = None) -> tx.Iterator[_Box]:
    """Walk the boxes of a JP2 file, or of a superbox.

    Each box is yielded as `(type, start, content_start, end)`, where
    `end` is `None` for a box that runs to the end of the file. After each
    box is yielded, the stream is moved to the end of that box.

    Parameters
    ----------
    file : IO
        The stream, at the start of the first box.
    end : int, optional
        The position at which the boxes end, or `None` for the end of the
        stream.

    Yields
    ------
    tuple of (bytes, int, int, int or None)
        The type, start, start of the content and end of each box.

    Raises
    ------
    ParserContentError
        If a box is truncated or has a length that is too small.
    """
    while end is None or file.tell() < end:
        start = file.tell()
        head = file.read(8)
        if not head:
            return
        if len(head) != 8:
            raise ParserContentError("This JP2 file has a truncated box.")
        length, kind = struct.unpack(">I4s", head)
        content = start + 8
        if length == 1:
            (length,) = struct.unpack(">Q", read_exact(file, 8))
            content += 8
        if length == 0:
            yield kind, start, content, None
            return
        if length < content - start:
            raise ParserContentError(f"This JP2 file has a bad {kind!r} box.")
        yield kind, start, content, start + length
        file.seek(start + length)


def read_exact(file: tx.IO, n: int) -> bytes:
    """Read exactly `n` bytes from a stream.

    Raises
    ------
    ParserContentError
        If the stream ends before.
    """
    data = file.read(n)
    if len(data) != n:
        raise ParserContentError("This JPEG 2000 file is truncated.")
    return data


def container_of(head: bytes) -> tx.Optional[str]:
    """Return `"jp2"`, `"j2k"` or `None` from the first 12 bytes of a file."""
    if head[:12] == JP2_SIGNATURE:
        return "jp2"
    if head[:4] == J2K_SIGNATURE:
        return "j2k"
    return None


def read_resolution(content: bytes) -> tx.Optional[tx.Tuple[float, float]]:
    """Decode a `resc` or `resd` box into `(x, y)` grid points per metre.

    A box that is too short, or whose numerator or denominator is zero,
    gives `None`.
    """
    if len(content) < 10:
        return None
    vn, vd, hn, hd, ve, he = struct.unpack(">HHHHbb", content[:10])
    if not (vn and vd and hn and hd):
        return None
    return hn / hd * 10.0**he, vn / vd * 10.0**ve


def rewrite(content: bytes, record: tx.Any) -> bytes:
    """Return a file whose header says what a record says.

    The file is a JP2 file or a codestream, whole or cut after the main
    header of its codestream. The result is in the container of the
    record. Its comments are those of the record, its resolution box
    holds the capture and display resolutions of the record, and its XML
    and UUID boxes before the codestream are those of the record. A JP2
    file keeps its other boxes, and a codestream that is put in a JP2
    file gets the signature, file type, header and colour boxes that a
    JP2 file needs. The coded image is copied as it is.

    Parameters
    ----------
    content : bytes
        The file, or its beginning.
    record : Jpeg2000Raw
        The record whose comments, resolutions, boxes and container the
        result has.

    Returns
    -------
    bytes
        The rewritten file.

    Raises
    ------
    WriterError
        If a comment cannot be stored, or a resolution is not positive.
    """
    container = container_of(content[:12])
    if container == "j2k":
        codestream = _with_comments(content, record.comments)
        if record.container == "j2k":
            return codestream
        return b"".join(
            [
                JP2_SIGNATURE,
                _box(b"ftyp", b"jp2 " + b"\x00" * 4 + b"jp2 "),
                _box(b"jp2h", _image_header(record) + _resolutions(record)),
                b"".join(_box(kind, data) for kind, data in record.boxes),
                _box(b"jp2c", codestream),
            ]
        )
    if container != "jp2":
        raise WriterError("This is neither a JP2 file nor a codestream.")
    before, codestream, after = [], b"", []
    f = BytesIO(content)
    found = False
    for kind, start, data, end in walk_boxes(f):
        stop = len(content) if end is None else min(end, len(content))
        if found:
            after.append(content[start:stop])
        elif kind == b"jp2c":
            found = True
            old = content[data:stop]
            codestream = _with_comments(old, record.comments)
            # A box cut short by the end of the content keeps its length,
            # changed by as much as the codestream changed.
            missing = 0 if end is None else max(0, end - len(content))
            jp2c_length = len(codestream) + missing
        elif kind == b"jp2h":
            children = [
                content[
                    sub_start : len(content) if sub_end is None else sub_end
                ]
                for sub, sub_start, _, sub_end in walk_boxes(f, end)
                if sub != b"res "
            ]
            children.append(_resolutions(record))
            before.append(_box(b"jp2h", b"".join(children)))
        elif kind not in KEPT_BOXES:
            before.append(content[start:stop])
        if end is None or end > len(content):
            break
    if not found:
        raise WriterError("This JP2 file has no codestream box.")
    if record.container == "j2k":
        return codestream
    kept = b"".join(_box(kind, data) for kind, data in record.boxes)
    return b"".join(
        before + [kept, _box_header(b"jp2c", jp2c_length), codestream] + after
    )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _box_header(kind: bytes, length: int) -> bytes:
    """Return the header of a box whose content has `length` bytes."""
    if length + 8 >= 2**32:
        return struct.pack(">I4sQ", 1, kind, length + 16)
    return struct.pack(">I4s", length + 8, kind)


def _box(kind: bytes, content: bytes) -> bytes:
    """Return a box of a type and a content."""
    return _box_header(kind, len(content)) + content


def _resolution_box(kind: bytes, value: tx.Sequence[float]) -> bytes:
    """Encode a `resc` or `resd` box from `(x, y)` grid points per metre.

    Each value is stored as a fraction `n / d` of 16-bit integers times a
    power of ten, so it is rounded to about six significant digits.

    Raises
    ------
    WriterError
        If a value is not positive and finite.
    """
    fields = []
    for v in (float(value[1]), float(value[0])):
        if not math.isfinite(v) or v <= 0:
            raise WriterError(f"A resolution must be positive, not {v!r}.")
        # The value is written as x * 10**e with x in [1, 10), and x as a
        # fraction n / d with d < 6554, so that n < 65536.
        exponent = math.floor(math.log10(v))
        ratio = Fraction(v / 10.0**exponent).limit_denominator(6553)
        fields.append((ratio.numerator, ratio.denominator, exponent))
    (vn, vd, ve), (hn, hd, he) = fields
    return _box(kind, struct.pack(">HHHHbb", vn, vd, hn, hd, ve, he))


def _resolutions(record: tx.Any) -> bytes:
    """Return the resolution box of a record, or nothing without one."""
    content = b""
    if record.capture_resolution is not None:
        content += _resolution_box(b"resc", record.capture_resolution)
    if record.display_resolution is not None:
        content += _resolution_box(b"resd", record.display_resolution)
    return _box(b"res ", content) if content else b""


def _image_header(record: tx.Any) -> bytes:
    """Return the image header and colour boxes of a JP2 file.

    The bit depth of the components is stored once when they share it,
    and in a box of its own otherwise. The colour space is greyscale for
    one or two components and sRGB for more, as OpenJPEG writes it.
    """
    width, height = record.size
    depths = {
        (depth - 1) | (0x80 if signed else 0)
        for depth, signed in zip(record.bit_depths, record.signed)
    }
    count = len(record.bit_depths)
    bpc = depths.pop() if len(depths) == 1 else 0xFF
    header = _box(
        b"ihdr", struct.pack(">IIHBBBB", height, width, count, bpc, 7, 0, 0)
    )
    if bpc == 0xFF:
        header += _box(
            b"bpcc",
            bytes(
                (depth - 1) | (0x80 if signed else 0)
                for depth, signed in zip(record.bit_depths, record.signed)
            ),
        )
    colour = 17 if count <= 2 else 16
    return header + _box(b"colr", struct.pack(">BBBI", 1, 0, 0, colour))


def _with_comments(codestream: bytes, comments: tx.Sequence[str]) -> bytes:
    """Return a codestream whose main header holds other comments.

    The text comments of the main header are removed, and the comments
    are written as text before the first tile-part. Binary comments and
    the other marker segments are kept.

    Raises
    ------
    WriterError
        If a comment is not Latin-1 text or is too long for a marker
        segment.
    """
    segments = []
    position = 2
    while position + 4 <= len(codestream):
        marker = codestream[position : position + 2]
        if marker[0] != 0xFF or marker[1] in (SOT, SOD):
            break
        (length,) = struct.unpack(
            ">H", codestream[position + 2 : position + 4]
        )
        segment = codestream[position : position + 2 + length]
        text = marker[1] == COM and segment[4:6] == b"\x00\x01"
        if not text:
            segments.append(segment)
        position += 2 + length
    for comment in comments:
        try:
            data = comment.encode("latin-1")
        except UnicodeEncodeError as e:
            raise WriterError(
                f"A JPEG 2000 comment is Latin-1 text, so {comment!r} "
                f"cannot be stored."
            ) from e
        if len(data) > 65531:
            raise WriterError(
                f"A JPEG 2000 comment holds at most 65531 bytes, not "
                f"{len(data)}."
            )
        segments.append(struct.pack(">BBHH", 0xFF, COM, len(data) + 4, 1))
        segments.append(data)
    return codestream[:2] + b"".join(segments) + codestream[position:]
