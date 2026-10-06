"""
The trailing tags of an MGH/MGZ file, which `nibabel` does not read.

After the footer, FreeSurfer writes a sequence of tags: a big-endian
`int32` tag id, a length, and the payload. The length is an `int64`,
except for a few legacy ids (`TAG_OLD_MGH_XFORM` has an `int32` length,
and `TAG_OLD_COLORTABLE`, `TAG_OLD_USEREALRAS` and `TAG_OLD_SURF_GEOM`
none). A command line (`TAG_CMDLINE = 3`) is a NUL-terminated string.
"""

__all__ = ["decode_history", "encode_history", "parse_tags"]

# stdlib
import struct

# dependencies
import typing_extensions as tx

# FreeSurfer tag ids (`utils/tags.h`).
TAG_OLD_COLORTABLE = 1
TAG_OLD_USEREALRAS = 2
TAG_CMDLINE = 3
TAG_OLD_SURF_GEOM = 20
TAG_OLD_MGH_XFORM = 30


def parse_tags(
    tags: bytes,
) -> tx.Optional[tx.List[tx.Tuple[int, bytes]]]:
    """
    Split the trailing tags of an MGH file into individual tags.

    Parameters
    ----------
    tags : bytes
        The bytes that follow the footer.

    Returns
    -------
    list of (int, bytes) or None
        One `(tag id, chunk)` pair per tag, where `chunk` holds the bytes
        of the tag verbatim (id, length and payload), or `None` when the
        bytes do not parse as a FreeSurfer tag stream.
    """
    out = []
    pos, end = 0, len(tags)
    while pos < end:
        if end - pos < 4:
            return None
        start = pos
        (tag,) = struct.unpack_from(">i", tags, pos)
        pos += 4
        if tag in _NO_LENGTH:
            # A legacy tag with no length: only safe when it is the last.
            if pos != end:
                return None
            out.append((tag, tags[start:]))
            break
        if tag == TAG_OLD_MGH_XFORM:
            if end - pos < 4:
                return None
            (length,) = struct.unpack_from(">i", tags, pos)
            pos += 4
        else:
            if end - pos < 8:
                return None
            (length,) = struct.unpack_from(">q", tags, pos)
            pos += 8
        if tag <= 0 or length < 0 or pos + length > end:
            return None
        pos += length
        out.append((tag, tags[start:pos]))
    return out


def decode_history(tags: bytes) -> tx.Optional[tx.Tuple[str, ...]]:
    """
    Read the command lines stored in the trailing tags of an MGH file.

    Parameters
    ----------
    tags : bytes
        The bytes that follow the footer.

    Returns
    -------
    tuple of str or None
        The command lines, oldest first, or `None` when there is none or
        when the tags do not parse.
    """
    parsed = parse_tags(tags)
    if not parsed:
        return None
    history = tuple(
        _cmdline_payload(chunk) for tag, chunk in parsed if tag == TAG_CMDLINE
    )
    return history or None


def encode_history(
    tags: bytes, history: tx.Optional[tx.Sequence[str]]
) -> tx.Optional[bytes]:
    """
    Replace the command lines stored in the trailing tags of an MGH file.

    The other tags are kept in place. The new command lines go where the
    first old one was, or at the end when there was none.

    Parameters
    ----------
    tags : bytes
        The bytes that follow the footer.
    history : sequence of str or None
        The new command lines. `None` removes them all.

    Returns
    -------
    bytes or None
        The new trailing tags, or `None` when `tags` does not parse.
    """
    parsed = parse_tags(tags)
    if parsed is None:
        return None
    new = b"".join(_cmdline_chunk(c) for c in history or ())
    out, placed = [], False
    for tag, chunk in parsed:
        if tag == TAG_CMDLINE:
            if not placed:
                out.append(new)
                placed = True
            continue
        out.append(chunk)
    if not placed:
        # FreeSurfer writes the command lines before the legacy tags with
        # no length, which must stay last.
        if out and parsed[-1][0] in _NO_LENGTH:
            out.insert(len(out) - 1, new)
        else:
            out.append(new)
    return b"".join(out)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

# Legacy tags with no length field.
_NO_LENGTH = (TAG_OLD_COLORTABLE, TAG_OLD_USEREALRAS, TAG_OLD_SURF_GEOM)


def _cmdline_payload(chunk: bytes) -> str:
    payload = chunk[12:]
    return payload.split(b"\0", 1)[0].decode("utf-8", "replace")


def _cmdline_chunk(command: str) -> bytes:
    payload = command.encode("utf-8") + b"\0"
    return struct.pack(">iq", TAG_CMDLINE, len(payload)) + payload
