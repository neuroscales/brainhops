"""
The JPEG 2000 backend: a small parser of the JP2 boxes and of the main
header of the codestream, and decoding and encoding with Pillow (which
wraps OpenJPEG).

Pillow decodes a whole resolution level at once (`reduce`), but has no
public way to decode one level of an image whose size is not a multiple of
its downsampling factor, as it rounds the size of a reduced level instead
of taking its ceiling. The level is decoded here with the size the
codestream defines (ITU-T T.800, B.5).
"""

# stdlib
import math
import struct
from fractions import Fraction
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic
from PIL import Image, ImageFile

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.images.pillow._utils import pillow_to_array

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
"""The JPEG 2000 signature box that a JP2 (or JPX) file starts with."""

J2K_SIGNATURE = b"\xff\x4f\xff\x51"
"""The start of a raw codestream: SOC, then the SIZ marker."""

JP2_EXTENSIONS = (".jp2", ".jpx", ".jpf")
J2K_EXTENSIONS = (".j2k", ".j2c", ".jpc")

# Top-level boxes kept from a JP2 file and written back: XML (GMLJP2, ...),
# UUID (GeoJP2, XMP, ...) and UUID info boxes.
_KEPT_BOXES = (b"xml ", b"uuid", b"uinf")

# The comment OpenJPEG writes on its own, which is not written back.
_OPENJPEG_COMMENT = "Created by OpenJPEG"

# Markers of the main header of a codestream.
_SOT, _SOD, _SIZ, _COD, _COC, _COM = 0x90, 0x93, 0x51, 0x52, 0x53, 0x64

# The number of resolutions written when nothing says how many: OpenJPEG's
# default, as long as the image is large enough.
_DEFAULT_RESOLUTIONS = 6


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


class Jpeg2000Header(Magic, frozen=True):
    """What a JPEG 2000 file says about its image, besides the pixels."""

    container: str
    """`"jp2"` for a JP2 file, `"j2k"` for a raw codestream."""

    size: tx.Tuple[int, int]
    """The size of the image area, `(width, height)` (`Xsiz - XOsiz`,
    `Ysiz - YOsiz`)."""

    image_offset: tx.Tuple[int, int]
    """The offset of the image area on the reference grid, `(XOsiz,
    YOsiz)`."""

    tile_size: tx.Tuple[int, int]
    """The size of a tile on the reference grid, `(XTsiz, YTsiz)`."""

    tile_offset: tx.Tuple[int, int]
    """The offset of the first tile, `(XTOsiz, YTOsiz)`."""

    bit_depths: tx.Tuple[int, ...]
    """The bit depth of each component."""

    signed: tx.Tuple[bool, ...]
    """Whether each component is signed."""

    subsampling: tx.Tuple[tx.Tuple[int, int], ...]
    """The subsampling of each component, `(XRsiz, YRsiz)`."""

    n_levels: int
    """The number of resolution levels (decomposition levels + 1) of the
    main header, the smallest over all components."""

    n_layers: int
    """The number of quality layers."""

    reversible: bool
    """Whether the wavelet transform is the reversible 5-3 one (lossless),
    rather than the irreversible 9-7 one."""

    comments: tx.Tuple[str, ...] = ()
    """The text comments (COM markers) of the main header."""

    capture_resolution: tx.Optional[tx.Tuple[float, float]] = None
    """The capture resolution of a JP2 file (`resc` box), `(x, y)` in
    grid points per metre."""

    display_resolution: tx.Optional[tx.Tuple[float, float]] = None
    """The default display resolution of a JP2 file (`resd` box), `(x,
    y)` in grid points per metre."""

    boxes: tx.Tuple[tx.Tuple[bytes, bytes], ...] = ()
    """The top-level XML, UUID and UUID info boxes of a JP2 file, as
    `(type, content)`, in file order."""

    def level_shape(self, level: int) -> tx.Tuple[int, int]:
        """The size `(width, height)` of a resolution level (0: the full
        resolution), as the codestream defines it: `ceil(X1 / 2**r) -
        ceil(X0 / 2**r)`."""
        return tuple(
            _ceil_div(offset + size, 2**level) - _ceil_div(offset, 2**level)
            for offset, size in zip(self.image_offset, self.size)
        )

    def level_start(self, level: int) -> tx.Tuple[int, int]:
        """
        The full-resolution pixel on which the first pixel of a level is
        centred, `(x, y)`.

        The wavelet transform keeps the low-pass samples at the even
        positions of the reference grid, so sample `u` of a level reduced
        by `f = 2**r` sits on reference coordinate `f * u`, and the first
        sample of the level is `u = ceil(X0 / f)`.
        """
        factor = 2**level
        return tuple(
            factor * _ceil_div(offset, factor) - offset
            for offset in self.image_offset
        )


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _read_exact(file: tx.IO, n: int) -> bytes:
    data = file.read(n)
    if len(data) != n:
        raise ParserContentError("This JPEG 2000 file is truncated.")
    return data


def _boxes(
    file: tx.IO, end: tx.Optional[int] = None
) -> tx.Iterator[tx.Tuple[bytes, int, int, int]]:
    """
    Walk the boxes of a JP2 file (or of a superbox), yielding `(type,
    start, content_start, end)`; `end` is `None` for a box that runs to the
    end of the file. The file is left at the end of the last box.
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
            (length,) = struct.unpack(">Q", _read_exact(file, 8))
            content += 8
        if length == 0:
            yield kind, start, content, None
            return
        if length < content - start:
            raise ParserContentError(f"This JP2 file has a bad {kind!r} box.")
        yield kind, start, content, start + length
        file.seek(start + length)


def _resolution(content: bytes) -> tx.Optional[tx.Tuple[float, float]]:
    """A `resc` or `resd` box: `(x, y)` in grid points per metre."""
    if len(content) < 10:
        return None
    vn, vd, hn, hd, ve, he = struct.unpack(">HHHHbb", content[:10])
    if not (vn and vd and hn and hd):
        return None
    return hn / hd * 10.0**he, vn / vd * 10.0**ve


def _parse_jp2(file: tx.IO) -> tx.Tuple[tx.Dict[str, tx.Any], int]:
    """The boxes of interest of a JP2 file, and where its codestream
    starts."""
    found: tx.Dict[str, tx.Any] = {"container": "jp2"}
    kept = []
    codestream = None
    for kind, _, content, end in _boxes(file):
        if kind == b"jp2c":
            codestream = content
            break
        if kind == b"jp2h":
            for sub, _, _, sub_end in _boxes(file, end):
                if sub != b"res ":
                    continue
                for res, _, res_content, res_end in _boxes(file, sub_end):
                    file.seek(res_content)
                    value = _resolution(file.read(res_end - res_content))
                    if res == b"resc":
                        found["capture_resolution"] = value
                    elif res == b"resd":
                        found["display_resolution"] = value
        elif kind in _KEPT_BOXES and end is not None:
            kept.append((kind, file.read(end - content)))
        if end is None:
            break
    if codestream is None:
        raise ParserContentError("This JP2 file has no codestream box.")
    found["boxes"] = tuple(kept)
    return found, codestream


def _parse_codestream(file: tx.IO) -> tx.Dict[str, tx.Any]:
    """The main header of a codestream, from SOC to the first tile."""
    if _read_exact(file, 2) != b"\xff\x4f":
        raise ParserContentError("This is not a JPEG 2000 codestream.")
    found: tx.Dict[str, tx.Any] = {}
    levels = []
    comments = []
    while True:
        marker = _read_exact(file, 2)
        if marker[0] != 0xFF:
            raise ParserContentError("This JPEG 2000 codestream is corrupt.")
        code = marker[1]
        if code in (_SOT, _SOD):
            break
        (length,) = struct.unpack(">H", _read_exact(file, 2))
        segment = _read_exact(file, length - 2)
        if code == _SIZ:
            (_, xs, ys, xo, yo, xt, yt, xto, yto, n) = struct.unpack(
                ">HIIIIIIIIH", segment[:36]
            )
            comps = [segment[36 + 3 * i : 39 + 3 * i] for i in range(n)]
            found["size"] = (xs - xo, ys - yo)
            found["image_offset"] = (xo, yo)
            found["tile_size"] = (xt, yt)
            found["tile_offset"] = (xto, yto)
            found["bit_depths"] = tuple((c[0] & 0x7F) + 1 for c in comps)
            found["signed"] = tuple(bool(c[0] & 0x80) for c in comps)
            found["subsampling"] = tuple((c[1], c[2]) for c in comps)
        elif code == _COD:
            found["n_layers"] = struct.unpack(">H", segment[2:4])[0]
            levels.append(segment[5] + 1)
            found["reversible"] = segment[9] == 1
        elif code == _COC:
            # The component index is one byte, or two with > 256 components.
            wide = len(found.get("bit_depths", ())) > 256
            levels.append(segment[3 if wide else 2] + 1)
        elif code == _COM and len(segment) > 2:
            if struct.unpack(">H", segment[:2])[0] == 1:
                comments.append(segment[2:].decode("latin-1"))
    if "size" not in found or not levels:
        raise ParserContentError(
            "This JPEG 2000 codestream has no SIZ or COD marker."
        )
    found["n_levels"] = min(levels)
    found["comments"] = tuple(comments)
    return found


def read_header(file: tx.IO) -> Jpeg2000Header:
    """
    Parse the header of a JP2 file or raw codestream, from an open file,
    which is left where it was.

    Raises
    ------
    ParserContentError
        If the file is neither, or is corrupt.
    """
    with preserve_position(file):
        head = file.read(12)
        file.seek(-len(head), 1)
        try:
            if head == JP2_SIGNATURE:
                found, codestream = _parse_jp2(file)
                file.seek(codestream)
            elif head[:4] == J2K_SIGNATURE:
                found = {"container": "j2k"}
            else:
                raise ParserContentError(
                    "This is neither a JP2 file nor a JPEG 2000 codestream."
                )
            found.update(_parse_codestream(file))
        except (struct.error, IndexError) as e:
            raise ParserContentError(
                f"This JPEG 2000 file is corrupt: {e}"
            ) from e
    return Jpeg2000Header(**found)


def container_of(head: bytes) -> tx.Optional[str]:
    """`"jp2"`, `"j2k"`, or `None`, from the first 12 bytes of a file."""
    if head[:12] == JP2_SIGNATURE:
        return "jp2"
    if head[:4] == J2K_SIGNATURE:
        return "j2k"
    return None


# ----------------------------------------------------------------------
#   SOURCE
# ----------------------------------------------------------------------


class Jpeg2000Source:
    """
    Where a JPEG 2000 file can be opened again, to decode a level after
    the header has been read: a local file by name, anything else from
    its bytes, held in memory.
    """

    def __init__(
        self,
        filename: tx.Optional[str] = None,
        content: tx.Optional[bytes] = None,
    ) -> None:
        if (filename is None) == (content is None):
            raise ValueError("Give either a filename or the content.")
        self.filename = filename
        self.content = content

    @classmethod
    def from_filename(cls, filename: path.FilenameLike) -> tx.Self:
        p = path.Path(filename) if isinstance(filename, str) else filename
        try:
            local = p.protocol.lower() in ("file", "local")
        except Exception:
            local = False
        if local:
            return cls(filename=str(p))
        with p.open("rb") as f:
            return cls(content=f.read())

    def open(self) -> tx.IO:
        """Open the file, in binary mode (a context manager)."""
        if self.filename is not None:
            return open(self.filename, "rb")
        return BytesIO(self.content)


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def decode_level(
    source: Jpeg2000Source, header: Jpeg2000Header, level: int
) -> tx.Tuple[np.ndarray, str]:
    """
    Decode one resolution level with Pillow.

    Returns the C-ordered pixels, `(rows, columns[, samples])`, with the
    values the codestream holds: Pillow's scaling of a bit depth that is
    not 8 or 16 to the full range of its 8- or 16-bit mode, and its offset
    of signed samples, are undone.

    Raises
    ------
    ParserContentError
        If Pillow cannot decode the file, or would lose precision (a
        colour image of more than 8 bits per component), or if a reduced
        level of an image whose area has an offset is asked for (Pillow
        misplaces it).
    PIL.Image.DecompressionBombError
        If the *full-resolution* image is larger than Pillow's limit,
        even when a coarse level is asked for.
    """
    if level and any(header.image_offset):
        # Pillow places a reduced level at the full-resolution offset,
        # outside the image, and refuses it.
        raise ParserContentError(
            f"Pillow cannot decode a reduced level of a JPEG 2000 image "
            f"whose area has an offset on the reference grid "
            f"{header.image_offset}: only level 0 can be read."
        )
    width, height = header.level_shape(level)
    with source.open() as f:
        try:
            im = Image.open(f, formats=["JPEG2000"])
        except Image.DecompressionBombError:
            raise
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot read this JPEG 2000 file: {e}"
            ) from e
        try:
            if level:
                # Pillow's own `reduce` rounds the size of the level, which
                # the decoder then refuses when it is not the ceiling.
                tile = im.tile[0]
                args = tile[3]
                new = (
                    tile[0],
                    (0, 0, width, height),
                    tile[2],
                    (args[0], int(level), 0) + tuple(args[3:]),
                )
                if hasattr(tile, "_replace"):
                    new = type(tile)(*new)
                im._size = (width, height)
                im.tile = [new]
                ImageFile.ImageFile.load(im)
            else:
                im.load()
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot decode level {level} of this JPEG 2000 "
                f"file: {e}"
            ) from e
        mode = im.mode
        array, axes = pillow_to_array(im)
    if mode not in ("P", "PA"):
        array = _undo_pillow_scaling(array, header)
    return array, axes


def _undo_pillow_scaling(
    array: np.ndarray, header: Jpeg2000Header
) -> np.ndarray:
    """
    Pillow decodes a component of `b` bits into its 8- or 16-bit mode as
    `(v + offset) << (bits - b)`, where `offset` is `2**(b - 1)` for a
    signed component and 0 otherwise. Undo it.
    """
    bits = array.dtype.itemsize * 8
    depth = max(header.bit_depths)
    if not np.issubdtype(array.dtype, np.unsignedinteger):
        return array
    if depth > bits:
        raise ParserContentError(
            f"This JPEG 2000 image has {depth} bits per component, which "
            f"Pillow decodes to {bits} bits: it cannot be read without "
            f"losing precision."
        )
    shift = bits - depth
    signed = any(header.signed)
    if not shift and not signed:
        return array
    if not signed:
        return array >> shift
    out = (array.astype(np.int32) >> shift) - 2 ** (depth - 1)
    return out.astype(np.int8 if depth <= 8 else np.int16)


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def default_resolutions(shape: tx.Sequence[int]) -> int:
    """The number of resolution levels written by default: OpenJPEG's
    six, or fewer when the image is too small for the coarsest one to have
    a pixel."""
    smallest = max(1, min(int(n) for n in shape))
    return max(1, min(_DEFAULT_RESOLUTIONS, int(math.log2(smallest)) + 1))


def _box(kind: bytes, content: bytes) -> bytes:
    if len(content) + 8 >= 2**32:
        return struct.pack(">I4sQ", 1, kind, len(content) + 16) + content
    return struct.pack(">I4s", len(content) + 8, kind) + content


def _resolution_box(kind: bytes, value: tx.Sequence[float]) -> bytes:
    """A `resc` or `resd` box, from `(x, y)` in grid points per metre."""
    fields = []
    for v in (float(value[1]), float(value[0])):
        if not math.isfinite(v) or v <= 0:
            raise WriterError(f"A resolution must be positive, not {v!r}.")
        # x = v / 10**e in [1, 10), as a fraction n / d with d < 6554, so
        # that n < 65536.
        exponent = math.floor(math.log10(v))
        ratio = Fraction(v / 10.0**exponent).limit_denominator(6553)
        numerator, denominator = ratio.numerator, ratio.denominator
        fields.append((round(numerator), denominator, exponent))
    (vn, vd, ve), (hn, hd, he) = fields
    return _box(kind, struct.pack(">HHHHbb", vn, vd, hn, hd, ve, he))


def add_boxes(
    content: bytes,
    capture: tx.Optional[tx.Sequence[float]] = None,
    display: tx.Optional[tx.Sequence[float]] = None,
    boxes: tx.Sequence[tx.Tuple[bytes, bytes]] = (),
) -> bytes:
    """
    Add a resolution box to the header box of a JP2 file Pillow wrote,
    and top-level boxes before its codestream.
    """
    if not (capture or display or boxes):
        return content
    res = b""
    if capture is not None:
        res += _resolution_box(b"resc", capture)
    if display is not None:
        res += _resolution_box(b"resd", display)
    out = []
    f = BytesIO(content)
    for kind, start, _, end in _boxes(f):
        stop = len(content) if end is None else end
        if kind == b"jp2h" and res:
            children = []
            for sub, sub_start, _, sub_end in _boxes(f, end):
                if sub != b"res ":
                    children.append(content[sub_start:sub_end])
            children.append(_box(b"res ", res))
            out.append(_box(b"jp2h", b"".join(children)))
        else:
            if kind == b"jp2c":
                out.extend(_box(k, c) for k, c in boxes)
                boxes = ()
            out.append(content[start:stop])
        f.seek(stop)
    return b"".join(out)


def is_openjpeg_comment(comment: str) -> bool:
    """Whether a comment is the one OpenJPEG writes on its own."""
    return comment.startswith(_OPENJPEG_COMMENT)
