"""The record of a JPEG 2000 file: what it says beside the coded image."""

# stdlib
import struct
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, NoRepr, replace

# internals
from brainhops._core.streams import preserve_position
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)

# this format
from ._boxes import (
    COC,
    COD,
    COM,
    J2K_SIGNATURE,
    KEPT_BOXES,
    SIZ,
    SOD,
    SOT,
    container_of,
    read_exact,
    read_resolution,
    rewrite,
    walk_boxes,
)

_LAYOUT = (
    "size",
    "image_offset",
    "tile_size",
    "tile_offset",
    "bit_depths",
    "signed",
    "subsampling",
    "n_levels",
    "n_layers",
    "reversible",
    "palette",
)
"""The fields of a record that the coded image decides."""


class Jpeg2000Raw(Magic, BinaryFileReader, BinaryFileWriter, frozen=True):
    """The header of a JPEG 2000 file, as brainhops reads it.

    A JPEG 2000 file is either a JP2 file, made of boxes, or a raw
    codestream. The record holds the main header of the codestream (the
    size of the image area and of the tiles, and the bit depth, the
    signedness and the subsampling of each component), the number of
    resolution levels and quality layers, the wavelet, the comments, and,
    for a JP2 file, the capture and display resolutions, the palette, and
    the XML and UUID boxes. It also keeps the bytes that it was read from,
    which are the file up to its first tile.

    Reading a record never decodes the image, so a file cut after the
    main header of its codestream can be read. The record is held by
    [`Jpeg2000Metadata`][brainhops.io.common.jpeg2000.Jpeg2000Metadata]. A
    record is frozen, and two records are equal when their fields are.

    Writing a record writes the bytes that it was read from. When the
    comments, the resolutions, the XML and UUID boxes or the container of
    the record differ from those of these bytes, they are written in
    their place. The other fields are decided by the coded image, so a
    record that changes them cannot be written.
    """

    container: str
    """`"jp2"` for a JP2 file, `"j2k"` for a raw codestream."""

    size: tx.Tuple[int, int]
    """The size of the image area, `(width, height)`.

    The size is `(Xsiz - XOsiz, Ysiz - YOsiz)` in the terms of the
    standard.
    """

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
    """The number of resolution levels, one more than the number of
    decomposition levels, and the smallest over all components."""

    n_layers: int
    """The number of quality layers."""

    reversible: bool
    """Whether the wavelet is the reversible 5-3 one, which is lossless,
    rather than the irreversible 9-7 one."""

    comments: tx.Tuple[str, ...] = ()
    """The text comments of the main header of the codestream."""

    capture_resolution: tx.Optional[tx.Tuple[float, float]] = None
    """The capture resolution of a JP2 file (`resc` box), `(x, y)` in grid
    points per metre."""

    display_resolution: tx.Optional[tx.Tuple[float, float]] = None
    """The default display resolution of a JP2 file (`resd` box), `(x, y)`
    in grid points per metre."""

    boxes: NoRepr[tx.Tuple[tx.Tuple[bytes, bytes], ...]] = ()
    """The top-level XML, UUID and UUID info boxes of a JP2 file, as `(type,
    content)` pairs, in the order of the file, up to the codestream."""

    palette: bool = False
    """Whether a JP2 file maps its samples through a palette.

    Pillow looks the colours up when it decodes such a file, so the image
    has three channels, or four with an alpha component.
    """

    stored: NoRepr[bytes] = b""
    """The bytes of the file up to its first tile, as they were read."""

    @property
    def dtype(self) -> np.dtype:
        """The type of the samples that Pillow decodes, once unscaled.

        A component of up to 8 bits gives `uint8` (`int8` if signed), and
        one of up to 16 bits gives `uint16` (`int16`). A palette gives
        `uint8` colours.
        """
        if self.palette:
            return np.dtype(np.uint8)
        signed = any(self.signed)
        if max(self.bit_depths) <= 8:
            return np.dtype(np.int8 if signed else np.uint8)
        return np.dtype(np.int16 if signed else np.uint16)

    @property
    def samples(self) -> int:
        """The number of samples of each pixel that Pillow decodes."""
        if self.palette:
            return 4 if len(self.bit_depths) == 2 else 3
        return len(self.bit_depths)

    def level_shape(self, level: int) -> tx.Tuple[int, int]:
        """Return the size `(width, height)` of a resolution level.

        Level 0 is the full resolution. The codestream defines the size of
        level `r` as `ceil(X1 / 2**r) - ceil(X0 / 2**r)`, where `X0` and
        `X1` are the edges of the image area on the reference grid.

        Parameters
        ----------
        level : int
            The resolution level.

        Returns
        -------
        tuple of int
            The width and the height of the level.
        """
        return tuple(
            _ceil_div(offset + size, 2**level) - _ceil_div(offset, 2**level)
            for offset, size in zip(self.image_offset, self.size)
        )

    def level_start(self, level: int) -> tx.Tuple[int, int]:
        """Return the full-resolution pixel on which a level starts.

        The wavelet keeps the low-pass samples at the even positions of
        the reference grid, so sample `u` of a level reduced by
        `f = 2**r` lies on reference coordinate `f * u`, and the first
        sample of the level is `u = ceil(X0 / f)`. The result is zero
        unless the image area has an offset that is not a multiple of
        `f`.

        Parameters
        ----------
        level : int
            The resolution level.

        Returns
        -------
        tuple of int
            The position `(x, y)` of the first pixel of the level, in
            full-resolution pixels.
        """
        factor = 2**level
        return tuple(
            factor * _ceil_div(offset, factor) - offset
            for offset in self.image_offset
        )

    def stored_shape(self, level: int) -> tx.Tuple[int, ...]:
        """Return the shape of a level as Pillow decodes it.

        The shape is `(rows, columns)` for one sample per pixel, and
        `(rows, columns, samples)` otherwise.

        Parameters
        ----------
        level : int
            The resolution level.

        Returns
        -------
        tuple of int
            The shape of the decoded level.
        """
        width, height = self.level_shape(level)
        if self.samples == 1:
            return (height, width)
        return (height, width, self.samples)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a JPEG 2000 header.

        The header is parsed, and the position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, which is read from its current position.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no JPEG 2000
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        try:
            with preserve_position(file):
                _read_record(file)
        except Exception:
            if error:
                if error is True:
                    error = SnifferContentError
                raise error(
                    "This content is neither a JP2 file nor a JPEG 2000 "
                    "codestream."
                ) from None
            return Confidence.NO
        return Confidence.LIKELY

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a JPEG 2000 header.

        Parameters
        ----------
        data : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no JPEG 2000
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(bytes(data)), error=error)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "Jpeg2000Raw":
        """Read the header of a JP2 file or a codestream from a stream.

        The stream is read up to the first tile of the codestream, and its
        position is restored. The image is not decoded.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        Jpeg2000Raw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream is neither a JP2 file nor a codestream, or is
            corrupt.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} takes no option, but was given "
                f"{', '.join(sorted(kwargs))}."
            )
        with preserve_position(file):
            return _read_record(file)

    # --- writing ------------------------------------------------------

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the file, up to its first tile.

        The bytes that the record was read from are returned as they are
        when they say what the record says. Otherwise the comments, the
        resolutions, the XML and UUID boxes and the container of the
        record are written in place of theirs.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        bytes
            The header.

        Raises
        ------
        TypeError
            If an option is given.
        WriterError
            If the record was not read from a file, or changes a field
            that the coded image decides.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} takes no option, but was given "
                f"{', '.join(sorted(kwargs))}."
            )
        if not self.stored:
            raise WriterError(
                "This JPEG 2000 record was not read from a file, so it has "
                "no codestream header to write."
            )
        if _read_record(BytesIO(self.stored)) == self:
            return self.stored
        written = rewrite(self.stored, self)
        check_layout(self, _read_record(BytesIO(written)))
        return written


def read_record(content: bytes) -> Jpeg2000Raw:
    """Read the record of a JPEG 2000 file from its bytes.

    Parameters
    ----------
    content : bytes
        The file, or its beginning up to its first tile.

    Returns
    -------
    Jpeg2000Raw
        The record.

    Raises
    ------
    ParserContentError
        If the bytes are neither a JP2 file nor a codestream.
    """
    return _read_record(BytesIO(content))


def check_layout(record: Jpeg2000Raw, written: Jpeg2000Raw) -> None:
    """Check that a record and the file written from it agree.

    The coded image decides the size, the tiles, the components, the
    levels, the layers, the wavelet and the palette, so a record whose
    fields differ from those of the written file asks for something that
    cannot be written.

    Parameters
    ----------
    record : Jpeg2000Raw
        The record that was asked for.
    written : Jpeg2000Raw
        The record of the written file.

    Raises
    ------
    WriterError
        If a field that the coded image decides differs.
    """
    changed = [
        name
        for name in _LAYOUT
        if getattr(record, name) != getattr(written, name)
    ]
    if changed:
        raise WriterError(
            f"The fields {changed} of a JPEG 2000 record are decided by "
            f"the coded image, so they cannot be changed without encoding "
            f"the image again."
        )


def with_container(record: Jpeg2000Raw, container: str) -> Jpeg2000Raw:
    """Return a copy of a record in a container.

    A raw codestream stores no resolution and no box, so a record in the
    `"j2k"` container has none.

    Parameters
    ----------
    record : Jpeg2000Raw
        The record.
    container : {"jp2", "j2k"}
        The container.

    Returns
    -------
    Jpeg2000Raw
        The copy, or `record` itself when it is already a record of a JP2
        file in the JP2 container.
    """
    if container == "jp2" and record.container == "jp2":
        return record
    if container == "j2k":
        return replace(
            record,
            container=container,
            capture_resolution=None,
            display_resolution=None,
            boxes=(),
        )
    return replace(record, container=container)


# ----------------------------------------------------------------------
#   PARSING
# ----------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    """Return the ceiling of `a / b` for integers."""
    return -(-int(a) // int(b))


def _read_record(file: tx.IO) -> Jpeg2000Raw:
    """Parse the header of a JP2 file or a codestream from a stream.

    The stream is left after the first marker that ends the main header
    of the codestream.

    Raises
    ------
    ParserContentError
        If the stream is neither a JP2 file nor a codestream, or is
        corrupt.
    """
    start = file.tell()
    head = file.read(12)
    file.seek(start)
    try:
        container = container_of(head)
        if container == "jp2":
            found = _parse_jp2(file)
        elif container == "j2k":
            found = {"container": "j2k"}
        else:
            raise ParserContentError(
                "This is neither a JP2 file nor a JPEG 2000 codestream."
            )
        found.update(_parse_codestream(file))
    except (struct.error, IndexError) as e:
        raise ParserContentError(f"This JPEG 2000 file is corrupt: {e}") from e
    end = file.tell()
    file.seek(start)
    found["stored"] = read_exact(file, end - start)
    return Jpeg2000Raw(**found)


def _parse_jp2(file: tx.IO) -> tx.Dict[str, tx.Any]:
    """Read the boxes of a JP2 file that the record keeps.

    The stream is left at the start of the codestream.
    """
    found: tx.Dict[str, tx.Any] = {"container": "jp2"}
    kept = []
    codestream = None
    for kind, _, content, end in walk_boxes(file):
        if kind == b"jp2c":
            codestream = content
            break
        if kind == b"jp2h":
            found["palette"] = _parse_jp2_header(file, end, found)
        elif kind in KEPT_BOXES and end is not None:
            kept.append((kind, read_exact(file, end - content)))
        if end is None:
            break
    if codestream is None:
        raise ParserContentError("This JP2 file has no codestream box.")
    found["boxes"] = tuple(kept)
    file.seek(codestream)
    return found


def _parse_jp2_header(
    file: tx.IO, end: tx.Optional[int], found: tx.Dict[str, tx.Any]
) -> bool:
    """Read the header box of a JP2 file.

    The resolutions are stored in `found`, and the function returns
    whether Pillow looks the samples up in a palette. Pillow does so for
    one component of up to 9 bits or for two components, when the
    entries of the palette have up to 9 bits.
    """
    grey = False
    palette = False
    for kind, _, content, sub_end in walk_boxes(file, end):
        if sub_end is None:
            break
        if kind == b"ihdr":
            data = read_exact(file, sub_end - content)
            _, _, count, depth = struct.unpack(">IIHB", data[:11])
            grey = (count == 1 and (depth & 0x7F) <= 8) or count == 2
        elif kind == b"pclr":
            data = read_exact(file, sub_end - content)
            columns = data[2]
            palette = grey and max(data[3 : 3 + columns]) <= 8
        elif kind == b"res ":
            for res, _, res_content, res_end in walk_boxes(file, sub_end):
                value = read_resolution(
                    read_exact(file, res_end - res_content)
                )
                if res == b"resc":
                    found["capture_resolution"] = value
                elif res == b"resd":
                    found["display_resolution"] = value
    return palette


def _parse_codestream(file: tx.IO) -> tx.Dict[str, tx.Any]:
    """Read the main header of a codestream, up to its first tile."""
    if read_exact(file, 2) != J2K_SIGNATURE[:2]:
        raise ParserContentError("This is not a JPEG 2000 codestream.")
    found: tx.Dict[str, tx.Any] = {}
    levels = []
    comments = []
    while True:
        marker = read_exact(file, 2)
        if marker[0] != 0xFF:
            raise ParserContentError("This JPEG 2000 codestream is corrupt.")
        code = marker[1]
        if code in (SOT, SOD):
            break
        (length,) = struct.unpack(">H", read_exact(file, 2))
        segment = read_exact(file, length - 2)
        if code == SIZ:
            (_, xs, ys, xo, yo, xt, yt, xto, yto, n) = struct.unpack(
                ">HIIIIIIIIH", segment[:36]
            )
            components = [segment[36 + 3 * i : 39 + 3 * i] for i in range(n)]
            found["size"] = (xs - xo, ys - yo)
            found["image_offset"] = (xo, yo)
            found["tile_size"] = (xt, yt)
            found["tile_offset"] = (xto, yto)
            found["bit_depths"] = tuple((c[0] & 0x7F) + 1 for c in components)
            found["signed"] = tuple(bool(c[0] & 0x80) for c in components)
            found["subsampling"] = tuple((c[1], c[2]) for c in components)
        elif code == COD:
            found["n_layers"] = struct.unpack(">H", segment[2:4])[0]
            levels.append(segment[5] + 1)
            found["reversible"] = segment[9] == 1
        elif code == COC:
            # The index of the component takes one byte, or two when there
            # are more than 256 components.
            wide = len(found.get("bit_depths", ())) > 256
            levels.append(segment[3 if wide else 2] + 1)
        elif code == COM and len(segment) > 2:
            if struct.unpack(">H", segment[:2])[0] == 1:
                comments.append(segment[2:].decode("latin-1"))
    if "size" not in found or not levels:
        raise ParserContentError(
            "This JPEG 2000 codestream has no SIZ or COD marker."
        )
    found["n_levels"] = min(levels)
    found["comments"] = tuple(comments)
    return found
