"""The record of a TrakEM2 project: its XML file, as parsed."""

__all__ = ["TrakEM2Raw"]

# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, NoRepr

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
from ._xml import _Element, attribute_span, document_name, parse_xml

_ROOT = "trakem2"
"""The name of the root element of a TrakEM2 project file."""

_TRANSFORMS = (
    "ict_transform",
    "iict_transform",
    "ict_transform_list",
    "iict_transform_list",
)
"""The elements that hold the coordinate transform of a patch."""

_Floats = tx.Tuple[float, ...]
_Ints = tx.Tuple[int, ...]


class TrakEM2Raw(Magic, BinaryFileReader, BinaryFileWriter, frozen=True):
    """Record of a TrakEM2 project, which holds its XML file.

    TrakEM2 is the Fiji plugin for serial-section electron microscopy. A
    project is an XML file that lists layers, one for each section, and
    in each layer the patches, which are the images of the section, each
    placed in the plane of its layer by an affine. The record holds the
    bytes of the XML file, so that a file that was read and not changed
    is written back byte for byte, and what brainhops reads from it: the
    patches, with their affine, their layer and the calibration of their
    layer set. The conventions are described in
    [`brainhops.io.transformations.trakem2`][].

    The record is held by
    [`TrakEM2Metadata`][brainhops.io.transformations.trakem2.TrakEM2Metadata].
    A record is frozen and never changed in place. Changing the affine of
    a patch gives a new record, by [`with_transform`][], whose bytes
    differ from the old ones in the value of one `transform` attribute
    only.
    """

    class Calibration(Magic, frozen=True):
        """The `<t2_calibration>` of a layer set.

        TrakEM2 writes the fields of an ImageJ calibration
        (`LayerSet.exportXML` in `ini/trakem2/display/LayerSet.java`,
        TrakEM2 commit cdd8b8f). The coordinates of a layer set are in
        pixels, and TrakEM2 multiplies x by `pixel_width`, y by
        `pixel_height` and the z of a layer by `pixel_width`, not by
        `pixel_depth`, to calibrate them (`Layer.getCalibratedZ` and
        `Tree.java`). The origins are written but are not used there.
        """

        pixel_width: float = 1.0
        """The `pixelWidth` attribute."""

        pixel_height: float = 1.0
        """The `pixelHeight` attribute."""

        pixel_depth: float = 1.0
        """The `pixelDepth` attribute."""

        x_origin: float = 0.0
        """The `xOrigin` attribute."""

        y_origin: float = 0.0
        """The `yOrigin` attribute."""

        z_origin: float = 0.0
        """The `zOrigin` attribute."""

        unit: tx.Optional[str] = None
        """The `unit` attribute, such as `"nm"` or `"pixel"`."""

    class CoordinateTransform(Magic, frozen=True):
        """A coordinate transform written inside a `<t2_patch>`.

        It is kept as the file writes it: brainhops decodes only the
        affine and similarity models (see
        [`TrakEM2PatchAffine.coordinate_transform`][brainhops.io.transformations.trakem2.TrakEM2PatchAffine.coordinate_transform]),
        and the writer never changes it.
        """

        element: str
        """The name of the element, such as `"iict_transform"` or
        `"ict_transform_list"`."""

        cls: tx.Optional[str] = None
        """The `class` attribute, the Java class of the transform."""

        data: tx.Optional[str] = None
        """The `data` attribute, the parameters of the transform."""

        children: tx.Tuple["TrakEM2Raw.CoordinateTransform", ...] = ()
        """The transforms of a list, in the order of the file."""

    class Patch(Magic, frozen=True):
        """One `<t2_patch>`: an image placed in the plane of its layer."""

        oid: int
        """The `oid` attribute, the id of the patch in the project."""

        title: tx.Optional[str] = None
        """The `title` attribute, often the name of the image file."""

        transform: _Floats = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        """The six values of the `transform` attribute, in file order.

        They are the values of `java.awt.geom.AffineTransform.getMatrix`,
        which are `m00 m10 m01 m11 m02 m12`.
        """

        span: tx.Optional[_Ints] = None
        """Where the value of the `transform` attribute lies in the bytes.

        It covers the text between the quotes, or is `None` for a patch
        that is not read from a file.
        """

        width: tx.Optional[float] = None
        """The `width` attribute, in pixels of the patch."""

        height: tx.Optional[float] = None
        """The `height` attribute, in pixels of the patch."""

        o_width: tx.Optional[int] = None
        """The `o_width` attribute, the width of the image file."""

        o_height: tx.Optional[int] = None
        """The `o_height` attribute, the height of the image file."""

        ct_id: tx.Optional[int] = None
        """The `ct_id` attribute, set when the patch has a coordinate
        transform, which is stored in a separate file unless it is also
        written inside the patch."""

        coordinate_transform: tx.Optional["TrakEM2Raw.CoordinateTransform"] = (
            None
        )
        """The coordinate transform written inside the patch, if any."""

        layer: tx.Optional[int] = None
        """The `oid` of the `<t2_layer>` that holds the patch."""

        z: tx.Optional[float] = None
        """The `z` attribute of the layer, in pixels of the layer set."""

        thickness: tx.Optional[float] = None
        """The `thickness` attribute of the layer, in pixels of the layer
        set."""

        calibration: tx.Optional["TrakEM2Raw.Calibration"] = None
        """The calibration of the layer set that holds the layer."""

    stored: NoRepr[bytes] = b""
    """The bytes of the XML file, as read or as last encoded."""

    patches: tx.Tuple["TrakEM2Raw.Patch", ...] = ()
    """The patches, in the order of the file."""

    # --- patches ------------------------------------------------------

    def patch(
        self, selector: tx.Optional[tx.Union[int, str]] = None
    ) -> "TrakEM2Raw.Patch":
        """Return the patch that a selector names.

        Parameters
        ----------
        selector : int or str, optional
            The `oid` of the patch when it is an integer, or its title
            when it is a string. By default, the first patch of the file.

        Returns
        -------
        TrakEM2Raw.Patch
            The patch.

        Raises
        ------
        ParserContentError
            If the record has no such patch, or several patches have the
            title.
        """
        if not self.patches:
            raise ParserContentError("This TrakEM2 project has no patch.")
        if selector is None:
            return self.patches[0]
        if isinstance(selector, str):
            found = [p for p in self.patches if p.title == selector]
            if len(found) > 1:
                oids = ", ".join(str(p.oid) for p in found)
                raise ParserContentError(
                    f"Several patches of this TrakEM2 project have the title "
                    f"{selector!r}, with oids {oids}. Select one by its oid."
                )
        else:
            found = [p for p in self.patches if p.oid == int(selector)]
        if not found:
            raise ParserContentError(
                f"This TrakEM2 project has no patch {selector!r}. Its "
                f"patches have the oids "
                f"{', '.join(str(p.oid) for p in self.patches)}."
            )
        return found[0]

    # --- encoding -----------------------------------------------------

    def with_transform(self, oid: int, values: _Floats) -> "TrakEM2Raw":
        """Return a record whose patch has another affine.

        The new bytes are the bytes of this record, in which the value of
        the `transform` attribute of the patch is written again, so
        everything else in the file is kept byte for byte. The values are
        written as `matrix(a,b,c,d,e,f)` in the shortest form that reads
        back as the same double, laid out as Java's `Double.toString`
        lays it out, which is how TrakEM2 writes them
        (`Displayable.exportXML` in `ini/trakem2/display/Displayable.java`,
        TrakEM2 commit cdd8b8f). Values equal to those of the file keep
        this record.

        Parameters
        ----------
        oid : int
            The `oid` of the patch.
        values : tuple of float
            The six values, in the order `m00 m10 m01 m11 m02 m12`.

        Returns
        -------
        TrakEM2Raw
            The new record.

        Raises
        ------
        WriterError
            If the record has no bytes or no such patch, or if its bytes
            are not in an encoding based on ASCII.
        """
        values = tuple(float(v) for v in values)
        try:
            patch = self.patch(int(oid))
        except ParserContentError as error:
            raise WriterError(str(error)) from error
        if patch.transform == values:
            return self
        if patch.span is None or not self.stored:
            raise WriterError(
                "This TrakEM2 record holds no file, so the affine of its "
                "patch cannot be written into it."
            )
        if not _ascii_based(self.stored):
            raise WriterError(
                "This TrakEM2 project is not in an encoding based on ASCII, "
                "such as ISO-8859-1 or UTF-8, so brainhops cannot change it "
                "in place."
            )
        text = "matrix(" + ",".join(_java_double(v) for v in values) + ")"
        start, stop = patch.span
        content = self.stored[:start] + text.encode("ascii") + self.stored[stop:]
        return TrakEM2Raw.from_bytes(content)

    # --- reading and writing ------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the document type of its XML.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` for a TrakEM2 project, and `NO` otherwise.
        """
        with preserve_position(file):
            head = file.read(4096)
        return cls.sniff_bytes(head, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes from the document type of their XML.

        TrakEM2 names the document type `trakem2_` followed by the type
        of the root of its template, such as `trakem2_anything`, and the
        root element `<trakem2>` (`Loader.writeXMLTo` in
        `ini/trakem2/persistence/Loader.java` and `Project.getDocType`,
        TrakEM2 commit cdd8b8f). A file without a document type is
        recognized by its root element.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` for a TrakEM2 project, and `NO` otherwise.
        """
        name = document_name(bytes(content))
        if name is not None and (name == _ROOT or name.startswith("trakem2_")):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("The content is not a TrakEM2 project.")
        return Confidence.NO

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "TrakEM2Raw":
        """Read the record from a binary stream of an XML file.

        Parameters
        ----------
        file : BinaryIO
            The stream, read from its position to its end, which is
            restored.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        TrakEM2Raw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not a TrakEM2 project.
        """
        _no_options(kwargs)
        with preserve_position(file):
            content = file.read()
        return _parse(bytes(content))

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "TrakEM2Raw":
        """Read the record from the bytes of an XML file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        TrakEM2Raw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not a TrakEM2 project.
        """
        _no_options(kwargs)
        return _parse(bytes(content))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the XML file.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, which are refused.

        Returns
        -------
        bytes
            The bytes that the record holds.

        Raises
        ------
        TypeError
            If an option is given.
        """
        _no_options(kwargs)
        return self.stored


# ----------------------------------------------------------------------
#   PARSING
# ----------------------------------------------------------------------
# The elements and attributes are those that TrakEM2 writes and reads at
# commit cdd8b8f: `Displayable.exportXML` and the `Displayable`
# constructor for `transform`, `Patch.exportXML` for the patch,
# `Layer.exportXML` for `z` and `thickness`, `LayerSet.exportXML` and
# `LayerSet.restoreCalibration` for the calibration, and
# `TMLHandler.makeCoordinateTransform` for the coordinate transforms.


def _parse(content: bytes) -> TrakEM2Raw:
    """Parse the bytes of a TrakEM2 project into a record.

    Raises
    ------
    ParserContentError
        If the content is not a TrakEM2 project, or a value cannot be
        read.
    """
    root = parse_xml(content)
    if root.name != _ROOT:
        raise ParserContentError(
            f"A TrakEM2 project has a <{_ROOT}> root element, and this file "
            f"has <{root.name}>."
        )
    patches: tx.List[TrakEM2Raw.Patch] = []
    try:
        _collect(content, root, None, None, patches)
    except (TypeError, ValueError) as error:
        raise ParserContentError(
            f"A value of this TrakEM2 project cannot be read: {error}"
        ) from error
    return TrakEM2Raw(stored=content, patches=tuple(patches))


def _collect(
    content: bytes,
    element: _Element,
    layer: tx.Optional[_Element],
    calibration: tx.Optional[TrakEM2Raw.Calibration],
    patches: tx.List[TrakEM2Raw.Patch],
) -> None:
    """Collect the patches under an element, in the order of the file.

    A patch takes the nearest layer that holds it and the calibration of
    the nearest layer set that holds that layer.
    """
    if element.name == "t2_layer_set":
        for child in element.children:
            if child.name == "t2_calibration":
                calibration = _calibration(child)
    elif element.name == "t2_layer":
        layer = element
    elif element.name == "t2_patch":
        patches.append(_patch(content, element, layer, calibration))
        return
    for child in element.children:
        _collect(content, child, layer, calibration, patches)


def _patch(
    content: bytes,
    element: _Element,
    layer: tx.Optional[_Element],
    calibration: tx.Optional[TrakEM2Raw.Calibration],
) -> TrakEM2Raw.Patch:
    """Read a `<t2_patch>` and the layer that holds it."""
    attrs = element.attrs
    oid = _integer(attrs.get("oid"))
    if oid is None:
        raise ValueError("a <t2_patch> has no oid")
    text = attrs.get("transform")
    if text is None:
        # TrakEM2 logs a warning and keeps the identity.
        transform, span = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), None
    else:
        transform = _matrix(text, oid)
        span = attribute_span(content, element, "transform")
    transforms = [c for c in element.children if c.name in _TRANSFORMS]
    layer_attrs = {} if layer is None else layer.attrs
    return TrakEM2Raw.Patch(
        oid=oid,
        title=attrs.get("title"),
        transform=transform,
        span=span,
        width=_number(attrs.get("width")),
        height=_number(attrs.get("height")),
        o_width=_integer(attrs.get("o_width")),
        o_height=_integer(attrs.get("o_height")),
        ct_id=_integer(attrs.get("ct_id")),
        coordinate_transform=(
            _coordinate_transform(transforms[0]) if transforms else None
        ),
        layer=_integer(layer_attrs.get("oid")),
        z=_number(layer_attrs.get("z")),
        thickness=_number(layer_attrs.get("thickness")),
        calibration=calibration,
    )


def _matrix(text: str, oid: int) -> _Floats:
    """Read the six values of a `transform` attribute.

    TrakEM2 reads the numbers between the first `(` and the last `)`,
    separated by commas, with `Double.parseDouble`.
    """
    values = text[text.find("(") + 1 : text.rfind(")")].split(",")
    if len(values) != 6:
        raise ValueError(
            f"the transform of patch {oid} has {len(values)} values instead "
            f"of 6: {text!r}"
        )
    return tuple(float(v) for v in values)


def _calibration(element: _Element) -> TrakEM2Raw.Calibration:
    """Read a `<t2_calibration>`, whose missing fields keep their default."""
    attrs = element.attrs
    names = {
        "pixel_width": "pixelwidth",
        "pixel_height": "pixelheight",
        "pixel_depth": "pixeldepth",
        "x_origin": "xorigin",
        "y_origin": "yorigin",
        "z_origin": "zorigin",
    }
    values = {
        field: float(attrs[name])
        for field, name in names.items()
        if name in attrs
    }
    return TrakEM2Raw.Calibration(unit=attrs.get("unit"), **values)


def _coordinate_transform(
    element: _Element,
) -> TrakEM2Raw.CoordinateTransform:
    """Read a coordinate transform, and the transforms of a list."""
    return TrakEM2Raw.CoordinateTransform(
        element=element.name,
        cls=element.attrs.get("class"),
        data=element.attrs.get("data"),
        children=tuple(
            _coordinate_transform(child)
            for child in element.children
            if child.name in _TRANSFORMS
        ),
    )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _number(text: tx.Optional[str]) -> tx.Optional[float]:
    """Read a number, or `None` when the attribute is missing."""
    return None if text is None else float(text)


def _integer(text: tx.Optional[str]) -> tx.Optional[int]:
    """Read an integer, or `None` when the attribute is missing."""
    return None if text is None else int(text)


def _java_double(value: float) -> str:
    """Return a double as Java's `Double.toString` writes it.

    The digits are the shortest that read back as the same double. Java
    writes a value whose magnitude is at least 1e-3 and less than 1e7 in
    decimal notation with at least one digit after the point, and any
    other value in scientific notation, such as `1.0E-5`.

    Raises
    ------
    WriterError
        If the value is not finite.
    """
    if not math.isfinite(value):
        raise WriterError(
            f"A TrakEM2 transform holds finite values, and this one holds "
            f"{value}."
        )
    if value == 0 or 1e-3 <= abs(value) < 1e7:
        return np.format_float_positional(value, unique=True, trim="0")
    text = np.format_float_scientific(
        value, unique=True, trim="0", exp_digits=1
    )
    return text.replace("e+", "E").replace("e", "E")


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse options, which the record does not take.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"A TrakEM2 project file takes no option, but was given "
            f"{', '.join(sorted(kwargs))}."
        )


def _ascii_based(content: bytes) -> bool:
    """Tell whether a document starts like XML in an ASCII-based encoding.

    Bytes that brainhops writes into a file are ASCII, so they fit any
    such encoding, but not UTF-16 or UTF-32.
    """
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]
    return content.lstrip(b" \t\r\n").startswith(b"<")
