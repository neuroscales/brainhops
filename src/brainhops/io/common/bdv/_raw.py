"""The record of a BigDataViewer dataset: its XML file, as parsed."""

__all__ = ["BdvRaw"]

# stdlib
import os

# dependencies
import typing_extensions as tx
from bagof.magic import Magic, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base._utils_files import sibling
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)

# this format
from ._xml import (
    _Element,
    indent_before,
    newline_of,
    parse_xml,
    render_registration,
    render_transform,
    root_name,
)

_ROOT = "SpimData"
"""The name of the root element of a BigDataViewer XML file."""

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
"""The identity, as the twelve values of a `<ViewTransform>`."""

_UNIT = b"  "
"""One level of indentation, as JDOM's pretty format writes it."""

_Floats = tx.Tuple[float, ...]
_Ints = tx.Tuple[int, ...]


class BdvRaw(Magic, BinaryFileReader, BinaryFileWriter, frozen=True):
    """Record of a BigDataViewer dataset, which holds its XML file.

    A BigDataViewer dataset is an XML file, often named `dataset.xml`,
    that describes the views of an acquisition and how they are placed in
    a common space, and an HDF5 file that holds their voxels. BigStitcher
    keeps its registrations in the same XML file. The record holds the
    bytes of the XML file, so that a file that was read and not changed
    is written back byte for byte, and what brainhops reads from it: the
    setups, the time points, the missing views, where the HDF5 file is,
    and the registration of each view. The layout is described in
    [`brainhops.io.common.bdv`][].

    The record is held by
    [`BdvMetadata`][brainhops.io.common.bdv.BdvMetadata]. A record is
    frozen and never changed in place. Changing the registration of a
    view gives a new record, by [`with_registration`][], whose bytes
    differ from the old ones in that `<ViewRegistration>` element only.

    An image read from the dataset also needs what the HDF5 file says
    about the resolution levels of each setup. The image reader reads it
    and stores it in `pyramids`, which is `None` in a record that was
    read from the XML file alone.
    """

    class Setup(Magic, frozen=True):
        """One `<ViewSetup>`: a channel, an angle or a tile of the
        acquisition."""

        id: int
        """The id of the setup, which the views and the HDF5 groups use."""

        name: tx.Optional[str] = None
        """The name of the setup, if the file gives one."""

        size: tx.Optional[_Ints] = None
        """The size of the full-resolution image, in the order x, y, z."""

        voxel_size: tx.Optional[_Floats] = None
        """The size of a voxel along x, y and z, in `unit`."""

        unit: tx.Optional[str] = None
        """The unit of the voxel size, as the file writes it."""

    class Transform(Magic, frozen=True):
        """One `<ViewTransform>` of a view registration."""

        type: str = "affine"
        """The type of the transform: `"affine"` or `"generic"`."""

        name: tx.Optional[str] = None
        """The text of the `<Name>` child, such as `"calibration"`."""

        affine: tx.Optional[_Floats] = _IDENTITY
        """The first three rows of the affine matrix, row after row.

        It is `None` for a generic transform, which a Java class reads
        and which brainhops cannot read.
        """

        span: tx.Optional[_Ints] = None
        """Where the element lies in the bytes of the record.

        It is the offset of its first byte and the offset just after its
        last byte, or `None` for a transform that is not read from a
        file.
        """

    class Registration(Magic, frozen=True):
        """One `<ViewRegistration>`: the transforms of one view."""

        timepoint: int
        """The id of the time point of the view."""

        setup: int
        """The id of the setup of the view."""

        transforms: tx.Tuple["BdvRaw.Transform", ...] = ()
        """The transforms, in the order of the file.

        The last transform is applied first, so the map from the voxels of
        the view to the global space is the product of their matrices in
        this order.
        """

        span: tx.Optional[_Ints] = None
        """Where the element lies in the bytes of the record."""

    class Pyramid(Magic, frozen=True):
        """The resolution levels of one setup, as the HDF5 file gives them."""

        setup: int
        """The id of the setup."""

        resolutions: tx.Tuple[_Floats, ...] = ()
        """The downsampling factors of each level along x, y and z."""

        subdivisions: tx.Tuple[_Ints, ...] = ()
        """The chunk shape of each level along x, y and z."""

        data_type: tx.Optional[str] = None
        """The `dataType` attribute of the setup group, such as `"uint8"`.

        It is `None` in a file written before the attribute existed, whose
        voxels are 16-bit unsigned integers stored as signed ones.
        """

    stored: NoRepr[bytes] = b""
    """The bytes of the XML file, as read or as last encoded."""

    version: tx.Optional[str] = None
    """The `version` attribute of the root element."""

    base_path: str = "."
    """The text of `<BasePath>`, against which the HDF5 path resolves."""

    base_path_relative: bool = True
    """Whether `<BasePath>` is relative to the directory of the XML file."""

    loader: tx.Optional[str] = None
    """The `format` attribute of `<ImageLoader>`, such as `"bdv.hdf5"`."""

    hdf5: tx.Optional[str] = None
    """The text of the `<hdf5>` child of `<ImageLoader>`, if any."""

    hdf5_relative: bool = False
    """Whether the HDF5 path is relative to the base path."""

    setups: tx.Tuple["BdvRaw.Setup", ...] = ()
    """The setups, in the order of the file."""

    timepoints: _Ints = ()
    """The ids of the time points, in increasing order."""

    missing: tx.Tuple[_Ints, ...] = ()
    """The views that have no data, as pairs of time point and setup."""

    registrations: tx.Tuple["BdvRaw.Registration", ...] = ()
    """The registrations of the views, in the order of the file."""

    pyramids: tx.Optional[tx.Tuple["BdvRaw.Pyramid", ...]] = None
    """The resolution levels of each setup, read from the HDF5 file.

    It is `None` when only the XML file was read.
    """

    # --- views --------------------------------------------------------

    def setup(self, setup: int) -> "BdvRaw.Setup":
        """Return the setup of a given id.

        Parameters
        ----------
        setup : int
            The id of the setup.

        Returns
        -------
        BdvRaw.Setup
            The setup.

        Raises
        ------
        ParserContentError
            If the dataset has no setup of that id.
        """
        for item in self.setups:
            if item.id == setup:
                return item
        raise ParserContentError(
            f"This BigDataViewer dataset has no setup {setup}. Its setups "
            f"are {_ids(item.id for item in self.setups)}."
        )

    def registration(
        self, timepoint: int, setup: int
    ) -> "BdvRaw.Registration":
        """Return the registration of a view.

        Parameters
        ----------
        timepoint : int
            The id of the time point of the view.
        setup : int
            The id of the setup of the view.

        Returns
        -------
        BdvRaw.Registration
            The registration.

        Raises
        ------
        ParserContentError
            If the dataset has no registration for that view.
        """
        for item in self.registrations:
            if item.timepoint == timepoint and item.setup == setup:
                return item
        raise ParserContentError(
            f"This BigDataViewer dataset has no registration for the view "
            f"of time point {timepoint} and setup {setup}."
        )

    def pyramid(self, setup: int) -> "BdvRaw.Pyramid":
        """Return the resolution levels of a setup.

        Parameters
        ----------
        setup : int
            The id of the setup.

        Returns
        -------
        BdvRaw.Pyramid
            The resolution levels.

        Raises
        ------
        ParserContentError
            If the record holds no levels for that setup, because the HDF5
            file was not read or does not describe the setup.
        """
        for item in self.pyramids or ():
            if item.setup == setup:
                return item
        raise ParserContentError(
            f"The HDF5 file of this BigDataViewer dataset does not describe "
            f"the resolution levels of setup {setup}, or it was not read."
        )

    def view(
        self,
        setup: tx.Optional[int] = None,
        timepoint: tx.Optional[int] = None,
    ) -> _Ints:
        """Return the ids of the setup and of the time point of a view.

        A setup or a time point that is not given is the first one, in the
        order of the ids, which is setup 0 and time point 0 in a dataset
        that BigDataViewer or BigStitcher wrote.

        Parameters
        ----------
        setup : int, optional
            The id of the setup.
        timepoint : int, optional
            The id of the time point.

        Returns
        -------
        setup : int
            The id of the setup.
        timepoint : int
            The id of the time point.

        Raises
        ------
        ParserContentError
            If the dataset has no such setup or time point, or none at all.
        """
        setups = sorted(item.id for item in self.setups)
        if setup is None:
            if not setups:
                raise ParserContentError(
                    "This BigDataViewer dataset has no setup."
                )
            setup = setups[0]
        elif int(setup) not in setups:
            self.setup(int(setup))
        if timepoint is None:
            if not self.timepoints:
                raise ParserContentError(
                    "This BigDataViewer dataset has no time point."
                )
            timepoint = self.timepoints[0]
        elif int(timepoint) not in self.timepoints:
            raise ParserContentError(
                f"This BigDataViewer dataset has no time point {timepoint}. "
                f"Its time points are {_ids(self.timepoints)}."
            )
        return int(setup), int(timepoint)

    def hdf5_filename(self, xml_filename: path.FilenameLike) -> path.Path:
        """Return the path of the HDF5 file of the dataset.

        The path resolves as BigDataViewer resolves it
        (`XmlIoAbstractSpimData.loadBasePathURI` and
        `XmlIoHdf5ImageLoader.fromXml`): `<BasePath>` against the
        directory of the XML file when it is relative, and the `<hdf5>`
        path against the base path when it is relative. A path is relative
        only when its `type` attribute says `"relative"`.

        Parameters
        ----------
        xml_filename : path-like
            The path of the XML file.

        Returns
        -------
        path.Path
            The path of the HDF5 file.

        Raises
        ------
        ParserContentError
            If the dataset does not name an HDF5 file.
        """
        if self.hdf5 is None:
            raise ParserContentError(
                f"This BigDataViewer dataset names no HDF5 file: its image "
                f"loader is {self.loader!r}, and brainhops reads the "
                f"'bdv.hdf5' loader only."
            )
        if not self.hdf5_relative:
            return path.Path(self.hdf5)
        if self.base_path_relative:
            base = sibling(xml_filename, self.base_path)
        else:
            base = path.Path(self.base_path)
        return path.Path(os.path.normpath(os.fspath(base / self.hdf5)))

    # --- encoding -----------------------------------------------------

    def with_registration(
        self,
        timepoint: int,
        setup: int,
        transforms: tx.Sequence["BdvRaw.Transform"],
    ) -> "BdvRaw":
        """Return a record whose view has other transforms.

        The new bytes are the bytes of this record, in which the
        `<ViewRegistration>` element of the view is written again, so
        everything else in the file is kept byte for byte. A transform
        that has a `span` is a transform of this record, and its element
        is copied as it is, with its name and the exact text of its
        values. Any other transform is written in the layout of
        BigDataViewer, with the indentation and the line separator of the
        file. A record without bytes, such as a new record, is written as
        a new file with that single view.

        Parameters
        ----------
        timepoint : int
            The id of the time point of the view.
        setup : int
            The id of the setup of the view.
        transforms : sequence of BdvRaw.Transform
            The transforms, in the order of the file.

        Returns
        -------
        BdvRaw
            The new record, which keeps the `pyramids` of this record.

        Raises
        ------
        WriterError
            If the bytes of the file do not start like an XML file in an
            encoding based on ASCII, or if the view has no registration.
        """
        if not self.stored:
            content = _new_file(timepoint, setup, transforms)
            return BdvRaw.from_bytes(content)
        if not _ascii_based(self.stored):
            raise WriterError(
                "This BigDataViewer XML file is not in an encoding based on "
                "ASCII, such as UTF-8, so brainhops cannot change it in "
                "place."
            )
        try:
            old = self.registration(timepoint, setup)
        except ParserContentError as error:
            raise WriterError(str(error)) from error
        newline = newline_of(self.stored)
        indent = indent_before(self.stored, old.span[0])
        unit = _UNIT
        for item in old.transforms:
            inner = indent_before(self.stored, item.span[0])
            if inner.startswith(indent) and len(inner) > len(indent):
                unit = inner[len(indent) :]
                break
        elements = []
        for item in transforms:
            if item.span is not None:
                elements.append(self.stored[item.span[0] : item.span[1]])
            else:
                elements.append(
                    render_transform(
                        item.name, item.affine, indent + unit, unit, newline
                    )
                )
        element = render_registration(
            timepoint, setup, elements, indent, unit, newline
        )
        start, stop = old.span
        content = self.stored[:start] + element + self.stored[stop:]
        return replace(BdvRaw.from_bytes(content), pyramids=self.pyramids)

    # --- reading and writing ------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the root element of its XML.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` when the root element is `<SpimData>`, and `NO`
            otherwise.
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
        """Score bytes from the root element of their XML.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` when the root element is `<SpimData>`, and `NO`
            otherwise.
        """
        if root_name(bytes(content)) == _ROOT:
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("The content is not a BigDataViewer XML file.")
        return Confidence.NO

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "BdvRaw":
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
        BdvRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not a BigDataViewer XML file.
        """
        _no_options(kwargs)
        with preserve_position(file):
            content = file.read()
        return _parse(bytes(content))

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "BdvRaw":
        """Read the record from the bytes of an XML file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        BdvRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the content is not a BigDataViewer XML file.
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
# The tags and attributes are those of `mpicbg/spim/data/XmlKeys.java`
# in spimdata (commit 84728ff), and the image loader is that of
# `bdv/img/hdf5/XmlIoHdf5ImageLoader.java` in bigdataviewer-core (commit
# dfe1a66).


def _parse(content: bytes) -> BdvRaw:
    """Parse the bytes of a BigDataViewer XML file into a record.

    Raises
    ------
    ParserContentError
        If the content is not a BigDataViewer XML file, or a value cannot
        be read.
    """
    root = parse_xml(content)
    if root.name != _ROOT:
        raise ParserContentError(
            f"A BigDataViewer XML file has a <{_ROOT}> root element, and "
            f"this file has <{root.name}>."
        )
    sequence = _required(root, "SequenceDescription")
    registrations = _required(root, "ViewRegistrations")
    base = root.child("BasePath")
    loader = sequence.child("ImageLoader")
    hdf5 = None if loader is None else loader.child("hdf5")
    try:
        return BdvRaw(
            stored=content,
            version=root.attrs.get("version"),
            base_path="." if base is None else base.text,
            base_path_relative=base is None or _relative(base),
            loader=None if loader is None else loader.attrs.get("format"),
            hdf5=None if hdf5 is None else hdf5.text,
            hdf5_relative=hdf5 is not None and _relative(hdf5),
            setups=_setups(sequence.child("ViewSetups")),
            timepoints=_timepoints(_required(sequence, "Timepoints")),
            missing=_missing(sequence.child("MissingViews")),
            registrations=tuple(
                _registration(element)
                for element in registrations.all("ViewRegistration")
            ),
        )
    except (TypeError, ValueError) as error:
        raise ParserContentError(
            f"A value of this BigDataViewer XML file cannot be read: {error}"
        ) from error


def _required(parent: _Element, name: str) -> _Element:
    """Return a child that BigDataViewer requires.

    Raises
    ------
    ParserContentError
        If the child is missing.
    """
    child = parent.child(name)
    if child is None:
        raise ParserContentError(
            f"This BigDataViewer XML file has no <{name}> element in "
            f"<{parent.name}>."
        )
    return child


def _relative(element: _Element) -> bool:
    """Tell whether a path element is relative, as `XmlHelpers.loadPath`
    decides: only when its `type` attribute is `"relative"`."""
    return element.attrs.get("type") == "relative"


def _setups(element: tx.Optional[_Element]) -> tx.Tuple[BdvRaw.Setup, ...]:
    """Read the `<ViewSetup>` elements of `<ViewSetups>`."""
    if element is None:
        return ()
    setups = []
    for item in element.all("ViewSetup"):
        size = item.child_text("size")
        voxels = item.child("voxelSize")
        setups.append(
            BdvRaw.Setup(
                id=int(_required(item, "id").text),
                name=item.child_text("name"),
                size=None if size is None else _numbers(size, int),
                voxel_size=None
                if voxels is None
                else _numbers(_required(voxels, "size").text, float),
                unit=None if voxels is None else voxels.child_text("unit"),
            )
        )
    return tuple(setups)


def _timepoints(element: _Element) -> _Ints:
    """Read the ids of `<Timepoints>`, as `XmlIoTimePoints.fromXml` does.

    Raises
    ------
    ParserContentError
        If the type of the element is unknown.
    """
    kind = element.attrs.get("type")
    if kind == "range":
        first = int(_required(element, "first").text)
        last = int(_required(element, "last").text)
        ids: tx.Iterable[int] = range(first, last + 1)
    elif kind == "list":
        ids = (
            int(_required(item, "id").text)
            for item in element.all("Timepoint")
        )
    elif kind == "pattern":
        ids = _integer_pattern(_required(element, "integerpattern").text)
    else:
        raise ParserContentError(
            f"The <Timepoints> of this BigDataViewer XML file have the "
            f"unknown type {kind!r}: the types are 'range', 'list' and "
            f"'pattern'."
        )
    return tuple(sorted(set(ids)))


def _integer_pattern(text: str) -> tx.List[int]:
    """Read a pattern of integers, as `IntegerPattern.parseIntegerString`.

    The pattern lists numbers separated by commas, each of which is a
    single number, a range `a-b`, or a range with a step `a-b:s`. An
    empty pattern is the single number 0. A range whose end is below its
    start counts down.
    """
    if not text.strip():
        return [0]
    ids = []
    for entry in text.split(","):
        entry = entry.replace(" ", "").strip()
        if "-" not in entry:
            ids.append(int(entry))
            continue
        start, _, rest = entry.partition("-")
        end, _, step = rest.partition(":")
        first, last, step = int(start), int(end), int(step or 1)
        if last > first:
            ids.extend(range(first, last + 1, step))
        else:
            ids.extend(range(first, last - 1, -step))
    return ids


def _missing(element: tx.Optional[_Element]) -> tx.Tuple[_Ints, ...]:
    """Read the views of `<MissingViews>`, as pairs of time point and
    setup."""
    if element is None:
        return ()
    return tuple(
        (int(item.attrs["timepoint"]), int(item.attrs["setup"]))
        for item in element.all("MissingView")
    )


def _registration(element: _Element) -> BdvRaw.Registration:
    """Read a `<ViewRegistration>`, as `XmlIoViewRegistration.fromXml`."""
    return BdvRaw.Registration(
        timepoint=int(element.attrs["timepoint"]),
        setup=int(element.attrs["setup"]),
        transforms=tuple(
            _transform(item) for item in element.all("ViewTransform")
        ),
        span=(element.start, element.end),
    )


def _transform(element: _Element) -> BdvRaw.Transform:
    """Read a `<ViewTransform>`, as `XmlIoViewTransform.fromXml` does.

    An affine transform holds twelve values, the first three rows of its
    matrix, row after row (`XmlHelpers.getAffineTransform3D` and
    `AffineTransform3D.set(double...)` in imglib2-realtransform, commit
    ca9039d). A generic transform is kept without its values.

    Raises
    ------
    ParserContentError
        If the type is unknown, or an affine does not hold twelve values.
    """
    kind = element.attrs.get("type")
    span = (element.start, element.end)
    name = element.child_text("Name")
    if kind == "generic":
        return BdvRaw.Transform(type=kind, name=name, affine=None, span=span)
    if kind != "affine":
        raise ParserContentError(
            f"A <ViewTransform> of this BigDataViewer XML file has the "
            f"unknown type {kind!r}: the types are 'affine' and 'generic'."
        )
    values = _numbers(_required(element, "affine").text, float)
    if len(values) != 12:
        raise ParserContentError(
            f"An affine <ViewTransform> holds 12 values, the first three "
            f"rows of its matrix, and this one holds {len(values)}."
        )
    return BdvRaw.Transform(type=kind, name=name, affine=values, span=span)


def _numbers(text: str, kind: type) -> tx.Tuple[tx.Any, ...]:
    """Read numbers separated by spaces."""
    return tuple(kind(value) for value in text.split())


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _ids(ids: tx.Iterable[int]) -> str:
    """Return a list of ids for a message."""
    ids = sorted(ids)
    return ", ".join(str(i) for i in ids) if ids else "none"


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse options, which the record does not take.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"A BigDataViewer XML file takes no option, but was given "
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


def _new_file(
    timepoint: int, setup: int, transforms: tx.Sequence[BdvRaw.Transform]
) -> bytes:
    """Return the bytes of a new XML file that holds a single view.

    The file is laid out as BigDataViewer writes one
    (`XmlIoAbstractSpimData.toXml`, with JDOM's pretty format), with CRLF
    line separators. It has one setup without size or voxel size, one
    time point and no image loader, which BigDataViewer reads as a
    dataset without images.
    """
    newline = b"\r\n"
    unit = _UNIT
    elements = [
        render_transform(item.name, item.affine, unit * 3, unit, newline)
        for item in transforms
    ]
    registration = render_registration(
        timepoint, setup, elements, unit * 2, unit, newline
    )
    lines = [
        b'<?xml version="1.0" encoding="UTF-8"?>',
        b'<SpimData version="0.2">',
        unit + b'<BasePath type="relative">.</BasePath>',
        unit + b"<SequenceDescription>",
        unit * 2 + b"<ViewSetups>",
        unit * 3 + b"<ViewSetup>",
        unit * 4 + b"<id>%d</id>" % setup,
        unit * 3 + b"</ViewSetup>",
        unit * 2 + b"</ViewSetups>",
        unit * 2 + b'<Timepoints type="range">',
        unit * 3 + b"<first>%d</first>" % timepoint,
        unit * 3 + b"<last>%d</last>" % timepoint,
        unit * 2 + b"</Timepoints>",
        unit + b"</SequenceDescription>",
        unit + b"<ViewRegistrations>",
        unit * 2 + registration,
        unit + b"</ViewRegistrations>",
        b"</SpimData>",
        b"",
    ]
    return newline.join(lines)
