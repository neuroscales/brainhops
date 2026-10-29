"""The image stack format."""

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core._internal_api import (
    ArrayProtocol,
    ArrayProxy,
    smartproperty,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    FIELDS,
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.datamodel.metadata._report import MetadataLossError
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    ParserTypeError,
    WriterError,
    disabled,
)
from brainhops.io.common._internal_api import (
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._data import _StackProxy
from ._files import (
    DEFAULT_NAME,
    name_parts,
    numbered_names,
    reads_from,
    stack_target,
    write_stack,
)
from ._geometry import encoded_geometry, stack_transformations, stored_geometry
from ._metadata import StackMetadata
from ._raw import (
    NOT_A_STREAM,
    StackMember,
    StackRaw,
    checked_axis,
    member_format,
    read_record,
    refuse_options,
    refuse_path,
    refuse_stream,
)
from ._views import _image_to_disk, _image_to_model, _permutation


@register_format
class StackImage(
    BinaryFileReader, BinaryFileWriter, ImageFormat, SingleScaleImage
):
    """A volume stored as a folder of numbered two-dimensional images.

    Light-sheet microscopes, electron microscopes and many export tools
    write a volume as one image file per slice, such as `z0000.tif`,
    `z0001.tif` and so on. This class reads such a folder as one image,
    and writes an image as such a folder. The files are TIFF files, read
    and written with tifffile, or images in a format that Pillow reads and
    writes, such as PNG. A stack is never recognized from its path, so a
    folder is read as a stack only with the format hint `"stack"`, as in
    `load(folder, hint="stack")`, or with `StackImage.load(folder)` (see
    [`brainhops.io.images.stack`][] for the rules).

    The image holds the list of its files, their headers and the geometry
    of the stack as [`StackMetadata`][], whose record is a
    [`StackRaw`][], and its pixels in `rawdata`, a lazy array that reads each
    file only when one of its pixels is needed. A number that no file
    carries is a missing slice, which the lazy array fills without reading
    a file. The files are stacked along the axis that the record names, z
    by default, so the data are indexed `(x, y, z)`, or `(x, y, z, c)` for
    files with channels.

    The image has one transformation, a [`Scaling`][] from voxels to
    `"physical"`, which the record decodes to: the size of the pixels of
    the files scales the axes along their columns and rows, and the
    spacing of the slices scales the stacking axis. A size that the record
    does not know is 1, without a unit, as for a single TIFF or Pillow
    image.

    Assigned data and transformations are pending until [`consolidate`][]
    or a write encodes them. A write consolidates a copy, writes it, and
    leaves the image as it is, except when it replaces files that the lazy
    array reads (#426).
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("stack",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The slices stacked along a first axis, or `None`.

    After a read, `rawdata` is a lazy array of shape `(slices, rows,
    columns[,
    channels])`. Assigning `data` leaves `rawdata` as it is, and
    [`consolidate`][] stores the assigned array here, with its spatial
    axes in the order of the stored volume.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending until [`consolidate`][] moves it into
    `rawdata`.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The transformations assigned to the image, or `None`.

    The image takes over the `transformations` field of
    [`SingleScaleImage`][] to store it under this private name. The
    constructor still takes `transformations` in the same place, and the
    property of the same name reads it. Only an assigned list, even an
    empty one, is stored here. `None`, the default, means that no list
    was assigned, so the property decodes the list of the record and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[StackMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the stack, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    @smartproperty(cache=True, invalidated_by=("rawdata", "metadata"))
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The pixels as `(x, y, z[, c])`, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. Converted to Dask, the data have one block per slice, and
        each file is read when its block is computed. To change the
        pixels, assign `data`, and assigning `None` gives the lazy array
        of the file again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata, _axis(self.metadata))

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access from the geometry fields of
        the record, and cached, apart from an assigned list, so
        `img.transformation is img.transformation`, and a change made to
        the list in place, such as an appended transformation, is kept.
        Assigning other metadata drops the decoded list, so that it is
        decoded from the new record, and keeps an assigned list. A change
        made in place to a decoded list is therefore lost when other
        metadata is assigned. An image without metadata has no
        transformations. Assigning a list replaces the stored one, and
        assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the files.

        The shape is that of the assigned data, or else the shape of `rawdata`
        with its axes in the order of the data. The lazy array knows its
        shape from the record. An image without data raises as
        [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        stored = tuple(int(d) for d in self.rawdata.shape)
        order = _permutation(len(stored), _axis(self.metadata))
        return tuple(stored[i] for i in order)

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        axis: tx.Optional[tx.Union[str, int]] = None,
        **fields: tx.Any,
    ) -> "StackImage":
        """Encode the pending data and geometry into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the files that a
        write of the image would write. The record is encoded as follows,
        from the record of the metadata when there is one.

        1. The stored volume follows the data. When data is pending, or
           the stacking axis changes, the data is cut into slices along
           the stacking axis and becomes `rawdata`. The slices keep the
           numbers of the record when there are as many of them, and the
           files keep their names. A slice that had no file, or a slice
           of an image without a record, gets a name made of the name of
           the first file with its number, or `slice_<number>.tif`. Every
           slice then has a file, since the data of the model takes
           precedence over the record. Otherwise, `rawdata` is kept, with its
           missing slices.
        2. The geometry comes from the preferred transformation. Only a
           scaling onto axes with a unit of length is recorded, as the
           pixel size along the columns and the rows of the files and the
           spacing of the slices, and only as far as the files record it
           (see the section on writing in [`brainhops.io.images.stack`][]).
        3. The members of a record that is encoded have no header, since
           the header of a file is known only once it is written.

        An image with a record, no pending data, and the transformations
        of its record or none read, consolidated without options or with
        its own stacking axis, is left as it is: the copy shares `rawdata` and
        the metadata. The transformations are compared with those that
        the record decodes to, exactly. Otherwise, the new record is held
        by new metadata, and the record held before is never changed. The
        transformations are then decoded from the new record.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an image
        stack holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        axis : {"x", "y", "z", 0, 1, 2}, optional
            The axis along which to cut the volume into files. By default,
            it is the axis of the record, or z.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        StackImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the image has no data, if its data does not have three
            spatial axes and at most one channel axis, if a name has no
            image extension, or if `axis` is not valid.
        """
        report = ConversionReport(source="StackImage", target="StackImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, StackMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        record = _record(self.metadata)
        if axis is None:
            axis = record.axis if record is not None else "z"
        else:
            try:
                axis = checked_axis(axis)
            except ValueError as e:
                raise WriterError(str(e)) from None
        held = held_transformations(self)
        kept = held is None or same_transformations(
            held, _record_transformations(self.metadata)
        )
        if (
            self._data is None
            and kept
            and record is not None
            and axis == record.axis
        ):
            if not inplace:
                return replace(self, transformations=None, metadata=metadata)
            self.transformations = None
            self.metadata = metadata
            return self
        record, raw = _encoded(self, axis)
        if metadata is None:
            metadata = StackMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                rawdata=raw,
                metadata=metadata,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        return self

    @takes_error_policy
    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image as numbered files, in a folder or beside a name.

        A path whose name ends with the extension of a TIFF file or of a
        format that Pillow writes, such as `out.tif` or `out.png`, names
        the files: the number of each slice is appended to the name, so
        that the files are `out_0.tif`, `out_1.tif` and so on, padded with
        zeros to the digits of the largest number, in the folder of the
        path, whose other files are kept. Any other path is the folder of
        the stack, whose files keep the names of the record, and which
        replaces any folder of that name that holds only images.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], which is written, and the image itself is
        not changed. A slice without a file is not written. The files of an
        image that was read and not changed are copied, byte for byte,
        when they keep their format. Every file is written before any is
        replaced. A folder from which the image reads some files is
        replaced only if it holds no other file. When the lazy array of
        the image reads a file that is replaced, the image is consolidated
        in place instead, so that it holds what the files hold (#426), and
        its lazy array reads the new files.

        Parameters
        ----------
        filename : path-like
            The local path of the folder, or the name of the files.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the path is not local, if the image cannot be encoded, or
            if the folder to replace holds files that are not images.
        """
        axis = kwargs.pop("axis", None)
        fields = {n: kwargs.pop(n) for n in list(kwargs) if n in FIELDS}
        refuse_options(type(self), kwargs)
        target = stack_target(filename)
        read = []
        raw = self.rawdata
        stored = raw.root if isinstance(raw, ArrayProxy) else raw
        if isinstance(stored, _StackProxy):
            read = stored.filenames()
        image = self.consolidate(axis=axis, **fields)
        reading = reads_from(self.rawdata, target, image.metadata.raw)
        if reading:
            fill = stored.fill
            # What the metadata loses was acted on by the copy above.
            image = self.consolidate(
                inplace=True,
                axis=axis,
                on_error={MetadataLossError: "ignore"},
                **fields,
            )
        written = write_stack(target, image.metadata.raw, image.rawdata, read)
        if not reading:
            return
        # The image holds what the files hold, and its lazy array reads the
        # new files. Copied files keep the geometry of the record, which
        # the files may not record.
        record = image.metadata.raw
        fresh = read_record(
            written.folder,
            written.numbered,
            pattern=written.pattern,
            axis=record.axis,
        )
        if written.copied:
            fresh = replace(
                fresh,
                pixel_size=record.pixel_size,
                spacing=record.spacing,
                unit=record.unit,
            )
        self.rawdata = _StackProxy(written.folder, fresh, fill)
        self.metadata = StackMetadata.from_raw(fresh)
        self.data = None

    to_fileobj = disabled(WriterError, NOT_A_STREAM)

    to_bytes = disabled(WriterError, NOT_A_STREAM)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        pattern: tx.Optional[str] = None,
        **kwargs,
    ) -> float:
        """Refuse a path, since a stack is read only with its hint.

        A stack is never recognized from its path, neither a folder nor a
        file, so [`load`][brainhops.io.load] chooses this class only with
        the hint `"stack"`, and this class never takes a folder or a file
        from another reader.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        error : bool or type[Exception], default=False
            Whether to raise an error, which tells why the path is not the
            folder of a stack, or that a stack is read only by hint.
        pattern : str, optional
            The glob pattern that selects the files of the stack.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_path(filename, pattern, error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Refuse a stream, which cannot hold a folder.

        Parameters
        ----------
        file : IO
            The stream.
        error : bool or type[Exception], default=False
            Whether to raise an error, since a stream is never a stack.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_stream(error)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Refuse bytes, which cannot hold a folder.

        Parameters
        ----------
        content : bytes
            The bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error, since bytes are never a stack.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_stream(error)

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        pattern: tx.Optional[str] = None,
        axis: tx.Optional[tx.Union[str, int]] = None,
        pixel_size: tx.Optional[tx.Union[float, tx.Sequence[float]]] = None,
        spacing: tx.Optional[float] = None,
        unit: tx.Optional[tx.Any] = None,
        fill: tx.Optional[tx.Any] = None,
        **kwargs,
    ) -> "StackImage":
        """Read the image stack in a folder, without reading pixels.

        The header of every file is read to check that the files can be
        stacked. The lazy array reads the pixels of a file, by name, when
        they are needed. The options `axis`, `pixel_size`, `spacing` and
        `unit` set the fields of the record, which the transformations are
        decoded from.

        Parameters
        ----------
        filename : path-like
            The path of a local folder.
        pattern : str, optional
            The glob pattern, relative to the folder, that selects the
            files of the stack, such as `"slice_*.tif"`. By default, every
            image file of the folder is taken. Hidden files are never
            taken.
        axis : {"x", "y", "z", 0, 1, 2}, optional
            The axis of the volume along which the files are stacked, by
            name or by its position in `(x, y, z)`. It is z by default, the
            slowest axis.
        pixel_size : float or Sequence[float], optional
            The size of a pixel of the files, one size for the columns and
            the rows or a pair `(columns, rows)`, which replaces the size
            that the files record.
        spacing : float, optional
            The distance from one slice to the next, which replaces the
            spacing that the files record. A negative spacing reverses the
            stacking axis.
        unit : str or Unit, optional
            The unit of length of `pixel_size` and `spacing`. The sizes
            that the files record are converted to it.
        fill : number, optional
            The value of the pixels of a slice whose number no file
            carries. It is NaN for a floating type and 0 for any other type
            by default.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        StackImage
            The image, whose pixels are read lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ValueError
            If an option is not valid, or if `fill` cannot be stored in
            the type of the files.
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the folder does not hold a stack that can be read: the names
            of its image files do not follow one numbering, a number is
            repeated, or the files differ in format, shape, type or number
            of channels.
        """
        refuse_options(cls, kwargs)
        record = StackRaw.from_folder(
            filename,
            pattern,
            axis=axis,
            pixel_size=pixel_size,
            spacing=spacing,
            unit=unit,
        )
        if isinstance(filename, str):
            filename = path.Path(filename)
        folder = os.path.abspath(os.fspath(filename))
        return cls(
            rawdata=_StackProxy(folder, record, fill),
            metadata=StackMetadata.from_raw(record),
        )

    from_fileobj = disabled(ParserTypeError, NOT_A_STREAM)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(metadata: tx.Optional[StackMetadata]) -> tx.Optional[StackRaw]:
    """Return the record of some metadata, or `None`."""
    return None if metadata is None else metadata.raw


def _axis(metadata: tx.Optional[StackMetadata]) -> str:
    """Return the stacking axis of the record of some metadata, or z."""
    record = _record(metadata)
    return "z" if record is None else record.axis


def _record_transformations(
    metadata: tx.Optional[StackMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record, or a record without files, gives no
    transformation.
    """
    record = _record(metadata)
    if record is None:
        return []
    return stack_transformations(record)


def _encoded(image: StackImage, axis: str) -> tx.Tuple[StackRaw, tx.Any]:
    """Encode the record and the stored volume of an image.

    The encoders read the transformations without storing them, so that
    a write leaves the image as it was.

    Raises
    ------
    WriterError
        If the image has no data, or data of another number of axes.
    """
    old = _record(image.metadata)
    xforms = held_transformations(image)
    if xforms is None:
        xforms = _record_transformations(image.metadata)
    xform = xforms[0] if xforms else None
    if (
        image._data is None
        and old is not None
        and image.rawdata is not None
        and axis == old.axis
    ):
        # Only the geometry changes, so the slices are those of
        # `rawdata`.
        raw = image.rawdata
        ndim = len(old.axes)
        members = tuple(replace(m, header=None) for m in old.members)
    else:
        data = image._data
        if data is None and image.rawdata is not None:
            data = _image_to_model(image.rawdata, _axis(image.metadata))
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        ndim = len(data.shape)
        if ndim not in (3, 4):
            raise WriterError(
                f"An image stack holds three spatial axes and at most one "
                f"channel axis, but this image has {ndim} axes (shape "
                f"{tuple(data.shape)})."
            )
        raw = _image_to_disk(data, axis)
        members = _new_members(old, axis, tuple(raw.shape), raw.dtype)
    fmt = members[0].format if members else None
    pixel_size, spacing, unit = stored_geometry(
        fmt, ndim == 4, encoded_geometry(xform, axis, ndim)
    )
    record = StackRaw(
        members=members,
        pattern=None if old is None else old.pattern,
        axis=axis,
        pixel_size=pixel_size,
        spacing=spacing,
        unit=unit,
    )
    return record, raw


def _new_members(
    old: tx.Optional[StackRaw],
    axis: str,
    shape: tx.Tuple[int, ...],
    dtype: tx.Any,
) -> tx.Tuple[StackMember, ...]:
    """Return the members of a stored volume of new slices.

    The slices keep the numbers of the old record when it is cut along the
    same axis into as many slices, and the files keep their names. A slice
    that had no file, and every slice otherwise, is named after the first
    file of the old record, or `slice_<number>.tif`, with its number.

    Raises
    ------
    WriterError
        If a name is not that of a TIFF file nor of an image that Pillow
        writes.
    """
    count = shape[0]
    first = old.members[0] if old is not None and old.members else None
    if first is not None and old.axis == axis and len(old.numbers) == count:
        numbers = list(old.numbers)
        held = {member.number: member.name for member in old.members}
    else:
        start = 0 if first is None else first.number
        numbers = list(range(start, start + count))
        held = {}
    prefix, suffix = DEFAULT_NAME if first is None else name_parts(first)
    names = [
        held.get(number, name)
        for number, name in zip(
            numbers, numbered_names(prefix, suffix, numbers)
        )
    ]
    dtype = np.dtype(dtype).newbyteorder("=")
    members = []
    for number, name in zip(numbers, names):
        fmt = member_format(name)
        if fmt is None:
            raise WriterError(
                f"Cannot write {name!r}: it is not the name of a TIFF file "
                f"nor of an image that Pillow writes (TIFF files need "
                f"tifffile and other images need Pillow)."
            )
        axes, member_shape = _member_layout(fmt, shape[1:], dtype)
        members.append(
            StackMember(
                name=name,
                number=number,
                format=fmt,
                axes=axes,
                shape=member_shape,
                dtype=str(dtype),
            )
        )
    return tuple(members)


def _member_layout(
    fmt: str, plane: tx.Tuple[int, ...], dtype: np.dtype
) -> tx.Tuple[str, tx.Tuple[int, ...]]:
    """Return the axes and the shape of a file that holds a plane.

    A plane is `(rows, columns[, channels])`. Pillow stores the channels
    of a pixel together, and so does the TIFF writer for the three or four
    samples of an 8-bit RGB or RGBA image. The TIFF writer stores other
    channels as separate planes, `(channels, rows, columns)`.
    """
    if len(plane) == 2:
        return "YX", plane
    rgb = dtype == np.uint8 and plane[2] in (3, 4)
    if fmt == "pillow" or rgb:
        return "YXS", plane
    return "CYX", (plane[2], plane[0], plane[1])
