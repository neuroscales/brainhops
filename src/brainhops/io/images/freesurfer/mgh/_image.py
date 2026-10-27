"""The MGH image format."""

# stdlib
import gzip
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace
from nibabel.arrayproxy import ArrayProxy
from nibabel.freesurfer import mghformat as _mgh
from nibabel.volumeutils import array_from_file, array_to_file

# internals
from brainhops._core import path
from brainhops._core.path import is_same_file, write_file
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import ArrayCoordinateSystem, CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    ParserExistsError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._geometry import (
    Arrangement,
    arrange_voxel_to_ras,
    declared_axes,
    held_transformations,
    same_transformations,
)
from brainhops.io.common.freesurfer import (
    FreesurferFormat,
    FreesurferTkrSystem,
)
from brainhops.io.common.freesurfer._geometry import fs_geometry_from_vox2ras
from brainhops.io.common.mgh import MghMetadata, MghRaw
from brainhops.io.common.mgh._constants import _MGH_AXES, _MRI_PARAMS
from brainhops.io.common.mgh._metadata import _encode as _encode_vocabulary
from brainhops.io.common.mgh._raw import (
    _compressed_name,
    encode_mgh,
    read_mgh,
)
from brainhops.io.common.nifti._geometry import _scale_spatial, _unit_scale
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)

# this format
from ._views import _image_to_disk, _image_to_model

_SCANNER = "scanner"
"""Name of the scanner RAS space, the preferred world space."""

_TKR = "tkr"
"""Name of the tkr (surface) RAS space."""

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_MGH_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
}


@register_format
class MghImage(
    BinaryFileReader,
    BinaryFileWriter,
    ImageFormat,
    FreesurferFormat,
    SingleScaleImage,
):
    """An image stored in a FreeSurfer MGH or MGZ file.

    An MGH image holds everything that its file stores beside the voxels
    as [`MghMetadata`][], whose record is an [`MghRaw`][], and the voxels
    as stored in `rawdata`. When the image is read from a file,
    `rawdata` is a nibabel proxy, which reads the voxels only when they
    are needed, and `data` is decoded from it on first access. The
    voxels are in Fortran order, and the data are indexed `(x, y, z)` or
    `(x, y, z, frames)`, where the frames form a time axis.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. They are, in order, a
    [`Scaling`][] to `"physical"` by the voxel size in millimeters and the
    TR in milliseconds, an [`Affine`][] to `"tkr"` (`MghRaw.vox2tkr`), and
    the preferred [`Affine`][] to `"scanner"` (`MghRaw.vox2ras`). The tkr
    RAS space of a 3-D image is a
    [`FreesurferTkrSystem`][brainhops.io.common.freesurfer.FreesurferTkrSystem]
    and not `RASmm`, so that a format that stores scanner RAS refuses to
    write it as scanner RAS. The tkr space of a 4-D image is a system of
    four axes named `"tkr"`, as its scanner space is a system of four
    axes named `"scanner"`.

    `rawdata` and the record describe the file as it was read, or as the image
    was last consolidated. Data that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the image is
    written, which writes such a copy and leaves the image as it was. The
    transformations, read or assigned, are compared at [`consolidate`][]
    with those that the record decodes to, and the geometry of the record
    is kept when they are the same.

    When the image is consolidated, the record of its metadata is the base
    of the new record, as [`consolidate`][] describes, so the acquisition
    parameters of the footer, the trailing tags and the other header
    fields are kept. An image that is read and written again without
    changes gives the same bytes, after decompression for an `.mgz` file.
    They are lost when the image is converted to another format.

    !!! note "`goodRASFlag`"
        When the flag is not positive, FreeSurfer and the reader assume 1
        mm voxels, LIA direction cosines and a zero centre. A file whose
        geometry is kept is written back with its flag, and a file whose
        geometry is encoded from the image sets the flag.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mgh", ".mgz", ".mgh.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mgh", "mgz")

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels as the file stores them, or `None` without data.

    After a read, `rawdata` is a nibabel proxy. `rawdata` describes the
    file as it was read, or as the image was last consolidated, so
    assigning `data` leaves it as it is.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending: it is encoded into `rawdata` only by
    [`consolidate`][] or by a write.
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

    _metadata: NoRepr[tx.Optional[MghMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning
    other metadata drops the transformations decoded from the previous
    record, so that they are decoded from the new one, and keeps the
    transformations that were assigned.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the voxels, assign `data`, and assigning `None`
        gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept.
        Assigning other metadata drops the decoded list, so that it is
        decoded from the new record, and keeps an assigned list. A change
        made in place to a decoded list is therefore lost when other
        metadata is assigned. An image without metadata has no
        transformations.

        Assigning a list replaces the stored one. In both cases, the list
        is compared at [`consolidate`][], and by a write, with the list
        that the record decodes to, exactly. An empty list means that the
        image has no geometry, and the image is then written as an image
        built from data alone. Assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else of `rawdata`: the data
        is `rawdata` as the file stores it, so the two have the same
        shape, and a proxy knows its shape from the header. An image
        without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the record.

        The axes are `x`, `y` and `z`, and `t` for a file with frames, in
        Fortran order. An image without metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return _mgh_system(self.metadata.raw)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MGH image.

        The stream is tested with [`MghRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, which is read from its current position.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MGH header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MghRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an MGH image.

        Parameters
        ----------
        data : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no MGH header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "MghImage":
        """Read an image from the path of an MGH or MGZ file.

        A local file whose name matches its content is handed to nibabel
        by name, so that nibabel opens it whenever the voxels are read and
        can memory-map them. Any other file is read into memory.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the nibabel proxy, which are `mmap` and
            `keep_file_open`. Other options are refused.

        Returns
        -------
        MghImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        raw, proxy = read_mgh(filename, **kwargs)
        return cls(rawdata=proxy, metadata=MghMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MghImage":
        """Read an image from an open MGH or MGZ stream.

        The record is read at once, and the proxy reads the voxels from
        the stream when they are needed, so the stream must stay open
        while the data may be read. A stream that cannot seek is first
        read into memory.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be compressed with gzip.
        **kwargs : Any
            Options of the nibabel proxy, which are `mmap` and
            `keep_file_open`. Other options are refused.

        Returns
        -------
        MghImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        """
        raw, proxy = read_mgh(file, **kwargs)
        return cls(rawdata=proxy, metadata=MghMetadata.from_raw(raw))

    @classmethod
    def from_nibabel(
        cls, mgh: tx.Union[_mgh.MGHImage, _mgh.MGHHeader], **kwargs
    ) -> "MghImage":
        """Build an image from a nibabel MGH image or header.

        The voxels of an image become `rawdata` as they are, and a copy of its
        header becomes the record of the metadata, without tags. A header
        alone gives an image without data.

        Parameters
        ----------
        mgh : nibabel.freesurfer.mghformat.MGHImage or MGHHeader
            The nibabel image or header.
        **kwargs : Any
            Other fields of the image.

        Returns
        -------
        MghImage
            The image.

        Raises
        ------
        TypeError
            If `mgh` is neither an MGH image nor an MGH header.
        """
        if isinstance(mgh, _mgh.MGHImage):
            raw, header = mgh.dataobj, mgh.header
        elif isinstance(mgh, _mgh.MGHHeader):
            raw, header = None, mgh
        else:
            raise TypeError(
                f"Expected an MGH image or header, got {type(mgh)}"
            )
        metadata = MghMetadata.from_raw(MghRaw(header=header).copy())
        return cls(rawdata=raw, metadata=metadata, **kwargs)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options: tx.Any,
    ) -> "MghImage":
        """Encode the pending data and geometry into `rawdata` and the record.

        The record is encoded in a fixed order. The base is a copy of the
        record of the metadata, or a new record for an image without
        metadata. The dimensions and the voxel type follow the data. The
        geometry comes from the transformations, as described in
        [`to_nibabel`][]: the voxel size, the cosines, the centre and the
        `goodRASFlag` of the base are kept when the voxel-to-scanner matrix
        and the spatial dimensions are those of the base, and they are
        encoded again otherwise. The MRI parameters of `like` are copied
        next, then a TR stated by the transformations, and the `overrides`
        are applied last. The tags and the other fields of the base are
        kept. The new record is held by new metadata, the record held
        before is never changed, and the data becomes `rawdata`, in the
        axis order of the file. Nothing is pending afterwards, so the
        transformations are decoded from the new record, with the
        precision of the file.

        When this encoding changes nothing, the metadata is kept as it is,
        so that an image that was read and not changed writes the record
        that it read. The transformations that the image holds, read or
        assigned, are compared with those that the record decodes to,
        exactly, and an image with a record, no pending data, and the
        transformations of its record or none read, consolidated without
        options, is left as it is: the copy shares `rawdata` and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, and its record is
        the base of the new record. The MRI parameters given by keyword,
        such as `flip_angle=` in degrees, take precedence over `like`. The
        image owns its geometry, so a repetition time that disagrees with
        the TR stated by the transformations is reported as approximated.
        The voxels are stored with the type given as `dtype=`, or else
        with the nearest type to `metadata.dtype` that MGH stores, or else
        with the nearest type to that of the data. What the write loses
        or approximates is acted on once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        like : path, nibabel MGH image or header, or MGH object, optional
            Template whose MRI parameters override those of the metadata.
            The fields given by keyword, and a TR stated by the
            transformations, override them.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect (see
            [`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]).
        **options : Any
            Fields of the metadata to override, such as `dtype`,
            `repetition_time` or `flip_angle`, and other header fields,
            which are set last. `dtype` must be a type that MGH stores
            (uint8, int16, int32, float32). The header fields `dims` and
            `type` are refused, because the dimensions follow the data
            and the voxel type is `dtype`, and so are `tr`, `te` and `ti`,
            which hold `repetition_time`, `echo_time` and
            `inversion_time`.

        Returns
        -------
        MghImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If an option names a header field that holds a field of the
            metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If there is no data, the data have more than four dimensions,
            the voxel type cannot be stored, or `dims` or `type` is
            overridden.
        UnrepresentableTransformationError
            If the transformation has no affine form, has a second
            non-spatial axis, or shifts or scales frames in a way that MGH
            cannot store.
        """
        fields, overrides = _split_options(options)
        report = ConversionReport(source="MghImage", target="MghImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, MghMetadata, fields, report
        )
        # The record is kept when the transformations were never read, or
        # are the same as those that it decodes to.
        held = held_transformations(self)
        kept = held is None or same_transformations(
            held, _record_transformations(self.metadata)
        )
        if (
            self._data is None
            and kept
            and like is None
            and not overrides
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            apply_loss_policy(report, on_loss, stacklevel=2)
            if not inplace:
                return replace(self, transformations=None)
            self.transformations = None
            return self
        record, raw = _encoded(self, like, overrides, metadata, fields)
        if self.metadata is not None and record is self.metadata.raw:
            metadata = self.metadata
        else:
            metadata = type(metadata or MghMetadata()).from_raw(record)
        overruled_fields(requested, metadata, report)
        apply_loss_policy(report, on_loss, stacklevel=2)
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

    def to_nibabel(self, like: tx.Any = None, **overrides) -> _mgh.MGHImage:
        """Build the nibabel MGH image that encodes this image.

        The voxel-to-scanner matrix comes from the `"scanner"`
        transformation, or else from the preferred one unless it maps to
        tkr RAS, which MGH cannot store. Without a transformation, the
        voxels are 1 mm along the RAS axes. The data are transposed to put
        the spatial axes first and at most one frame axis after them (see
        [`plan_axes`][brainhops.io.common._geometry.plan_axes]). The TR of
        a time axis is stored in milliseconds.

        The image is consolidated with the options, as by
        [`consolidate`][], into a copy. The header of the nibabel image
        starts from the header of the record of that copy, and nibabel
        encodes the geometry again from `MghRaw.vox2ras`. The geometry is
        therefore the one that FreeSurfer reads, with the FreeSurfer
        defaults for a record whose `goodRASFlag` is not positive, and the
        flag is set. A nibabel image has no place for the trailing tags,
        which are left out. The image itself is not changed.

        Parameters
        ----------
        like : path, nibabel MGH image or header, or MGH object, optional
            Template whose MRI parameters override those of this image. A
            TR stated by the transformation overrides both.
        **overrides : Any
            `on_loss`, and the fields of the metadata and of the header,
            as for [`consolidate`][].

        Returns
        -------
        nibabel.freesurfer.mghformat.MGHImage
            The encoded image.

        Raises
        ------
        WriterError
            If there is no data, the data have more than four dimensions,
            or the voxel type cannot be stored.
        UnrepresentableTransformationError
            If the transformation cannot be stored, as for
            [`consolidate`][].
        """
        image = self.consolidate(like=like, **overrides)
        record, raw = _mgh_written(image)
        return _mgh.MGHImage(raw, record.vox2ras, header=record.header)

    def to_bytes(self, compress: bool = False, **kwargs) -> bytes:
        """Return the MGH encoding: header, voxels, footer and tags.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], and the record and `rawdata` of the copy are
        written. A proxy that reaches the writer unchanged is written as
        the file stores it, so that an untouched image gives the bytes of
        its file. The image itself is not changed.

        Parameters
        ----------
        compress : bool, default=False
            Whether to gzip the result into MGZ.
        **kwargs : Any
            The template `like`, `on_loss` and the options of
            [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        WriterError
            If the image cannot be encoded, as in [`consolidate`][].
        """
        content = encode_mgh(*_mgh_written(self.consolidate(**kwargs)))
        return gzip.compress(content) if compress else content

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a path.

        The file is gzipped when its name ends with `.mgz` or `.gz`, unless
        `compress` says otherwise. The other keyword arguments are those of
        [`consolidate`][]. The content is encoded before the file is
        written, and a local file is replaced at once, so an image can be
        written over the file that it was read from. The image is then
        consolidated in place, so that it holds what the file holds
        (#426), and its lazy array reads the new file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Arguments of [`to_bytes`][], which are `compress` and the
            options of [`consolidate`][].
        """
        compress = kwargs.pop("compress", _compressed_name(filename))
        if not _reads_file(self.rawdata, filename):
            write_file(filename, self.to_bytes(compress=compress, **kwargs))
            return
        self.consolidate(inplace=True, **kwargs)
        content = encode_mgh(*_mgh_written(self))
        write_file(filename, gzip.compress(content) if compress else content)
        # The image holds what the file holds, and its lazy array reads the
        # new file, since the lazy array of the old file describes how that
        # file stored the voxels.
        self.rawdata = read_mgh(filename)[1]
        self.data = None


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _record_transformations(
    metadata: tx.Optional[MghMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _mgh_to_transformations(metadata.raw)


def _mgh_system(record: MghRaw) -> CoordinateSystem:
    """Return the voxel coordinate system of a record."""
    axes = _MGH_AXES[: len(record.shape)]
    return ArrayCoordinateSystem(name="voxel", axes=axes, order="F")


def _mgh_to_transformations(record: MghRaw) -> tx.List[Transformation]:
    """Return the scaling, tkr RAS and scanner RAS transformations."""
    voxel_space = _mgh_system(record)
    axes = list(voxel_space.axes)
    tr = record.mri_params["tr"]

    # Frames are in milliseconds only when the TR is known; a unit is
    # never invented.
    units = {"space": "mm", "time": "ms" if tr > 0 else None}
    phys_axes = [replace(axis, unit=units.get(axis.type)) for axis in axes]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)

    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=_RAS_ORIENTATION[axis.name]
            ),
        )
        if axis.name in _RAS_ORIENTATION
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    zooms = list(record.voxel_size)
    if len(axes) > 3:
        zooms.append(tr if tr > 0 else 1.0)
    vox2phys = Scaling(
        input=voxel_space, output=phys_space, scale=zooms[: len(axes)]
    )

    def _affine(matrix: np.ndarray, output: CoordinateSystem) -> Affine:
        return Affine(
            input=voxel_space,
            output=output,
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    # The shared tkregister system has three axes, so a 4-D image keeps a
    # system of four axes that is only named after it.
    if len(ras_axes) == 3:
        tkr_space = FreesurferTkrSystem(name=_TKR, axes=ras_axes)
    else:
        tkr_space = replace(ras_space, name=_TKR)
    return [
        vox2phys,
        _affine(record.vox2tkr, tkr_space),
        _affine(record.vox2ras, replace(ras_space, name=_SCANNER)),
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


_LAYOUT_FIELDS = frozenset({"dims", "type"})
"""Header fields that describe the stored voxels, which the data sets."""

_VOCABULARY_SLOTS = {
    "tr": "repetition_time",
    "te": "echo_time",
    "ti": "inversion_time",
}
"""Header fields that hold a field of the metadata, with that field."""

_FOOTER_FIELDS = (
    "repetition_time",
    "echo_time",
    "inversion_time",
    "flip_angle",
)
"""The fields of the metadata that the MRI parameters of the footer hold."""


def _split_options(
    options: tx.Mapping[str, tx.Any],
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """Split the options of the MGH writer into fields and header overrides.

    An option named after a field of the metadata, such as `dtype` or
    `flip_angle`, is that field. Any other option names a field of the
    header, except the header fields that hold a field of the metadata,
    which are refused.

    Raises
    ------
    TypeError
        If an option is `tr`, `te` or `ti`.
    """
    refused = sorted(set(options) & set(_VOCABULARY_SLOTS))
    if refused:
        hints = ", ".join(
            f"{name}= (give {_VOCABULARY_SLOTS[name]}= instead)"
            for name in refused
        )
        raise TypeError(
            f"These header fields hold metadata, and cannot be overridden "
            f"by name: {hints}."
        )
    fields = {name: v for name, v in options.items() if name in FIELDS}
    overrides = {name: v for name, v in options.items() if name not in FIELDS}
    return fields, overrides


def _stored_dtype(
    fields: tx.Mapping[str, tx.Any], metadata: tx.Optional[MghMetadata]
) -> tx.Optional[np.dtype]:
    """Return the voxel type that a write is asked to store, if any.

    The `dtype` given by keyword is returned as it is, and refused later
    if MGH cannot store it. Otherwise the `dtype` of the metadata, which
    may come from another format, gives the nearest type that MGH stores.
    `None` leaves the type to the data.
    """
    if "dtype" in fields:
        return fields["dtype"]
    if metadata is None or metadata.dtype is None:
        return None
    return _mgh_dtype(np.empty(0, metadata.dtype))


def _encoded(
    image: MghImage,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    metadata: tx.Optional[MghMetadata],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Tuple[MghRaw, ArrayProtocol]:
    """Return the record and the voxels that consolidating an image stores.

    The record is encoded as [`MghImage.consolidate`][] describes, over a
    copy of the record of the consolidated `metadata`, and the voxels are
    the assigned data, or else `rawdata`, arranged in the axis order
    of the file when the transformations declare another order. The held
    record is returned when the encoding changes nothing. Voxels of
    another type than the stored one are converted as the writer converts
    them, with `_stored_voxels`.
    """
    if image._data is not None:
        source = _image_to_disk(image._data)
    else:
        source = image.rawdata
    if source is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    overrides = dict(overrides)
    refused = sorted(_LAYOUT_FIELDS.intersection(overrides))
    if refused:
        raise WriterError(
            f"The header fields {refused} cannot be overridden: the "
            f"dimensions come from the data, and the voxel type is set "
            f"with dtype=."
        )
    dtype = _stored_dtype(fields, image.metadata)
    # The list of the image is read without being stored, so that a write
    # leaves the image as it was.
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _record_transformations(image.metadata)
    vox2ras, layout, tr = _scanner_geometry(transformations, source)
    data = source if layout is None else layout.apply(image.data)
    shape = _mgh_shape(data)

    # The base is a copy of the record of the metadata, or a new record.
    held = None if metadata is None else metadata.raw
    record = MghRaw() if held is None else held.copy()
    header = record.header

    # The layout of the voxels and the geometry come from the image.
    header["dims"] = shape + (1,) * (4 - len(shape))
    header.set_data_dtype(_mgh_dtype(data, dtype))
    if held is None or not _same_geometry(held, vox2ras, shape):
        _set_geometry(header, vox2ras, shape)

    # The template comes next, then the TR of the image, then the
    # overrides.
    template = _like_header(like)
    if template is not None:
        for name in _MRI_PARAMS:
            header[name] = template[name]
        # The parameters given by keyword take precedence over the
        # template, so they are encoded again.
        given = {
            name: fields[name] for name in _FOOTER_FIELDS if name in fields
        }
        _encode_vocabulary(record, given, ConversionReport())
    if tr is not None:
        header["tr"] = tr
    for name, value in overrides.items():
        header[name] = value

    # The voxels are converted as the writer converts them, so that `rawdata`
    # holds what the file holds.
    data = _stored_voxels(data, header.get_data_dtype())
    if record == held:
        return held, data
    return record, data


def _mgh_shape(data: ArrayProtocol) -> tx.Tuple[int, ...]:
    """Return the shape that MGH stores for an array, with at least 3 axes.

    Raises
    ------
    WriterError
        If the array has more than four dimensions, or none.
    """
    shape = tuple(int(d) for d in data.shape)
    if not 1 <= len(shape) <= 4:
        raise WriterError(
            f"MGH stores volumes of up to four dimensions "
            f"(x, y, z, frames), not {len(shape)}."
        )
    return shape + (1,) * max(0, 3 - len(shape))


def _same_geometry(
    record: MghRaw, vox2ras: np.ndarray, shape: tx.Tuple[int, ...]
) -> bool:
    """Tell whether a record already stores a geometry.

    The geometry is the same when the spatial dimensions are those of the
    record and the voxel-to-scanner matrix is the one that the record
    decodes to, exactly. A record read from a file and transformations
    decoded from it therefore keep the stored fields, which encoding them
    again could round differently.
    """
    dims = tuple(int(d) for d in record.header["dims"][:3])
    return dims == shape[:3] and np.array_equal(record.vox2ras, vox2ras)


def _set_geometry(
    header: _mgh.MGHHeader, vox2ras: np.ndarray, shape: tx.Tuple[int, ...]
) -> None:
    """Encode a voxel-to-scanner matrix in the geometry fields of a header.

    The voxel size, the direction cosines and the centre are recovered
    from the matrix, and the `goodRASFlag` is set so that FreeSurfer reads
    them.
    """
    voxel_size, xras, yras, zras, cras = fs_geometry_from_vox2ras(
        vox2ras, shape[:3]
    )
    header["delta"] = voxel_size
    header["Mdc"] = (xras, yras, zras)
    header["Pxyz_c"] = cras
    header["goodRASFlag"] = 1


_MGH_POLICY = dict(fill_space=True, max_nonspatial=1)
"""Placement of axes in MGH files: three spatial axes, then frames."""


def _scanner_transformation(
    transformations: tx.Sequence[Transformation],
) -> tx.Optional[Transformation]:
    """Pick the transformation that gives the voxel-to-scanner matrix.

    A transformation named `"scanner"` wins. Otherwise the last
    transformation that does not map to tkr RAS is used, or `None` if
    there is none. A transformation maps to tkr RAS when its output is a
    `FreesurferTkrSystem` or is named `"tkr"`.
    """
    transformations = list(transformations or [])
    chosen = None
    for xform in transformations:
        if getattr(getattr(xform, "output", None), "name", None) == _SCANNER:
            chosen = xform
    if chosen is None:
        for xform in reversed(transformations):
            output = getattr(xform, "output", None)
            tkr = isinstance(output, FreesurferTkrSystem)
            if not tkr and getattr(output, "name", None) != _TKR:
                chosen = xform
                break
    return chosen


def _scanner_geometry(
    transformations: tx.Sequence[Transformation], data: tx.Any
) -> tx.Tuple[np.ndarray, tx.Any, tx.Optional[float]]:
    """Return the scanner matrix, the axis layout and the TR to store.

    The matrix is in millimeters and the TR in milliseconds (see
    [`MghImage.to_nibabel`][]). Without a transformation, the result is
    the identity, no layout and no TR.
    """
    chosen = _scanner_transformation(transformations)
    if chosen is None:
        return np.eye(4), None, None
    ndim = len(getattr(data, "shape", ()) or ())
    voxel_axes = declared_axes(getattr(chosen, "input", None), ndim)
    arranged = arrange_voxel_to_ras(
        chosen, voxel_axes, "MGH", "scanner", **_MGH_POLICY
    )
    output = getattr(chosen, "output", None)
    matrix = _scale_spatial(arranged.matrix, _unit_scale(output, "mm"))
    return matrix, arranged.layout, _frame_tr(arranged)


def _frame_tr(arranged: Arrangement) -> tx.Optional[float]:
    """Return the TR in milliseconds of the frame axis, or `None`.

    A time axis that still counts frames, with a scale of one and no time
    unit, states no TR. Frames that are not time can be neither scaled
    nor shifted.
    """
    if not arranged.others:
        return None
    scale, offset = arranged.others[0]
    groups = [
        g[3]
        for g in (arranged.voxel_groups, arranged.world_groups)
        if g is not None and len(g) > 3
    ]
    if any(g != "time" for g in groups):
        if (scale, offset) != (1.0, 0.0):
            raise UnrepresentableTransformationError(
                "MGH stores no spacing or origin for frames that are not "
                f"time, so a map that scales or shifts them ({scale}, "
                f"{offset}) cannot be written."
            )
        return None
    if offset != 0:
        raise UnrepresentableTransformationError(
            f"MGH stores no origin for the time axis, so a map that "
            f"shifts it ({offset}) cannot be written."
        )
    if scale <= 0:
        raise UnrepresentableTransformationError(
            f"MGH stores the repetition time as a positive number, so a "
            f"map that scales time by {scale} cannot be written."
        )
    axes = list(getattr(arranged.world, "axes", None) or [])
    unit = getattr(axes[3], "unit", None) if len(axes) > 3 else None
    if is_physicalunit(unit) and is_timeunit(unit):
        return float(scale) * float(unit.scale) * 1000.0
    if scale == 1.0:
        # Still a frame index, as read from a file without a TR.
        return None
    return float(scale)


def _mgh_dtype(data: tx.Any, dtype: tx.Any = None) -> np.dtype:
    """Return the voxel type to store (see [`MghImage.to_nibabel`][])."""
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype not in _MGH_DTYPES:
            raise WriterError(
                f"MGH cannot store voxels of type {dtype}; it stores uint8, "
                f"int16, int32 and float32."
            )
        return dtype
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype in _MGH_DTYPES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype.kind == "f":
        return np.dtype(np.float32)
    if dtype.kind in "iu":
        if np.can_cast(dtype, np.int16):
            return np.dtype(np.int16)
        if np.can_cast(dtype, np.int32):
            return np.dtype(np.int32)
        info = np.iinfo(np.int32)
        values = np.asarray(data)
        if values.size == 0 or (
            values.min() >= info.min and values.max() <= info.max
        ):
            return np.dtype(np.int32)
        raise WriterError(
            f"MGH stores integers in at most 32 bits, and this {dtype} "
            f"array holds values outside the int32 range."
        )
    raise WriterError(
        f"MGH cannot store voxels of type {dtype}; it stores uint8, int16, "
        f"int32 and float32."
    )


def _like_header(like: tx.Any) -> tx.Optional[_mgh.MGHHeader]:
    """Resolve a `like` template to the MGH header to copy fields from.

    The template is a nibabel MGH header or image, an [`MghRaw`][] record,
    MGH metadata, an MGH image, or the path of an MGH file, whose record
    alone is read. Any other template gives `None`.
    """
    if like is None:
        return None
    if isinstance(like, _mgh.MGHHeader):
        return like
    if isinstance(like, _mgh.MGHImage):
        return like.header
    if isinstance(like, MghImage):
        like = like.metadata
    if isinstance(like, MghMetadata):
        like = like.raw
    if isinstance(like, (str, path.PathLike)):
        like = MghRaw.from_filename(like)
    if isinstance(like, MghRaw):
        return like.header
    return None


def _mgh_written(image: MghImage) -> tx.Tuple[MghRaw, ArrayProtocol]:
    """Return the record and the voxels of a consolidated image.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image.rawdata is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    return image.metadata.raw, image.rawdata


def _reads_file(raw: tx.Any, file: tx.Any) -> bool:
    """Tell whether a lazy array of voxels reads a file by its name."""
    return isinstance(raw, ArrayProxy) and is_same_file(raw.file_like, file)


def _stored_voxels(data: ArrayProtocol, dtype: np.dtype) -> ArrayProtocol:
    """Return voxels in the type that the file stores, as written.

    The writer, `encode_mgh`, writes the voxels with nibabel's
    `array_to_file`, which rounds floating-point values and clips them to
    the range of an integer type. Voxels of another type are written to
    memory and read back in the same way, so that they are the values of
    the file, in its type and byte order. Voxels of the stored type, in
    either byte order, are returned as they are, so that a proxy or a
    lazy array is not read.
    """
    if np.dtype(data.dtype).newbyteorder("=") == dtype.newbyteorder("="):
        return data
    shape = tuple(int(d) for d in data.shape)
    buffer = BytesIO()
    array_to_file(np.asarray(data), buffer, dtype, order="F")
    buffer.seek(0)
    return array_from_file(shape, dtype, buffer, order="F")
