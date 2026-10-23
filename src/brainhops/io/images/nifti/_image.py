"""The NIfTI image format."""

# stdlib
from io import BytesIO

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel._backends import dataobjproperty, dataproperty
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import ArrayCoordinateSystem, CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    WriterError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._constants import (
    _NIFTI_FIELD_INTENTS,
    _NIFTI_INTENT_NONE,
    _NIFTI_XCODES,
)
from brainhops.io.common.nifti._geometry import _image_with_geometry
from brainhops.io.common.nifti._header import (
    _keep_given_fields,
    _nifti_intent,
    _nifti_shape,
    _nifti_to_axes,
    _NiftiObject,
    _split_options,
    _storage_override,
    _stored_parts,
    _written_nifti,
)
from brainhops.io.common.nifti._raw import (
    _sniffed_header,
    read_nifti,
    reads_file,
    write_nifti,
)
from brainhops.io.common.nifti._units import nifti_to_unit
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._base import (
    metadata_to_write,
    overruled_fields,
    requested_fields,
)

# this format
from ._views import _image_to_disk, _image_to_model


@register_format
class NiftiImage(
    BinaryFileReader, BinaryFileWriter, ImageFormat, SingleScaleImage
):
    """An image stored in a NIfTI file.

    A NIfTI image holds the header of its file as [`NiftiMetadata`][] and
    the voxels as stored in `rawdata`. When the image is read from a
    file, `rawdata` is a nibabel proxy, which reads the voxels only when
    they are needed, and `data` is decoded from it on first access. The
    transformations are decoded from the header in the same way.

    `rawdata` and the header describe the file as it was read, or as the image
    was last consolidated. Data assigned to the image are pending: they
    are kept beside `rawdata` and the header, and `data` returns them,
    until [`consolidate`][] encodes them into a copy of the image, or
    until the image is written. The transformations, read or assigned,
    are kept in the same way and compared at [`consolidate`][] with
    those that the header decodes to, so that they are encoded only when
    they differ. A write encodes the image in the same way, into a copy,
    and leaves the image as it was, except when the image is written
    over the file that its proxy reads.

    When the image is consolidated, the header of its file is the base of
    the new header. The fields that describe the layout of the voxels and
    the geometry are encoded again from the image, and the other fields,
    such as the description, the intent and the extensions, are kept. An
    image that is read and written again without changes is written from
    its proxy and its header as they are, so the file is copied byte for
    byte.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels as the file stores them, or `None` without data.

    After a read, `rawdata` is a nibabel proxy, which applies the
    scaling of the header when it is read. `rawdata` describes the file
    as it was read, or as the image was last consolidated, so assigning
    `data` leaves it as it is until [`consolidate`][] encodes the data
    into a copy.
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
    was assigned, so the property decodes the list of the header and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its header, or `None`.

    An image built from data has no metadata. The header describes the
    file as it was read, or as the image was last consolidated. Assigning
    other metadata drops the transformations decoded from the previous
    header, so that they are decoded from the new one, and keeps the
    transformations that were assigned.
    """

    data = dataproperty()
    """The image data, assigned or decoded from `rawdata`.

    An array assigned to the image is returned as it is, and stays
    pending until [`consolidate`][] or a write encodes it. Otherwise
    the data is decoded from `rawdata` on first access and cached.
    Data decoded from a file is read-only, because changing it in
    place would not change `rawdata`, which is what the image
    writes. To change the voxels, assign `data`. Assigning `None`
    decodes `rawdata` again.

    The first read converts the lazy array of the file with the
    backend setting in effect and caches the result. Each later read
    returns the cached array converted to the setting in effect then,
    such as `with backend("dask"):`, and the cached array does not
    change.
    """

    @dataobjproperty
    def dataobj(self) -> tx.Optional[ArrayProtocol]:
        """The array of the image in the layout of `data`, without reading it.

        An array assigned as `data` is returned as it is. Otherwise, the
        lazy array of the file is returned through the same view that
        `data` reads, and nothing is read until it is converted.
        """
        if self._data is not None:
            return self._data
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept.
        Assigning other metadata drops the decoded list, so that it is
        decoded from the new header, and keeps an assigned list. A change
        made in place to a decoded list is therefore lost when other
        metadata is assigned. A header modified in place is not noticed,
        so a modified copy should be assigned as new metadata instead.

        Assigning a list replaces the stored one. In both cases, the list
        is compared at [`consolidate`][], and by a write, with the list
        that the header decodes to: the geometry of the header is kept
        when they are the same, exactly, and encoded from the list
        otherwise. An empty list means that the image has no geometry,
        and the image is then written as an image built from data alone.
        Assigning `None` decodes the header again.
        """
        return _header_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else of `rawdata`: an image
        decodes its data from `rawdata` without changing its shape, and a proxy
        knows its shape from the header. An image without data raises as
        [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the header.

        The axes are those of the stored array, in Fortran order, with the
        names and types that the intent code gives them. An image without
        metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        axes = [
            axis
            for axis in _nifti_to_axes(self.metadata.raw.header)
            if axis.name is not None
        ]
        # In F order, the first axis changes fastest.
        return ArrayCoordinateSystem(axes=axes, name="voxel", order="F")

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a NIfTI image.

        A stream that starts with a NIfTI header is scored with
        `_score_nibabel`. Any other stream is declined by
        [`NiftiRaw.sniff_fileobj`][], which raises the requested error.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode. Its position is
            restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no NIfTI
            header. With `True`, the error is a `SnifferContentError`, and
            an exception class is raised instead when one is given.
        **kwargs : Any
            The NIfTI `version` to look for, 1 or 2. Other options are
            ignored.

        Returns
        -------
        float
            The confidence that the stream holds an image.

        Raises
        ------
        SnifferContentError
            If the stream holds no NIfTI header and `error` is `True`.
        """
        header = _sniffed_header(file, kwargs.get("version"))
        if header is None:
            return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)
        return cls._score_nibabel(header)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a NIfTI image.

        The bytes are scored as [`sniff_fileobj`][] scores a stream.

        Parameters
        ----------
        data : bytes
            The content to test, which may be compressed with gzip.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no NIfTI header,
            as in [`sniff_fileobj`][].
        **kwargs : Any
            The NIfTI `version` to look for, 1 or 2. Other options are
            ignored.

        Returns
        -------
        float
            The confidence that the bytes hold an image.

        Raises
        ------
        SnifferContentError
            If the bytes hold no NIfTI header and `error` is `True`.
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a plain image.

        Any NIfTI file can be read as an image, but a displacement-field
        intent suggests that the file is meant as a transformation, so it
        is only a weak match.
        """
        intent = _nifti_intent(header)
        if intent is None:
            return Confidence.MAYBE
        if intent in _NIFTI_FIELD_INTENTS:
            return Confidence.WEAK
        # The intent is often unset, so the shape is checked too: a trailing
        # axis of length 3 on a 4D or larger image may be a deformation.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.WEAK
        if intent == _NIFTI_INTENT_NONE:
            return Confidence.LIKELY
        return Confidence.MAYBE

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "NiftiImage":
        """Read an image from the path of a NIfTI file.

        A local file is handed to nibabel by name, so that nibabel opens
        it whenever the voxels are read and can memory-map them. A remote
        file is read into memory instead. The keyword arguments `mmap` and
        `keep_file_open` are passed to nibabel.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the nibabel readers, which are `mmap` and
            `keep_file_open`. Other options are refused.

        Returns
        -------
        NiftiImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the path does not exist.
        """
        _nibabel_options(kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        raw, proxy = read_nifti(filename, **kwargs)
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NiftiImage":
        """Read an image from an open NIfTI stream.

        The proxy reads the voxels from the stream when they are needed,
        so the stream must stay open while the data may be read. The
        keyword arguments `mmap` and `keep_file_open` are passed to
        nibabel, which ignores them for a stream.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be compressed with gzip.
        **kwargs : Any
            Options of the nibabel readers, which are `mmap` and
            `keep_file_open`. Other options are refused.

        Returns
        -------
        NiftiImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        """
        _nibabel_options(kwargs)
        raw, proxy = read_nifti(file, **kwargs)
        return cls(rawdata=proxy, metadata=NiftiMetadata.from_raw(raw))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> "NiftiImage":
        """Build an image from a nibabel image or header.

        The voxels of an image become `rawdata` as they are, and a copy of its
        header becomes the record of the metadata. A header alone gives an
        image without data.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header. NIfTI-2 images and headers are
            accepted too.
        **kwargs : Any
            Other fields of the image.

        Returns
        -------
        NiftiImage
            The image.

        Raises
        ------
        TypeError
            If `nifti` is neither a NIfTI image nor a NIfTI header.
        """
        if isinstance(nifti, nb.Nifti1Image):
            raw, header = nifti.dataobj, nifti.header
        elif isinstance(nifti, nb.Nifti1Header):
            raw, header = None, nifti
        else:
            raise TypeError(
                f"Expected a NIfTI image or header, got {type(nifti)}"
            )
        metadata = NiftiMetadata.from_raw(NiftiRaw(header=header.copy()))
        return cls(rawdata=raw, metadata=metadata, **kwargs)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        like: tx.Any = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options: tx.Any,
    ) -> "NiftiImage":
        """Encode the pending data and geometry into `rawdata` and the header.

        Consolidating makes `rawdata` and the header describe the image, as a
        file written from it would. A copy of the header of the metadata
        is the base of the new header, and the fields that describe the
        layout of the voxels and the geometry are encoded again over it.
        The data becomes `rawdata`, laid out as NIfTI stores it. An
        override that changes the stored data type or the scaling
        converts the data as a write would, and `rawdata` is then a
        proxy of the converted voxels in memory, so that the copy holds
        what its file holds. The preferred transformation becomes the
        sform, and the qform is a rigid transformation among the others
        or the rigid part of the sform. Large arrays are encoded as
        NIfTI-2, and so is an image whose header is NIfTI-2. The new
        header is held by new metadata, and the header held before is
        never changed. Nothing is pending afterwards, so the
        transformations are decoded from the new header, with the
        precision of the file: the sform is stored in single precision
        and the qform as a quaternion.

        The transformations that the image holds, read or assigned, are
        compared with those that the header decodes to, exactly. When
        they are the same, or were never read, the geometry of the header
        is kept as it is stored, as long as the data have the shape that
        the header records. An image that has a header, no pending data,
        and the transformations of its header, consolidated without
        options, is left as it is: the copy shares `rawdata` and the metadata.
        Consolidating a consolidated image changes nothing.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the options that name its fields, so any field of
        the metadata can be overridden here, and its record is the base of
        the new header. The image owns its geometry, so a field that the
        geometry encodes over, such as the repetition time, which is the
        time step of the image, is reported as approximated when it
        disagrees. The data are stored with the type given as `dtype=`,
        or else with `metadata.dtype`, or else with the type of the data,
        and nibabel scales them into an integer type. What the write
        loses or approximates is acted on once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose description and intent are copied. The geometry
            always comes from this image.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the header cannot hold. By
            default, the policy in effect (see
            [`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy]).
        **options : Any
            Fields of the metadata to override, such as `dtype`, `intent`
            or `description`, and fields of the header, which are set
            last. A header field that holds a field of the metadata, such
            as `descrip` or `scl_slope`, is refused.

        Returns
        -------
        NiftiImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If an option names a header field that holds a field of the
            metadata.
        WriterError
            If the image has no data to encode.
        UnrepresentableTransformationError
            If the preferred transformation is not affine, as for a
            displacement field.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        fields, overrides = _split_options(options)
        report = ConversionReport(source="NiftiImage", target="NiftiImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NiftiMetadata, fields, report
        )
        # The geometry of the header is kept when the transformations
        # were never read, or are the same as those that it decodes to.
        held = held_transformations(self)
        kept = held is None or same_transformations(
            held, _header_transformations(self.metadata)
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
        raw, record = _nifti_encoded(
            self, kept, like, overrides, metadata, fields
        )
        metadata = type(metadata or NiftiMetadata()).from_raw(record)
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

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Build the nibabel image that a file written from the image holds.

        The image is consolidated with the options, as by
        [`consolidate`][], into a copy, and the nibabel image holds the
        `rawdata` and the header of that copy. The image itself is not
        changed.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose description and intent are copied, as in
            [`consolidate`][].
        **overrides : Any
            `on_loss`, and the fields of the metadata and of the header,
            as in [`consolidate`][].

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.

        Raises
        ------
        WriterError
            If the image has no data.
        UnrepresentableTransformationError
            If the preferred transformation is not affine, as for a
            displacement field.
        """
        return _nifti_written(self.consolidate(like=like, **overrides))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a path, compressed if its name ends with `.gz`.

        The image is consolidated into a copy, which is written, and the
        image itself is not changed. A local file is replaced at once, so
        an image can be written over the file that it was read from. The
        image is then consolidated in place, so that it holds what the
        file holds: its lazy array reads the new file, and its
        record is the header of that file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The template `like`, `on_loss` and the options of
            [`consolidate`][].

        Raises
        ------
        WriterError
            If the image cannot be encoded, as in [`consolidate`][].
        """
        if not reads_file(self.rawdata, filename):
            write_nifti(self.to_nibabel(**kwargs), filename)
            return
        self.consolidate(inplace=True, **kwargs)
        write_nifti(_nifti_written(self), filename)
        # The image holds what the file holds: its lazy array reads the new
        # file, since the lazy array of the old file describes how that
        # file stored the voxels, and its record is the header of the new
        # file.
        record, proxy = read_nifti(filename)
        self.rawdata = proxy
        self.metadata = type(self.metadata).from_raw(record)
        self.data = None
        self.transformations = None

    def to_fileobj(self, file: tx.BinaryIO, **kwargs) -> None:
        """Write the image to a stream as an uncompressed NIfTI file.

        The image is consolidated into a copy, which is written, and the
        image itself is not changed.

        Parameters
        ----------
        file : file object
            The stream to write to, open in binary mode.
        **kwargs : Any
            The template `like`, `on_loss` and the options of
            [`consolidate`][].

        Raises
        ------
        WriterError
            If the image cannot be encoded, as in [`consolidate`][].
        """
        write_nifti(self.to_nibabel(**kwargs), file)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the uncompressed NIfTI encoding of the image.

        The image is consolidated into a copy, which is encoded, and the
        image itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The template `like`, `on_loss` and the options of
            [`consolidate`][].

        Returns
        -------
        bytes
            The content of an uncompressed NIfTI file.

        Raises
        ------
        WriterError
            If the image cannot be encoded, as in [`consolidate`][].
        """
        return self.to_nibabel(**kwargs).to_bytes()


def _nifti_to_transformations(
    header: nb.Nifti1Header,
) -> tx.List[Transformation]:
    """Convert a NIfTI header to transformations.

    The transformations are, in order:

    1. A transformation from voxel to scaled voxel space, named "physical";
    2. The qform voxel-to-RAS rigid transformation, named "qform";
    3. The sform voxel-to-RAS affine transformation, named "sform";
    4. The qform voxel-to-RAS rigid transformation, named after its code.
    5. The sform voxel-to-RAS affine transformation, named after its code.

    The code names are `"unknown"`, `"scanner"`, `"aligned"`,
    `"talairach"`, `"mni"` and `"template"`. A form whose code is zero is
    not set and is omitted, along with its entry named after the code.
    When the qform and the sform have the same code, the fourth entry is
    omitted. As a result, the last transformation is the affine that
    nibabel's `get_best_affine()` returns, named after its code.

    The NIfTI affine applies to the spatial axes. When there are other
    axes, each voxel-to-RAS transformation is a `Sequence` made of a
    `SubspaceTransformation` of the spatial axes and, for a time axis, one
    that scales by `pixdim[4]` and translates by `toffset`. Other axes pass
    through, and a time axis without a valid spacing stays a frame index
    in every space. An image with only spatial axes gets a plain `Affine`.
    """

    xforms = []

    # --- preliminaries ------------------------------------------------

    axes = _nifti_to_axes(header)
    # Units go through the single NIfTI-to-brainhops converter, which reads
    # an unknown spatial unit as millimeters and leaves an unknown time
    # unit unspecified (see `brainhops.io.common.nifti._units`).
    space, time = header.get_xyzt_units()
    units = {
        "space": nifti_to_unit(space, "space"),
        "time": nifti_to_unit(time, "time"),
    }
    orientation = {
        "x": "left-to-right",
        "y": "posterior-to-anterior",
        "z": "inferior-to-superior",
    }
    keep_dims = [
        i
        for i, axis in enumerate(axes)
        if axis.name is not None and axis.type == "space"
    ]

    # --- coordinate systems -------------------------------------------

    named_axes = [axis for axis in axes if axis.name is not None]

    # A time axis whose spacing is zero or non-finite (pixdim[4] = 0 when
    # the step is missing) has no TR. It is not mapped to time and stays a
    # frame index in every space.
    zooms = header.get_zooms()
    zooms = [zooms[i] for i, axis in enumerate(axes) if axis.name is not None]
    untimed = {
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and not (np.isfinite(zooms[j]) and zooms[j])
    }
    zooms = [1.0 if j in untimed else z for j, z in enumerate(zooms)]

    # The axes count samples, in Fortran order as nibabel reads them.
    voxel_space = ArrayCoordinateSystem(
        name="voxel", axes=named_axes, order="F"
    )

    # Axes that are neither space nor time, such as channels, have no unit
    # rather than inheriting the index unit.
    phys_axes = [
        axis if j in untimed else replace(axis, unit=units.get(axis.type))
        for j, axis in enumerate(named_axes)
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=orientation[axis.name]
            ),
        )
        if axis.name in orientation
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    # --- voxel-to-physical --------------------------------------------
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=zooms)
    xforms.append(vox2phys)

    # The voxel space holds only the named axes, so these positions are not
    # always the positions in the header.
    named = [i for i, axis in enumerate(axes) if axis.name is not None]
    space_pos = [named.index(i) for i in keep_dims]
    time_pos = [
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and j not in untimed
    ]
    product = bool(space_pos) and len(space_pos) < len(named_axes)

    def _coded_affine(matrix: np.ndarray, label: str) -> Transformation:
        """Build the voxel-to-RAS transformation of a qform or sform."""
        matrix = matrix[keep_dims + [-1], :][:, keep_dims + [-1]]
        if product:
            return _coded_product(matrix, label)
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=label),
            matrix=matrix[:-1],
        )

    def _coded_product(matrix: np.ndarray, label: str) -> Sequence:
        """Build the voxel-to-RAS transformation of an image with more axes.

        The affine maps the spatial axes and the time axis is mapped by its
        own spacing and origin, as a sequence of subspace transformations.
        """
        world = replace(ras_space, name=label)
        # Between the two steps, the spatial axes are in world coordinates
        # while the others are still voxel indices. That space is neither
        # the voxel nor the world space, so it has no name.
        middle_axes = list(voxel_space.axes)
        for j in space_pos:
            middle_axes[j] = world.axes[j]
        middle = CoordinateSystem(axes=middle_axes) if time_pos else world
        spatial = SubspaceTransformation(
            transformation=Affine(
                matrix=matrix[:-1],
                input=voxel_space.restrict(space_pos),
                output=world.restrict(space_pos),
            ),
            input_axes=space_pos,
            output_axes=space_pos,
            input=voxel_space,
            output=middle,
        )
        steps = [spatial]
        if time_pos:
            # t_world = toffset + dt * t_index: the spacing gives the time
            # since the first frame, and `toffset` shifts it to world time.
            (it,) = time_pos
            dt = zooms[it]
            toffset = float(header["toffset"])
            temporal = Sequence(
                transformations=[
                    Scaling(
                        scale=[dt],
                        input=voxel_space.restrict([it]),
                        output=phys_space.restrict([it]),
                    ),
                    Translation(
                        translation=[toffset],
                        input=phys_space.restrict([it]),
                        output=world.restrict([it]),
                    ),
                ],
                input=voxel_space.restrict([it]),
                output=world.restrict([it]),
            )
            steps.append(
                SubspaceTransformation(
                    transformation=temporal,
                    input_axes=[it],
                    output_axes=[it],
                    input=middle,
                    output=world,
                )
            )
        return Sequence(transformations=steps, input=voxel_space, output=world)

    # --- qform --------------------------------------------------------
    # get_qform and get_sform return None for a form whose code is 0, and
    # only one of the two forms is required.
    qmatrix, qcode = header.get_qform(coded=True)
    qform = None
    if qmatrix is not None:
        qname = _NIFTI_XCODES.get(qcode, "unknown")
        qform = _coded_affine(qmatrix, "qform")
        xforms.append(qform)

    # --- sform --------------------------------------------------------
    smatrix, scode = header.get_sform(coded=True)
    sform = None
    if smatrix is not None:
        sname = _NIFTI_XCODES.get(scode, "unknown")
        sform = _coded_affine(smatrix, "sform")
        xforms.append(sform)

    # --- named & best affines -----------------------------------------
    # The last transformation must be nibabel's best affine, named after
    # its code. It is rebuilt under that name rather than renamed, so that
    # the world space is named consistently in every step of a sequence.
    if sform is not None and qform is not None:
        if scode == qcode:
            xforms.append(_coded_affine(smatrix, sname))
        else:
            xforms.append(_coded_affine(qmatrix, qname))
            xforms.append(_coded_affine(smatrix, sname))
    elif sform is not None:
        xforms.append(_coded_affine(smatrix, sname))
    elif qform is not None:
        xforms.append(_coded_affine(qmatrix, qname))

    return xforms


def _nifti_encoded(
    image: NiftiImage,
    kept: bool,
    like: tx.Any,
    overrides: tx.Mapping[str, tx.Any],
    metadata: tx.Optional[NiftiMetadata],
    fields: tx.Mapping[str, tx.Any],
) -> tx.Tuple[ArrayProtocol, NiftiRaw]:
    """Encode the pending state of an image into `rawdata` and a new record.

    The data is the assigned array, or `rawdata` when no array is
    pending. A copy of the record of the consolidated `metadata` is the
    base of the new header, and the data type to store is the `dtype` of
    `fields`, or else that of the metadata of the image. The geometry is
    encoded from the transformations that the image holds, or from those
    that the record decodes to when the image holds none, so that the
    image is not changed. With `kept`, the transformations are those of
    the record, and the geometry fields of the record are copied over the
    new header when the data have the shape that it records, so that they
    are kept as they are stored. When an override changes the data type
    or the scaling, `rawdata` is a proxy of the voxels as the file stores
    them, as `_stored_parts` returns it.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image._data is not None:
        source = _image_to_disk(image._data)
    else:
        source = image.rawdata
    if source is None:
        raise WriterError(
            "This image has no data, so there is nothing to encode."
        )
    if metadata is None or metadata.raw is None:
        header = None
    else:
        header = metadata.raw.copy().header
    dtype = _storage_override(fields, image.metadata, source)
    if dtype is not None:
        overrides = {**overrides, "dtype": dtype}
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _header_transformations(image.metadata)
    preferred = transformations[-1] if transformations else Identity()
    nifti = _image_with_geometry(
        source,
        preferred,
        transformations,
        header=header,
        like=like,
        overrides=overrides,
    )
    _keep_given_fields(nifti, fields)
    stored = None if image.metadata is None else image.metadata.raw
    if kept and header is not None and stored is not None:
        _keep_geometry(nifti.header, stored.header, overrides)
    return _stored_parts(nifti)


def _nifti_written(
    image: NiftiImage,
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Return the nibabel image that writes a consolidated image.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image.rawdata is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    return _written_nifti(image.rawdata, image.metadata.raw)


_NIBABEL_OPTIONS = ("mmap", "keep_file_open")
"""The options of the nibabel readers that a NIfTI image accepts."""


def _nibabel_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the reader options that nibabel does not take.

    Raises
    ------
    TypeError
        If another option is given.
    """
    unknown = [name for name in kwargs if name not in _NIBABEL_OPTIONS]
    if unknown:
        raise TypeError(
            f"Unknown NIfTI reader option(s): {', '.join(unknown)}"
        )


def _header_transformations(
    metadata: tx.Optional[NiftiMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the header of some metadata.

    Metadata without a header gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _nifti_to_transformations(metadata.raw.header)


def _keep_geometry(
    header: nb.Nifti1Header,
    stored: nb.Nifti1Header,
    overrides: tx.Mapping[str, tx.Any],
) -> None:
    """Copy the geometry fields of a stored header over a new header.

    The fields are copied only when the new header has the dimensions of
    the stored one, and a field that an override sets is left as it is.
    The geometry that the new header encodes from the same transformations
    is then the one that the file stores, byte for byte.
    """
    if not np.array_equal(header["dim"], stored["dim"]):
        return
    for name in (
        "pixdim",
        "xyzt_units",
        "toffset",
        "qform_code",
        "sform_code",
        "quatern_b",
        "quatern_c",
        "quatern_d",
        "qoffset_x",
        "qoffset_y",
        "qoffset_z",
        "srow_x",
        "srow_y",
        "srow_z",
    ):
        if name not in overrides:
            header[name] = stored[name]
