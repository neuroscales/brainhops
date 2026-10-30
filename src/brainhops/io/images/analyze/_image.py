"""The Analyze 7.5 image format, which is read only."""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import ArrayCoordinateSystem, CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base._internal_api import no_options
from brainhops.io.base.parsers import (
    BinaryFileReader,
    ParserContentError,
    ParserExistsError,
    disabled,
)
from brainhops.io.common.analyze import AnalyzeMetadata, AnalyzeRaw
from brainhops.io.common.analyze._internal_api import (
    analyze_files,
    analyze_to_ras,
)
from brainhops.io.common.nifti._internal_api import NIFTI_AXES, NiftiProxy
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.nifti._internal_api import (
    image_to_disk,
    image_to_model,
)
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)

_OPTION_OWNER = "Analyze reader"
"""What refuses unknown options, as the error message names it."""

_PHYSICAL = "physical"
"""Name of the space of the voxels scaled by their size."""

_WORLD = "world"
"""Name of the RAS world space, the preferred world space."""

_MM = "mm"
"""The unit of the voxel sizes of the spatial axes."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}


@register_format
class AnalyzeImage(BinaryFileReader, ImageFormat, SingleScaleImage):
    """An image stored as an Analyze 7.5 pair of files.

    Old data sets, and the files that SPM99 and SPM2 write, store their
    images as a header file (`.hdr`) and an image file (`.img`). This
    class reads such a pair from the path of either file, and answers to
    the format hint `"analyze"`. A header with the NIfTI-1 magic number
    belongs to a NIfTI pair and is refused, because
    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] reads it.

    An Analyze image holds the header as [`AnalyzeMetadata`][], whose
    record is an [`AnalyzeRaw`][], and the voxels in `rawdata`, a lazy
    array that reads them from the image file only when they are
    needed. The voxels are stored as in a NIfTI file, so the lazy array
    is the one of NIfTI, and the data are in the same Fortran order. The
    voxels are scaled by the scale factor and the intercept that SPM2
    stores in the unused fields of the header (`funused1` and
    `funused2`), as nibabel scales them.

    The transformations are decoded from the header on first access,
    unless other transformations are assigned. They are, in order, a
    [`Scaling`][] to `"physical"` by the voxel size of each spatial axis,
    in millimetres, and the preferred [`Affine`][] to the RAS space
    `"world"`. Analyze stores no orientation, so the affine follows the
    convention of SPM: the first axis runs from right to left, and the
    origin of the world is the voxel that SPM stores in the header, or
    else the centre of the grid (see
    [`brainhops.io.common.analyze`][brainhops.io.common.analyze]).

    Analyze files cannot be written by brainhops. The format would need
    an encoder that turns any transformation into voxel sizes and an
    origin, and reports what it loses, while NIfTI stores the same
    voxels with a full geometry. An image is therefore saved to another
    format, such as NIfTI, and [`save`][brainhops.io.save] refuses to
    write it as Analyze. Assigned data is pending until
    [`consolidate`][] moves it into `rawdata`, and the transformations
    that the image holds, read or assigned, are kept, since no record
    can be encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("analyze",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".hdr", ".img")

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels as the image file stores them, or `None` without data.

    After a read, `rawdata` is a lazy array of the voxels of the image
    file, which applies the scaling of the header when it is read.
    Assigning `data` leaves `rawdata` as it is, and [`consolidate`][]
    stores the assigned array here.
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
    was assigned, so the property decodes the list of the header and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[AnalyzeMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the image, which holds its header, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous header, so that
    they are decoded from the new one, and keeps the transformations
    that were assigned.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels in Fortran order, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`.
        Otherwise, the data are the lazy array of the image file, which
        reads its values when a computation needs them and refuses an
        edit in place. To change the voxels, assign `data`, and assigning
        `None` gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept. Assigning other metadata drops the
        decoded list, so that it is decoded from the new header, and
        keeps an assigned list. An image without metadata has no
        transformations. Assigning `None` decodes the header again.
        """
        return _header_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else of `rawdata`,
        whose lazy array knows its shape from the header. An image
        without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the header.

        The axes are named `x`, `y`, `z` and `t`, in Fortran order, as for
        a NIfTI image. An image without metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return _voxel_system(self.metadata.raw)

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> "AnalyzeImage":
        """Move the pending data into `rawdata`.

        Analyze files cannot be written, so no header can be encoded, and
        consolidating an Analyze image is limited to its data: an assigned
        array becomes `rawdata` and is no longer pending. The metadata is
        kept, and so are the transformations that the image holds, read
        or assigned. An image without pending data is left as it is: the
        copy shares `rawdata` and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The header of an
        Analyze image holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, under the error policy of
        the call, and the copy holds the consolidated metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        AnalyzeImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(source="AnalyzeImage", target="AnalyzeImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, AnalyzeMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        if self._data is None and not requested:
            return self if inplace else replace(self)
        raw = self.rawdata if self._data is None else image_to_disk(self._data)
        if not inplace:
            return replace(self, data=None, rawdata=raw, metadata=metadata)
        self.rawdata = raw
        self.data = None
        self.metadata = metadata
        return self

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names an Analyze image.

        The header file of the pair is tested, whichever file of the pair
        the path names.

        Parameters
        ----------
        filename : path-like
            The path of the header file or of the image file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the path names no Analyze
            image.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AnalyzeRaw.sniff_filename(filename, error=error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an Analyze header.

        The stream is scored as [`AnalyzeRaw.sniff_fileobj`][] scores it,
        so a NIfTI-1 header is declined.

        Parameters
        ----------
        file : IO
            The stream, which may be compressed with gzip.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no Analyze
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AnalyzeRaw.sniff_fileobj(file, error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "AnalyzeImage":
        """Read an image from the path of its header file or image file.

        The header is read from the header file, and the image file is
        checked but not read: the lazy array reads it, by name, when the
        voxels are needed. A file that is not local is read into memory
        when the voxels are read.

        Parameters
        ----------
        filename : path-like
            The path of the header file or of the image file.
        mmap : bool, default=True
            Whether the lazy array memory-maps a local image file that is
            not compressed.
        **kwargs : Any
            Refused, since Analyze has no other option.

        Returns
        -------
        AnalyzeImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the header file or the image file does not exist.
        ParserContentError
            If the path names no file of an Analyze pair, or if the header
            is a NIfTI-1 header.
        """
        no_options(kwargs, _OPTION_OWNER)
        header, image = analyze_files(filename)
        for name in (header, image):
            if not name.exists():
                raise ParserExistsError(f"No such file: {name}")
        record = AnalyzeRaw.from_filename(header)
        return _read(record, image, mmap=mmap)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "AnalyzeImage":
        """Read an image from an open header file.

        The image file is found from the name of the stream, which must
        have one, and is opened by name when the voxels are needed,
        without memory mapping. The position of the stream is restored.

        Parameters
        ----------
        file : BinaryIO
            The open header file, which must have a name.
        **kwargs : Any
            Refused, since Analyze has no option for a stream.

        Returns
        -------
        AnalyzeImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream has no file name, or if it holds a NIfTI-1
            header or no Analyze header.
        ParserExistsError
            If the image file does not exist.
        """
        no_options(kwargs, _OPTION_OWNER)
        name = getattr(file, "name", None)
        if not isinstance(name, (str, bytes, os.PathLike)):
            raise ParserContentError(
                "An Analyze image is two files (.hdr and .img), and this "
                "stream has no file name to find the image file from. "
                "Read the image from its path instead."
            )
        record = AnalyzeRaw.from_fileobj(file)
        _, image = analyze_files(os.fsdecode(name))
        if not image.exists():
            raise ParserExistsError(f"No such file: {image}")
        return _read(record, image, mmap=False)

    from_bytes = disabled(
        ParserContentError,
        "An Analyze image is two files (.hdr and .img), which bytes alone "
        "cannot hold. Read it from its path instead.",
    )


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read(record: AnalyzeRaw, image: path.Path, mmap: bool) -> AnalyzeImage:
    """Build the image of a header and of the path of its image file.

    The lazy array is the one of NIfTI, which reads the shape, the type,
    the byte order, the offset and the scaling of the voxels from the
    header, as it reads them from a NIfTI header.
    """
    source = image if image.is_local() else BytesIO(image.read_bytes())
    proxy = NiftiProxy.from_header(source, record.header, mmap=mmap)
    return AnalyzeImage(
        rawdata=proxy, metadata=AnalyzeMetadata.from_raw(record)
    )


def _voxel_system(record: AnalyzeRaw) -> ArrayCoordinateSystem:
    """Return the voxel system of a record, in Fortran order."""
    ndim = len(record.header.get_data_shape())
    axes = [replace(axis) for axis in NIFTI_AXES[:ndim]]
    return ArrayCoordinateSystem(name="voxel", axes=axes, order="F")


def _header_transformations(
    metadata: tx.Optional[AnalyzeMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the header of some metadata.

    Metadata without a header gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _analyze_to_transformations(metadata.raw)


def _analyze_to_transformations(
    record: AnalyzeRaw,
) -> tx.List[Transformation]:
    """Return the voxel-to-physical and voxel-to-world transformations.

    The first three axes are spatial. Other axes, such as time, are not
    scaled, because Analyze gives no unit to their step, and they have
    zero columns in the matrix of the affine, so they do not move a world
    point.
    """
    header = record.header
    voxel_space = _voxel_system(record)
    axes = list(voxel_space.axes)
    spatial = [i for i, axis in enumerate(axes) if axis.type == "space"]

    zooms = [float(z) for z in header.get_zooms()]
    scale = [zooms[i] if i in spatial else 1.0 for i in range(len(axes))]
    phys_axes = [
        replace(axis, unit=_MM) if i in spatial else axis
        for i, axis in enumerate(axes)
    ]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    xforms = [Scaling(input=voxel_space, output=phys_space, scale=scale)]

    world_axes = [
        Axis(
            name,
            "space",
            unit=_MM,
            orientation=Orientation(type="anatomical", value=value),
        )
        for name, value in _RAS_ORIENTATION.items()
    ]
    world_space = CoordinateSystem(name=_WORLD, axes=world_axes)
    vox2ras = analyze_to_ras(header)
    matrix = np.zeros((3, len(axes) + 1))
    matrix[:, spatial] = vox2ras[:3, spatial]
    matrix[:, -1] = vox2ras[:3, 3]
    xforms.append(Affine(input=voxel_space, output=world_space, matrix=matrix))
    return xforms
