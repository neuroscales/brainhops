"""The image sequence format, which is read only."""

# stdlib
import os
import warnings

# dependencies
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, ParserTypeError
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._data import _SequenceProxy
from ._metadata import SequenceMetadata
from ._raw import (
    NOT_A_STREAM,
    SequenceRaw,
    refuse_options,
    refuse_stream,
    sniff_folder,
    sniff_score,
)
from ._views import _image_to_disk, _image_to_model, _permutation


@register_format
class SequenceImage(ImageFormat, SingleScaleImage, BinaryFileReader):
    """A volume stored as a folder of numbered two-dimensional images.

    Light-sheet microscopes, electron microscopes and many export tools
    write a volume as one image file per slice, such as `z0000.tif`,
    `z0001.tif` and so on. This class reads such a folder as one image.
    The files are TIFF files, read with tifffile, or images in a format
    that Pillow reads, such as PNG. The folder is recognized when it is
    passed to [`load`][brainhops.io.load] (see
    [`brainhops.io.images.sequence`][] for the rules), and the format hint
    `"sequence"` selects this class.

    The image holds the list of its files and their headers as
    [`SequenceMetadata`][], whose record is a [`SequenceRaw`][], and its
    pixels in `raw`, a lazy array that reads each file only when one of
    its pixels is needed. The files are stacked along the slowest axis,
    z, so the data are indexed `(x, y, z)`, or `(x, y, z, c)` for files
    with channels.

    The image has one transformation, a [`Scaling`][] from voxels to
    `"physical"`. The size of a pixel along x and y is the one that every
    TIFF file records, through OME-XML, ImageJ metadata or resolution
    tags, and the spacing of the slices along z is the ImageJ `spacing`
    that every file records. A size that the files do not record, or on
    which they disagree, is unknown: the scaling is 1 along that axis,
    without a unit, as for a single TIFF or Pillow image. The options
    `pixel_size` and `unit` set the sizes instead, so that
    `pixel_size={"z": 2.0}` gives the spacing of the slices.

    Image sequences cannot be written, so the class has no writer: an
    image is saved to another format, such as NIfTI or a multi-page TIFF,
    and [`save`][brainhops.io.save] refuses to write it as a sequence.
    Assigned data are pending until [`consolidate`][] moves them into
    `raw`, and the transformations that the image holds, read or
    assigned, are kept, since no record can be encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("sequence",)

    raw: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The pixels of the files stacked along a first axis, or `None`.

    After a read, `raw` is a lazy array of shape `(files, rows, columns[,
    channels])`. Assigning `data` leaves `raw` as it is, and
    [`consolidate`][] stores the assigned array here, with its spatial
    axes reversed.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
    """The data given to the image, or `None` to decode it from `raw`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending until [`consolidate`][] moves it into
    `raw`.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The transformations of the image, or `None` when never read or set.

    The image takes over the `transformations` field of
    [`SingleScaleImage`][] to store it under this private name. The
    constructor still takes `transformations` in the same place, and the
    property of the same name reads it. `None`, the default, means that
    the transformations have been neither read nor assigned. The property
    stores here the list that it decodes, and an assigned list, even an
    empty one, is stored as it is.
    """

    _metadata: NoRepr[tx.Optional[SequenceMetadata]] = None

    metadata = smartproperty("metadata")
    """The metadata of the sequence, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations that the image holds, read or assigned, so
    that they are decoded from the new record.
    """

    @metadata.setter
    def metadata(self, value: tx.Optional[SequenceMetadata]) -> None:
        self._metadata = value
        self._transformations = None

    pixel_size: tx.Optional[tx.Any] = None
    """The size of a voxel given when the sequence was read, or `None`.

    It is one size for every spatial axis, one size per spatial axis in
    the order x, y and z, or a mapping that sets the named axes only, such
    as `{"z": 2.0}`. It takes precedence over the sizes that the files
    record, when the transformations are decoded from the record.
    """

    unit: tx.Optional[tx.Any] = None
    """The unit of `pixel_size`, or the unit to which the sizes that the
    files record are converted, or `None`.

    Without a unit, `pixel_size` is in the unit that the files record, or
    has no unit if they record none.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The pixels as `(x, y, z[, c])`, assigned or decoded from `raw`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `raw`. Otherwise the
        data is decoded from `raw` on first access and cached. With NumPy
        as the array backend, every file is read and the data is
        read-only. With Dask, the data is a Dask array with one chunk per
        file, and each file is read when its chunk is computed. To change
        the pixels, assign `data`. Assigning `None` decodes `raw` again.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @smartproperty
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access, with the options
        `pixel_size` and `unit`, and stored in the image, so
        `img.transformation is img.transformation`, and a change made to
        the list in place, such as an appended transformation, is kept.
        Assigning other metadata drops the list. An image without metadata
        has no transformations. Assigning a list replaces the stored one,
        and assigning `None` decodes the record again.
        """
        self._transformations = _record_transformations(
            self.metadata, self.pixel_size, self.unit
        )
        return self._transformations

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the files.

        The shape is that of the assigned data, or else the shape of `raw`
        with its spatial axes reversed. The lazy array knows its shape
        from the record. An image without data raises as
        [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.raw is None:
            return self.data.shape
        stored = tuple(int(d) for d in self.raw.shape)
        return tuple(stored[i] for i in _permutation(len(stored)))

    def consolidate(self, *, inplace: bool = False) -> "SequenceImage":
        """Move the pending data into `raw`.

        Image sequences cannot be written, so no record can be encoded,
        and consolidating a sequence is limited to its data: an assigned
        array becomes `raw`, with its spatial axes reversed, and is no
        longer pending. The metadata is kept, and so are the
        transformations that the image holds, read or assigned. An image
        without pending data is left as it is: the copy shares `raw` and
        the metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.

        Returns
        -------
        SequenceImage
            The consolidated copy, or this image with `inplace=True`.
        """
        if self._data is None:
            return self if inplace else replace(self)
        raw = _image_to_disk(self._data)
        if not inplace:
            return replace(self, data=None, raw=raw)
        self.raw = raw
        self.data = None
        return self

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        pattern: tx.Optional[str] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a path is the folder of a sequence.

        A local folder that is not a Zarr store, and whose image files
        (or the files that `pattern` selects) are numbered without a gap,
        scores `LIKELY`. Only the names of the files are checked, so no
        file is opened. A file never scores, so this class never takes a
        file from the TIFF and Pillow readers.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        error : bool or type[Exception], default=False
            Whether to raise an error when the path is not the folder of a
            sequence.
        pattern : str, optional
            The glob pattern that selects the files of the sequence.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `LIKELY` or `NO`.
        """
        return sniff_score(sniff_folder(filename, pattern), error)

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
            Whether to raise an error, since a stream is never a sequence.
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
            Whether to raise an error, since bytes are never a sequence.
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
        pixel_size: tx.Optional[tx.Any] = None,
        unit: tx.Optional[tx.Any] = None,
        **kwargs,
    ) -> "SequenceImage":
        """Read the image sequence in a folder, without reading pixels.

        The header of every file is read to check that the files can be
        stacked. The lazy array reads the pixels of a file, by name, when
        they are needed.

        Parameters
        ----------
        filename : path-like
            The path of a local folder.
        pattern : str, optional
            The glob pattern, relative to the folder, that selects the
            files of the sequence, such as `"slice_*.tif"`. By default,
            every image file of the folder is taken, except hidden files.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The size of a voxel, which takes precedence over the sizes that
            the files record. It is one size for every spatial axis, one
            size per spatial axis in the order x, y and z, or a mapping
            that sets the named axes only, such as `{"z": 2.0}` for the
            spacing of the slices.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the sizes that
            the files record are converted.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        SequenceImage
            The image, whose pixels are read lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ValueError
            If `pixel_size` or `unit` is not valid.
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the folder does not hold a sequence that can be read: the
            names of its image files do not follow one numbering, a number
            is missing or repeated, or the files differ in format, shape,
            type or number of channels.
        """
        refuse_options(cls, kwargs)
        record = SequenceRaw.from_folder(filename, pattern)
        _check_options(record, pixel_size, unit)
        if isinstance(filename, str):
            filename = path.Path(filename)
        folder = os.path.abspath(os.fspath(filename))
        return cls(
            raw=_SequenceProxy(folder, record),
            metadata=SequenceMetadata.from_raw(record),
            pixel_size=pixel_size,
            unit=unit,
        )

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "SequenceImage":
        """Refuse a stream, which cannot hold the files of a sequence.

        Parameters
        ----------
        file : IO
            The stream.
        **kwargs : Any
            Options of the reader, which are ignored.

        Raises
        ------
        ParserTypeError
            Always, since a sequence is read from the path of its folder.
        """
        raise ParserTypeError(NOT_A_STREAM)


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def _check_options(
    record: SequenceRaw, pixel_size: tx.Any, unit: tx.Any
) -> None:
    """Check the options of the geometry against the axes of a record.

    Raises
    ------
    ValueError
        If a size is not positive and finite, if `pixel_size` does not
        match the spatial axes, or if `unit` is not a unit.
    """
    raster.resolve_pixel_size(
        _model_axes(record), pixel_size=pixel_size, unit=unit
    )


def _record_transformations(
    metadata: tx.Optional[SequenceMetadata],
    pixel_size: tx.Any,
    unit: tx.Any,
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record, or a record without files, gives no
    transformation.
    """
    if metadata is None or metadata.raw is None or not metadata.raw.members:
        return []
    return _sequence_transformations(metadata.raw, pixel_size, unit)


def _sequence_transformations(
    record: SequenceRaw, pixel_size: tx.Any, unit: tx.Any
) -> tx.List[Transformation]:
    """Return the voxel-to-physical scaling of a sequence.

    The sizes that every file records agree on are used, the options take
    precedence over them, and a negative ImageJ spacing flips z unless
    `pixel_size` sets z.
    """
    axes = _model_axes(record)
    recorded, signs = _recorded_scales(record)
    scales = raster.resolve_pixel_size(
        axes, recorded, pixel_size=pixel_size, unit=unit
    )
    if isinstance(pixel_size, tx.Mapping):
        overridden = set(pixel_size)
    elif pixel_size is not None:
        overridden = {axis.name for axis in raster.spatial_axes(axes)}
    else:
        overridden = set()
    for name, sign in signs.items():
        if name in scales and name not in overridden:
            size, size_unit = scales[name]
            scales[name] = (sign * size, size_unit)
    return raster.raster_transformations(axes, scales)


def _model_axes(record: SequenceRaw) -> tx.List[tx.Any]:
    """Return the axes of the data of a sequence, `x, y, z[, c]`."""
    axes = raster.storage_axes(record.axes)
    return [axes[p] for p in raster.canonical_permutation(axes)]


def _recorded_scales(
    record: SequenceRaw,
) -> tx.Tuple[tx.Dict[str, raster.AxisScale], tx.Dict[str, float]]:
    """Return the sizes and signs on which every file of a sequence agrees.

    A TIFF file records the size of its pixels along x and y, as the TIFF
    reader reads it (`series_metadata`), and an ImageJ file may also
    record the spacing of its slices, which is the size along z. A Pillow
    file records no size that the Pillow reader uses by default. A size
    that some files do not record, or on which they disagree, is left
    out, with a warning when some files record it.
    """
    per_file = [_file_scales(member) for member in record.members]
    scales: tx.Dict[str, raster.AxisScale] = {}
    signs: tx.Dict[str, float] = {}
    for name in ("x", "y", "z"):
        found = [file_scales.get(name) for file_scales, _ in per_file]
        if all(scale is None for scale in found):
            continue
        if any(scale != found[0] for scale in found):
            warnings.warn(
                f"The files of the sequence do not all record the same "
                f"size along {name}, so that size is unknown. Pass "
                f"pixel_size= to set it.",
                stacklevel=3,
            )
            continue
        scales[name] = found[0]
        file_signs = {file_signs.get(name) for _, file_signs in per_file}
        if file_signs == {-1.0}:
            signs[name] = -1.0
    return scales, signs


def _file_scales(
    member: tx.Any,
) -> tx.Tuple[tx.Dict[str, raster.AxisScale], tx.Dict[str, float]]:
    """Return the sizes that one file records, and the signs of its steps.

    The sizes along x and y are those that the TIFF reader takes from the
    file, and the size along z is the ImageJ `spacing`, read with the
    function that the TIFF reader uses for ImageJ hyperstacks.
    """
    if member.format != "tiff":
        return {}, {}
    # A TIFF member exists only when tifffile is installed, so the TIFF
    # package, which registers its formats when imported, can be imported.
    from brainhops.io.images.tiff import _utils as tiff

    header = member.header
    scales = {
        name: scale
        for name, scale in header.scales.items()
        if name in ("x", "y")
    }
    spacing, signs = tiff.imagej_scales(header.imagej, None, ("z",))
    scales.update(spacing)
    return scales, signs
