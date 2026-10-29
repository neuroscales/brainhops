"""One resolution level of one view of a BigDataViewer dataset, read only."""

__all__ = ["BdvImage"]

# stdlib

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core._internal_api import ArrayProtocol, smartproperty
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferContentError,
    disabled,
)
from brainhops.io.common._internal_api import (
    held_transformations,
    same_transformations,
)
from brainhops.io.common.bdv import BdvMetadata, BdvRaw
from brainhops.io.common.bdv._internal_api import (
    global_system,
    level_count,
    level_matrix,
    physical_system,
    read_dataset,
    registration_matrix,
    voxel_system,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._views import image_to_disk, image_to_model


@register_format
class BdvImage(BinaryFileReader, ImageFormat, SingleScaleImage):
    """One resolution level of one view of a BigDataViewer dataset.

    A BigDataViewer dataset is an XML file and an HDF5 file (see
    [`brainhops.io.common.bdv`][]). A view is one setup, such as a
    channel, an angle or a tile, at one time point, and each view is
    stored at several resolutions. This class reads one level of one
    view, which the reader options `setup`, `timepoint` and `level`
    select. By default it reads the full resolution of the first setup
    at the first time point, in the order of their ids, which is setup 0
    and time point 0 in the datasets that BigDataViewer and BigStitcher
    write. [`BdvMultiScaleImage`][brainhops.io.images.bdv.BdvMultiScaleImage]
    reads every level of a view, and it is what `load` returns for a view
    with several levels unless `level` is given.

    The image holds the XML file as [`BdvMetadata`][], whose record, a
    [`BdvRaw`][], also holds the resolution levels that the HDF5 file
    describes. The voxels are in `rawdata`, a lazy array that reads them only
    when `data` is first needed, in the order z, y, x of the file. The
    data reverse it into the order x, y, z of ImgLib2.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. They are, in order, an
    [`Affine`][] to `"physical"`, which scales the voxels of the level by
    the voxel size of the setup, in its unit, and the preferred
    [`Affine`][] to `"global"`, the registration of the view, which is
    the product of its transforms, as BigDataViewer composes them. Both
    include the downsampling of the level, which places voxel `i` of a
    level downsampled by `f` at `f * i + (f - 1) / 2` in full-resolution
    voxels. The global space has no unit, because BigStitcher measures it
    in units of the smallest voxel size of the dataset. A setup without a
    voxel size has no `"physical"` transformation.

    The class cannot write, because a view is one dataset among many in
    an HDF5 file that BigDataViewer writes chunked and compressed, with
    every level, and an XML file that describes the whole acquisition.
    An image is saved to another format, such as NIfTI, and
    [`save`][brainhops.io.save] refuses to write it as a BigDataViewer
    dataset. The registration of a view is written by
    [`BdvViewRegistration`][brainhops.io.transformations.bdv.BdvViewRegistration].
    Assigned data is pending until [`consolidate`][] moves it into `rawdata`,
    and the transformations that the image holds, read or assigned, are
    kept, since no record is encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("bdv", "bigdataviewer")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels in the order of the file, or `None` without data.

    After a read, `rawdata` is a lazy array, which reads the voxels when it is
    read. Assigning `data` leaves `rawdata` as it is, and
    [`consolidate`][] stores the assigned array here, with its axes
    reversed.
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

    _metadata: NoRepr[tx.Optional[BdvMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the dataset, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    setup: tx.Optional[int] = None
    """The id of the setup of the view that the image holds.

    It is the reader option `setup`. A reader stores the id of the setup
    that it read, and `None` stands for the first setup.
    """

    timepoint: tx.Optional[int] = None
    """The id of the time point of the view that the image holds.

    It is the reader option `timepoint`. A reader stores the id of the
    time point that it read, and `None` stands for the first one.
    """

    level: int = 0
    """The resolution level that the image holds, from 0 for the finest.

    It is the reader option `level`.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels in the order x, y, z, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the voxels, assign `data`, and assigning `None`
        gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place is kept. Assigning other
        metadata drops the decoded list, so that it is decoded from the
        new record, and keeps an assigned list. A change made in place to
        a decoded list is therefore lost when other metadata is assigned.
        An image without metadata has no transformations. Assigning a list
        replaces the stored one, and assigning `None` decodes the record
        again.
        """
        return _record_transformations(self)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else the reverse of the
        shape of `rawdata`, which the lazy array knows from the HDF5 file. An
        image without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in reversed(self.rawdata.shape))

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> "BdvImage":
        """Move the pending data into `rawdata`.

        A BigDataViewer image cannot be written, so no record can be
        encoded, and consolidating it is limited to its data: an assigned
        array becomes `rawdata`, with its axes reversed, and is no
        longer pending. The metadata is kept, and so are the
        transformations that the image holds, read or assigned. An image
        without pending data is left as it is: the copy shares `rawdata`
        and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a
        BigDataViewer dataset holds no field of the vocabulary yet, so a
        field that is set is reported as lost, once, under the error
        policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        BdvImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(source="BdvImage", target="BdvImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, BdvMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        # The record cannot be encoded, so a copy is given the list that
        # the image holds when it is not the list that the record decodes
        # to, as a list decoded and then changed in place would otherwise
        # be decoded again and lost.
        held = held_transformations(self)
        if held is None or same_transformations(
            held, _record_transformations(self)
        ):
            held = self._transformations
        if self._data is None:
            if not inplace:
                return replace(self, transformations=held, metadata=metadata)
            if requested:
                self.transformations = held
                self.metadata = metadata
            return self
        raw = image_to_disk(self._data)
        if not inplace:
            return replace(
                self,
                data=None,
                rawdata=raw,
                transformations=held,
                metadata=metadata,
            )
        self.rawdata = raw
        self.data = None
        if requested:
            self.transformations = held
            self.metadata = metadata
        return self

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        setup: tx.Optional[int] = None,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score the XML file of a dataset at a path.

        The score is `CERTAIN` for a dataset stored in HDF5 when `level`
        is given or when the setup has a single level, and `LIKELY`
        otherwise, so that
        [`BdvMultiScaleImage`][brainhops.io.images.bdv.BdvMultiScaleImage]
        reads a view with several levels.

        Parameters
        ----------
        filename : path-like
            The path of the XML file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a dataset that
            this class reads.
        setup : int, optional
            The setup that the caller asks for.
        level : int, optional
            The level that the caller asks for.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = sniff_path(filename, error)
        if not score or level is not None:
            return score
        if level_count(filename, setup) == 1:
            return Confidence.CERTAIN
        return Confidence.LIKELY

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score a stream from the XML of a dataset.

        A stream has no location against which to find the HDF5 file, so
        the score is `CERTAIN` when `level` is given and `LIKELY`
        otherwise, for a dataset stored in HDF5. Reading the stream then
        fails with a message that says to read the path of the file.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a dataset
            that this class reads.
        level : int, optional
            The level that the caller asks for.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = sniff_stream(file, error)
        if not score or level is not None:
            return score
        return Confidence.LIKELY

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        setup: tx.Optional[int] = None,
        timepoint: tx.Optional[int] = None,
        level: int = 0,
        **kwargs,
    ) -> "BdvImage":
        """Read one level of one view from the XML file of a dataset.

        The XML file is read, and the HDF5 file that it names is opened
        for the resolution levels and for the shape of the level, but no
        voxel is read. The lazy array reads the voxels from the HDF5 file,
        by name, when they are needed.

        Parameters
        ----------
        filename : path-like
            The path of the XML file.
        setup : int, optional
            The id of the setup of the view, by default the first one.
        timepoint : int, optional
            The id of the time point of the view, by default the first
            one.
        level : int, default=0
            The resolution level, from 0 for the finest.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        BdvImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserExistsError
            If the XML file or the HDF5 file does not exist.
        ParserContentError
            If the dataset is not stored in HDF5, or if it has no such
            view or level.
        """
        no_options(cls, kwargs)
        raw, setup, timepoint, levels = read_dataset(
            filename, setup=setup, timepoint=timepoint
        )
        level = int(level)
        if not 0 <= level < len(levels):
            raise ParserContentError(
                f"Setup {setup} of this BigDataViewer dataset has "
                f"{len(levels)} level(s), so it has no level {level}."
            )
        return cls(
            rawdata=levels[level],
            metadata=BdvMetadata.from_raw(raw),
            setup=setup,
            timepoint=timepoint,
            level=level,
        )

    from_fileobj = disabled(
        ParserContentError,
        "A BigDataViewer dataset is an XML file and the HDF5 file that "
        "it names, so an image is read from the path of the XML file, "
        "not from a stream or from bytes.",
    )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse unknown reader options.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(kwargs))}."
        )


def _hdf5_dataset(content: bytes) -> bool:
    """Tell whether bytes are the XML file of a dataset stored in HDF5."""
    try:
        return BdvRaw.from_bytes(content).loader == "bdv.hdf5"
    except Exception:  # noqa: BLE001
        return False


def _score(found: bool, error: tx.Union[bool, tx.Type[Exception]]) -> float:
    """Return `CERTAIN` for a dataset stored in HDF5, or decline.

    Raises
    ------
    SnifferContentError
        If the content is not such a dataset and `error` is set.
    """
    if found:
        return Confidence.CERTAIN
    if error:
        if error is True:
            error = SnifferContentError
        raise error(
            "The content is not the XML file of a BigDataViewer dataset "
            "stored in HDF5."
        )
    return Confidence.NO


def sniff_path(
    filename: path.FilenameLike, error: tx.Union[bool, tx.Type[Exception]]
) -> float:
    """Score the XML file at a path as a dataset stored in HDF5."""
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not path.exists(filename):
        return _score(False, error)
    with filename.open("rb") as file:
        return sniff_stream(file, error)


def sniff_stream(
    file: tx.IO, error: tx.Union[bool, tx.Type[Exception]]
) -> float:
    """Score a binary stream as the XML file of a dataset stored in HDF5.

    The root element is checked first, so that a file that is not a
    dataset is declined without being read whole.
    """
    position = file.tell()
    try:
        head = file.read(4096)
        if BdvRaw.sniff_bytes(head) == Confidence.NO:
            return _score(False, error)
        return _score(_hdf5_dataset(head + file.read()), error)
    finally:
        file.seek(position)


def _record_transformations(image: BdvImage) -> tx.List[Transformation]:
    """Decode the transformations of an image from its record.

    An image without a record has no transformation.

    Raises
    ------
    ParserContentError
        If the record does not describe the view or the level, or if the
        registration has a generic transform.
    """
    metadata = image.metadata
    if metadata is None or metadata.raw is None:
        return []
    raw = metadata.raw
    setup, timepoint = raw.view(image.setup, image.timepoint)
    resolutions = raw.pyramid(setup).resolutions
    if not 0 <= image.level < len(resolutions):
        raise ParserContentError(
            f"Setup {setup} of this BigDataViewer dataset has "
            f"{len(resolutions)} level(s), so it has no level {image.level}."
        )
    level = level_matrix(resolutions[image.level])
    try:
        world = registration_matrix(raw.registration(timepoint, setup))
    except ValueError as error:
        raise ParserContentError(str(error)) from error
    voxel = voxel_system()
    xforms: tx.List[Transformation] = []
    item = raw.setup(setup)
    if item.voxel_size is not None:
        scale = np.diag(list(item.voxel_size) + [1.0])
        xforms.append(
            Affine(
                input=voxel,
                output=physical_system(item.unit),
                matrix=(scale @ level)[:3],
            )
        )
    xforms.append(
        Affine(input=voxel, output=global_system(), matrix=(world @ level)[:3])
    )
    return xforms
