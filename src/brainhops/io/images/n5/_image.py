"""One N5 dataset, or one level of an N5 pyramid, read as an image."""

# dependencies
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_TENSORSTORE
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferTypeError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base._utils_raster import (
    axis_group,
    default_axes,
    level_transformation,
)
from brainhops.io.metadata._base import metadata_to_write, requested_fields

# this format
from ._data import (
    READABLE_COMPRESSIONS,
    _N5Proxy,
    dataset_location,
    read_node,
    stream_location,
)
from ._metadata import N5Metadata
from ._raw import N5Raw, no_options, score
from ._spatial import placement
from ._views import _image_to_disk, _image_to_model


class N5Image(ImageFormat, SingleScaleImage, BinaryFileReader):
    """One N5 dataset, or one level of an N5 pyramid.

    An N5 container is a directory tree in which a dataset is a directory
    that holds an `attributes.json` file and the blocks of an array (see
    [`brainhops.io.images.n5`][brainhops.io.images.n5]). This class reads
    a dataset, named by its directory or its `attributes.json` file. It
    also reads one level of a pyramid, a group whose datasets are `s0`,
    `s1`, ..., the finest by default or the one that `level` selects. A
    pyramid of several levels is read whole by
    [`N5MultiScaleImage`][brainhops.io.images.n5.N5MultiScaleImage]
    unless `level` is given.

    The image holds the attributes as [`N5Metadata`][], whose record is
    an [`N5Raw`][], and the samples in `rawdata`, a lazy array that
    TensorStore reads block by block when the data are needed. The data
    are indexed in the order of the `dimensions` of the dataset, whose
    first dimension varies fastest in a block, so the axes are named
    `x`, `y`, `z`, `t` and `c`, as for an array without a description.

    The transformation maps the voxels to a `"physical"` space with the
    voxel size, the downsampling factors and the offset that the spatial
    attributes of the tools of the N5 ecosystem give, as
    [`brainhops.io.images.n5`][brainhops.io.images.n5] describes them.
    Without such attributes, the transformation is the identity and the
    physical axes have no unit, as for a raster image whose pixel size is
    unknown.

    N5 containers are not written, so the class has no writer, and
    [`save`][brainhops.io.save] refuses to write the image as an N5
    dataset. Assigned data are pending until [`consolidate`][] moves them
    into `rawdata`, and the transformations that the image holds, read
    or assigned, are kept, since no record can be encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("n5",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The samples as the dataset stores them, or `None` without data.

    After a read, `rawdata` is a lazy array of the dataset. Assigning
    `data` leaves `rawdata` as it is, and [`consolidate`][] stores the
    assigned array here.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
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

    _metadata: NoRepr[tx.Optional[N5Metadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the dataset or pyramid, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    level: tx.Annotated[
        tx.Optional[int],
        tx.Doc(
            "The index of the level of a pyramid, where 0 is the finest "
            "level `s0`. It is `None` for a dataset that was read on its "
            "own."
        ),
    ] = None

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The samples, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`. Otherwise the
        data is decoded from `rawdata` on first access and cached. Data
        decoded from a dataset is read-only. To change the samples, assign
        `data`. Assigning `None` decodes `rawdata` again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-physical transformations, decoded from the record.

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
        return _record_transformations(self.metadata, self.level)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the samples.

        The shape is that of the assigned data, or else the shape of
        `rawdata`, which the lazy array knows from the record. An image
        without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> "N5Image":
        """Move the pending data into `rawdata`.

        N5 containers are not written, so no record can be encoded, and
        consolidating an image is limited to its data: an assigned array
        becomes `rawdata` and is no longer pending. The metadata is kept, and
        so are the transformations that the image holds, read or assigned.
        An image without pending data is left as it is: the copy shares
        `rawdata` and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an N5
        dataset holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        N5Image
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(source="N5Image", target="N5Image")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(self.metadata, N5Metadata, fields, report)
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not requested:
            metadata = self.metadata
        # The record cannot be encoded, so a copy is given the list that
        # the image holds when it is not the list that the record decodes
        # to, as a list decoded and then changed in place would otherwise
        # be decoded again and lost.
        held = held_transformations(self)
        if held is None or same_transformations(
            held, _record_transformations(self.metadata, self.level)
        ):
            held = self._transformations
        if self._data is None:
            if not inplace:
                return replace(self, transformations=held, metadata=metadata)
            if requested:
                self.transformations = held
                self.metadata = metadata
            return self
        raw = _image_to_disk(self._data)
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
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a name designates a dataset.

        The name is the directory or the `attributes.json` file of a
        dataset or of a pyramid. A dataset is certainly read by this
        class, and so is a pyramid when `level` is given. A pyramid
        without `level` scores `Confidence.LIKELY`, below
        [`N5MultiScaleImage`][brainhops.io.images.n5.N5MultiScaleImage],
        which reads a pyramid of several levels whole.

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of a node.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no dataset
            and no pyramid.
        level : int, optional
            The level of a pyramid to read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        try:
            _, record = read_node(filename)
        except Exception:  # noqa: BLE001
            return score(Confidence.NO, error)
        if record.levels and level is None:
            return Confidence.LIKELY
        return Confidence.CERTAIN

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream reads dataset attributes.

        Parameters
        ----------
        file : IO
            The stream of the `attributes.json` file, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no attributes
            of an N5 dataset.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        return N5Metadata.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `Confidence.NO`, since bytes cannot hold a whole dataset.

        Parameters
        ----------
        content : bytes
            The bytes, which are not read.
        error : bool or type[Exception], default=False
            Whether to raise an error, which is then always raised.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(
                "An N5 dataset is a directory of files, which bytes alone "
                "cannot hold."
            )
        return Confidence.NO

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> "N5Image":
        """Read a dataset, or one level of a pyramid, from its directory.

        Only the attributes are read. The lazy array reads the blocks when
        the samples are needed.

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of a dataset or of
            a pyramid.
        level : int, optional
            The index of the level of a pyramid, 0 (the finest) by
            default. A negative index counts from the coarsest level. A
            dataset read on its own takes no level other than 0.
        **kwargs : Any
            Refused, since the format has no other option.

        Returns
        -------
        N5Image
            The image, whose samples are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If there is no node at that name.
        ParserContentError
            If the node is neither a dataset nor a pyramid, if there is no
            such level, or if the samples cannot be read.
        """
        no_options(kwargs)
        location, record = read_node(filename)
        return _read_dataset(cls, location, N5Metadata.from_raw(record), level)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> "N5Image":
        """Read a dataset from an open `attributes.json` file.

        The dataset is found from the name of the stream, which must have
        one, and its attributes are read again from there.

        Parameters
        ----------
        file : BinaryIO
            The open `attributes.json` file, which must have a name.
        level : int, optional
            The index of the level of a pyramid, 0 by default.
        **kwargs : Any
            Refused, since the format has no other option.

        Returns
        -------
        N5Image
            The image, whose samples are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserTypeError
            If the stream has no file name.
        """
        no_options(kwargs)
        location, record = read_node(stream_location(file))
        return _read_dataset(cls, location, N5Metadata.from_raw(record), level)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "N5Image":
        """Refuse to read bytes, which cannot hold a dataset of many files.

        Parameters
        ----------
        content : bytes
            The bytes, which are not read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        N5Image
            Never, since an error is always raised.

        Raises
        ------
        ParserContentError
            Always.
        """
        raise ParserContentError(
            "An N5 dataset is a directory of files, which bytes alone cannot "
            "hold. Read it from the path of its directory or of its "
            "attributes.json file instead."
        )


# Without TensorStore, the samples cannot be read, so the format is not
# registered, and a request for it by hint reports which package to
# install (see `brainhops.io.images`).
if HAS_TENSORSTORE:
    register_format(N5Image)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_dataset(
    cls: type,
    location: str,
    metadata: N5Metadata,
    level: tx.Optional[int],
) -> N5Image:
    """Build the image of a dataset, or of a level of a pyramid.

    Raises
    ------
    ParserContentError
        If there is no such level, or its samples cannot be read.
    """
    record = metadata.raw
    index = _level_index(record, level)
    dataset = record.levels[index] if record.levels else record
    where = dataset_location(location, record, index)
    if dataset.data_type == "object":
        raise ParserContentError(
            f"The N5 dataset at {where} holds serialized objects, which are "
            f"not an image."
        )
    kind = dataset.compression["type"]
    if (
        kind not in READABLE_COMPRESSIONS
        or "compression" not in dataset.attributes
    ):
        raise ParserContentError(
            f"The N5 dataset at {where} is compressed with {kind!r}, which "
            f"brainhops cannot decode. The compressions that it reads are "
            f"{', '.join(READABLE_COMPRESSIONS)}, given as a compression "
            f"object."
        )
    proxy = _N5Proxy(where, dataset)
    return cls(
        rawdata=proxy,
        metadata=metadata,
        level=index if record.levels else None,
    )


def _level_index(record: N5Raw, level: tx.Optional[int]) -> int:
    """Return the index of a level, 0 for `None`, counting from 0.

    A dataset read on its own has the single level 0.

    Raises
    ------
    ParserContentError
        If the node has no such level.
    """
    count = len(record.levels) or 1
    index = 0 if level is None else int(level)
    if not -count <= index < count:
        raise ParserContentError(
            f"The N5 node has {count} level(s), so it has no level {index}."
        )
    return index % count


def _record_transformations(
    metadata: tx.Optional[N5Metadata], level: tx.Optional[int]
) -> tx.List[Transformation]:
    """Decode the transformation of a dataset from the record of metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    record = metadata.raw
    index = _level_index(record, level)
    dataset = record.levels[index] if record.levels else record
    return [_dataset_transformation(record, index, len(dataset.dimensions))]


def _dataset_transformation(
    record: N5Raw, index: int, ndim: int
) -> Transformation:
    """Return the voxel-to-physical transformation of a dataset.

    The voxel `i` of the dataset lies at
    `size * (factor * i + (factor - 1) / 2) + origin` along each axis,
    with the values that the spatial attributes give, and a size and a
    factor of 1 and an origin of 0 where they give none. The unit of the
    sizes applies to the spatial axes `x`, `y` and `z` only.
    """
    found = placement(record, index if record.levels else None)
    axes = default_axes(ndim)
    scales = {}
    for axis, size in zip(axes, found.sizes):
        unit = found.unit if axis_group(axis) == "space" else None
        scales[axis.name] = (size, unit)
    factors = list(found.factors) + [1.0] * (ndim - len(found.factors))
    origin = {axis.name: o for axis, o in zip(axes, found.origin)}
    return level_transformation(axes, scales, factors, origin)
