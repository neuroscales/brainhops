"""The shared base of the transformations whose file is an image."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Field, InitVar, KwOnly, NoRepr, fields, replace

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import (
    FIELDS,
    apply_loss_policy,
)
from brainhops.io.base.parsers import BinaryFileReader
from brainhops.io.common._internal_api import held_transformations
from brainhops.io.metadata._internal_api import requested_fields

# this package
from ._base import TransformationFormat
from .affines import VoxelToRAS


class ImageBackedTransformation(BinaryFileReader, TransformationFormat):
    """A transformation whose file is an image, which it keeps.

    A field of displacements or of coordinates is often stored as an
    image whose voxels hold vectors, such as a NIfTI image, an AFNI
    dataset or an MRtrix image. A transformation of this class keeps
    that file, as it was read or as the transformation was last
    consolidated, in a private image of the class that `IMAGE` names,
    and it reads, consolidates and writes through that image. The
    properties `rawdata` and `metadata` read the array and the metadata
    of the image, and assigning one of them replaces the image with a
    copy that holds the new value. The data and the chain of the data
    model are kept in the private fields of the transformation instead.
    They are pending until the transformation is consolidated, and they
    take precedence over the image. The design memo
    `spec/design/image-backed-transformations.md` explains these two
    tiers of state.

    The constructor takes the positional arguments of the data model,
    and the keywords `image`, `rawdata` and `metadata`. It applies the
    keywords in that order, so that a later keyword overrides an earlier
    one. `image` becomes the stored image, and `rawdata` and `metadata`
    replace the matching parts of it. Without `image`, the stored image
    is built from `rawdata` and `metadata`. The fields of the data model
    are stored last, and they are never given to the image, because the
    `data` and the `transformations` of a transformation do not mean the
    `data` and the `transformations` of its image. An image of another
    class than `IMAGE` raises `TypeError`, and an image that has pending
    data or transformations is consolidated when it is given, so that
    the stored image always describes a file.

    A concrete subclass names its image class in `IMAGE`, and takes the
    `metadata` field of its data model over as a value that the
    constructor takes and does not keep, with
    `_metadata: ... = Inherit("metadata", default=None, var=True)`. A
    view that it decodes from the image, such as its chain, is cached
    with `invalidated_by="_image"`, so that the view is decoded again
    whenever the stored image is replaced. The subclass implements
    [`consolidate`][] in `_consolidate`: it builds the image of its
    pending state with [`image_with`][] or [`data_image`][], and passes
    that image to [`consolidate_through`][].
    """

    IMAGE: tx.ClassVar[type]
    """The class of the image that stores the transformation."""

    # The image is not converted, so that an image of another class is
    # refused rather than read as an image of the class of the format.
    _image: KwOnly[NoRepr[tx.Optional[tx.Any]]] = Field(
        default=None, convert=False
    )
    """The image that holds the file, which the constructor always sets.

    The image describes the file as it was read, or as the
    transformation was last consolidated, and it never holds pending
    data or transformations.
    """

    _rawdata: KwOnly[NoRepr[InitVar[tx.Optional[ArrayProtocol]]]] = Field(
        default=None, alias="rawdata"
    )

    def __post_init__(self, arguments: tx.Any) -> None:
        image = self._image
        if image is None:
            image = self.IMAGE()
        elif not isinstance(image, self.IMAGE):
            raise TypeError(
                f"A {type(self).__name__} is stored in a "
                f"{self.IMAGE.__name__}, and cannot be built from a "
                f"{type(image).__name__}."
            )
        elif (
            image.data is not image.rawdata
            or held_transformations(image) is not None
        ):
            image = image.consolidate()
        parts = {
            name: arguments[name]
            for name in ("rawdata", "metadata")
            if arguments.get(name) is not None
        }
        self._image = replace(image, **parts) if parts else image
        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init(arguments)

    # --- stored image -------------------------------------------------

    @property
    def rawdata(self) -> tx.Optional[ArrayProtocol]:
        """The array of the file as it is stored, or `None` without one.

        The array is the `rawdata` of the stored image. After a read, it
        is a lazy array of the file, which applies the scaling of the
        header when it is read. It describes the file as it was read, or
        as the transformation was last consolidated, so assigning `data`
        leaves it as it is. Assigning `rawdata` replaces the stored
        image with a copy that holds the new array.
        """
        return self._image.rawdata

    @rawdata.setter
    def rawdata(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._image = replace(self._image, rawdata=value)

    @property
    def metadata(self) -> tx.Optional[tx.Any]:
        """The metadata of the file, which holds its header, or `None`.

        The metadata is that of the stored image. A transformation built
        from its data alone has none. Assigning other metadata replaces
        the stored image with a copy that holds it, converted into the
        metadata of the image class, and drops the views decoded from
        the previous header.
        """
        return self._image.metadata

    @metadata.setter
    def metadata(self, value: tx.Optional[tx.Any]) -> None:
        self._image = replace(self._image, metadata=value)

    # --- reading ------------------------------------------------------

    @classmethod
    def from_image(cls, image: tx.Any, **options: tx.Any) -> tx.Self:
        """Build a transformation from the image that stores it.

        The image becomes the stored image of the transformation, and
        the transformation decodes its data and its chain from that
        image when they are needed. The call is the same as
        `cls(image=image, **options)`.

        Parameters
        ----------
        image : Any
            The image, of the class that `IMAGE` names, which is
            consolidated first if it has pending data or
            transformations.
        **options : Any
            The other fields of the transformation.

        Returns
        -------
        Self
            The transformation.

        Raises
        ------
        TypeError
            If the image is not of the class that `IMAGE` names.
        """
        return cls(image=image, **options)

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs: tx.Any) -> tx.Self:
        """Read a transformation from the path of its file.

        The file is read by the reader of the image class, whose array
        is read lazily, and the transformation is built from that image
        with [`from_image`][]. The keywords that name an argument of the
        constructor of the transformation, under any of their aliases,
        go to the transformation, and the other keywords go to the
        reader of the image, which refuses those that it does not know.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of the reader of the image, such as `mmap`, and
            the other fields of the transformation.

        Returns
        -------
        Self
            The transformation, whose array is read lazily.
        """
        reader = _reader_options(cls, kwargs)
        image = cls.IMAGE.from_filename(filename, **reader)
        return cls.from_image(image, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs: tx.Any) -> tx.Self:
        """Read a transformation from an open stream of its file.

        The stream is read by the reader of the image class, whose lazy
        array keeps the stream, so the stream must stay open while the
        array may be read. The transformation is built from that image
        with [`from_image`][], and the keywords are shared out as by
        [`from_filename`][].

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **kwargs : Any
            The options of the reader of the image, and the other fields
            of the transformation.

        Returns
        -------
        Self
            The transformation, whose array is read lazily.
        """
        reader = _reader_options(cls, kwargs)
        image = cls.IMAGE.from_fileobj(file, **reader)
        return cls.from_image(image, **kwargs)

    # --- writing ------------------------------------------------------

    def to_image(self) -> tx.Any:
        """Return the image that a file of this transformation holds.

        The transformation is consolidated into a copy, as by
        [`consolidate`][], and the stored image of that copy is
        returned. The transformation itself is not changed.

        Returns
        -------
        Any
            The image, of the class that `IMAGE` names, without pending
            data or transformations.
        """
        return self.consolidate()._image


# ----------------------------------------------------------------------
#   HELPERS OF THE CONCRETE FORMATS
# ----------------------------------------------------------------------


def image_with(
    xform: ImageBackedTransformation,
    raw: tx.Optional[ArrayProtocol] = None,
    vox2ras: tx.Optional[np.ndarray] = None,
) -> tx.Any:
    """Return the stored image of a transformation, with pending data.

    A concrete format calls this function to build the image of its
    pending state. The array `raw`, laid out as the file stores it,
    replaces the array of a copy of the stored image. When the grid is
    given as the (4, 4) voxel-to-RAS matrix `vox2ras`, it becomes the
    voxel-to-world transformation of the copy, as a
    [`VoxelToRAS`][brainhops.io.transformations.base.affines.VoxelToRAS],
    and the image encodes its geometry from it. Otherwise, the copy
    keeps the geometry of its header. Without `raw`, the stored image is
    returned as it is.
    """
    if raw is None:
        return xform._image
    if vox2ras is None:
        return replace(xform._image, data=raw)
    grid = [VoxelToRAS(matrix=np.asarray(vox2ras)[:3])]
    return replace(xform._image, data=raw, transformations=grid)


def data_image(
    xform: ImageBackedTransformation,
    to_disk: tx.Callable[[ArrayProtocol], ArrayProtocol],
) -> tx.Any:
    """Return the stored image of a transformation, with its assigned data.

    A concrete format keeps the data that is assigned to it in `_data`.
    That data, laid out by `to_disk` as the file stores it, replaces the
    array of a copy of the stored image, whose header keeps its
    geometry, as [`image_with`][] does. Without assigned data, the
    stored image is returned as it is.
    """
    data = getattr(xform, "_data", None)
    return image_with(xform, None if data is None else to_disk(data))


def consolidate_through(
    xform: ImageBackedTransformation,
    image: tx.Any,
    intent: tx.Mapping[str, tx.Any],
    changes: tx.Mapping[str, tx.Any],
    *,
    inplace: bool = False,
    check: tx.Optional[tx.Callable[[tx.Any], None]] = None,
    **options: tx.Any,
) -> ImageBackedTransformation:
    """Consolidate a transformation through the image of its pending state.

    A concrete format implements `_consolidate` with this function. It
    builds `image`, the image that holds its pending data or chain, with
    [`image_with`][] or [`data_image`][], and it removes from `options`
    the options that are its own rather than the image's. The image is
    consolidated with the options of the call, so the precedence on
    write and the loss report are those of the image class. The options
    in `intent`, such as the NIfTI intent code or the MRtrix header keys
    that mark the format, are given to the image only when it is
    encoded again, and the options of the call override them. An option
    whose value is `None` counts as not given. A transformation that has
    an image and nothing pending, consolidated without options,
    therefore keeps its image, and it is still saved byte for byte. A
    format whose files hold arrays of a fixed shape gives a function as
    `check`, which receives the consolidated image and raises
    `WriterError` when the file cannot hold it. The consolidated
    transformation then holds that image and receives the fields in
    `changes`, such as `data=None`.

    Returns
    -------
    ImageBackedTransformation
        The consolidated copy, or `xform` with `inplace=True`.

    Raises
    ------
    WriterError
        If the image has no data, or if the file cannot hold it.
    MetadataLossError
        If metadata is lost under the `"raise"` policy.
    """
    fields = {name: v for name, v in options.items() if name in FIELDS}
    given = any(
        value is not None
        for name, value in options.items()
        if name not in FIELDS
    )
    if (
        image is xform._image
        and not given
        and not requested_fields(xform.metadata, fields)
    ):
        intent = {}
    # The image reports what it loses to a report of the format, on which
    # the policy in effect then acts once, at the level of the caller of
    # `consolidate`.
    name = type(xform).__name__
    report = ConversionReport(source=name, target=name)
    options = {**intent, **options}
    image = image.consolidate(on_loss=report, **options)
    apply_loss_policy(report, stacklevel=3)
    if check is not None:
        check(image)
    if not inplace:
        return replace(xform, image=image, **changes)
    xform._image = image
    for name, value in changes.items():
        setattr(xform, name, value)
    return xform


def _reader_options(
    cls: type, kwargs: tx.Dict[str, tx.Any]
) -> tx.Dict[str, tx.Any]:
    # The keywords that the constructor of the transformation takes, under
    # any of their aliases, stay in `kwargs`, and the others go to the
    # reader of the image.
    names = {
        alias
        for field in fields(cls, var=None)
        if field.init
        for alias in field.aliases
    }
    return {
        name: kwargs.pop(name) for name in list(kwargs) if name not in names
    }
