"""A Neuroglancer precomputed volume, read as a multiscale image."""

# dependencies
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_TENSORSTORE
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferTypeError,
)
from brainhops.io.common.precomputed import (
    PrecomputedMetadata,
    PrecomputedRaw,
)
from brainhops.io.common.precomputed._data import (
    read_precomputed,
    stream_location,
)
from brainhops.io.common.precomputed._raw import _no_options, _score
from brainhops.io.images.base import ImageFormat

# this format
from ._image import PrecomputedImage, _read_scale


class PrecomputedMultiScaleImage(
    ImageFormat, MultiScaleImage, BinaryFileReader
):
    """A Neuroglancer precomputed volume, read as a multiscale image.

    A precomputed volume is a directory, local or remote, that holds an
    `info` file and the chunks of each scale (see
    [`brainhops.io.common.precomputed`][brainhops.io.common.precomputed]).
    Each scale is a [`PrecomputedImage`][], finest first, whose voxels
    are read lazily, and every scale maps its voxels to the same
    `"physical"` space, in nanometres, by its own resolution and
    `voxel_offset`. The pyramid therefore has no transformation of its
    own, as a TIFF pyramid. A scale that Neuroglancer hides is read too.

    The pyramid holds the `info` file as [`PrecomputedMetadata`][], which
    its levels share, and it has no `raw` of its own: each level holds the
    lazy array of its scale. A volume of a single scale is read as a
    [`PrecomputedImage`][] by [`load`][brainhops.io.images.load], unless
    this class is asked for.

    Precomputed volumes are not written, so the class has no writer.
    [`consolidate`][] only moves the pending data of each level into its
    `raw`.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("precomputed", "neuroglancer")

    metadata: KwOnly[NoRepr[tx.Optional[PrecomputedMetadata]]] = None
    """The metadata of the volume, which holds its record, or `None`.

    A pyramid built from images has no metadata.
    """

    def consolidate(
        self, *, inplace: bool = False
    ) -> "PrecomputedMultiScaleImage":
        """Move the pending data of each level into its `raw`.

        Precomputed volumes are not written, so no record can be encoded.
        Each level that is a [`PrecomputedImage`][] is consolidated, which
        moves its assigned data into its `raw`, and the other levels are
        kept as they are. The metadata and the transformations are kept.
        A level without pending data is consolidated into a copy that
        shares its `raw` and its metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this pyramid rather than a copy.

        Returns
        -------
        PrecomputedMultiScaleImage
            The consolidated copy, or this pyramid with `inplace=True`.
        """
        images = [
            image.consolidate()
            if isinstance(image, PrecomputedImage)
            else image
            for image in self.images
        ]
        if not inplace:
            return replace(self, images=images)
        self.images = images
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
        """Return the confidence that a name designates a pyramid.

        The name is the directory of a volume or its `info` file, local
        or remote. The score is `Confidence.CERTAIN` for a volume of
        several scales when no `level` is given, and `Confidence.NO`
        otherwise, so that [`PrecomputedImage`][] reads a single scale.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no volume.
        level : int, optional
            The scale to read, which this class does not read alone.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        if level is not None:
            return Confidence.NO
        try:
            _, record = read_precomputed(filename)
        except Exception:  # noqa: BLE001
            return _score(Confidence.NO, error)
        return _score_record(record)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream reads the `info` of a pyramid.

        Parameters
        ----------
        file : IO
            The stream of the `info` file, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no `info`
            file of a volume.
        level : int, optional
            The scale to read, which this class does not read alone.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, `Confidence.CERTAIN` or `Confidence.NO`.
        """
        if level is not None:
            return Confidence.NO
        try:
            record = PrecomputedRaw.from_fileobj(file)
        except Exception:  # noqa: BLE001
            return _score(Confidence.NO, error)
        return _score_record(record)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `Confidence.NO`, since bytes cannot hold a whole volume.

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
                "A precomputed volume is a directory of files, which bytes "
                "alone cannot hold."
            )
        return Confidence.NO

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "PrecomputedMultiScaleImage":
        """Read every scale of a volume from its directory or `info` file.

        Only the `info` file is read. The lazy array of each level reads
        the chunks of its scale when its voxels are needed.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume, local or remote.
        **kwargs : Any
            Refused, since the pyramid has no option.

        Returns
        -------
        PrecomputedMultiScaleImage
            The pyramid, whose levels are read lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If there is no volume at that name.
        ParserContentError
            If the `info` file breaks the specification, or the encoding
            of a scale cannot be read.
        """
        _no_options(kwargs)
        location, record = read_precomputed(filename)
        return _read_pyramid(cls, location, record)

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, **kwargs
    ) -> "PrecomputedMultiScaleImage":
        """Read every scale of a volume from an open `info` file.

        The volume is found from the name of the stream, which must have
        one, and the `info` file is read again from there.

        Parameters
        ----------
        file : BinaryIO
            The open `info` file, which must have a name.
        **kwargs : Any
            Refused, since the pyramid has no option.

        Returns
        -------
        PrecomputedMultiScaleImage
            The pyramid, whose levels are read lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserTypeError
            If the stream has no file name.
        """
        _no_options(kwargs)
        location, record = read_precomputed(stream_location(file))
        return _read_pyramid(cls, location, record)

    @classmethod
    def from_bytes(
        cls, content: bytes, **kwargs
    ) -> "PrecomputedMultiScaleImage":
        """Refuse to read bytes, which cannot hold a volume of many files.

        Parameters
        ----------
        content : bytes
            The bytes, which are not read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        PrecomputedMultiScaleImage
            Never, since an error is always raised.

        Raises
        ------
        ParserContentError
            Always.
        """
        raise ParserContentError(
            "A precomputed volume is a directory of files, which bytes "
            "alone cannot hold. Read it from the path of its directory or "
            "of its info file instead."
        )


# Without TensorStore, the voxels cannot be read, so the format is not
# registered, and a request for it by hint reports which package to
# install (see `brainhops.io.images`).
if HAS_TENSORSTORE:
    register_format(PrecomputedMultiScaleImage)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_pyramid(
    cls: type, location: str, record: PrecomputedRaw
) -> PrecomputedMultiScaleImage:
    """Build the pyramid of every scale of a volume.

    The levels share the metadata of the pyramid.

    Raises
    ------
    ParserContentError
        If the encoding of a scale cannot be read.
    """
    metadata = PrecomputedMetadata.from_raw(record)
    images: tx.List[SingleScaleImage] = [
        _read_scale(PrecomputedImage, location, metadata, index)
        for index in range(len(record.scales))
    ]
    return cls(images=images, transformations=[], metadata=metadata)


def _score_record(record: PrecomputedRaw) -> float:
    """Score a volume as a pyramid: certain when it has several scales."""
    if len(record.scales) > 1:
        return Confidence.CERTAIN
    return Confidence.NO
