"""An N5 pyramid, read as a multiscale image."""

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
    disabled,
)
from brainhops.io.images.base import ImageFormat

# this format
from ._data import read_node, stream_location
from ._image import N5Image, _read_dataset
from ._metadata import N5Metadata
from ._raw import N5Raw, no_options, score


class N5MultiScaleImage(BinaryFileReader, ImageFormat, MultiScaleImage):
    """An N5 pyramid, read as a multiscale image.

    An N5 pyramid is a group whose datasets `s0`, `s1`, ... hold the same
    image at decreasing resolutions, finest first, as the n5-viewer,
    Paintera and BigDataViewer write it (see
    [`brainhops.io.images.n5`][brainhops.io.images.n5]). Each level is an
    [`N5Image`][] whose samples are read lazily, and every level maps its
    voxels to the same `"physical"` space by the spatial attributes of
    its dataset and of the group. The pyramid therefore has no
    transformation of its own.

    The pyramid holds the attributes of the group and of its datasets as
    [`N5Metadata`][], which its levels share, and it has no `rawdata` of
    its own: each level holds the lazy array of its dataset. A pyramid
    of a single level is read as an [`N5Image`][] by
    [`load`][brainhops.io.images.load], unless this class is asked for.

    N5 containers are not written, so the class has no writer.
    [`consolidate`][] only moves the pending data of each level into its
    `rawdata`.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("n5",)

    metadata: KwOnly[NoRepr[tx.Optional[N5Metadata]]] = None
    """The metadata of the pyramid, which holds its record, or `None`.

    A pyramid built from images has no metadata.
    """

    def consolidate(self, *, inplace: bool = False) -> "N5MultiScaleImage":
        """Move the pending data of each level into its `rawdata`.

        N5 containers are not written, so no record can be encoded. Each
        level that is an [`N5Image`][] is consolidated, which moves its
        assigned data into its `rawdata`, and the other levels are kept
        as they are. The metadata and the transformations are kept. A
        level without pending data is consolidated into a copy that
        shares its `rawdata` and its metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this pyramid rather than a copy.

        Returns
        -------
        N5MultiScaleImage
            The consolidated copy, or this pyramid with `inplace=True`.
        """
        images = [
            image.consolidate() if isinstance(image, N5Image) else image
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

        The name is the directory or the `attributes.json` file of a
        group. The score is `Confidence.CERTAIN` for a pyramid of several
        levels when no `level` is given, and `Confidence.NO` otherwise, so
        that [`N5Image`][] reads a dataset or a single level.

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of a group.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no pyramid.
        level : int, optional
            The level to read, which this class does not read alone.
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
            _, record = read_node(filename)
        except Exception:  # noqa: BLE001
            return score(Confidence.NO, error)
        return _score_record(record)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream reads the group of a pyramid.

        The levels of a pyramid are found from the name of the stream,
        which must have one.

        Parameters
        ----------
        file : IO
            The stream of the `attributes.json` file of a group.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream reads no pyramid.
        level : int, optional
            The level to read, which this class does not read alone.
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
            _, record = read_node(stream_location(file))
        except Exception:  # noqa: BLE001
            return score(Confidence.NO, error)
        return _score_record(record)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `Confidence.NO`, since bytes cannot hold a pyramid.

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
                "An N5 pyramid is a directory of files, which bytes alone "
                "cannot hold."
            )
        return Confidence.NO

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "N5MultiScaleImage":
        """Read every level of a pyramid from the directory of its group.

        Only the attributes are read. The lazy array of each level reads
        the blocks of its dataset when its samples are needed.

        Parameters
        ----------
        filename : path-like
            The directory or the `attributes.json` file of the group.
        **kwargs : Any
            Refused, since the pyramid has no option.

        Returns
        -------
        N5MultiScaleImage
            The pyramid, whose levels are read lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If there is no node at that name.
        ParserContentError
            If the node is not a pyramid, or a level cannot be read.
        """
        no_options(kwargs)
        location, record = read_node(filename)
        return _read_pyramid(cls, location, record)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "N5MultiScaleImage":
        """Read every level of a pyramid from an open `attributes.json`.

        The pyramid is found from the name of the stream, which must have
        one, and its attributes are read again from there.

        Parameters
        ----------
        file : BinaryIO
            The open `attributes.json` file of the group, which must have
            a name.
        **kwargs : Any
            Refused, since the pyramid has no option.

        Returns
        -------
        N5MultiScaleImage
            The pyramid, whose levels are read lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserTypeError
            If the stream has no file name.
        """
        no_options(kwargs)
        location, record = read_node(stream_location(file))
        return _read_pyramid(cls, location, record)

    from_bytes = disabled(
        ParserContentError,
        "An N5 pyramid is a directory of files, which bytes alone cannot "
        "hold. Read it from the path of its group instead.",
    )


# Without TensorStore, the samples cannot be read, so the format is not
# registered, and a request for it by hint reports which package to
# install (see `brainhops.io.images`).
if HAS_TENSORSTORE:
    register_format(N5MultiScaleImage)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_pyramid(
    cls: type, location: path.Path, record: N5Raw
) -> N5MultiScaleImage:
    """Build the pyramid of every level of a group.

    The levels share the metadata of the pyramid.

    Raises
    ------
    ParserContentError
        If the node is a dataset rather than a pyramid, or a level cannot
        be read.
    """
    if not record.levels:
        raise ParserContentError(
            f"The N5 node at {location} is a dataset, not a pyramid."
        )
    metadata = N5Metadata.from_raw(record)
    images: tx.List[SingleScaleImage] = [
        _read_dataset(N5Image, location, metadata, index)
        for index in range(len(record.levels))
    ]
    return cls(images=images, transformations=[], metadata=metadata)


def _score_record(record: N5Raw) -> float:
    """Score a node as a pyramid: certain when it has several levels."""
    if len(record.levels) > 1:
        return Confidence.CERTAIN
    return Confidence.NO
