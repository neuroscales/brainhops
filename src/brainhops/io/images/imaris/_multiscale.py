"""Imaris pyramids: every resolution level of an Imaris file."""

# stdlib
from io import BytesIO

# dependencies
import h5py
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.datamodel.images import MultiScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, SnifferContentError
from brainhops.io.common.hdf5 import Hdf5Reader
from brainhops.io.common.imaris import ImarisMetadata, ImarisRaw
from brainhops.io.common.imaris._data import read_imaris_level
from brainhops.io.common.imaris._raw import no_options
from brainhops.io.images.base import ImageFormat

# this format
from ._image import ImarisImage, _level_count, _source


@register_format
class ImarisMultiScaleImage(ImageFormat, MultiScaleImage, Hdf5Reader):
    """Every resolution level of an Imaris file, read as a pyramid.

    Each level is an [`ImarisImage`][], finest first, whose voxels are read
    on first access. The levels share the metadata of the file. Every
    level spans the extent of the file and maps its voxels to the same
    `"physical"` system, so the pyramid has no transformation of its own,
    like a TIFF pyramid. The voxel size of a level is the size of the
    extent divided by the size of the level, which places the levels as in
    the other brainhops pyramids. A file with a single level, or a read
    with the option `level`, gives an [`ImarisImage`][] instead.

    Imaris files cannot be written by brainhops, so the pyramid has no
    writer. Its levels hold the record of the file, but the pyramid itself
    holds no record, since the record of a pyramid is not designed yet.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ims",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("imaris", "ims")

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is an Imaris
        pyramid.

        The score is `CERTAIN` for an Imaris file with several resolution
        levels, and `NO` for any other file.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris file
            with several levels.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if ImarisRaw.sniff_h5(h5file, error=error) and _level_count(h5file) > 1:
            return Confidence.CERTAIN
        return _declined(error, "The Imaris file has a single level.")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score the Imaris file at a path.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris
            pyramid.
        level : int, optional
            The level that the caller asks for, which this class does not
            read, so that the score is then `NO`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if level is not None:
            return _declined(error, "The option level selects one level.")
        return super().sniff_filename(filename, error=error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score an open, seekable binary stream.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not an Imaris
            pyramid.
        level : int, optional
            The level that the caller asks for, which this class does not
            read, so that the score is then `NO`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if level is not None:
            return _declined(error, "The option level selects one level.")
        return super().sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score the bytes of an Imaris file.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an Imaris
            pyramid.
        level : int, optional
            The level that the caller asks for, which this class does not
            read, so that the score is then `NO`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error, level=level)

    # --- read ---------------------------------------------------------

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> tx.Self:
        """Read every level of an open Imaris file, without the voxels.

        The lazy arrays read the voxels again from the file by its name,
        when they are needed. A file that h5py opened over a stream has no
        name, so the lazy arrays read that open file, which the caller
        must keep open while the pyramid is used.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        ImarisMultiScaleImage
            The pyramid.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the file is not an Imaris file that brainhops reads.
        """
        no_options(kwargs)
        return _read_pyramid(cls, h5file, _source(h5file))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read every level of an Imaris file from an open stream.

        A stream has no name to open again, so its content is read into
        memory, where the lazy arrays read the voxels when they are needed.
        The position of the stream is restored.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        ImarisMultiScaleImage
            The pyramid.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream is not an Imaris file that brainhops reads.
        """
        no_options(kwargs)
        with preserve_position(file):
            content = bytes(file.read())
        with h5py.File(BytesIO(content), "r") as f:
            return _read_pyramid(cls, f, content)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _read_pyramid(cls: type, h5file: h5py.File, source: tx.Any) -> tx.Any:
    """Read the record of an open file and build every level."""
    record = ImarisRaw.from_h5(h5file)
    metadata = ImarisMetadata.from_raw(record)
    levels = [
        ImarisImage(
            raw=read_imaris_level(source, record, index),
            metadata=metadata,
            level=index,
        )
        for index in range(len(record.levels))
    ]
    return cls(images=levels, transformations=[])


def _declined(error: tx.Union[bool, tx.Type[Exception]], reason: str) -> float:
    """Return `NO`, or raise the error that the caller asks for.

    Raises
    ------
    SnifferContentError
        If `error` is `True`, or the class that `error` gives.
    """
    if error:
        if error is True:
            error = SnifferContentError
        raise error(reason)
    return Confidence.NO
