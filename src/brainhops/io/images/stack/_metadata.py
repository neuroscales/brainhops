"""The metadata of an image stack."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    ParserTypeError,
    disabled,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import (
    NOT_A_STREAM,
    StackRaw,
    refuse_options,
    refuse_path,
    refuse_stream,
)


@register_format
class StackMetadata(BinaryFileReader, MetadataFormat):
    """Metadata of an image stack, which is the list of its files.

    An image stack is a folder of numbered two-dimensional images that
    form one volume. Its metadata holds a [`StackRaw`][] record, which
    lists the files, holds the header of each one and the geometry of the
    stack. Reading the metadata reads the header of every file, but never
    their pixels, and the metadata keeps no open file. A folder of headers
    alone would not be a stack, so the metadata is not written on its
    own: it is written with the image, by
    [`StackImage`][brainhops.io.images.stack.StackImage].

    A stack is never recognized from its path, so this class never takes
    a folder or a file from another reader. The format hint `"stack"`
    selects it, as in `MetadataFormat.load(folder, hint="stack")` (see
    [`brainhops.io.images.stack`][] for the rules).
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("stack",)

    raw: NoRepr[tx.Optional[StackRaw]] = None
    """The record of the stack, or `None` for metadata without a record.
    """

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        pattern: tx.Optional[str] = None,
        **kwargs,
    ) -> float:
        """Refuse a path, since a stack is read only with its hint.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        error : bool or type[Exception], default=False
            Whether to raise an error, which tells why the path is not the
            folder of a stack, or that a stack is read only by hint.
        pattern : str, optional
            The glob pattern that selects the files of the stack.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_path(filename, pattern, error)

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
            Whether to raise an error, since a stream is never a stack.
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
            Whether to raise an error, since bytes are never a stack.
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
        axis: tx.Optional[tx.Union[str, int]] = None,
        pixel_size: tx.Optional[tx.Union[float, tx.Sequence[float]]] = None,
        spacing: tx.Optional[float] = None,
        unit: tx.Optional[tx.Any] = None,
        **kwargs,
    ) -> "StackMetadata":
        """Read the metadata of the image stack in a folder.

        The record is read with [`StackRaw.from_folder`][], which reads
        the header of every file but not their pixels. The options are
        those of the reader of the image, except `fill`, which concerns
        the pixels.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        pattern : str, optional
            The glob pattern, relative to the folder, that selects the
            files of the stack. By default, every image file of the
            folder is taken.
        axis : {"x", "y", "z", 0, 1, 2}, optional
            The axis of the volume along which the files are stacked, z
            by default.
        pixel_size : float or Sequence[float], optional
            The size of a pixel of the files, one size or a pair
            `(columns, rows)`, which replaces the size that they record.
        spacing : float, optional
            The distance from one slice to the next, which replaces the
            spacing that the files record.
        unit : str or Unit, optional
            The unit of length of `pixel_size` and `spacing`.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        StackMetadata
            The metadata, which holds the record of the stack.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ValueError
            If an option is not valid.
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the folder does not hold a stack that can be read.
        """
        refuse_options(cls, kwargs)
        record = StackRaw.from_folder(
            filename,
            pattern,
            axis=axis,
            pixel_size=pixel_size,
            spacing=spacing,
            unit=unit,
        )
        return cls.from_raw(record)

    from_fileobj = disabled(ParserTypeError, NOT_A_STREAM)
