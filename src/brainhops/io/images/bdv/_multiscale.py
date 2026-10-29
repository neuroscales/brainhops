"""Every resolution level of one view of a BigDataViewer dataset."""

__all__ = ["BdvMultiScaleImage"]

# stdlib

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.datamodel.images import MultiScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    disabled,
)
from brainhops.io.common.bdv import BdvMetadata
from brainhops.io.common.bdv._internal_api import level_count, read_dataset
from brainhops.io.images.base import ImageFormat

# this format
from ._image import BdvImage, no_options, sniff_path, sniff_stream


@register_format
class BdvMultiScaleImage(BinaryFileReader, ImageFormat, MultiScaleImage):
    """Every resolution level of one view of a BigDataViewer dataset.

    Each level is a [`BdvImage`][], finest first, whose voxels are read
    on first access. The levels share the metadata of the dataset, and
    each of them maps its voxels to the `"physical"` and `"global"`
    spaces with the downsampling of its level included, so the pyramid
    has no transformation of its own, as for a pyramidal TIFF file. The
    reader options `setup` and `timepoint` select the view, by default
    the first setup at the first time point, in the order of their ids.
    A view with a single level, or a read with the option `level`, gives
    a [`BdvImage`][] instead.

    The class cannot write, for the reasons given in [`BdvImage`][]. The
    levels hold their record, but the pyramid itself has none yet, as
    the other pyramids of brainhops.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("bdv", "bigdataviewer")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)

    setup: tx.Optional[int] = None
    """The id of the setup of the view that the pyramid holds.

    It is the reader option `setup`, and a reader stores the id of the
    setup that it read.
    """

    timepoint: tx.Optional[int] = None
    """The id of the time point of the view that the pyramid holds.

    It is the reader option `timepoint`, and a reader stores the id of
    the time point that it read.
    """

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

        The score is `CERTAIN` for a dataset stored in HDF5, unless
        `level` is given or the setup has a single level, which
        [`BdvImage`][] reads, and then it is `NO`.

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
            The level that the caller asks for, which this class does not
            take.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if level is not None:
            return Confidence.NO
        score = sniff_path(filename, error)
        if score and level_count(filename, setup) == 1:
            return Confidence.NO
        return score

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score a stream from the XML of a dataset.

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a dataset
            that this class reads.
        level : int, optional
            The level that the caller asks for, which this class does not
            take.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` for a dataset stored in HDF5 when `level` is not
            given, and `NO` otherwise.
        """
        if level is not None:
            return Confidence.NO
        return sniff_stream(file, error)

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        setup: tx.Optional[int] = None,
        timepoint: tx.Optional[int] = None,
        **kwargs,
    ) -> "BdvMultiScaleImage":
        """Read every level of one view from the XML file of a dataset.

        The XML file and the description of the levels in the HDF5 file
        are read once, and no voxel is read.

        Parameters
        ----------
        filename : path-like
            The path of the XML file.
        setup : int, optional
            The id of the setup of the view, by default the first one.
        timepoint : int, optional
            The id of the time point of the view, by default the first
            one.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        BdvMultiScaleImage
            The pyramid, finest level first.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserExistsError
            If the XML file or the HDF5 file does not exist.
        ParserContentError
            If the dataset is not stored in HDF5, or if it has no such
            view.
        """
        no_options(cls, kwargs)
        raw, setup, timepoint, levels = read_dataset(
            filename, setup=setup, timepoint=timepoint
        )
        metadata = BdvMetadata.from_raw(raw)
        images = [
            BdvImage(
                rawdata=proxy,
                metadata=metadata,
                setup=setup,
                timepoint=timepoint,
                level=index,
            )
            for index, proxy in enumerate(levels)
        ]
        return cls(
            images=images,
            transformations=[],
            setup=setup,
            timepoint=timepoint,
        )

    from_fileobj = disabled(
        ParserContentError,
        "A BigDataViewer dataset is an XML file and the HDF5 file that "
        "it names, so a pyramid is read from the path of the XML file, "
        "not from a stream or from bytes.",
    )
