"""The multiscale TIFF image: a pyramidal series, one level per image."""

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.streams import preserve_position
from brainhops.datamodel.images import MultiScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    WriterError,
)
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.tiff import _utils as backend
from brainhops.io.images.tiff._image import (
    TiffImage,
    _bigtiff,
    _TiffMixin,
    _wrap_write,
    _write_geometry,
)

# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


@register_format
class TiffMultiScaleImage(
    _TiffMixin, BinaryFileParserWriter, WritableFileBasedImage, MultiScaleImage
):
    """
    A pyramidal TIFF series -- OME-TIFF or plain TIFF with SubIFDs, or any
    pyramid tifffile recognizes (`series.levels`) -- as a multiscale image.

    Each level is a [`TiffImage`][brainhops.io.images.tiff.TiffImage],
    finest first, whose pixels are read when its data is first accessed.
    Every level maps its pixels onto the same `"physical"` system, so the
    pyramid's own transformations are empty (the identity), as for an
    OME-Zarr pyramid that declares no common transformation.

    A level's pixel size is the base pixel size times its downsampling
    factor, the ratio of the base shape to its own along each axis. Levels
    are aligned by their *extent*: the edges of a level's first and last
    pixels coincide with those of the base level, so pixel `i` of a level
    downsampled by `f` is centred on the base level's pixel coordinate
    `f * i + (f - 1) / 2`, the centre of the block of base pixels it
    summarizes. This is the convention of block-averaged pyramids (and of
    OME-Zarr pyramids whose levels carry the matching translation).
    """

    dialect: tx.Annotated[
        tx.Optional[str],
        tx.Doc("The metadata dialect of the file: 'ome', 'imagej' or None."),
    ] = None

    ome_xml: tx.Annotated[
        tx.Optional[str], tx.Doc("The OME-XML of an OME-TIFF file.")
    ] = None

    imagej_metadata: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc("The ImageJ metadata of an ImageJ file."),
    ] = None

    tags: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc("The tags of interest of the first page, by name."),
    ] = None

    series: tx.Annotated[
        tx.Optional[int], tx.Doc("The index of the series that was read.")
    ] = None

    n_series: tx.Annotated[
        tx.Optional[int], tx.Doc("The number of series in the file.")
    ] = None

    storage_axes: tx.Annotated[
        tx.Optional[str],
        tx.Doc("The axes of the series as tifffile stores them."),
    ] = None

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        series: int = 0,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Score how confident the class is that an open file holds a
        pyramid: `CERTAIN` when the series asked for (the first by
        default) has several levels and no `level` is asked for, and `NO`
        otherwise.
        """
        if level is not None:
            return Confidence.NO
        if not cls._sniff_head(file, error):
            return Confidence.NO
        tf = backend._require_tifffile()
        try:
            with preserve_position(file):
                with tf.TiffFile(file) as tif:
                    levels = len(tif.series[int(series)].levels)
        except Exception:
            return Confidence.NO
        return Confidence.CERTAIN if levels > 1 else Confidence.NO

    # --- load ---------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source: backend.TiffSource,
        series: int = 0,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        origin: tx.Any = None,
        mmap: bool = True,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read every level of one series of a TIFF file. The options are
        those of [`TiffImage`][brainhops.io.images.tiff.TiffImage]
        `.from_source`, but for `level`.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        with source.open() as tif:
            n = len(tif.series)
            backend._select(tif, int(series), 0)
            series = int(series) % n
            base = tif.series[series]
            count = len(base.levels)
        levels = [
            TiffImage.from_source(
                source,
                series=series,
                level=index,
                pixel_size=pixel_size,
                unit=unit,
                origin=origin,
                mmap=mmap,
                lazy=lazy,
            )
            for index in range(count)
        ]
        first = levels[0]
        image = cls(
            images=levels,
            transformations=[],
            dialect=first.dialect,
            ome_xml=first.ome_xml,
            imagej_metadata=first.imagej_metadata,
            tags=first.tags,
            series=series,
            n_series=first.n_series,
            storage_axes=first.storage_axes,
        )
        image._ome_index = getattr(first, "_ome_index", None)
        return image

    # --- save ---------------------------------------------------------

    def _writer(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
        **options,
    ) -> tx.Callable[[tx.Any], None]:
        """
        Write the pyramid: the full-resolution level as the main image,
        and the others as its SubIFDs, in OME-TIFF or plain TIFF (ImageJ
        stores no pyramid). The pixel size is that of the full-resolution
        level; the other levels' placement is not stored, and is derived
        again from their shapes when the file is read.
        """
        tf = backend._require_tifffile()
        images = list(self.images or [])
        if not images:
            raise WriterError(
                "This multiscale image has no levels, so there is nothing "
                "to write."
            )
        xform = images[0].transformation
        if self.transformations:
            xform = self.transformation @ xform
        storage, codes, axes = self._level_storage(
            images[0].data, images[0].transformation, self.storage_axes
        )
        levels = [storage]
        for index, image in enumerate(images[1:], start=1):
            stored, _, _ = self._level_storage(
                image.data, image.transformation, self.storage_axes, codes
            )
            if stored.ndim != storage.ndim:
                raise WriterError(
                    f"Level {index} of this pyramid does not have as many "
                    f"axes as its first level."
                )
            levels.append(stored)
        geometry = _write_geometry(xform, axes)
        _, opener, write = self._options(
            storage, codes, geometry, name, dialect, allow_imagej=False
        )
        write.update(options)
        nbytes = sum(level.nbytes for level in levels)
        opener["bigtiff"] = _bigtiff(nbytes, bigtiff)
        sub = {
            key: write[key]
            for key in ("photometric", "planarconfig")
            if key in write
        }
        sub.update(options)

        def run(target: tx.Any) -> None:
            def go() -> None:
                with tf.TiffWriter(target, **opener) as writer:
                    writer.write(levels[0], subifds=len(levels) - 1, **write)
                    for level in levels[1:]:
                        writer.write(level, subfiletype=1, **sub)

            _wrap_write(go)

        return run
