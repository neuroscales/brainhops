"""Multiscale TIFF images: pyramidal series with one image per level."""

import typing_extensions as tx

from brainhops._core.streams import preserve_position
from brainhops.datamodel.images import MultiScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    WriterError,
)
from brainhops.io.images.base import FileBasedImage
from brainhops.io.images.tiff import _utils as backend
from brainhops.io.images.tiff._image import (
    TiffImage,
    _bigtiff,
    _TiffMixin,
    _wrap_write,
    _write_geometry,
)

# A vendor whole-slide pyramid scores above a single-scale TIFF (LIKELY) and
# below the dedicated OpenSlide reader (CERTAIN).
_WHOLE_SLIDE = 0.9

# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


@register_format
class TiffMultiScaleImage(
    _TiffMixin,
    BinaryFileReader,
    BinaryFileWriter,
    FileBasedImage,
    MultiScaleImage,
):
    """A pyramidal TIFF series, read as a multiscale image.

    The series is an OME-TIFF or plain TIFF with SubIFD levels, or any other
    pyramid that tifffile recognises. Each level is a [`TiffImage`][], finest
    first, whose pixels are read on first access. All levels map to the same
    `"physical"` system, so the pyramid has no transformation of its own, like
    an OME-Zarr pyramid that declares no common transformation.

    The pixel size of a level is the pixel size of the base level multiplied
    by the downsampling factor of the level. Along each axis, this factor is
    the base shape divided by the shape of the level. The levels are aligned by
    their extent, so pixel `i` of a level downsampled by `f` is centred on the
    base coordinate `f * i + (f - 1) / 2`. This coordinate is the centre of the
    block of base pixels that the pixel summarises, as in block-averaged and
    OME-Zarr pyramids.
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
        """Return the confidence that an open file holds a pyramid.

        The score is `CERTAIN` when the requested series (the first by default)
        has several levels and no level is requested, and `NO` otherwise. A
        vendor whole-slide image (Aperio SVS, Hamamatsu NDPI, Philips, Leica
        SCN or Ventana BIF) scores slightly less (0.9), still above
        [`TiffImage`][], so that the dedicated reader of
        [`brainhops.io.images.openslide`][] takes it when installed.
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
                    slide = backend.is_whole_slide(tif)
        except Exception:
            return Confidence.NO
        if levels <= 1:
            return Confidence.NO
        return _WHOLE_SLIDE if slide else Confidence.CERTAIN

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
        """Read every level of one series.

        The options are those of [`TiffImage.from_source`][], except `level`.

        Raises
        ------
        TypeError
            If an unknown option is given.
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
        """Return a function that writes the pyramid in OME-TIFF or plain TIFF,
        since ImageJ cannot store pyramids.

        The full-resolution level is the main image, and the other levels are
        its SubIFDs. The pixel size is the one of the full-resolution level,
        and the placement of the other levels is derived again from their
        shapes on reading.
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
