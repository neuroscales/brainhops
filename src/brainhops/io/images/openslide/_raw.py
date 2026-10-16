"""The record of a whole-slide image that OpenSlide reads."""

# dependencies
import typing_extensions as tx
from bagof.magic import Factory

from brainhops.io.base._raw import Raw


class OpenSlideLevelRaw(Raw):
    """Storage description of one level of a whole-slide image.

    A slide stores the same image at several resolutions, its levels,
    finest first. This record describes how OpenSlide presents one level,
    without its pixels.
    """

    shape: tx.Tuple[int, ...] = ()
    """The shape of the level as OpenSlide decodes it: its width and its
    height in pixels, as `level_dimensions` gives them, and its three RGB
    samples."""

    dtype: str = "uint8"
    """The name of the NumPy data type of the pixels, always `"uint8"`."""

    downsample: float = 1.0
    """The downsampling factor of the level with respect to the finest
    level, as `level_downsamples` gives it."""


class OpenSlideRaw(Raw):
    """Record of a whole-slide image, without its pixels.

    A whole-slide image, or slide, is the pyramid that a slide scanner
    writes, in the format of its vendor. OpenSlide describes every slide in
    the same way, whatever its vendor: a set of named properties, the
    dimensions and the downsampling factor of each level, and the names of
    the associated images, such as the label and the macro image. The
    record holds this description, as OpenSlide reports it when the slide
    is opened. [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage]
    and
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage]
    hold it through
    [`OpenSlideMetadata`][brainhops.io.images.openslide.OpenSlideMetadata],
    and every level of a pyramid holds the same record, in which its index
    `level` finds it.

    The record holds no open file and can be pickled. It is never changed
    in place. OpenSlide cannot write a slide, so the record is never
    encoded: it describes the slide as it was read.
    """

    properties: tx.Dict[str, str] = Factory(dict)
    """Every property that OpenSlide reports for the slide, by name: the
    standard `openslide.*` properties and those of the vendor, such as
    `aperio.AppMag` or `tiff.Software`."""

    levels: tx.Tuple[OpenSlideLevelRaw, ...] = ()
    """The storage description of each level, finest first."""

    associated_images: tx.Tuple[str, ...] = ()
    """The names of the associated images of the slide, such as `"label"`,
    `"macro"` and `"thumbnail"`, as `associated_images` lists them."""

    @property
    def vendor(self) -> tx.Optional[str]:
        """The vendor of the slide, as OpenSlide names it.

        OpenSlide sets the property `openslide.vendor` to the name of the
        format that opened the slide (`openslide.c` of OpenSlide 4.0.0), which
        is the name that `OpenSlide.detect_format` gives, such as `"aperio"`
        or `"generic-tiff"`.
        """
        return self.properties.get("openslide.vendor")

    @property
    def level_downsamples(self) -> tx.Tuple[float, ...]:
        """The downsampling factor of each level, finest first."""
        return tuple(level.downsample for level in self.levels)

    @property
    def background_color(self) -> tx.Optional[str]:
        """The background colour of the slide, as `RRGGBB`, or `None`.

        The colour fills the pixels outside the scanned area, and white is
        used when the slide records none.
        """
        return self.properties.get("openslide.background-color")

    @property
    def bounds(self) -> tx.Optional[tx.Tuple[int, int, int, int]]:
        """The bounding box of the scanned area, or `None`.

        The box is `(x, y, width, height)` in full-resolution pixels, as the
        properties `openslide.bounds-*` record it. It is metadata only, and
        it does not change the geometry of the levels.
        """
        try:
            return tuple(
                int(self.properties[f"openslide.bounds-{key}"])
                for key in ("x", "y", "width", "height")
            )
        except (KeyError, ValueError):
            return None


def read_record(handle: tx.Any) -> OpenSlideRaw:
    """Read the record of a slide that OpenSlide has opened.

    The record is read from the properties, the level dimensions, the
    level downsamples and the names of the associated images, and no pixel
    is decoded. OpenSlide itself reads the raw tiles of the coarsest level
    of a TIFF-based slide when it opens it, to compute the property
    `openslide.quickhash-1`.

    Parameters
    ----------
    handle : openslide.OpenSlide
        The open slide.

    Returns
    -------
    OpenSlideRaw
        The record of the slide.
    """
    levels = tuple(
        OpenSlideLevelRaw(
            shape=(int(width), int(height), 3),
            dtype="uint8",
            downsample=float(downsample),
        )
        for (width, height), downsample in zip(
            handle.level_dimensions, handle.level_downsamples
        )
    )
    return OpenSlideRaw(
        properties={str(k): str(v) for k, v in handle.properties.items()},
        levels=levels,
        associated_images=tuple(
            str(name) for name in handle.associated_images.keys()
        ),
    )
