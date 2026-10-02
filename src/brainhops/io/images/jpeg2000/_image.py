# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.base import _utils_raster as raster
from brainhops.io.images.jpeg2000 import _utils as backend
from brainhops.io.images.pillow._utils import array_to_pillow, encode_pillow

# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def _pixel_size(
    header: backend.Jpeg2000Header,
) -> tx.Optional[tx.Tuple[float, float]]:
    """
    The pixel size `(x, y)` in millimetres that a JP2 file records: from
    its capture resolution, or else its display resolution. A placeholder
    (72 or 96 dpi, or one point per inch) is unknown.
    """
    for resolution in (header.capture_resolution, header.display_resolution):
        if resolution is None:
            continue
        dpi = [value * raster.MM_PER_INCH / 1000 for value in resolution]
        if raster.is_default_dpi(dpi):
            continue
        return 1000 / resolution[0], 1000 / resolution[1]
    return None


def _level_transformation(
    axes: tx.Sequence[Axis],
    scales: tx.Mapping[str, raster.AxisScale],
    factor: int,
    start: tx.Sequence[int],
) -> Transformation:
    """
    The pixel-to-physical transformation of a resolution level reduced by
    `factor`, whose first pixel is centred on the full-resolution pixel
    `start` (`(x, y)`): pixel `i` lands at `size * (factor * i + start)`.
    """
    if factor == 1:
        return raster.raster_transformations(axes, scales)[0]
    level_scales = dict(scales)
    translation = []
    offsets = {"x": start[0], "y": start[1]}
    for axis in axes:
        size, unit = scales.get(axis.name, (1.0, None))
        if axis.name in offsets:
            level_scales[axis.name] = (size * factor, unit)
        translation.append(size * offsets.get(axis.name, 0))
    scaling = raster.raster_transformations(axes, level_scales)[0]
    if not any(translation):
        return scaling
    n = len(axes)
    matrix = np.zeros((n, n + 1))
    matrix[:, :-1] = np.diag(np.asarray(scaling.scale, dtype=float))
    matrix[:, -1] = translation
    return Affine(matrix=matrix, input=scaling.input, output=scaling.output)


def _storage_codes(header: backend.Jpeg2000Header) -> str:
    return "YXS" if len(header.bit_depths) > 1 else "YX"


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _encode(
    container: str,
    data: tx.Any,
    xform: tx.Optional[Transformation],
    header: tx.Optional[backend.Jpeg2000Header],
    num_resolutions: tx.Optional[int] = None,
    **options,
) -> bytes:
    """
    Encode an image as a JP2 file or raw codestream, losslessly unless
    `options` ask otherwise. The pixel size (or the resolutions of the file
    it was read from) and the extra boxes and comment of that file are
    written back.
    """
    storage, sizes = raster.planar_storage(data, xform)
    storage = np.ascontiguousarray(np.asarray(storage))
    rows, columns = storage.shape[:2]
    most = backend.default_resolutions((rows, columns))
    if num_resolutions is None:
        num_resolutions = most
        if header is not None:
            num_resolutions = min(header.n_levels, most)
    elif not 1 <= int(num_resolutions) <= most:
        raise WriterError(
            f"An image of {columns} x {rows} pixels has between 1 and {most} "
            f"resolution levels, not {num_resolutions}."
        )
    options["num_resolutions"] = int(num_resolutions)
    if container == "j2k":
        options["no_jp2"] = True
    if header is not None and "comment" not in options:
        comments = [
            c for c in header.comments if not backend.is_openjpeg_comment(c)
        ]
        if comments:
            options["comment"] = comments[0]
    content = encode_pillow(array_to_pillow(storage), "JPEG2000", **options)
    if container == "j2k":
        return content
    capture = display = None
    boxes: tx.Sequence[tx.Tuple[bytes, bytes]] = ()
    if header is not None:
        capture = header.capture_resolution
        display = header.display_resolution
        boxes = header.boxes
    if sizes is not None:
        capture = (1000 / sizes["x"], 1000 / sizes["y"])
    return backend.add_boxes(content, capture, display, boxes)


# ----------------------------------------------------------------------
#   SHARED BASE
# ----------------------------------------------------------------------


class _Jpeg2000Format:
    """Sniffing, reading and writing shared by every JPEG 2000 image."""

    HINTS = ("jpeg2000",)

    _CONTAINER: tx.ClassVar[str] = "jp2"

    @classmethod
    def _sniff_header(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> tx.Optional[backend.Jpeg2000Header]:
        """The header of a file in this class's container, or `None`."""
        with preserve_position(file):
            head = file.read(12)
        if backend.container_of(head) == cls._CONTAINER:
            try:
                return backend.read_header(file)
            except Exception:
                pass
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                f"This content is not a JPEG 2000 file in the "
                f"{cls._CONTAINER.upper()} container."
            )
        return None

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold its
        format."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """
        Read the image from a file. A local file is reopened by name when
        a level is decoded; a remote file is read into memory. See
        `from_source` for the options.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        source = backend.Jpeg2000Source.from_filename(filename)
        return cls.from_source(source, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read the image from an open binary file, which is read into
        memory and left where it was. See `from_source` for the
        options."""
        with preserve_position(file):
            content = file.read()
        return cls.from_bytes(content, **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read the image from the bytes of a file. See `from_source` for
        the options."""
        source = backend.Jpeg2000Source(content=bytes(content))
        return cls.from_source(source, **kwargs)

    @classmethod
    def from_source(cls, source: backend.Jpeg2000Source, **kwargs) -> tx.Self:
        raise NotImplementedError  # pragma: no cover

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the image to an open binary file. See `to_bytes` for the
        options."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the image to a file. See `to_bytes` for the options.

        The image is encoded before the file is opened, so an image that
        cannot be written leaves no file behind.
        """
        content = self.to_bytes(**kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("wb") as f:
            f.write(content)

    def to_bytes(self, **kwargs) -> bytes:
        raise NotImplementedError  # pragma: no cover


# ----------------------------------------------------------------------
#   SINGLE-SCALE IMAGE
# ----------------------------------------------------------------------


class Jpeg2000Image(
    _Jpeg2000Format,
    BinaryFileParserWriter,
    WritableFileBasedImage,
    SingleScaleImage,
):
    """
    One resolution level of a JPEG 2000 image, read and written with
    Pillow (OpenJPEG). Its concrete formats are
    [`Jp2Image`][brainhops.io.images.jpeg2000.Jp2Image] (the JP2
    container) and [`J2kImage`][brainhops.io.images.jpeg2000.J2kImage]
    (a raw codestream).

    The data is F-ordered, `(x, y)` or `(x, y, c)`, and is decoded when it
    is first accessed: only the level asked for is decoded. The only
    transformation is a scaling (with a translation for a reduced level
    whose first pixel is not centred on the first full-resolution pixel)
    from the pixel system to a `"physical"` system.
    """

    header: tx.Annotated[
        tx.Optional[backend.Jpeg2000Header],
        tx.Doc(
            "What the file records besides the pixels: the codestream's "
            "main header (offsets, tiles, bit depths, levels, layers, "
            "comments) and, for a JP2 file, its resolution and extra boxes."
        ),
    ] = None

    level: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The resolution level that was read (0: full resolution)."),
    ] = None

    @property
    def n_levels(self) -> tx.Optional[int]:
        """The number of resolution levels of the file."""
        return None if self.header is None else self.header.n_levels

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        source = getattr(self, "_source", None)
        if source is None or self.header is None:
            return None
        raw, codes = backend.decode_level(
            source, self.header, int(self.level or 0)
        )
        data, _ = raster.to_canonical(raw, codes)
        return data

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score how confident the class is that an open file holds an image
        in its container: `LIKELY` when it does. An image with several
        resolution levels is claimed more confidently by the multiscale
        class, unless a `level` is asked for.
        """
        if cls._sniff_header(file, error) is not None:
            return Confidence.LIKELY
        return Confidence.NO

    # --- load ---------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source: backend.Jpeg2000Source,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        header: tx.Optional[backend.Jpeg2000Header] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read one resolution level of a JPEG 2000 file. Its pixels are
        decoded when its data is first accessed.

        Parameters
        ----------
        source : Jpeg2000Source
            Where the file is.
        level : int, optional
            The resolution level to read: 0 (the default) is the full
            resolution, and level `r` is reduced by `2**r`. Negative values
            count from the coarsest.
        pixel_size : float | Sequence[float] | Mapping[str, float], optional
            The pixel size of the full-resolution level, which overrides
            the file's: one size, `(x, y)`, or a mapping by axis name.
        unit : str | Unit, optional
            The unit of `pixel_size`, or the unit to convert the file's
            size (in millimetres) to.
        header : Jpeg2000Header, optional
            The header, if it has been parsed already.

        Raises
        ------
        ParserContentError
            If the file is not a JPEG 2000 file.
        IndexError
            If it has no such level.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        if header is None:
            with source.open() as f:
                header = backend.read_header(f)
        n = header.n_levels
        index = 0 if level is None else int(level)
        if not -n <= index < n:
            raise IndexError(
                f"This file has {n} resolution level(s), so it has no level "
                f"{level}."
            )
        index %= n
        axes = raster.storage_axes(_storage_codes(header))
        axes = [axes[i] for i in raster.canonical_permutation(axes)]
        metadata = {}
        sizes = _pixel_size(header)
        if sizes is not None:
            metadata = {"x": (sizes[0], "mm"), "y": (sizes[1], "mm")}
        scales = raster.resolve_pixel_size(
            axes, metadata, pixel_size=pixel_size, unit=unit
        )
        xform = _level_transformation(
            axes, scales, 2**index, header.level_start(index)
        )
        image = cls(transformations=[xform], header=header, level=index)
        image._source = source
        return image

    # --- save ---------------------------------------------------------

    def to_bytes(
        self, num_resolutions: tx.Optional[int] = None, **options
    ) -> bytes:
        """
        Encode the image, losslessly by default (the reversible 5-3
        wavelet, one quality layer).

        Parameters
        ----------
        num_resolutions : int, optional
            The number of resolution levels to store. By default, as many
            as the file the image was read from had, or six, as long as
            the coarsest level has a pixel.
        **options
            Passed on to Pillow's JPEG 2000 writer, e.g.
            `quality_mode="rates", quality_layers=[20], irreversible=True`
            for lossy compression, or `tile_size=(256, 256)`.

        Raises
        ------
        WriterError
            If the image is not a two-dimensional raster, or its data type
            cannot be stored (Pillow writes 8-bit grey, grey + alpha, RGB
            and RGBA, and 16-bit grey).
        """
        return _encode(
            self._CONTAINER,
            self.data,
            self.transformation,
            self.header,
            num_resolutions,
            **options,
        )


@register_format
class Jp2Image(Jpeg2000Image):
    """
    One resolution level of an image in a JP2 file (the JPEG 2000 file
    format, with boxes), as a single-scale image. See
    [`Jpeg2000Image`][brainhops.io.images.jpeg2000.Jpeg2000Image].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.JP2_EXTENSIONS
    HINTS = ("jp2",)
    _CONTAINER = "jp2"


@register_format
class J2kImage(Jpeg2000Image):
    """
    One resolution level of a raw JPEG 2000 codestream, as a single-scale
    image. See
    [`Jpeg2000Image`][brainhops.io.images.jpeg2000.Jpeg2000Image].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.J2K_EXTENSIONS
    HINTS = ("j2k", "j2c")
    _CONTAINER = "j2k"


# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


class Jpeg2000MultiScaleImage(
    _Jpeg2000Format,
    BinaryFileParserWriter,
    WritableFileBasedImage,
    MultiScaleImage,
):
    """
    Every resolution level of a JPEG 2000 image, as a multiscale image,
    finest first. Its concrete formats are
    [`Jp2MultiScaleImage`][brainhops.io.images.jpeg2000.Jp2MultiScaleImage]
    and
    [`J2kMultiScaleImage`][brainhops.io.images.jpeg2000.J2kMultiScaleImage].

    Each level is a single-scale image of the same format, decoded only
    when its data is accessed. Level `r` is reduced by `2**r`, and its
    pixel `i` is centred on the full-resolution pixel `2**r * i` (plus a
    shift when the image area has an offset): the wavelet transform keeps
    the even samples of each level. Every level maps onto the same
    `"physical"` system, so the pyramid's own transformations are empty.
    """

    _LEVEL: tx.ClassVar[tx.Type[Jpeg2000Image]] = Jpeg2000Image

    header: tx.Annotated[
        tx.Optional[backend.Jpeg2000Header],
        tx.Doc(
            "What the file records besides the pixels: the codestream's "
            "main header (offsets, tiles, bit depths, levels, layers, "
            "comments) and, for a JP2 file, its resolution and extra boxes."
        ),
    ] = None

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Score how confident the class is that an open file holds a
        multiscale image in its container: `CERTAIN` when it has several
        resolution levels and no `level` is asked for, and `NO`
        otherwise. An image whose area has an offset on the reference grid
        scores `NO` too, as Pillow cannot decode its reduced levels: it is
        read at full resolution by the single-scale class.
        """
        if level is not None:
            return Confidence.NO
        header = cls._sniff_header(file, error)
        if (
            header is not None
            and header.n_levels > 1
            and not any(header.image_offset)
        ):
            return Confidence.CERTAIN
        return Confidence.NO

    # --- load ---------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source: backend.Jpeg2000Source,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read every resolution level of a JPEG 2000 file. The options are
        those of the single-scale image's `from_source`, but for `level`.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        with source.open() as f:
            header = backend.read_header(f)
        levels = [
            cls._LEVEL.from_source(
                source,
                level=index,
                pixel_size=pixel_size,
                unit=unit,
                header=header,
            )
            for index in range(header.n_levels)
        ]
        return cls(images=levels, transformations=[], header=header)

    # --- save ---------------------------------------------------------

    def to_bytes(self, **options) -> bytes:
        """
        Encode the pyramid: the full-resolution level, with as many
        resolution levels as the pyramid has. The coarser levels are not
        stored as they are: the codec derives them from the first, so they
        must have the shapes it gives them (`ceil(n / 2**r)`). See the
        single-scale image's `to_bytes` for the options.

        Raises
        ------
        WriterError
            If the pyramid has no level, or levels of other shapes.
        """
        images = list(self.images or [])
        if not images:
            raise WriterError(
                "This multiscale image has no levels, so there is nothing "
                "to write."
            )
        base = images[0]
        if base.data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        shape = tuple(int(n) for n in base.data.shape)
        for index, image in enumerate(images[1:], start=1):
            factor = 2**index
            expected = tuple(-(-n // factor) for n in shape[:2]) + shape[2:]
            got = tuple(int(n) for n in image.data.shape)
            if got != expected:
                raise WriterError(
                    f"Level {index} of this pyramid has shape {got}, but "
                    f"JPEG 2000 derives a level of shape {expected} from "
                    f"the full resolution level {shape}."
                )
        xform = base.transformation
        if self.transformations:
            xform = self.transformation @ xform
        options.setdefault("num_resolutions", len(images))
        return _encode(
            self._CONTAINER, base.data, xform, self.header, **options
        )


@register_format
class Jp2MultiScaleImage(Jpeg2000MultiScaleImage):
    """
    Every resolution level of an image in a JP2 file, as a multiscale
    image whose levels are
    [`Jp2Image`][brainhops.io.images.jpeg2000.Jp2Image]s. See
    [`Jpeg2000MultiScaleImage`][brainhops.io.images.jpeg2000.Jpeg2000MultiScaleImage].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.JP2_EXTENSIONS
    HINTS = ("jp2",)
    _CONTAINER = "jp2"
    _LEVEL = Jp2Image


@register_format
class J2kMultiScaleImage(Jpeg2000MultiScaleImage):
    """
    Every resolution level of a raw JPEG 2000 codestream, as a multiscale
    image whose levels are
    [`J2kImage`][brainhops.io.images.jpeg2000.J2kImage]s. See
    [`Jpeg2000MultiScaleImage`][brainhops.io.images.jpeg2000.Jpeg2000MultiScaleImage].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.J2K_EXTENSIONS
    HINTS = ("j2k", "j2c")
    _CONTAINER = "j2k"
    _LEVEL = J2kImage
