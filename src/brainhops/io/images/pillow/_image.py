# stdlib
from io import BytesIO
from numbers import Number

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.datamodel.images import SingleScaleImage
from brainhops.io.base import raster
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileParserWriter,
    Confidence,
    SnifferContentError,
    WriterError,
)
from brainhops.io.base.pillow import (
    EXTENSIONS,
    array_to_pillow,
    can_write,
    encode_pillow,
    format_for_name,
    read_pillow,
    sniff_pillow,
)
from brainhops.io.images.base import WritableFileBasedImage

_DpiLike = tx.Union[None, bool, float, tx.Sequence[float]]

# Metadata that Pillow reads into `info` and that its writers take back as
# a save option of the same name. It is carried over when the image is
# written, so that it survives a round trip.
_ROUND_TRIP_INFO = ("icc_profile", "exif")

# The format an image is written in when nothing says which.
_DEFAULT_FORMAT = "PNG"


def _dpi_pair(dpi: tx.Any) -> tx.Tuple[float, float]:
    """A resolution given as one number or as an `(x, y)` pair."""
    if isinstance(dpi, (Number, np.number)):
        return float(dpi), float(dpi)
    values = [float(v) for v in np.ravel(dpi)]
    if len(values) == 1:
        values = values * 2
    if len(values) != 2:
        raise ValueError(
            f"A resolution is one number or an (x, y) pair, not {dpi!r}."
        )
    return values[0], values[1]


@register_format
class PillowImage(
    BinaryFileParserWriter, WritableFileBasedImage, SingleScaleImage
):
    """
    A two-dimensional raster image (PNG, JPEG, BMP, GIF, WebP, ...) read
    and written with Pillow.

    The data is F-ordered: `(x, y)` for a single-component image, and
    `(x, y, c)` when each pixel has several samples (grey + alpha, RGB,
    RGBA, ...). The only transformation is a scaling from the pixel system
    to a `"physical"` system, which is the identity, in no unit, unless a
    pixel size is known (see the module documentation).

    What the file records beyond the pixels is kept on the object --
    `image_format`, `mode`, `info`, `frame` and `n_frames` -- and the ICC
    profile and EXIF block in `info` are written back when the image is
    saved.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = EXTENSIONS
    HINTS = ("pillow",)

    # --- format-specific metadata -------------------------------------

    image_format: tx.Annotated[
        tx.Optional[str],
        tx.Doc("Pillow's name for the format of the file, such as 'PNG'."),
    ] = None

    mode: tx.Annotated[
        tx.Optional[str],
        tx.Doc(
            "Pillow's mode of the frame as stored, before any conversion, "
            "such as 'P', 'RGB' or 'I;16'."
        ),
    ] = None

    info: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc("Pillow's `info` dictionary: the format-specific metadata."),
    ] = None

    frame: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The index of the frame that was read."),
    ] = None

    n_frames: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The number of frames in the file."),
    ] = None

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
        Pillow reads.

        A format recognized from its magic number scores `LIKELY`. TIFF
        scores `WEAK`: Pillow reads it, but the dedicated TIFF reader
        reads it better, so Pillow is only a fallback.
        """
        fmt = sniff_pillow(file)
        if fmt == "TIFF":
            return Confidence.WEAK
        if fmt is not None:
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Pillow does not recognize this content as an image.")
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the class is that bytes hold an image
        Pillow reads."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    # --- load ---------------------------------------------------------

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        frame: int = 0,
        palette: bool = True,
        dpi: _DpiLike = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read an image from an open file.

        Parameters
        ----------
        file : IO
            A binary file object, open for reading. It is not closed.
        frame : int
            The frame of a multi-frame file (an animated GIF, PNG or WebP)
            to read. Negative values count from the end. Default: the
            first.
        palette : bool
            Look the colours of a palette image up, giving an RGB(A) image
            (the default), or keep the palette indices.
        dpi : bool | float | (float, float), optional
            Whether to take the pixel size from the resolution, in dots
            per inch. By default (`None` or `False`) the resolution the
            file records is ignored, as it most often describes a screen
            or a printer rather than the scene. `True` uses it, unless it
            is absent or a placeholder (72 or 96 dpi). A number, or an
            `(x, y)` pair, is a resolution to use instead. A pixel is then
            `25.4 / dpi` millimetres.
        pixel_size : float | Sequence[float] | Mapping[str, float], optional
            The pixel size, which overrides the resolution: one size, or
            `(x, y)`.
        unit : str | Unit, optional
            The unit of `pixel_size`, or the unit to convert the size
            from `dpi` to (default: millimetres).

        Raises
        ------
        ParserContentError
            If Pillow cannot read the file.
        IndexError
            If the file has no frame `frame`.
        PIL.Image.DecompressionBombError
            If the image is larger than Pillow's safety limit,
            `PIL.Image.MAX_IMAGE_PIXELS`.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        frame_data = read_pillow(file, frame=frame, palette=palette)
        data, axes = raster.to_canonical(frame_data.array, frame_data.axes)

        metadata: tx.Dict[str, raster.AxisScale] = {}
        resolution = None
        if dpi is True:
            if not raster.is_default_dpi(frame_data.dpi):
                resolution = frame_data.dpi
        elif dpi is not None and dpi is not False:
            resolution = _dpi_pair(dpi)
        if resolution is not None:
            names = [axis.name for axis in raster.spatial_axes(axes)]
            metadata = {
                name: (raster.dpi_to_size(value), "mm")
                for name, value in zip(names, resolution)
            }
        scales = raster.resolve_pixel_size(
            axes, metadata, pixel_size=pixel_size, unit=unit
        )
        return cls(
            data=data,
            transformations=raster.raster_transformations(axes, scales),
            image_format=frame_data.format,
            mode=frame_data.mode,
            info=frame_data.info,
            frame=frame_data.frame,
            n_frames=frame_data.n_frames,
        )

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read an image from the bytes of a file. See `from_fileobj` for
        the options."""
        return cls.from_fileobj(BytesIO(content), **kwargs)

    # --- save ---------------------------------------------------------

    def _storage(
        self,
    ) -> tx.Tuple[np.ndarray, tx.Optional[tx.Dict[str, float]]]:
        """
        The pixels as Pillow stores them, `(rows, columns[, samples])`, and
        the pixel size along `x` and `y` in millimetres, if it is known.
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        shape = tuple(int(d) for d in data.shape)
        ndim = len(shape)
        xform = self.transformation
        axes = raster.image_axes(ndim, getattr(xform, "input", None))
        if axes is None:
            if ndim == 2:
                axes = raster.default_axes(2)
            elif ndim == 3 and shape[-1] <= 4:
                axes = raster.default_axes(3, channel=True)
            else:
                raise WriterError(
                    f"Cannot tell the axes of an array of shape {shape}: a "
                    f"raster image is (x, y) or (x, y, c) with at most four "
                    f"channels. Give the image a pixel coordinate system "
                    f"whose axes say which is which."
                )
        sizes = raster.physical_pixel_size(xform, axes, "mm")

        # Keep two spatial axes and at most one channel axis; any other
        # axis must be a singleton, and is dropped.
        space = [
            i for i, a in enumerate(axes) if raster.axis_group(a) == "space"
        ]
        channel = [
            i for i, a in enumerate(axes) if raster.axis_group(a) == "channel"
        ]
        if len(space) > 2:
            # A slice of a volume: keep the two spatial axes that are not
            # singletons (or, failing that, the first ones), in order.
            wide = [i for i in space if shape[i] != 1]
            narrow = [i for i in space if shape[i] == 1]
            if len(wide) <= 2:
                space = sorted(wide + narrow[: 2 - len(wide)])
        if len(space) != 2:
            raise WriterError(
                f"A raster image has two spatial axes, but this one has "
                f"{len(space)}, not counting singletons (shape {shape}). "
                f"Extract a 2D slice before writing it."
            )
        channel = [i for i in channel if shape[i] != 1] or channel[:1]
        if len(channel) > 1:
            raise WriterError(
                f"A raster image has at most one channel axis, but this one "
                f"has {len(channel)} (shape {shape})."
            )
        keep = space + channel
        drop = [i for i in range(ndim) if i not in keep]
        if any(shape[i] != 1 for i in drop):
            names = [axes[i].name for i in drop if shape[i] != 1]
            raise WriterError(
                f"A raster image has two spatial axes and one channel axis, "
                f"so the axes {names} (shape {shape}) cannot be written. "
                f"Select one of their elements before writing."
            )
        index = tuple(slice(None) if i in keep else 0 for i in range(ndim))
        data = data[index]
        # The two spatial axes kept are, in order, the columns (x) and the
        # rows (y) of the raster, whatever their names: a sagittal slice
        # (y, z) is written with y along the columns. Pillow stores rows
        # first, then columns, then samples.
        position = {axis: k for k, axis in enumerate(sorted(keep))}
        order = [position[space[1]], position[space[0]]]
        order += [position[i] for i in channel]
        storage = data.transpose(order)

        if sizes is not None:
            x, y = (axes[i].name for i in space)
            sizes = {"x": sizes[x], "y": sizes[y]}
        return storage, sizes

    def to_bytes(
        self,
        format: tx.Optional[str] = None,
        dpi: _DpiLike = None,
        **options,
    ) -> bytes:
        """
        Encode the image in a raster format.

        Parameters
        ----------
        format : str, optional
            Pillow's name for the format, such as `"PNG"` or `"JPEG"`. By
            default, the format the image was read from, if Pillow can
            write it, or else PNG.
        dpi : bool | float | (float, float), optional
            The resolution to record, in dots per inch, in the formats that
            store one (PNG, JPEG, BMP, TIFF). By default (`None` or
            `True`), it is computed from the pixel size when the
            preferred transformation is a scaling onto axes measured in a
            unit of length; failing that, a resolution the file was read
            with is written back. `False` records none. A number or an
            `(x, y)` pair is recorded as it is.
        **options
            Passed on to Pillow's `Image.save`, e.g. `quality=95` for JPEG
            or `compress_level=9` for PNG.

        Raises
        ------
        WriterError
            If the data cannot be stored in the format: more than two
            spatial axes, a data type Pillow does not store (e.g. `float64`
            or `int16`), or one the format does not (e.g. `uint16` in
            JPEG). Nothing is converted silently.
        """
        storage, sizes = self._storage()
        im = array_to_pillow(storage)

        if format is None:
            format = self.image_format
            if format == "MPO":
                format = "JPEG"
            if format is not None and not can_write(format):
                format = None
        format = (format or _DEFAULT_FORMAT).upper()

        resolution = None
        if dpi is None or dpi is True:
            if sizes is not None:
                resolution = (
                    raster.size_to_dpi(sizes["x"]),
                    raster.size_to_dpi(sizes["y"]),
                )
            elif self.info and "dpi" in self.info:
                resolution = _dpi_pair(self.info["dpi"])
        elif dpi is not False:
            resolution = _dpi_pair(dpi)

        for key in _ROUND_TRIP_INFO:
            if self.info and self.info.get(key) and key not in options:
                options[key] = self.info[key]
        return encode_pillow(im, format, dpi=resolution, **options)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """
        Write the image to an open binary file.

        The format is the `format` keyword if given, or else the one the
        file's name calls for, if it has one (see `to_bytes`).
        """
        if kwargs.get("format") is None:
            kwargs["format"] = format_for_name(_to_filename(file))
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the image to a file, in the format its extension calls for
        (or the `format` keyword). See `to_bytes` for the options.

        The image is encoded before the file is opened, so an image the
        format cannot store leaves no file behind.
        """
        if kwargs.get("format") is None:
            fmt = format_for_name(filename)
            if fmt is None:
                raise WriterError(
                    f"Cannot tell which format to write {str(filename)!r} "
                    f"in from its extension. Pass format=, such as "
                    f"format='PNG'."
                )
            kwargs["format"] = fmt
        content = self.to_bytes(**kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("wb") as f:
            f.write(content)
