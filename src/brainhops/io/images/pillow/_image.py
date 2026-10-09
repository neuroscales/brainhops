from io import BytesIO
from numbers import Number

import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops.datamodel.images import SingleScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base import _utils_raster as raster
from brainhops.io.images.pillow._utils import (
    EXTENSIONS,
    array_to_pillow,
    can_write,
    encode_pillow,
    format_for_name,
    read_pillow,
    sniff_pillow,
)

_DpiLike = tx.Union[None, bool, float, tx.Sequence[float]]

# Pillow writers accept these keys of `info` back as save options. They are
# passed on when the image is saved, so that they survive a round trip.
_ROUND_TRIP_INFO = ("icc_profile", "exif")

# The output format when neither `format=` nor a file name names one.
_DEFAULT_FORMAT = "PNG"


def _dpi_pair(dpi: tx.Any) -> tx.Tuple[float, float]:
    """Convert a resolution, one number or an `(x, y)` pair, to a pair of
    floats.
    """
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
    BinaryFileReader,
    BinaryFileWriter,
    ImageFormat,
    SingleScaleImage,
):
    """A two-dimensional raster image, read and written with Pillow.

    The data is F-ordered, `(x, y)` for single-component images and `(x, y, c)`
    otherwise. The image has a single transformation, which scales pixels to
    `"physical"` and is the identity, with no unit, unless the pixel size is
    known (see [`brainhops.io.images.pillow`][]). The file metadata is kept in
    `image_format`, `mode`, `info`, `frame` and `n_frames`, and the ICC profile
    and EXIF block are written back on save.
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
        tx.Doc(
            "Pillow's `info` dictionary, which holds the format-specific "
            "metadata."
        ),
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
        """Return the confidence that a file holds an image that Pillow reads.

        A format that is recognised by its magic number scores `LIKELY`. TIFF
        scores only `WEAK`, because the dedicated TIFF reader reads it better.
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
        """Return the confidence that bytes hold an image that Pillow reads."""
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
        """Read an image from an open file.

        Parameters
        ----------
        file : IO
            A binary file open for reading, which is not closed.
        frame : int, optional
            The frame of a multi-frame file, the first by default. A negative
            index counts from the end.
        palette : bool, optional
            Whether palette colours are looked up (the default) or kept as
            indices.
        dpi : bool or float or tuple of float, optional
            The resolution in dots per inch, which makes pixels `25.4 / dpi`
            millimetres wide. By default, the resolution of the file is
            ignored, since it usually describes a screen or a printer. `True`
            uses the resolution of the file unless it is absent or a
            placeholder (72 or 96 dpi), and a number or an `(x, y)` pair
            replaces it.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size, which overrides any resolution.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which a size derived from
            a resolution is converted. Millimetres by default.

        Raises
        ------
        ParserContentError
            If Pillow cannot read the file.
        IndexError
            If the file has no such frame.
        PIL.Image.DecompressionBombError
            If the image exceeds the limit set by `PIL.Image.MAX_IMAGE_PIXELS`.
        TypeError
            If an unknown option is given.
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
        """Read an image from bytes.

        The options are those of `from_fileobj`.
        """
        return cls.from_fileobj(BytesIO(content), **kwargs)

    # --- save ---------------------------------------------------------

    def _storage(
        self,
    ) -> tx.Tuple[np.ndarray, tx.Optional[tx.Dict[str, float]]]:
        """Return the pixels as Pillow stores them,
        `(rows, columns[, samples])`, together with the pixel size along x and
        y in millimetres, if it is known.
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

        # Keep two spatial axes and at most one channel axis. Any other axis
        # must be a singleton, and is dropped.
        space = [
            i for i, a in enumerate(axes) if raster.axis_group(a) == "space"
        ]
        channel = [
            i for i, a in enumerate(axes) if raster.axis_group(a) == "channel"
        ]
        if len(space) > 2:
            # The image is a slice of a volume. Keep the spatial axes that are
            # not singletons and, if fewer than two remain, add the first
            # singleton axes, keeping the original axis order.
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
        # The kept spatial axes are the columns (x) and rows (y) of the raster,
        # in order, whatever their names. For example, a sagittal (y, z) slice
        # has y along the columns.
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
        """Encode the image in a raster format.

        Parameters
        ----------
        format : str, optional
            The Pillow name of the format, such as `"PNG"`. By default, the
            image is written in the format that it was read from (JPEG for
            MPO) if Pillow can write that format, and in PNG otherwise.
        dpi : bool or float or tuple of float, optional
            The resolution recorded by PNG, JPEG, BMP and TIFF. By default, it
            is computed from the pixel size when the preferred transformation
            scales onto axes with a unit of length, and is otherwise the
            resolution read with the image. `False` records none, and a number
            or an `(x, y)` pair is recorded as is.
        **options : Any
            Options passed to `Image.save`, such as `quality=95` for JPEG. The
            ICC profile and EXIF block of `info` are added unless they are
            given.

        Raises
        ------
        WriterError
            If the data has more than two spatial axes, or a type that Pillow
            or the format does not store (such as `float64`, or `uint16` in
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
        """Write the image to an open binary file, in the `format` given or the
        one that the name of the file calls for (see `to_bytes`).
        """
        if kwargs.get("format") is None:
            kwargs["format"] = format_for_name(_to_filename(file))
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a file, in the format its extension calls for.

        The options are those of `to_bytes`. The image is encoded before the
        file is opened, so that an image that cannot be stored leaves no file
        behind.

        Raises
        ------
        WriterError
            If neither the extension nor `format` names a format, or if the
            image cannot be stored.
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
