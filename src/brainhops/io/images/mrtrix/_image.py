# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._geometry import Arrangement, declared_axes
from brainhops.io.common.mrtrix import MrtrixHeader, MrtrixParser
from brainhops.io.common.mrtrix._codecs import (
    default_layout,
    dtype_to_mrtrix,
    parse_layout,
)
from brainhops.io.common.mrtrix._geometry import (
    split_voxel_to_scanner,
    voxel_to_ras,
)
from brainhops.io.images.base import WritableFileBasedImage

_INDEX = "index"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}
_SCANNER = "scanner"
"""The name of the world space MRtrix's transform maps into."""


def _mrtrix_axes(ndim: int) -> tx.List[Axis]:
    """
    The axes of the voxel space of an MRtrix image.

    The first three are spatial (`x`, `y`, `z`). MRtrix gives the axes
    beyond them no meaning of their own -- the fourth holds diffusion
    volumes, time points, spherical-harmonic coefficients, ... -- so they
    are named `dim3`, `dim4`, ... and given no type.
    """
    axes = []
    for i in range(ndim):
        if i < 3:
            axes.append(Axis(_SPATIAL[i], "space", unit=_INDEX))
        else:
            axes.append(Axis(f"dim{i}", unit=_INDEX))
    return axes


@register_format
class MrtrixImage(MrtrixParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by an MRtrix image file
    (`.mif`, `.mif.gz`, or `.mih` with its data file).

    The data are indexed in the order of the header's axes
    (`[x, y, z, ...]`, F order), whatever the `layout` they are stored
    in: a negative or permuted layout is undone by a view, so the data of
    an uncompressed local file stay memory-mapped. Intensity scaling, when
    the header has one, is applied on access; `dataobj` holds the stored
    values.

    The voxel-to-world transformations are:

    1. `voxel` -> `physical`: a `Scaling` by the voxel sizes (`vox`);
    2. `voxel` -> `scanner`: the `Affine` to scanner RAS+ millimetres,
       `transform @ diag(vox)`. The header's `transform` maps voxel
       coordinates *multiplied by the voxel sizes*, not voxel indices.
       Without a `transform`, MRtrix's default is used, which centres the
       field of view on the origin; it is not the identity.

    The last one is the preferred transformation. Header keys that the
    data model has no slot for (`dw_scheme`, `command_history`, ...) are
    kept in `header.keyval` and written back.

    !!! note "Why the bases are in this order"
        As for `NiftiImage`: `SingleScaleImage` comes last so that its
        `data` field follows the defaulted fields of the parser, and the
        lazy properties of this class take precedence over the plain
        fields.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mif.gz", ".mih")

    # --- sniff --------------------------------------------------------

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """
        Score an MRtrix header as a plain image.

        Any MRtrix file can be read as an image. A 4D file whose last
        axis has three volumes may be a warp, so it is only a weak match,
        as a NIfTI file of that shape is.
        """
        if header.ndim == 4 and header.dim[3] == 3:
            return Confidence.WEAK
        return Confidence.LIKELY

    # --- data model ---------------------------------------------------

    @property
    def data(self) -> tx.Optional[tx.Any]:
        """The image data, scaled, unless set explicitly."""
        if getattr(self, "_data", None) is not None:
            return self._data
        data = self._scaled_data()
        if data is not None and data is not getattr(self, "dataobj", None):
            self._data = data
        return data

    @data.setter
    def data(self, value: tx.Optional[tx.Any]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system, derived from the header, unless
        set explicitly. `None` when there is no header."""
        if getattr(self, "_system", None) is not None:
            return self._system
        if self.header is None:
            return None
        axes = _mrtrix_axes(self.header.ndim)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The voxel-to-world transformations recorded by the header,
        decoded on access unless set explicitly.

        An image built from data alone has no header, so it records no
        transformation and the list is empty.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _mrtrix_to_transformations(self.header)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    # --- writing ------------------------------------------------------

    def _mrtrix_geometry(self) -> tx.Tuple[tx.Any, Arrangement]:
        """
        The data, and its geometry with the axes placed where MRtrix
        stores them (see [`voxel_to_ras`][brainhops.io.common.mrtrix._geometry.
        voxel_to_ras]).
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        xform = self.transformation
        ndim = len(np.shape(data))
        voxel_axes = declared_axes(getattr(xform, "input", None), ndim)
        return data, voxel_to_ras(xform, voxel_axes)

    def _mrtrix_data(self) -> tx.Any:
        data, arranged = self._mrtrix_geometry()
        if arranged.layout is not None:
            data = arranged.layout.apply(data)
        return data

    def _mrtrix_header(
        self,
        layout: tx.Optional[tx.Union[str, tx.Sequence[int]]] = None,
        datatype: tx.Optional[tx.Any] = None,
        scaling: tx.Optional[tx.Sequence[float]] = None,
        keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
        **kwargs,
    ) -> MrtrixHeader:
        """
        Build the header that encodes this image.

        The geometry always comes from the preferred transformation,
        converted to voxel-to-scanner RAS+ and split into unit direction
        cosines (`transform`) and voxel sizes (`vox`).

        The axes are placed by the types and names the voxel space of the
        preferred transformation declares: the spatial axes first (`x, y,
        z` in that order when they are so named), then time, then the
        channels, then the others, and the data is transposed to match
        (lazily, for a lazy array). A slice with other axes is given a
        `z` axis of size one, as MRtrix reads its first three axes as
        spatial. A voxel space that declares nothing is written in the
        order it has.

        The voxel sizes of the non-spatial axes are their spacings in the
        preferred transformation, when it maps them -- a time axis in
        seconds, when its unit is a time unit, as MRtrix and BIDS state
        times. MRtrix stores no origin for them, so a map that shifts one
        raises `UnrepresentableTransformationError`. When the
        transformation does not map them, they come from the header the
        image was read from, else from a `Scaling` among the
        transformations (such as the `physical` one a NIfTI reader
        builds), else are 1.

        Parameters
        ----------
        layout : str | sequence of int, optional
            The layout to store the data in, as MRtrix spells it
            (`"-0,-1,+2"`) or as signed one-based strides. Defaults to the
            layout the image was read with, if it has as many axes, else
            to `+0,+1,+2,...`.
        datatype : str | dtype, optional
            The MRtrix data type (`"Float32LE"`, `"UInt16BE"`, `"Bit"`)
            or a numpy one. Defaults to the data's own type.
        scaling : (offset, scale), optional
            Intensity scaling to store; the data are divided accordingly.
            Defaults to none.
        keyval : mapping, optional
            Extra header keys, merged into those read from the source
            header; a value of `None` removes a key. A value with several
            lines is written as one line per entry.
        """
        if kwargs:
            raise TypeError(
                f"Unknown MRtrix writer option(s): {', '.join(kwargs)}"
            )
        data, arranged = self._mrtrix_geometry()
        if arranged.layout is not None:
            data = arranged.layout.apply(data)
        shape = tuple(int(d) for d in np.shape(data))
        ndim = len(shape)
        if ndim == 0:
            raise WriterError("MRtrix cannot store a zero-dimensional array.")
        source = self.header

        # --- geometry -------------------------------------------------
        transform, spatial_vox = split_voxel_to_scanner(arranged.matrix)
        vox = list(spatial_vox[: min(3, ndim)])
        extra = _mapped_vox(arranged, ndim)
        if extra is None:
            extra = _extra_vox(self.transformations, source, ndim)
        vox += extra[len(vox) :]

        # --- storage --------------------------------------------------
        if layout is None:
            if source is not None and len(source.layout) == ndim:
                strides = source.layout
            else:
                strides = default_layout(ndim)
        elif isinstance(layout, str):
            strides = parse_layout(layout, ndim)
        else:
            strides = parse_layout(
                ",".join(
                    ("+" if s > 0 else "-") + str(abs(int(s)) - 1)
                    for s in layout
                ),
                ndim,
            )
        if datatype is None:
            datatype = getattr(data, "dtype", np.float32)
        datatype = dtype_to_mrtrix(datatype)

        # --- free-form keys -------------------------------------------
        merged = dict(source.keyval) if source is not None else {}
        for key, value in (keyval or {}).items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value

        return MrtrixHeader(
            dim=shape,
            vox=vox,
            layout=strides,
            datatype=datatype,
            transform=transform,
            scaling=scaling,
            keyval=merged,
        )


def _mapped_vox(
    arranged: Arrangement, ndim: int
) -> tx.Optional[tx.List[float]]:
    """
    The voxel sizes of the axes after the spatial ones, from their
    spacings in the voxel-to-world map, or `None` when it does not map
    them all.

    A time axis whose world unit is a time unit is given in seconds. A
    map that shifts one of these axes, or reverses it, has no MRtrix form.
    """
    others = list(arranged.others)
    if ndim <= 3 or len(others) != ndim - 3:
        return None
    world = list(getattr(arranged.world, "axes", None) or [])
    vox = [math.nan] * 3
    for k, (scale, offset) in enumerate(others):
        if offset != 0:
            raise UnrepresentableTransformationError(
                f"MRtrix stores no origin for the axes after the spatial "
                f"ones, so a map that shifts axis {3 + k} ({offset}) cannot "
                f"be written."
            )
        if scale < 0:
            raise UnrepresentableTransformationError(
                f"MRtrix stores the voxel size of axis {3 + k} as a "
                f"positive number, so a map that reverses it ({scale}) "
                f"cannot be written."
            )
        unit = None
        if len(world) > 3 + k:
            unit = getattr(world[3 + k], "unit", None)
        if is_physicalunit(unit) and is_timeunit(unit):
            scale = scale * float(unit.scale)
        vox.append(float(scale))
    return vox


def _extra_vox(
    transformations: tx.Sequence[Transformation],
    source: tx.Optional[MrtrixHeader],
    ndim: int,
) -> tx.List[float]:
    """
    The voxel size of every axis, used for the axes beyond the third.

    The header the image was read from wins when it has as many axes, so
    a `nan` volume size is written back as it was read. Otherwise a
    `Scaling` with one scale per axis among the transformations (such as
    the `physical` one of a NIfTI image) provides them. Otherwise they
    are 1.
    """
    if source is not None and source.ndim == ndim:
        vox = list(source.vox[:ndim])
        return vox + [math.nan] * (ndim - len(vox))
    for xform in reversed(list(transformations or [])):
        if isinstance(xform, Scaling):
            scale = np.ravel(np.asarray(getattr(xform, "scale", []), float))
            if scale.size == ndim:
                return [float(v) for v in scale]
    return [1.0] * ndim


def _mrtrix_to_transformations(
    header: MrtrixHeader,
) -> tx.List[Transformation]:
    """
    Convert an MRtrix header to a list of transformations.

    1. voxel -> "physical": a `Scaling` by the voxel sizes;
    2. voxel -> "scanner": the `Affine` to scanner RAS+ millimetres,
       `transform @ diag(vox)` (or MRtrix's centred default).

    A voxel size that is not finite (`nan` is common on the volume axis)
    is a scale of 1 on the physical axis.
    """
    ndim = header.ndim
    voxel_axes = _mrtrix_axes(ndim)
    voxel_space = CoordinateSystem(name="voxel", axes=voxel_axes, order="F")

    phys_axes = [
        replace(axis, unit=_MM if axis.type == "space" else None)
        for axis in voxel_axes
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

    spacing = header.spacing
    scale = [v if math.isfinite(v) and v > 0 else 1.0 for v in spacing]
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=scale)

    # The scanner space is always three-dimensional RAS+ mm, whatever the
    # number of axes: MRtrix's transform maps the first three.
    ras_axes = [
        Axis(
            name,
            "space",
            unit=_MM,
            orientation=Orientation(
                type="anatomical", value=_ORIENTATION[name]
            ),
        )
        for name in _SPATIAL
    ]
    nspatial = min(3, ndim)
    world = CoordinateSystem(name=_SCANNER, axes=ras_axes)
    vox2ras = header.voxel_to_scanner()
    columns = list(range(nspatial)) + [3]
    matrix = vox2ras[:3][:, columns]
    vox2scanner = Affine(input=voxel_space, output=world, matrix=matrix)
    return [vox2phys, vox2scanner]
