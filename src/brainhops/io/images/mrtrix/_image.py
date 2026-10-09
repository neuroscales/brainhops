import math

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

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
from brainhops.io.images.base import ImageFormat

_INDEX = "index"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}
_SCANNER = "scanner"
"""Name of the world space into which the MRtrix transform maps."""


def _mrtrix_axes(ndim: int) -> tx.List[Axis]:
    """Return the voxel axes of an MRtrix image.

    The first three axes are the spatial axes `x`, `y` and `z`. Further
    axes, such as diffusion volumes, time or spherical harmonic
    coefficients, have no meaning defined by MRtrix, so they are named
    `dim3`, `dim4` and so on, without a type.
    """
    axes = []
    for i in range(ndim):
        if i < 3:
            axes.append(Axis(_SPATIAL[i], "space", unit=_INDEX))
        else:
            axes.append(Axis(f"dim{i}", unit=_INDEX))
    return axes


@register_format
class MrtrixImage(MrtrixParser, ImageFormat, SingleScaleImage):
    """An image stored in an MRtrix file (`.mif`, `.mif.gz` or `.mih`).

    The data are indexed `[x, y, z, ...]` in Fortran order, whatever
    `layout` the file uses. The intensity scaling is applied to `data`,
    while `dataobj` holds the stored values.

    The transformations are a [`Scaling`][] to `"physical"` by the voxel
    sizes and the preferred [`Affine`][] to scanner RAS+ in millimeters,
    `transform @ diag(vox)`. Header keys without a place in the data
    model, such as `dw_scheme`, are kept in `header.keyval` and written
    back.

    !!! note "Why the bases are in this order"
        As for `NiftiImage`, [`SingleScaleImage`][] comes last so that its
        `data` field follows the defaulted fields of the parser, and so
        that the lazy properties of this class override plain fields.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mif.gz", ".mih")

    # --- sniff --------------------------------------------------------

    @classmethod
    def _score_header(cls, header: MrtrixHeader) -> float:
        """Score a header as a plain image.

        A 4D image with three volumes may be a warp, as may a NIfTI file
        of that shape, so it is only a weak match.
        """
        if header.ndim == 4 and header.dim[3] == 3:
            return Confidence.WEAK
        return Confidence.LIKELY

    # --- data model ---------------------------------------------------

    @property
    def data(self) -> tx.Optional[tx.Any]:
        """Scaled image data, unless set explicitly."""
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
        """Voxel coordinate system, derived from the header unless set."""
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
        """Voxel-to-world transformations, decoded from the header.

        Transformations that were set explicitly are returned as they are.
        An image built from data alone has none.
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
        """Return the data and the geometry with axes placed for MRtrix.

        See [`voxel_to_ras`][] for the placement of the axes.
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
        """Build a header that encodes this image.

        The preferred transformation is split into unit direction cosines
        (`transform`) and voxel sizes (`vox`). Spatial axes are stored
        first, and a slice with further axes gets a `z` axis of size one,
        because MRtrix reads the first three axes as spatial. Non-spatial
        voxel sizes come from the transformation, with time in seconds, or
        else from the source header, from a [`Scaling`][], or are one.

        Parameters
        ----------
        layout : str or sequence of int, optional
            Strides, in the MRtrix spelling (`"-0,-1,+2"`) or as signed
            one-based integers. The default is the layout that was read,
            if it has as many axes, and otherwise `+0,+1,+2,...`.
        datatype : str or dtype, optional
            MRtrix type (`"Float32LE"`, `"UInt16BE"`, `"Bit"`) or numpy
            type. The default is the type of the data.
        scaling : (offset, scale), optional
            Intensity scaling to store. By default, none is stored.
        keyval : mapping, optional
            Header keys merged into those of the source header. A value of
            `None` removes a key.

        Returns
        -------
        MrtrixHeader
            The header.

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If there is no data, or the data are zero-dimensional.
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
    """Return the voxel sizes that the transformation gives non-spatial axes.

    The result is `None` unless every axis after the spatial ones is
    mapped. A time axis with a time unit in the world is given in seconds.

    Raises
    ------
    UnrepresentableTransformationError
        If the map shifts or reverses one of these axes, which MRtrix
        cannot store.
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
    """Return a voxel size for every axis, used beyond the third.

    The sizes come from the source header when it has as many axes, so
    that a NaN volume size survives a round trip. Otherwise they come from
    a [`Scaling`][] with one scale per axis, or default to one.
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
    """Convert an MRtrix header to transformations, in this order.

    1. voxel -> "physical": a `Scaling` by the voxel sizes;
    2. voxel -> "scanner": the `Affine` to scanner RAS+ millimetres,
       `transform @ diag(vox)` (or MRtrix's centred default).

    A non-finite voxel size, which is common on the volume axis, gives a
    scale of one.
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

    # The scanner space is always 3D, because the MRtrix transform maps
    # only the first three axes.
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
