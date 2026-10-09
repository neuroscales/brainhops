import numpy as np
import typing_extensions as tx
from bagof.magic import replace
from nibabel.freesurfer import mghformat as _mgh

from brainhops._core import path
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._geometry import (
    Arrangement,
    arrange_voxel_to_ras,
    declared_axes,
)
from brainhops.io.common.mgh import MghReaderWriter
from brainhops.io.common.mgh._constants import _MRI_PARAMS
from brainhops.io.common.nifti._geometry import _scale_spatial, _unit_scale
from brainhops.io.images.base import ImageFormat

_SCANNER = "scanner"
"""Name of the scanner RAS space, the preferred world space."""

_TKR = "tkr"
"""Name of the tkr (surface) RAS space."""

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_MGH_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
}


@register_format
class MghImage(MghReaderWriter, ImageFormat, SingleScaleImage):
    """An image stored in a FreeSurfer MGH or MGZ file.

    The voxels are in Fortran order, and the data are indexed `(x, y, z)`
    or `(x, y, z, frames)`, where the frames form a time axis.

    The transformations are, in order, a [`Scaling`][] to `"physical"` by
    the voxel size in millimeters and the TR in milliseconds, an
    [`Affine`][] to `"tkr"` (`header.get_vox2ras_tkr()`), and the
    preferred [`Affine`][] to `"scanner"` (`header.get_vox2ras()`). The
    footer parameters
    ([`mri_params`][brainhops.io.common.mgh.MghReaderWriter.mri_params]),
    the raw `goodRASFlag` and the trailing tags
    ([`tags`][brainhops.io.common.mgh.MghReaderWriter.tags]) are kept on the
    object and written back.

    !!! note "`goodRASFlag`"
        When the flag is not positive, FreeSurfer and the reader assume 1
        mm voxels, LIA direction cosines and a zero centre. Written files
        always set the flag.

    !!! note "Why the bases are in this order"
        As for [`NiftiImage`][brainhops.io.images.nifti.NiftiImage],
        [`SingleScaleImage`][] comes last because its `data` field has no
        default, and [`MghReaderWriter`][] comes first so that its lazy
        properties take precedence.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".mgh",
        ".mgz",
        ".mgh.gz",
    )

    @property
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        Transformations that were set explicitly are returned as they are.
        An image built from data alone has no header and no
        transformations.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _mgh_to_transformations(self)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    def to_nibabel(self, like: tx.Any = None, **overrides) -> _mgh.MGHImage:
        """Build the nibabel MGH image that encodes this image.

        The voxel-to-scanner matrix comes from the `"scanner"`
        transformation, or else from the preferred one unless it maps to
        tkr RAS, which MGH cannot store. Without a transformation, the
        voxels are 1 mm along the RAS axes. The data are transposed to put
        the spatial axes first and at most one frame axis after them (see
        [`plan_axes`][brainhops.io.common._geometry.plan_axes]). The TR of
        a time axis is stored in milliseconds.

        Parameters
        ----------
        like : path, nibabel MGH image or header, or MGH object, optional
            Template whose MRI parameters override those of this image. A
            TR stated by the transformation overrides both.
        **overrides : Any
            Header fields set last. `dtype` sets the stored voxel type,
            which by default is the type of the data if MGH can store it
            (uint8, int16, int32, float32), or else the nearest one.

        Returns
        -------
        nibabel.freesurfer.mghformat.MGHImage
            The encoded image.

        Raises
        ------
        WriterError
            If there is no data, the data have more than four dimensions,
            or the voxel type cannot be stored.
        UnrepresentableTransformationError
            If the transformation has no affine form, has a second
            non-spatial axis, or shifts or scales frames in a way that MGH
            cannot store.
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        vox2ras, layout, tr = _scanner_geometry(self.transformations, data)
        if layout is not None:
            data = layout.apply(data)
        ndim = len(getattr(data, "shape", ()))
        if not 1 <= ndim <= 4:
            raise WriterError(
                f"MGH stores volumes of up to four dimensions "
                f"(x, y, z, frames), not {ndim}."
            )

        overrides = dict(overrides)
        dtype = overrides.pop("dtype", None)
        dtype = _mgh_dtype(data, dtype)

        header = _mgh.MGHHeader()
        for source in (self.header, _like_header(like)):
            if source is None:
                continue
            for name in _MRI_PARAMS:
                header[name] = source[name]
        if tr is not None:
            header["tr"] = tr
        for name, value in overrides.items():
            header[name] = value
        header.set_data_dtype(dtype)

        return _mgh.MGHImage(data, vox2ras, header=header)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _mgh_to_transformations(image: MghReaderWriter) -> tx.List[Transformation]:
    """Return the scaling, tkr RAS and scanner RAS transformations."""
    voxel_space = image.system
    axes = list(voxel_space.axes)
    tr = image.mri_params["tr"]

    # Frames are in milliseconds only when the TR is known; a unit is
    # never invented.
    units = {"space": "mm", "time": "ms" if tr > 0 else None}
    phys_axes = [replace(axis, unit=units.get(axis.type)) for axis in axes]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)

    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=_RAS_ORIENTATION[axis.name]
            ),
        )
        if axis.name in _RAS_ORIENTATION
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    zooms = list(image.voxel_size)
    if len(axes) > 3:
        zooms.append(tr if tr > 0 else 1.0)
    vox2phys = Scaling(
        input=voxel_space, output=phys_space, scale=zooms[: len(axes)]
    )

    def _affine(matrix: np.ndarray, name: str) -> Affine:
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=name),
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    return [
        vox2phys,
        _affine(image.vox2tkr, _TKR),
        _affine(image.vox2ras, _SCANNER),
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


_MGH_POLICY = dict(fill_space=True, max_nonspatial=1)
"""Placement of axes in MGH files: three spatial axes, then frames."""


def _scanner_transformation(
    transformations: tx.Sequence[Transformation],
) -> tx.Optional[Transformation]:
    """Pick the transformation that gives the voxel-to-scanner matrix.

    A transformation named `"scanner"` wins. Otherwise the last
    transformation that does not map to tkr RAS is used, or `None` if
    there is none.
    """
    transformations = list(transformations or [])
    chosen = None
    for xform in transformations:
        if getattr(getattr(xform, "output", None), "name", None) == _SCANNER:
            chosen = xform
    if chosen is None:
        for xform in reversed(transformations):
            name = getattr(getattr(xform, "output", None), "name", None)
            if name != _TKR:
                chosen = xform
                break
    return chosen


def _scanner_geometry(
    transformations: tx.Sequence[Transformation], data: tx.Any
) -> tx.Tuple[np.ndarray, tx.Any, tx.Optional[float]]:
    """Return the scanner matrix, the axis layout and the TR to store.

    The matrix is in millimeters and the TR in milliseconds (see
    [`MghImage.to_nibabel`][]). Without a transformation, the result is
    the identity, no layout and no TR.
    """
    chosen = _scanner_transformation(transformations)
    if chosen is None:
        return np.eye(4), None, None
    ndim = len(getattr(data, "shape", ()) or ())
    voxel_axes = declared_axes(getattr(chosen, "input", None), ndim)
    arranged = arrange_voxel_to_ras(
        chosen, voxel_axes, "MGH", "scanner", **_MGH_POLICY
    )
    output = getattr(chosen, "output", None)
    matrix = _scale_spatial(arranged.matrix, _unit_scale(output, "mm"))
    return matrix, arranged.layout, _frame_tr(arranged)


def _frame_tr(arranged: Arrangement) -> tx.Optional[float]:
    """Return the TR in milliseconds of the frame axis, or `None`.

    A time axis that still counts frames, with a scale of one and no time
    unit, states no TR. Frames that are not time can be neither scaled
    nor shifted.
    """
    if not arranged.others:
        return None
    scale, offset = arranged.others[0]
    groups = [
        g[3]
        for g in (arranged.voxel_groups, arranged.world_groups)
        if g is not None and len(g) > 3
    ]
    if any(g != "time" for g in groups):
        if (scale, offset) != (1.0, 0.0):
            raise UnrepresentableTransformationError(
                "MGH stores no spacing or origin for frames that are not "
                f"time, so a map that scales or shifts them ({scale}, "
                f"{offset}) cannot be written."
            )
        return None
    if offset != 0:
        raise UnrepresentableTransformationError(
            f"MGH stores no origin for the time axis, so a map that "
            f"shifts it ({offset}) cannot be written."
        )
    if scale <= 0:
        raise UnrepresentableTransformationError(
            f"MGH stores the repetition time as a positive number, so a "
            f"map that scales time by {scale} cannot be written."
        )
    axes = list(getattr(arranged.world, "axes", None) or [])
    unit = getattr(axes[3], "unit", None) if len(axes) > 3 else None
    if is_physicalunit(unit) and is_timeunit(unit):
        return float(scale) * float(unit.scale) * 1000.0
    if scale == 1.0:
        # Still a frame index, as read from a file without a TR.
        return None
    return float(scale)


def _mgh_dtype(data: tx.Any, dtype: tx.Any = None) -> np.dtype:
    """Return the voxel type to store (see [`MghImage.to_nibabel`][])."""
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype not in _MGH_DTYPES:
            raise WriterError(
                f"MGH cannot store voxels of type {dtype}; it stores uint8, "
                f"int16, int32 and float32."
            )
        return dtype
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype in _MGH_DTYPES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype.kind == "f":
        return np.dtype(np.float32)
    if dtype.kind in "iu":
        if np.can_cast(dtype, np.int16):
            return np.dtype(np.int16)
        if np.can_cast(dtype, np.int32):
            return np.dtype(np.int32)
        info = np.iinfo(np.int32)
        values = np.asarray(data)
        if values.size == 0 or (
            values.min() >= info.min and values.max() <= info.max
        ):
            return np.dtype(np.int32)
        raise WriterError(
            f"MGH stores integers in at most 32 bits, and this {dtype} "
            f"array holds values outside the int32 range."
        )
    raise WriterError(
        f"MGH cannot store voxels of type {dtype}; it stores uint8, int16, "
        f"int32 and float32."
    )


def _like_header(like: tx.Any) -> tx.Optional[_mgh.MGHHeader]:
    """Resolve a `like` template to the MGH header to copy fields from."""
    if like is None:
        return None
    if isinstance(like, _mgh.MGHHeader):
        return like
    if isinstance(like, _mgh.MGHImage):
        return like.header
    header = getattr(like, "header", None)
    if isinstance(header, _mgh.MGHHeader):
        return header
    if isinstance(like, (str, path.PathLike)):
        return MghImage.from_file(like).header
    return None
