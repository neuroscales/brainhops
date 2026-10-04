# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops._core import dependencies as deps
from brainhops._core import path
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.mgh import _MRI_PARAMS, MghParser, _MGHHeader, _MGHImage
from brainhops.io.base.nifti import (
    _scale_spatial,
    _unit_scale,
    _voxel_to_ras,
)
from brainhops.io.base.parsers import WriterError
from brainhops.io.images.base import WritableFileBasedImage

if tx.TYPE_CHECKING:
    from nibabel.freesurfer import mghformat as _mgh

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

# The voxel types MGH stores, by numpy dtype.
_MGH_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
}


@register_format
class MghImage(MghParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a FreeSurfer MGH or MGZ file.

    The voxels are stored x fastest (F order), so `data` has shape
    `(x, y, z)` or, for a multi-frame volume, `(x, y, z, frames)`, with
    the frames read as a time axis.

    The transformations are, in order:

    1. a [`Scaling`][brainhops.datamodel.transformations.Scaling] from
       voxels to the scaled voxel space `"physical"`: the voxel size in
       mm, and, for a multi-frame volume, the repetition time in ms (or 1,
       with no unit, when the footer records none);
    2. the voxel-to-tkr (surface) RAS affine, whose output is named
       `"tkr"` (`header.get_vox2ras_tkr()`);
    3. the voxel-to-scanner RAS affine, whose output is named
       `"scanner"` (`header.get_vox2ras()`). It is the last one, so it is
       the preferred transformation.

    FreeSurfer-specific header content -- the MRI acquisition parameters
    of the footer, the raw `goodRASFlag` and the trailing tags -- is kept
    on the object ([`mri_params`][brainhops.io.base.mgh.MghParser.
    mri_params], the private `_good_ras` and
    [`tags`][brainhops.io.base.mgh.MghParser.tags]) and written back.

    !!! note "`goodRASFlag`"
        When the header's `goodRASFlag` is not positive, FreeSurfer
        ignores the stored geometry and uses 1 mm voxels, coronal (LIA)
        direction cosines and a zero centre. So does this reader. A file
        written back always records its geometry, with `goodRASFlag = 1`.

    !!! note "Why the bases are in this order"
        As for [`NiftiImage`][brainhops.io.images.nifti.NiftiImage]:
        `SingleScaleImage.data` has no default, so it comes last, and
        `MghParser` leads so that its lazy `data`/`system` properties win.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".mgh",
        ".mgz",
        ".mgh.gz",
    )

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The voxel-to-world transformations recorded by the header,
        decoded on first access unless set explicitly.

        An image built from data alone has no header, so it records no
        transformation and the list is empty.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _mgh_to_transformations(self)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    def to_nibabel(self, like: tx.Any = None, **overrides) -> "_mgh.MGHImage":
        """
        Build the `nibabel` image that encodes this image.

        The image data becomes the voxels. The voxel-to-scanner RAS
        matrix is taken from the transformation whose output is named
        `"scanner"`, or else from the preferred transformation unless it
        maps to tkr RAS (which MGH cannot store, as it follows from the
        shape and the voxel size). It is converted to millimetres and
        decomposed into voxel sizes, direction cosines and a centre. An
        image with no transformation is written with 1 mm voxels and RAS
        axes. A transformation with no affine representation raises
        `UnrepresentableTransformationError`.

        The MRI parameters of the footer (`tr`, `flip_angle`, `te`, `ti`,
        `fov`) are copied from this image's header, then from `like` when
        it is given (a path to an MGH/MGZ file, a `nibabel` MGH image or
        header, or another object read from MGH), then from the keyword
        arguments. `dtype` sets the stored voxel type. Without it, the
        array's type is kept when MGH can store it (uint8, int16, int32,
        float32), and otherwise converted to the nearest one MGH can:
        booleans to uint8, other floats to float32, other integers to
        int16 or int32. Integers that int32 cannot hold raise
        `WriterError`.
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        ndim = len(getattr(data, "shape", ()))
        if not 1 <= ndim <= 4:
            raise WriterError(
                f"MGH stores volumes of up to four dimensions "
                f"(x, y, z, frames), not {ndim}."
            )

        overrides = dict(overrides)
        dtype = overrides.pop("dtype", None)
        dtype = _mgh_dtype(data, dtype)

        header = deps.nb.freesurfer.mghformat.MGHHeader()
        for source in (self.header, _like_header(like)):
            if source is None:
                continue
            for name in _MRI_PARAMS:
                header[name] = source[name]
        for name, value in overrides.items():
            header[name] = value
        header.set_data_dtype(dtype)

        vox2ras = _scanner_matrix(self.transformations)
        return deps.nb.freesurfer.mghformat.MGHImage(
            data, vox2ras, header=header
        )


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _mgh_to_transformations(image: MghParser) -> tx.List[Transformation]:
    """
    Convert the header of an MGH image to its list of transformations:
    voxel-to-physical scaling, voxel-to-tkr RAS and voxel-to-scanner RAS.
    """
    voxel_space = image.system
    axes = list(voxel_space.axes)
    tr = image.mri_params["tr"]

    # >> Physical space: the same axes, in mm (and ms for the frames,
    # when the repetition time is known -- it is never invented).
    units = {"space": "mm", "time": "ms" if tr > 0 else None}
    phys_axes = [replace(axis, unit=units.get(axis.type)) for axis in axes]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)

    # >> RAS space: the spatial axes run along R, A, S.
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


def _scanner_matrix(
    transformations: tx.Sequence[Transformation],
) -> np.ndarray:
    """
    The `(4, 4)` voxel-to-scanner RAS matrix, in mm, to store.

    The transformation named `"scanner"` wins; failing that, the
    preferred one, unless it maps to tkr RAS, in which case the last
    transformation that does not. No transformation gives the identity.
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
    if chosen is None:
        return np.eye(4)
    matrix = _voxel_to_ras(chosen)
    output = getattr(chosen, "output", None)
    return _scale_spatial(matrix, _unit_scale(output, "mm"))


def _mgh_dtype(data: tx.Any, dtype: tx.Any = None) -> np.dtype:
    """The voxel type to store an array as (see `MghImage.to_nibabel`)."""
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


def _like_header(like: tx.Any) -> "tx.Optional[_mgh.MGHHeader]":
    """Resolve a `like` template to the MGH header to copy fields from."""
    if like is None:
        return None
    if isinstance(like, _MGHHeader):
        return like
    if isinstance(like, _MGHImage):
        return like.header
    header = getattr(like, "header", None)
    if isinstance(header, _MGHHeader):
        return header
    if isinstance(like, (str, path.PathLike)):
        return MghImage.from_file(like).header
    return None
