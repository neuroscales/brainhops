"""The tkregister RAS space of a volume, shared by the FreeSurfer formats."""

__all__ = [
    "FreesurferTkrSystem",
    "scanner_vox2ras",
    "tkr_to_scanner",
]

import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, KwOnly

from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems
from brainhops.datamodel._internal_api import get_axes
from brainhops.datamodel.geometry import Geometry
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.common._internal_api import (
    RAS_FROM_ORIENTATION,
    ras_conversion,
    reduce_to_affine,
)

from ._geometry import fs_vox2tkr

_MM = "mm"


class FreesurferTkrSystem(
    _systems.SpatialCoordinateSystem3D, _systems.MillimetricCoordinateSystem
):
    """The tkregister RAS space of a volume, in which surfaces live.

    FreeSurfer gives each volume a second RAS space beside scanner RAS,
    which it calls tkregister RAS, tkr RAS or surface RAS
    (`mri_info --vox2ras-tkr`). Surfaces are stored in this space, an MGH
    image maps its voxels to it, and a `register.dat` file maps the
    tkregister space of one volume to that of another. Its axes point
    towards the right, anterior and superior directions in millimetres,
    as those of scanner RAS do, but its voxel-to-RAS matrix uses the
    conformed LIA directions and puts the origin at the centre of the
    volume (`MRIxfmCRS2XYZtkreg` in `utils/mri.cpp` of FreeSurfer,
    commit 4f8728e). The space therefore depends only on the shape and
    the voxel size of the volume, and it differs from scanner RAS by a
    rotation and a translation that the geometry of the volume gives.

    The space is not [`RASmm`][brainhops.datamodel.systems.RASmm], which
    stands for scanner RAS. A format that stores scanner RAS and checks
    the class of its systems, such as LTA, therefore refuses to write an
    affine from or to a tkregister space as an affine between scanner
    spaces.

    A system may hold the geometry of its volume, a
    [`Geometry`][brainhops.datamodel.geometry.Geometry], which places the
    volume in scanner RAS. A system without a geometry stands for the
    tkregister space of a volume that is not known. The geometry must be
    that of a 3-D volume whose voxel-to-world transformation is an
    affine into an anatomical space in millimetres, such as `RASmm` or
    `LPSmm`, which FreeSurfer reads as scanner RAS. Any other geometry
    raises a `ValueError` when the system is built.
    """

    name: ... = "tkr"
    axes: ... = Factory(
        lambda: (
            _axes.R("x", unit=_MM),
            _axes.A("y", unit=_MM),
            _axes.S("z", unit=_MM),
        )
    )
    geometry: KwOnly[tx.Optional[Geometry]] = None
    """The geometry of the volume, or `None` when it is not known."""

    DISTINCT_ORIGIN: tx.ClassVar[bool] = True
    """Whether the space differs from other spaces with the same axes.

    A tkregister space has the axes of scanner RAS but another origin,
    so a conversion into a file format never bridges it to a system of
    another class by its axes alone (see
    [`affine_between`][brainhops.io.transformations.base._conversions.affine_between]).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.geometry is not None:
            scanner_vox2ras(self.geometry, "the volume of a tkregister space")


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def scanner_vox2ras(
    geometry: Geometry, what: str = "the volume"
) -> tx.Tuple[tx.Tuple[int, int, int], np.ndarray]:
    """Return the shape and the voxel-to-scanner-RAS matrix of a geometry.

    FreeSurfer places a volume in tkregister RAS from its shape and its
    voxel size, and in scanner RAS from its voxel-to-RAS matrix. A
    geometry gives both only when it describes a 3-D volume whose
    voxel-to-world transformation is an affine, or reduces to one, and
    whose world space is anatomical and in millimetres, such as `RASmm`
    or `LPSmm`. A world in LPS or RSA is converted to RAS. A world that
    is itself a tkregister space is refused, because it is not scanner
    RAS.

    Parameters
    ----------
    geometry : Geometry
        The geometry of the volume.
    what : str, default="the volume"
        The role of the volume, for error messages.

    Returns
    -------
    shape : tuple of int
        The three dimensions of the volume.
    vox2ras : np.ndarray
        The `(4, 4)` voxel-to-scanner-RAS matrix.

    Raises
    ------
    TypeError
        If `geometry` is not a `Geometry`.
    ValueError
        If the geometry is not that of a 3-D volume placed by an affine
        in an anatomical space in millimetres.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(
            f"The geometry of {what} is a Geometry, such as the `geometry` "
            f"of an image, not a {type(geometry).__name__}."
        )
    shape = tuple(int(n) for n in geometry.shape or ())
    if len(shape) != 3:
        raise ValueError(
            f"FreeSurfer places a 3-D volume in tkregister RAS, but the "
            f"geometry of {what} has shape {shape}. Give the geometry of "
            f"its three spatial axes, a Geometry whose shape has three "
            f"dimensions and whose transformation is the 3-D "
            f"voxel-to-RAS affine."
        )
    xform = geometry.transformation
    for side in ("input", "output"):
        for axis in getattr(getattr(xform, side), "axes", None) or ():
            kind = getattr(axis, "type", None)
            if axis is not ... and kind not in (None, "space"):
                raise ValueError(
                    f"FreeSurfer places a volume in space, but the "
                    f"voxel-to-world transformation of {what} has a "
                    f"{kind} axis in its {side}."
                )
    world = xform.output
    if isinstance(world, FreesurferTkrSystem):
        raise ValueError(
            f"The geometry of {what} places it in tkregister RAS, but "
            f"FreeSurfer needs its voxel-to-scanner-RAS matrix. Give the "
            f"geometry whose world is scanner RAS."
        )
    if not _anatomical_mm(world):
        raise ValueError(
            f"The world space of the geometry of {what} is not known to "
            f"be anatomical and in millimetres, so it cannot be read as "
            f"scanner RAS. Set the output of its transformation to RASmm "
            f"(or LPSmm) when that is what it maps to."
        )
    try:
        affine = reduce_to_affine(xform, "FreeSurfer", "scanner RAS")
    except UnrepresentableTransformationError as error:
        raise ValueError(
            f"The voxel-to-world transformation of {what} is a "
            f"{type(xform).__name__} without an affine form, and "
            f"FreeSurfer places a volume in scanner RAS by an affine."
        ) from error
    matrix = np.asarray(affine.matrix, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(
            f"The voxel-to-world affine of {what} has shape "
            f"{matrix.shape}, but FreeSurfer places a 3-D volume by a "
            f"(3, 4) affine."
        )
    vox2world = np.concatenate([matrix, [[0.0, 0.0, 0.0, 1.0]]])
    vox2ras = ras_conversion(world) @ vox2world
    if not np.all(np.isfinite(vox2ras)) or np.linalg.det(vox2ras) == 0:
        raise ValueError(
            f"The voxel-to-world affine of {what} is not invertible, so "
            f"it does not place a volume in space."
        )
    return shape, vox2ras


def tkr_to_scanner(geometry: Geometry, what: str = "the volume") -> np.ndarray:
    """Return the matrix from the tkregister RAS to the scanner RAS of a
    volume.

    The matrix is `vox2ras @ inv(vox2tkr)`, where `vox2ras` is the
    voxel-to-scanner-RAS matrix of the geometry and `vox2tkr` is the
    matrix of `MRIxfmCRS2XYZtkreg` in `utils/mri.cpp` of FreeSurfer
    (commit 4f8728e), built from the shape and the voxel size. The voxel
    size is the length of each column of `vox2ras`, which FreeSurfer
    stores as `xsize`, `ysize` and `zsize`.

    Parameters
    ----------
    geometry : Geometry
        The geometry of the volume, as [`scanner_vox2ras`][] accepts it.
    what : str, default="the volume"
        The role of the volume, for error messages.

    Returns
    -------
    np.ndarray
        The `(4, 4)` matrix.

    Raises
    ------
    TypeError
        If `geometry` is not a `Geometry`.
    ValueError
        If the geometry does not place a 3-D volume in scanner RAS.
    """
    shape, vox2ras = scanner_vox2ras(geometry, what)
    voxelsize = np.linalg.norm(vox2ras[:3, :3], axis=0)
    return vox2ras @ np.linalg.inv(fs_vox2tkr(shape, voxelsize))


def _anatomical_mm(world: tx.Optional[_systems.CoordinateSystem]) -> bool:
    # A world whose three axes have an anatomical orientation and are in
    # millimetres, which `ras_conversion` turns into RAS.
    if not isinstance(world, _systems.MillimetricCoordinateSystem):
        return False
    axes = list(get_axes(world))
    if len(axes) != 3:
        return False
    for axis in axes:
        orientation = getattr(axis, "orientation", None)
        if getattr(orientation, "value", None) not in RAS_FROM_ORIENTATION:
            return False
    return True
