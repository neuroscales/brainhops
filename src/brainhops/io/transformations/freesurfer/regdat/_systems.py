"""The tkregister RAS space of a volume, which a `register.dat` maps."""

__all__ = ["RegisterDatTkrSystem"]

import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems

from ..lta import LtaRaw

_MM = "mm"

_3SpatialAxes = tx.Tuple[
    _axes.SpaceAxis,
    _axes.SpaceAxis,
    _axes.SpaceAxis,
]


class RegisterDatTkrSystem(
    _systems.SpatialCoordinateSystem3D, _systems.MillimetricCoordinateSystem
):
    """The tkregister RAS space of a volume, in which `register.dat` works.

    Each volume has its own tkregister RAS space, which FreeSurfer also
    calls tkr RAS or surface RAS (`mri_info --vox2ras-tkr`). Its axes point
    towards the right, anterior and superior directions in millimetres,
    as scanner RAS does, but its voxel-to-RAS matrix uses the conformed
    LIA directions and puts the origin at the centre of the volume
    (`MRIxfmCRS2XYZtkreg` in `utils/mri.cpp` of FreeSurfer, commit
    4f8728e). The space therefore depends only on the shape and the voxel
    size of the volume, and it differs from scanner RAS by a rotation and
    a translation that the geometry of the volume gives.

    The space is not [`RASmm`][brainhops.datamodel.systems.RASmm], which
    is scanner RAS, so that a format that stores scanner RAS refuses to
    write a transformation between tkregister spaces instead of writing it
    as a transformation between scanner spaces.

    A system built from the geometry of a volume keeps that geometry in
    `raw`, as the systems of an LTA file do. A system without a geometry
    stands for the tkregister space of a volume that is not known.
    """

    name: ... = "tkr"
    axes: _3SpatialAxes = (
        _axes.R("x", unit=_MM),
        _axes.A("y", unit=_MM),
        _axes.S("z", unit=_MM),
    )
    raw: KwOnly[tx.Optional[LtaRaw.VolumeInfo]] = None
    """The geometry of the volume, or `None` when it is not known."""

    @classmethod
    def from_raw(
        cls, raw: tx.Optional[LtaRaw.VolumeInfo], name: str = "tkr"
    ) -> tx.Self:
        """Return the tkregister space of the volume that `raw` describes.

        Parameters
        ----------
        raw : LtaRaw.VolumeInfo, optional
            The geometry of the volume, which the system keeps, or `None`
            for a volume whose geometry is not known.
        name : str, default="tkr"
            The name of the system when the geometry names no file.

        Returns
        -------
        Self
            The system, named after the file of the volume, or `name`.
        """
        if raw is not None and raw.filename:
            name = raw.filename
        return cls(name=name, raw=raw)
