"""Readers and writers of FreeSurfer Linear Transform Arrays (LTA).

An LTA file stores an affine transformation together with the coordinate
systems that it maps between. These systems are recorded as a type and as
the geometries of the source and destination volumes.
"""

__all__ = [
    "LtaType",
    "LtaMatrixType",
    "LtaValidity",
    "LtaStruct",
    "LtaCoordinateSystem",
    "LtaVoxelSystem",
    "LtaScaledSystem",
    "LtaPhysicalSystem",
    "LtaTransformation",
    "LtaTransformationVoxToVox",
    "LtaTransformationPhysToPhys",
    "LtaTransformationRASToRAS",
]

from ._enums import LtaMatrixType, LtaType, LtaValidity
from ._struct import LtaStruct
from ._systems import (
    LtaCoordinateSystem,
    LtaPhysicalSystem,
    LtaScaledSystem,
    LtaVoxelSystem,
)
from ._xforms import (
    LtaTransformation,
    LtaTransformationPhysToPhys,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
)
