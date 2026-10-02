"""Readers and writers for FreeSurfer's Linear Transform Array (LTA) format.

An LTA file stores a linear (affine) transformation together with the
coordinate systems it maps between, recorded as its type and as the
geometry of its source and destination volumes.
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
