"""Readers and writers for FreeSurfer's Linear Transform Array (LTA) format.

An LTA file stores a linear (affine) transformation together with the
coordinate systems it maps between, recorded as its type and as the
geometry of its source and destination volumes.
"""

__all__ = [
    "LTAType",
    "LTAMatrixType",
    "LTAValidity",
    "LTAStruct",
    "LTACoordinateSystem",
    "LTAVoxelSystem",
    "LTAScaledSystem",
    "LTAPhysicalSystem",
    "LTATransformation",
    "LTATransformationVoxToVox",
    "LTATransformationPhysToPhys",
    "LTATransformationRASToRAS",
]

from ._enums import LTAMatrixType, LTAType, LTAValidity
from ._struct import LTAStruct
from ._systems import (
    LTACoordinateSystem,
    LTAPhysicalSystem,
    LTAScaledSystem,
    LTAVoxelSystem,
)
from ._xforms import (
    LTATransformation,
    LTATransformationPhysToPhys,
    LTATransformationRASToRAS,
    LTATransformationVoxToVox,
)
