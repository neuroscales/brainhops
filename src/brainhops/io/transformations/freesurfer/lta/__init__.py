"""Readers and writers of FreeSurfer Linear Transform Arrays (LTA).

An LTA file stores an affine transformation together with the coordinate
systems that it maps between. The file records these systems through a type
code and through the geometries of the source and destination volumes.
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

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses rather than relabels.
from . import _converters  # noqa: E402, F401  isort: skip
