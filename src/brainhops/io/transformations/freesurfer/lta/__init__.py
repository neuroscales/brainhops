"""Readers and writers of FreeSurfer Linear Transform Arrays (LTA).

An LTA file stores an affine transformation together with the coordinate
systems that it maps between. The file records these systems through a type
code and through the geometries of the source and destination volumes.

An LTA file is short and is read in one pass. Its whole content is held as
an [`LtaRaw`][] record by [`LtaMetadata`][], the metadata of the file, and
an [`LtaTransformation`][] holds that metadata. The matrix of the affine
lives in the record, so the transformation reads its matrix and its
coordinate systems from the record, and a transformation that is read and
written again without changes is written as its file stored it.
"""

__all__ = [
    "LtaType",
    "LtaMatrixType",
    "LtaValidity",
    "LtaRaw",
    "LtaMetadata",
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
from ._metadata import LtaMetadata
from ._raw import LtaRaw
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

# Importing the private converters registers them with the data model,
# so that a conversion into one of these formats, such as
# `t.to(Format)`, fails with a reason instead of building a wrong
# object.
from . import _converters  # noqa: E402, F401  isort: skip
