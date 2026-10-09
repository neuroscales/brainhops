"""The shared NIfTI-reading and NIfTI-writing machinery behind every
NIfTI-based image and transformation format."""

__all__ = [
    "NiftiParser",
    "NiftiUnitWarning",
]

from ._parsers import NiftiParser
from ._units import NiftiUnitWarning
