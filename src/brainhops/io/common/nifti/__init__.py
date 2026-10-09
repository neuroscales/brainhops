"""Reading and writing machinery shared by all NIfTI-based formats.

The image formats and the transformation formats stored as NIfTI files are both
built on [`NiftiParser`][].
"""

__all__ = [
    "NiftiParser",
    "NiftiUnitWarning",
]

from ._parsers import NiftiParser
from ._units import NiftiUnitWarning
