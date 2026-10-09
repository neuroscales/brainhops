"""Reading and writing machinery shared by all NIfTI-based formats.

The image formats and the transformation formats stored as NIfTI files are both
built on [`NiftiReaderWriter`][].
"""

__all__ = [
    "NiftiReaderWriter",
    "NiftiUnitWarning",
]

from ._parsers import NiftiReaderWriter
from ._units import NiftiUnitWarning
