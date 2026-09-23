"""Readers and writers for ITK transformation formats."""

__all__ = [
    "ITKAffineBase",
    "ITKBlockBase",
    "ITKDisplacementBase",
    "ITKPrecision",
    "ITKStruct",
    "ITKTransform",
    "ITKTransformClass",
    "tfm",
]

from . import tfm
from ._common import (
    ITKAffineBase,
    ITKBlockBase,
    ITKDisplacementBase,
    ITKPrecision,
    ITKStruct,
    ITKTransformClass,
)
from ._xform import ITKTransform

# The h5 reader needs h5py, which is optional. It is imported only when
# h5py is available, mirroring how the transformations package imports
# its own optional-dependency submodules.
try:
    from . import h5

    __all__ += ["h5"]
except ImportError:  # h5py is optional
    pass
