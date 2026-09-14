__all__ = ["base", "itk", "freesurfer"]

from . import base, freesurfer, itk

try:
    from . import nifti

    __all__ += ["nifti"]
except ImportError:
    pass

try:
    from . import spm

    __all__ += ["spm"]
except ImportError:
    pass
