__all__ = ["parsers"]

from . import parsers

try:
    import nibabel as _nb
except ImportError:
    _nb = None

if _nb:
    from . import nifti

    __all__ += ["nifti"]
