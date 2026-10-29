"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.freesurfer` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "code2orient",
    "FS_DEFAULT_XRAS",
    "FS_DEFAULT_YRAS",
    "FS_DEFAULT_ZRAS",
    "fs_geometry_from_vox2ras",
    "fs_phys2ras",
    "fs_vox2phys",
    "fs_vox2ras",
    "fs_vox2tkr",
    "mat2code",
    "mat2orient",
    "scanner_vox2ras",
    "tkr_to_scanner",
]

from ._geometry import (
    FS_DEFAULT_XRAS,
    FS_DEFAULT_YRAS,
    FS_DEFAULT_ZRAS,
    code2orient,
    fs_geometry_from_vox2ras,
    fs_phys2ras,
    fs_vox2phys,
    fs_vox2ras,
    fs_vox2tkr,
    mat2code,
    mat2orient,
)
from ._tkr import scanner_vox2ras, tkr_to_scanner
