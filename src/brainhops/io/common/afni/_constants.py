"""The constants of the AFNI dataset format."""

# dependencies
import numpy as np
import typing_extensions as tx

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

AFNI_VIEWS: tx.Tuple[str, ...] = ("orig", "acpc", "tlrc")
"""The views, by their `SCENE_DATA[0]` code."""

AFNI_ORIENTATIONS: tx.Tuple[str, ...] = (
    "right-to-left",
    "left-to-right",
    "posterior-to-anterior",
    "anterior-to-posterior",
    "inferior-to-superior",
    "superior-to-inferior",
)
"""The anatomical orientation of each `ORIENT_SPECIFIC` code."""

# The DICOM axis an orientation code runs along is `code // 2`, and its
# direction is positive in DICOM (LPS) for R2L, A2P and I2S
# (`ORIENT_sign = "+--++-"` in `3ddata.h`).
_ORIENT_SIGN = (1, -1, -1, 1, 1, -1)

# The orientation code of a voxel axis that runs along a DICOM axis
# (0 = x, 1 = y, 2 = z), in the positive DICOM direction (towards L, P
# or S) or not.
_ORIENT_CODE = {
    (0, True): 0,  # R2L
    (0, False): 1,  # L2R
    (1, True): 3,  # A2P
    (1, False): 2,  # P2A
    (2, True): 4,  # I2S
    (2, False): 5,  # S2I
}

# The attributes AFNI holds as floats, so that one given with integer
# values is still written as a float attribute.
_FLOAT_ATTRIBUTES = frozenset(
    {
        "ORIGIN",
        "DELTA",
        "IJK_TO_DICOM",
        "IJK_TO_DICOM_REAL",
        "BRICK_FLOAT_FACS",
        "BRICK_STATS",
        "BRICK_STATAUX",
        "STAT_AUX",
        "TAXIS_FLOATS",
        "TAXIS_OFFSETS",
        "MARKS_XYZ",
        "TAGSET_FLOATS",
        "TAGALIGN_MATVEC",
        "WARP_DATA",
        "VOLREG_MATVEC_000000",
        "VOLREG_CENTER_OLD",
        "VOLREG_CENTER_BASE",
    }
)

DICOM_TO_RAS: np.ndarray = np.diag([-1.0, -1.0, 1.0, 1.0])
"""The `(4, 4)` matrix from AFNI's DICOM (LPS) coordinates to RAS. It is
its own inverse."""

# The data types of `BRICK_TYPES` (`MRI_TYPE` in `mrilib.h`), without
# byte order. RGB (6) and RGBA (7) are not read.
_BRICK_DTYPES = {
    0: np.dtype(np.uint8),
    1: np.dtype(np.int16),
    2: np.dtype(np.int32),
    3: np.dtype(np.float32),
    4: np.dtype(np.float64),
    5: np.dtype(np.complex64),
}
_BRICK_CODES = {dtype: code for code, dtype in _BRICK_DTYPES.items()}
_BRICK_NAMES = {
    "byte": 0,
    "short": 1,
    "int": 2,
    "float": 3,
    "double": 4,
    "complex": 5,
}

_BYTEORDERS = {"LSB_FIRST": "<", "MSB_FIRST": ">"}

# The suffixes AFNI tries after `.BRIK` (`COMPRESS_suffix` in
# `thd_compress.h`), the uncompressed file first. Only gzip and bzip2
# can be decompressed here.
_BRIK_SUFFIXES = ("", ".gz", ".bz2", ".Z", ".briz")
_READABLE_SUFFIXES = ("", ".gz", ".bz2")

_TAXIS_UNITS = {77001: "millisecond", 77002: "second", 77003: "hertz"}
"""The time units of `TAXIS_NUMS[2]`."""

# Names a world space may carry, and the view they mean. NIfTI's
# template references go to `tlrc`, as AFNI reads them
# (`thd_niftiread.c`).
_VIEW_NAMES = {
    "orig": "orig",
    "acpc": "acpc",
    "tlrc": "tlrc",
    "talairach": "tlrc",
    "mni": "tlrc",
    "template": "tlrc",
}
