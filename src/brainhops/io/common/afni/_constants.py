"""The constants of the AFNI dataset format."""

# dependencies
import numpy as np
import typing_extensions as tx

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

AFNI_VIEWS: tx.Tuple[str, ...] = ("orig", "acpc", "tlrc")
"""The AFNI views, indexed by their `SCENE_DATA[0]` code."""

AFNI_ORIENTATIONS: tx.Tuple[str, ...] = (
    "right-to-left",
    "left-to-right",
    "posterior-to-anterior",
    "anterior-to-posterior",
    "inferior-to-superior",
    "superior-to-inferior",
)
"""The anatomical orientation name of each `ORIENT_SPECIFIC` code."""

# DICOM axis of a code is code // 2; R2L, A2P and I2S point toward +LPS.
_ORIENT_SIGN = (1, -1, -1, 1, 1, -1)

# (DICOM axis, points toward positive LPS) -> orientation code.
_ORIENT_CODE = {
    (0, True): 0,
    (0, False): 1,
    (1, True): 3,
    (1, False): 2,
    (2, True): 4,
    (2, False): 5,
}

# AFNI stores these attributes as floats, so integer values are still
# written as a float attribute.
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
"""The `(4, 4)` matrix from DICOM (LPS) to RAS, its own inverse."""

# BRICK_TYPES code -> dtype without byte order (MRI_TYPE in mrilib.h).
# RGB (6) and RGBA (7) are not read.
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

# Suffixes AFNI tries after .BRIK, uncompressed first; only gzip and
# bzip2 can be decompressed.
_BRIK_SUFFIXES = ("", ".gz", ".bz2", ".Z", ".briz")
_READABLE_SUFFIXES = ("", ".gz", ".bz2")

_TAXIS_UNITS = {77001: "millisecond", 77002: "second", 77003: "hertz"}
"""The time units of the `TAXIS_NUMS[2]` codes."""

# Like AFNI (thd_niftiread.c), template world spaces map to the tlrc view.
_VIEW_NAMES = {
    "orig": "orig",
    "acpc": "acpc",
    "tlrc": "tlrc",
    "talairach": "tlrc",
    "mni": "tlrc",
    "template": "tlrc",
}
