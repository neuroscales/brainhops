"""The constants of the MRtrix image format."""

MRTRIX_MAGIC = "mrtrix image"
"""The line every MRtrix header starts with."""

_MAGIC_BYTES = MRTRIX_MAGIC.encode("ascii")

_END = "END"
"""The line every MRtrix header ends with."""

_RESERVED = ("dim", "vox", "layout", "datatype", "transform", "scaling")
"""The keys MRtrix decodes itself, matched case-insensitively."""

_FILE = "file"
"""The key that says where the voxel data are."""

_MAX_HEADER_LINES = 1_000_000
"""A bound on the header length, so a stray binary file is not read whole
while looking for an `END` that never comes."""


# ----------------------------------------------------------------------
#   DATA TYPES
# ----------------------------------------------------------------------

_DTYPES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "int64": "i8",
    "uint64": "u8",
    "float32": "f4",
    "float64": "f8",
    "cfloat32": "c8",
    "cfloat64": "c16",
}
"""The numpy type of each MRtrix data type, without its byte order."""

_NAMES = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
    "cfloat32": "CFloat32",
    "cfloat64": "CFloat64",
}
"""The spelling MRtrix writes each data type with."""

_BIT = "bit"
