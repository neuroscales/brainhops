"""The constants of the NRRD format."""

NRRD_MAGIC = b"NRRD000"
"""The first seven bytes of every NRRD file; a version digit follows."""

_MAX_HEADER_LINES = 1_000_000

# ----------------------------------------------------------------------
#   VOCABULARY
# ----------------------------------------------------------------------

_TYPES = {}
for _names, _code in (
    (("signed char", "int8", "int8_t"), "i1"),
    (("uchar", "unsigned char", "uint8", "uint8_t"), "u1"),
    (
        (
            "short",
            "short int",
            "signed short",
            "signed short int",
            "int16",
            "int16_t",
        ),
        "i2",
    ),
    (
        (
            "ushort",
            "unsigned short",
            "unsigned short int",
            "uint16",
            "uint16_t",
        ),
        "u2",
    ),
    (("int", "signed int", "int32", "int32_t"), "i4"),
    (("uint", "unsigned int", "uint32", "uint32_t"), "u4"),
    (
        (
            "longlong",
            "long long",
            "long long int",
            "signed long long",
            "signed long long int",
            "int64",
            "int64_t",
        ),
        "i8",
    ),
    (
        (
            "ulonglong",
            "unsigned long long",
            "unsigned long long int",
            "uint64",
            "uint64_t",
        ),
        "u8",
    ),
    (("float",), "f4"),
    (("double",), "f8"),
):
    for _name in _names:
        _TYPES[_name] = _code

_TYPE_NAMES = {
    "i1": "int8",
    "u1": "uint8",
    "i2": "int16",
    "u2": "uint16",
    "i4": "int32",
    "u4": "uint32",
    "i8": "int64",
    "u8": "uint64",
    "f4": "float",
    "f8": "double",
}
"""The type name written for each numpy type (the `pynrrd` spelling)."""

_ENCODINGS = {
    "raw": "raw",
    "txt": "ascii",
    "text": "ascii",
    "ascii": "ascii",
    "hex": "hex",
    "gz": "gzip",
    "gzip": "gzip",
    "bz2": "bzip2",
    "bzip2": "bzip2",
}

_ALIASES = {
    "datafile": "data file",
    "lineskip": "line skip",
    "byteskip": "byte skip",
    "centerings": "centers",
    "axismins": "axis mins",
    "axismaxs": "axis maxs",
    "oldmin": "old min",
    "oldmax": "old max",
    "sampleunits": "sample units",
    "blocksize": "block size",
}

_FIELDS = (
    "content",
    "type",
    "block size",
    "dimension",
    "space",
    "space dimension",
    "sizes",
    "space directions",
    "spacings",
    "thicknesses",
    "axis mins",
    "axis maxs",
    "centers",
    "kinds",
    "labels",
    "units",
    "space units",
    "space origin",
    "measurement frame",
    "sample units",
    "number",
    "min",
    "max",
    "old min",
    "old max",
    "endian",
    "encoding",
    "line skip",
    "byte skip",
    "data file",
)
"""Every field of the specification, in the order they are written."""

SPACES = {
    "right-anterior-superior": ("RAS", 3),
    "left-anterior-superior": ("LAS", 3),
    "left-posterior-superior": ("LPS", 3),
    "right-anterior-superior-time": ("RAST", 4),
    "left-anterior-superior-time": ("LAST", 4),
    "left-posterior-superior-time": ("LPST", 4),
    "scanner-xyz": ("scanner-xyz", 3),
    "scanner-xyz-time": ("scanner-xyz-time", 4),
    "3d-right-handed": ("3D-right-handed", 3),
    "3d-left-handed": ("3D-left-handed", 3),
    "3d-right-handed-time": ("3D-right-handed-time", 4),
    "3d-left-handed-time": ("3D-left-handed-time", 4),
}
"""Each `space` of the specification: its abbreviation and dimension."""

_SPACE_NAMES = {abbr.lower(): name for name, (abbr, _) in SPACES.items()}

_NONE = ("none", "???")


_DATA_EXTENSIONS = {
    "raw": ".raw",
    "gzip": ".raw.gz",
    "bzip2": ".raw.bz2",
    "ascii": ".txt",
    "hex": ".hex",
}
