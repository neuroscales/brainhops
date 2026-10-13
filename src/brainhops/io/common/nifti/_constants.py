"""The constants of the NIfTI format: axes, intents, xform codes."""

# internals
from brainhops.datamodel.axes import Axis

# The array axes are axes of voxel space, so they count samples. A reader that
# builds a physical space gives the axes its own unit.
_INDEX = "index"
_NIFTI_AXES = [
    Axis("x", "space", unit=_INDEX),
    Axis("y", "space", unit=_INDEX),
    Axis("z", "space", unit=_INDEX),
    Axis("t", "time", unit=_INDEX),
    Axis("c", "channel", unit=_INDEX),
    Axis("dim5", unit=_INDEX),
    Axis("dim6", unit=_INDEX),
]
_FLAT_AXES = {
    # The number of points, vertices, triangles and so on.
    0: Axis("n", unit=_INDEX),
    1: Axis("x", unit=_INDEX),
    2: Axis("y", unit=_INDEX),
    3: Axis("z", unit=_INDEX),
}
_FLAT_AXES_CHANNEL = {**_FLAT_AXES, 4: Axis("c", "channel", unit=_INDEX)}
_FLAT_AXES_TIME = {**_FLAT_AXES, 4: Axis("t", "time", unit=_INDEX)}
_AXES_DISP = {4: Axis("c", "displacement", unit=_INDEX)}
_NIFTI_SPECIFIC_AXES = {
    1004: {5: Axis("k", "channel", unit=_INDEX)},  # GENMATRIX
    1006: _AXES_DISP,  # DISPVECT
    1008: _FLAT_AXES_CHANNEL,  # POINTSET
    1009: _FLAT_AXES_CHANNEL,  # TRIANGLE
    # GIFTI intents
    2001: _FLAT_AXES_TIME,  # TIME_SERIES
    2002: _FLAT_AXES_CHANNEL,  # NODE_INDEX
    2003: _FLAT_AXES_CHANNEL,  # RGB_VECTOR
    2004: _FLAT_AXES_CHANNEL,  # RGBA_VECTOR
    2005: _FLAT_AXES_CHANNEL,  # SHAPE
    # FSL intents
    2006: _AXES_DISP,  # FSL_FNIRT_DISPLACEMENT_FIELD
    2007: _AXES_DISP,  # FSL_CUBIC_SPLINE_COEFFICIENTS
    2008: _AXES_DISP,  # FSL_DCT_COEFFICIENTS
    2009: _AXES_DISP,  # FSL_QUADRATIC_SPLINE_COEFFICIENTS
}


_NIFTI_FIELD_INTENTS = frozenset(
    {
        1006,  # DISPVECT
        1007,  # VECTOR
        2006,  # FSL_FNIRT_DISPLACEMENT_FIELD
        2007,  # FSL_CUBIC_SPLINE_COEFFICIENTS
        2008,  # FSL_DCT_COEFFICIENTS
        2009,  # FSL_QUADRATIC_SPLINE_COEFFICIENTS
    }
)
"""Intent codes that mark a file as holding a deformation field.

A NIfTI file can hold an image, affine maps or a field, so the container alone
does not say which object is wanted. The intent code does, which lets each
sniffer score itself instead of relying on an arbitrary precedence.
"""

_NIFTI_FSL_INTENTS = frozenset({2006, 2007, 2008, 2009})
"""FSL field intent codes, left to the FSL readers because the generic field
reader does not decode FSL storage conventions.
"""

_NIFTI_INTENT_NONE = 0
"""Intent code of a plain image."""

_NIFTI_INTENT_DISPVECT = 1006
"""Intent code of a displacement field.

ITK 5.4 and later read a three-component `DISPVECT` image as RAS displacements
in millimetres. brainhops reads it in the same way and writes it only for
displacement fields.
"""

_NIFTI_INTENT_VECTOR = 1007
"""Intent code of a generic vector image.

brainhops writes coordinate fields in RAS with this code, as SPM does for its
`y_` deformations. ITK writes it for every vector image, including its LPS
displacement fields, so the code says nothing about the frame; the intent name
[`_NIFTI_INTENT_NAME_MAPPING`][] marks RAS coordinate maps.
"""

_NIFTI_INTENT_NAME_NIFTYREG = "NREG_TRANS"
"""Intent name of every NiftyReg transformation.

NiftyReg stores its transformations as `VECTOR` images with this name and tells
them apart by `intent_p1`. Only the NiftyReg readers decode such files, so the
generic vector readers decline them.
"""

_NIFTI_INTENT_NAME_MAPPING = "Mapping"
"""Intent name of SPM coordinate fields (`y_` files).

brainhops writes it with `VECTOR` on RAS coordinate fields. ITK never writes an
intent name, so the name also distinguishes such a map from an ITK or ANTs
field in LPS.
"""


_NIFTI_XCODES = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}
"""World-space name of each xform code.

The reader names each affine map after its code, and the writer reads that name
back to choose the stored code.
"""

_NIFTI_XFORM_CODE_BY_NAME = {
    name: code for code, name in _NIFTI_XCODES.items()
}
"""Xform code of each world-space name."""

_QFORM_NAME = "qform"
"""Name that the reader gives to the world space of the qform."""

_NIFTI_DEFAULT_XFORM_CODE = 2
"""Xform code (`aligned`) stored for a world space without a known reference,
because NIfTI ignores forms with a zero code.
"""

_NIFTI1_MAX_DIM = 2**15 - 1
"""Largest extent that NIfTI-1 stores; larger images are written as NIfTI-2."""
