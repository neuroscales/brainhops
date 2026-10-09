"""The constants of the NIfTI format: axes, intents, xform codes."""

# internals
from brainhops.datamodel.axes import Axis

# The axes of a NIfTI array, by position. They are the axes of its voxel
# space, so they count samples (`IndexUnit`): reversing one shifts its
# origin by one less than its extent. A reader that builds a physical space
# from them gives them its own unit.
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
    # number of points / vertices / triangles / ...
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
    # --- GIFTI ---
    2001: _FLAT_AXES_TIME,  # TIME_SERIES
    2002: _FLAT_AXES_CHANNEL,  # NODE_INDEX
    2003: _FLAT_AXES_CHANNEL,  # RGB_VECTOR
    2004: _FLAT_AXES_CHANNEL,  # RGBA_VECTOR
    2005: _FLAT_AXES_CHANNEL,  # SHAPE
    # --- FSL ---
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
"""
Intent codes that mark a NIfTI file as holding a deformation field.

A NIfTI file is legitimately an image *and* a set of affines *and*,
sometimes, a field -- so the container alone cannot say which object the
caller wants. The intent code can, and is what lets sniffers score
themselves instead of relying on an arbitrary precedence between kinds.
"""

_NIFTI_FSL_INTENTS = frozenset({2006, 2007, 2008, 2009})
"""FSL-specific field intent codes, decoded by the FSL readers.

A file with one of these codes is left to the FSL readers rather than
claimed by the generic field reader, which does not decode FSL's storage
conventions.
"""

_NIFTI_INTENT_NONE = 0
"""Intent code of a plain image: no specialized interpretation."""

_NIFTI_INTENT_DISPVECT = 1006
"""
Intent code of a field of displacement vectors.

The NIfTI-1 standard reserves it "specifically for displacements", and
ITK 5.4 and later reads a three-component `DISPVECT` image as RAS
displacements in millimetres. brainhops reads it the same way, and
writes it only for displacement fields.
"""

_NIFTI_INTENT_VECTOR = 1007
"""
Intent code of a generic vector image.

The NIfTI-1 standard reserves it "for any other type of vector" than a
displacement. brainhops writes its fields of RAS coordinates with it,
as SPM writes its `y_` deformations (coordinate maps), and ITK writes it
for every vector image unless told otherwise, so it is also the code of
ITK's (LPS) displacement fields. It says nothing about the frame its
vectors are in; the intent name `"Mapping"` (see below) marks the RAS
coordinate maps.
"""

_NIFTI_INTENT_NAME_NIFTYREG = "NREG_TRANS"
"""
The intent name NiftyReg gives every transformation it writes.

NiftyReg stores its deformation and displacement fields and its
control-point grids as `VECTOR` (1007) images named `"NREG_TRANS"`, and
tells them apart with `intent_p1` (`reg-lib/cpu/Maths.hpp`,
`NREG_TRANS_TYPE`). The name is evidence of a NiftyReg file, which only
the NiftyReg readers decode, so the generic `VECTOR` readers decline it.
"""

_NIFTI_INTENT_NAME_MAPPING = "Mapping"
"""
The intent name SPM gives a field of coordinates (`y_` files).

brainhops writes it next to `VECTOR` on a field of RAS coordinates, so
the file says what its vectors are, not only that they are vectors. ITK's
`NiftiImageIO` never writes an intent name, so neither ITK nor ANTs
files carry it, and it also tells such a map from an ITK (LPS) field.
"""


_NIFTI_XCODES = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}
"""
The world space each NIfTI xform code names.

A qform or sform code labels the world space its matrix maps voxels into.
The reader names each affine after its code, and the writer reads that
name back to choose the code to store.
"""

_NIFTI_XFORM_CODE_BY_NAME = {
    name: code for code, name in _NIFTI_XCODES.items()
}
"""The xform code for a world-space name, the reverse of `_NIFTI_XCODES`."""

_QFORM_NAME = "qform"
"""The name the reader gives the rigid voxel-to-RAS affine of the qform."""

_NIFTI_DEFAULT_XFORM_CODE = 2
"""
The xform code stored when the world space names no known reference.

NIfTI ignores a form whose code is zero, so a form that carries real
geometry is stored with a non-zero code. The value `2` is NIfTI's
"aligned" code.
"""

_NIFTI1_MAX_DIM = 2**15 - 1
"""
The largest array dimension NIfTI-1 can store.

NIfTI-1 records each dimension in a signed 16-bit field. An array with a
larger extent along any axis is written as NIfTI-2 instead.
"""
