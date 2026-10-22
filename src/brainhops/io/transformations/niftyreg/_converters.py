"""
Private converters into the dense NiftyReg fields.

`t.to(NiftyRegDisplacementField)`,
`NiftyRegDisplacementField.from_any(t)` and
`io.save(t, "warp.nii.gz", hint="niftyreg.disp")` all run these
converters. Each one returns an instance of its format that is the map
`t` is, raises a `LossyConversionError` that carries an approximation
of it, or raises a `ConversionError` that says what the format cannot
hold.

| Source                             | Format                        |
| ---------------------------------- | ----------------------------- |
| any field, `Sequence`              | `NiftyRegDisplacementField`   |
| any field, `Sequence`              | `NiftyRegDeformationField`    |
| a velocity, `Sequence`             | `NiftyRegVelocityField`       |

NiftyReg stores a field in the RAS world, in millimetres, on the grid
of its file. The endpoints of `t` are bridged exactly to RAS, so a
field between LPS spaces is flipped into RAS, and an endpoint that is
not known is taken to be RAS.

The three formats store the same chain as a field of displacements
between a world-to-grid affine and its inverse, and differ in how the
file encodes it: as displacements, as positions, or as a stationary
velocity whose flow is the map. A field of coordinates becomes the
displacement of each node, since the two are the same map inside the
grid and only extend differently beyond it, and the affines on either
side of a field are carried into its values when they do not undo
each other. A velocity is integrated into a displacement or a
deformation field, and it is kept as it is in a velocity field. A
velocity field holds only a velocity, since brainhops computes no
logarithm of a displacement. A field interpolated with another degree
than 1 is sampled at its nodes, which is a loss: the converter raises a
`LossyConversionError`, and `t.to(Format, lossy=True)` returns the
sampled field.

The control-point grids of NiftyReg have converters of their own. A
conversion takes no option, since every option would change the map.
An object that is already in its format is changed by the rules of any
chain, which keep its type, and another NiftyReg field is converted as
any other field is.
"""

import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    FieldParts,
    displacement_parts,
    format_options,
    held,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

from ._fields import (
    NiftyRegDeformationField,
    NiftyRegDisplacementField,
    NiftyRegSequence,
    NiftyRegVelocityField,
)

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that a NiftyReg field maps from and to."""

SOURCES = (*FIELD_SOURCES, NiftyRegSequence)
"""The families that the converters take. The base class of the NiftyReg
fields is listed, so that another NiftyReg field reaches the converter of
the format it is converted to."""


def _chain(parts: FieldParts) -> tx.Tuple[_xforms.Transformation, ...]:
    """Return the chain `(ras2voxel, displacement, voxel2ras)` of parts."""
    return (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        parts.field,
        VoxelToRAS(matrix=parts.grid2world[:-1]),
    )


@register_converter(SOURCES, NiftyRegDisplacementField)
@register_converter(SOURCES, NiftyRegDeformationField)
def _(
    t: _xforms.Transformation,
    cls: tx.Type[NiftyRegSequence],
    **kwargs,
) -> NiftyRegSequence:
    # A field of displacements between a world-to-grid affine and its
    # inverse, which the format writes as displacements or as positions.
    # Anything else is turned into one, and what loses is reported.
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    format_options(t, cls, kwargs)
    parts = displacement_parts(t, RAS, cls)
    return held(cls(transformations=_chain(parts)), parts)


@register_converter(SOURCES, NiftyRegVelocityField)
def _(
    t: _xforms.Transformation,
    cls: tx.Type[NiftyRegVelocityField],
    **kwargs,
) -> NiftyRegVelocityField:
    # A stationary velocity between a world-to-grid affine and its inverse,
    # kept as a velocity. A displacement is refused, since brainhops
    # computes no logarithm.
    cls = NiftyRegVelocityField
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    format_options(t, cls, kwargs)
    parts = displacement_parts(t, RAS, cls, keep_log=True)
    if not parts.field.log:
        raise unrepresentable(
            t,
            cls,
            "a NiftyReg velocity field stores a stationary velocity, and "
            "this transformation is not a linear velocity between a "
            "world-to-grid affine and its inverse. brainhops computes no "
            "logarithm of a displacement.",
        )
    return held(cls(transformations=_chain(parts)), parts)
