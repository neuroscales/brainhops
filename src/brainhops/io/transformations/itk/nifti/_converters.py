"""
Private converters into the ITK NIfTI field formats.

`t.to(ItkNiftiDisplacementField)`,
`ItkNiftiDisplacementField.from_any(t)` and
`io.save(t, "warp.nii.gz")` all run these converters. Each one returns
an instance of its format that is the map `t` is, raises a
`LossyConversionError` that carries an approximation of it, or raises a
`ConversionError` that says what the format cannot hold.

| Source                             | Format                        |
| ---------------------------------- | ----------------------------- |
| any field, `Sequence`              | `ItkNiftiDisplacementField`   |
| any field, `Sequence`              | `ItkNiftiCoordinateField`    |

ITK stores a field in the LPS world, in millimetres, on the grid of its
file. The endpoints of `t` are bridged exactly to LPS, so a field
between RAS spaces is flipped into LPS, and an endpoint that is not
known, when the other is, is taken to be LPS. The converters write 3-D
fields, whose world is `LPSmm`. A 2-D field that was read from a file
is changed within its own format by the rules of any chain.

A field is stored as its values at the nodes of its grid, which ITK
reads back with linear interpolation and with the nearest value outside
the grid. A field of coordinates is written to a displacement file as
the displacement of each node, and a field of displacements to a
coordinates file as the position of each node, since the two are the
same map inside the grid and only extend differently beyond it. A
velocity is integrated, and the affines on either side of a field of
displacements are carried into its values when they do not undo each
other. A field interpolated with another degree than 1 is sampled at
its nodes, which is a loss: the converter raises a
`LossyConversionError`, and `t.to(Format, on_loss="ignore")` returns the
sampled field.

A field whose endpoints are both unknown is refused, since ITK maps LPS
where the other formats of NIfTI fields map RAS and take such a field
to be in RAS. `io.save` then writes it to a `.nii.gz` name as a NIfTI
field of RAS displacements.

A conversion takes no option, since every option would change the map.
An object that is already in its format is changed by the rules of any
chain, which keep its type, and the other variant of the format is
converted as any other field is.
"""

import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import smart_replace
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    coordinate_parts,
    displacement_parts,
    format_options,
    held,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.base.fields import LPSCoordinateField

from ._fields import (
    ItkNiftiCoordinateField,
    ItkNiftiDisplacementField,
    ItkNiftiField,
)

LPS = _systems.LPSmm()
"""The LPS world, in millimetres, that a 3-D ITK field maps from and to."""

_NDIM = 3
"""The number of dimensions of the fields that the converters write."""

SOURCES = (*FIELD_SOURCES, ItkNiftiField)
"""The families that the converters take. The base class of the ITK
fields is listed, so that the other variant of the format reaches the
converter of the format it is converted to."""


@register_converter(SOURCES, ItkNiftiDisplacementField)
def _(
    t: _xforms.Transformation,
    cls: tx.Type[ItkNiftiDisplacementField],
    **kwargs,
) -> ItkNiftiDisplacementField:
    # One field of displacements between an LPS-to-grid affine and its
    # inverse. Anything else is turned into one, and what loses is
    # reported.
    cls = ItkNiftiDisplacementField
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    format_options(t, cls, kwargs)
    _check_known(t, cls)
    parts = displacement_parts(t, LPS, cls)
    world, voxel = cls._spaces(_NDIM)
    chain = (
        LPSToVoxel(matrix=parts.world2grid[:-1], input=world, output=voxel),
        replace(parts.field, input=voxel, output=voxel),
        VoxelToLPS(matrix=parts.grid2world[:-1], input=voxel, output=world),
    )
    return held(cls(transformations=chain), parts)


@register_converter(SOURCES, ItkNiftiCoordinateField)
def _(
    t: _xforms.Transformation,
    cls: tx.Type[ItkNiftiCoordinateField],
    **kwargs,
) -> ItkNiftiCoordinateField:
    # An affine from LPS to a grid, then the LPS position of each node of
    # that grid. Anything else is turned into one, and what loses is
    # reported.
    cls = ItkNiftiCoordinateField
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    format_options(t, cls, kwargs)
    _check_known(t, cls)
    parts = coordinate_parts(t, LPS, LPS, cls)
    world, voxel = cls._spaces(_NDIM)
    chain = (
        LPSToVoxel(matrix=parts.world2grid[:-1], input=world, output=voxel),
        LPSCoordinateField(
            field=parts.field.values, input=voxel, output=world
        ),
    )
    return held(cls(transformations=chain), parts)


def _check_known(t: _xforms.Transformation, cls: type) -> None:
    """
    Refuse a transformation whose endpoints are both unknown.

    Every other format of fields in a NIfTI file maps RAS, and a field
    whose world is not known is taken to be in RAS when it is converted
    to one of them. ITK maps LPS instead, so taking such a field to be
    in LPS would flip it against what the other formats read, and it is
    refused. `io.save` then writes it to a `.nii.gz` name as a NIfTI
    field of RAS displacements.
    """
    if t.input is None and t.output is None:
        raise unrepresentable(
            t,
            cls,
            "its endpoints are not known, and ITK maps LPS where the other "
            "formats of NIfTI fields map RAS, so a field is held only when "
            "its world is known. Give it LPS or RAS endpoints.",
        )
