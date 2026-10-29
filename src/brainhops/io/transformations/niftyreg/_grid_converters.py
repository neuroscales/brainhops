"""
Private converters into the NiftyReg grids of control points.

`t.to(NiftyRegControlPointGrid)`, `NiftyRegControlPointGrid.from_any(t)`
and `io.save(t, "cpp.nii.gz", hint="niftyreg.cpp")` all run these
converters. A grid of control points, which `reg_f3d -cpp` writes,
stores the coefficients of a cubic or linear B-spline of displacements
in RAS millimetres, and a velocity grid stores those of a cubic
velocity.

| Source                                     | Result                    |
| ------------------------------------------ | ------------------------- |
| field of the degree of the grid            | exact                     |
| field of another degree                    | interpolated, lossy       |
| any field, with `spacing=`                 | evaluated, lossy          |
| velocity (velocity grid)                   | as a field, above         |
| displacement (velocity grid)               | refused                   |

A grid of control points is a field like any other, interpolated with
cubic or linear B-splines, so it is converted as the other field
formats are (see
[`displacement_parts`][brainhops.io.transformations.base._conversions.displacement_parts]).
The field keeps the grid on which it is placed, and the affines that
come before and after it are stored in the two affine slots of the grid,
which NiftyReg reads from the header extensions: it applies the affine
of the first extension before the spline and composes the affine of the
second extension after it, as a symmetric registration writes them. The
spline then holds the displacement alone (see
[`world_pieces`][brainhops.io.transformations.base._conversions.world_pieces]).
By default, the control points are the nodes of the field of `t`. A
field of the degree of the grid is then kept as it is, and any other
field is evaluated at its nodes, and its values are converted to the
coefficients of the grid, which is a loss. A field of coordinates
becomes the displacement of each node. With `spacing=`, the distance
between the control points in millimetres, the field is evaluated on a
grid laid out as NiftyReg lays it out, with one control point before
the first voxel of the field, which is a loss.

A velocity grid holds a velocity, and NiftyReg composes the affine of
its second header extension after the flow. The affines around the
velocity are carried to its end by a change of coordinates, which is
exact (see
[`split_affine`][brainhops.io.transformations.base._conversions.split_affine]),
and the velocity is placed on its grid in the input world. The options of a
conversion are `degree=` (3, the default, or 1 for a linear grid of
displacements) and `spacing=`.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import converter, smart_replace
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.transformations.base._internal_api import (
    FIELD_SOURCES,
    FieldParts,
    control_grid,
    displacement_parts,
    displacements_of,
    field_slots,
    format_options,
    held,
    register_converter,
    sample_on_grid,
    source_grid,
    split_affine,
    split_field_chain,
    undoes,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import (
    RASToRAS,
    RASToVoxel,
    VoxelToRAS,
)

from ._fields import (
    NiftyRegControlPointGrid,
    NiftyRegSequence,
    NiftyRegVelocityGrid,
)

RAS = _systems.RASmm()
"""The world of every NiftyReg transformation, RAS in millimetres."""

VOXEL = _systems.VoxelCoordinateSystem()
"""The indices of the control points of a grid."""

GRID_OPTIONS = ("degree", "spacing")
"""What a conversion to a grid of control points may set."""

VELOCITY_OPTIONS = ("spacing",)
"""What a conversion to a velocity grid may set."""


@register_converter(FIELD_SOURCES, NiftyRegControlPointGrid)
def _to_grid(
    t: _xforms.Transformation,
    cls: tx.Type[NiftyRegControlPointGrid],
    **kwargs,
) -> NiftyRegControlPointGrid:
    # The field of `t`, as the coefficients of a spline of the degree of
    # the grid, on its own nodes or on a grid of the asked spacing.
    cls = NiftyRegControlPointGrid
    options = format_options(t, cls, dict(kwargs), GRID_OPTIONS)
    degree = int(options.get("degree", 3))
    if degree not in (1, 3):
        raise unrepresentable(
            t,
            cls,
            f"a NiftyReg grid of control points is cubic or linear, and "
            f"degree={degree} was asked.",
        )
    spacing = options.get("spacing")
    # The field keeps its own grid, and the affines on either side of it
    # go to the two slots that NiftyReg reads from the header extensions,
    # so that the spline holds the displacement alone.
    before, field, after = field_slots(t, RAS, cls)
    if spacing is None:
        parts = displacement_parts(field, RAS, cls, degree=degree)
    else:
        grid2world, shape = _control_grid(field, cls, spacing)
        parts = displacements_of(
            sample_on_grid(
                field, np.linalg.inv(grid2world), shape, RAS, RAS, cls, degree
            )
        )
    _check_3d(parts, t, cls)
    chain = _sandwich(parts)
    if before is not None or after is not None:
        identity = np.eye(len(parts.world2grid))
        before = identity if before is None else before
        chain = (RASToRAS(matrix=before[:-1]), *chain)
    if after is not None:
        chain = (*chain, RASToRAS(matrix=after[:-1]))
    return held(cls(transformations=chain), parts)


@register_converter(FIELD_SOURCES, NiftyRegVelocityGrid)
def _to_velocity_grid(
    t: _xforms.Transformation,
    cls: tx.Type[NiftyRegVelocityGrid],
    **kwargs,
) -> NiftyRegVelocityGrid:
    # The velocity of `t`, as the coefficients of a cubic spline, on its
    # own nodes or on a grid of the asked spacing.
    cls = NiftyRegVelocityGrid
    options = format_options(t, cls, dict(kwargs), VELOCITY_OPTIONS)
    spacing = options.get("spacing")
    # The affines around the velocity are carried to its end, where
    # NiftyReg composes the affine of the second header extension: a
    # change of coordinates carries the flow of a velocity exactly.
    after, velocity = split_affine(t, RAS, RAS, cls, "after")
    velocity = _velocity(velocity if velocity is not None else t, cls)
    if spacing is None:
        parts = displacement_parts(velocity, RAS, cls, keep_log=True, degree=3)
    else:
        # The values of the velocity are evaluated on the new grid, as
        # those of a displacement would be, and are read again as a
        # velocity.
        grid2world, shape = _control_grid(velocity, cls, spacing)
        parts = displacements_of(
            sample_on_grid(
                _as_displacement(velocity),
                np.linalg.inv(grid2world),
                shape,
                RAS,
                RAS,
                cls,
                3,
            )
        )
        field = parts.field
        parts = parts._replace(
            field=_xforms.DisplacementField(
                data=field.values,
                log=True,
                steps=velocity.steps,
                degree=3,
                input=VOXEL,
                output=VOXEL,
            )
        )
    _check_3d(parts, t, cls)
    chain = _sandwich(parts)
    if not undoes(after, np.eye(len(after))):
        identity = RASToRAS(matrix=np.eye(len(after))[:-1])
        chain = (identity, *chain, RASToRAS(matrix=after[:-1]))
    return held(cls(transformations=chain), parts)


@converter(NiftyRegSequence, NiftyRegControlPointGrid)
@converter(NiftyRegSequence, NiftyRegVelocityGrid)
def _(
    t: NiftyRegSequence, cls: tx.Type[NiftyRegSequence], **kwargs
) -> NiftyRegSequence:
    # A grid converted to its own class is changed by the rules of any
    # chain. Another NiftyReg field is converted as any other field is.
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    if issubclass(cls, NiftyRegVelocityGrid):
        return _to_velocity_grid(t, cls, **kwargs)
    return _to_grid(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _control_grid(
    t: _xforms.Transformation, cls: type, spacing: tx.Any
) -> tx.Tuple[np.ndarray, tx.Tuple[int, ...]]:
    # The grid that NiftyReg would build over the grid of the first field
    # of `t`, with control points `spacing` millimetres apart.
    grid2world, shape = source_grid(t, RAS, cls)
    return control_grid(grid2world, shape, spacing)


def _velocity(t: _xforms.Transformation, cls: type) -> _xforms.Transformation:
    # The velocity of `t`, which must lie between affines that undo each
    # other once the affine that NiftyReg composes after it is taken out.
    before, field, after = split_field_chain(t, RAS, RAS, cls)
    if not isinstance(field, _xforms.DisplacementField) or not field.log:
        raise unrepresentable(
            t,
            cls,
            "the format stores a velocity, and its field holds a "
            "displacement, whose logarithm brainhops does not compute.",
        )
    if not undoes(before, after):
        raise unrepresentable(
            t,
            cls,
            "its velocity lies between affines that do not undo each "
            "other, and a NiftyReg velocity grid holds an affine only "
            "after its flow.",
        )
    return t


def _as_displacement(t: _xforms.Transformation) -> _xforms.Transformation:
    # The chain of `t` with its velocity read as a displacement, so that
    # evaluating the chain gives the values of the velocity rather than
    # those of its flow.
    before, field, after = split_field_chain(t, RAS, RAS, NiftyRegVelocityGrid)
    values = _xforms.DisplacementField(
        data=field.data,
        degree=field.degree,
        bound=field.bound,
        store=field.store,
        input=VOXEL,
        output=VOXEL,
    )
    return _xforms.Sequence(
        transformations=[
            _xforms.Affine(matrix=before[:-1], input=RAS, output=VOXEL),
            values,
            _xforms.Affine(matrix=after[:-1], input=VOXEL, output=RAS),
        ]
    )


def _check_3d(parts: FieldParts, t: _xforms.Transformation, cls: type) -> None:
    # NiftyReg grids are three-dimensional.
    data = parts.field.data
    if data is None or data.shape[-1] != 3:
        raise unrepresentable(
            t,
            cls,
            "NiftyReg grids of control points hold the coefficients of a "
            "three-dimensional field on each control point.",
        )


def _sandwich(parts: FieldParts) -> tx.Tuple[_xforms.Transformation, ...]:
    # The three slots of a NiftyReg grid: RAS to the control points, the
    # spline in their units, stored as its coefficients, and back to RAS.
    field = parts.field
    degree = int(field.degree)
    velocity = {}
    if field.log:
        velocity["log"] = True
        if getattr(field, "steps", None) is not None:
            velocity["steps"] = int(field.steps)
    coefficients = field.coefficients if degree > 1 else field.values
    return (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        _xforms.DisplacementField(
            data=coefficients,
            input=VOXEL,
            output=VOXEL,
            degree=degree,
            bound=BoundaryCondition.nearest,
            store=[StoreEnum.values, StoreEnum.coefficients][degree > 1],
            **velocity,
        ),
        VoxelToRAS(matrix=parts.grid2world[:-1]),
    )
