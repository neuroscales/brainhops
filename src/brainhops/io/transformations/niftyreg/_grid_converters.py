"""
Private converters into the NiftyReg grids of control points.

`t.to(NiftyRegControlPointGrid)`, `NiftyRegControlPointGrid.from_any(t)`
and `io.save(t, "cpp.nii.gz", hint="niftyreg.cpp")` all run these
converters. A grid of control points, which `reg_f3d -cpp` writes,
stores a cubic or linear B-spline of displacements in RAS millimetres,
and it may hold an affine that is applied before the spline.

| Source                                   | Result                      |
| ---------------------------------------- | --------------------------- |
| B-spline of the degree of the grid       | exact                       |
| any other field, or chain with a field   | fitted B-spline, lossy      |
| velocity of degree 3 (velocity grid)     | exact                       |
| other velocity (velocity grid)           | fitted B-spline, lossy      |
| displacement (velocity grid)             | refused                     |

A transformation that holds a field between two affines is held
exactly when its field is a B-spline of the degree that the grid
stores, because the affine that does not undo the other one is moved
into the affine slot of the grid (see
[`_splines`][brainhops.io.transformations.base._splines]). A velocity
grid has no affine slot, so a velocity must lie between affines that
undo each other. Any other field is sampled at the control points of a
grid laid out as NiftyReg lays it out, with one control point before
the first voxel of the field, and the coefficients are fitted to the
samples. By default, the control points are the voxels of the field, so
that the fit is the cubic interpolation of its values, which is a loss.

The options of a conversion are `degree=` (3, the default, or 1 for a
linear grid) and `spacing=`, the distance between the control points in
millimetres, which always asks for a fit.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.errors import LossyConversionError
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    format_options,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base._splines import (
    SplineParts,
    control_grid,
    exact_spline_parts,
    fit_spline,
    source_grid,
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
    # The spline of `t` when it is one of the degree of the grid, and a
    # fitted spline otherwise.
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
    parts = None
    if spacing is None:
        parts = exact_spline_parts(t, RAS, RAS, cls, (degree,))
    if parts is None:
        parts = _fitted(t, cls, degree, spacing, log=False)
    chain = _sandwich(parts, log=False)
    if not parts.sandwich:
        # The affine that the field does not undo is applied first, in
        # the slot that NiftyReg reads from the header extension.
        chain = (RASToRAS(matrix=(parts.after @ parts.before)[:-1]), *chain)
    return _held(cls(transformations=chain), parts)


@register_converter(FIELD_SOURCES, NiftyRegVelocityGrid)
def _to_velocity_grid(
    t: _xforms.Transformation,
    cls: tx.Type[NiftyRegVelocityGrid],
    **kwargs,
) -> NiftyRegVelocityGrid:
    # The cubic velocity of `t`, or a cubic spline fitted to it.
    cls = NiftyRegVelocityGrid
    options = format_options(t, cls, dict(kwargs), VELOCITY_OPTIONS)
    spacing = options.get("spacing")
    parts = None
    if spacing is None:
        parts = exact_spline_parts(t, RAS, RAS, cls, (3,), log=True)
    if parts is None:
        parts = _fitted(t, cls, 3, spacing, log=True)
    if not parts.sandwich:
        raise unrepresentable(
            t,
            cls,
            "its velocity lies between affines that do not undo each "
            "other, and a NiftyReg velocity grid holds no affine.",
        )
    return _held(cls(transformations=_sandwich(parts, log=True)), parts)


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


def _fitted(
    t: _xforms.Transformation,
    cls: type,
    degree: int,
    spacing: tx.Any,
    log: bool,
) -> SplineParts:
    # A spline fitted on the grid that NiftyReg would build over the
    # grid of the first field of `t`, sampled once per control point.
    grid2world, shape = source_grid(t, RAS, cls)
    if len(shape) != 3:
        raise unrepresentable(
            t, cls, "NiftyReg grids of control points are three-dimensional."
        )
    grid2world, shape = control_grid(grid2world, shape, spacing)
    return fit_spline(
        t,
        grid2world,
        shape,
        RAS,
        RAS,
        cls,
        degree,
        BoundaryCondition.nearest,
        log=log,
    )


def _sandwich(
    parts: SplineParts, log: bool
) -> tx.Tuple[_xforms.Transformation, ...]:
    # The three slots of a NiftyReg grid: RAS to the control points, the
    # spline in their units, and back to RAS, placed by `after`.
    degree = int(parts.degree)
    velocity = {"log": True} if log else {}
    if log and parts.steps is not None:
        velocity["steps"] = int(parts.steps)
    voxel = _systems.VoxelCoordinateSystem()
    return (
        RASToVoxel(matrix=np.linalg.inv(parts.after)[:-1]),
        _xforms.DisplacementField(
            data=parts.coefficients,
            input=voxel,
            output=voxel,
            degree=degree,
            bound=BoundaryCondition.nearest,
            store=StoreEnum.from_coefficients(degree > 1),
            **velocity,
        ),
        VoxelToRAS(matrix=parts.after[:-1]),
    )


def _held(result: tx.Any, parts: SplineParts) -> tx.Any:
    # The result, or the error that carries it after a fit.
    if parts.loss is not None:
        raise LossyConversionError(
            f"The {type(result).__name__} that this transformation "
            f"converts to is not the same map: {parts.loss} Pass "
            f"lossy=True to accept it.",
            result=result,
        )
    return result
