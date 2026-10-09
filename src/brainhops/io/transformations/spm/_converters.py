"""Private converters into SPM transformation formats."""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import (
    format_options,
    split_field_chain,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.nifti._converters import (
    RAS,
    check_coordinates,
    ras_coordinate_values,
)
from brainhops.io.transformations.nifti.fields import RASCoordinatesField

from .y import SpmCoordinatesField


@converter(_xforms.DisplacementField, SpmCoordinatesField)
@converter(_xforms.Sequence, SpmCoordinatesField)
@converter
def _(
    t: _xforms.CoordinatesField,
    cls: tx.Type[SpmCoordinatesField],
    **kwargs,
) -> SpmCoordinatesField:
    # Exactly, or raise: an affine from RAS to a grid, then one field of
    # coordinates on that grid, sampled as NIfTI stores it, followed by an
    # affine into RAS. A field of displacements is refused there, with the
    # reason.
    format_options(t, cls, kwargs)
    ras2voxel, field, voxel2ras = _spm_coordinates(t, cls)
    coordinates = ras_coordinate_values(t, field, voxel2ras, cls)
    chain = (
        RASToVoxel(matrix=ras2voxel[:-1]),
        RASCoordinatesField(field=coordinates),
    )
    return cls(transformations=chain)


@converter
def _(
    t: SpmCoordinatesField,
    cls: tx.Type[SpmCoordinatesField],
    **kwargs,
) -> SpmCoordinatesField:
    # Within its own format, a chain is changed by the rules of any chain,
    # which keep its type.
    return smart_replace(t, cls, **kwargs)


def _spm_coordinates(t: _xforms.Transformation, cls: type) -> tuple:
    """
    `t` as an SPM deformation field, or the reason it is not.

    Returns `(ras2voxel, field, voxel2ras)`: the homogeneous affine from
    RAS to the grid of the field, the field of coordinates, and the affine
    its coordinates are carried into RAS by.
    """
    ras2voxel, field, voxel2ras = split_field_chain(t, RAS, RAS, cls)
    check_coordinates(field, t, cls)
    if np.linalg.cond(ras2voxel) > _MAX_CONDITION:
        raise unrepresentable(
            t,
            cls,
            "the affine before its field is singular, or nearly so, and SPM "
            "stores the grid of the field by its inverse.",
        )
    return ras2voxel, field, voxel2ras


_MAX_CONDITION = 1e12
"""
The largest condition number of an affine whose inverse SPM is given.

Inverting a matrix loses about as many digits as its condition number
has, so beyond this one the grid affine written would keep fewer than
four of the sixteen significant digits of a float64.
"""
