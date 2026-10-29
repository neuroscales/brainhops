"""Private converters into SPM transformation formats."""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import converter, smart_replace
from brainhops.io.transformations.base._internal_api import (
    FIELD_SOURCES,
    FieldParts,
    coordinate_parts,
    format_options,
    held,
    split_field_chain,
    undoes,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import RASCoordinateField
from brainhops.io.transformations.nifti._internal_api import RAS

from ._fields import SpmCoordinateField


def _(
    t: _xforms.Transformation,
    cls: tx.Type[SpmCoordinateField],
    **kwargs,
) -> SpmCoordinateField:
    # An affine from RAS to a grid, then the RAS position of each node of
    # that grid. A field of displacements becomes the position of each
    # node, and a field that is not linear is sampled at its nodes, which
    # loses.
    cls = SpmCoordinateField
    options = format_options(t, cls, kwargs, ("metadata",))
    # The metadata of `t` rides along, converted by the field.
    options.setdefault("metadata", t.metadata)
    if _holds_no_coordinates(t, cls):
        chain = (RASToVoxel(), RASCoordinateField())
        return cls(transformations=chain, **options)
    parts = _spm_coordinates(t, cls)
    chain = (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        RASCoordinateField(field=parts.field.values),
    )
    return held(cls(transformations=chain, **options), parts)


for _source in FIELD_SOURCES:
    converter(_source, SpmCoordinateField)(_)


@converter
def _(
    t: SpmCoordinateField,
    cls: tx.Type[SpmCoordinateField],
    **kwargs,
) -> SpmCoordinateField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


def _holds_no_coordinates(t: _xforms.Transformation, cls: type) -> bool:
    """
    Whether `t` is a field of coordinates without data, the identity.

    Such a field is held without data, as a displacement field without
    data is. Nothing may then come before or after it, since an affine
    has no values to be carried into.
    """
    before, field, after = split_field_chain(t, RAS, RAS, cls)
    if field.data is not None:
        return False
    identity = np.eye(len(before))
    if not undoes(before, identity) or not undoes(after, identity):
        raise unrepresentable(
            t,
            cls,
            "it holds no coordinates, and an affine before or after them "
            "has no values to be carried into.",
        )
    return True


def _spm_coordinates(t: _xforms.Transformation, cls: type) -> FieldParts:
    """
    `t` as an SPM deformation field, or the reason it is not.

    Returns the parts that
    [`coordinate_parts`][brainhops.io.transformations.base._conversions.coordinate_parts]
    gives: the homogeneous affine from RAS to the grid of the field, and
    the RAS position of each node of that grid.
    """
    parts = coordinate_parts(t, RAS, RAS, cls)
    if np.linalg.cond(parts.world2grid) > _MAX_CONDITION:
        raise unrepresentable(
            t,
            cls,
            "the affine before its field is singular, or nearly so, and SPM "
            "stores the grid of the field by its inverse.",
        )
    return parts


_MAX_CONDITION = 1e12
"""
The largest condition number of an affine whose inverse SPM is given.

Inverting a matrix loses about as many digits as its condition number
has, so beyond this one the grid affine written would keep fewer than
four of the sixteen significant digits of a float64.
"""
