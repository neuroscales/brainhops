"""
Private converters into the formats of MNI transform files.

An MNI transform file stores affines and chains of transforms from RAS
to RAS. No exact conversion into an `XfmAffine` is written yet (#312),
so each family with an affine form is relabelled as one by the
converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format.

A field, or any chain, is converted to an `XfmTransform` by
[`grid_chain`][brainhops.io.transformations.minc._grid.grid_chain]: its
endpoints are bridged exactly to RAS, each piece of affines becomes a
linear transform, and each field becomes a grid transform, whose
displacements are written as a new MINC volume beside the file. libminc
interpolates a grid with cubic splines, so a field that is interpolated
otherwise is the same map at the nodes of its grid only, and the
converter raises a `LossyConversionError` that carries the result,
which `t.to(XfmTransform, lossy=True)` returns.

An `XfmAffine` that is converted to its own class is changed by the
rules of any affine, and an `XfmTransform` by the rules of any chain.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)
from brainhops.errors import LossyConversionError
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    format_options,
    register_converter,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)

from ._affine import XfmAffine
from ._chain import XfmTransform
from ._grid import grid_chain

# A file of one linear transform stores an affine from RAS to RAS, and no
# exact conversion into it is written yet, so each family with an affine
# form is relabelled.
converter(_xforms.Identity, XfmAffine)(relabelling)
converter(_xforms.Translation, XfmAffine)(relabelling)
converter(_xforms.Scaling, XfmAffine)(relabelling)
converter(_xforms.Permutation, XfmAffine)(relabelling)
converter(_xforms.Linear, XfmAffine)(relabelling)
converter(_xforms.Affine, XfmAffine)(relabelling)
converter(_xforms.AffineExponential, XfmAffine)(relabelling)
converter(_xforms.SubspaceTransformation, XfmAffine)(relabelling)
converter(_xforms.Sequence, XfmAffine)(relabelling)

# ----------------------------------------------------------------------
#   FIELDS AND CHAINS
# ----------------------------------------------------------------------
# A field, or a chain that holds fields, is cut into affines and grid
# transforms by `grid_chain`, whose displacements the file writes as new
# MINC volumes.


def _(
    t: _xforms.Transformation, cls: tx.Type[XfmTransform], **kwargs
) -> XfmTransform:
    # One linear transform per piece of affines and one grid transform per
    # field, each from RAS to RAS. What loses is reported once.
    cls = XfmTransform
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    format_options(t, cls, kwargs)
    transforms, loss = grid_chain(t, cls)
    result = cls(transformations=transforms)
    if loss is None:
        return result
    raise LossyConversionError(
        f"The XfmTransform that this transformation converts to is not "
        f"the same map: {loss} Pass lossy=True to accept it.",
        result=result,
    )


register_converter(FIELD_SOURCES, XfmTransform)(_)


@converter
def _(t: XfmAffine, cls: tx.Type[XfmAffine], **kwargs) -> XfmAffine:
    # An affine that is already in this format is changed by the rules of
    # any affine, for example when it is given a new `matrix=`, and these
    # rules keep its type.
    if isinstance(t, cls):
        return _convert_withlog(t, cls, "matrix", **kwargs)
    return relabel(t, cls, **kwargs)


@converter
def _(  # noqa: F811
    t: XfmTransform, cls: tx.Type[XfmTransform], **kwargs
) -> XfmTransform:
    # A transform that is already in this format is changed by the rules
    # of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
