"""
Private converters into the formats of MNI transform files.

`t.to(XfmAffine)`, `XfmAffine.from_any(t)` and `io.save(t, "t.xfm")`
all run these converters. Each one returns an instance of its format
that is the map `t` is, raises a `LossyConversionError` that carries an
approximation of it, or raises a `ConversionError` that says what the
format cannot hold.

| Source                                 | Format         |
| -------------------------------------- | -------------- |
| any affine, `Sequence` of affines      | `XfmAffine`    |
| any affine, any field, any `Sequence`  | `XfmTransform` |

In this table, "any affine" means every family that has an affine form
(see
[`AFFINE_SOURCES`][brainhops.io.transformations.base._conversions.AFFINE_SOURCES]),
and "any field" every family of fields (see
[`FIELD_SOURCES`][brainhops.io.transformations.base._conversions.FIELD_SOURCES]).
Every transform of an MNI transform file maps RAS to RAS, so the
endpoints of `t` are bridged to RAS, which flips an affine between LPS
spaces. An endpoint that is not known is taken to be RAS. An
`XfmAffine` stores a chain of affines composed into one matrix, as its
single linear transform.

A field, or any chain, is converted to an `XfmTransform` by
[`grid_chain`][brainhops.io.transformations.minc._grid.grid_chain]: each
piece of affines becomes a linear transform, and each field becomes a
grid transform, whose displacements are written as a new MINC volume
beside the file. libminc interpolates a grid with cubic splines, so a
field that is interpolated otherwise is the same map at the nodes of its
grid only, and the converter raises a `LossyConversionError` that
carries the result, which `t.to(XfmTransform, on_loss="ignore")`
returns.

A conversion takes no option, since every option would change the map.
An `XfmAffine` that is converted to its own class is changed by the
rules of any affine, and an `XfmTransform` by the rules of any chain.

An `.xfm` file name is claimed by both formats, which hold an affine
equally well. `XfmTransform` therefore has a lower `PRIORITY`, so that
`io.save` writes an affine as an `XfmAffine`, the format that `io.load`
also prefers for a file of one linear transform.
"""

import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import (
    convert_withlog,
    converter,
    smart_replace,
)
from brainhops.errors import LossyConversionError
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    affine_between,
    format_options,
    register_converter,
)

from ._affine import XfmAffine
from ._chain import XfmTransform
from ._grid import grid_chain

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that every MNI transform maps."""


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------


@register_converter(AFFINE_SOURCES, XfmAffine)
def _(t: _xforms.Affine, cls: tx.Type[XfmAffine], **kwargs) -> XfmAffine:
    # Exactly, or raise: `affine_between` bridges the endpoints of `t` to
    # RAS and reduces a chain to one affine.
    cls = XfmAffine
    format_options(t, cls, kwargs)
    matrix = affine_between(t, RAS, RAS, cls)
    return cls(matrix=matrix[:-1])


@converter
def _(t: XfmAffine, cls: tx.Type[XfmAffine], **kwargs) -> XfmAffine:
    # An affine that is already in this format is changed by the rules of
    # any affine, for example when it is given a new `matrix=`, and these
    # rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)


# ----------------------------------------------------------------------
#   FIELDS AND CHAINS
# ----------------------------------------------------------------------
# A field, or a chain that holds fields, is cut into affines and grid
# transforms by `grid_chain`, whose displacements the file writes as new
# MINC volumes. A chain of affines alone becomes one linear transform.


@register_converter(
    dict.fromkeys(AFFINE_SOURCES + FIELD_SOURCES), XfmTransform
)
def _(
    t: _xforms.Transformation, cls: tx.Type[XfmTransform], **kwargs
) -> XfmTransform:
    # One linear transform per piece of affines and one grid transform per
    # field, each from RAS to RAS. What loses is reported once.
    cls = XfmTransform
    format_options(t, cls, kwargs)
    transforms, loss = grid_chain(t, cls)
    result = cls(transformations=transforms)
    if loss is None:
        return result
    raise LossyConversionError(
        f"The XfmTransform that this transformation converts to is not "
        f"the same map: {loss} Pass on_loss='ignore' to accept it.",
        result=result,
    )


@converter
def _(t: XfmTransform, cls: tx.Type[XfmTransform], **kwargs) -> XfmTransform:
    # A transform that is already in this format is changed by the rules
    # of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
