"""
Private converters into the formats of MNI transform files.

`t.to(XfmAffine)`, `XfmAffine.from_any(t)` and `io.save(t, "t.xfm")`
all run these converters. Each one returns an instance of its format
that is the map `t` is, or raises a `ConversionError` that says what the
format cannot hold.

| Source                             | Format         |
| ---------------------------------- | -------------- |
| any affine, `Sequence` of affines  | `XfmAffine`    |
| any affine, `Sequence` of affines  | `XfmTransform` |

In this table, "any affine" means every family that has an affine form
(see
[`AFFINE_SOURCES`][brainhops.io.transformations.base._conversions.AFFINE_SOURCES]).
Every transform of an MNI transform file maps RAS to RAS, so the
endpoints of `t` are bridged to RAS, which flips an affine between LPS
spaces, and a chain of affines is composed into one matrix. An endpoint
that is not known is taken to be RAS. An `XfmAffine` stores that matrix
as its single linear transform, and an `XfmTransform` stores it as a
chain of one linear transform.

A chain that holds a field is refused by both formats for now. An MNI
transform file stores a field as a grid transform, whose displacements
are kept in a separate MINC volume, and the converter of an
`XfmTransform` is written so that a branch for such a chain can be added
to it.

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
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    affine_between,
    format_options,
    register_converter,
)

from ._affine import XfmAffine
from ._chain import XfmTransform

RAS = _systems.RASmm()
"""The RAS world, in millimetres, that every MNI transform maps."""


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
    return _convert_withlog(t, cls, "matrix", **kwargs)


@register_converter(AFFINE_SOURCES, XfmTransform)
def _(
    t: _xforms.Transformation, cls: tx.Type[XfmTransform], **kwargs
) -> XfmTransform:
    # Exactly, or raise. A chain of affines becomes one linear transform.
    # A chain that holds a field needs a grid transform, which is not
    # converted yet, and `affine_between` refuses it with a reason.
    cls = XfmTransform
    format_options(t, cls, kwargs)
    return cls(transformations=(_linear(t, cls),))


@converter
def _(t: XfmTransform, cls: tx.Type[XfmTransform], **kwargs) -> XfmTransform:
    # A transform that is already in this format is changed by the rules
    # of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _linear(t: _xforms.Transformation, cls: type) -> _xforms.Affine:
    """
    Return `t` as the affine of one linear transform, from RAS to RAS.

    Raises
    ------
    ConversionError
        If `t` is not an affine, or a chain of affines, between systems
        that are bridged to RAS.
    """
    matrix = affine_between(t, RAS, RAS, cls)
    return _xforms.Affine(matrix=matrix[:-1], input=RAS, output=RAS)
