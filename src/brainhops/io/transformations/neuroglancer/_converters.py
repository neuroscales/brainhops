"""
Private converters into the Neuroglancer coordinate transform.

`t.to(NeuroglancerTransform)`, `NeuroglancerTransform.from_any(t)` and
`io.save(t, "t.ngtransform.json")` all run these converters. Each one
either returns a transform that is exactly the map `t` is, or raises a
`ConversionError` that says what the format cannot hold.

A Neuroglancer transform has no coordinate systems of its own: it names
the dimensions of its input and output, with their units. Every family
that has an affine form, and a `Sequence` of affines, is therefore
reduced to one matrix between the systems of `t` itself, with no bridge,
and the format keeps those systems. The conversion is exact only when
the systems can be written as dimensions, which means that every axis
has a name that Neuroglancer accepts and a unit that it has (see
[`NeuroglancerTransform`][brainhops.io.transformations.neuroglancer.NeuroglancerTransform]).
Otherwise the converter refuses, rather than leaving the refusal to the
writer. A conversion takes no option, since nothing may change the map
that it converts.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import convert_withlog, converter
from brainhops.io.base.parsers import UnrepresentableTransformationError
from brainhops.io.transformations.base._internal_api import (
    affine_between,
    format_options,
    unrepresentable,
)

from ._xform import NeuroglancerTransform, encoded


@converter(_xforms.Identity, NeuroglancerTransform)
@converter(_xforms.Translation, NeuroglancerTransform)
@converter(_xforms.Scaling, NeuroglancerTransform)
@converter(_xforms.Permutation, NeuroglancerTransform)
@converter(_xforms.Linear, NeuroglancerTransform)
@converter(_xforms.AffineExponential, NeuroglancerTransform)
@converter(_xforms.SubspaceTransformation, NeuroglancerTransform)
@converter(_xforms.Sequence, NeuroglancerTransform)
@converter
def _(
    t: _xforms.Affine, cls: tx.Type[NeuroglancerTransform], **kwargs
) -> NeuroglancerTransform:
    # Exactly, or raise: the reduction to one affine between the systems
    # of `t`, and the encoding of those systems as dimensions.
    cls = NeuroglancerTransform
    options = format_options(t, cls, kwargs)
    matrix = affine_between(t, t.input, t.output, cls)
    result = cls(matrix=matrix[:-1], input=t.input, output=t.output, **options)
    try:
        encoded(result)
    except UnrepresentableTransformationError as error:
        raise unrepresentable(t, cls, str(error)) from error
    return result


@converter
def _(
    t: NeuroglancerTransform, cls: tx.Type[NeuroglancerTransform], **kwargs
) -> NeuroglancerTransform:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)
