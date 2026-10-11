"""
Private converter of an Elastix chain of transforms to its own format.

An Elastix file holds a chain of transforms, and no exact conversion
into it is written yet (#312). Another transformation is therefore
relabelled as an `ElastixTransform` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. An `ElastixTransform` that is converted to its own class
is changed by the rules of any chain, and a chain in the other Elastix
syntax is relabelled.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import relabel

from ._xform import ElastixTransform


@converter
def _(
    t: ElastixTransform, cls: tx.Type[ElastixTransform], **kwargs
) -> ElastixTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type. The other syntax
    # holds the same chain, so it is relabelled, as any other
    # transformation is.
    if not isinstance(t, cls):
        return relabel(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)
