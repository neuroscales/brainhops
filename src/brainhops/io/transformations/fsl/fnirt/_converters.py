"""
Private converter of the FNIRT format to itself.

FNIRT stores a field between the scaled voxels of two images, and no
exact conversion into it is written yet (#312). Another transformation
is therefore relabelled as a `FnirtWarpField` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. A `FnirtWarpField` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)

from ._base import FnirtWarpField


@converter
def _(
    t: FnirtWarpField, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
