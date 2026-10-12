"""
Private converters into the FNIRT format.

FNIRT stores a field between the scaled voxels of two images, and no
exact conversion into it is written yet (#312). A chain is therefore
relabelled as a `FnirtWarpField` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format. A `FnirtWarpField` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import relabelling

from ._base import FnirtWarpField

# FNIRT stores a field between the scaled voxels of two images, and no
# exact conversion into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, FnirtWarpField)(relabelling)


@converter
def _(
    t: FnirtWarpField, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
