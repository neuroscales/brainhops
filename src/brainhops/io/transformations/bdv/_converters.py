"""
Private converters into the BigDataViewer registration format.

A BigDataViewer registration holds a chain of affines, and no exact
conversion into it is written yet (#312). A chain is therefore
relabelled as a `BdvViewRegistration` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format, the voxels of a view and the global space. A
`BdvViewRegistration` that is converted to its own class is changed by
the rules of any chain, as any other transformation of that family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import relabelling

from ._xform import BdvViewRegistration

# A BigDataViewer registration holds a chain of affines, and no exact
# conversion into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, BdvViewRegistration)(relabelling)


@converter
def _(
    t: BdvViewRegistration, cls: tx.Type[BdvViewRegistration], **kwargs
) -> BdvViewRegistration:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
