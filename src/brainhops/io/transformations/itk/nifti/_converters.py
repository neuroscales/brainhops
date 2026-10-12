"""
Private converters into the ITK NIfTI field format.

ITK stores a field in LPS on the grid of its file, and no exact
conversion into it is written yet (#312). A chain is therefore
relabelled as an `ItkNiftiField` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format. An `ItkNiftiField` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is, and another variant of the format is relabelled.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)

from ._fields import ItkNiftiField

# ITK stores a field in LPS on the grid of its file, and no exact
# conversion into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, ItkNiftiField)(relabelling)


@converter
def _(
    t: ItkNiftiField, cls: tx.Type[ItkNiftiField], **kwargs
) -> ItkNiftiField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type. Another variant
    # of the format holds the same chain as other values, so it is
    # relabelled, as any other transformation is.
    if not isinstance(t, cls):
        return relabel(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)
