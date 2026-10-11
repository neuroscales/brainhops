"""
Private converter of the LTA format to itself.

An LTA file stores an affine together with the geometry of the two
volumes that it maps, and no exact conversion into it is written yet
(#312). Another transformation is therefore relabelled as an
`LtaTransformation` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. Whether the LTA format can write those systems is decided
when it is written. An `LtaTransformation` that is converted to its own
class is changed by the rules of any affine, and another variant of the
format is relabelled, as any other transformation is.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base._converters import relabel

from ._xforms import LtaTransformation


@converter
def _(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type. Another variant of the format reads
    # its matrix in other systems, so it is relabelled only when those
    # systems are compatible.
    if not isinstance(t, cls):
        return relabel(t, cls, **kwargs)
    return _convert_withlog(t, cls, "matrix", **kwargs)
