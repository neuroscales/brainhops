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
class is changed by the rules of any affine. Another variant of the
format that reads its matrix from a record reads the same record in its
own coordinate systems, which is exact, and one that holds a matrix of
its own is relabelled, as any other transformation is.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
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
    # the record of `t`, when it has one, in its own systems. A matrix of
    # its own is relabelled only when its systems are compatible.
    if isinstance(t, cls):
        return _convert_withlog(t, cls, "matrix", **kwargs)
    if t._data is None:
        return _view(t, cls, **kwargs)
    return relabel(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _view(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # Another variant of the format that shares the record of `t`, whose
    # matrix it decodes rather than copies.
    kwargs.setdefault("data", None)
    return smart_replace(t, cls, **kwargs)
