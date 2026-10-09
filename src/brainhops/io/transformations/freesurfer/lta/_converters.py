"""
Private converters into the LTA format.

These converters make a conversion into the format, such as
`t.to(LtaTransformation)`, fail with a reason instead of building a
wrong object. The format derives its coordinate systems from the content
it holds, so it has no endpoints to check. Without these converters, the
converters of the data model would rebuild a transformation as an
`LtaTransformation` with the same parameters, and the result would not
be the map that the format says it holds. Each family that would be
rebuilt in this way is refused until an exact conversion is written
(#312). An `LtaTransformation` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base._conversions import no_exact_conversion

from ._xforms import LtaTransformation


@converter(_xforms.Identity, LtaTransformation)
@converter(_xforms.Translation, LtaTransformation)
@converter(_xforms.Scaling, LtaTransformation)
@converter(_xforms.Permutation, LtaTransformation)
@converter(_xforms.Linear, LtaTransformation)
@converter(_xforms.AffineExponential, LtaTransformation)
@converter(_xforms.SubspaceTransformation, LtaTransformation)
@converter(_xforms.Sequence, LtaTransformation)
@converter
def _(
    t: _xforms.Affine, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # An LTA file stores an affine together with the geometry of the two
    # volumes that it maps, and no exact conversion into it is written
    # yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type. A conversion into another variant
    # of the format is not written yet.
    if not isinstance(t, cls):
        raise no_exact_conversion(t, cls)
    return _convert_withlog(t, cls, "matrix", **kwargs)
