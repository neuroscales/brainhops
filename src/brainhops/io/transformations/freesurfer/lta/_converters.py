"""
Private converters into the LTA format.

Each family that the converters of the data model would otherwise
rebuild as an `LtaTransformation` is refused here until an exact
conversion is written (#312), so that a transformation is never
relabelled as the format. A format converted to its own class is changed
as any transformation of its family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base.conversions import no_exact_conversion

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
    # volumes it maps, and nothing converts to it yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: LtaTransformation, cls: tx.Type[LtaTransformation], **kwargs
) -> LtaTransformation:
    # Within its own format, an affine is changed by the rules of any
    # affine (a new `matrix=`, say), which keep its type. Another variant
    # of the format is not converted to yet.
    if not isinstance(t, cls):
        raise no_exact_conversion(t, cls)
    return _convert_withlog(t, cls, "matrix", **kwargs)
