"""
Private converters into the LTA format.

An LTA file stores an affine together with the geometry of the two
volumes that it maps, and no exact conversion into it is written yet
(#312). Each family with an affine form is therefore relabelled as an
`LtaTransformation` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format. Whether the LTA format can write
those systems is decided when it is written. An `LtaTransformation`
that is converted to its own class is changed by the rules of any
affine, and another variant of the format is relabelled, as any other
transformation is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)

from ._xforms import LtaTransformation

# An LTA file stores an affine together with the geometry of the two
# volumes that it maps, and no exact conversion into it is written yet,
# so each family with an affine form is relabelled.
converter(_xforms.Identity, LtaTransformation)(relabelling)
converter(_xforms.Translation, LtaTransformation)(relabelling)
converter(_xforms.Scaling, LtaTransformation)(relabelling)
converter(_xforms.Permutation, LtaTransformation)(relabelling)
converter(_xforms.Linear, LtaTransformation)(relabelling)
converter(_xforms.Affine, LtaTransformation)(relabelling)
converter(_xforms.AffineExponential, LtaTransformation)(relabelling)
converter(_xforms.SubspaceTransformation, LtaTransformation)(relabelling)
converter(_xforms.Sequence, LtaTransformation)(relabelling)


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
