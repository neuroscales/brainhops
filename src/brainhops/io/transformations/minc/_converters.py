"""
Private converters into the formats of MNI transform files.

An MNI transform file stores affines and chains of transforms from RAS
to RAS, and no exact conversion into it is written yet (#312). Each
family with an affine form is therefore relabelled as an `XfmAffine`,
and a chain as an `XfmTransform`, by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format. An `XfmAffine` that is converted to
its own class is changed by the rules of any affine, and an
`XfmTransform` by the rules of any chain.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
    smart_replace,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)

from ._affine import XfmAffine
from ._chain import XfmTransform

# A file of one linear transform stores an affine from RAS to RAS, and no
# exact conversion into it is written yet, so each family with an affine
# form is relabelled.
converter(_xforms.Identity, XfmAffine)(relabelling)
converter(_xforms.Translation, XfmAffine)(relabelling)
converter(_xforms.Scaling, XfmAffine)(relabelling)
converter(_xforms.Permutation, XfmAffine)(relabelling)
converter(_xforms.Linear, XfmAffine)(relabelling)
converter(_xforms.Affine, XfmAffine)(relabelling)
converter(_xforms.AffineExponential, XfmAffine)(relabelling)
converter(_xforms.SubspaceTransformation, XfmAffine)(relabelling)
converter(_xforms.Sequence, XfmAffine)(relabelling)

# A file of several transforms stores a chain, and no exact conversion into
# it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, XfmTransform)(relabelling)


@converter
def _(t: XfmAffine, cls: tx.Type[XfmAffine], **kwargs) -> XfmAffine:
    # An affine that is already in this format is changed by the rules of
    # any affine, for example when it is given a new `matrix=`, and these
    # rules keep its type.
    if isinstance(t, cls):
        return _convert_withlog(t, cls, "matrix", **kwargs)
    return relabel(t, cls, **kwargs)


@converter
def _(  # noqa: F811
    t: XfmTransform, cls: tx.Type[XfmTransform], **kwargs
) -> XfmTransform:
    # A transform that is already in this format is changed by the rules
    # of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
