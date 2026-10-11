"""
Private converters into the X5 format.

An X5 file holds a chain of transforms, and no exact conversion into it
is written yet (#312). A chain is therefore relabelled as an
`X5Transform` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format. An `X5Transform` that is converted to its own class is
changed by the rules of any chain, through the converters of the data
model.
"""

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.io.transformations.base._converters import relabelling

from ._xform import X5Transform

# An X5 file holds a chain of transforms, and no exact conversion into it
# is written yet, so a chain is relabelled.
converter(_xforms.Sequence, X5Transform)(relabelling)
