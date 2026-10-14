"""
Private converters into the TrakEM2 format.

A TrakEM2 affine places a patch in the plane of its layer, and no exact
conversion into it is written yet (#312). Each family with a 2D affine
form is therefore relabelled as a `TrakEM2PatchAffine` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format. A relabelled affine has no project,
so it can be read and compared but not written, until its matrix is
assigned to a patch of a project that was read.
"""

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.io.transformations.base._converters import relabelling

from ._xform import TrakEM2PatchAffine

converter(_xforms.Identity, TrakEM2PatchAffine)(relabelling)
converter(_xforms.Translation, TrakEM2PatchAffine)(relabelling)
converter(_xforms.Scaling, TrakEM2PatchAffine)(relabelling)
converter(_xforms.Linear, TrakEM2PatchAffine)(relabelling)
converter(_xforms.Affine, TrakEM2PatchAffine)(relabelling)
