"""
Private converters into the `register.dat` format.

A `register.dat` file stores an affine between the tkregister RAS spaces
of two volumes, and no exact conversion into it is written yet (#312).
Each family with an affine form is therefore relabelled as a
`RegisterDatTransformation` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the
transformation only when its coordinate systems are not known or are
compatible with those of the format. Whether the format can write those
systems is decided when it is written. A `RegisterDatTransformation`
that is converted to its own class is changed by the rules of any
affine. Another variant of the format that reads its matrix from a
record reads the same record in its own coordinate systems, which is
exact, and one that holds a matrix of its own is relabelled, as any
other transformation is.
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

from ._xforms import RegisterDatTransformation

# A `register.dat` file stores an affine between tkregister spaces, and
# no exact conversion into it is written yet, so each family with an
# affine form is relabelled.
converter(_xforms.Identity, RegisterDatTransformation)(relabelling)
converter(_xforms.Translation, RegisterDatTransformation)(relabelling)
converter(_xforms.Scaling, RegisterDatTransformation)(relabelling)
converter(_xforms.Permutation, RegisterDatTransformation)(relabelling)
converter(_xforms.Linear, RegisterDatTransformation)(relabelling)
converter(_xforms.Affine, RegisterDatTransformation)(relabelling)
converter(_xforms.AffineExponential, RegisterDatTransformation)(relabelling)
converter(_xforms.SubspaceTransformation, RegisterDatTransformation)(
    relabelling
)
converter(_xforms.Sequence, RegisterDatTransformation)(relabelling)


@converter
def _(
    t: RegisterDatTransformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type. Another variant of the format reads
    # the record of `t`, when it has one, in its own systems, with the
    # same geometries. A matrix of its own is relabelled only when its
    # systems are compatible.
    if isinstance(t, cls):
        return _convert_withlog(t, cls, "matrix", **kwargs)
    if t._data is None:
        return _view(t, cls, **kwargs)
    return relabel(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _view(
    t: RegisterDatTransformation,
    cls: tx.Type[RegisterDatTransformation],
    **kwargs,
) -> RegisterDatTransformation:
    # Another variant of the format that shares the record of `t`, whose
    # matrix it decodes rather than copies.
    kwargs.setdefault("data", None)
    return smart_replace(t, cls, **kwargs)
