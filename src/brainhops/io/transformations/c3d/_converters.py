"""
Private converters into the Convert3D matrix format.

A Convert3D matrix maps RAS to RAS, so every family with an affine form
converts into it exactly: the endpoints of the transformation are
bridged to RAS, which flips an affine between LPS spaces, and a chain of
affines is composed into one matrix
([`affine_between`][brainhops.io.transformations.base._conversions.affine_between]).
A transformation whose systems are not known is taken to map RAS to RAS,
because the endpoints of the format accept unknown systems.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import convert_withlog, converter
from brainhops.io.transformations.base._internal_api import (
    affine_between,
    format_options,
)
from brainhops.io.transformations.base.affines import _RAS

from ._affine import C3dAffine


@converter(_xforms.Identity, C3dAffine)
@converter(_xforms.Translation, C3dAffine)
@converter(_xforms.Scaling, C3dAffine)
@converter(_xforms.Permutation, C3dAffine)
@converter(_xforms.Linear, C3dAffine)
@converter(_xforms.AffineExponential, C3dAffine)
@converter(_xforms.SubspaceTransformation, C3dAffine)
@converter(_xforms.Sequence, C3dAffine)
@converter
def _(t: _xforms.Affine, cls: tx.Type[C3dAffine], **kwargs) -> C3dAffine:
    # `affine_between` either converts `t` exactly or raises.
    cls = C3dAffine
    format_options(t, cls, kwargs)
    matrix = affine_between(t, _RAS, _RAS, cls)
    return cls(matrix=matrix[:-1])


@converter
def _(t: C3dAffine, cls: tx.Type[C3dAffine], **kwargs) -> C3dAffine:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)
