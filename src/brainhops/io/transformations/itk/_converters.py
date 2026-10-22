"""
Private converters into the ITK formats that store parameters.

`t.to(MatTransform)`, `MatTransform.from_any(t)` and
`io.save(t, "0GenericAffine.mat")` all run these converters, and so do
the same calls for `TfmTransform` and `H5Transform`. Each one returns an
instance of its format that is the map `t` is, or raises a
`ConversionError` that says what the format cannot hold.

| Source                             | Format                          |
| ---------------------------------- | ------------------------------- |
| any affine, `Sequence` of affines  | `MatTransform`                  |
| any affine, `Sequence` of affines  | `TfmTransform`, `H5Transform`   |
| a chain of ITK blocks              | `TfmTransform`, `H5Transform`   |

In this table, "any affine" means every family that has an affine form
(see
[`AFFINE_SOURCES`][brainhops.io.transformations.base._conversions.AFFINE_SOURCES]).
Every block of an ITK file maps LPS millimetres to LPS millimetres, so
the endpoints of `t` are bridged to LPS, which flips an affine between
RAS spaces, and a chain of affines is composed into one matrix, which is
written as an `AffineTransform` centered on the origin. An endpoint that
is not known is taken to be LPS. The number of dimensions is read from
the systems of `t`, or from its matrix, so a 2-D affine is written as a
2-D ITK transform.

A chain read from another ITK file already holds ITK blocks, such as an
`Euler3DTransform` with its center. A text or HDF5 file keeps these
blocks as they are, warps included, since it holds any chain. A MATLAB
file holds one block, as ANTs writes it, so it keeps the block of a
chain of one, and composes a longer chain of affines into one affine.

A conversion takes no option, since every option would change the map.
A format that is converted to its own class is changed by the rules of
any chain, which keep its type.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.errors import ConversionError
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    affine_between,
    format_options,
    register_converter,
    unrepresentable,
)

from ._common import ItkStruct
from ._systems import _make_system
from ._xform import ItkTransform
from .mat import MatTransform
from .tfm import TfmTransform

_FORMATS: tx.List[type] = [MatTransform, TfmTransform]

try:
    from .h5 import H5Transform

    _FORMATS.append(H5Transform)
except ImportError:  # pragma: no cover
    # The HDF5 format needs the optional h5py.
    pass


def _convert(
    t: _xforms.Transformation, cls: tx.Type[ItkTransform], **kwargs
) -> ItkTransform:
    # Exactly, or raise. A chain of ITK blocks is kept when the format can
    # hold it, and anything else is reduced to one affine between LPS
    # spaces.
    format_options(t, cls, kwargs)
    blocks = _itk_blocks(t)
    if blocks is not None and (cls is not MatTransform or len(blocks) == 1):
        return cls(transformations=blocks)
    return cls(transformations=[_lps_affine(t, cls)])


def _within(
    t: ItkTransform, cls: tx.Type[ItkTransform], **kwargs
) -> ItkTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


for _format in _FORMATS:
    register_converter(AFFINE_SOURCES, _format)(_convert)
    converter(_format, _format)(_within)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _itk_blocks(
    t: _xforms.Transformation,
) -> tx.Optional[tx.List[_xforms.Transformation]]:
    """
    Return the blocks of a chain read from an ITK file, or `None`.

    Only a chain whose every element is an ITK block is returned, since
    its blocks are exactly what an ITK file stores.
    """
    if isinstance(t, ItkStruct):
        return [t]
    if not isinstance(t, ItkTransform):
        return None
    blocks = list(t)
    if blocks and all(isinstance(block, ItkStruct) for block in blocks):
        return blocks
    return None


def _lps_affine(t: _xforms.Transformation, cls: type) -> _xforms.Affine:
    """
    Return `t` as one affine between the LPS spaces of ITK.

    Raises
    ------
    ConversionError
        If `t` is not an affine, or a chain of affines, between systems
        that are bridged to LPS.
    """
    space = _make_system(_ndim(t, cls))
    matrix = affine_between(t, space, space, cls)
    return _xforms.Affine(matrix=matrix[:-1], input=space, output=space)


def _ndim(t: _xforms.Transformation, cls: type) -> int:
    """
    Return the number of dimensions of the space that `t` maps.

    It is read from the systems of `t`, and otherwise from its matrix.
    """
    for system in (t.input, t.output):
        ndim = None if system is None else system.ndim
        if ndim is not None:
            return int(ndim)
    try:
        matrix = t.to(_xforms.Affine).matrix
    except ConversionError as error:
        raise unrepresentable(
            t, cls, f"it is not an affine. {error}"
        ) from error
    return 3 if matrix is None else int(matrix.shape[0])
