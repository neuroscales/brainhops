"""
Private converters into the ITK formats that store parameters.

`t.to(MatTransform)`, `MatTransform.from_any(t)` and
`io.save(t, "0GenericAffine.mat")` all run these converters, and so do
the same calls for `TfmTransform` and `H5Transform`. Each one returns an
instance of its format that is the map `t` is, raises a
`LossyConversionError` that carries an approximation of it, or raises a
`ConversionError` that says what the format cannot hold.

| Source                             | Format                          |
| ---------------------------------- | ------------------------------- |
| any affine, `Sequence` of affines  | `MatTransform`                  |
| any affine, `Sequence` of affines  | `TfmTransform`, `H5Transform`   |
| any field, `Sequence` with fields  | `TfmTransform`, `H5Transform`   |
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

A chain that holds fields is cut into pieces that each map LPS to LPS.
A text or HDF5 file stores each piece of affines as an
`AffineTransform`, and each field as a `DisplacementFieldTransform`,
which holds the displacement of each node of its grid in LPS
millimetres. A field of coordinates becomes the displacement of each
node, and a velocity is integrated. A field interpolated with another
degree than 1 is sampled at its nodes, which is a loss: the converter
then raises a `LossyConversionError` that carries the result.

A conversion takes no option, since every option would change the map.
A format that is converted to its own class is changed by the rules of
any chain, which keep its type.
"""

import numpy as np
import typing_extensions as tx

from brainhops.backends import get_array_backend
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.errors import ConversionError
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    _is_field,
    _leaves,
    affine_between,
    displacement_parts,
    format_options,
    held,
    register_converter,
    unrepresentable,
    world_pieces,
)

from ._common import (
    ItkPrecision,
    ItkStruct,
    ItkTransformClass,
)
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
    # hold it. A chain that holds fields is cut into affines and dense
    # displacement fields, which a text or HDF5 file holds, and anything
    # else is reduced to one affine between LPS spaces.
    format_options(t, cls, kwargs)
    blocks = _itk_blocks(t)
    if blocks is not None and (cls is not MatTransform or len(blocks) == 1):
        return cls(transformations=blocks)
    if cls is not MatTransform and _holds_a_field(t):
        return _field_chain(t, cls)
    return cls(transformations=[_lps_affine(t, cls)])


def _within(
    t: ItkTransform, cls: tx.Type[ItkTransform], **kwargs
) -> ItkTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


for _format in _FORMATS:
    sources = AFFINE_SOURCES
    if _format is not MatTransform:
        sources = tuple(dict.fromkeys(AFFINE_SOURCES + FIELD_SOURCES))
    register_converter(sources, _format)(_convert)
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


def _holds_a_field(t: _xforms.Transformation) -> bool:
    """Return whether `t` is a field, or a chain that holds one."""
    return any(_is_field(leaf) for leaf in _leaves(t))


def _field_chain(
    t: _xforms.Transformation, cls: tx.Type[ItkTransform]
) -> ItkTransform:
    """
    Return `t` as a chain of affines and dense displacement fields.

    `t` is cut into pieces that each map LPS to LPS (see
    [`world_pieces`][brainhops.io.transformations.base._conversions.world_pieces]).
    Each piece of affines becomes an affine, which the file writes as an
    `AffineTransform`, and each field becomes a
    `DisplacementFieldTransform`, which stores the displacement of each
    node of its grid in LPS millimetres. A field of coordinates becomes
    the displacement of each node, and a velocity is integrated. A field
    interpolated with another degree than 1 is sampled at its nodes,
    which loses, and the result is then carried by a
    `LossyConversionError`.

    Raises
    ------
    ConversionError
        If a bridge to LPS does not exist, or what lies between the
        fields is not an affine.
    LossyConversionError
        If a field is sampled at its nodes.
    """
    ndim = _field_space_ndim(t)
    space = _make_system(ndim)
    transforms = []
    loss = None
    for piece in world_pieces(t, space, cls):
        if isinstance(piece, _xforms.Affine):
            matrix = affine_between(piece, space, space, cls)
            transforms.append(
                _xforms.Affine(matrix=matrix[:-1], input=space, output=space)
            )
            continue
        parts = displacement_parts(piece, space, cls)
        if parts.field.data is None:
            continue
        loss = loss or parts.loss
        transforms.append(_displacement_block(parts, ndim))
    result = cls(transformations=transforms)
    return held(result, parts._replace(loss=loss)) if transforms else result


def _field_space_ndim(t: _xforms.Transformation) -> int:
    """Return the number of dimensions of the fields that `t` holds."""
    for leaf in _leaves(t):
        if _is_field(leaf) and leaf.data is not None:
            return int(leaf.data.shape[-1])
    return 3


def _displacement_block(parts: tx.Any, ndim: int) -> ItkStruct:
    """
    Return a `DisplacementFieldTransform` block that stores a field.

    The fixed parameters hold the shape, the origin, the spacing and the
    direction of the grid, which are read from its voxel-to-LPS affine.
    The parameters hold the displacement of each node in LPS millimetres,
    with the first axis varying fastest and the components of each node
    next to each other, as ITK stores an image of vectors.
    """
    vox2lps = np.asarray(parts.grid2world, dtype=np.float64)[:ndim]
    linear = vox2lps[:, :ndim]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing
    values = parts.field.values
    shape = tuple(int(n) for n in values.shape[:-1])
    backend = get_array_backend(values)
    rotate = backend.asarray(linear, dtype=values.dtype)
    world = backend.matmul(rotate, values[..., None])[..., 0]
    order = tuple(range(ndim - 1, -1, -1)) + (ndim,)
    flat = np.asarray(world.transpose(order)).reshape(-1)
    fixed = np.concatenate(
        [
            np.asarray(shape, dtype=np.float64),
            vox2lps[:, ndim],
            spacing,
            direction.reshape(-1),
        ]
    )
    return ItkStruct(
        type=ItkTransformClass.DisplacementFieldTransform,
        precision=ItkPrecision.Double,
        ndim_input=ndim,
        ndim_output=ndim,
        parameters=flat.astype(np.float64),
        fixed_parameters=fixed,
    )
