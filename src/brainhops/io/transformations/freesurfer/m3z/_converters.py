"""
Private converters into the M3Z format.

`t.to(M3zMorph)`, `M3zMorph.from_any(t)` and `io.save(t, "warp.m3z")`
all run these converters. The converter returns a morph that is the map
`t` is, raises a `LossyConversionError` that carries an approximation
of it, or raises a `ConversionError` that says what the morph cannot
hold.

| Source                             | Format       |
| ---------------------------------- | ------------ |
| any field, `Sequence`              | `M3zMorph`   |

A morph maps the RAS world of the atlas to the RAS world of the source
image, so the endpoints of `t` are bridged exactly to RAS, and an
endpoint that is not known is taken to be RAS. The morph is built as
`mri_ca_register` writes it, with positions in the voxels of the source
image (`GCAM_VOX`):

1. the affine from atlas RAS to the nodes, which are the nodes of the
   grid of the field of `t`, one atlas voxel apart;
2. the position of each node, in the voxels of the source image;
3. the voxel-to-RAS affine of the source image.

The source image is given with the conversion option `moving=`, as a
nibabel image or header, an image of brainhops, or a pair of its
voxel-to-RAS matrix and its shape. Without it, the source image is
taken to have the geometry of the atlas, which is the grid of the
field, so that the positions are the voxels of the atlas that each node
maps to. The map is the same either way, and only the geometry that the
file records changes. A morph built with `moving=` holds a record of
the two geometries, which the writer reads, and a morph built without
it holds no record.

A field of displacements becomes the position of each node, since the
two are the same map inside the grid and only extend differently
beyond it, and a velocity is integrated. A field interpolated with
another degree than 1 is sampled at its nodes, which is a loss: the
converter raises a `LossyConversionError`, and
`t.to(M3zMorph, on_loss="ignore")` returns the sampled morph. A morph that
is converted to its own class is changed by the rules of any chain,
which keep its type.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import (
    FIELD_SOURCES,
    apply_affine,
    coordinate_parts,
    format_options,
    held,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

from ._metadata import M3zMetadata
from ._raw import GCAM_VOX, M3zGeometry, M3zRaw
from ._xform import M3zMorph

RAS = _systems.RASmm()
"""The RAS world, in millimetres, of the atlas and of the source image."""

VOXEL = _systems.VoxelCoordinateSystem()
"""The voxels of the source image, in which the positions are given."""

OPTIONS = ("moving",)
"""What a conversion to a morph may set: the source image."""


@register_converter(FIELD_SOURCES, M3zMorph)
def _(t: _xforms.Transformation, cls: tx.Type[M3zMorph], **kwargs) -> M3zMorph:
    # An affine from atlas RAS to the nodes, the position of each node in
    # the voxels of the source image, and the voxel-to-RAS affine of the
    # source image. What loses is reported.
    cls = M3zMorph
    if isinstance(t, cls):
        return smart_replace(t, cls, **kwargs)
    options = format_options(t, cls, kwargs, OPTIONS)
    parts = coordinate_parts(t, RAS, RAS, cls)
    values = parts.field.values
    shape = tuple(int(n) for n in values.shape[:-1])
    if len(shape) != 3 or int(values.shape[-1]) != 3:
        raise unrepresentable(
            t,
            cls,
            f"a morph holds one 3-vector per node of a 3-D grid, and its "
            f"field has the shape {tuple(values.shape)}.",
        )
    atlas = parts.grid2world
    moving = options.get("moving")
    if moving is None:
        source, source_shape = atlas, shape
    else:
        source, source_shape = _geometry(moving, t, cls)
    positions = apply_affine(np.linalg.inv(source), values)
    chain = (
        RASToVoxel(matrix=parts.world2grid[:-1]),
        _xforms.CoordinateField(field=positions, input=VOXEL, output=VOXEL),
        VoxelToRAS(matrix=source[:-1]),
    )
    if moving is None:
        # The writer takes a source image placed as the atlas to have the
        # shape of the atlas, so the morph needs no record.
        morph = cls(transformations=chain)
    else:
        # The record holds the shape of the source image, which the chain
        # does not say, for the writer.
        record = M3zRaw(
            spacing=1,
            type=GCAM_VOX,
            image=M3zGeometry.from_vox2ras(source, source_shape),
            atlas=M3zGeometry.from_vox2ras(atlas, shape),
        )
        morph = cls(metadata=M3zMetadata.from_raw(record))
        morph.transformations = chain
    return held(morph, parts)


def _geometry(
    image: tx.Any, t: _xforms.Transformation, cls: type
) -> tx.Tuple[np.ndarray, tx.Tuple[int, int, int]]:
    """
    Return the voxel-to-RAS matrix and the shape of the source image.

    The image is a pair of a matrix and a shape, or an image whose
    geometry is read as an FSL reference image is read.
    """
    if isinstance(image, (tuple, list)) and len(image) == 2:
        vox2ras, shape = image
    else:
        from brainhops.io.transformations.fsl._affines import (
            _best_affine,
            _shape,
        )

        try:
            vox2ras, shape = _best_affine(image), _shape(image)
        except ValueError as error:
            raise unrepresentable(
                t, cls, f"the geometry of moving= cannot be read. {error}"
            ) from error
    vox2ras = np.asarray(vox2ras, dtype=np.float64)
    if vox2ras.shape == (3, 4):
        vox2ras = np.concatenate([vox2ras, [[0.0, 0.0, 0.0, 1.0]]])
    shape = tuple(int(n) for n in tuple(shape)[:3])
    if vox2ras.shape != (4, 4) or len(shape) != 3:
        raise unrepresentable(
            t,
            cls,
            "moving= gives the geometry of a 3-D image: its 4x4 "
            "voxel-to-RAS matrix and its shape.",
        )
    return vox2ras, shape
