"""
Private converters into the FLIRT format.

`t.to(FlirtTransform, reference=ref, moving=mov)`,
`FlirtTransform.from_any(t, reference=ref, moving=mov)` and
`io.save(t, "t.mat", conversion={"reference": ref, "moving": mov})` all
run these converters. Each one returns a `FlirtTransform` that is the
map `t` is, or raises a `ConversionError` that says what the format
cannot hold.

A FLIRT matrix maps the scaled millimetres of the moving image to those
of the reference image, so it is placed in world coordinates by the
geometry of both images, which the file does not store. The converter
therefore takes the two images as options, `reference=` and `moving=`,
under the names that the format takes them when a file is loaded, and
refuses a conversion that is given only one of them, or neither. The
images are nibabel images or headers, or any object that gives the
voxel-to-world affine, the shape and the voxel sizes of an image.

Every family that has an affine form, and a `Sequence` of affines, is
first read as one affine from the RAS world of the reference image to
that of the moving image, which is the direction of the data model:
the endpoints of `t` are bridged to RAS, which flips an affine between
LPS spaces, and an endpoint that is not known is taken to be RAS. The
FLIRT matrix is then `ref.ras2fsl @ inv(W) @ mov.fsl2ras`, where `W` is
that affine, and the transformation reads `W` back from it. The only
difference is the rounding of the inversion. A `FlirtTransform` that is
converted to its own class is changed by the rules of any affine.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import convert_withlog, converter
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    affine_between,
    format_options,
    register_converter,
    unrepresentable,
)
from brainhops.io.transformations.base.affines import _RAS

from .._affines import _ImageGeometry
from ._xform import FlirtTransform

OPTIONS = ("reference", "moving")
"""What a conversion to a FLIRT transform takes: the two images that
place its matrix in world coordinates."""


@register_converter(AFFINE_SOURCES, FlirtTransform)
def _(
    t: _xforms.Transformation, cls: tx.Type[FlirtTransform], **kwargs
) -> FlirtTransform:
    # Exactly, or raise: the affine from reference RAS to moving RAS, and
    # the FLIRT matrix that the two images give it.
    cls = FlirtTransform
    options = format_options(t, cls, kwargs, OPTIONS)
    missing = [name for name in OPTIONS if options.get(name) is None]
    if missing:
        names = " and ".join(f"{name}=" for name in missing)
        raise unrepresentable(
            t,
            cls,
            f"a FLIRT matrix maps the scaled millimetres of a moving and a "
            f"reference image, and the conversion is not given {names}. "
            f"Pass both images, as in t.to(FlirtTransform, reference=ref, "
            f"moving=mov), or conversion={{'reference': ref, 'moving': "
            f"mov}} to io.save.",
        )
    world = affine_between(t, _RAS, _RAS, cls)
    if world.shape != (4, 4):
        raise unrepresentable(
            t,
            cls,
            f"a FLIRT matrix holds a 3-D affine, and this one has "
            f"{len(world) - 1} dimensions.",
        )
    try:
        reference = _ImageGeometry(options["reference"])
        moving = _ImageGeometry(options["moving"])
        flirt = reference.ras2fsl @ np.linalg.inv(world) @ moving.fsl2ras
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        raise unrepresentable(
            t,
            cls,
            f"the FLIRT matrix cannot be computed from the images and the "
            f"affine. {error}",
        ) from error
    return cls(flirt_matrix=flirt, **options)


@converter
def _(
    t: FlirtTransform, cls: tx.Type[FlirtTransform], **kwargs
) -> FlirtTransform:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)
