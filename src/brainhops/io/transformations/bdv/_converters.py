"""
Private converters into the BigDataViewer registration format.

`t.to(BdvViewRegistration)`, `BdvViewRegistration.from_any(t)` and
`io.save(t, "dataset.xml")` all run these converters. Each one returns a
registration that is the map `t` is, or raises a `ConversionError` that
says what the format cannot hold.

A BigDataViewer registration maps the voxels of a view to the global
space of its dataset, through a chain of affines. Every family that has
an affine form, and a `Sequence` of affines, is reduced to one affine
between these two systems
([`affine_between`][brainhops.io.transformations.base._conversions.affine_between]),
and the registration holds it as a chain of one affine. The voxels of
another volume are paired with those of the view axis by axis, and the
global space, whose axes have no unit, takes any world, such as RAS. An
endpoint that is not known is taken to be the system of the format. A
transformation whose input is a world rather than voxels is refused,
since a world is placed in the voxels of a view only by the geometry of
the view.

A conversion takes the options of the format that do not change the
map: `metadata=`, the dataset whose XML file the registration is written
into, and `setup=` and `timepoint=`, which select its view. Without
them, the registration is written as a new XML file that holds a single
view. A `BdvViewRegistration` that is converted to its own class is
changed by the rules of any chain, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import converter, smart_replace
from brainhops.io.common.bdv._geometry import global_system, voxel_system
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    affine_between,
    format_options,
    register_converter,
    unrepresentable,
)

from ._xform import BdvViewRegistration

OPTIONS = ("metadata", "setup", "timepoint")
"""What a conversion to a registration may set: the dataset that it is
written into, and the view of that dataset."""


@register_converter(AFFINE_SOURCES, BdvViewRegistration)
def _(
    t: _xforms.Transformation, cls: tx.Type[BdvViewRegistration], **kwargs
) -> BdvViewRegistration:
    # Exactly, or raise: one affine from the voxels of the view to the
    # global space.
    cls = BdvViewRegistration
    options = format_options(t, cls, kwargs, OPTIONS)
    voxels, world = voxel_system(), global_system()
    matrix = affine_between(t, voxels, world, cls)
    if matrix.shape != (4, 4):
        raise unrepresentable(
            t,
            cls,
            f"BigDataViewer stores 3-D affines, and this one has "
            f"{len(matrix) - 1} dimensions.",
        )
    affine = _xforms.Affine(matrix=matrix[:-1], input=voxels, output=world)
    return cls(transformations=(affine,), **options)


@converter
def _(
    t: BdvViewRegistration, cls: tx.Type[BdvViewRegistration], **kwargs
) -> BdvViewRegistration:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
