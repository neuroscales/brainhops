"""
Private converters into the TrakEM2 format.

`t.to(TrakEM2PatchAffine)` and `TrakEM2PatchAffine.from_any(t)` run
these converters. Each one returns an affine that is the map `t` is, or
raises a `ConversionError` that says what the format cannot hold.

A TrakEM2 affine places a patch in the plane of its layer, so every
family that has a 2-D affine form, and a `Sequence` of such affines, is
reduced to one affine from the pixels of the patch to the plane of the
layer
([`affine_between`][brainhops.io.transformations.base._conversions.affine_between]).
The pixels of another image are paired with those of the patch axis by
axis, and the plane of the layer, whose axes have no unit, takes any
plane. An endpoint that is not known is taken to be the system of the
format.

A TrakEM2 affine is written into the project that it belongs to, since
a project cannot be made from an affine alone. The conversion therefore
takes the options `project=`, an affine read from the project, whose
metadata holds its XML file, or `metadata=`, that metadata itself, and
`patch=`, the `oid` or the title of the patch whose affine is replaced,
by default the first patch of the project. Without a project, the
converted affine can be read and compared but not written.
"""

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

from ._xform import TrakEM2PatchAffine, layer_system, patch_system

OPTIONS = ("project", "metadata", "patch")
"""What a conversion to a TrakEM2 affine takes: the project that it is
written into, and the patch of that project."""


@register_converter(AFFINE_SOURCES, TrakEM2PatchAffine)
def _(
    t: _xforms.Transformation, cls: tx.Type[TrakEM2PatchAffine], **kwargs
) -> TrakEM2PatchAffine:
    # Exactly, or raise: one 2-D affine from the pixels of the patch to the
    # plane of its layer.
    cls = TrakEM2PatchAffine
    options = dict(format_options(t, cls, kwargs, OPTIONS))
    project = options.pop("project", None)
    if project is not None:
        if "metadata" in options:
            raise unrepresentable(
                t,
                cls,
                "project= gives the metadata of the project, so metadata= "
                "is not given with it.",
            )
        if not isinstance(project, TrakEM2PatchAffine):
            raise unrepresentable(
                t,
                cls,
                f"project= is an affine read from a TrakEM2 project, not a "
                f"{type(project).__name__}.",
            )
        options["metadata"] = project.metadata
    matrix = affine_between(t, patch_system(), layer_system(), cls)
    if matrix.shape != (3, 3):
        raise unrepresentable(
            t,
            cls,
            f"a TrakEM2 affine is a 2-D affine, and this one has "
            f"{len(matrix) - 1} dimensions.",
        )
    return cls(matrix=matrix[:-1], **options)


@converter
def _(
    t: TrakEM2PatchAffine, cls: tx.Type[TrakEM2PatchAffine], **kwargs
) -> TrakEM2PatchAffine:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return convert_withlog(t, cls, "matrix", **kwargs)
