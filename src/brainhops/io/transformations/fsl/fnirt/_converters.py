"""
Private converters into the FNIRT format.

`t.to(FnirtWarpField, reference=..., moving=...)`,
`FnirtWarpField.from_any(t, reference=..., moving=...)` and
`io.save(t, "warp.nii.gz", hint="fnirt", conversion={...})` all run
these converters. A FNIRT warp maps the scaled millimetres of a
reference image to those of a moving image, so the conversion needs the
geometry of both images. They are the options `reference=` and
`moving=` (or `ref=`, `mov=` and `src=`), and a source that holds them
itself, such as a FLIRT transform or another FNIRT warp, gives them when
they are not passed. Without them, the converter refuses and names the
missing option. A path is read as the header of a NIfTI file.

| Option             | Result                                       |
| ------------------ | -------------------------------------------- |
| none               | deformation field on the reference grid      |
| `knot_spacing=`    | cubic coefficient field on a knot grid       |

A deformation field (`--fout`) stores the relative displacement of each
voxel of the reference, in scaled millimetres. The transformation is
evaluated at these voxels, so the conversion is exact when the
transformation is an affine, which linear interpolation reproduces, or
a linear field on the grid of the reference. Any other transformation
is sampled, which loses: the converter raises a `LossyConversionError`
that carries the result.

A coefficient field (`--cout`) stores a cubic B-spline whose knots lie
every `knot_spacing` voxels of the reference, as FNIRT places them, and
an initial affine. The affine that the field of the transformation does
not undo is stored in the initial affine, as FNIRT stores it, so that
the spline holds the displacement alone (see
[`split_affine`][brainhops.io.transformations.base._conversions.split_affine]),
and an affine alone is held exactly by an initial affine and a spline
of zeros. The field is evaluated at the knots as a cubic B-spline (see
[`sample_on_grid`][brainhops.io.transformations.base._conversions.sample_on_grid]),
and the values are converted to coefficients. A cubic B-spline on that
knot grid is held exactly, and any other field is evaluated at the
knots and interpolated, which loses.
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.backends import array_backend

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.errors import ConversionError, LossyConversionError

# io
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._raw import NiftiRaw
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    affine_between,
    apply_affine,
    format_options,
    register_converter,
    sample_on_grid,
    split_affine,
    unrepresentable,
)
from brainhops.io.transformations.base.fields import voxel_grid_coordinates

# this format
from .._affines import _ImageGeometry
from ._base import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSL_FNIRT_DISPLACEMENT_FIELD,
    FnirtWarpField,
    _anchor_offsets,
    _read_image,
)

RAS = _systems.RASmm()
"""The world in which a FNIRT warp places both images."""

IMAGE_OPTIONS = {
    "reference": ("reference", "ref"),
    "moving": ("moving", "mov", "src"),
}
"""The names under which a conversion is given each image."""

FNIRT_OPTIONS = (*(n for names in IMAGE_OPTIONS.values() for n in names),)
FNIRT_OPTIONS += ("knot_spacing",)
"""What a conversion to a FNIRT warp may set: the two images, and the
knot spacing of a coefficient field."""


@register_converter(
    tuple(dict.fromkeys((*AFFINE_SOURCES, *FIELD_SOURCES))), FnirtWarpField
)
def to_fnirt(
    t: _xforms.Transformation, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    """
    Return `t` as a FNIRT warp between two images.

    Parameters
    ----------
    t : Transformation
        The transformation, from the RAS world of the reference to the
        RAS world of the moving image.
    cls : type
        The format, `FnirtWarpField`.
    **kwargs : Any
        The options of the conversion, listed in `FNIRT_OPTIONS`.

    Returns
    -------
    FnirtWarpField
        The warp.

    Raises
    ------
    ConversionError
        If an image is missing, or `t` cannot be evaluated.
    LossyConversionError
        If the warp only approximates `t`. It carries the warp.
    """
    cls = FnirtWarpField
    options = format_options(t, cls, dict(kwargs), FNIRT_OPTIONS)
    images = {
        name: _option(options, names, getattr(t, name, None))
        for name, names in IMAGE_OPTIONS.items()
    }
    missing = [name for name, image in images.items() if image is None]
    if missing:
        names = " and ".join(f"{name}=" for name in missing)
        raise unrepresentable(
            t,
            cls,
            f"a FNIRT warp is stored on the grid of its reference image, in "
            f"the scaled millimetres of its reference and moving images, "
            f"and the conversion is not given {names}. Pass the images as "
            f"options, as in t.to(FnirtWarpField, reference=..., "
            f"moving=...), or in conversion= of io.save.",
        )
    reference = _read_image(images["reference"])
    moving = _read_image(images["moving"])
    spacing = options.get("knot_spacing")
    if spacing is None:
        return _deformation(t, cls, reference, moving)
    return _coefficients(t, cls, reference, moving, spacing)


@converter
def _(
    t: FnirtWarpField, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # A warp converted to its own class is changed by the rules of any
    # chain, which keep its type, unless it is asked for a coefficient
    # field, which is a conversion.
    if "knot_spacing" in kwargs:
        return to_fnirt(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   DEFORMATION FIELDS
# ----------------------------------------------------------------------


def _deformation(
    t: _xforms.Transformation,
    cls: type,
    reference: tx.Any,
    moving: tx.Any,
) -> FnirtWarpField:
    # The relative displacement, in scaled millimetres, of each voxel of
    # the reference, which keeps the backend of the field of `t`.
    ref = _ImageGeometry(reference)
    mov = _ImageGeometry(moving)
    shape = tuple(int(n) for n in ref.shape[:3])
    parts = sample_on_grid(t, ref.ras2vox, shape, RAS, RAS, cls)
    positions = parts.field.data
    backend = array_backend(positions)
    scaled = apply_affine(mov.ras2fsl, positions)
    nodes = voxel_grid_coordinates(shape, ref.vox2fsl, backend)
    displacements = scaled - backend.asarray(nodes, dtype=scaled.dtype)
    header = nb.Nifti1Header()
    header.set_data_shape((*shape, 3))
    header.set_data_dtype(np.float32)
    header.set_sform(ref.vox2ras, 1)
    header.set_qform(ref.vox2ras, 1)
    header.set_intent(FSL_FNIRT_DISPLACEMENT_FIELD)
    result = cls(
        metadata=NiftiMetadata.from_raw(NiftiRaw(header=header)),
        data=displacements,
        reference=reference,
        moving=moving,
        deformation_type="relative",
    )
    loss = None if _is_affine(t, cls) else parts.loss
    return _held(result, loss)


# ----------------------------------------------------------------------
#   COEFFICIENT FIELDS
# ----------------------------------------------------------------------


def _coefficients(
    t: _xforms.Transformation,
    cls: type,
    reference: tx.Any,
    moving: tx.Any,
    spacing: tx.Any,
) -> FnirtWarpField:
    # A cubic B-spline on the knot grid that FNIRT places over the
    # reference, with an initial affine.
    ref = _ImageGeometry(reference)
    mov = _ImageGeometry(moving)
    shape = tuple(int(n) for n in ref.shape[:3])
    spacing = np.broadcast_to(np.asarray(spacing), (3,))
    if np.any(spacing != np.round(spacing)) or np.any(spacing < 1):
        raise unrepresentable(
            t,
            cls,
            f"FNIRT places its knots every whole number of voxels of the "
            f"reference, and knot_spacing={spacing.tolist()} was asked.",
        )
    spacing = spacing.astype(int)
    sizes = tuple(
        n if k == 1 else int(np.ceil((n + 1) / k)) + 2
        for n, k in zip(shape, spacing)
    )
    # The affine from the scaled millimetres of the reference to the
    # knot grid, as `FnirtWarpField._ref_to_field` places it.
    fsl2grid = np.diag(np.concatenate([1.0 / (ref.pixdim * spacing), [1.0]]))
    fsl2grid[:3, 3] += _anchor_offsets(3, spacing.astype(float))
    ras2grid = fsl2grid @ ref.ras2fsl
    # FNIRT maps a reference position `x`, in scaled millimetres, to
    # `M x + D(x)` in the source, where `M` is the inverse of the sform
    # and `D` is the B-spline. The affine that the field does not undo is
    # stored in `M`, as FNIRT stores its initial affine, so that the
    # spline holds the displacement alone (see `split_affine`).
    affine, field = split_affine(t, RAS, RAS, cls, "after")
    to_source = mov.ras2fsl @ affine @ ref.fsl2ras
    if field is None:
        values = np.zeros((*sizes, 3))
        loss = None
    else:
        # The field is evaluated at the knots as a cubic B-spline, which
        # keeps the coefficients of a cubic B-spline already on this knot
        # grid. Its displacement, in the scaled millimetres of the
        # reference, is carried into the source by the linear part of
        # `M`. Both maps are affine, so they act on the coefficients of
        # the B-spline as on its values.
        parts = sample_on_grid(field, ras2grid, sizes, RAS, RAS, cls, degree=3)
        knots = voxel_grid_coordinates(sizes, np.linalg.inv(fsl2grid))
        moved = apply_affine(ref.ras2fsl, parts.field.data) - knots
        displacement = _xforms.DisplacementField(
            data=apply_affine(_linear(to_source), moved),
            degree=3,
            store=parts.field.store,
            bound="constant",
        )
        values = displacement.coefficients
        loss = parts.loss
    initial = np.linalg.inv(to_source)
    header = nb.Nifti1Header()
    header.set_data_shape((*sizes, 3))
    header.set_data_dtype(np.float32)
    header.set_sform(initial, 1)
    # FNIRT stores the knot spacing in the pixel sizes, and the shape of
    # the reference in the offsets of the qform.
    qform = np.diag([*spacing.astype(float), 1.0])
    qform[:3, 3] = shape
    header.set_qform(qform, 1)
    header.set_intent(FSL_CUBIC_SPLINE_COEFFICIENTS)
    for i, size in enumerate(ref.pixdim[:3], start=1):
        header[f"intent_p{i}"] = float(size)
    result = cls(
        metadata=NiftiMetadata.from_raw(NiftiRaw(header=header)),
        data=values,
        reference=reference,
        moving=moving,
    )
    return _held(result, loss)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _option(
    options: tx.Dict[str, tx.Any], names: tx.Sequence[str], default: tx.Any
) -> tx.Any:
    # The value given under one of the names of an option, or `default`.
    for name in names:
        if options.get(name) is not None:
            return options[name]
    return default


def _is_affine(t: _xforms.Transformation, cls: type) -> bool:
    # Whether `t` is an affine, which linear interpolation reproduces.
    try:
        affine_between(t, RAS, RAS, cls)
    except ConversionError:
        return False
    return True


def _linear(matrix: np.ndarray) -> np.ndarray:
    # The linear part of a homogeneous affine, as a homogeneous affine.
    linear = np.eye(len(matrix))
    linear[:-1, :-1] = matrix[:-1, :-1]
    return linear


def _held(result: FnirtWarpField, loss: tx.Optional[str]) -> FnirtWarpField:
    # The warp, or the error that carries it when it approximates.
    if loss is not None:
        raise LossyConversionError(
            f"The FnirtWarpField that this transformation converts to is "
            f"not the same map: {loss} Pass on_loss='ignore' to accept it.",
            result=result,
        )
    return result
