"""
Private converters into the elastix formats.

`t.to(ElastixParameterTransform)`,
`ElastixParameterTransform.from_any(t)` and
`io.save(t, "TransformParameters.0.txt")` all run these converters, and
so do the same calls for an
[`ElastixTomlTransform`][brainhops.io.transformations.elastix.ElastixTomlTransform].
elastix maps the LPS world of the fixed image to the LPS world of the
moving image, so the endpoints of a transformation are bridged to LPS
first: an affine between RAS spaces has its first two axes flipped on
both sides.

| Source                                     | Result                       |
| ------------------------------------------ | ---------------------------- |
| any affine, chain of affines               | one linear transform, exact  |
| field of degree 1 to 3 between affines     | B-spline, exact              |
| any other field, or chain with a field     | cubic B-spline, lossy        |

An affine is written as the simplest elastix transform that holds its
matrix: a `TranslationTransform` when its linear part is the identity,
an `EulerTransform` when it is a rotation in three dimensions, and an
`AffineTransform` otherwise. The center of rotation is the option
`center=`, in LPS millimetres, or the center of the fixed image given
as `reference=`, or the origin.

A field is written as a `BSplineTransform`, which is a field like any
other, interpolated with B-splines: of the degree of the field when
elastix has it (1 to 3), and cubic otherwise. The affine that the field
does not undo is written as a transform of its own, so that the
B-spline holds the displacement alone, as elastix writes its own
registrations (see
[`split_affine`][brainhops.io.transformations.base._conversions.split_affine]).
It is applied after the B-spline, which keeps the grid of the field, or
before it when ITK cannot place that grid by orthonormal directions. The
two transforms are chained as elastix chains them: the file of the
transform applied first is the initial transform of the other, named in
`InitialTransformParameterFileName`. The initial file is named after a
hash of its content, and it is written beside the file that names it
when the transformation is written to a path. A field of another degree
is evaluated at its nodes and interpolated with the cubic B-spline,
which is a loss, and so is a field that ITK cannot place on either
side: it is evaluated on the nearest grid that ITK can place.

The option `reference=`, a NIfTI image or header or the path of a NIfTI
file, gives the geometry of the fixed image (`Size`, `Spacing`, `Origin`
and `Direction`), which transformix needs to resample. A transformation
of either syntax that holds its record, and whose chain was not
assigned, is converted to the other syntax by holding the same
metadata, and its consolidation then writes the same parameter map in
the other syntax. A transformation that is converted to its own class is
changed by the rules of any chain.
"""

import hashlib
import math
import os

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.errors import ConversionError, LossyConversionError
from brainhops.io.transformations.base._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    _is_field,
    _leaves,
    affine_between,
    displacement_parts,
    displacements_of,
    field_slots,
    format_options,
    register_converter,
    sample_on_grid,
    split_affine,
    undoes,
)
from brainhops.io.transformations.base._converters import relabel
from brainhops.io.transformations.base._splines import (
    orthonormal_placement,
)
from brainhops.io.transformations.itk._common import (
    ItkTransformClass,
    _euler_to_matrix,
)
from brainhops.io.transformations.itk._systems import _make_system

from ._blocks import _BSPLINES, _column_major, _struct, transformation_to_map
from ._raw import ElastixRaw
from ._xform import (
    ElastixParameterTransform,
    ElastixTomlTransform,
    ElastixTransform,
)

_ITKT = ItkTransformClass

ELASTIX_OPTIONS = ("center", "reference")
"""What a conversion to an elastix file may set: the center of rotation
of an affine, and the fixed image."""

_SOURCES = tuple(dict.fromkeys((*AFFINE_SOURCES, *FIELD_SOURCES)))
"""The families that convert to an elastix file."""


@register_converter(_SOURCES, ElastixParameterTransform)
@register_converter(_SOURCES, ElastixTomlTransform)
def to_elastix(
    t: _xforms.Transformation, cls: tx.Type[ElastixTransform], **kwargs
) -> ElastixTransform:
    """
    Return `t` as an elastix transform parameter file.

    Parameters
    ----------
    t : Transformation
        The transformation, from the fixed world to the moving world.
    cls : type
        The format, which is one of the two elastix syntaxes.
    **kwargs : Any
        The options of the conversion, `center=` and `reference=`.

    Returns
    -------
    ElastixTransform
        The transformation, which holds the record of its file.

    Raises
    ------
    ConversionError
        If the endpoints of `t` cannot be bridged to LPS, or `t` holds
        neither an affine nor a field.
    LossyConversionError
        If the B-spline only approximates the field of `t`. It carries
        the result.
    """
    syntax = "toml" if issubclass(cls, ElastixTomlTransform) else "text"
    options = format_options(t, cls, dict(kwargs), ELASTIX_OPTIONS)
    ndim = _ndim(t)
    world = _make_system(ndim)
    base, fixed_center = _fixed_map(options.get("reference"), ndim)
    center = options.get("center")
    center = fixed_center if center is None else center
    center = np.zeros(ndim) if center is None else np.asarray(center, float)
    loss = None
    try:
        matrix = affine_between(t, world, world, cls)
    except ConversionError:
        matrix = None
    if matrix is not None:
        blocks = [_affine_block(matrix, center)]
    else:
        blocks, loss = _field_blocks(t, cls, world, center)
    if len(blocks) == 1 and not base:
        # One transform is held as the chain of the format, which its
        # consolidation encodes, and the metadata of `t` rides along.
        result = cls(transformations=tuple(blocks), metadata=t.metadata)
    else:
        result = _from_blocks(cls, syntax, blocks, base)
    if loss is not None:
        raise LossyConversionError(
            f"The {cls.__name__} that this transformation converts to is "
            f"not the same map: {loss} Pass lossy=True to accept it.",
            result=result,
        )
    return result


@converter
def _(
    t: ElastixParameterTransform,
    cls: tx.Type[ElastixParameterTransform],
    **kwargs,
) -> ElastixParameterTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: ElastixTomlTransform,
    cls: tx.Type[ElastixTomlTransform],
    **kwargs,
) -> ElastixTomlTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: ElastixTomlTransform,
    cls: tx.Type[ElastixParameterTransform],
    **kwargs,
) -> ElastixParameterTransform:
    # The text syntax holds the parameter map of a TOML file.
    return _other_syntax(t, cls, **kwargs)


@converter
def _(
    t: ElastixParameterTransform,
    cls: tx.Type[ElastixTomlTransform],
    **kwargs,
) -> ElastixTomlTransform:
    # The TOML syntax holds the parameter map of a text file.
    return _other_syntax(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------


def _affine_block(matrix: np.ndarray, center: np.ndarray) -> tx.Any:
    """
    Return the simplest ITK block that holds a homogeneous affine.

    The linear blocks map `x` to `M (x - c) + c + t`, so the translation
    of the block is the offset of the affine minus `c - M c`.
    """
    ndim = matrix.shape[0] - 1
    linear = matrix[:ndim, :ndim]
    offset = matrix[:ndim, ndim]
    if np.allclose(linear, np.eye(ndim), rtol=0, atol=1e-12):
        return _struct(_ITKT.TranslationTransform, ndim, offset)
    translation = offset - center + linear @ center
    if ndim == 3:
        angles = _euler_angles(linear)
        if angles is not None:
            params = np.concatenate([angles, translation])
            fixed = np.append(center, 1.0)
            return _struct(_ITKT.Euler3DTransform, 3, params, fixed)
    params = np.concatenate([linear.ravel(), translation])
    return _struct(_ITKT.AffineTransform, ndim, params, center)


def _euler_angles(linear: np.ndarray) -> tx.Optional[np.ndarray]:
    """
    Return the angles of a rotation, composed in the order Z, Y and X.

    The angles are those of an ITK `Euler3DTransform` whose `ComputeZYX`
    is true, which computes `Rz @ Ry @ Rx`. `None` is returned when the
    matrix is not a rotation, or when the angles do not give back the
    matrix to the rounding of the float arithmetic.
    """
    if not np.allclose(linear.T @ linear, np.eye(3), rtol=0, atol=1e-10):
        return None
    if np.linalg.det(linear) <= 0:
        return None
    ax = math.atan2(linear[2, 1], linear[2, 2])
    ay = math.asin(max(-1.0, min(1.0, -linear[2, 0])))
    az = math.atan2(linear[1, 0], linear[0, 0])
    angles = np.array([ax, ay, az])
    if not np.allclose(
        _euler_to_matrix(angles, True), linear, rtol=0, atol=1e-12
    ):
        return None
    return angles


# ----------------------------------------------------------------------
#   FIELDS
# ----------------------------------------------------------------------


def _field_blocks(
    t: _xforms.Transformation,
    cls: type,
    world: tx.Any,
    center: np.ndarray,
) -> tx.Tuple[tx.List[tx.Any], tx.Optional[str]]:
    """
    Return the blocks that hold the field of `t`, in the order of
    application, and what they lose.

    The affines around the field are written as transforms of their own,
    chained to the B-spline as elastix chains transforms, so that the
    B-spline holds the displacement alone. The field keeps the grid on
    which it is placed, with the affines that come before and after it on
    either side (see
    [`field_slots`][brainhops.io.transformations.base._conversions.field_slots]).
    ITK places a grid by orthonormal directions, so when it cannot place
    that grid, the affine that the field does not undo is applied after
    the B-spline, which is placed by the affines before it, or before the
    B-spline, which is then placed by the affines after it (see
    [`split_affine`][brainhops.io.transformations.base._conversions.split_affine]).
    The field is held as a B-spline of its own
    degree when elastix has it, and as a cubic B-spline otherwise. A
    field that ITK cannot place on either side is evaluated on the
    nearest grid that it can place, which loses.
    """
    degree = _spline_degree(t)
    first, field, last = field_slots(t, world, cls)
    parts = displacement_parts(field, world, cls, degree=degree)
    if orthonormal_placement(parts.grid2world) is not None:
        data = parts.field
        coefficients = data.coefficients if degree > 1 else data.values
        blocks = [_bspline_block(parts.grid2world, coefficients, degree)]
        if first is not None and not undoes(first, np.eye(len(first))):
            blocks.insert(0, _affine_block(first, center))
        if last is not None and not undoes(last, np.eye(len(last))):
            blocks.append(_affine_block(last, center))
        return blocks, parts.loss
    for side in ("after", "before"):
        affine, field = split_affine(t, world, world, cls, side)
        parts = displacement_parts(field, world, cls, degree=degree)
        if orthonormal_placement(parts.grid2world) is not None:
            break
    else:
        grid2world = _orthogonalized(parts.grid2world)
        shape = tuple(int(n) for n in parts.field.data.shape[:-1])
        parts = displacements_of(
            sample_on_grid(
                field,
                np.linalg.inv(grid2world),
                shape,
                world,
                world,
                cls,
                degree,
            )
        )
    data = parts.field
    coefficients = data.coefficients if degree > 1 else data.values
    blocks = [_bspline_block(parts.grid2world, coefficients, degree)]
    if not np.allclose(affine, np.eye(len(affine))):
        block = _affine_block(affine, center)
        blocks = blocks + [block] if side == "after" else [block, *blocks]
    return blocks, parts.loss


def _spline_degree(t: _xforms.Transformation) -> int:
    """
    Return the degree of the B-spline that holds the field of `t`.

    It is the degree of the field when elastix has a B-spline of that
    degree, and 3 otherwise.
    """
    for leaf in _leaves(t):
        if _is_field(leaf):
            degree = int(leaf.degree)
            return degree if degree in _BSPLINES else 3
    return 3


def _bspline_block(
    grid2world: np.ndarray, coefficients: tx.Any, degree: int
) -> tx.Any:
    """
    Return the ITK B-spline block of coefficients on a grid.

    ITK stores the coefficients of the displacement in the world, one
    image per axis, each with x varying fastest, and places the grid by
    its size, origin, spacing and directions.
    """
    direction, spacing, origin = orthonormal_placement(grid2world)
    ndim = len(spacing)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    world = coefficients @ grid2world[:ndim, :ndim].T
    params = np.concatenate(
        [world[..., k].ravel(order="F") for k in range(ndim)]
    )
    size = np.asarray(coefficients.shape[:-1], dtype=np.float64)
    fixed = np.concatenate([size, origin, spacing, direction.ravel()])
    return _struct(
        _ITKT.BSplineTransform, ndim, params, fixed, cls=_BSPLINES[degree]
    )


def _orthogonalized(grid2world: np.ndarray) -> np.ndarray:
    """
    Return the affine of a grid with its directions made orthogonal.

    ITK places a grid by orthonormal directions, so a grid whose columns
    are not orthogonal is replaced by the nearest one that has the same
    origin and the same distance between its nodes.
    """
    if orthonormal_placement(grid2world) is not None:
        return grid2world
    ndim = grid2world.shape[0] - 1
    linear = grid2world[:ndim, :ndim]
    u, _, vt = np.linalg.svd(linear)
    result = grid2world.copy()
    result[:ndim, :ndim] = (u @ vt) * np.linalg.norm(linear, axis=0)
    return result


# ----------------------------------------------------------------------
#   RECORDS
# ----------------------------------------------------------------------


def _from_blocks(
    cls: type,
    syntax: str,
    blocks: tx.Sequence[tx.Any],
    base: tx.Dict[str, tx.Any],
) -> ElastixTransform:
    """
    Return the transformation of a chain of blocks, in its records.

    Each block is the transform of one file, and the file of each block
    is the initial transform of the file of the next one.
    """
    raw = None
    for block in blocks:
        pmap = transformation_to_map(block, base)
        if raw is not None:
            pmap["InitialTransformParameterFileName"] = (_initial_name(raw),)
        raw = ElastixRaw(parameters=pmap, syntax=syntax, initial=raw)
    return cls.from_raw(raw)


def _initial_name(raw: ElastixRaw) -> str:
    """
    Return the name of the file of an initial transform.

    A name that depends on the content alone never names two different
    files, so the files of two chains written in the same directory do
    not replace each other.
    """
    digest = hashlib.sha1(raw.to_bytes()).hexdigest()[:12]  # noqa: S324
    extension = ".toml" if raw.syntax == "toml" else ".txt"
    return f"TransformParameters.initial-{digest}{extension}"


def _fixed_map(
    reference: tx.Any, ndim: int
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Optional[np.ndarray]]:
    """
    Return the parameters of the fixed image, and its center in LPS.

    Without a reference, there are no parameters and no center.
    """
    if reference is None:
        return {}, None
    if isinstance(reference, (str, os.PathLike)):
        import nibabel as nb

        reference = nb.load(os.fspath(reference)).header
    from brainhops.io.transformations.fsl._affines import (
        _best_affine,
        _shape,
    )

    vox2ras = _best_affine(reference)
    shape = _shape(reference)[:ndim]
    flip = np.diag([-1.0, -1.0, *([1.0] * (ndim - 2)), 1.0])
    vox2lps = flip @ vox2ras
    placement = orthonormal_placement(vox2lps)
    if placement is None:
        placement = orthonormal_placement(_orthogonalized(vox2lps))
    direction, spacing, origin = placement
    center = vox2lps[:ndim, :ndim] @ ((np.asarray(shape) - 1) / 2) + origin
    pmap = {
        "Size": tuple(int(n) for n in shape),
        "Index": (0,) * ndim,
        "Spacing": tuple(float(s) for s in spacing),
        "Origin": tuple(float(o) for o in origin),
        "Direction": _column_major(direction),
        "UseDirectionCosines": ("true",),
    }
    return pmap, center


def _ndim(t: _xforms.Transformation) -> int:
    """Return the number of dimensions of `t`, three when not known."""
    for system in (t.input, t.output):
        axes = getattr(system, "axes", None)
        if axes:
            return len(axes)
    return 3


def _other_syntax(
    t: ElastixTransform, cls: tx.Type[ElastixTransform], **kwargs
) -> ElastixTransform:
    # A transformation that holds its record, and whose chain was not
    # assigned, keeps its metadata, and its consolidation writes the
    # parameter map in the syntax of `cls`. Any other one is converted
    # as any other chain is.
    if t.metadata is None or t._transformations is not None:
        if kwargs:
            return relabel(t, cls, **kwargs)
        return to_elastix(t, cls)
    return smart_replace(t, cls, **kwargs)
