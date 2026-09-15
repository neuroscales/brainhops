"""
This module implements function that compute the composition of two
transformations. It assumes that their input and output coordinate
systems are compatible (see _xform_adaptors for composing transformations
that do not have compatible coordinate systems). It also assumes that
transfmations are fully defined (i.e., their parameters are not `None`).

A composer takes two transformations To and Ti, and returns a
transformation T such that T = To @ Ti, i.e. such that T(x) = To(Ti(x))
for any x in the domain of Ti. The input coordinate system of T is the
input coordinate system of Ti, and the output coordinate system of T is
the output coordinate system of To.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.bsplines import pull_field
from brainhops.backends import get_array_backend

# internals
from .transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
    _compose,
    _composer,
    _Evaluated,
)

# ----------------------------------------------------------------------
#     IDENTITY
# ----------------------------------------------------------------------


@_composer
def _(To: Identity, Ti: Transformation) -> Transformation:
    return replace(Ti, output=To.output).compute()


@_composer
def _(To: Transformation, Ti: Identity) -> Transformation:
    return replace(To, input=Ti.input).compute()


# ----------------------------------------------------------------------
#     SEQUENCE
# ----------------------------------------------------------------------


@_composer
def _(To: Sequence, Ti: Transformation) -> Transformation:
    return Sequence(
        transformations=[Ti] + list(To.transformations),
        input=Ti.input,
        output=To.output,
    ).compute()


@_composer
def _(To: Transformation, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + [To],
        input=Ti.input,
        output=To.output,
    ).compute()


@_composer
def _(To: Sequence, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + list(To.transformations),
        input=Ti.input,
        output=To.output,
    ).compute()


# ----------------------------------------------------------------------
#     SAME KIND (AFFINE-ish)
# ----------------------------------------------------------------------


@_composer
def _(To: Affine, Ti: Affine) -> Affine:
    ba = get_array_backend(To.matrix)
    No = To.matrix.shape[0]
    Ni = Ti.matrix.shape[1] - 1
    A = ba.empty_like(To.matrix, shape=(No, Ni + 1))
    A[:, :-1] = To.matrix[:, :-1] @ Ti.matrix[:, :-1]
    A[:, -1:] = To.matrix[:, :-1] @ Ti.matrix[:, -1:] + To.matrix[:, -1:]
    return Affine(matrix=A, input=Ti.input, output=To.output)


@_composer
def _(To: Linear, Ti: Linear) -> Linear:
    return Linear(
        matrix=To.matrix @ Ti.matrix, input=Ti.input, output=To.output
    )


@_composer
def _(To: Permutation, Ti: Permutation) -> Permutation:
    return Permutation(
        permutation=To.permutation[Ti.permutation],
        input=Ti.input,
        output=To.output,
    )


@_composer
def _(To: Scaling, Ti: Scaling) -> Scaling:
    return Scaling(scale=To.scale * Ti.scale, input=Ti.input, output=To.output)


@_composer
def _(To: Translation, Ti: Translation) -> Translation:
    return Translation(
        translation=To.translation + Ti.translation,
        input=Ti.input,
        output=To.output,
    )


_LinearIsh = tx.Union[Linear, Scaling, Permutation]
_AffineIsh = tx.Union[_LinearIsh, Affine, Translation]


@_composer
def _(To: _LinearIsh, Ti: _LinearIsh) -> Linear:
    return (To.to(Linear) @ Ti.to(Linear)).compute()


@_composer
def _(To: _AffineIsh, Ti: _AffineIsh) -> Affine:
    return (To.to(Affine) @ Ti.to(Affine)).compute()


# ----------------------------------------------------------------------
#     AFFINE-ISH o SAMPLING DOMAIN
# ----------------------------------------------------------------------
#
# An affine applied to a sampling domain -- a query grid or an explicit
# point set that has already been evaluated -- is folded onto the
# coordinates pointwise. This is exact, because the coordinates are the
# values the affine acts on, and it is the terminal step of reslicing.
#
# There is deliberately no composer for an affine applied to a raw stored
# `CoordinatesField` or `DisplacementField`. Folding an affine into a
# stored field changes the transform once the field is interpolated, so a
# stored field that does not lead a sequence matches nothing here, raises
# `CompositionError`, and is kept as a separate step in application order.


def _apply_affine(To: _AffineIsh, coords: object) -> object:
    # Apply an affine-like transform to a coordinate array, pointwise. The
    # transform is taken through its `Affine` form, so every affine-like
    # kind is handled by one expression.
    affine = To.to(Affine)
    return coords @ affine.matrix[:, :-1].T + affine.matrix[:, -1]


@_composer
def _(To: _AffineIsh, Ti: CartesianField) -> _Evaluated:
    field = _apply_affine(To, Ti.field)
    return _Evaluated(field=field, input=Ti.input, output=To.output)


@_composer
def _(To: _AffineIsh, Ti: _Evaluated) -> _Evaluated:
    field = _apply_affine(To, Ti.field)
    return _Evaluated(field=field, input=Ti.input, output=To.output)


# ----------------------------------------------------------------------
#     STORED FIELD o SAMPLING DOMAIN
# ----------------------------------------------------------------------
#
# A stored field evaluated on a sampling domain is the one legitimate
# interpolation. The field is interpolated at the domain's coordinates,
# in the field's own frame, and the result is again a set of evaluated
# coordinates. The right operand must be a sampling domain, so these fire
# only when the field is applied to a leading grid or point set, never
# when one stored field is folded into another.

_Domain = tx.Union[CartesianField, _Evaluated]


@_composer
def _(To: DisplacementField, Ti: _Domain) -> _Evaluated:
    coords = Ti.field
    field = (
        pull_field(
            To.field,
            coords=coords,
            order=To.order,
            bound=To.bound,
            coeff=To.coeff,
        )
        + coords
    )
    return _Evaluated(field=field, input=Ti.input, output=To.output)


@_composer
def _(To: CoordinatesField, Ti: _Domain) -> _Evaluated:
    field = pull_field(
        To.field,
        coords=Ti.field,
        order=To.order,
        bound=To.bound,
        coeff=To.coeff,
    )
    return _Evaluated(field=field, input=Ti.input, output=To.output)


# ----------------------------------------------------------------------
#     SUBSPACE o SAMPLING DOMAIN
# ----------------------------------------------------------------------


@_composer
def _(To: SubspaceTransformation, Ti: _Evaluated) -> _Evaluated:
    # Apply the inner transform to the input-axis columns of the evaluated
    # coordinates, write the result onto the output-axis columns, and pass
    # every other column through unchanged.
    coords = Ti.field
    ab = get_array_backend(coords)
    in_axes = [int(a) for a in To.input_axes]
    out_axes = (
        in_axes if To.output_axes is None else [int(a) for a in To.output_axes]
    )
    sub = coords[..., in_axes]
    inner = To.transformation
    if inner is None:
        moved = sub
    else:
        moved = _compose(inner, _Evaluated(field=sub)).field
    scatter = {axis: k for k, axis in enumerate(out_axes)}
    columns = [
        moved[..., scatter[axis]] if axis in scatter else coords[..., axis]
        for axis in range(coords.shape[-1])
    ]
    field = ab.stack(columns, -1)
    return _Evaluated(field=field, input=Ti.input, output=To.output)
