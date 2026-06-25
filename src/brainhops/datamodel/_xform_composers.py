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
import typing_extensions as _tx

# core
from brainhops._core.bsplines import pull_field
from brainhops._core.backends import get_array_backend

# internals
from .transformations import (
    Transformation, _composer,
    Identity, Translation, Scaling, Permutation, Linear, Affine, Sequence,
    CoordinatesField, CartesianField, DisplacementField
)


# ----------------------------------------------------------------------
#     IDENTITY
# ----------------------------------------------------------------------


@_composer
def _(To: Identity, Ti: Transformation) -> Transformation:
    return type(Ti)(Ti, output=To.output).compute()


@_composer
def _(To: Transformation, Ti: Identity) -> Transformation:
    return type(To)(To, input=Ti.input).compute()


# ----------------------------------------------------------------------
#     SEQUENCE
# ----------------------------------------------------------------------


@_composer
def _(To: Sequence, Ti: Transformation) -> Transformation:
    return Sequence(
        transformations=[Ti] + list(To.transformations),
        input=Ti.input,
        output=To.output
    ).compute()


@_composer
def _(To: Transformation, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + [To],
        input=Ti.input,
        output=To.output
    ).compute()


@_composer
def _(To: Sequence, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + list(To.transformations),
        input=Ti.input,
        output=To.output
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
    return Affine(
        matrix=A,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Linear, Ti: Linear) -> Linear:
    return Linear(
        matrix=To.matrix @ Ti.matrix,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Permutation, Ti: Permutation) -> Permutation:
    return Permutation(
        permutation=To.permutation[Ti.permutation],
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Scaling, Ti: Scaling) -> Scaling:
    return Scaling(
        scale=To.scale * Ti.scale,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Translation, Ti: Translation) -> Translation:
    return Translation(
        translation=To.translation + Ti.translation,
        input=Ti.input,
        output=To.output
    )


_LinearIsh = _tx.Union[Linear, Scaling, Permutation]
_AffineIsh = _tx.Union[_LinearIsh, Affine, Translation]


@_composer
def _(To: _LinearIsh, Ti: _LinearIsh) -> Linear:
    return (To.to(Linear) @ Ti.to(Linear)).compute()


@_composer
def _(To: _AffineIsh, Ti: _AffineIsh) -> Affine:
    return (To.to(Affine) @ Ti.to(Affine)).compute()


# ----------------------------------------------------------------------
#     AFFINE o COORDINATES
# ----------------------------------------------------------------------


@_composer
def _(To: Translation, Ti: CoordinatesField) -> CoordinatesField:
    field = Ti.field + To.translation
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Scaling, Ti: CoordinatesField) -> CoordinatesField:
    field = Ti.field * To.scale
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Permutation, Ti: CoordinatesField) -> CoordinatesField:
    field = Ti.field[..., To.permutation]

    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Linear, Ti: CoordinatesField) -> CoordinatesField:
    field = Ti.field @ To.matrix.T
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Affine, Ti: CoordinatesField) -> CoordinatesField:
    field = Ti.field @ To.matrix[:, :-1].T + To.matrix[:, -1]
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output
    )


# ----------------------------------------------------------------------
#     AFFINE o DISPLACEMENTS
# ----------------------------------------------------------------------


@_composer
def _(To: Translation, Ti: DisplacementField) -> DisplacementField:
    field = Ti.field + To.translation
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Scaling, Ti: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = To.scale * Ti.field + (To.scale - 1) * grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Permutation, Ti: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field)[..., To.permutation] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Linear, Ti: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix.T - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output
    )


@_composer
def _(To: Affine, Ti: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix[:, :-1].T + To.matrix[:, -1] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output
    )


# ----------------------------------------------------------------------
#     DISPLACEMENTS & COORDINATES
# ----------------------------------------------------------------------


@_composer
def _(To: DisplacementField, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.to(coeff=False)
    x2 = Ti.to(CoordinatesField)
    field = pull_field(
        To.field,
        coords=x2.field,
        order=To.order,
        bound=To.bound,
        coeff=To.coeff
    ) + Ti.field
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=To.order,
        bound=To.bound,
        coeff=False
    ).to(coeff=To.coeff)


@_composer
def _(To: DisplacementField, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.to(coeff=False)
    x2 = Ti.to(CoordinatesField)
    field = pull_field(
        To.field,
        coords=x2.field,
        order=To.order,
        bound=To.bound,
        coeff=To.coeff
    ) + x2.field
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False
    ).to(coeff=Ti.coeff)


@_composer
def _(To: CoordinatesField, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.to(coeff=False)
    field = pull_field(
        To.field,
        coords=Ti.field,
        order=To.order,
        bound=To.bound,
        coeff=To.coeff
    )
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False
    ).to(coeff=coeff)
