# externals
import typing_extensions as _tx

# internals
from .transformations import (
    Transformation, _composer,
    Identity, Translation, Scaling, Permutation, Linear, Affine, Sequence,
    CoordinatesField, CartesianField, DisplacementField
)
from ._utils import _pull_field


# ----------------------------------------------------------------------
#     IDENTITY
# ----------------------------------------------------------------------


@_composer
def _(t1: Identity, t2: Transformation) -> Transformation:
    return type(t2)(t2, output=t1.output).compute()


@_composer
def _(t1: Transformation, t2: Identity) -> Transformation:
    return type(t1)(t1, input=t2.input).compute()


# ----------------------------------------------------------------------
#     SEQUENCE
# ----------------------------------------------------------------------


@_composer
def _(t1: Sequence, t2: Transformation) -> Transformation:
    return Sequence(
        transformations=[t2] + list(t1.transformations),
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Transformation, t2: Sequence) -> Transformation:
    return Sequence(
        transformations=list(t2.transformations) + [t1],
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Sequence, t2: Sequence) -> Sequence:
    return Sequence(
        transformations=list(t2.transformations) + list(t1.transformations),
        input=t2.input,
        output=t1.output
    ).compute()


# ----------------------------------------------------------------------
#     SAME KIND (AFFINE-ish)
# ----------------------------------------------------------------------


@_composer
def _(t1: Affine, t2: Affine) -> Affine:
    return Affine(
        matrix=t1.matrix @ t2.matrix,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Linear, t2: Linear) -> Linear:
    return Linear(
        matrix=t1.matrix @ t2.matrix,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Permutation, t2: Permutation) -> Permutation:
    return Permutation(
        permutation=[t1.permutation[i] for i in t2.permutation],
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Scaling, t2: Scaling) -> Scaling:
    return Scaling(
        scale=[s1 * s2 for s1, s2 in zip(t1.scale, t2.scale)],
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Translation, t2: Translation) -> Translation:
    return Translation(
        translation=[t1 + t2 for t1, t2 in zip(t1.translation, t2.translation)],
        input=t2.input,
        output=t1.output
    )


_LinearIsh = _tx.Union[Linear, Scaling, Permutation]
_AffineIsh = _tx.Union[_LinearIsh, Affine, Translation]


@_composer
def _(t1: _LinearIsh, t2: _LinearIsh) -> Linear:
    return (t1.to(Linear) @ t2.to(Linear)).compute()


@_composer
def _(t1: _AffineIsh, t2: _AffineIsh) -> Affine:
    return (t1.to(Affine) @ t2.to(Affine)).compute()


# ----------------------------------------------------------------------
#     AFFINE o COORDINATES
# ----------------------------------------------------------------------


@_composer
def _(t1: Translation, t2: CoordinatesField) -> CoordinatesField:
    field = t2.field + t1.translation
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Scaling, t2: CoordinatesField) -> CoordinatesField:
    field = t2.field * t1.scale
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Permutation, t2: CoordinatesField) -> CoordinatesField:
    field = t2.field[..., t1.permutation]
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Linear, t2: CoordinatesField) -> CoordinatesField:
    field = t2.field @ t1.matrix.T
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Affine, t2: CoordinatesField) -> CoordinatesField:
    field = t2.field @ t1.matrix[:, :-1].T + t1.matrix[:, -1]
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output
    )


# ----------------------------------------------------------------------
#     AFFINE o DISPLACEMENTS
# ----------------------------------------------------------------------


@_composer
def _(t1: Translation, t2: DisplacementField) -> DisplacementField:
    field = t2.field + t1.translation
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Scaling, t2: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=t2.field.shape[:-1]).field
    field = t1.scale * t2.field + (t1.scale - 1) * grid
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Permutation, t2: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=t2.field.shape[:-1]).field
    field = (grid + t2.field)[..., t1.permutation] - grid
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Linear, t2: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=t2.field.shape[:-1]).field
    field = (grid + t2.field) @ t1.matrix.T - grid
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Affine, t2: DisplacementField) -> DisplacementField:
    grid = CartesianField(shape=t2.field.shape[:-1]).field
    field = (grid + t2.field) @ t1.matrix[:, :-1].T + t1.matrix[:, -1] - grid
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output
    )


# ----------------------------------------------------------------------
#     DISPLACEMENTS & COORDINATES
# ----------------------------------------------------------------------


@_composer
def _(t1: DisplacementField, t2: DisplacementField) -> DisplacementField:
    t2 = t2.to(coeff=False)
    x2 = t2.to(CoordinatesField)
    field = _pull_field(
        t1.field,
        coords=x2.field,
        order=t1.order,
        bound=t1.bound,
        coeff=t1.coeff
    ) + t2.field
    return DisplacementField(
        field=field,
        input=t2.input,
        output=t1.output,
        order=t1.order, 
        bound=t1.bound, 
        coeff=False
    ).to(coeff=t1.coeff)


@_composer
def _(t1: DisplacementField, t2: CoordinatesField) -> CoordinatesField:
    t2 = t2.to(coeff=False)
    x2 = t2.to(CoordinatesField)
    field = _pull_field(
        t1.field,
        coords=x2.field,
        order=t1.order,
        bound=t1.bound,
        coeff=t1.coeff
    ) + x2
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output,
        order=t2.order, 
        bound=t2.bound, 
        coeff=False
    ).to(coeff=t2.coeff)


@_composer
def _(t1: CoordinatesField, t2: CoordinatesField) -> CoordinatesField:
    coeff = t2.coeff
    t2 = t2.to(coeff=False)
    field = _pull_field(
        t1.field,
        coords=t2.field,
        order=t1.order,
        bound=t1.bound,
        coeff=t1.coeff
    )
    return CoordinatesField(
        field=field,
        input=t2.input,
        output=t1.output,
        order=t2.order, 
        bound=t2.bound, 
        coeff=False
    ).to(coeff=coeff)