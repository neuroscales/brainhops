# externals
import typing_extensions as _tx

# internals
from .transformations import (
    Transformation, _composer,
    Identity, Translation, Scaling, Permutation, Linear, Affine, Sequence
)


@_composer
def _(t1: Identity, t2: Transformation) -> Transformation:
    return type(t2)(t2, output=t1.output).compute()


@_composer
def _(t1: Transformation, t2: Identity) -> Transformation:
    return type(t1)(t1, input=t2.input).compute()


@_composer
def _(t1: Sequence, t2: Transformation) -> Transformation:
    return Sequence(
        transforms=[t2] + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Transformation, t2: Sequence) -> Transformation:
    return Sequence(
        transforms=list(t2.transforms) + [t1],
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Sequence, t2: Sequence) -> Sequence:
    return Sequence(
        transforms=list(t2.transforms) + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute()


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


