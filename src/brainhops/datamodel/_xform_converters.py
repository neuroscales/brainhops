from types import ModuleType

from .transforms import (
    Identity, Translation, Scale, Permutation, Linear, Affine,
    DisplacementField, CoordinatesField, LossyConversionError
)
from .transforms import _get_ndim, _converter, _to
from .typing import ArrayProtocol, npt, cpt, dkt


def _get_array_package(x: ArrayProtocol) -> ModuleType:
    if npt.np and isinstance(x, npt.np.ndarray):
        return npt.np
    elif cpt.cp and isinstance(x, cpt.cp.ndarray):
        return cpt.cp
    elif dkt.dk and isinstance(x, dkt.dk.ndarray):
        return dkt.dk
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")


# ----------------------------------------------------------------------
#   LOSSLESS
# ----------------------------------------------------------------------


@_converter
def _(t: Identity) -> Translation:
    ndim = _get_ndim(t)
    return Translation(
        translation=[0.0] * ndim if ndim is not None else None,
        input=t.input,
        output=t.output
    )


@_converter
def _(t: Identity) -> Scale:
    ndim = _get_ndim(t)
    return Scale(
        scale=[1.0] * ndim if ndim is not None else None,
        input=t.input,
        output=t.output
    )


@_converter
def _(t: Identity) -> Permutation:
    ndim = _get_ndim(t)
    return Permutation(
        permutation=range(ndim) if ndim is not None else None,
        input=t.input,
        output=t.output
    )


@_converter
def _(t: Identity) -> Linear:
    ndim = _get_ndim(t)
    return Linear(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim)] 
            for i in range(ndim)
        ] if ndim is not None else None,
        input=t.input,
        output=t.output
    )


@_converter
def _(t: Identity) -> Affine:
    ndim = _get_ndim(t)
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim+1)] 
            for i in range(ndim)
        ] if ndim is not None else None,
        input=t.input,
        output=t.output
    )


@_converter
def _(t: Translation) -> Affine:
    if t.translation is None:
        return _to(Identity(input=t.input, output=t.output), Affine)
    ndim = max(_get_ndim(t, 0), len(t.translation))
    u = Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim+1)] 
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output
    )
    u.matrix[:, -1] = t.translation
    return u


@_converter
def _(t: Scale) -> Linear:
    if t.scale is None:
        return _to(Identity(input=t.input, output=t.output), Linear)
    ndim = max(_get_ndim(t, 0), len(t.scale))
    u = Linear(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim)] 
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output
    )
    u.matrix[range(ndim), range(ndim)] = t.scale
    return u


@_converter
def _(t: Permutation) -> Linear:
    if t.permutation is None:
        return _to(Identity(input=t.input, output=t.output), Linear)
    ndim = max(_get_ndim(t, 0), len(t.permutation))
    u = Linear(
        matrix=[
            [1.0 if j == t.permutation[i] else 0.0 for j in range(ndim)] 
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output
    )
    return u


@_converter
def _(t: Linear) -> Affine:
    if t.matrix is None:
        return _to(Identity(input=t.input, output=t.output), Affine)
    ndim = len(t.matrix)
    u = Affine(
        matrix=[
            [t.matrix[i][j] if j < ndim else 0.0 for j in range(ndim+1)] 
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output
    )
    return u


@_converter
def _(t: DisplacementField) -> CoordinatesField:
    if t.field is None:
        return CoordinatesField(
            coordinates=None,
            input=t.input,
            output=t.output
        )
    xp = _get_array_package(t.field)
    g = xp.meshgrid(*(xp.arange(s) for s in t.field.shape[:-1]), indexing='ij')
    g = xp.stack(g, axis=-1)
    return CoordinatesField(
        field=t.field + g,
        input=t.input,
        output=t.output
    )


# ----------------------------------------------------------------------
#   LOSSY
# ----------------------------------------------------------------------


@_converter
def _(t: Translation) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if (
        t.translation is not None and 
        any(x != 0.0 for x in t.translation)
    ):
        raise LossyConversionError(result=u)
    return u
    

@_converter
def _(t: Scale) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if (
        t.scale is not None and 
        any(s != 1.0 for s in t.scale)
    ):
        raise LossyConversionError(result=u)
    return u
    

@_converter
def _(t: Permutation) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if (
        t.permutation is not None and 
        any(i != p for i, p in enumerate(t.permutation))
    ):
        raise LossyConversionError(result=u)
    return u


@_converter
def _(t: Linear) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if (
                    (i == j and t.matrix[i][j] != 1.0) or 
                    (i != j and t.matrix[i][j] != 0.0)
                ):
                    raise LossyConversionError(result=u)
    return u


@_converter
def _(t: Linear) -> Permutation:
    if t.matrix is None:
        return _to(_to(t, Identity), Permutation)
    nx = _get_array_package(t.matrix)
    matrix = nx.round(nx.abs(t.matrix))
    permutation = nx.argmax(matrix, axis=1)
    u = Permutation(input=t.input, output=t.output, permutation=permutation)
    if t.matrix is not None:
        for row in t.matrix:
            if any([x not in (1, 0) for x in row]) or row.sum() != 1:
                raise LossyConversionError(result=u)
        for col in t.matrix.T:
            if any([x not in (1, 0) for x in col]) or col.sum() != 1:
                raise LossyConversionError(result=u)
    return u


@_converter
def _(t: Linear) -> Scale:
    if t.matrix is None:
        return _to(_to(t, Identity), Scale)
    nx = _get_array_package(t.matrix)
    scale = nx.diagonal(t.matrix)  # TODO: use svd instead?
    u = Scale(input=t.input, output=t.output, scale=scale)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if i != j and t.matrix[i][j] != 0:
                    raise LossyConversionError(result=u)
    return u


@_converter
def _(t: Affine) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if (
                    (i == j and t.matrix[i][j] != 1.0) or 
                    (i != j and t.matrix[i][j] != 0.0)
                ):
                    raise LossyConversionError(result=u)
    return u


@_converter
def _(t: Affine) -> Linear:
    if t.matrix is None:
        return _to(_to(t, Identity), Linear)
    matrix = t.matrix[:, :-1]
    u = Linear(input=t.input, output=t.output, matrix=matrix)
    if t.matrix[:, -1].any():
        raise LossyConversionError(result=u)
    return u


@_converter
def _(t: DisplacementField) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.field is not None:
        if t.field.any():
            raise LossyConversionError(result=u)
    return u



# ----------------------------------------------------------------------
#   MULTI-HOPS
# ----------------------------------------------------------------------


def _make_converter_chain(*types):

    T0, TN = types[0], types[-1]

    @_converter
    def _(t: T0) -> TN:
        for T1 in types[1:]:
            t = _to(t, T1)
        return t


# Lossless
_make_converter_chain(Scale, Linear, Affine)
_make_converter_chain(Permutation, Linear, Affine)

# Lossy
_make_converter_chain(Affine, Linear, Scale)
_make_converter_chain(Affine, Linear, Permutation)
_make_converter_chain(Linear, Identity, Translation)
_make_converter_chain(Scale, Identity, Translation)
_make_converter_chain(Permutation, Identity, Translation)