# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops.backends import get_array_backend
from brainhops.datamodel import hierarchy

# locals
from .base import Transformation
from .check import is_kind
from .concrete import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from .convert import convert, converter
from .errors import ConversionError, LossyConversionError
from .meta import SubspaceTransformation
from .utils import get_ndim


def smart_replace(t: Transformation, **kwargs) -> Transformation:
    return replace(t, **kwargs) if kwargs else t


# ----------------------------------------------------------------------
#   SAME TYPE
# ----------------------------------------------------------------------


@converter(Identity, Identity)
@converter(Translation, Translation)
@converter(Scaling, Scaling)
@converter(Permutation, Permutation)
@converter(Rotation, Rotation)
@converter(Affine, Affine)
@converter(Linear, Linear)
@converter
def _(t: Transformation, **kwargs) -> Transformation:
    # A same-type conversion with no overrides is a pass-through; with
    # overrides it rebuilds the transform of the same type, applying the
    # named fields on top of the existing ones.
    return smart_replace(t, **kwargs)


@converter
def _(t: DisplacementField, **kwargs) -> DisplacementField:
    if "field" not in kwargs and t.field is not None:
        if t.coeff and not kwargs.get("coeff", t.coeff):
            field = coeff2value_field(t.field, order=t.order, bound=t.bound)
            kwargs["field"] = field
        elif not t.coeff and kwargs.get("coeff", t.coeff):
            order = kwargs.get("order", t.order)
            bound = kwargs.get("bound", t.bound)
            field = value2coeff_field(t.field, order=order, bound=bound)
            kwargs["field"] = field
    return smart_replace(t, **kwargs)


@converter
def _(t: CoordinatesField, **kwargs) -> CoordinatesField:
    if "field" not in kwargs and t.field is not None:
        if t.coeff and not kwargs.get("coeff", t.coeff):
            field = coeff2value_field(t.field, order=t.order, bound=t.bound)
            kwargs["field"] = field
        if not t.coeff and kwargs.get("coeff", t.coeff):
            order = kwargs.get("order", t.order)
            bound = kwargs.get("bound", t.bound)
            field = value2coeff_field(t.field, order=order, bound=bound)
            kwargs["field"] = field
    return smart_replace(t, **kwargs)


@converter
def _(t: CartesianField, **kwargs) -> CartesianField:
    # A CartesianField generates its `field` on demand from `shape`, so
    # `field` is not a constructor argument here. A rebuild carries
    # `shape` over and lets the new instance regenerate the field, and any
    # `field` override is dropped.
    kwargs.pop("field", None)
    return smart_replace(t, **kwargs)


# ----------------------------------------------------------------------
#   LOSSLESS
# ----------------------------------------------------------------------


@converter
def _(t: Identity, **kwargs) -> Translation:
    ndim = get_ndim(t)
    return Translation(
        translation=[0.0] * ndim if ndim is not None else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Identity, **kwargs) -> Scaling:
    ndim = get_ndim(t)
    return Scaling(
        scale=[1.0] * ndim if ndim is not None else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Identity, **kwargs) -> Permutation:
    ndim = get_ndim(t)
    return Permutation(
        permutation=range(ndim) if ndim is not None else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Identity, **kwargs) -> Linear:
    ndim = get_ndim(t)
    return Linear(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim)] for i in range(ndim)
        ]
        if ndim is not None
        else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Identity, **kwargs) -> Affine:
    ndim = get_ndim(t)
    return Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ]
        if ndim is not None
        else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Translation) -> Affine:
    if t.translation is None:
        return convert(Identity(input=t.input, output=t.output), Affine)
    ndim = max(get_ndim(t, 0), len(t.translation))
    u = Affine(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output,
    )
    u.matrix[:, -1] = t.translation
    return u


@converter
def _(t: Scaling) -> Linear:
    if t.scale is None:
        return convert(Identity(input=t.input, output=t.output), Linear)
    ndim = max(get_ndim(t, 0), len(t.scale))
    u = Linear(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim)] for i in range(ndim)
        ],
        input=t.input,
        output=t.output,
    )
    u.matrix[range(ndim), range(ndim)] = t.scale
    return u


@converter
def _(t: Permutation) -> Linear:
    if t.permutation is None:
        return convert(Identity(input=t.input, output=t.output), Linear)
    ndim = max(get_ndim(t, 0), len(t.permutation))
    u = Linear(
        matrix=[
            [1.0 if j == t.permutation[i] else 0.0 for j in range(ndim)]
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output,
    )
    return u


@converter
def _(t: Linear, **kwargs) -> Rotation:
    # A `Rotation` stores the same matrix as a `Linear`, but promises it is
    # orthogonal with determinant +1 -- which is what lets it invert by
    # transposing instead of solving. The promise is checked, so converting
    # a matrix that does not keep it is lossy.
    if t.matrix is None:
        return convert(Identity(input=t.input, output=t.output), Rotation)
    kwargs.setdefault("matrix", t.matrix)
    kwargs.setdefault("input", t.input)
    kwargs.setdefault("output", t.output)
    u = Rotation(**kwargs)
    if not is_kind(
        t, hierarchy.SpecialOrthogonalTransformation, compute=True
    ):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Identity, **kwargs) -> Rotation:
    ndim = get_ndim(t)
    return Rotation(
        matrix=[
            [1.0 if i == j else 0.0 for j in range(ndim)] for i in range(ndim)
        ]
        if ndim is not None
        else None,
        input=t.input,
        output=t.output,
    )


@converter
def _(t: Linear) -> Affine:
    if t.matrix is None:
        return convert(Identity(input=t.input, output=t.output), Affine)
    ndim = len(t.matrix)
    u = Affine(
        matrix=[
            [t.matrix[i][j] if j < ndim else 0.0 for j in range(ndim + 1)]
            for i in range(ndim)
        ],
        input=t.input,
        output=t.output,
    )
    return u


@converter
def _(t: SubspaceTransformation) -> Affine:
    # Embed the inner transform, reduced to an affine, into a full-size
    # affine over the whole space. The inner transform occupies the rows
    # and columns of the acted-on axes. Every other axis passes through as
    # the identity, mapping each input pass-through axis to the output
    # pass-through axis in the same position in the ordering.
    if (
        t.input is None
        or t.output is None
        or t.input.axes is None
        or t.output.axes is None
    ):
        raise ConversionError(
            "A subspace transformation is embedded into a full affine only "
            "when its input and output systems are known, because the "
            "number of axes is read from them."
        )
    n_in = len(t.input.axes)
    n_out = len(t.output.axes)
    input_axes = [
        int(i) for i in (t.input_axes if t.input_axes is not None else [])
    ]
    output_axes = [
        int(o) for o in (t.output_axes if t.output_axes is not None else [])
    ]
    if t.transformation is None:
        inner_matrix = None
        ba = get_array_backend()
    else:
        inner_affine = t.transformation.compute().to(Affine)
        if not isinstance(inner_affine, Affine):
            raise ConversionError(
                "a subspace transformation that wraps a field is applied by "
                "composing it with a sampling domain, not by reduction to an "
                "affine"
            )
        inner_matrix = inner_affine.matrix
        ba = get_array_backend(inner_matrix)
    matrix = ba.zeros((n_out, n_in + 1))
    acted_in = set(input_axes)
    acted_out = set(output_axes)
    passthrough_in = [i for i in range(n_in) if i not in acted_in]
    passthrough_out = [o for o in range(n_out) if o not in acted_out]
    for out_axis, in_axis in zip(passthrough_out, passthrough_in):
        matrix[out_axis, in_axis] = 1.0
    if inner_matrix is None:
        for out_axis, in_axis in zip(output_axes, input_axes):
            matrix[out_axis, in_axis] = 1.0
    else:
        m_in = len(input_axes)
        for r, out_axis in enumerate(output_axes):
            for c, in_axis in enumerate(input_axes):
                matrix[out_axis, in_axis] = inner_matrix[r, c]
            matrix[out_axis, n_in] = inner_matrix[r, m_in]
    return Affine(matrix=matrix, input=t.input, output=t.output)


@converter
def _(t: DisplacementField) -> CoordinatesField:
    if t.field is None:
        return CoordinatesField(
            coordinates=None, input=t.input, output=t.output
        )
    ba = get_array_backend(t.field)
    g = ba.meshgrid(*(ba.arange(s) for s in t.field.shape[:-1]), indexing="ij")
    g = ba.stack(g, axis=-1)
    return CoordinatesField(field=t.field + g, input=t.input, output=t.output)


# ----------------------------------------------------------------------
#   LOSSY
# ----------------------------------------------------------------------


@converter
def _(t: Translation) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.translation is not None and any(x != 0.0 for x in t.translation):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Scaling) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.scale is not None and any(s != 1.0 for s in t.scale):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Permutation) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.permutation is not None and any(
        i != p for i, p in enumerate(t.permutation)
    ):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if (i == j and t.matrix[i][j] != 1.0) or (
                    i != j and t.matrix[i][j] != 0.0
                ):
                    raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear) -> Permutation:
    if t.matrix is None:
        return convert(convert(t, Identity), Permutation)
    ba = get_array_backend(t.matrix)
    matrix = ba.round(ba.abs(t.matrix))
    permutation = ba.argmax(matrix, axis=1)
    u = Permutation(input=t.input, output=t.output, permutation=permutation)
    if t.matrix is not None:
        for row in t.matrix:
            if any([x not in (1, 0) for x in row]) or row.sum() != 1:
                raise LossyConversionError(result=u)
        for col in t.matrix.T:
            if any([x not in (1, 0) for x in col]) or col.sum() != 1:
                raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear) -> Scaling:
    if t.matrix is None:
        return convert(convert(t, Identity), Scaling)
    ba = get_array_backend(t.matrix)
    scale = ba.diagonal(t.matrix)  # TODO: use svd instead?
    u = Scaling(input=t.input, output=t.output, scale=scale)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if i != j and t.matrix[i][j] != 0:
                    raise LossyConversionError(result=u)
    return u


@converter
def _(t: Affine) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.matrix is not None:
        for i in range(len(t.matrix)):
            for j in range(len(t.matrix[i])):
                if (i == j and t.matrix[i][j] != 1.0) or (
                    i != j and t.matrix[i][j] != 0.0
                ):
                    raise LossyConversionError(result=u)
    return u


@converter
def _(t: Affine) -> Linear:
    if t.matrix is None:
        return convert(convert(t, Identity), Linear)
    matrix = t.matrix[:, :-1]
    u = Linear(input=t.input, output=t.output, matrix=matrix)
    if t.matrix[:, -1].any():
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: DisplacementField) -> Identity:
    u = Identity(input=t.input, output=t.output)
    if t.field is not None:
        if t.field.any():
            raise LossyConversionError(result=u)
    return u


# ----------------------------------------------------------------------
#   MULTI-HOPS
# ----------------------------------------------------------------------


def _make_converter_chain(*types: tx.List[type]) -> None:
    T0, TN = types[0], types[-1]

    @converter(T0, TN)
    def _(t: Transformation) -> Transformation:
        for T1 in types[1:]:
            t = convert(t, T1)
        return t


# Lossless
_make_converter_chain(Scaling, Linear, Affine)
_make_converter_chain(Permutation, Linear, Affine)

# Lossy
_make_converter_chain(Affine, Linear, Rotation)
_make_converter_chain(Affine, Linear, Scaling)
_make_converter_chain(Affine, Linear, Permutation)
_make_converter_chain(Linear, Identity, Translation)
_make_converter_chain(Scaling, Identity, Translation)
_make_converter_chain(Permutation, Identity, Translation)
