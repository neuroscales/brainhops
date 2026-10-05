# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel import kinds
from brainhops.datamodel.systems import _axes_or_unknown

# locals
from .base import Transformation
from .check import is_kind
from .concrete import (
    Affine,
    AffineExponential,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    LinearExponential,
    Permutation,
    Rotation,
    RotationExponential,
    Scaling,
    ScalingExponential,
    StationaryVelocityField,
    Translation,
    _decode,
    _encode,
    _integrate,
)
from .convert import convert, converter
from .errors import CompositionError, ConversionError, LossyConversionError
from .matfuncs import affine_logm, log_scale, logm
from .meta import SubspaceTransformation, _close_subspace
from .sequence import Sequence
from .utils import get_ndim


def smart_replace(t: Transformation, **kwargs) -> Transformation:
    return replace(t, **kwargs) if kwargs else t


# ----------------------------------------------------------------------
#   SAME TYPE
# ----------------------------------------------------------------------


@converter(Identity, Identity)
@converter
def _(t: Transformation, **kwargs) -> Transformation:
    # A same-type conversion with no overrides is a pass-through; with
    # overrides it rebuilds the transform of the same type, applying the
    # named fields on top of the existing ones.
    return smart_replace(t, **kwargs)


@converter
def _(t: Affine, **kwargs) -> Affine:
    if kwargs.pop("log", False):
        # The tangent of the map, which an `AffineExponential` stores: the
        # principal logarithm of the matrix (or of a new `matrix=`).
        if "data" not in kwargs:
            matrix = kwargs.pop("matrix", t.matrix)
            kwargs["data"] = _affine_log(matrix, t)
        _endpoints(t, kwargs)
        return AffineExponential(**kwargs)
    # A new `matrix=` replaces the matrix stored in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: AffineExponential, **kwargs) -> Affine:
    if not kwargs.pop("log", True):
        # The map itself, which a plain `Affine` stores: the matrix (or a
        # new `matrix=`). `replace` would keep this class, so the result
        # is built as an `Affine`.
        if "data" not in kwargs:
            kwargs["data"] = kwargs.pop("matrix", t.matrix)
        _endpoints(t, kwargs)
        return Affine(**kwargs)
    # A new `matrix=` is stored as its logarithm, in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: Linear, **kwargs) -> Linear:
    if kwargs.pop("log", False):
        # The tangent of the map, which a `LinearExponential` stores: the
        # principal logarithm of the matrix (or of a new `matrix=`).
        if "data" not in kwargs:
            matrix = kwargs.pop("matrix", t.matrix)
            kwargs["data"] = _linear_log(matrix, t)
        _endpoints(t, kwargs)
        return LinearExponential(**kwargs)
    # A new `matrix=` replaces the matrix stored in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: LinearExponential, **kwargs) -> Linear:
    if not kwargs.pop("log", True):
        # The map itself, which a plain `Linear` stores (see the affine).
        if "data" not in kwargs:
            kwargs["data"] = kwargs.pop("matrix", t.matrix)
        _endpoints(t, kwargs)
        return Linear(**kwargs)
    # A new `matrix=` is stored as its logarithm, in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: Rotation, **kwargs) -> Rotation:
    if kwargs.pop("log", False):
        # The tangent of the rotation, which a `RotationExponential`
        # stores: the principal logarithm of the matrix.
        if "data" not in kwargs:
            matrix = kwargs.pop("matrix", t.matrix)
            kwargs["data"] = _linear_log(matrix, t)
        _endpoints(t, kwargs)
        return RotationExponential(**kwargs)
    # A new `matrix=` replaces the matrix stored in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: RotationExponential, **kwargs) -> Rotation:
    if not kwargs.pop("log", True):
        # The map itself, which a plain `Rotation` stores (see the affine).
        if "data" not in kwargs:
            kwargs["data"] = kwargs.pop("matrix", t.matrix)
        _endpoints(t, kwargs)
        return Rotation(**kwargs)
    # A new `matrix=` is stored as its logarithm, in `data`.
    if "matrix" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: Permutation, **kwargs) -> Permutation:
    # A new `permutation=` replaces the vector stored in `data`.
    if "permutation" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: Scaling, **kwargs) -> Scaling:
    if kwargs.pop("log", False):
        # The tangent of the map, which a `ScalingExponential` stores: the
        # logarithm of the scaling factors (or of a new `scale=`).
        if "data" not in kwargs:
            scale = kwargs.pop("scale", t.scale)
            kwargs["data"] = _scale_log(scale, t)
        _endpoints(t, kwargs)
        return ScalingExponential(**kwargs)
    # A new `scale=` replaces the vector stored in `data`.
    if "scale" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: ScalingExponential, **kwargs) -> Scaling:
    if not kwargs.pop("log", True):
        # The map itself, which a plain `Scaling` stores (see the affine).
        if "data" not in kwargs:
            kwargs["data"] = kwargs.pop("scale", t.scale)
        _endpoints(t, kwargs)
        return Scaling(**kwargs)
    # A new `scale=` is stored as its logarithm, in `data`.
    if "scale" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: Translation, **kwargs) -> Translation:
    # A new `translation=` replaces the vector stored in `data`.
    if "translation" in kwargs:
        kwargs.setdefault("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: DisplacementField, **kwargs) -> DisplacementField:
    if kwargs.get("log"):
        # The velocity of the map, which a `StationaryVelocityField`
        # stores. A displacement has no logarithm that brainhops computes,
        # so only a field with no displacement (the identity, whose
        # velocity is zero) is converted, or one given a velocity as
        # `data=`, which is stored as it is.
        if "field" in kwargs or ("data" not in kwargs and t.data is not None):
            raise NotImplementedError(
                f"{type(t).__name__}.to(log=True) would store the velocity "
                f"of a displacement field, whose logarithm is not "
                f"implemented. A velocity is built with log=True, from "
                f"its data: DisplacementField(data=v, log=True)."
            )
        return smart_replace(t, **kwargs)
    kwargs.pop("log", None)
    # The flags of the result: those given, or else those of `t`.
    coeff = kwargs.get("coeff", t.coeff)
    degree = kwargs.get("degree", t.degree)
    bound = kwargs.get("bound", t.bound)
    if "field" in kwargs:
        # A new map, as values, stored under the flags of the result.
        if "data" in kwargs:
            raise _data_and_field(t)
        kwargs["data"] = _encode(kwargs.pop("field"), coeff, degree, bound)
    elif "data" not in kwargs and (coeff, degree, bound) != (
        t.coeff,
        t.degree,
        t.bound,
    ):
        # New flags, and no new data: the map is kept, and re-encoded.
        kwargs["data"] = _encode(t.field, coeff, degree, bound)
    return smart_replace(t, **kwargs)


@converter
def _(t: StationaryVelocityField, **kwargs) -> DisplacementField:
    # The flags of the result: those given, or else those of `t`.
    coeff = kwargs.get("coeff", t.coeff)
    degree = kwargs.get("degree", t.degree)
    bound = kwargs.get("bound", t.bound)
    if not kwargs.pop("log", True):
        # The map itself, which a plain `DisplacementField` stores: the
        # displacement of the flow (integrated with `steps=`, if given),
        # encoded under the flags of the result. `replace` would keep this
        # class, so the result is built as a `DisplacementField`.
        steps = kwargs.pop("steps", t.steps)
        if "field" in kwargs:
            if "data" in kwargs:
                raise _data_and_field(t)
            kwargs["data"] = _encode(kwargs.pop("field"), coeff, degree, bound)
        elif "data" not in kwargs:
            field = t.field
            if steps != t.steps:
                flags = t.coeff, t.degree, t.bound
                field = _integrate(t.data, *flags, steps)
            kwargs["data"] = _encode(field, coeff, degree, bound)
        kwargs.update(coeff=coeff, degree=degree, bound=bound)
        _endpoints(t, kwargs)
        return DisplacementField(**kwargs)
    if "field" in kwargs:
        raise NotImplementedError(
            f"{type(t).__name__}.to() got field=, which is the map, as "
            f"displacement values: storing it as a velocity needs the "
            f"logarithm of a field, which is not implemented. Pass the "
            f"velocity as data=, or the field with log=False."
        )
    if "data" not in kwargs and (coeff, degree, bound) != (
        t.coeff,
        t.degree,
        t.bound,
    ):
        # New flags, and no new data: the velocity is kept, and re-encoded.
        velocity = _decode(t.data, t.coeff, t.degree, t.bound)
        kwargs["data"] = _encode(velocity, coeff, degree, bound)
    return smart_replace(t, **kwargs)


@converter
def _(t: CoordinatesField, **kwargs) -> CoordinatesField:
    # The flags of the result: those given, or else those of `t`.
    coeff = kwargs.get("coeff", t.coeff)
    degree = kwargs.get("degree", t.degree)
    bound = kwargs.get("bound", t.bound)
    if "field" in kwargs:
        # A new map, as values, stored under the flags of the result.
        if "data" in kwargs:
            raise _data_and_field(t)
        kwargs["data"] = _encode(kwargs.pop("field"), coeff, degree, bound)
    elif "data" not in kwargs and (coeff, degree, bound) != (
        t.coeff,
        t.degree,
        t.bound,
    ):
        # New flags, and no new data: the map is kept, and re-encoded.
        kwargs["data"] = _encode(t.field, coeff, degree, bound)
    return smart_replace(t, **kwargs)


@converter
def _(t: CartesianField, **kwargs) -> CartesianField:
    # A CartesianField generates its `field` and its `data` on demand
    # from `shape` and the flags, so neither is a constructor argument
    # here, and nothing needs re-encoding. A rebuild carries `shape` over
    # and lets the new instance regenerate them, and any `field` or
    # `data` override is dropped.
    kwargs.pop("field", None)
    kwargs.pop("data", None)
    return smart_replace(t, **kwargs)


@converter
def _(t: SubspaceTransformation, **kwargs) -> SubspaceTransformation:
    if "log" in kwargs:
        # The encoding of the inner transformation. The axes a subspace
        # does not act on are the identity, whose tangent is zero, so a
        # tangent inner pads to the tangent of the whole: the map is kept.
        log = kwargs.pop("log")
        inner = kwargs.get("transformation", t.transformation)
        if inner is not None:
            kwargs["transformation"] = inner.to(log=log)
    return smart_replace(t, **kwargs)


def _data_and_field(t: Transformation) -> TypeError:
    return TypeError(
        f"{type(t).__name__}.to() got both data= and field=: field= is "
        f"the map, as values, encoded under the flags of the result, while "
        f"data= is stored as given. Pass one of them."
    )


def _endpoints(t: Transformation, kwargs: tx.Dict[str, tx.Any]) -> None:
    # The endpoints of a result built afresh: those given, or those of `t`.
    kwargs.setdefault("input", t.input)
    kwargs.setdefault("output", t.output)


def _affine_log(
    matrix: tx.Optional[ArrayProtocol], t: Transformation
) -> tx.Optional[ArrayProtocol]:
    if matrix is None:
        return None
    return affine_logm(matrix, f"The logarithm of this {type(t).__name__}")


def _linear_log(
    matrix: tx.Optional[ArrayProtocol], t: Transformation
) -> tx.Optional[ArrayProtocol]:
    if matrix is None:
        return None
    return logm(matrix, f"The logarithm of this {type(t).__name__}")


def _scale_log(
    scale: tx.Optional[ArrayProtocol], t: Transformation
) -> tx.Optional[ArrayProtocol]:
    if scale is None:
        return None
    return log_scale(scale, f"The logarithm of this {type(t).__name__}")


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
    if "data" not in kwargs and "matrix" not in kwargs:
        kwargs["data"] = t.matrix
    kwargs.setdefault("input", t.input)
    kwargs.setdefault("output", t.output)
    u = Rotation(**kwargs)
    if not is_kind(t, kinds.SpecialOrthogonal, compute=True):
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
    n_in = _axes_or_unknown(t.input).ndim
    n_out = _axes_or_unknown(t.output).ndim
    if n_in is None or n_out is None:
        raise ConversionError(
            "The axis count of this subspace transformation is unknown, so "
            "it cannot be embedded into a full affine, whose size is read "
            "from it. Its input and output systems must be closed, but one "
            "of them is missing or open (its axes hold `...`). Declare the "
            "full-space systems, or close them with "
            "CoordinateSystem.expand."
        )
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
def _(t: Sequence) -> Affine:
    # The map a sequence describes, as one affine over its endpoints.
    #
    # `compute()` (every kind admitted) hands each adjacent pair to
    # `compose` and keeps the two side by side only when the composer
    # raises `CompositionError` (`sequence._compose_mode`). Among pieces
    # that each have an affine form, the one composer that does so is
    # `SubspaceTransformation @ SubspaceTransformation` when the axes the
    # second writes are not the axes the first reads -- such as the
    # spatial and the temporal subspace a 4D NIfTI image is read as. An
    # affine-like piece next to a subspace composes with it, and every
    # other refusal involves a piece with no affine form (a field, bare or
    # in a subspace, or a `Projection`). So a sequence that `compute()`
    # leaves is reduced here only when it is a chain of subspace
    # transforms: each is embedded in an affine over its full space, and
    # their product is the affine of the sequence (block-diagonal when the
    # axes are disjoint). Anything else has no affine form.
    reduced = t.compute()
    if not isinstance(reduced, Sequence):
        return convert(reduced, Affine)
    pieces = reduced.transformations
    # The space between two pieces is one, so the number of axes a piece
    # does not state is read from its neighbour, carried along the chain
    # from the start of the sequence, and from its end for the last piece.
    # The endpoints are those of `t`: a sequence that `compute()` leaves
    # is a new one, which does not carry the endpoints `t` declares.
    n = _axes_or_unknown(t.input).ndim
    n_end = _axes_or_unknown(t.output).ndim
    matrix = None
    for i, piece in enumerate(pieces):
        if not isinstance(piece, SubspaceTransformation):
            raise ConversionError(
                f"This sequence does not compose into one transform, and "
                f"its {type(piece).__name__} has no affine form, so the "
                f"sequence has none either."
            )
        last = i == len(pieces) - 1
        try:
            piece = _close_subspace(piece, n, n_end if last else None)
        except CompositionError as error:
            raise ConversionError(str(error)) from error
        n = _axes_or_unknown(piece.output).ndim
        block = convert(piece, Affine).matrix
        if matrix is None:
            matrix = block
            continue
        ba = get_array_backend(block)
        product = ba.empty_like(block, shape=(block.shape[0], matrix.shape[1]))
        product[:, :-1] = block[:, :-1] @ matrix[:, :-1]
        product[:, -1:] = block[:, :-1] @ matrix[:, -1:] + block[:, -1:]
        matrix = product
    return Affine(matrix=matrix, input=t.input, output=t.output)


@converter
def _(t: DisplacementField) -> CoordinatesField:
    # The coordinate of a grid point is the point plus its displacement.
    # The result keeps the encoding of `t` (its `coeff`, `degree` and
    # `bound`). Fitting spline coefficients is linear, so the encoded
    # arrays add up the same way the values do: the coordinates' `data`
    # is the displacements' `data` plus the grid, encoded under the same
    # flags. Nothing is decoded, and a field of coefficients stays one.
    flags = dict(coeff=t.coeff, degree=t.degree, bound=t.bound)
    data = t.data
    if data is None:
        return CoordinatesField(input=t.input, output=t.output, **flags)
    ba = get_array_backend(data)
    grid = ba.meshgrid(*(ba.arange(s) for s in data.shape[:-1]), indexing="ij")
    grid = _encode(ba.stack(grid, axis=-1).astype(data.dtype), **flags)
    return CoordinatesField(
        data=data + grid, input=t.input, output=t.output, **flags
    )


@converter
def _(t: StationaryVelocityField) -> CoordinatesField:
    # The coordinates of the map: those of the displacement of the flow,
    # which the `field` view integrates. They are stored under the
    # encoding of `t` -- its `coeff`, `degree` and `bound` -- but not as a
    # velocity: `log` and `steps` say how a velocity is stored, and do not
    # carry over.
    flags = dict(coeff=t.coeff, degree=t.degree, bound=t.bound)
    displacement = DisplacementField(
        field=t.field, input=t.input, output=t.output, **flags
    )
    return convert(displacement, CoordinatesField)


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
