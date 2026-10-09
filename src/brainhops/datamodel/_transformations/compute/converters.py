import numpy as np
import typing_extensions as tx

from brainhops._core.magic import replace, stores
from brainhops.backends import get_array_backend
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import (
    CompositionError,
    ConversionError,
    LossyConversionError,
)

from ..base import Transformation
from ..concrete import (
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
    _is_tangent,
    _values2data,
)
from ..meta import SubspaceTransformation, _close_subspace
from ..sequence import Sequence
from ..tangents import (
    AffineExponential,
    LinearExponential,
    RotationExponential,
    ScalingExponential,
    StationaryVelocityField,
)
from .convert import convert, converter
from .utils import get_ndim

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)
DISP = tx.TypeVar("DISP", bound=DisplacementField)
COORD = tx.TypeVar("COORD", bound=CoordinatesField)
AFFINE = tx.TypeVar("AFFINE", bound=Affine)
LINEAR = tx.TypeVar("LINEAR", bound=Linear)
ROTATION = tx.TypeVar("ROTATION", bound=Rotation)
SCALING = tx.TypeVar("SCALING", bound=Scaling)
PERMUTATION = tx.TypeVar("PERMUTATION", bound=Permutation)
TRANSLATION = tx.TypeVar("TRANSLATION", bound=Translation)
IDENTITY = tx.TypeVar("IDENTITY", bound=Identity)
SVF = tx.TypeVar("SVF", bound=StationaryVelocityField)
AFFINE_EXP = tx.TypeVar("AFFINE_EXP", bound=AffineExponential)
LINEAR_EXP = tx.TypeVar("LINEAR_EXP", bound=LinearExponential)
ROTATION_EXP = tx.TypeVar("ROTATION_EXP", bound=RotationExponential)
SCALING_EXP = tx.TypeVar("SCALING_EXP", bound=ScalingExponential)


def smart_replace(
    t: Transformation,
    cls: tx.Optional[tx.Type[TRANSFORMATION]] = None,
    **kwargs,
) -> TRANSFORMATION:
    """Rebuild a transformation as `cls`, overriding fields with `kwargs`.

    The transformation itself is returned, without a copy, when no field is
    overridden and `cls` is either missing or the type of `t`.
    """
    doit = kwargs or (cls and cls is not type(t))
    return replace(t, cls, **kwargs) if doit else t


# ----------------------------------------------------------------------
#   SAME TYPE
# ----------------------------------------------------------------------


@converter(Identity, Identity)
@converter
def _(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # Every request can reach this catch-all, so a target from an unrelated
    # family is discarded instead of built (an Affine would receive a field
    # array as its matrix). The result keeps the type of `t`, which is how
    # `convert` signals an impossible conversion.
    if not (issubclass(cls, type(t)) or issubclass(type(t), cls)):
        cls = None
    return smart_replace(t, cls, **kwargs)


def _convert_withdata(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # A view such as `matrix=` fills `data`, so the carried-over data is
    # cleared. Passing both is left to the constructor, which rejects it.
    if "data" not in kwargs and set(cls.derived_fields) & set(kwargs):
        kwargs["data"] = None
    return smart_replace(t, cls, **kwargs)


def _convert_withlog(
    t: Transformation, cls: tx.Type[TRANSFORMATION], _fallback: str, **kwargs
) -> TRANSFORMATION:
    # Crossing between log and exp transfers the exponentiated parameter,
    # because `data` means different things on each side. Families without a
    # `log` flag have no tangent, so nothing is transferred.
    out_log = kwargs.get("log", _is_tangent(cls))
    crossing = getattr(t, "log", False) != out_log
    # Transformations that derive their data (a file reader, a grid built from
    # a shape) do not store it under the name the rebuild reads, so a foreign
    # class receives the map through `_fallback`.
    derived = not stores(type(t), "_data") and not issubclass(cls, type(t))
    if crossing or derived:
        fields = set(cls.data_fields) | set(cls.derived_fields)
        if not (fields & set(kwargs)):
            kwargs.setdefault(_fallback, getattr(t, _fallback))
        if crossing:
            # Crossing is what this conversion does, so the flag is set instead
            # of carried over.
            kwargs["log"] = out_log
    return _convert_withdata(t, cls, **kwargs)


_SPLINE_FLAGS = ("degree", "bound", "store")


def _convert_withsplines(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # A spline flag is written back only if it differs from that of `t`: an
    # unchanged flag carries no information, and classes with derived flags (a
    # lazy inverse) reject it.
    before = tuple(getattr(t, name) for name in _SPLINE_FLAGS)
    after = tuple(
        kwargs.pop(name, value) for name, value in zip(_SPLINE_FLAGS, before)
    )
    kwargs.update(
        (name, new)
        for name, old, new in zip(_SPLINE_FLAGS, before, after)
        if old != new
    )

    # Move to the requested log or exp domain first.
    logkwargs = {"log": kwargs.pop("log")} if "log" in kwargs else {}
    t = _convert_withlog(t, cls, "field", **logkwargs)
    cls = type(t)

    # Changed flags require the values, not the data, so that the coefficients
    # are fitted again.
    if before != after:
        fields = set(cls.data_fields) | set(cls.derived_fields)
        if not (fields & set(kwargs)):
            kwargs.setdefault("values", t.values)

    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Translation, cls: tx.Type[TRANSLATION], **kwargs) -> TRANSLATION:
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Permutation, cls: tx.Type[PERMUTATION], **kwargs) -> PERMUTATION:
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Affine, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: AffineExponential, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    out_log = kwargs.get("log", _is_tangent(cls))
    if not out_log and issubclass(cls, AffineExponential):
        cls = Affine
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: Linear, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: LinearExponential, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    out_log = kwargs.get("log", _is_tangent(cls))
    if not out_log and issubclass(cls, LinearExponential):
        cls = Linear
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: Rotation, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: RotationExponential, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    out_log = kwargs.get("log", _is_tangent(cls))
    if not out_log and issubclass(cls, RotationExponential):
        cls = Rotation
    return _convert_withlog(t, cls, "matrix", **kwargs)


@converter
def _(t: Scaling, cls: tx.Type[SCALING], **kwargs) -> SCALING:
    return _convert_withlog(t, cls, "scale", **kwargs)


@converter
def _(t: ScalingExponential, cls: tx.Type[SCALING], **kwargs) -> SCALING:
    out_log = kwargs.get("log", _is_tangent(cls))
    if not out_log and issubclass(cls, ScalingExponential):
        cls = Scaling
    return _convert_withlog(t, cls, "scale", **kwargs)


@converter
def _(t: DisplacementField, cls: tx.Type[DISP], **kwargs) -> DISP:
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(t: DisplacementField, cls: tx.Type[SVF], **kwargs) -> SVF:
    # A type is not an encoding: a field becomes a velocity through
    # `.to(log=True)`.
    raise ConversionError(
        f"A {type(t).__name__} is converted to a StationaryVelocityField "
        f"with t.to(log=True), not t.to(StationaryVelocityField). A field "
        f"that holds a displacement has no logarithm that brainhops "
        f"computes; one whose data is a velocity is built with "
        f"DisplacementField(data=v, log=True)."
    )


@converter(StationaryVelocityField, SVF)
@converter
def _(t: StationaryVelocityField, cls: tx.Type[DISP], **kwargs) -> DISP:
    out_log = kwargs.get("log", _is_tangent(cls))
    if not out_log and issubclass(cls, StationaryVelocityField):
        cls = DisplacementField
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(t: CoordinatesField, cls: tx.Type[COORD], **kwargs) -> COORD:
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(t: CartesianField, cls: tx.Type[COORD], **kwargs) -> COORD:
    # A grid generates its coordinates from its shape, so a storing class must
    # receive them as `values`, encoded under its own flags.
    kwargs.setdefault("values", t.values)
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(
    t: CartesianField, cls: tx.Type[CartesianField], **kwargs
) -> CartesianField:
    # A CartesianField regenerates its field and data from its shape and flags,
    # which are not constructor arguments, so overrides of them are dropped.
    for key in cls.derived_fields:
        kwargs.pop(key, None)
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: SubspaceTransformation, cls: tx.Type[SubspaceTransformation], **kwargs
) -> SubspaceTransformation:
    # TODO: pass keywords that are not meta-class fields to the inner
    # transformation, perhaps for all meta classes.
    if "log" in kwargs:
        # `log` re-encodes the inner transformation and keeps the map.
        # Untouched axes are an identity with zero tangent, so the padded inner
        # tangent is the whole tangent; a reindexing subspace has no tangent of
        # its own.
        log = kwargs.pop("log")
        inner = kwargs.get("transformation", t.transformation)
        if inner is not None:
            kwargs["transformation"] = inner.to(log=log)
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   LOSSLESS
# ----------------------------------------------------------------------
# Each lossless converter supplies a view and goes through `_convert_withdata`,
# which clears `data`, because the stored parameter differs between families.


def _sized(
    t: Identity, make: tx.Callable[[int], tx.Any]
) -> tx.Optional[tx.Any]:
    # When the endpoints give no axis count, the parameter is left unset, which
    # also reads as the identity.
    ndim = get_ndim(t)
    return None if ndim is None else make(ndim)


@converter
def _(t: Identity, cls: tx.Type[TRANSLATION], **kwargs) -> TRANSLATION:
    kwargs.setdefault("translation", _sized(t, lambda n: [0.0] * n))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Identity, cls: tx.Type[SCALING], **kwargs) -> SCALING:
    kwargs.setdefault("scale", _sized(t, lambda n: [1.0] * n))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Identity, cls: tx.Type[PERMUTATION], **kwargs) -> PERMUTATION:
    kwargs.setdefault("permutation", _sized(t, lambda n: list(range(n))))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Identity, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    kwargs.setdefault("matrix", _sized(t, np.eye))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Identity, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    kwargs.setdefault("matrix", _sized(t, np.eye))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Identity, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    kwargs.setdefault("matrix", _sized(t, lambda n: np.eye(n + 1)[:-1]))
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Translation, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    if t.is_identity():
        return convert(convert(t, Identity), cls, **kwargs)
    if "matrix" not in kwargs:
        ndim = max(get_ndim(t, 0), len(t.translation))
        matrix = np.eye(ndim + 1)[:-1]
        matrix[:, -1] = t.translation
        kwargs["matrix"] = matrix
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Scaling, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    if t.is_identity():
        return convert(convert(t, Identity), cls, **kwargs)
    if "matrix" not in kwargs:
        ndim = max(get_ndim(t, 0), len(t.scale))
        matrix = np.eye(ndim)
        matrix[range(ndim), range(ndim)] = t.scale
        kwargs["matrix"] = matrix
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Permutation, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    if t.is_identity():
        return convert(convert(t, Identity), cls, **kwargs)
    if "matrix" not in kwargs:
        ndim = max(get_ndim(t, 0), len(t.permutation))
        matrix = np.zeros((ndim, ndim))
        matrix[range(ndim), t.permutation] = 1.0
        kwargs["matrix"] = matrix
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: Linear, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    if t.is_identity():
        return convert(convert(t, Identity), cls, **kwargs)
    if "matrix" not in kwargs:
        # Append a zero translation column, whatever the (possibly unspecified)
        # endpoints.
        ba = get_array_backend(t.matrix)
        zeros = ba.zeros((*t.matrix.shape[:-1], 1), dtype=t.matrix.dtype)
        kwargs["matrix"] = ba.concatenate([t.matrix, zeros], axis=-1)
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: SubspaceTransformation, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    # TODO: make this converter generic like the others (smart_replace?).

    # The inner affine occupies the rows and columns of the acted axes. The
    # other axes form an identity mapping the i-th pass-through input axis to
    # the i-th pass-through output axis.
    n_in = get_axes(t.input).ndim
    n_out = get_axes(t.output).ndim
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
    return cls(matrix=matrix, input=t.input, output=t.output)


@converter
def _(t: Sequence, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    # TODO: make this converter generic like the others (smart_replace?).

    # `compute()` leaves adjacent pieces separate only when composition raises
    # CompositionError. For pieces with an affine form, this happens only
    # between subspace transformations whose written and read axes differ
    # (spatial and temporal parts of a 4D NIfTI); other refusals involve pieces
    # without an affine form. A leftover sequence therefore reduces only if it
    # is a chain of subspace transformations, whose embedded affines are
    # multiplied.
    reduced = t.compute()
    if not isinstance(reduced, Sequence):
        return convert(reduced, cls)
    pieces = reduced.transformations
    # Neighbouring pieces share a space, so unknown axis counts are propagated
    # along the chain. The endpoints are those of `t`, because the result of
    # `compute()` lacks the declared ones.
    n = get_axes(t.input).ndim
    n_end = get_axes(t.output).ndim
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
        n = get_axes(piece.output).ndim
        block = convert(piece, Affine).matrix
        if matrix is None:
            matrix = block
            continue
        ba = get_array_backend(block)
        product = ba.empty_like(block, shape=(block.shape[0], matrix.shape[1]))
        product[:, :-1] = block[:, :-1] @ matrix[:, :-1]
        product[:, -1:] = block[:, :-1] @ matrix[:, -1:] + block[:, -1:]
        matrix = product
    return cls(matrix=matrix, input=t.input, output=t.output)


@converter
def _(t: DisplacementField, cls: tx.Type[COORD], **kwargs) -> COORD:
    # TODO: make this converter generic like the others (smart_replace?).

    # Coordinates are the grid plus the displacement, in the encoding of `t`.
    # Spline fitting is linear, so the encoded grid is added to the encoded
    # data without decoding.
    flags = dict(store=t.store, degree=t.degree, bound=t.bound)
    data = t.data
    if data is None:
        return cls(input=t.input, output=t.output, **flags)
    ba = get_array_backend(data)
    grid = ba.meshgrid(*(ba.arange(s) for s in data.shape[:-1]), indexing="ij")
    grid = _values2data(ba.stack(grid, axis=-1).astype(data.dtype), **flags)
    return cls(data=data + grid, input=t.input, output=t.output, **flags)


# ----------------------------------------------------------------------
#   LOSSY
# ----------------------------------------------------------------------
# The result is built before the promise is checked, so that it travels with
# the LossyConversionError.


@converter
def _(t: Transformation, cls: tx.Type[IDENTITY], **kwargs) -> IDENTITY:
    u = smart_replace(t, cls, **kwargs)
    if not t.is_identity(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[PERMUTATION], **kwargs) -> PERMUTATION:
    if t.is_identity():
        return convert(convert(t, Identity), Permutation, **kwargs)
    ba = get_array_backend(t.matrix)
    matrix = ba.round(ba.abs(t.matrix))
    permutation = ba.argmax(matrix, axis=1)
    kwargs.setdefault("permutation", permutation)
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_permutation(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[SCALING], **kwargs) -> SCALING:
    if t.is_identity():
        return convert(convert(t, Identity), Scaling, **kwargs)
    ba = get_array_backend(t.matrix)
    scale = ba.diagonal(t.matrix)  # TODO: use SVD instead?
    kwargs.setdefault("scale", scale)
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_scaling(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    # A Rotation promises an orthogonal matrix with determinant +1 (inverted by
    # transposition); a matrix that breaks the promise makes the conversion
    # lossy.
    if t.is_identity():
        return convert(convert(t, Identity), Rotation, **kwargs)
    u = smart_replace(t, cls, **kwargs)
    if not t.is_rotation(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Affine, cls: tx.Type[IDENTITY], **kwargs) -> IDENTITY:
    u = smart_replace(t, cls, **kwargs)
    if not t.is_identity(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Affine, cls: tx.Type[Linear], **kwargs) -> Linear:
    if t.matrix is None:
        return convert(convert(t, Identity), Linear, **kwargs)
    kwargs.setdefault("matrix", t.matrix[:, :-1])
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_linear(compute=True):
        raise LossyConversionError(result=u)
    return u


# ----------------------------------------------------------------------
#   MULTI-HOPS
# ----------------------------------------------------------------------


def _make_converter_chain(*types: type) -> None:
    """Register a converter that walks from the first type to the last one.

    The intermediate hops are converted to in order; a TypeVar hop stands for
    its bound. The endpoints only key the registration.
    """
    first, *hops, last = types
    hops = tuple(_concrete(hop) for hop in hops)

    @converter(first, last)
    def _(t: Transformation, cls: type, **kwargs) -> Transformation:
        for hop in hops:
            t = convert(t, hop)
        t = convert(t, cls, **kwargs)  # applied on the last hop only
        return t


def _concrete(hop: tx.Any) -> type:
    return hop.__bound__ if isinstance(hop, tx.TypeVar) else hop


_make_converter_chain(SCALING, LINEAR, AFFINE)
_make_converter_chain(PERMUTATION, LINEAR, AFFINE)
_make_converter_chain(SVF, DISP, COORD)

_make_converter_chain(AFFINE, LINEAR, ROTATION)
_make_converter_chain(AFFINE, LINEAR, SCALING)
_make_converter_chain(AFFINE, LINEAR, PERMUTATION)
_make_converter_chain(LINEAR, IDENTITY, TRANSLATION)
_make_converter_chain(SCALING, IDENTITY, TRANSLATION)
_make_converter_chain(PERMUTATION, IDENTITY, TRANSLATION)
