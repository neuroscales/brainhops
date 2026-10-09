# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.magic import replace, stores
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import (
    CompositionError,
    ConversionError,
    LossyConversionError,
)

# locals
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

# typing
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
# - tangents
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
    """
    Rebuild `t` as `cls`, with `kwargs` on top of the fields it holds.

    A request that changes neither the class nor a field is answered with
    `t` itself: a conversion that has nothing to do copies nothing.
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
    # A same-type conversion with no overrides is a pass-through; with
    # overrides it rebuilds the transform of the same type, applying the
    # named fields on top of the existing ones.
    #
    # This is the catch-all, which every request reaches, so a `cls` from
    # another family is dropped rather than built: a field rebuilt as an
    # `Affine` would hold the field array as its matrix. What comes back
    # is then of the wrong type, which is how `convert` reports a
    # conversion nothing can make.
    if not (issubclass(cls, type(t)) or issubclass(type(t), cls)):
        cls = None
    return smart_replace(t, cls, **kwargs)


def _convert_withdata(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # A view (`matrix=`, `field=`, ...) is the map, and fills `data`, so
    # the `data` the rebuild would carry over is cleared and the view is
    # what lands. A caller who names both is left to the constructor,
    # which refuses the two together.
    if "data" not in kwargs and set(cls.derived_fields) & set(kwargs):
        kwargs["data"] = None
    return smart_replace(t, cls, **kwargs)


def _convert_withlog(
    t: Transformation, cls: tx.Type[TRANSFORMATION], _fallback: str, **kwargs
) -> TRANSFORMATION:
    # If log<->exp conversion, transfer the exponentiated parameter
    # (as `data` has different semantics with and without log).
    #
    # A family with no `log` flag -- a coordinates field -- has no tangent
    # to cross to, so both sides read as the map and nothing is
    # transferred.
    out_log = kwargs.get("log", _is_tangent(cls))
    crossing = getattr(t, "log", False) != out_log
    # A transform that *derives* its `data` -- a reader whose matrix comes
    # from the file it parsed, a grid that builds its coordinates from
    # `shape` -- does not keep it under the name a rebuild reads, so a
    # rebuild as a class that is not one of its own is handed the map
    # through `_fallback` instead.
    derived = not stores(type(t), "_data") and not issubclass(cls, type(t))
    if crossing or derived:
        fields = set(cls.data_fields) | set(cls.derived_fields)
        if not (fields & set(kwargs)):
            kwargs.setdefault(_fallback, getattr(t, _fallback))
        if crossing:
            # The rebuild carries the flag over from `t`, and crossing the
            # boundary is what this conversion does, so it is named.
            kwargs["log"] = out_log
    # If a data-like field is provided, set `data=None` to avoid conflicts
    return _convert_withdata(t, cls, **kwargs)


_SPLINE_FLAGS = ("degree", "bound", "store")


def _convert_withsplines(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # The encoding asked for, beside the one `t` has. Each flag is taken
    # out of `kwargs` and written back only if it changed: a flag that is
    # the one `t` already has says nothing, and a class whose flags are
    # derived -- a lazy inverse reports its forward's -- does not take it.
    before = tuple(getattr(t, name) for name in _SPLINE_FLAGS)
    after = tuple(
        kwargs.pop(name, value) for name, value in zip(_SPLINE_FLAGS, before)
    )
    kwargs.update(
        (name, new)
        for name, old, new in zip(_SPLINE_FLAGS, before, after)
        if old != new
    )

    # Ensure we are working in the same log/exp domain. The conversion is
    # a no-op when the field is already in the one asked for, and a
    # coordinates field has no tangent domain at all.
    logkwargs = {"log": kwargs.pop("log")} if "log" in kwargs else {}
    t = _convert_withlog(t, cls, "field", **logkwargs)
    cls = type(t)

    # If the spline flags differ, ensure that it is the values that
    # are transferred, so that spline fitting is correctly triggered.
    if before != after:
        fields = set(cls.data_fields) | set(cls.derived_fields)
        if not (fields & set(kwargs)):
            kwargs.setdefault("values", t.values)

    # If a data-like field is provided, set `data=None` to avoid conflicts
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
    # A type is not an encoding: a field becomes a velocity with
    # `.to(log=True)`, which says what it does to the stored `data`.
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
    # A grid generates its coordinates from `shape`, and does not store
    # them, so a rebuild as a class that *does* store them has to be
    # handed the generated array -- as values, which the result encodes
    # under its own flags.
    kwargs.setdefault("values", t.values)
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(
    t: CartesianField, cls: tx.Type[CartesianField], **kwargs
) -> CartesianField:
    # A CartesianField generates its `field` and its `data` on demand
    # from `shape` and the flags, so neither is a constructor argument
    # here, and nothing needs re-encoding. A rebuild carries `shape` over
    # and lets the new instance regenerate them, and any `field` or
    # `data` override is dropped.
    for key in cls.derived_fields:
        kwargs.pop(key, None)
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: SubspaceTransformation, cls: tx.Type[SubspaceTransformation], **kwargs
) -> SubspaceTransformation:
    # FIXME: Can we not simply pass all kwargs that are not fields
    # of the meta class to the inner class? Should we do this for all
    # meta classes?
    if "log" in kwargs:
        # The encoding of the inner transformation; the map is kept. When
        # the subspace reads and writes the same axes, the axes it does not
        # act on are the identity, whose tangent is zero, so the padded
        # inner tangent is the tangent of the whole. A subspace that
        # reindexes its axes has no such tangent of its own, and only its
        # inner transformation is re-encoded.
        log = kwargs.pop("log")
        inner = kwargs.get("transformation", t.transformation)
        if inner is not None:
            kwargs["transformation"] = inner.to(log=log)
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   LOSSLESS
# ----------------------------------------------------------------------
# Each of these hands over a view of the map -- `matrix=`, `scale=`,
# `translation=` -- so each goes through `_convert_withdata`, which
# clears the `data` the rebuild would otherwise carry over: the stored
# parameter of one family is not the stored parameter of another.


def _sized(
    t: Identity, make: tx.Callable[[int], tx.Any]
) -> tx.Optional[tx.Any]:
    # The identity parameter of a family, at the axis count of `t`. An
    # identity whose endpoints do not say how many axes it has converts
    # with its parameter unset, which reads as the identity too.
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
        # The affine of a `(No, Ni)` matrix is that matrix with a zero
        # translation column, whatever its endpoints say -- and they may
        # say nothing. It is built on the backend the matrix lives on.
        ba = get_array_backend(t.matrix)
        zeros = ba.zeros((*t.matrix.shape[:-1], 1), dtype=t.matrix.dtype)
        kwargs["matrix"] = ba.concatenate([t.matrix, zeros], axis=-1)
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: SubspaceTransformation, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    # FIXME: Do we need to be a bit more generic, like in the other cases?
    # (use smart_replace?)

    # Embed the inner transform, reduced to an affine, into a full-size
    # affine over the whole space. The inner transform occupies the rows
    # and columns of the acted-on axes. Every other axis passes through as
    # the identity, mapping each input pass-through axis to the output
    # pass-through axis in the same position in the ordering.
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
    # FIXME: Do we need to be a bit more generic, like in the other cases?
    # (use smart_replace?)

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
        return convert(reduced, cls)
    pieces = reduced.transformations
    # The space between two pieces is one, so the number of axes a piece
    # does not state is read from its neighbour, carried along the chain
    # from the start of the sequence, and from its end for the last piece.
    # The endpoints are those of `t`: a sequence that `compute()` leaves
    # is a new one, which does not carry the endpoints `t` declares.
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
    # FIXME: Do we need to be a bit more generic, like in the other cases?
    # (use smart_replace?)

    # The coordinate of a grid point is the point plus its displacement.
    # The result keeps the encoding of `t` (its `store`, `degree` and
    # `bound`). Fitting spline coefficients is linear, so the encoded
    # arrays add up the same way the values do: the coordinates' `data`
    # is the displacements' `data` plus the grid, encoded under the same
    # flags. Nothing is decoded, and a field of coefficients stays one.
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
# A lossy conversion builds the result first and checks the promise it
# makes afterwards, so the transform it would have produced is what
# travels on the `LossyConversionError`.


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
    scale = ba.diagonal(t.matrix)  # TODO: use svd instead?
    kwargs.setdefault("scale", scale)
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_scaling(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    # A `Rotation` stores the same matrix as a `Linear`, but promises it is
    # orthogonal with determinant +1 -- which is what lets it invert by
    # transposing instead of solving. The promise is checked, so converting
    # a matrix that does not keep it is lossy.
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
    """
    Register the converter that walks from the first type to the last.

    The types between them are the hops it goes through, each a concrete
    class: `convert` builds an instance of the class it is handed, so a
    `TypeVar` named here stands for its bound. The endpoints only key the
    registration, and may stay `TypeVar`s.
    """
    first, *hops, last = types
    hops = tuple(_concrete(hop) for hop in hops)

    @converter(first, last)
    def _(t: Transformation, cls: type, **kwargs) -> Transformation:
        for hop in hops:
            t = convert(t, hop)
        t = convert(t, cls, **kwargs)  # only use cls/kwargs on the last hop
        return t


def _concrete(hop: tx.Any) -> type:
    """The class a hop names: a `TypeVar` stands for its bound."""
    return hop.__bound__ if isinstance(hop, tx.TypeVar) else hop


# Lossless
_make_converter_chain(SCALING, LINEAR, AFFINE)
_make_converter_chain(PERMUTATION, LINEAR, AFFINE)
_make_converter_chain(SVF, DISP, COORD)

# Lossy
_make_converter_chain(AFFINE, LINEAR, ROTATION)
_make_converter_chain(AFFINE, LINEAR, SCALING)
_make_converter_chain(AFFINE, LINEAR, PERMUTATION)
_make_converter_chain(LINEAR, IDENTITY, TRANSLATION)
_make_converter_chain(SCALING, IDENTITY, TRANSLATION)
_make_converter_chain(PERMUTATION, IDENTITY, TRANSLATION)
