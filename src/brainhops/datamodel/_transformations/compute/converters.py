# stdlib
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.backends import _creation_options, array_backend
from brainhops._core.magic import replace, stores
from brainhops._core.typing import ArrayProtocol

# api
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import (
    CompositionError,
    ConversionError,
    LossyConversionError,
)

# transformations
from ..base import Transformation
from ..concrete import (
    Affine,
    CartesianField,
    CoordinateField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    TransformationField,
    Translation,
    _is_tangent,
    _values2data,
)
from ..meta import Projection, SubspaceTransformation, _close_subspace
from ..sequence import Sequence
from ..tangents import (
    AffineExponential,
    LinearExponential,
    RotationExponential,
    ScalingExponential,
    StationaryVelocityField,
)
from .convert import convert, converter
from .utils import axis_counts, axis_list, get_ndim

# typing
TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)
DISP = tx.TypeVar("DISP", bound=DisplacementField)
COORD = tx.TypeVar("COORD", bound=CoordinateField)
AFFINE = tx.TypeVar("AFFINE", bound=Affine)
LINEAR = tx.TypeVar("LINEAR", bound=Linear)
ROTATION = tx.TypeVar("ROTATION", bound=Rotation)
SCALING = tx.TypeVar("SCALING", bound=Scaling)
PERMUTATION = tx.TypeVar("PERMUTATION", bound=Permutation)
TRANSLATION = tx.TypeVar("TRANSLATION", bound=Translation)
IDENTITY = tx.TypeVar("IDENTITY", bound=Identity)
SEQUENCE = tx.TypeVar("SEQUENCE", bound=Sequence)
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
    # Every request can reach this catch-all converter, so a target class
    # from an unrelated family is not built. For example, an Affine rebuilt
    # from a field would receive the field array as its matrix. `t` is then
    # returned as it is, which is how `convert` learns that the conversion
    # is impossible. The fields in `kwargs` belong to the target class, so
    # they are not applied to `t`.
    if not (issubclass(cls, type(t)) or issubclass(type(t), cls)):
        return t
    return smart_replace(t, cls, **kwargs)


@converter
def _(t: Transformation, cls: tx.Type[SEQUENCE], **kwargs) -> SEQUENCE:
    if cls._data_model() is not cls:
        # A file format that stores a chain decides with its own converters
        # what it holds. Without one, it is rebuilt from a chain of a related
        # class, as the catch-all converter rebuilds it, and anything else is
        # returned unchanged, which tells `convert` that the format does not
        # hold it.
        if issubclass(cls, type(t)) or issubclass(type(t), cls):
            return smart_replace(t, cls, **kwargs)
        return t
    # Another chain holds the same fields, and is rebuilt as the requested
    # chain. Any other transformation becomes a chain of one element.
    if isinstance(t, Sequence):
        return smart_replace(t, cls, **kwargs)
    return cls([t], **kwargs)


def _convert_withdata(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # A view such as `matrix=` is what fills `data`, so the data that the
    # rebuild would carry over from `t` is cleared. A caller who passes both
    # a view and `data` is left to the constructor, which rejects the pair.
    if "data" not in kwargs and set(cls.derived_fields) & set(kwargs):
        kwargs["data"] = None
    return smart_replace(t, cls, **kwargs)


def _convert_withlog(
    t: Transformation, cls: tx.Type[TRANSFORMATION], _fallback: str, **kwargs
) -> TRANSFORMATION:
    # When the conversion crosses between the log and exp encodings, the
    # exponentiated parameter is transferred instead of `data`, because
    # `data` means different things in the two encodings. Families without
    # a `log` flag have no tangent, so for them nothing is transferred.
    out_log = kwargs.get("log", _is_tangent(cls))
    crossing = getattr(t, "log", False) != out_log
    # Keep stored parameters in their existing encoding when only the class
    # changes: reading matrix/field could exponentiate a tangent or integrate
    # a velocity. Readers with an empty inherited storage slot must transfer
    # their public data, which is decoded from metadata instead.
    changes_class = not issubclass(cls, type(t))
    if crossing or changes_class:
        fields = set(cls.data_fields) | set(cls.derived_fields)
        if not (fields & set(kwargs)):
            if crossing or not stores(type(t), "_data"):
                kwargs[_fallback] = getattr(t, _fallback)
            elif getattr(t, "_data", None) is None:
                kwargs["data"] = t.data
    if changes_class:
        # The input and output systems follow the same rule as the data.
        # A file format such as LTA leaves the stored system empty and
        # decodes it from its record, so the rebuild would carry over
        # nothing. The public system is transferred in that case.
        for name in ("input", "output"):
            private = "_" + name
            if (
                name in kwargs
                or private in kwargs
                or not stores(cls, private)
                or getattr(t, private, None) is not None
            ):
                continue
            system = getattr(t, name)
            if system is not None:
                kwargs[name] = system
    if crossing:
        # The rebuild would otherwise carry the flag over from `t`, but
        # changing the encoding is the purpose of this conversion, so the
        # flag is set explicitly.
        kwargs["log"] = out_log
    return _convert_withdata(t, cls, **kwargs)


_SPLINE_FLAGS = ("degree", "bound", "store")


def _convert_withsplines(
    t: Transformation, cls: tx.Type[TRANSFORMATION], **kwargs
) -> TRANSFORMATION:
    # A spline flag is written back into `kwargs` only if it differs from
    # the flag of `t`. An unchanged flag carries no information, and classes
    # whose flags are derived, such as a lazy inverse, would reject it.
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

    # When the spline flags change, the values are transferred instead of
    # the data, so that the spline coefficients are fitted again.
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
    # A velocity is an encoding of a field rather than a separate type, so
    # a field is turned into a velocity with `.to(log=True)` instead.
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
def _(t: CoordinateField, cls: tx.Type[COORD], **kwargs) -> COORD:
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(t: CartesianField, cls: tx.Type[COORD], **kwargs) -> COORD:
    # A CartesianField generates its coordinates from its shape instead of
    # storing them. A class that stores coordinates must therefore receive
    # them as `values`, which it encodes under its own spline flags.
    kwargs.setdefault("values", t.values)
    return _convert_withsplines(t, cls, **kwargs)


@converter
def _(
    t: CartesianField, cls: tx.Type[CartesianField], **kwargs
) -> CartesianField:
    # A CartesianField regenerates its field and its data from its shape and
    # flags. Because the field and the data are not constructor arguments,
    # any override of them is dropped.
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
        # The `log` flag re-encodes the inner transformation and keeps the
        # map. The axes that the subspace does not act on follow the
        # identity, whose tangent is zero, so the inner tangent padded with
        # zeros is the tangent of the whole transformation. A subspace that
        # reindexes its axes has no tangent of its own.
        log = kwargs.pop("log")
        inner = kwargs.get("transformation", t.transformation)
        if inner is not None:
            kwargs["transformation"] = inner.to(log=log)
    return smart_replace(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   LOSSLESS
# ----------------------------------------------------------------------
# Each lossless converter supplies a view of the map, such as `matrix=` or
# `scale=`, and goes through `_convert_withdata`. That helper clears the data
# carried over from `t`, because the stored parameter of one family is not the
# stored parameter of another.


def _sized(
    t: Identity, make: tx.Callable[[int], tx.Any]
) -> tx.Optional[tx.Any]:
    # This helper builds the identity parameter of a family at the axis count
    # of `t`. When the endpoints do not give an axis count, the parameter is
    # left unset, which is also read as the identity.
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
        # Append a zero translation column. The result does not depend on
        # the endpoints, which may be unspecified.
        xp = array_backend(t.matrix)
        zeros = xp.zeros((*t.matrix.shape[:-1], 1), dtype=t.matrix.dtype)
        kwargs["matrix"] = xp.concatenate([t.matrix, zeros], axis=-1)
    return _convert_withdata(t, cls, **kwargs)


@converter
def _(t: SubspaceTransformation, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    # TODO: make this converter generic like the others (smart_replace?).

    # The inner affine fills the rows and columns of the axes that the
    # subspace acts on. The remaining axes form an identity that maps the
    # i-th pass-through input axis to the i-th pass-through output axis.
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
        xp = np
    else:
        inner_affine = t.transformation.compute().to(Affine)
        if not isinstance(inner_affine, Affine):
            raise ConversionError(
                "a subspace transformation that wraps a field is applied by "
                "composing it with a sampling domain, not by reduction to an "
                "affine"
            )
        inner_matrix = inner_affine.matrix
        xp = array_backend(inner_matrix)
    matrix = xp.zeros((n_out, n_in + 1))
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
def _(t: Projection, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    return _projection_to_matrix(t, cls, homogeneous=True, **kwargs)


@converter
def _(t: Projection, cls: tx.Type[LINEAR], **kwargs) -> LINEAR:
    return _projection_to_matrix(t, cls, homogeneous=False, **kwargs)


def _projection_to_matrix(
    t: Projection, cls: tx.Type[TRANSFORMATION], homogeneous: bool, **kwargs
) -> TRANSFORMATION:
    # The matrix of a projection selects the kept input axes, in order,
    # and has a row of zeros for each created axis, whose coordinate is
    # zero. Its size is read from the systems of the projection, which a
    # composer closes from a neighbour when they are missing.
    if t.is_identity():
        return convert(convert(t, Identity), cls, **kwargs)
    n_in, n_out = axis_counts(t)
    if n_in is None or n_out is None:
        raise ConversionError(
            "The number of axes of this projection is unknown, so it "
            "cannot be written as a matrix, whose size is read from it. "
            "Declare a closed input or output system, or compose the "
            "projection with a neighbour that states the number of axes."
        )
    dropped, created = axis_list(t.dropped), axis_list(t.created)
    kept = [i for i in range(n_in) if i not in dropped]
    rows = [o for o in range(n_out) if o not in created]
    if len(kept) != len(rows):
        raise ConversionError(
            f"A projection that drops the axes {dropped} and creates the "
            f"axes {created} cannot map {n_in} axes onto {n_out} axes."
        )
    if "matrix" not in kwargs:
        matrix = np.zeros((n_out, n_in + 1 if homogeneous else n_in))
        matrix[rows, kept] = 1.0
        kwargs["matrix"] = matrix
    kwargs.setdefault("input", t.input)
    kwargs.setdefault("output", t.output)
    return cls(**kwargs)


@converter
def _(t: Sequence, cls: tx.Type[AFFINE], **kwargs) -> AFFINE:
    # TODO: make this converter generic like the others (smart_replace?).

    # `compute()` leaves adjacent pieces separate only when composing them
    # raises a CompositionError. Among pieces that have an affine form, this
    # happens only between subspace transformations where the axes written by
    # one piece differ from the axes read by the next, as with the spatial
    # and temporal parts of a 4D NIfTI. Every other refusal involves a piece
    # without an affine form. A sequence left over by `compute()` can
    # therefore be reduced only if it is a chain of subspace transformations,
    # in which case their embedded affines are multiplied.
    reduced = t.compute()
    if not isinstance(reduced, Sequence):
        return convert(reduced, cls)
    pieces = reduced.transformations
    # Neighbouring pieces share a space, so an unknown axis count is taken
    # from the previous piece as the loop walks along the chain. The axis
    # counts at the two ends are read from `t`, because the result of
    # `compute()` does not keep the declared endpoints.
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
        xp = array_backend(block)
        product = xp.empty_like(block, shape=(block.shape[0], matrix.shape[1]))
        product[:, :-1] = block[:, :-1] @ matrix[:, :-1]
        product[:, -1:] = block[:, :-1] @ matrix[:, -1:] + block[:, -1:]
        matrix = product
    return cls(matrix=matrix, input=t.input, output=t.output)


@converter
def _(t: DisplacementField, cls: tx.Type[COORD], **kwargs) -> COORD:
    # TODO: make this converter generic like the others (smart_replace?).

    # The coordinates are the identity grid plus the displacement, expressed
    # in the spline encoding of `t`. Because spline fitting is linear, the
    # encoded grid can be added to the encoded data without decoding either.
    # This is exact as data, but not as a map outside the grid, nor near its
    # border for degree 3 (#313).
    flags = dict(store=t.store, degree=t.degree, bound=t.bound)
    data = t.data
    if data is not None:
        data = data + _encoded_grid(t)
    u = cls(data=data, input=t.input, output=t.output, **flags)
    # Requested flags (`store="values"`) re-encode the result within its type.
    return convert(u, cls, **kwargs) if kwargs else u


@converter
def _(t: CoordinateField, cls: tx.Type[DISP], **kwargs) -> DISP:
    # The displacement is the coordinates minus the grid, in the encoding of
    # `t`: the inverse of the converter above, with the same caveat (#313).
    flags = dict(store=t.store, degree=t.degree, bound=t.bound)
    data = t.data
    if data is not None:
        data = data - _encoded_grid(t)
    u = cls(data=data, input=t.input, output=t.output, **flags)
    # Requested flags (`store="values"`) re-encode the result within its type.
    return convert(u, cls, **kwargs) if kwargs else u


@converter
def _(t: CoordinateField, cls: tx.Type[SVF], **kwargs) -> SVF:
    # Coordinates minus the grid are a displacement, not a velocity.
    raise ConversionError(
        f"A {type(t).__name__} converts to a DisplacementField, not to a "
        f"StationaryVelocityField: coordinates minus the grid are a "
        f"displacement, whose logarithm brainhops does not compute."
    )


def _encoded_grid(t: TransformationField) -> ArrayProtocol:
    """Return the identity grid of the nodes of `t`, encoded like its data."""
    data = t.data
    xp = array_backend(data)
    # The grid is created in the backend of the data, with CuPy chunks when
    # its chunks are CuPy arrays.
    options = _creation_options(data)
    axes = (xp.arange(s, **options) for s in data.shape[:-1])
    grid = xp.meshgrid(*axes, indexing="ij")
    grid = xp.stack(grid, axis=-1).astype(data.dtype)
    return _values2data(grid, t.store, t.degree, t.bound)


# ----------------------------------------------------------------------
#   LOSSY
# ----------------------------------------------------------------------
# Each lossy converter builds its result before checking that the source keeps
# the promise of the target class, so that the result can be attached to the
# LossyConversionError when the check fails.


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
    xp = array_backend(t.matrix)
    matrix = xp.round(xp.abs(t.matrix))
    permutation = xp.argmax(matrix, axis=1)
    kwargs.setdefault("permutation", permutation)
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_permutation(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[SCALING], **kwargs) -> SCALING:
    if t.is_identity():
        return convert(convert(t, Identity), Scaling, **kwargs)
    xp = array_backend(t.matrix)
    scale = xp.diagonal(t.matrix)  # TODO: use SVD instead?
    kwargs.setdefault("scale", scale)
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_scaling(compute=True):
        raise LossyConversionError(result=u)
    return u


@converter
def _(t: Linear, cls: tx.Type[ROTATION], **kwargs) -> ROTATION:
    # A Rotation promises an orthogonal matrix with determinant +1, which it
    # inverts by transposition. A matrix that breaks this promise makes the
    # conversion lossy.
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


@converter
def _(t: Affine, cls: tx.Type[TRANSLATION], **kwargs) -> TRANSLATION:
    if t.matrix is None:
        return convert(convert(t, Identity), cls, **kwargs)
    kwargs.setdefault("translation", t.matrix[:, -1])
    u = _convert_withdata(t, cls, **kwargs)
    if not t.is_translation(compute=True):
        raise LossyConversionError(result=u)
    return u


# ----------------------------------------------------------------------
#   MULTI-HOPS
# ----------------------------------------------------------------------


def _make_converter_chain(*types: type) -> None:
    """Register a converter that walks from the first type to the last one.

    The converter converts its input to each intermediate type in turn, and
    then to the requested class. An intermediate type given as a TypeVar
    stands for its bound. The first and last types only determine the pair
    of types for which the converter is registered.
    """
    first, *hops, last = types
    hops = tuple(_concrete(hop) for hop in hops)

    @converter(first, last)
    def _(t: Transformation, cls: type, **kwargs) -> Transformation:
        # A hop that loses information does not stop the walk. Its result is
        # converted through the remaining hops, so that the loss is reported
        # with a result of the requested class rather than of an
        # intermediate type.
        lossy = False
        for hop in hops:
            try:
                t = convert(t, hop)
            except LossyConversionError as e:
                lossy, t = True, e.result
        t = convert(t, cls, **kwargs)  # overrides apply to the last hop
        if lossy:
            raise LossyConversionError(result=t)
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
_make_converter_chain(TRANSLATION, IDENTITY, LINEAR)
_make_converter_chain(TRANSLATION, IDENTITY, ROTATION)
_make_converter_chain(TRANSLATION, IDENTITY, SCALING)
_make_converter_chain(TRANSLATION, IDENTITY, PERMUTATION)
_make_converter_chain(SCALING, LINEAR, PERMUTATION)
_make_converter_chain(PERMUTATION, LINEAR, SCALING)
# A linear tangent matches both the converter of its own family and the
# converter from Linear to Rotation, so the request is ambiguous unless the
# tangent is first turned into its map.
_make_converter_chain(LINEAR_EXP, LINEAR, ROTATION)
_make_converter_chain(Sequence, AFFINE, TRANSLATION)
_make_converter_chain(Sequence, AFFINE, LINEAR)
_make_converter_chain(Sequence, AFFINE, ROTATION)
_make_converter_chain(Sequence, AFFINE, SCALING)
_make_converter_chain(Sequence, AFFINE, PERMUTATION)
