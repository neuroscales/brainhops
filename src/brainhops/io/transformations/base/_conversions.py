"""
The pieces of an exact conversion into a file format.

A format holds a map between the coordinate systems it is defined on:
NIfTI's voxels and RAS, say. A general transformation is held by it only
when it is the same map. Its endpoints are read from the transformation
itself, not from its class, and the exact bridge between them and the
format's is put on either side of it, as composition does (with the
adaptor's `bridge`): LPS becomes RAS by flipping two axes, and two voxel
systems are paired axis by axis. A system that is not known is taken to
be the format's.

What remains is then read as the format reads its own content: one
affine, or one field between two affines. When it is not that, or a
bridge does not exist, nothing is approximated: the converter raises a
[`ConversionError`][brainhops.errors.ConversionError],
built by [`unrepresentable`][], that says what cannot be held.

These helpers only take apart and check. Each converter decides, in one
place, whether what they return is held exactly by its format. A format
that has converters builds itself from another transformation through
them ([`converts_to`][] and [`convert_instance`][]), so that
`Format.from_any(t)`, `Format.from_instance(t)` and `t.to(Format)` are
one conversion.

The generic converters of the data model rebuild a transformation as the
class they are asked for. A format bound to coordinate systems checks its
endpoints as it is built, so a transformation between other systems is
not relabelled as it, and its exact converters, named for each family,
bridge the endpoints first.
A format that derives its systems from what it holds cannot check them,
and refuses each family with [`no_exact_conversion`][] until it has
exact converters. A format converted to its own class is changed by the
rules of its family, as any transformation of that family.
"""

__all__ = [
    "affine_between",
    "apply_affine",
    "convert_instance",
    "converts_to",
    "format_options",
    "no_exact_conversion",
    "split_field_chain",
    "undoes",
    "unrepresentable",
]

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.adaptors import bridge
from brainhops.datamodel._transformations.compute.convert import convert
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import AdaptationError, ConversionError

_Field = tx.Union[_xforms.DisplacementField, _xforms.CoordinatesField]


def unrepresentable(
    t: _xforms.Transformation, cls: type, reason: str
) -> ConversionError:
    """
    The error a converter raises when its format cannot hold `t` exactly.

    Parameters
    ----------
    t : Transformation
        The transformation being converted.
    cls : type
        The format it is converted to.
    reason : str
        What the format cannot hold, as the end of a sentence.
    """
    return ConversionError(
        f"This {type(t).__name__} cannot be held exactly by a "
        f"{cls.__name__}: {reason}"
    )


def no_exact_conversion(
    t: _xforms.Transformation, cls: type
) -> ConversionError:
    """
    The error a converter raises when its format has no exact conversion
    yet.

    The format may hold some transformations of the family of `t`, but
    which ones, and how, is for an exact converter to decide (#312). Until
    it has one, `t` is refused: the format derives its systems from what it
    holds, so `t` relabelled as it would not be what it says it holds.
    """
    return unrepresentable(
        t,
        cls,
        "no exact conversion into this format is implemented yet (#312), "
        "and a transformation is not relabelled as a format. Build the "
        "format from its own parameters instead.",
    )


def converts_to(cls: type, other: tx.Any) -> bool:
    """
    Whether the format `cls` builds itself from `other` by converting it.

    It does for any transformation that is not already of that format: an
    instance of the format is copied, and anything else (a file, a
    mapping) is read as the format's bases read it.
    """
    return isinstance(other, _xforms.Transformation) and not isinstance(
        other, cls
    )


def convert_instance(cls: type, other: tx.Any, *args, **kwargs) -> tx.Any:
    """
    `other` converted to the format `cls`, exactly as `other.to(cls)`.

    This is what `from_any` and `from_instance` of a format with
    converters return for another transformation (see [`converts_to`][]),
    so the three give the same result and raise the same errors.
    """
    if args:
        raise TypeError(
            f"{cls.__name__} converts a transformation with keyword options "
            f"only, but was given {len(args)} positional argument(s)."
        )
    return convert(other, cls, **kwargs)


def format_options(
    t: _xforms.Transformation,
    cls: type,
    options: tx.Dict[str, tx.Any],
    allowed: tx.Iterable[str] = (),
) -> tx.Dict[str, tx.Any]:
    """
    The options of a conversion that the format `cls` takes as they are.

    A converter decides what map the format holds, so nothing it is given
    may change that map afterwards: neither the endpoints (`input=`,
    `output=`), nor the map itself (`matrix=`, `field=`, `data=`,
    `transformations=`), nor anything else that is not one of the
    format's own options, which are `allowed`.

    Raises
    ------
    ConversionError
        If an option is not one of `allowed`.
    """
    refused = sorted(set(options) - set(allowed))
    if refused:
        names = ", ".join(f"{name}=" for name in refused)
        takes = ", ".join(f"{name}=" for name in allowed) or "no option"
        raise unrepresentable(
            t,
            cls,
            f"{names} cannot be overridden by a conversion, which gives the "
            f"format the very map it converts, between the format's own "
            f"coordinate systems. It takes {takes}.",
        )
    return options


def affine_between(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
) -> np.ndarray:
    """
    The homogeneous matrix of `t`, read from `input` and written in
    `output`.

    The bridges from `input` to the input of `t`, and from the output of
    `t` to `output`, are put around it, and the three are composed into
    one affine.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` is not an affine (a field, or a
        chain that holds one).
    """
    chain = [
        _bridge(input, t.input, t, cls),
        t,
        _bridge(t.output, output, t, cls),
    ]
    return _matrix(chain, t, cls)


def split_field_chain(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
) -> tx.Tuple[np.ndarray, _Field, np.ndarray]:
    """
    Take `t`, read from `input` and written in `output`, apart around its
    field.

    `t` is a field, or a chain that holds one field and affines. The
    bridges from `input` and to `output` are put around it, and the
    affines on either side of the field are composed.

    Returns
    -------
    before : array, shape `(D + 1, D + 1)`
        The affine that carries `input` to the grid of the field.
    field : DisplacementField or CoordinatesField
        The field, as it is.
    after : array, shape `(D + 1, D + 1)`
        The affine that carries the output of the field to `output`.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` does not hold exactly one field,
        or what is on either side of it is not an affine.
    """
    leaves = [
        _bridge(input, t.input, t, cls),
        *_leaves(t),
        _bridge(t.output, output, t, cls),
    ]
    found = [i for i, leaf in enumerate(leaves) if _is_field(leaf)]
    if len(found) != 1:
        raise unrepresentable(
            t,
            cls,
            f"it holds {len(found)} fields, and the format holds one. "
            f"Fields are not composed into one, which would resample them.",
        )
    i = found[0]
    field = leaves[i]
    before = [*leaves[:i], _bridge(leaves[i - 1].output, field.input, t, cls)]
    after = [
        _bridge(field.output, leaves[i + 1].input, t, cls),
        *leaves[i + 1 :],
    ]
    ndim = _field_ndim(field)
    return (
        _matrix(before, t, cls, ndim),
        field,
        _matrix(after, t, cls, ndim),
    )


def undoes(before: np.ndarray, after: np.ndarray) -> bool:
    """
    Whether the affine `after` undoes the affine `before`.

    They undo each other when their product is the identity, up to the
    rounding of the float arithmetic that inverted one into the other: a
    matrix and its inverse, computed in floats, never multiply to the
    identity exactly. That is the only tolerance taken, and it is
    relative to the size of the matrices.
    """
    product = after @ before
    scale = np.abs(after).max() * np.abs(before).max() * len(product)
    tolerance = 64 * np.finfo(np.float64).eps * max(scale, 1.0)
    return bool(
        np.allclose(product, np.eye(len(product)), rtol=0, atol=tolerance)
    )


def apply_affine(matrix: np.ndarray, values: ArrayProtocol) -> ArrayProtocol:
    """
    The affine `matrix` applied to an array of vectors.

    Parameters
    ----------
    matrix : array, shape `(D + 1, D + 1)`
        A homogeneous matrix.
    values : array, shape `(*shape, D)`
        The vectors, such as the values of a field of coordinates.

    Returns
    -------
    array, shape `(*shape, D)`
        Each vector `x` mapped to `matrix @ [x, 1]`. The array keeps the
        backend of `values`.
    """
    backend = get_array_backend(values)
    ndim = matrix.shape[0] - 1
    linear = backend.asarray(matrix[:ndim, :ndim])
    offset = backend.asarray(matrix[:ndim, ndim])
    return backend.matmul(linear, values[..., None])[..., 0] + offset


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _bridge(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
    t: _xforms.Transformation,
    cls: type,
) -> _xforms.Transformation:
    # The exact bridge between two systems, built as a composition builds
    # it (pairing still-unmatched axes by position within their type, as
    # `Sequence.compute` does), or the reason why there is none.
    try:
        return bridge(source, target, allow_type_grouped_positional=True)
    except AdaptationError as error:
        raise unrepresentable(t, cls, str(error)) from error


def _leaves(t: _xforms.Transformation) -> tx.List[_xforms.Transformation]:
    # The transformations a chain is made of, nested chains unrolled.
    if not isinstance(t, _xforms.Sequence):
        return [t]
    return [
        leaf for piece in t.transformations or () for leaf in _leaves(piece)
    ]


def _is_field(t: _xforms.Transformation) -> bool:
    return isinstance(t, (_xforms.DisplacementField, _xforms.CoordinatesField))


def _field_ndim(field: _Field) -> tx.Optional[int]:
    # The number of components of a field, read off its data.
    data = field.data
    return None if data is None else int(data.shape[-1])


def _matrix(
    chain: tx.List[_xforms.Transformation],
    t: _xforms.Transformation,
    cls: type,
    ndim: tx.Optional[int] = None,
) -> np.ndarray:
    # The homogeneous matrix of a chain of affines. A chain of identities
    # between unknown systems has no size of its own, and takes `ndim`.
    try:
        affine = _xforms.Sequence(transformations=chain).to(_xforms.Affine)
    except ConversionError as error:
        raise unrepresentable(
            t, cls, f"what it holds is not an affine there. {error}"
        ) from error
    if affine.matrix is None:
        if ndim is None:
            raise unrepresentable(
                t, cls, "the number of its dimensions is not known."
            )
        return np.eye(ndim + 1)
    return np.asarray(affine.homogeneous_matrix, dtype=np.float64)
