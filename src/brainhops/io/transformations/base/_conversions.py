"""
The pieces of a conversion into a file format.

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
bridge does not exist, the converter raises a
[`ConversionError`][brainhops.errors.ConversionError],
built by [`unrepresentable`][], that says what cannot be held.

A field is taken apart into what a field format stores by
[`displacement_parts`][] and [`coordinates_parts`][]: the affine from
the world to the grid of the field, and the displacement or the
position of each node of that grid. A field of displacements and a
field of coordinates are the same map inside their grid, and are only
extended differently beyond it, so each one is stored as the other
without loss. A field interpolated with another degree than the linear
interpolation that formats store is sampled at its nodes, and a field
that a format stores on a grid of its own is sampled on that grid
([`sample_on_grid`][]). Both are losses: [`held`][] then raises a
[`LossyConversionError`][brainhops.errors.LossyConversionError] that
carries the result, which `t.to(Format, lossy=True)` returns.

These helpers only take apart and check. Each converter decides, in one
place, whether what they return is held exactly by its format. Every
format builds itself from another transformation by converting it
([`converts_to`][] and [`convert_instance`][], which
`TransformationFormat.from_any` and `TransformationFormat.from_instance`
call), so that `Format.from_any(t)`, `Format.from_instance(t)` and
`t.to(Format)` are one conversion.

The converters of the data model, which know nothing of file formats,
rebuild a transformation as whatever class they are asked for and keep
its parameters. This is called relabelling, and it is wrong when the
new class means a different map. A format that is bound to coordinate
systems therefore checks its endpoints when it is built, so that a
transformation between other systems cannot be relabelled as that
format. The exact converters of such a format, one for each family of
transformations, bridge the endpoints before they build the format.

Every writable format now has converters of its own, built on these
helpers. Only the plain matrix files, which are not written, and the
NiftyReg chain that the field formats of NiftyReg share register one
shared converter
([`relabelling`][brainhops.io.transformations.base._converters.relabelling])
that relabels a transformation as the format, and they do so only when
the endpoints of the transformation are not known or are compatible
with the coordinate systems of the format, as the bound classes check
them. Otherwise it raises the error that [`unrepresentable`][] builds.
Each registration names its format and a family, as a converter does,
so it is more specific than the converters of the data model. A format
that is converted to its own class is changed by the rules of its
family, as any other transformation of that family is.
"""

__all__ = [
    "AFFINE_SOURCES",
    "FIELD_SOURCES",
    "FieldParts",
    "affine_between",
    "apply_affine",
    "cardinal_grid",
    "cardinal_parts",
    "convert_instance",
    "converts_to",
    "coordinates_parts",
    "displacement_parts",
    "displacements_of",
    "is_cardinal",
    "format_options",
    "held",
    "register_converter",
    "sample_on_grid",
    "split_affine",
    "split_field_chain",
    "UnusedOptionError",
    "undoes",
    "unrepresentable",
    "world_pieces",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.adaptors import bridge
from brainhops.datamodel._transformations.compute.convert import (
    convert,
    converter,
)
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.datamodel.systems import CoordinateSystem, VoxelCoordinateSystem
from brainhops.errors import (
    AdaptationError,
    ConversionError,
    LossyConversionError,
)

_Field = tx.Union[_xforms.DisplacementField, _xforms.CoordinatesField]

AFFINE_SOURCES: tx.Tuple[type, ...] = (
    _xforms.Identity,
    _xforms.Translation,
    _xforms.Scaling,
    _xforms.Permutation,
    _xforms.Linear,
    _xforms.Affine,
    _xforms.AffineExponential,
    _xforms.SubspaceTransformation,
    _xforms.Sequence,
)
"""
The families that a converter into an affine format is registered for.

Every family that has an affine form is listed, together with `Sequence`,
so that each one reaches the converter of the format rather than a
converter of the data model, which would keep the endpoints of the
transformation instead of bridging them to those of the format.
"""

FIELD_SOURCES: tx.Tuple[type, ...] = (
    _xforms.DisplacementField,
    _xforms.StationaryVelocityField,
    _xforms.CoordinatesField,
    _xforms.CartesianField,
    _xforms.Sequence,
)
"""
The families that a converter into a field format is registered for.

A field of displacements, a velocity, a field of coordinates and a chain
that holds one of them all reach the converter of the format, which
turns the field into what the format stores.
"""

_VOXEL = VoxelCoordinateSystem()


def register_converter(
    sources: tx.Iterable[type], cls: type
) -> tx.Callable[[tx.Callable], tx.Callable]:
    """
    Register a converter into a format for each of several families.

    A format registers its converter once for each family that it holds,
    such as every family in [`AFFINE_SOURCES`][] for an affine format, so
    that each one reaches the converter of the format.

    Parameters
    ----------
    sources : iterable of type
        The families of transformations that the converter takes.
    cls : type
        The format that the converter builds.

    Returns
    -------
    callable
        A decorator that registers its function and returns it unchanged.
    """
    sources = tuple(sources)

    def decorate(func: tx.Callable) -> tx.Callable:
        for source in sources:
            converter(source, cls)(func)
        return func

    return decorate


class UnusedOptionError(ConversionError):
    """
    A conversion was given options that the format does not take.

    [`format_options`][] raises it, with the names of the options in
    `names`. `io.save` passes the same conversion options to every
    format it may write, such as the reference image that an FSL warp
    needs, so it converts again without these options when a format
    does not take them.
    """

    def __init__(self, *args: tx.Any, names: tx.Sequence[str] = ()) -> None:
        super().__init__(*args)
        self.names = tuple(names)


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


def converts_to(cls: type, other: tx.Any) -> bool:
    """
    Whether the format `cls` builds itself from `other` by converting it.

    It does for any transformation that is not already of that format: an
    instance of the format is copied, and anything else (a file, a
    mapping) is read as the format's bases read it.

    Parameters
    ----------
    cls : type
        The format.
    other : Any
        The value that the format is built from.

    Returns
    -------
    bool
        Whether `other` is converted.
    """
    return isinstance(other, _xforms.Transformation) and not isinstance(
        other, cls
    )


def convert_instance(cls: type, other: tx.Any, *args, **kwargs) -> tx.Any:
    """
    Return `other` converted to the format `cls`, as `other.to(cls)` does.

    This is what `from_any` and `from_instance` of every format return
    for another transformation (see [`converts_to`][]), so the three give
    the same result and raise the same errors.

    Parameters
    ----------
    cls : type
        The format.
    other : Any
        The transformation to convert.
    *args : Any
        Positional arguments, which a conversion does not take.
    **kwargs : Any
        Options of the conversion.

    Returns
    -------
    Any
        The converted transformation, an instance of `cls`.

    Raises
    ------
    TypeError
        If positional arguments are given.
    ConversionError
        If the format cannot hold `other`.
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
    UnusedOptionError
        If an option is not one of `allowed`.
    """
    refused = sorted(set(options) - set(allowed))
    if refused:
        names = ", ".join(f"{name}=" for name in refused)
        takes = ", ".join(f"{name}=" for name in allowed) or "no option"
        error = unrepresentable(
            t,
            cls,
            f"{names} cannot be overridden by a conversion, which gives the "
            f"format the very map it converts, between the format's own "
            f"coordinate systems. It takes {takes}.",
        )
        raise UnusedOptionError(str(error), names=refused)
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
    one affine. A system that declares a distinct origin, such as a
    tkregister space, is bridged only to a system of its own class.

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
    xp = get_array_backend(values)
    ndim = matrix.shape[0] - 1
    linear = xp.asarray(matrix[:ndim, :ndim])
    offset = xp.asarray(matrix[:ndim, ndim])
    return xp.matmul(linear, values[..., None])[..., 0] + offset


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
    # `Sequence.compute` does), or the reason why there is none. A system
    # that declares a distinct origin, such as a tkregister space, has the
    # axes of another space but is not it, so it is bridged only to a
    # system of its own class.
    distinct = [
        getattr(system, "DISTINCT_ORIGIN", False)
        for system in (source, target)
    ]
    if (
        source is not None
        and target is not None
        and any(distinct)
        and type(source) is not type(target)
    ):
        odd = source if distinct[0] else target
        other = target if distinct[0] else source
        raise unrepresentable(
            t,
            cls,
            f"the {odd.name} space has the axes of the {other.name} space "
            f"but another origin, so one is not the other. Compose the "
            f"transformation with the affine between the two spaces first.",
        )
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


# ----------------------------------------------------------------------
#   FIELDS AS A FORMAT STORES THEM
# ----------------------------------------------------------------------


class FieldParts(tx.NamedTuple):
    """
    A field taken apart into what a field format stores.

    A field format stores the values of a field on the nodes of a grid,
    together with the affine that carries the world to that grid. These
    parts are what [`displacement_parts`][] and [`coordinates_parts`][]
    return, so that each format only arranges them as its file does.
    """

    world2grid: np.ndarray
    """The homogeneous affine from the input world to the voxels of the
    grid, of shape `(D + 1, D + 1)`."""

    field: _Field
    """The field on the voxels of the grid, interpolated with the degree
    that the format asked for. It is a field of displacements in voxels,
    or a field of coordinates in the output world. A format that stores
    the coefficients of a spline reads `field.coefficients`, and one that
    stores values reads `field.values`."""

    grid2world: np.ndarray
    """The homogeneous affine from the voxels of the grid back to the
    world, which is the inverse of `world2grid`."""

    loss: tx.Optional[str] = None
    """What the field loses, as the end of a sentence, or `None` when it
    is the same map as the transformation inside the grid. It loses when
    the transformation interpolates its field with another degree than
    the one that the format stores, or when the field is sampled on
    another grid."""

    @property
    def exact(self) -> bool:
        """Whether the field is the same map as the transformation inside
        its grid."""
        return self.loss is None


def held(result: tx.Any, parts: FieldParts) -> tx.Any:
    """
    Return the converted format, or report that the conversion loses.

    A converter builds its format from the parts of a field and returns
    it through this function. When the parts are not exact, the result
    is carried by a
    [`LossyConversionError`][brainhops.errors.LossyConversionError],
    which `t.to(cls, lossy=True)` turns back into the result.

    Parameters
    ----------
    result : Any
        The format built from `parts`.
    parts : FieldParts
        The parts it was built from.

    Returns
    -------
    Any
        `result`, when the parts are exact.

    Raises
    ------
    LossyConversionError
        If the parts are not exact.
    """
    if parts.loss is not None:
        raise LossyConversionError(
            f"The {type(result).__name__} that this transformation converts "
            f"to is not the same map: {parts.loss} Pass lossy=True to "
            f"accept it.",
            result=result,
        )
    return result


def displacement_parts(
    t: _xforms.Transformation,
    world: CoordinateSystem,
    cls: type,
    keep_log: bool = False,
    degree: int = 1,
) -> FieldParts:
    """
    Take `t` apart as a field of displacements between two affines.

    This is what a format such as a NIfTI `DISPVECT` file, an ITK
    displacement field or a grid of B-spline coefficients stores: an
    affine from the world to the voxels of a grid, displacements in
    those voxels, interpolated with the degree of the format, and the
    inverse affine back to the world. `t` is read from `world` and
    written in `world`, with the bridges between them and its own
    endpoints put on either side of it (see [`split_field_chain`][]).

    The field that `t` holds may be a field of displacements or of
    coordinates, of any degree, and the affines on either side of it
    need not undo each other. The position that each node of the grid
    maps to is computed, and the displacement is that position minus the
    node. A spline of degree 1 or more reproduces the affine functions,
    so the result is the same map as `t` inside the grid, and differs
    from it only in how it is extended beyond the grid, which a field
    format decides. A velocity is integrated into its displacement,
    unless `keep_log` asks for it as it is and the affines undo each
    other, in which case the velocity is kept. A field interpolated with
    another degree than `degree` is evaluated at its nodes and
    interpolated with `degree`, and the parts then report a loss.

    Parameters
    ----------
    t : Transformation
        The transformation to take apart.
    world : CoordinateSystem
        The world that the format maps from and to.
    cls : type
        The format, which names the error when `t` cannot be held.
    keep_log : bool, default=False
        Whether a velocity is kept as a velocity, for a format that can
        store one.
    degree : int, default=1
        The degree of the spline with which the format interpolates its
        field: 1 for the values of a dense field, 3 for the coefficients
        of a cubic B-spline.

    Returns
    -------
    FieldParts
        The affine to the grid, the displacements in voxels, the inverse
        affine, and what the field loses, if anything.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` does not hold exactly one field
        between affines.
    """
    before, field, after = split_field_chain(t, world, world, cls)
    world2grid = before
    grid2world = np.linalg.inv(before)
    sandwich = isinstance(field, _xforms.DisplacementField) and undoes(
        before, after
    )
    velocity = keep_log and sandwich and bool(field.log)
    if (
        sandwich
        and int(field.degree) == degree
        and (velocity or not field.log)
    ):
        # The field is already what the format stores, and is kept as it
        # is, with its data still unread.
        field = replace(field, input=_VOXEL, output=_VOXEL)
        return FieldParts(world2grid, field, grid2world)
    data, store, bound, loss = _node_data(field, degree, keep_log=velocity)
    if data is None and not undoes(before, after):
        raise unrepresentable(
            t,
            cls,
            "its field has no values, so the size of its grid is not "
            "known, and the affines on either side of it do not undo each "
            "other, so the format would need the displacement of every "
            "node of a grid.",
        )
    if data is None:
        result = _xforms.DisplacementField(input=_VOXEL, output=_VOXEL)
        return FieldParts(world2grid, result, grid2world, loss)
    if sandwich:
        displacements = data
    else:
        # The position of each node, in the voxels of the grid, minus the
        # node. A B-spline is linear in its coefficients and reproduces
        # the identity with the coefficient `k` at node `k`, so this holds
        # for its coefficients as it does for values.
        positions = data
        if isinstance(field, _xforms.DisplacementField):
            positions = positions + _grid_like(data)
        positions = apply_affine(before @ after, positions)
        displacements = positions - _grid_like(positions)
    result = _xforms.DisplacementField(
        data=displacements,
        log=velocity,
        degree=degree,
        store=store,
        bound=bound,
        input=_VOXEL,
        output=_VOXEL,
    )
    return FieldParts(world2grid, result, grid2world, loss)


def split_affine(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
    side: str,
) -> tx.Tuple[np.ndarray, tx.Optional[_xforms.Transformation]]:
    """
    Take the affine part of `t` out of its field.

    Some formats store an affine of their own beside a field, in their
    header: the affine that NiftyReg applies before its grid of control
    points, the initial affine of an FSL coefficient field, or another
    transform of a chain. Such a format stores what it can in that slot,
    as the tools write their own files, so that the field holds only a
    displacement.

    A transformation that holds one field of displacements between two
    affines is the map `T = A o (id + d) o B`, where `B` carries the
    input world to the grid of the field and `A` carries the grid to the
    output world. It is split exactly, without changing the field, into
    the affine `R = A B` and a field `S` between an affine and its
    inverse: `T = R o S`, with `S` placed by `B` in the input world, when
    the affine is applied last, and `T = S o R`, with `S` placed by the
    inverse of `A` in the output world, when it is applied first. A
    velocity is kept, since its flow is carried by a change of
    coordinates. A field of coordinates holds its affines in its values,
    so it is a displacement between `B` and its inverse, with no affine
    taken out. A transformation without a field is all affine.

    Parameters
    ----------
    t : Transformation
        The transformation to split.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the format maps to.
    cls : type
        The format, which names the error when `t` cannot be split.
    side : {"before", "after"}
        Whether the format applies its affine before or after its field.

    Returns
    -------
    affine : array, shape `(D + 1, D + 1)`
        The homogeneous affine `R`, from `input` to `output`.
    field : Transformation or None
        The field `S`, a chain of an affine, the field and the inverse
        affine, from `output` to `output` when `side` is `"before"` and
        from `input` to `input` otherwise. It is `None` when `t` holds no
        field.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` is neither an affine nor one
        field between affines.
    """
    if side not in ("before", "after"):
        raise ValueError(f"side must be 'before' or 'after', not {side!r}.")
    try:
        return affine_between(t, input, output, cls), None
    except ConversionError:
        pass
    before, field, after = split_field_chain(t, input, output, cls)
    if isinstance(field, _xforms.CoordinatesField):
        parts = displacement_parts(t, input, cls, degree=int(field.degree))
        before, field = parts.world2grid, parts.field
        after = parts.grid2world
        affine = np.eye(len(before))
    else:
        affine = after @ before
    if side == "after":
        world, place = input, before
    else:
        world, place = output, np.linalg.inv(after)
    chain = _xforms.Sequence(
        transformations=[
            _xforms.Affine(matrix=place[:-1], input=world, output=_VOXEL),
            replace(field, input=_VOXEL, output=_VOXEL),
            _xforms.Affine(
                matrix=np.linalg.inv(place)[:-1], input=_VOXEL, output=world
            ),
        ]
    )
    return affine, chain


def coordinates_parts(
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
    degree: int = 1,
) -> FieldParts:
    """
    Take `t` apart as an affine followed by a field of coordinates.

    This is what a format such as an SPM deformation or an MRtrix
    deformation stores: an affine from the input world to the voxels of
    a grid, and the position in the output world that each node maps
    to. `t` is read from `input` and written in `output`, with the
    bridges between them and its own endpoints put on either side of it
    (see [`split_field_chain`][]).

    The field that `t` holds may be a field of displacements or of
    coordinates. The position that each node maps to is the same in both
    cases, so the result is the same map as `t` inside the grid, and
    differs from it only in how it is extended beyond the grid. A
    velocity is integrated into its displacement. A field interpolated
    with another degree than `degree` is evaluated at its nodes and
    interpolated with `degree`, and the parts then report a loss.

    Parameters
    ----------
    t : Transformation
        The transformation to take apart.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the coordinates of the format are written in.
    cls : type
        The format, which names the error when `t` cannot be held.
    degree : int, default=1
        The degree of the spline with which the format interpolates its
        field.

    Returns
    -------
    FieldParts
        The affine to the grid, the field of coordinates in `output`,
        the inverse affine, and what the field loses, if anything.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` does not hold exactly one field
        between affines, or the field holds no values.
    """
    before, field, after = split_field_chain(t, input, output, cls)
    values, store, bound, loss = _node_data(field, degree)
    if values is None and isinstance(field, _xforms.CoordinatesField):
        raise unrepresentable(
            t,
            cls,
            "it holds no coordinates, and the format stores the coordinates "
            "of every node of its grid.",
        )
    if values is None:
        raise unrepresentable(
            t,
            cls,
            "its field has no values, so the size of its grid is not known, "
            "and the format stores the coordinates of every node of its "
            "grid.",
        )
    positions = values
    if isinstance(field, _xforms.DisplacementField):
        positions = positions + _grid_like(values)
    coordinates = apply_affine(after, positions)
    result = _xforms.CoordinatesField(
        data=coordinates,
        degree=degree,
        store=store,
        bound=bound,
        input=_VOXEL,
        output=output,
    )
    return FieldParts(before, result, np.linalg.inv(before), loss)


def sample_on_grid(
    t: _xforms.Transformation,
    world2grid: np.ndarray,
    shape: tx.Sequence[int],
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
    degree: int = 1,
) -> FieldParts:
    """
    Sample `t` on the nodes of a grid that a format imposes.

    Some formats store a field on a grid that is fixed by something else
    than the field, such as the reference image of an FSL warp or the
    knots of its B-spline. `t` is then evaluated at the nodes of that
    grid, which gives the position in `output` that each node maps to,
    and these positions are interpolated with `degree`. This is the same
    map as `t` only when `t` is itself interpolated with `degree` on the
    same grid, so the parts are exact only then.

    Parameters
    ----------
    t : Transformation
        The transformation to sample.
    world2grid : array, shape `(D + 1, D + 1)`
        The homogeneous affine from `input` to the voxels of the grid.
    shape : sequence of int
        The shape of the grid.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the coordinates are written in.
    cls : type
        The format, which names the error when `t` cannot be sampled.
    degree : int, default=1
        The degree of the spline with which the format interpolates its
        field.

    Returns
    -------
    FieldParts
        The affine to the grid, the field of coordinates in `output`, the
        inverse affine, and what the field loses.

    Raises
    ------
    ConversionError
        If a bridge between the systems does not exist.
    """
    world2grid = np.asarray(world2grid, dtype=np.float64)
    shape = tuple(int(n) for n in shape)
    try:
        native = coordinates_parts(t, input, output, cls, degree)
    except ConversionError:
        native = None
    if (
        native is not None
        and native.exact
        and native.field.data is not None
        and tuple(native.field.data.shape[:-1]) == shape
        and undoes(native.world2grid, np.linalg.inv(world2grid))
    ):
        return native
    grid2world = np.linalg.inv(world2grid)
    ndim = len(shape)
    nodes = _xforms.CartesianField(shape=shape).field
    points = apply_affine(grid2world, nodes)
    chain = [
        _xforms.CoordinatesField(field=points, input=_VOXEL, output=input),
        _bridge(input, t.input, t, cls),
        t,
        _bridge(t.output, output, t, cls),
    ]
    try:
        mapped = _xforms.Sequence(transformations=chain).compute()
        mapped = mapped.to(_xforms.CoordinatesField)
    except ConversionError as error:
        raise unrepresentable(
            t, cls, f"it cannot be sampled on the grid of the format. {error}"
        ) from error
    if mapped.data is None or mapped.data.shape[-1] != ndim:
        raise unrepresentable(
            t,
            cls,
            "it does not map the grid of the format into a space of the "
            "same dimension.",
        )
    result = _xforms.CoordinatesField(
        field=mapped.values, degree=degree, input=_VOXEL, output=output
    )
    loss = (
        f"it is evaluated on the grid of the format, and is read back with "
        f"an interpolation of degree {degree} between these samples."
    )
    return FieldParts(world2grid, result, grid2world, loss)


def _node_data(
    field: _Field, degree: int, keep_log: bool = False
) -> tx.Tuple[
    tx.Optional[ArrayProtocol],
    StoreEnum,
    tx.Any,
    tx.Optional[str],
]:
    # The data of a field in the encoding of a spline of `degree`, with
    # that encoding, its boundary condition, and what it loses against the
    # field, if anything. A field of that degree gives its coefficients,
    # on which affines act as they act on values, and keeps its boundary.
    # A field of another degree is evaluated at its nodes, and its values
    # are stored, which a format interpolates with `degree`. A velocity is
    # integrated first, unless it is kept. How the field is extended
    # beyond its grid does not count: every format extends it in its own
    # way.
    if (
        isinstance(field, _xforms.DisplacementField)
        and field.log
        and not keep_log
    ):
        field = convert(field, _xforms.DisplacementField, log=False)
    if field.data is None:
        return None, StoreEnum.values, field.bound, None
    source = int(field.degree)
    if source == degree:
        if degree > 1:
            return (
                field.coefficients,
                StoreEnum.coefficients,
                field.bound,
                None,
            )
        return field.values, StoreEnum.values, field.bound, None
    return (
        field.values,
        StoreEnum.values,
        BoundaryCondition.nearest,
        (
            f"its field is interpolated with degree {source}, and the format "
            f"interpolates the values at its nodes with degree {degree}."
        ),
    )


def _grid_like(values: ArrayProtocol) -> ArrayProtocol:
    # The voxel coordinates of the nodes of a grid, shaped and placed as an
    # array of vectors `values` on that grid, in the backend of `values`.
    backend = get_array_backend(values)
    shape = tuple(int(n) for n in values.shape[:-1])
    axes = [backend.arange(n, dtype=values.dtype) for n in shape]
    return backend.stack(backend.meshgrid(*axes, indexing="ij"), axis=-1)


# ----------------------------------------------------------------------
#   CHAINS AS A FORMAT OF SEVERAL TRANSFORMS STORES THEM
# ----------------------------------------------------------------------


def world_pieces(
    t: _xforms.Transformation,
    world: CoordinateSystem,
    cls: type,
) -> tx.List[_xforms.Transformation]:
    """
    Cut `t` into pieces that each map `world` to `world`.

    A format such as X5 or an MNI transform file stores a chain of
    transforms, each of which maps the world to the world: an affine, or
    a field placed on a grid of that world. `t` is read from `world` and
    written in `world`, with the bridges between them and its own
    endpoints put on either side of it, as in [`split_field_chain`][],
    and it may hold any number of fields.

    Each field becomes one piece, which is a chain of three: the affine
    from the world to the grid of the field, the field as it is, and an
    affine back to the world. When the leaf just before a field of
    displacements is an affine from a world system, the field keeps the
    grid that this leaf places it on, and the affines that come before
    that leaf become an affine piece of their own. Each affine is then
    stored as a transform of the file, as the tools write their own
    files, rather than in the placement of the grid. Otherwise, the
    affines that come before a field are composed into its first affine.
    A field of displacements is closed by the inverse of its first
    affine, so that it is the sandwich that a field format stores, and
    the affines that follow it start the next piece. A field of
    coordinates is closed by the bridge from its output to the world
    when there is one, by the affines that follow it when it is the last
    field, and otherwise by the inverse of its first affine. The affines
    that remain after the last field become one affine piece, which is
    left out when it is the identity. A chain without a field is one
    affine piece. The pieces, applied in turn, are the same map as `t`.

    Parameters
    ----------
    t : Transformation
        The transformation to cut.
    world : CoordinateSystem
        The world that every piece maps from and to.
    cls : type
        The format, which names the error when `t` cannot be cut.

    Returns
    -------
    list of Transformation
        The pieces, in the order in which they are applied. An affine
        piece is an `Affine` from `world` to `world`, and a field piece
        is a `Sequence` of an affine, the field and an affine.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or what lies between the fields is
        not an affine.
    """
    leaves = _leaves(t)
    fields = [i for i, leaf in enumerate(leaves) if _is_field(leaf)]
    ndim = next(
        (
            _field_ndim(leaves[i])
            for i in fields
            if _field_ndim(leaves[i]) is not None
        ),
        3,
    )

    def segment(lo: int, hi: int) -> np.ndarray:
        # The affine from the output of leaf `lo - 1` (the world when
        # `lo` is 0) to the input of leaf `hi` (the world at the end).
        # The world is bridged to the endpoints of `t` itself, which its
        # leaves may not repeat: the blocks of an ITK chain between LPS
        # spaces, for example, are made of pieces without systems.
        chain = []
        if lo == 0:
            chain.append(_bridge(world, t.input, t, cls))
            source = t.input
        else:
            source = leaves[lo - 1].output
        for leaf in leaves[lo:hi]:
            chain += [_bridge(source, leaf.input, t, cls), leaf]
            source = leaf.output
        if hi == len(leaves):
            chain += [
                _bridge(source, t.output, t, cls),
                _bridge(t.output, world, t, cls),
            ]
        else:
            chain.append(_bridge(source, leaves[hi].input, t, cls))
        return _matrix(chain, t, cls, ndim)

    def affine(
        matrix: np.ndarray,
        input: tx.Optional[CoordinateSystem],
        output: tx.Optional[CoordinateSystem],
    ) -> _xforms.Affine:
        return _xforms.Affine(matrix=matrix[:-1], input=input, output=output)

    if not fields:
        return [affine(segment(0, len(leaves)), world, world)]

    def placement(lo: int, i: int) -> tx.Optional[np.ndarray]:
        # The affine from the world to the grid of field `i`, read from
        # the leaf just before the field when that leaf is an affine from
        # a world system, or `None`. This keeps the field on the grid that
        # it was placed on.
        if i - 1 < lo or _is_field(leaves[i - 1]):
            return None
        leaf = leaves[i - 1]
        to_leaf = _world_bridge(world, leaf.input, ndim)
        if to_leaf is None:
            return None
        try:
            chain = [leaf, _bridge(leaf.output, leaves[i].input, t, cls)]
            return _matrix(chain, t, cls, ndim) @ to_leaf
        except ConversionError:
            return None

    pieces: tx.List[_xforms.Transformation] = []
    incoming = np.eye(ndim + 1)
    for k, i in enumerate(fields):
        field = leaves[i]
        lo = 0 if k == 0 else fields[k - 1] + 1
        before = segment(lo, i) @ incoming
        last = k == len(fields) - 1
        after = segment(i + 1, len(leaves)) if last else None
        place = placement(lo, i)
        if isinstance(field, _xforms.DisplacementField) and place is not None:
            # The affines that come before the placement of the field
            # become an affine piece of their own.
            first = np.linalg.inv(place) @ before
            if not undoes(first, np.eye(len(first))):
                pieces.append(affine(first, world, world))
            before = place
        if isinstance(field, _xforms.DisplacementField):
            close = np.linalg.inv(before)
        elif last:
            close = after
        else:
            close = _world_bridge(field.output, world, ndim)
            if close is None:
                close = np.linalg.inv(before)
        pieces.append(
            _xforms.Sequence(
                transformations=[
                    affine(before, world, field.input),
                    field,
                    affine(close, field.output, world),
                ]
            )
        )
        incoming = np.linalg.inv(close)
    rest = after @ incoming
    if not undoes(rest, np.eye(len(rest))):
        pieces.append(affine(rest, world, world))
    return pieces


def field_slots(
    t: _xforms.Transformation,
    world: CoordinateSystem,
    cls: type,
) -> tx.Tuple[
    tx.Optional[np.ndarray], _xforms.Transformation, tx.Optional[np.ndarray]
]:
    """
    Take `t` apart as an affine, a field on its own grid, and an affine.

    Some formats store a field with an affine on either side of it, in
    their header or as other transforms of a chain: NiftyReg applies the
    affine of the first extension of a grid of control points before the
    spline and composes the affine of the second one after it, and an
    elastix file chains transforms. Such a format stores the affines
    there, as the tools write their own files, and the field keeps the
    grid on which it is placed. `t` is cut as [`world_pieces`][] cuts
    it, and must hold one field.

    Parameters
    ----------
    t : Transformation
        The transformation to take apart.
    world : CoordinateSystem
        The world that the format maps from and to.
    cls : type
        The format, which names the error when `t` cannot be held.

    Returns
    -------
    before : array or None
        The homogeneous affine applied before the field, in `world`, or
        `None` when there is none.
    field : Transformation
        The field, as a chain of an affine from `world` to its grid, the
        field and an affine back to `world`.
    after : array or None
        The homogeneous affine applied after the field, in `world`, or
        `None` when there is none.

    Raises
    ------
    ConversionError
        If a bridge does not exist, or `t` does not hold exactly one
        field, or what lies around it is not an affine.
    """
    pieces = world_pieces(t, world, cls)
    fields = [
        k
        for k, piece in enumerate(pieces)
        if not isinstance(piece, _xforms.Affine)
    ]
    if len(fields) != 1:
        raise unrepresentable(
            t,
            cls,
            f"it holds {len(fields)} fields, and the format holds one. "
            f"Fields are not composed into one, which would resample them.",
        )
    k = fields[0]
    return (
        _composed(pieces[:k]),
        pieces[k],
        _composed(pieces[k + 1 :]),
    )


def _composed(
    pieces: tx.Sequence[_xforms.Transformation],
) -> tx.Optional[np.ndarray]:
    # The homogeneous matrix of affine pieces applied in turn, or `None`
    # when there is no piece.
    if not pieces:
        return None
    matrix = np.eye(len(pieces[0].homogeneous_matrix))
    for piece in pieces:
        matrix = np.asarray(piece.homogeneous_matrix, np.float64) @ matrix
    return matrix


def cardinal_grid(
    grid2world: np.ndarray, shape: tx.Sequence[int]
) -> np.ndarray:
    """
    Return the grid whose axes run along the world axes, nearest to a grid.

    A format such as an AFNI warp or an OME-Zarr field places its grid
    with a scale and a translation along the world axes, and a field on
    an oblique grid is sampled on this grid instead
    ([`sample_on_grid`][]). Each axis of the grid is turned onto the
    world axis that it is closest to, keeping its direction and its
    voxel size, and the centre of the grid stays where it is.

    Parameters
    ----------
    grid2world : array, shape `(D + 1, D + 1)`
        The homogeneous affine from the voxels of the grid to the world.
    shape : sequence of int
        The shape of the grid.

    Returns
    -------
    array, shape `(D + 1, D + 1)`
        The affine from the voxels of the cardinal grid to the world.
    """
    grid2world = np.asarray(grid2world, dtype=np.float64)
    ndim = len(grid2world) - 1
    linear = grid2world[:ndim, :ndim]
    sizes = np.linalg.norm(linear, axis=0)
    free = list(range(ndim))
    world_axis = [0] * ndim
    # The axes that are closest to a world axis choose first.
    weights = np.abs(linear) / np.where(sizes > 0, sizes, 1.0)
    for j in np.argsort(-weights.max(axis=0)):
        best = max(free, key=lambda a: weights[a, j])
        world_axis[j] = best
        free.remove(best)
    cardinal = np.zeros((ndim, ndim))
    for j, a in enumerate(world_axis):
        sign = 1.0 if linear[a, j] >= 0 else -1.0
        cardinal[a, j] = sign * sizes[j]
    centre = (np.asarray(shape, dtype=np.float64) - 1) / 2
    result = np.eye(ndim + 1)
    result[:ndim, :ndim] = cardinal
    result[:ndim, ndim] = (
        linear @ centre + grid2world[:ndim, ndim] - cardinal @ centre
    )
    return result


def _world_bridge(
    source: tx.Optional[CoordinateSystem],
    world: CoordinateSystem,
    ndim: int,
) -> tx.Optional[np.ndarray]:
    # The matrix of the exact bridge from a system to the world, or `None`
    # when there is none, as from a system of voxels.
    if source is None:
        return None
    try:
        chain = [bridge(source, world, allow_type_grouped_positional=True)]
        affine = _xforms.Sequence(transformations=chain).to(_xforms.Affine)
    except (AdaptationError, ConversionError):
        return None
    if affine.matrix is None:
        return np.eye(ndim + 1)
    return np.asarray(affine.homogeneous_matrix, dtype=np.float64)


def is_cardinal(grid2world: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Whether each axis of a grid runs along one axis of the world.

    Parameters
    ----------
    grid2world : array, shape `(D + 1, D + 1)`
        The homogeneous affine from the voxels of the grid to the world.
    tol : float, default=1e-6
        The tolerance on the cosine between an axis of the grid and the
        world axis that it runs along.

    Returns
    -------
    bool
        Whether the grid is cardinal.
    """
    grid2world = np.asarray(grid2world, dtype=np.float64)
    ndim = len(grid2world) - 1
    linear = grid2world[:ndim, :ndim]
    norms = np.linalg.norm(linear, axis=0)
    if not np.all(norms > 0):
        return False
    unit = np.abs(linear / norms)
    return bool(np.all(np.abs(unit.max(axis=0) - 1) < tol))


def cardinal_parts(
    parts: FieldParts,
    t: _xforms.Transformation,
    input: CoordinateSystem,
    output: CoordinateSystem,
    cls: type,
) -> FieldParts:
    """
    Return the parts of a field on a grid whose axes run along the world.

    A format that places its grid with a scale and a translation along
    the world axes, such as an AFNI warp, holds the parts of a field on
    an oblique grid only once the field is sampled on the nearest
    cardinal grid ([`cardinal_grid`][]). Parts that are already on a
    cardinal grid are returned as they are. Otherwise, `t` is sampled
    on the cardinal grid with [`sample_on_grid`][], which is a loss, and
    the result is a field of coordinates in `output`.

    Parameters
    ----------
    parts : FieldParts
        The parts of `t`, as [`displacement_parts`][] or
        [`coordinates_parts`][] give them.
    t : Transformation
        The transformation that the parts are taken from.
    input : CoordinateSystem
        The world that the format maps from.
    output : CoordinateSystem
        The world that the coordinates are written in.
    cls : type
        The format, which names the error when `t` cannot be sampled.

    Returns
    -------
    FieldParts
        The parts on a cardinal grid.
    """
    if is_cardinal(parts.grid2world):
        return parts
    data = parts.field.data
    if data is None:
        raise unrepresentable(
            t,
            cls,
            "its grid is oblique, and the format places its grid along the "
            "world axes, so the field would be sampled on another grid, "
            "which needs its values.",
        )
    shape = tuple(int(n) for n in data.shape[:-1])
    grid2world = cardinal_grid(parts.grid2world, shape)
    sampled = sample_on_grid(
        t, np.linalg.inv(grid2world), shape, input, output, cls
    )
    reason = (
        "its grid is oblique, and the format places its grid along the "
        "world axes, so it is sampled on the nearest such grid."
    )
    return sampled._replace(loss=_joined(parts.loss, reason))


def displacements_of(parts: FieldParts) -> FieldParts:
    """
    Return parts of coordinates as parts of displacements in voxels.

    A field of coordinates in the world that the grid is placed in,
    such as what [`sample_on_grid`][] returns, is the same map inside
    the grid as the field of displacements that moves each node by the
    difference between its position and the node, in the voxels of the
    grid. Parts that already hold displacements are returned as they
    are.

    Parameters
    ----------
    parts : FieldParts
        The parts, whose coordinates are in the world of the grid.

    Returns
    -------
    FieldParts
        The parts, with a field of displacements in voxels.
    """
    field = parts.field
    if isinstance(field, _xforms.DisplacementField):
        return parts
    positions = apply_affine(parts.world2grid, field.values)
    displacements = positions - _grid_like(positions)
    result = _xforms.DisplacementField(
        field=displacements,
        degree=int(field.degree),
        input=_VOXEL,
        output=_VOXEL,
    )
    return parts._replace(field=result)


def _joined(*losses: tx.Optional[str]) -> tx.Optional[str]:
    # The losses that are reported, in one sentence each.
    kept = [loss for loss in losses if loss]
    return " ".join(kept) if kept else None
