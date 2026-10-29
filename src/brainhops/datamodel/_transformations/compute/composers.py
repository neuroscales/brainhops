"""Composers for pairs of transformations.

Each composer computes `T = To @ Ti`, that is `T(x) = To(Ti(x))` on the
domain of `Ti`. The input system of `T` is the input system of `Ti`,
and its output system is the output system of `To`. The composers
assume that the coordinate systems where the two transformations meet
are compatible, which the `adaptors` module ensures. They also assume
that both transformations are fully defined, with no unset parameters.
"""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.affines import matmul

# core
from brainhops._core.backends import (
    array_backend,
    to_backend,
    to_promoted_backend,
)
from brainhops._core.bsplines import pull_field

# api
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.errors import CompositionError, ConversionError

# transformations
from ..base import Transformation
from ..batched import (
    BatchedTransformation,
    apply_to_points,
    broadcast_element,
    compose_with,
    match_positions,
)
from ..concrete import (
    Affine,
    CoordinateField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
    _grid_like,
)
from ..meta import (
    Projection,
    SubspaceTransformation,
    _close_projection,
    _close_subspace,
)
from ..sequence import Sequence, _interpolates
from .compose import compose, composer
from .prune import close_counts, read_pattern, restrict_block, stage_counts
from .simplifiers import section_before
from .simplify import SimplifyTable
from .utils import UNREADABLE, axis_counts

# ----------------------------------------------------------------------
#     SEQUENCE
# ----------------------------------------------------------------------


# A sequence composes with its neighbour as the chain of their members,
# which is computed.


def _chain(To: Transformation, Ti: Transformation) -> Transformation:
    def members(t: Transformation) -> tx.List[Transformation]:
        return list(t.transformations) if isinstance(t, Sequence) else [t]

    return Sequence(
        transformations=members(Ti) + members(To),
        input=Ti.input,
        output=To.output,
    ).compute()


@composer
def _(To: Sequence, Ti: Transformation) -> Transformation:
    return _chain(To, Ti)


@composer
def _(To: Transformation, Ti: Sequence) -> Transformation:
    return _chain(To, Ti)


@composer
def _(To: Sequence, Ti: Sequence) -> Transformation:
    return _chain(To, Ti)


# ----------------------------------------------------------------------
#     SAME KIND (AFFINE-ish)
# ----------------------------------------------------------------------


@composer
def _(To: Affine, Ti: Affine) -> Affine:
    # A NaN in the translation of `Ti` is a free axis, which only the
    # outputs that read it inherit (see `matmul` in
    # `brainhops._core.affines`).
    A = matmul(To.matrix, Ti.matrix, zero_cancels_nan=True)
    return Affine(matrix=A, input=Ti.input, output=To.output)


@composer
def _(To: Linear, Ti: Linear) -> Linear:
    return Linear(
        matrix=To.matrix @ Ti.matrix, input=Ti.input, output=To.output
    )


@composer
def _(To: Rotation, Ti: Rotation) -> Rotation:
    # Rotations form a group, so the product of two rotations is a rotation.
    return Rotation(
        matrix=To.matrix @ Ti.matrix, input=Ti.input, output=To.output
    )


@composer
def _(To: Permutation, Ti: Permutation) -> Permutation:
    return Permutation(
        permutation=Ti.permutation[To.permutation],
        input=Ti.input,
        output=To.output,
    )


@composer
def _(To: Scaling, Ti: Scaling) -> Scaling:
    return Scaling(scale=To.scale * Ti.scale, input=Ti.input, output=To.output)


@composer
def _(To: Translation, Ti: Translation) -> Translation:
    return Translation(
        translation=To.translation + Ti.translation,
        input=Ti.input,
        output=To.output,
    )


_LinearIsh = tx.Union[Linear, Scaling, Permutation]
_AffineIsh = tx.Union[_LinearIsh, Affine, Translation]


@composer
def _(To: _LinearIsh, Ti: _LinearIsh) -> Linear:
    return (To.to(Linear) @ Ti.to(Linear)).compute()


@composer
def _(To: _AffineIsh, Ti: _AffineIsh) -> Affine:
    return (To.to(Affine) @ Ti.to(Affine)).compute()


# ----------------------------------------------------------------------
#     AFFINE o COORDINATES
# ----------------------------------------------------------------------


@composer
def _(To: Translation, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    field, translation = to_promoted_backend(Ti.field, To.translation)
    field = field + translation
    return CoordinateField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Scaling, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    data, scale = to_promoted_backend(Ti.data, To.scale)
    data = data * scale
    return CoordinateField(
        data=data,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Permutation, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    data = Ti.data[..., To.permutation]
    return CoordinateField(
        data=data,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Linear, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    field, matrix = to_promoted_backend(Ti.field, To.matrix)
    field = field @ matrix.T
    return CoordinateField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Affine, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    field, matrix = to_promoted_backend(Ti.field, To.matrix)
    field = field @ matrix[:, :-1].T + matrix[:, -1]
    return CoordinateField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


# ----------------------------------------------------------------------
#     AFFINE o DISPLACEMENTS
# ----------------------------------------------------------------------


@composer
def _(To: Translation, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.compute()
    field, translation = to_promoted_backend(Ti.field, To.translation)
    field = field + translation
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Scaling, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.compute()
    field, scale = to_promoted_backend(Ti.field, To.scale)
    grid = _grid_like(field)
    field = scale * field + (scale - 1) * grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Permutation, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.compute()
    field = to_backend(Ti.field)
    grid = _grid_like(field)
    field = (grid + field)[..., To.permutation] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


def _check_square(To: tx.Union[Linear, Affine]) -> None:
    # A displacement field maps a space to itself and cannot absorb a matrix
    # that changes the number of axes. The pair is refused, and the compose
    # pass keeps both transformations side by side.
    rows, cols = To.matrix.shape
    if isinstance(To, Affine):
        cols -= 1
    if rows != cols:
        raise CompositionError(
            "Cannot fold a matrix that changes the number of axes into a "
            "displacement field."
        )


@composer
def _(To: Linear, Ti: DisplacementField) -> DisplacementField:
    _check_square(To)
    Ti = Ti.compute()
    field, matrix = to_promoted_backend(Ti.field, To.matrix)
    grid = _grid_like(field)
    field = (grid + field) @ matrix.T - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Affine, Ti: DisplacementField) -> DisplacementField:
    _check_square(To)
    Ti = Ti.compute()
    field, matrix = to_promoted_backend(Ti.field, To.matrix)
    grid = _grid_like(field)
    field = (grid + field) @ matrix[:, :-1].T + matrix[:, -1] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


# ----------------------------------------------------------------------
#     DISPLACEMENTS & COORDINATES
# ----------------------------------------------------------------------


@composer
def _(To: DisplacementField, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.compute()
    # Read the displacement as spline coefficients. The conversion matters
    # because a StationaryVelocityField stores its velocity, not its
    # displacement, under `data`. The three arrays are matched before
    # anything is computed from them.
    coeffs, coords, field = to_promoted_backend(
        To.to(log=False, store="coefficients").data,
        Ti.to(CoordinateField).field,
        Ti.field,
    )
    moved = pull_field(
        coeffs,
        coords=coords,
        degree=To.degree,
        bound=To.bound,
        coeff=True,
    )
    field = moved + field
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=To.degree,
        bound=To.bound,
        store=To.store,
    )


@composer
def _(To: DisplacementField, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    coeffs, coords = to_promoted_backend(
        To.to(log=False, store="coefficients").data,
        Ti.to(CoordinateField).field,
    )
    moved = pull_field(
        coeffs,
        coords=coords,
        degree=To.degree,
        bound=To.bound,
        coeff=True,
    )
    field = moved + coords
    return CoordinateField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: CoordinateField, Ti: CoordinateField) -> CoordinateField:
    Ti = Ti.compute()
    field = pull_field(
        To.to(store="coefficients").data,
        coords=Ti.field,
        degree=To.degree,
        bound=To.bound,
        coeff=True,
    )
    return CoordinateField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


# ----------------------------------------------------------------------
#     SUBSPACE
# ----------------------------------------------------------------------


@composer
def _(To: SubspaceTransformation, Ti: CoordinateField) -> CoordinateField:
    # The components that the subspace acts on go through the inner
    # transformation, and the other components pass through unchanged. The
    # inner transformation is evaluated only at the coordinates of the axes
    # it acts on, so the other axes are never tiled to the full size.
    Ti = Ti.compute()
    x = Ti.field
    xp = array_backend(x)
    if To.input_axes is None or To.output_axes is None:
        raise CompositionError(
            "A subspace transform composes with a field only when it names "
            "the axes it acts on, but its input or output axes are unset."
        )
    in_axes = [int(i) for i in To.input_axes]
    out_axes = [int(i) for i in To.output_axes]
    if To.transformation is None:
        # A missing inner transformation stands for the identity. When the
        # input and output axes are the same, the field only receives a new
        # output system. Otherwise, the subspace moves the values of the input
        # axes to the output axes without changing them.
        if in_axes == out_axes:
            return replace(Ti, output=To.output)
        acted = x[..., in_axes]
    else:
        if _interpolates(To.transformation):
            # Looking up the axes by position reads the axes that the input
            # system declares, even when the system is open.
            axes = get_axes(To.input)
            for i in in_axes:
                axis = axes.at(i)
                if getattr(axis, "discrete", None):
                    raise CompositionError(
                        "Cannot apply an interpolating transform along the "
                        "discrete axis {!r}. A field is sampled between grid "
                        "points, which a discrete axis does not allow.".format(
                            getattr(axis, "name", None)
                            or getattr(axis, "type", None)
                        )
                    )
        domain = CoordinateField(
            field=x[..., in_axes],
            degree=Ti.degree,
            bound=Ti.bound,
            store="values",
        )
        result = Sequence([domain, To.transformation]).compute()
        if isinstance(result, DisplacementField):
            result = result.to(CoordinateField)
        if not isinstance(result, CoordinateField):
            raise CompositionError(
                "The inner transform of a subspace transform did not reduce "
                "to a field of coordinates, so it cannot be applied to a "
                "field."
            )
        acted = result.field

    # The pass-through components fill, in order, the output positions that
    # the subspace does not write, as when a subspace is reduced to an
    # affine.
    n = x.shape[-1]
    acted_in = set(in_axes)
    acted_out = set(out_axes)
    passthrough_in = [i for i in range(n) if i not in acted_in]
    passthrough_out = [o for o in range(n) if o not in acted_out]
    columns: tx.List[tx.Any] = [None] * n
    for k, o in enumerate(out_axes):
        columns[o] = acted[..., k]
    for o, i in zip(passthrough_out, passthrough_in):
        columns[o] = x[..., i]
    y = xp.stack(columns, axis=-1)

    return CoordinateField(
        field=y,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: _AffineIsh, Ti: SubspaceTransformation) -> Affine:
    # A subspace that wraps a field cannot be reduced to an affine. Such a
    # subspace is applied by composing it with a sampling domain instead.
    if _interpolates(Ti):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    # The output space of the subspace is the input space of the affine. When
    # the systems of the subspace do not state how many axes the full space
    # has, the input axis count of the affine supplies that number.
    Ti = _close_subspace(Ti, n_out=axis_counts(To)[0])
    return compose(To, Ti.to(Affine))


@composer
def _(To: SubspaceTransformation, Ti: _AffineIsh) -> Affine:
    # This composer mirrors the one above, with the subspace on the left.
    if _interpolates(To):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    To = _close_subspace(To, n_in=axis_counts(Ti)[1])
    return compose(To.to(Affine), Ti)


# ----------------------------------------------------------------------
#     PROJECTION o AFFINE
# ----------------------------------------------------------------------
# A projection is composed as its matrix, which selects the kept axes and has
# a row of zeros for each created axis. The size of the matrix comes from the
# systems of the projection or, when they are missing, from the neighbour,
# since the space between two transformations is a single space. Two
# projections never reach these composers, because a pair simplifier merges
# them first.


def _projection_matrix(t: Projection) -> Affine:
    # Without a number of axes, the projection stays next to its neighbour,
    # which the sequence engine does on a CompositionError.
    try:
        return t.to(Affine)
    except ConversionError as error:
        raise CompositionError(str(error)) from error


@composer
def _(To: Projection, Ti: _AffineIsh) -> Affine:
    To = _close_projection(To, n_in=axis_counts(Ti)[1])
    return compose(_projection_matrix(To), Ti)


@composer
def _(To: _AffineIsh, Ti: Projection) -> Affine:
    Ti = _close_projection(Ti, n_out=axis_counts(To)[0])
    return compose(To, _projection_matrix(Ti))


@composer
def _(To: Projection, Ti: SubspaceTransformation) -> Transformation:
    # The subspace composer that receives the matrix of the projection
    # refuses a subspace that wraps a field, and closes an open subspace.
    To = _close_projection(To, n_in=axis_counts(Ti)[1])
    return compose(_projection_matrix(To), Ti)


@composer
def _(To: SubspaceTransformation, Ti: Projection) -> Transformation:
    Ti = _close_projection(Ti, n_out=axis_counts(To)[0])
    return compose(To, _projection_matrix(Ti))


@composer
def _(
    To: SubspaceTransformation,
    Ti: SubspaceTransformation,
) -> Transformation:
    # Two subspace transformations compose into one only when the axes written
    # by the first are the axes read by the second. The inner transformations
    # are composed through a Sequence, which cancels an inner transformation
    # against its inverse before any field is materialized.
    if (
        To.input_axes is None
        or Ti.output_axes is None
        or list(To.input_axes) != list(Ti.output_axes)
    ):
        raise CompositionError(
            "Two subspace transforms compose only when they act on the same "
            "axes, but the axes the first writes differ from the axes the "
            "second reads."
        )
    inner_prev = Ti.transformation or Identity()
    inner_next = To.transformation or Identity()
    inner = Sequence([inner_prev, inner_next]).compute()
    # An identity inner transformation still moves values between axes when
    # the axes read by the first subspace differ from the axes written by the
    # second. The result therefore collapses to Identity only when these axes
    # match.
    same_axes = (Ti.input_axes is None) == (To.output_axes is None) and (
        Ti.input_axes is None or list(Ti.input_axes) == list(To.output_axes)
    )
    if isinstance(inner, Identity) and same_axes:
        return Identity(input=Ti.input, output=To.output)
    return SubspaceTransformation(
        transformation=inner,
        input_axes=Ti.input_axes,
        output_axes=To.output_axes,
        input=Ti.input,
        output=To.output,
    )


@composer
def _(To: SubspaceTransformation, Ti: DisplacementField) -> DisplacementField:
    # Read the displacements as coordinates, apply the subspace, and subtract
    # the grid again.
    Ti = Ti.compute()
    y = compose(To, Ti.to(CoordinateField))
    field = y.field - _grid_like(y.field)
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


# ----------------------------------------------------------------------
#     BATCHES
# ----------------------------------------------------------------------
# Two batches that meet on the same axes compose element by element. A
# batch composes with a transformation that carries its batch axes
# through unchanged, and that acts on the axes of its elements only, as
# a batch of the compositions of each element with that transformation.
# A batch that follows a transformation that creates some batch axes at
# a value reduces to the elements at that value. A batch applied to a
# field of coordinates applies to each point the element that the point
# selects. Any other neighbour reads or writes a batch axis in a way
# that no batch can hold, such as a scaling of the time axis, and stays
# next to the batch.

_NUMERIC = SimplifyTable({None: SimplifyPolicy.numeric})
"""Policy under which the composers read their operands, values included."""


@composer
def _(
    To: BatchedTransformation, Ti: BatchedTransformation
) -> BatchedTransformation:
    if (
        To.batch_input_axes != Ti.batch_output_axes
        or To.input_axes != Ti.output_axes
    ):
        raise CompositionError(
            "Two batches compose only when the first one writes the batch "
            "axes and the axes of the elements that the second one reads, "
            "in the same order."
        )
    positions = [
        _shared_positions(outer, inner)
        for outer, inner in zip(To.positions, Ti.positions)
    ]
    return compose_with(
        To,
        Ti,
        input_axes=Ti.input_axes,
        output_axes=To.output_axes,
        positions=positions,
        input=Ti.input,
        output=To.output,
    )


@composer
def _(To: BatchedTransformation, Ti: Transformation) -> Transformation:
    # A composer reads values, so the selection reads the values of `Ti`
    # whatever the policy of the caller, and the chain that it gives is
    # computed with the default mode and policy of `compute`, as the
    # composers of a sequence compute their chain.
    section = section_before(Ti, To, _NUMERIC)
    if section is not None:
        return section.compute()
    counts = _counts(Ti, To)
    piece, carried = _split_at_batch(Ti, counts, To, before=True)
    fields = {
        "input_axes": [c for c in range(counts[0]) if c not in carried],
        "output_axes": To.output_axes,
        "positions": To.positions,
        "input": Ti.input,
        "output": To.output,
    }
    if isinstance(piece, Identity):
        return replace(To, **fields)
    piece = broadcast_element(piece, To.batch_ndim)
    return compose_with(To, piece, **fields)


@composer
def _(To: Transformation, Ti: BatchedTransformation) -> Transformation:
    counts = _counts(Ti, To)
    piece, carried = _split_at_batch(To, counts[1:], Ti, before=False)
    fields = {
        "input_axes": Ti.input_axes,
        "output_axes": [r for r in range(counts[2]) if r not in carried],
        "positions": Ti.positions,
        "input": Ti.input,
        "output": To.output,
    }
    if isinstance(piece, Identity):
        return replace(Ti, **fields)
    piece = broadcast_element(piece, Ti.batch_ndim)
    return compose_with(piece, Ti, **fields)


@composer
def _(To: BatchedTransformation, Ti: CoordinateField) -> CoordinateField:
    # The batch coordinates of each point select its element, and they
    # are carried to the output unchanged, even for a point whose
    # coordinates match no element and whose other outputs are NaN.
    Ti = Ti.compute()
    field = np.asarray(to_backend(Ti.field, "numpy"), dtype=float)
    n_in, n_out = To.counts
    if field.shape[-1] != n_in:
        raise CompositionError(
            f"A batch of {n_in} input axes cannot be applied to a field of "
            f"{field.shape[-1]} components."
        )
    points = field.reshape(-1, n_in)
    coordinates = points[:, To.batch_input_axes]
    index = np.stack(
        [
            match_positions(positions, coordinates[:, j])
            for j, positions in enumerate(To.positions)
        ],
        axis=-1,
    )
    valid = np.all(index >= 0, axis=-1)
    values = np.empty((len(points), n_out))
    values[:, To.output_axes] = apply_to_points(
        To, points[:, To.input_axes], index, valid
    )
    values[:, To.batch_output_axes] = coordinates
    return CoordinateField(
        field=values.reshape(field.shape[:-1] + (n_out,)),
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: BatchedTransformation, Ti: DisplacementField) -> Transformation:
    # The displacements are read as coordinates, and the grid is
    # subtracted again when the batch maps the space to itself.
    Ti = Ti.compute()
    y = compose(To, Ti.to(CoordinateField))
    if To.counts[0] != To.counts[1]:
        return y
    return DisplacementField(
        field=y.field - _grid_like(y.field),
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


# A batch next to a sequence composes as the chain of their members, as
# any other transformation does. These two composers are more specific
# than both the composers of a batch with any transformation and the
# composers of a sequence with any transformation, which would otherwise
# tie.


@composer
def _(To: BatchedTransformation, Ti: Sequence) -> Transformation:
    return _chain(To, Ti)


@composer
def _(To: Sequence, Ti: BatchedTransformation) -> Transformation:
    return _chain(To, Ti)


def _shared_positions(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    # Return the positions along a batch axis that two batches share. A
    # batch axis of length 1 is broadcast, and takes the positions of
    # the other batch.
    if len(outer) == 1 and len(inner) > 1:
        return inner
    if len(inner) == 1 and len(outer) > 1:
        return outer
    if len(outer) != len(inner) or np.any(
        match_positions(inner, outer) != np.arange(len(outer))
    ):
        raise CompositionError(
            f"Two batches compose only when their elements sit at the same "
            f"positions, but they sit at {outer.tolist()} and at "
            f"{inner.tolist()}."
        )
    return inner


def _counts(Ti: Transformation, To: Transformation) -> tx.List[int]:
    # Return the numbers of axes before, between and after two
    # neighbours, and raise when they are unknown or disagree.
    counts = stage_counts([Ti, To])
    if counts is None:
        raise CompositionError(
            f"A {type(To).__name__} cannot follow a {type(Ti).__name__} "
            f"whose numbers of axes are unknown or disagree with its own."
        )
    return counts


def _split_at_batch(
    t: Transformation,
    counts: tx.Sequence[int],
    batch: BatchedTransformation,
    before: bool,
) -> tx.Tuple[Transformation, tx.List[int]]:
    # Split a neighbour of a batch into the part that carries the batch
    # axes through unchanged and the part that acts on the axes of the
    # elements. `counts` holds the numbers of axes before and after `t`.
    # Return that second part, and the positions of the batch axes on
    # the side of `t` that does not touch the batch. Raise when `t`
    # reads or writes a batch axis in any other way.
    refused = CompositionError(
        "A batch composes with a transformation only when that "
        "transformation carries the batch axes through unchanged, in "
        "their order, and acts on the axes of the elements only, in "
        "their order."
    )
    t = close_counts(t, counts[0], counts[1])
    pattern = UNREADABLE if t is None else read_pattern(t, _NUMERIC)
    if pattern is UNREADABLE:
        raise refused
    # The rows of `links` are the axes that `t` shares with the batch.
    pattern = np.asarray(pattern, dtype=bool)
    links = pattern if before else pattern.T
    if before:
        shared, inner = batch.batch_input_axes, batch.input_axes
    else:
        shared, inner = batch.batch_output_axes, batch.output_axes
    carried = []
    for axis in shared:
        partners = np.nonzero(links[axis])[0]
        if len(partners) != 1 or links[:, partners[0]].sum() != 1:
            raise refused
        carried.append(int(partners[0]))
    if carried != sorted(set(carried)) or inner != sorted(inner):
        raise refused
    outer = [a for a in range(links.shape[1]) if a not in carried]
    if before:
        through = restrict_block(t, carried, shared, _NUMERIC)
        piece = restrict_block(t, outer, inner, _NUMERIC)
    else:
        through = restrict_block(t, shared, carried, _NUMERIC)
        piece = restrict_block(t, inner, outer, _NUMERIC)
    if through is None or not through.is_identity(compute=True):
        raise refused
    if piece is None:
        raise refused
    n_in, n_out = (
        (len(outer), len(inner)) if before else (len(inner), len(outer))
    )
    return close_counts(piece, n_in, n_out) or piece, carried
