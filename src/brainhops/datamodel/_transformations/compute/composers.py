"""Composers for pairs of transformations.

Each composer computes `T = To @ Ti`, that is `T(x) = To(Ti(x))` on the
domain of `Ti`. The input system of `T` is the input system of `Ti`,
and its output system is the output system of `To`. The composers
assume that the coordinate systems where the two transformations meet
are compatible, which the `adaptors` module ensures. They also assume
that both transformations are fully defined, with no unset parameters.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core._internal_api import (
    array_backend,
    matmul,
    pull_field,
    to_backend,
    to_promoted_backend,
)

# api
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import CompositionError, ConversionError

# transformations
from ..base import Transformation
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
    grid_like,
)
from ..meta import (
    Projection,
    SubspaceTransformation,
    close_projection,
    close_subspace,
)
from ..sequence import Sequence, interpolates
from .compose import compose, composer
from .utils import axis_counts

# ----------------------------------------------------------------------
#     SEQUENCE
# ----------------------------------------------------------------------


@composer
def _(To: Sequence, Ti: Transformation) -> Transformation:
    return Sequence(
        transformations=[Ti] + list(To.transformations),
        input=Ti.input,
        output=To.output,
    ).compute()


@composer
def _(To: Transformation, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + [To],
        input=Ti.input,
        output=To.output,
    ).compute()


@composer
def _(To: Sequence, Ti: Sequence) -> Transformation:
    return Sequence(
        transformations=list(Ti.transformations) + list(To.transformations),
        input=Ti.input,
        output=To.output,
    ).compute()


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
    grid = grid_like(field)
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
    grid = grid_like(field)
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
    grid = grid_like(field)
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
    grid = grid_like(field)
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
        if interpolates(To.transformation):
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
    if interpolates(Ti):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    # The output space of the subspace is the input space of the affine. When
    # the systems of the subspace do not state how many axes the full space
    # has, the input axis count of the affine supplies that number.
    Ti = close_subspace(Ti, n_out=axis_counts(To)[0])
    return compose(To, Ti.to(Affine))


@composer
def _(To: SubspaceTransformation, Ti: _AffineIsh) -> Affine:
    # This composer mirrors the one above, with the subspace on the left.
    if interpolates(To):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    To = close_subspace(To, n_in=axis_counts(Ti)[1])
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
    To = close_projection(To, n_in=axis_counts(Ti)[1])
    return compose(_projection_matrix(To), Ti)


@composer
def _(To: _AffineIsh, Ti: Projection) -> Affine:
    Ti = close_projection(Ti, n_out=axis_counts(To)[0])
    return compose(To, _projection_matrix(Ti))


@composer
def _(To: Projection, Ti: SubspaceTransformation) -> Transformation:
    # The subspace composer that receives the matrix of the projection
    # refuses a subspace that wraps a field, and closes an open subspace.
    To = close_projection(To, n_in=axis_counts(Ti)[1])
    return compose(_projection_matrix(To), Ti)


@composer
def _(To: SubspaceTransformation, Ti: Projection) -> Transformation:
    Ti = close_projection(Ti, n_out=axis_counts(To)[0])
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
    field = y.field - grid_like(y.field)
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )
