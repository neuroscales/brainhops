"""Composers for pairs of transformations.

Each composer computes `T = To @ Ti`, that is `T(x) = To(Ti(x))` on the
domain of `Ti`. The input system of `T` is the input system of `Ti`,
and its output system is the output system of `To`. The composers
assume that the adjoining coordinate systems are compatible, which
`adaptors` ensures, and that both transformations are fully defined,
with no unset parameter.
"""

import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.bsplines import pull_field
from brainhops.backends import get_array_backend
from brainhops.datamodel._sugar import get_axes
from brainhops.errors import CompositionError

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
)
from ..meta import SubspaceTransformation, _close_subspace
from ..sequence import Sequence, _interpolates
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
    ba = get_array_backend(To.matrix)
    No = To.matrix.shape[0]
    Ni = Ti.matrix.shape[1] - 1
    A = ba.empty_like(To.matrix, shape=(No, Ni + 1))
    A[:, :-1] = To.matrix[:, :-1] @ Ti.matrix[:, :-1]
    A[:, -1:] = To.matrix[:, :-1] @ Ti.matrix[:, -1:] + To.matrix[:, -1:]
    return Affine(matrix=A, input=Ti.input, output=To.output)


@composer
def _(To: Linear, Ti: Linear) -> Linear:
    return Linear(
        matrix=To.matrix @ Ti.matrix, input=Ti.input, output=To.output
    )


@composer
def _(To: Rotation, Ti: Rotation) -> Rotation:
    # Rotations form a group.
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
def _(To: Translation, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    field = Ti.field + To.translation
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Scaling, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    data = Ti.data * To.scale
    return CoordinatesField(
        data=data,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Permutation, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    data = Ti.data[..., To.permutation]
    return CoordinatesField(
        data=data,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Linear, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    field = Ti.field @ To.matrix.T
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: Affine, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    field = Ti.field @ To.matrix[:, :-1].T + To.matrix[:, -1]
    return CoordinatesField(
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
    field = Ti.field + To.translation
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
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = To.scale * Ti.field + (To.scale - 1) * grid
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
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field)[..., To.permutation] - grid
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
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix.T - grid
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
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix[:, :-1].T + To.matrix[:, -1] - grid
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
    x2 = Ti.to(CoordinatesField)
    field = (
        pull_field(
            # The displacement as spline coefficients; a
            # StationaryVelocityField keeps its velocity under `data`.
            To.to(log=False, store="coefficients").data,
            coords=x2.field,
            degree=To.degree,
            bound=To.bound,
            coeff=True,
        )
        + Ti.field
    )
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=To.degree,
        bound=To.bound,
        store=To.store,
    )


@composer
def _(To: DisplacementField, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    x2 = Ti.to(CoordinatesField)
    field = (
        pull_field(
            To.to(log=False, store="coefficients").data,
            coords=x2.field,
            degree=To.degree,
            bound=To.bound,
            coeff=True,
        )
        + x2.field
    )
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: CoordinatesField, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute()
    field = pull_field(
        To.to(store="coefficients").data,
        coords=Ti.field,
        degree=To.degree,
        bound=To.bound,
        coeff=True,
    )
    return CoordinatesField(
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
def _(To: SubspaceTransformation, Ti: CoordinatesField) -> CoordinatesField:
    # The components acted on go through the inner transformation, and the
    # others pass unchanged. The inner transformation is pulled at the
    # sub-coordinates, so the extra axes are never tiled to full size.
    Ti = Ti.compute()
    x = Ti.field
    ba = get_array_backend(x)
    if To.input_axes is None or To.output_axes is None:
        raise CompositionError(
            "A subspace transform composes with a field only when it names "
            "the axes it acts on, but its input or output axes are unset."
        )
    in_axes = [int(i) for i in To.input_axes]
    out_axes = [int(i) for i in To.output_axes]
    if To.transformation is None:
        # A missing inner transformation is the identity. With equal axis
        # vectors the field is only relabelled; otherwise the subspace is a
        # pure reindexing of axes.
        if in_axes == out_axes:
            return replace(Ti, output=To.output)
        acted = x[..., in_axes]
    else:
        if _interpolates(To.transformation):
            # Positional access reads the axes that the system states, even
            # when it is open.
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
        domain = CoordinatesField(
            field=x[..., in_axes],
            degree=Ti.degree,
            bound=Ti.bound,
            store="values",
        )
        result = Sequence([domain, To.transformation]).compute()
        if isinstance(result, DisplacementField):
            result = result.to(CoordinatesField)
        if not isinstance(result, CoordinatesField):
            raise CompositionError(
                "The inner transform of a subspace transform did not reduce "
                "to a field of coordinates, so it cannot be applied to a "
                "field."
            )
        acted = result.field

    # Pass-through components are matched to the free output positions in
    # order, as in the reduction of a subspace to an affine.
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
    y = ba.stack(columns, axis=-1)

    return CoordinatesField(
        field=y,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )


@composer
def _(To: _AffineIsh, Ti: SubspaceTransformation) -> Affine:
    # A subspace that wraps a field cannot be reduced to an affine: fields are
    # applied by composing them with a sampling domain.
    if _interpolates(Ti):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    # The output space of the subspace is the input space of the affine, so the
    # axis count of the affine closes a subspace whose systems leave it
    # unknown.
    Ti = _close_subspace(Ti, n_out=axis_counts(To)[0])
    return compose(To, Ti.to(Affine))


@composer
def _(To: SubspaceTransformation, Ti: _AffineIsh) -> Affine:
    # Mirror of the composer above, with the subspace on the left.
    if _interpolates(To):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    To = _close_subspace(To, n_in=axis_counts(Ti)[1])
    return compose(To.to(Affine), Ti)


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
    # An identity inner transformation over differing axis vectors is still a
    # pure reindexing, so it collapses to Identity only when the axes match.
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
    y = compose(To, Ti.to(CoordinatesField))
    field = y.field - CartesianField(shape=Ti.field.shape[:-1]).field
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        degree=Ti.degree,
        bound=Ti.bound,
        store=Ti.store,
    )
