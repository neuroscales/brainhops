"""
This module implements function that compute the composition of two
transformations. It assumes that their input and output coordinate
systems are compatible (see `adaptors` for composing transformations
that do not have compatible coordinate systems). It also assumes that
transfmations are fully defined (i.e., their parameters are not `None`).

A composer takes two transformations To and Ti, and returns a
transformation T such that T = To @ Ti, i.e. such that T(x) = To(Ti(x))
for any x in the domain of Ti. The input coordinate system of T is the
input coordinate system of Ti, and the output coordinate system of T is
the output coordinate system of To.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# core
from brainhops._core.bsplines import pull_field
from brainhops.backends import get_array_backend

# internals
from .base import Transformation
from .compose import compose, composer
from .concrete import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Scaling,
    Translation,
)
from .errors import CompositionError
from .meta import SubspaceTransformation
from .sequence import Sequence, _interpolates

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
def _(To: Permutation, Ti: Permutation) -> Permutation:
    return Permutation(
        permutation=To.permutation[Ti.permutation],
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
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field + To.translation
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Scaling, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field * To.scale
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Permutation, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field[..., To.permutation]
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Linear, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field @ To.matrix.T
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Affine, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field @ To.matrix[:, :-1].T + To.matrix[:, -1]
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


# ----------------------------------------------------------------------
#     AFFINE o DISPLACEMENTS
# ----------------------------------------------------------------------


@composer
def _(To: Translation, Ti: DisplacementField) -> DisplacementField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = Ti.field + To.translation
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Scaling, Ti: DisplacementField) -> DisplacementField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = To.scale * Ti.field + (To.scale - 1) * grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Permutation, Ti: DisplacementField) -> DisplacementField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field)[..., To.permutation] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Linear, Ti: DisplacementField) -> DisplacementField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix.T - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: Affine, Ti: DisplacementField) -> DisplacementField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    grid = CartesianField(shape=Ti.field.shape[:-1]).field
    field = (grid + Ti.field) @ To.matrix[:, :-1].T + To.matrix[:, -1] - grid
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


# ----------------------------------------------------------------------
#     DISPLACEMENTS & COORDINATES
# ----------------------------------------------------------------------


@composer
def _(To: DisplacementField, Ti: DisplacementField) -> DisplacementField:
    Ti = Ti.compute().to(coeff=False)
    x2 = Ti.to(CoordinatesField)
    field = (
        pull_field(
            To.field,
            coords=x2.field,
            order=To.order,
            bound=To.bound,
            coeff=To.coeff,
        )
        + Ti.field
    )
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=To.order,
        bound=To.bound,
        coeff=False,
    ).to(coeff=To.coeff)


@composer
def _(To: DisplacementField, Ti: CoordinatesField) -> CoordinatesField:
    Ti = Ti.compute().to(coeff=False)
    x2 = Ti.to(CoordinatesField)
    field = (
        pull_field(
            To.field,
            coords=x2.field,
            order=To.order,
            bound=To.bound,
            coeff=To.coeff,
        )
        + x2.field
    )
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=Ti.coeff)


@composer
def _(To: CoordinatesField, Ti: CoordinatesField) -> CoordinatesField:
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
    field = pull_field(
        To.field,
        coords=Ti.field,
        order=To.order,
        bound=To.bound,
        coeff=To.coeff,
    )
    return CoordinatesField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


# ----------------------------------------------------------------------
#     SUBSPACE
# ----------------------------------------------------------------------


@composer
def _(To: SubspaceTransformation, Ti: CoordinatesField) -> CoordinatesField:
    # Apply a transform that acts on a subset of the axes to a field of
    # coordinates. The acted-on components of the field are carried through
    # the inner transform, and the remaining components pass through
    # unchanged. The inner transform is evaluated lazily, by pulling it at
    # the acted-on sub-coordinates, so the extra axes are never tiled to
    # full size.
    coeff = Ti.coeff
    Ti = Ti.compute().to(coeff=False)
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
        return replace(Ti, output=To.output)
    if _interpolates(To.transformation) and (
        To.input is not None and To.input.axes is not None
    ):
        for i in in_axes:
            axis = To.input.axes[i]
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
        field=x[..., in_axes], order=Ti.order, bound=Ti.bound, coeff=False
    )
    result = Sequence(transformations=[domain, To.transformation]).compute()
    if isinstance(result, DisplacementField):
        result = result.to(CoordinatesField)
    if not isinstance(result, CoordinatesField):
        raise CompositionError(
            "The inner transform of a subspace transform did not reduce to "
            "a field of coordinates, so it cannot be applied to a field."
        )
    result = result.to(coeff=False)

    # Reassemble the full field. The acted-on components take their new
    # values from the inner result, and each pass-through component is
    # copied straight from the input, matched to its output position in
    # order, exactly as the subspace-to-affine reduction matches them.
    n = x.shape[-1]
    acted_in = set(in_axes)
    acted_out = set(out_axes)
    passthrough_in = [i for i in range(n) if i not in acted_in]
    passthrough_out = [o for o in range(n) if o not in acted_out]
    columns: tx.List[tx.Any] = [None] * n
    for k, o in enumerate(out_axes):
        columns[o] = result.field[..., k]
    for o, i in zip(passthrough_out, passthrough_in):
        columns[o] = x[..., i]
    y = ba.stack(columns, axis=-1)
    return CoordinatesField(
        field=y,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=coeff)


@composer
def _(To: _AffineIsh, Ti: SubspaceTransformation) -> Affine:
    # Embed a subspace transform that merely lifts an affine into a larger
    # space, then compose the two as plain affines. A subspace that wraps a
    # field cannot be reduced this way -- a field is applied by composing it
    # with a sampling domain, not by reduction to an affine -- so it stays a
    # wrapper and this composition is declined.
    if _interpolates(Ti):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    return compose(To, Ti.to(Affine))


@composer
def _(To: SubspaceTransformation, Ti: _AffineIsh) -> Affine:
    # The mirror of the embed above, with the subspace transform on the left.
    if _interpolates(To):
        raise CompositionError(
            "A subspace transform that wraps a field is not embedded into an "
            "affine; it stays a wrapper and is applied by composing it with a "
            "sampling domain."
        )
    return compose(To.to(Affine), Ti)


@composer
def _(
    To: SubspaceTransformation,
    Ti: SubspaceTransformation,
) -> Transformation:
    # Compose two transforms that act on subsets of the axes. The two
    # compose into one subspace transform only when the axes the first
    # writes are the axes the second reads. The inner transforms compose in
    # order (through a `Sequence`, which cancels an inner/inner inverse pair
    # symbolically before any field is materialized), and a pair that
    # reduces to the identity collapses the whole composition to the
    # identity.
    #
    # This composer is mode-free: it always composes the inners it is
    # handed. Whether a pair of adjacent subspace transforms is handed here
    # at all (for instance, whether two subspace-wrapped fields are composed
    # under a restrictive mode, which would resample one field through the
    # other) is decided by the sequence engine's gate, not here.
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
    inner_prev = Ti.transformation
    inner_next = To.transformation
    inner = Sequence(
        transformations=[
            inner_prev or Identity(),
            inner_next or Identity(),
        ]
    ).compute()
    # The inner transforms cancelling to the identity only tells half the
    # story: the composition still reindexes the axes unless the axes the
    # first reads are the axes the second writes. It collapses to a bare
    # identity only when the inner is the identity AND those axes match.
    # Otherwise it stays a subspace transform: an identity inner over a
    # differing pair of axis vectors is a pure axis reindex, which embeds to
    # the affine that maps each input axis to its output axis, and which
    # `is_identity` (unlike a `transformation=None` subspace) correctly
    # reports as non-identity.
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
    # Apply a transform that acts on a subset of the axes to a field of
    # displacements. The displacement field is read as a field of
    # coordinates, the subspace transform is applied, and the grid is
    # subtracted back off to return to displacements.
    Ti = Ti.compute().to(coeff=False)
    y = compose(To, Ti.to(CoordinatesField))
    field = y.field - CartesianField(shape=Ti.field.shape[:-1]).field
    return DisplacementField(
        field=field,
        input=Ti.input,
        output=To.output,
        order=Ti.order,
        bound=Ti.bound,
        coeff=False,
    ).to(coeff=Ti.coeff)
