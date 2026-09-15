"""Fold-ordering of an affine and an interpolated field (#66).

Folding an affine into a stored displacement or coordinates field changes
the encoded transform once the field is interpolated. The value the affine
is folded into is a stored sample, not the query coordinate, so the folded
field is wrong wherever interpolation reaches past the samples: outside the
field of view for any order, and near the border for order two or more.

The compose engine therefore never folds an affine into a stored field.
An affine is folded only onto a *sampling domain*: a query grid or an
explicit point set that leads a sequence and has already been evaluated.
That fold is exact, because the coordinates are the values the affine acts
on, and it is the terminal step of reslicing. A stored field that does not
lead a sequence stays a separate step, applied in order.

These tests pin the contract and re-confirm the premise numerically: the
in-order path is exact to machine precision, including outside the field of
view, exactly where folding was wrong by an O(1) amount.
"""

import numpy as np
import pytest
import scipy.ndimage as ndi
import typing_extensions as tx

from brainhops import backends
from brainhops._core.bsplines import pull_field
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    CartesianField,
    CompositionError,
    CoordinatesField,
    DisplacementField,
    Sequence,
    SubspaceTransformation,
    Translation,
    _compose,
    _Evaluated,
)


@pytest.fixture(autouse=True)
def _numpy_backend() -> tx.Iterator[None]:
    # Every test in this module works with concrete numpy arrays.
    with backends.backend("numpy"):
        yield


# An anisotropic affine with scale, shear, and shift, so that folding it
# into a field is wrong in a way that a diagonal affine would hide.
_AFFINE = np.array([[1.7, 0.4, 3.0], [0.2, 0.8, -2.0]])
_SHAPE = (20, 20)


def _affine() -> Affine:
    return Affine(matrix=_AFFINE.copy())


def _apply_affine(points: np.ndarray) -> np.ndarray:
    return points @ _AFFINE[:, :-1].T + _AFFINE[:, -1]


def _displacement_samples(rng) -> np.ndarray:  # noqa: ANN001
    return rng.standard_normal((*_SHAPE, 2)) * 0.7


def _query_points(rng) -> np.ndarray:  # noqa: ANN001
    # Two hundred points, half strictly inside the field of view and half
    # spilling well past its borders.
    interior = rng.uniform(1.0, 18.0, size=(100, 2))
    outside = rng.uniform(-8.0, 28.0, size=(100, 2))
    return np.concatenate([interior, outside], axis=0)


# ----------------------------------------------------------------------
#   The no-fold shape
# ----------------------------------------------------------------------


def test_stored_field_and_affine_stay_separate_without_a_domain() -> None:
    # `[D, A]` with no leading sampling domain must not fold. The affine
    # cannot be pushed into the stored displacement field, so the two
    # remain separate steps, in application order.
    rng = np.random.default_rng(0)
    disp = DisplacementField(field=_displacement_samples(rng))
    result = Sequence([disp, _affine()]).compute()
    assert isinstance(result, Sequence)
    assert [type(t).__name__ for t in result] == [
        "DisplacementField",
        "Affine",
    ]


def test_affine_on_a_raw_field_is_refused() -> None:
    # No composer folds an affine into a stored field, so the composition
    # is refused. The sequence machinery catches the refusal and keeps the
    # elements separate.
    rng = np.random.default_rng(1)
    disp = DisplacementField(field=_displacement_samples(rng))
    with pytest.raises(CompositionError):
        _compose(_affine(), disp)
    coords = CoordinatesField(field=rng.standard_normal((*_SHAPE, 2)))
    with pytest.raises(CompositionError):
        _compose(_affine(), coords)


def test_two_raw_displacement_fields_are_refused() -> None:
    # A displacement field is not a sampling domain, so folding one into
    # another is refused too (#66, Q1). Reslicing loses nothing, because a
    # leading grid makes both interpolate in order.
    rng = np.random.default_rng(2)
    first = DisplacementField(field=_displacement_samples(rng))
    second = DisplacementField(field=_displacement_samples(rng))
    with pytest.raises(CompositionError):
        _compose(first, second)


# ----------------------------------------------------------------------
#   In-order exactness, and the premise it rests on
# ----------------------------------------------------------------------


def _combinations() -> tx.Iterator[tuple]:
    for order in (1, 3):
        for bound in ("nearest", "reflect"):
            for coeff in (False, True):
                if coeff and order < 2:
                    # A coefficient field is only meaningful for an order
                    # that prefilters, which is order two or more.
                    continue
                yield order, bound, coeff


@pytest.mark.parametrize("order, bound, coeff", list(_combinations()))
def test_in_order_is_exact_including_outside_fov(order, bound, coeff) -> None:  # noqa: ANN001
    # Evaluating `[points, D, A]` in order gives `A(y + d(y))` at each
    # query point `y`, where `d` is the interpolated displacement. This is
    # exact to machine precision, including at points that fall outside the
    # field of view.
    rng = np.random.default_rng(10)
    samples = _displacement_samples(rng)
    points = _query_points(rng)

    disp = DisplacementField(field=samples.copy(), order=order, bound=bound)
    if coeff:
        disp = disp.to(coeff=True)

    # Independent reference: interpolate the displacement with scipy, add
    # the query point, then apply the affine.
    interpolated = pull_field(
        np.asarray(disp.field),
        points,
        order=order,
        bound=bound,
        coeff=coeff,
    )
    reference = _apply_affine(points + interpolated)

    # The engine, with the point set leading as the sampling domain.
    query = CoordinatesField(field=points.copy())
    computed = Sequence([query, disp, _affine()]).compute()
    engine = np.asarray(computed.field)

    assert np.abs(engine - reference).max() < 1e-12

    # Premise: folding the affine into the stored field would be wrong by
    # an O(1) amount outside the field of view. Reconstruct that fold and
    # show it disagrees with the in-order truth where interpolation reaches
    # past the samples.
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in _SHAPE], indexing="ij"), -1
    ).astype(float)
    folded_samples = _apply_affine(grid + samples) - grid
    folded = pull_field(
        folded_samples, points, order=order, bound=bound, coeff=False
    )
    fold_result = points + folded
    outside = np.any((points < 0) | (points > 19), axis=1)
    assert np.abs(fold_result[outside] - reference[outside]).max() > 0.1


# ----------------------------------------------------------------------
#   Affine runs still merge
# ----------------------------------------------------------------------


def test_affine_runs_merge_without_a_domain() -> None:
    # Two adjacent affines still compose into one affine. Only a fold into
    # a stored field is refused.
    first = _affine()
    second = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    result = Sequence([first, second]).compute()
    assert isinstance(result, Affine)


def test_affines_pre_merge_before_folding_onto_a_domain() -> None:
    # A run of affines ahead of a sampling grid merges into a single affine
    # and is then folded onto the grid. The merged result equals folding
    # each affine in turn.
    grid = CartesianField(shape=_SHAPE)
    first = _affine()
    second = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0]]))
    merged = Sequence([grid, first, second]).compute()
    stepwise = Sequence([grid, second @ first]).compute()
    np.testing.assert_allclose(
        np.asarray(merged.field), np.asarray(stepwise.field)
    )


# ----------------------------------------------------------------------
#   The domain marker: promotion and demotion
# ----------------------------------------------------------------------


def test_grid_leading_collapses_while_no_grid_stays_a_sequence() -> None:
    # This is the distinction the design rests on. A leading grid is the
    # sampling domain, so `[grid, D, A]` collapses to a single field of
    # coordinates ready to feed a pull. With no leading grid, `[D, A]`
    # stays a sequence, because the affine must not fold into the field.
    rng = np.random.default_rng(20)
    disp = DisplacementField(field=_displacement_samples(rng))
    grid = CartesianField(shape=_SHAPE)

    with_grid = Sequence([grid, disp, _affine()]).compute()
    assert isinstance(with_grid, CoordinatesField)
    assert not isinstance(with_grid, Sequence)
    assert with_grid.field is not None

    without_grid = Sequence([disp, _affine()]).compute()
    assert isinstance(without_grid, Sequence)


def test_a_leading_point_set_is_a_sampling_domain() -> None:
    # A leading raw `CoordinatesField` is a point set, and is treated as
    # the sampling domain (#66, Q2). The transforms after it are evaluated
    # on the points, which is what `points.py` and the multiscale tests
    # rely on.
    rng = np.random.default_rng(21)
    disp = DisplacementField(field=_displacement_samples(rng))
    points = CoordinatesField(field=_query_points(rng))
    result = Sequence([points, disp, _affine()]).compute()
    assert isinstance(result, CoordinatesField)
    assert np.asarray(result.field).shape == (200, 2)


def test_a_computed_result_never_exposes_the_private_marker() -> None:
    # The evaluated-domain marker is internal. A computed result is a plain
    # `CoordinatesField`, never the private `_Evaluated`, whether it
    # collapses to one field or stays a sequence with an un-appliable step.
    rng = np.random.default_rng(22)
    disp = DisplacementField(field=_displacement_samples(rng))
    grid = CartesianField(shape=_SHAPE)

    collapsed = Sequence([grid, disp, _affine()]).compute()
    assert type(collapsed) is CoordinatesField

    stranded = Sequence(
        [grid, Bijection(forward=_affine(), backward=_affine())]
    ).compute()
    assert isinstance(stranded, Sequence)
    assert not any(isinstance(t, _Evaluated) for t in stranded)


# ----------------------------------------------------------------------
#   Subspace transform on a sampling domain
# ----------------------------------------------------------------------


def test_subspace_transform_applies_only_to_its_axes() -> None:
    # A subspace transform, evaluated on a sampling domain, applies its
    # inner transform to the input-axis columns and passes the rest
    # through. Here the inner translation shifts the first axis by ten and
    # leaves the second axis untouched.
    inner = Translation(translation=np.array([10.0]))
    subspace = SubspaceTransformation(
        transformation=inner, input_axes=[0], output_axes=[0]
    )
    domain = _Evaluated(field=np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = _compose(subspace, domain)
    np.testing.assert_allclose(
        np.asarray(result.field), [[11.0, 2.0], [13.0, 4.0]]
    )


# ----------------------------------------------------------------------
#   End-to-end reslice
# ----------------------------------------------------------------------


def test_reslice_matches_scipy_on_a_reference_grid() -> None:
    # Reslicing samples the reference grid through the composed
    # voxel-to-voxel map and pulls the data. The map is built with the grid
    # leading, so every affine folds onto the sampled coordinates exactly.
    # The result matches a direct scipy interpolation at the same
    # coordinates.
    vox = CoordinateSystem(name="vox")
    world = CoordinateSystem(name="world")
    rng = np.random.default_rng(30)
    data = rng.standard_normal((10, 12))

    source = Affine(
        matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]]),
        input=vox,
        output=world,
    )
    image = SingleScaleImage(data=data, transformations=[source])

    target = Affine(
        matrix=np.array([[1.5, 0.0, 0.5], [0.0, 2.0, 1.0]]),
        input=vox,
        output=world,
    )
    reference = Geometry(
        (CartesianField(shape=(8, 9), input=vox, output=vox), target)
    )
    resliced = image.reslice(reference, order=1, bound="nearest")

    grid = np.stack(
        np.meshgrid(np.arange(8), np.arange(9), indexing="ij"), -1
    ).astype(float)
    world_points = grid @ target.matrix[:, :-1].T + target.matrix[:, -1]
    source_homogeneous = np.array(
        [[2.0, 0.0, 1.0], [0.0, 3.0, -2.0], [0.0, 0.0, 1.0]]
    )
    source_inverse = np.linalg.inv(source_homogeneous)[:-1]
    voxel_points = (
        world_points @ source_inverse[:, :-1].T + source_inverse[:, -1]
    )
    expected = ndi.map_coordinates(
        data, np.moveaxis(voxel_points, -1, 0), order=1, mode="nearest"
    )

    np.testing.assert_allclose(np.asarray(resliced), expected, atol=1e-12)


def test_reslice_reports_an_unappliable_step_clearly() -> None:
    # A step that cannot be applied to the reference grid, such as a
    # bijection with no affine or field reduction, leaves the computed map
    # a sequence rather than a field of coordinates. Reslice reports this
    # with a clear error naming the step, rather than failing on a missing
    # attribute.
    vox = CoordinateSystem(name="vox")
    rng = np.random.default_rng(31)
    bijection = Bijection(
        forward=DisplacementField(field=np.zeros((5, 6, 2))),
        backward=DisplacementField(field=np.zeros((5, 6, 2))),
        input=vox,
        output=vox,
    )
    image = SingleScaleImage(
        data=rng.standard_normal((5, 6)), transformations=[bijection]
    )
    with pytest.raises(ValueError, match="Bijection"):
        image.reslice()
