"""Regression tests for folding an affine into a stored field.

Composing an affine-like transformation with a ``CoordinatesField`` or a
``DisplacementField`` folds the affine into the stored field. The folded
field must keep the interpolation settings of the input field (``order``,
``bound`` and ``coeff``); otherwise it is later re-interpolated at the wrong
settings. A field of spline coefficients must be converted to sampled values
before the affine arithmetic and converted back afterwards; otherwise the
arithmetic runs on coefficients and yields garbage.

Both faults were present in the affine-into-field composers. This file locks
the fix in two ways: it checks that the folded field reports the same
``order``, ``bound`` and ``coeff`` as its input, and it checks that
evaluating the folded field at interior points reproduces the in-order
reference ``A(interp(f))``.
"""

import numpy as np
import pytest

from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Sequence,
)

# An anisotropic affine with shear and a shift, so a dropped setting or
# coefficient arithmetic shows up as a large discrepancy rather than a small
# one. The linear part is invertible and far from the identity.
AFFINE_MATRIX = np.array([[1.7, 0.4, 2.0], [-0.3, 0.9, -1.5]])

# A grid large enough to hold query points several nodes from every edge.
GRID_SHAPE = (14, 15)

# Query points in the interior of the grid, offset from the nodes so that a
# wrong interpolation order changes the result.
QUERY_POINTS = np.array(
    [[5.5, 6.5], [7.2, 8.1], [6.3, 5.7], [8.0, 9.0], [5.8, 7.4]]
)

# Coefficients require a spline order of at least two, so `coeff=True` is
# paired only with the cubic order.
ORDER_COEFF = [(1, False), (3, False), (3, True)]


def _evaluate(field, points):  # noqa: ANN001, ANN202
    """The coordinate map of ``field`` sampled at ``points``.

    Leading with the query points as a sampling domain evaluates the field on
    those points and returns the resulting coordinates.
    """
    computed = Sequence(
        transformations=[CoordinatesField(field=points), field]
    ).compute()
    return np.asarray(computed.to(CoordinatesField).field)


@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("order, coeff", ORDER_COEFF)
def test_fold_affine_into_field_keeps_interpolation_settings(
    field_type, order, coeff  # noqa: ANN001
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale
    field = field_type(
        field=values, order=order, bound=BoundaryCondition.mirror
    ).to(coeff=coeff)

    folded = (Affine(matrix=AFFINE_MATRIX) @ field).compute()

    assert folded.order == field.order
    assert folded.bound == field.bound
    assert folded.coeff == field.coeff


@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("order, coeff", ORDER_COEFF)
def test_fold_affine_into_field_matches_inorder_reference(
    field_type, order, coeff  # noqa: ANN001
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale
    field = field_type(
        field=values, order=order, bound=BoundaryCondition.mirror
    ).to(coeff=coeff)

    matrix = AFFINE_MATRIX
    sampled = _evaluate(field, QUERY_POINTS)
    reference = sampled @ matrix[:, :-1].T + matrix[:, -1]

    folded = (Affine(matrix=matrix) @ field).compute()
    result = _evaluate(folded, QUERY_POINTS)

    # Folding a coordinate field is exact within the field of view, because
    # interpolation is linear and the affine is affine. Folding a
    # displacement field carries an additional interior term under a spline
    # order above one: the representation subtracts the node grid, whose
    # cubic interpolation departs from the identity by an amount that decays
    # away from the edges but does not vanish on a finite grid.
    interior_term = field_type is DisplacementField and order > 1
    atol = 1e-3 if interior_term else 1e-10
    np.testing.assert_allclose(result, reference, atol=atol, rtol=0)
