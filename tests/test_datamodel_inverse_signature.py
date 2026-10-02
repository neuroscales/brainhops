"""
Regression tests: every `inverse()` override takes the signature of
`Transformation.inverse`.

`Sequence.inverse` forwards `compute` and the options of `compute()` to
every child, so an override that takes fewer arguments breaks the
inverse of any sequence that holds it.
"""

from unittest import mock

import numpy as np
import pytest

from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Inverse,
    MultiscaleField,
    Projection,
    Sequence,
)

AFFINE = np.array([[2.0, 0.0, 1.0], [0.0, 0.5, -3.0]])


def _affine() -> Affine:
    return Affine(matrix=AFFINE)


def _multiscale() -> MultiscaleField:
    rng = np.random.default_rng(0)
    return MultiscaleField(
        scales=[
            Sequence([_affine(), DisplacementField(field=rng.normal(size=s))])
            for s in [(6, 5, 2), (3, 3, 2)]
        ]
    )


CASES = {
    "projection": lambda: Projection(dropped=[2]),
    "inverse": lambda: Inverse(forward=_affine()),
    "multiscale": _multiscale,
}

# Past `compute`, the keywords are options of `compute()`, which
# `Sequence.inverse` passes on to every child.
KWARGS = [
    {},
    {"compute": False},
    {"compute": True},
    {"compute": True, "simplify": "analytic", "factor": False},
]


@pytest.mark.parametrize("kwargs", KWARGS, ids=str)
@pytest.mark.parametrize("name", CASES)
def test_the_inverse_takes_the_base_signature(name, kwargs) -> None:  # noqa: ANN001
    forward = CASES[name]()
    forward.inverse(**kwargs)


@pytest.mark.parametrize("kwargs", KWARGS, ids=str)
@pytest.mark.parametrize("name", CASES)
def test_a_sequence_holding_it_inverts(name, kwargs) -> None:  # noqa: ANN001
    forward = CASES[name]()
    inverse = Sequence([forward]).inverse(**kwargs)
    assert len(inverse) == 1


def test_the_projection_inverse_swaps_dropped_and_created() -> None:
    inverse = Projection(dropped=[2]).inverse(compute=True)
    assert list(inverse.created) == [2]
    assert len(inverse.dropped) == 0


@pytest.mark.parametrize("compute", [False, True])
def test_the_inverse_of_an_inverse_is_the_forward(compute) -> None:  # noqa: ANN001
    inverse = Inverse(forward=_affine()).inverse(
        compute=compute, simplify="analytic"
    )
    np.testing.assert_allclose(inverse.matrix, AFFINE)


@pytest.mark.parametrize("compute", [False, True])
def test_the_multiscale_inverse_reaches_every_scale(compute) -> None:  # noqa: ANN001
    field = _multiscale()
    with mock.patch.object(
        Sequence, "inverse", autospec=True, side_effect=Sequence.inverse
    ) as spy:
        inverse = MultiscaleField.inverse(field, compute=compute, factor=False)
    assert isinstance(inverse, MultiscaleField)
    assert inverse.nscales == 2
    # Compared by identity: `==` on a sequence compares its arrays.
    calls = {id(c.args[0]): c.kwargs for c in spy.call_args_list}
    for scale in field.scales:
        assert calls[id(scale)] == {"compute": compute, "factor": False}
