"""Tests of the shortest decimal that a file stores as the same float32."""

import math

import numpy as np
import pytest

from brainhops._core.numeric import float32_repr, shortest_decimal


def test_float32_repr_is_numpys_shortest_round_trip() -> None:
    rng = np.random.default_rng(0)
    scales = 10.0 ** rng.integers(-8, 8, 20000)
    values = (rng.standard_normal(20000) * scales).astype(np.float32)
    for value in values:
        assert float32_repr(value) == float(str(value))


@pytest.mark.parametrize("written", [0.3, 2.3, 1e-7, 123456.7, -4.25])
def test_float32_repr_reads_back_what_was_written(written: float) -> None:
    assert float32_repr(np.float32(written)) == written


def test_zero_and_non_finite_values_are_kept() -> None:
    assert shortest_decimal(0.0) == 0.0
    assert math.isnan(shortest_decimal(float("nan")))
    assert shortest_decimal(float("inf")) == float("inf")


@pytest.mark.parametrize("ms", [3.5, 2.3, 4.73, 2000.7, 8.2])
def test_milliseconds_read_as_seconds(ms: float) -> None:
    stored = np.float32(ms)
    seconds = shortest_decimal(float(stored) * 1e-3, lambda s: s / 1e-3)
    assert seconds == round(ms * 1e-3, 12)
    # Writing the value again stores the same bits.
    assert np.float32(seconds / 1e-3) == stored


@pytest.mark.parametrize("degrees", [9.0, 8.6, 30.0, 90.0, 0.5])
def test_radians_read_as_degrees(degrees: float) -> None:
    stored = np.float32(math.radians(degrees))
    assert shortest_decimal(math.degrees(float(stored)), math.radians) == (
        degrees
    )
