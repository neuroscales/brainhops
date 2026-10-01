"""Tests for units: the three-state predicates, and parsing unit names."""

import pytest

from brainhops.datamodel import units
from brainhops.datamodel.units import (
    SampleUnit,
    Unit,
    is_physicalunit,
    is_sampleunit,
    is_spaceunit,
    is_timeunit,
)

# ----------------------------------------------------------------------
#   is_sampleunit / is_physicalunit
# ----------------------------------------------------------------------


def test_the_sample_unit_is_a_singleton() -> None:
    assert Unit("sample") is SampleUnit()
    assert Unit("sample") is Unit("sample")


@pytest.mark.parametrize(
    "unit, sample, physical",
    [
        # An instance.
        (Unit("mm"), False, True),
        (Unit("s"), False, True),
        (Unit("sample"), True, False),
        # A class.
        (units.Meter, False, True),
        (units.MilliMeter, False, True),
        (units.Second, False, True),
        (SampleUnit, True, False),
        # An abstract class measures nothing.
        (Unit, False, False),
        (units.SpaceUnit, False, False),
        (units.TimeUnit, False, False),
        # Unspecified.
        (None, False, False),
        # A name `Unit` does not know is a unit of nothing.
        (Unit("not-a-unit"), False, False),
    ],
)
def test_three_state_predicates(
    unit: object, sample: bool, physical: bool
) -> None:
    assert is_sampleunit(unit) is sample
    assert is_physicalunit(unit) is physical


@pytest.mark.parametrize(
    "unit, space, time",
    [
        (Unit("mm"), True, False),
        (units.Meter, True, False),
        (Unit("ms"), False, True),
        (units.Second, False, True),
        (Unit("sample"), False, False),
        (None, False, False),
    ],
)
def test_kind_predicates(unit: object, space: bool, time: bool) -> None:
    assert is_spaceunit(unit) is space
    assert is_timeunit(unit) is time


def test_kind_predicates_are_exported() -> None:
    assert "is_spaceunit" in units.__all__
    assert "is_timeunit" in units.__all__


# ----------------------------------------------------------------------
#   Parsing prefixed names
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mm", units.MilliMeter),
        ("um", units.MicroMeter),
        ("kilometers", units.KiloMeter),
        ("ms", units.MilliSecond),
        # A prefix followed by a spelled-out unit. These were read one
        # character at a time and either missed or raised.
        ("msec", units.MilliSecond),
        ("usec", units.MicroSecond),
        ("millisec", units.MilliSecond),
        ("micrometres", units.MicroMeter),
    ],
)
def test_prefixed_names_are_parsed(name: str, expected: type) -> None:
    assert type(Unit(name)) is expected


def test_an_unknown_prefixed_name_does_not_raise() -> None:
    # Regression: "micron" is "micro" + "n", and "n" was taken for the
    # name of a unit because the suffixes were iterated character by
    # character. It is not a name this registry knows, so it is a unit of
    # nothing rather than an error.
    assert not is_physicalunit(Unit("micron"))
