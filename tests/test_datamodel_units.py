"""Tests for units: the three-state predicates, and parsing unit names."""

import math

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


@pytest.mark.parametrize(
    "build",
    [
        lambda: units.SpaceUnit("s"),
        lambda: units.SpaceUnit(units.Second()),
        lambda: units.TimeUnit("mm"),
        lambda: SampleUnit("mm"),
        lambda: SampleUnit(units.Second()),
        lambda: units.Second("mm"),
    ],
)
def test_a_unit_class_only_builds_its_own_units(build: object) -> None:
    # `SpaceUnit("s")` returned a second, and `SampleUnit(anything)` the
    # sample, so a field typed `SpaceUnit` or `SampleUnit` held whatever it
    # was given.
    with pytest.raises(ValueError, match="is not a"):
        build()


def test_a_unit_class_builds_its_own_units() -> None:
    assert units.SpaceUnit("mm") is units.MilliMeter()
    assert units.SpaceUnit(units.MilliMeter()) is units.MilliMeter()
    assert units.TimeUnit("ms") is units.MilliSecond()
    assert SampleUnit("sample") is SampleUnit()
    assert units.Meter("metre") is units.Meter()


# ----------------------------------------------------------------------
#   scale / log10_scale (issue #257)
# ----------------------------------------------------------------------

KNOWN_SCALES = [
    # Time, in seconds.
    (units.Minute, "minute", 60.0),
    (units.Hour, "hour", 3600.0),
    (units.Day, "day", 86400.0),
    (units.Week, "week", 604800.0),
    (units.Year, "year", 52 * 604800.0),
    # Space, in meters.
    (units.Inch, "inch", 0.0254),
    (units.Foot, "foot", 0.3048),
    (units.Yard, "yard", 0.9144),
    (units.Mile, "mile", 1609.344),
    (units.Angstrom, "angstrom", 1e-10),
    (units.Parsec, "parsec", 3.085677581491367e16),
]


@pytest.mark.parametrize("cls, name, scale", KNOWN_SCALES)
def test_known_units_have_their_scale(
    cls: type, name: str, scale: float
) -> None:
    unit = Unit(name)
    assert unit is cls()
    assert isinstance(unit, units.KnownUnit)
    assert isinstance(unit.scale, float)
    assert unit.scale == pytest.approx(scale, rel=1e-15)
    assert cls.scale == unit.scale
    assert unit.log10_scale == pytest.approx(math.log10(scale))
    assert unit.prefix is None


def test_every_registered_known_unit_has_a_table_scale() -> None:
    for cls in units._REGISTERED_UNITS:
        if issubclass(cls, units.KnownUnit):
            assert cls().scale == float(units.UNITS[cls.name][0])
            assert cls().scale != 1.0


@pytest.mark.parametrize(
    "name, scale, log10_scale",
    [
        ("meter", 1.0, 0),
        ("millimeter", 1e-3, -3),
        ("micrometer", 1e-6, -6),
        ("kilometer", 1e3, 3),
        ("second", 1.0, 0),
        ("millisecond", 1e-3, -3),
    ],
)
def test_si_units_have_their_scale(
    name: str, scale: float, log10_scale: int
) -> None:
    unit = Unit(name)
    assert unit.log10_scale == log10_scale
    assert unit.scale == pytest.approx(scale, rel=1e-15)


def test_known_unit_symbols() -> None:
    assert Unit("inch").symbol == '"'
    assert Unit("min").symbol == "min"
    assert Unit("ft") is units.Foot()
