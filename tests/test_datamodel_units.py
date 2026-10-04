"""Tests for units: parsing names through pint, the dimension-restricted
classes, the predicates, and the value semantics of a unit."""

import math
import pickle
import subprocess
import sys
import textwrap
import typing as t
import warnings

import pytest
from bagof.converters import ConversionError

from brainhops.datamodel import units
from brainhops.datamodel.axes import Axis, SpaceAxis, TimeAxis
from brainhops.datamodel.units import (
    IndexUnit,
    SpaceUnit,
    TimeUnit,
    Unit,
    is_indexunit,
    is_physicalunit,
    is_spaceunit,
    is_timeunit,
)

# ----------------------------------------------------------------------
#   Parsing
# ----------------------------------------------------------------------

PARSED = [
    # name, class, canonical name, scale in base units
    ("mm", SpaceUnit, "millimeter", 1e-3),
    ("um", SpaceUnit, "micrometer", 1e-6),
    ("µm", SpaceUnit, "micrometer", 1e-6),  # micro sign
    ("μm", SpaceUnit, "micrometer", 1e-6),  # greek mu
    ("mcm", SpaceUnit, "micrometer", 1e-6),
    ("micm", SpaceUnit, "micrometer", 1e-6),
    ("micron", SpaceUnit, "micrometer", 1e-6),
    ("microns", SpaceUnit, "micrometer", 1e-6),
    ("micrometres", SpaceUnit, "micrometer", 1e-6),
    ("kilometers", SpaceUnit, "kilometer", 1e3),
    ("angstrom", SpaceUnit, "angstrom", 1e-10),
    ("parsec", SpaceUnit, "parsec", 3.0856775814671916e16),
    ("inch", SpaceUnit, "inch", 0.0254),
    ("mile", SpaceUnit, "mile", 1609.344),
    ('"', SpaceUnit, "inch", 0.0254),
    ("'", SpaceUnit, "foot", 0.3048),
    ("light year", SpaceUnit, "light_year", 9.4607304725808e15),
    ("ms", TimeUnit, "millisecond", 1e-3),
    ("msec", TimeUnit, "millisecond", 1e-3),
    ("millisec", TimeUnit, "millisecond", 1e-3),
    ("usec", TimeUnit, "microsecond", 1e-6),
    ("mcs", TimeUnit, "microsecond", 1e-6),
    ("minute", TimeUnit, "minute", 60.0),
    ("min", TimeUnit, "minute", 60.0),
    ("hour", TimeUnit, "hour", 3600.0),
    ("day", TimeUnit, "day", 86400.0),
    ("week", TimeUnit, "week", 604800.0),
    ("year", TimeUnit, "year", 365.25 * 86400.0),
    ("Hz", Unit, "hertz", 1.0),
    ("1/s", Unit, "1 / second", 1.0),
    ("deg", Unit, "degree", math.pi / 180),
    ("rad", Unit, "radian", 1.0),
    ("s/mm^2", Unit, "second / millimeter ** 2", 1e6),
    ("s/mm²", Unit, "second / millimeter ** 2", 1e6),
    ("s/mm**2", Unit, "second / millimeter ** 2", 1e6),
    ("mm/s", Unit, "millimeter / second", 1e-3),
    ("Bq/ml", Unit, "becquerel / milliliter", 1e6),
    ("T", Unit, "tesla", 1.0),
    ("mT", Unit, "millitesla", 1e-3),
    ("%", Unit, "percent", 1e-2),
    ("ppm", Unit, "ppm", 1e-6),
    ("count", Unit, "count", 1.0),
    ("a.u.", Unit, "arbitrary_unit", 1.0),
    ("A.U.", Unit, "arbitrary_unit", 1.0),
    ("au", Unit, "arbitrary_unit", 1.0),
    ("HU", Unit, "hounsfield_unit", 1.0),
    ("index", IndexUnit, "index", 1.0),
    ("voxel", IndexUnit, "voxel", 1.0),
    ("voxels", IndexUnit, "voxel", 1.0),
    ("pixel", IndexUnit, "pixel", 1.0),
    ("pixels", IndexUnit, "pixel", 1.0),
]


@pytest.mark.parametrize("name, cls, canonical, scale", PARSED)
def test_names_are_parsed(
    name: str, cls: type, canonical: str, scale: float
) -> None:
    unit = Unit(name)
    assert type(unit) is cls
    assert unit.name == canonical
    assert str(unit) == canonical
    assert repr(unit) == repr(canonical)
    assert isinstance(unit.scale, float)
    assert unit.scale == pytest.approx(scale, rel=1e-12)


def test_the_arbitrary_unit_is_not_parsed_by_pint() -> None:
    # pint's tokenizer splits "a.u." on the dots, and reads it as
    # `unified_atomic_mass_unit * year`; "au" is pint's astronomical unit.
    for name in ("a.u.", "A.U.", "a.u", "au", "AU", "arbitrary unit"):
        unit = Unit(name)
        assert unit is Unit("arbitrary_unit")
        assert unit.type == "dimensionless"
        assert unit.symbol == "a.u."


def test_the_pint_free_names_are_what_pint_gives() -> None:
    # A few names are resolved without pint, so that building them at
    # import time does not import it. They must agree with the registry.
    for name, (canonical, dimension) in units._PINT_FREE.items():
        parsed = units._parse(units._ALIASES.get(name, name))
        assert units._canonical_name(parsed) == canonical, name
        assert str(parsed.dimensionality) == dimension, name


@pytest.mark.parametrize(
    "name, symbol",
    [
        ("mm", "mm"),
        ("micrometer", "µm"),
        ("inch", "in"),
        ("min", "min"),
        ("s/mm^2", "s / mm ** 2"),
        ("voxel", "vox"),
        ("HU", "HU"),
    ],
)
def test_symbols(name: str, symbol: str) -> None:
    assert Unit(name).symbol == symbol


@pytest.mark.parametrize(
    "name, log10_scale",
    [
        ("meter", 0),
        ("millimeter", -3),
        ("micrometer", -6),
        ("kilometer", 3),
        ("angstrom", -10),
        ("second", 0),
        ("millisecond", -3),
        ("voxel", 0),
    ],
)
def test_a_power_of_ten_has_an_integer_log10_scale(
    name: str, log10_scale: int
) -> None:
    # The adaptors compute the ratio of two such units exactly, as a
    # power of ten.
    assert Unit(name).log10_scale == log10_scale
    assert isinstance(Unit(name).log10_scale, int)


@pytest.mark.parametrize("name", ["inch", "foot", "minute", "parsec"])
def test_other_units_have_a_float_log10_scale(name: str) -> None:
    unit = Unit(name)
    assert isinstance(unit.log10_scale, float)
    assert unit.log10_scale == pytest.approx(math.log10(unit.scale))


@pytest.mark.parametrize(
    "name, prefix",
    [("mm", "milli"), ("um", "micro"), ("km", "kilo"), ("m", None),
     ("inch", None), ("s/mm^2", None), ("mm^2", None)],
)  # fmt: skip
def test_prefixes(name: str, prefix: t.Optional[str]) -> None:
    assert Unit(name).prefix == prefix


@pytest.mark.parametrize(
    "name, kind",
    [
        ("mm", "space"),
        ("ms", "time"),
        ("voxel", "index"),
        ("%", "dimensionless"),
        ("Hz", "1 / [time]"),
        ("s/mm^2", "[time] / [length] ** 2"),
    ],
)
def test_types(name: str, kind: str) -> None:
    assert Unit(name).type == kind


def test_compatibility() -> None:
    assert Unit("mm").is_compatible_with("inch")
    assert Unit("Hz").is_compatible_with(Unit("1/s"))
    assert Unit("voxel").is_compatible_with("pixel")
    assert not Unit("mm").is_compatible_with("s")
    assert not Unit("voxel").is_compatible_with("mm")
    # Angles are dimensionless, as pint has them.
    assert Unit("deg").is_compatible_with("%")


# ----------------------------------------------------------------------
#   Errors
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["not-a-unit", "ms-1", "10 mm", "mm)", "decimeterz", "''"]
)
def test_a_name_that_is_not_a_unit_raises(name: str) -> None:
    with pytest.raises(ValueError, match="is not a unit brainhops"):
        Unit(name)


def test_an_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="empty name"):
        Unit("")
    with pytest.raises(ValueError, match="empty name"):
        Unit("   ")


@pytest.mark.parametrize("value", [1.0, object(), b"mm"])
def test_a_unit_is_built_from_a_name_or_a_unit(value: object) -> None:
    with pytest.raises(TypeError, match="built from a name or a unit"):
        Unit(value)


def test_an_unrestricted_unit_needs_a_name() -> None:
    with pytest.raises(TypeError, match="needs the name of a unit"):
        Unit()
    with pytest.raises(TypeError, match="needs the name of a unit"):
        SpaceUnit()


def test_a_unit_is_immutable() -> None:
    unit = Unit("mm")
    with pytest.raises(AttributeError, match="immutable"):
        unit._name = "second"
    with pytest.raises(AttributeError):
        unit.anything = 1


# ----------------------------------------------------------------------
#   Dimension restriction
# ----------------------------------------------------------------------


def test_unit_returns_the_class_of_the_dimension() -> None:
    assert isinstance(Unit("mm"), SpaceUnit)
    assert isinstance(Unit("ms"), TimeUnit)
    assert isinstance(Unit("voxel"), IndexUnit)
    assert type(Unit("Hz")) is Unit


def test_the_declared_dimensions() -> None:
    assert SpaceUnit.dimension == "[length]"
    assert TimeUnit.dimension == "[time]"
    assert IndexUnit.dimension == "[index]"
    assert Unit.dimension is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: SpaceUnit("s"),
        lambda: SpaceUnit(Unit("s")),
        lambda: SpaceUnit("voxel"),
        lambda: SpaceUnit(IndexUnit()),
        lambda: TimeUnit("mm"),
        lambda: TimeUnit("Hz"),
        lambda: IndexUnit("mm"),
        lambda: IndexUnit(Unit("s")),
        lambda: Unit["[time] / [length] ** 2"]("mm"),
    ],
)
def test_a_unit_class_only_builds_its_own_units(build: object) -> None:
    with pytest.raises(ValueError, match="is not a"):
        build()


def test_the_restriction_error_names_both_dimensions() -> None:
    with pytest.raises(ValueError) as error:
        SpaceUnit("s")
    message = str(error.value)
    assert "'s' is not a SpaceUnit" in message
    assert "[time]" in message and "[length]" in message


def test_a_unit_class_builds_its_own_units() -> None:
    assert SpaceUnit("mm") is Unit("mm")
    assert SpaceUnit(Unit("mm")) is Unit("mm")
    assert TimeUnit("ms") is Unit("millisecond")
    assert IndexUnit() is Unit("index")
    assert IndexUnit("voxel") is Unit("voxel")
    assert SpaceUnit(name="metre") is Unit("meter")


def test_a_parametrized_class() -> None:
    Diffusion = Unit["[time] / [length] ** 2"]
    unit = Diffusion("s/mm^2")
    assert isinstance(unit, Diffusion)
    assert isinstance(unit, Unit)
    assert Diffusion.dimension == "[time] / [length] ** 2"
    assert unit == Unit("s/mm^2")
    assert hash(unit) == hash(Unit("s/mm^2"))
    assert Diffusion(unit) is unit
    assert Diffusion(Unit("s/mm^2")) is unit
    # Built once per dimension, however it is spaced.
    assert Unit["[time]/[length]**2"] is Diffusion
    # A declared dimension gives its declared class.
    assert Unit["[length]"] is SpaceUnit
    assert Unit["[index]"] is IndexUnit


def test_only_unit_is_subscripted() -> None:
    with pytest.raises(TypeError, match="already restricted"):
        SpaceUnit["[time]"]
    with pytest.raises(TypeError, match="dimension is a string"):
        Unit[1]


def test_an_invalid_dimension_raises_when_used() -> None:
    kls = Unit["[not_a_dimension]"]
    with pytest.raises(ValueError):
        kls("mm")


def test_a_subclass_declares_a_dimension() -> None:
    class Frequency(Unit, dimension="1 / [time]"):
        __slots__ = ()

    assert type(Frequency("Hz")) is Frequency
    assert Frequency("Hz") == Unit("Hz")
    with pytest.raises(ValueError, match="is not a Frequency"):
        Frequency("s")


# ----------------------------------------------------------------------
#   bagof hints
# ----------------------------------------------------------------------


def test_axis_hints_convert_names() -> None:
    axis = SpaceAxis(name="x", unit="mm")
    assert axis.unit is Unit("mm")
    assert SpaceAxis(name="x", unit="voxel").unit is Unit("voxel")
    assert TimeAxis(name="t", unit="ms").unit is Unit("ms")
    assert TimeAxis(name="t", unit="index").unit is IndexUnit()
    assert Axis(name="x", type="space", unit="um").unit is Unit("um")
    assert SpaceAxis(name="x").unit is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: SpaceAxis(name="x", unit="s"),
        lambda: SpaceAxis(name="x", unit="Hz"),
        lambda: TimeAxis(name="t", unit="mm"),
        lambda: SpaceAxis(name="x", unit="not-a-unit"),
    ],
)
def test_axis_hints_refuse_other_units(build: t.Callable) -> None:
    with pytest.raises(ConversionError):
        build()


def test_a_parametrized_class_as_a_hint() -> None:
    from bagof.converters import get_converter

    Diffusion = Unit["[time] / [length] ** 2"]
    convert = get_converter(t.Optional[t.Union[Diffusion, IndexUnit]])
    assert convert("s/mm^2") is Diffusion("s/mm^2")
    assert convert("voxel") is Unit("voxel")
    assert convert(None) is None
    with pytest.raises(ConversionError):
        convert("mm")


# ----------------------------------------------------------------------
#   Interning, equality, hash, pickling
# ----------------------------------------------------------------------


def test_units_are_interned() -> None:
    assert Unit("mm") is Unit("millimeter")
    assert Unit("mm") is Unit("millimetre")
    assert Unit("mm") is SpaceUnit("mm")
    assert Unit("s/mm^2") is Unit("s / mm ** 2")
    assert Unit("mm/s") is Unit("1/s*mm")


def test_equal_units_are_equal() -> None:
    assert SpaceUnit("mm") == Unit("millimeter")
    assert Unit("um") == Unit("micron")
    assert Unit("mm") != Unit("um")
    assert Unit("voxel") != Unit("pixel")
    assert Unit("mm") != "mm"
    assert len({Unit("mm"), Unit("millimeter"), SpaceUnit("mm")}) == 1
    assert {Unit("mm"): 1}[Unit("millimetre")] == 1


@pytest.mark.parametrize(
    "unit",
    [
        Unit("mm"),
        Unit("ms"),
        Unit("s/mm^2"),
        Unit("a.u."),
        Unit("voxel"),
        IndexUnit(),
        Unit["[time] / [length] ** 2"]("s/mm^2"),
    ],
)
def test_units_pickle(unit: Unit) -> None:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        back = pickle.loads(pickle.dumps(unit, protocol=protocol))
        assert back is unit


def test_units_copy_to_themselves() -> None:
    import copy

    unit = Unit("mm")
    assert copy.copy(unit) is unit
    assert copy.deepcopy(unit) is unit


def test_the_pint_escape_hatch() -> None:
    pint_unit = Unit("mm").pint
    assert str(pint_unit) == "millimeter"
    assert Unit.from_pint(pint_unit) is Unit("mm")
    assert Unit.from_pint(pint_unit / Unit("s").pint) is Unit("mm/s")
    assert SpaceUnit.from_pint(pint_unit) is Unit("mm")
    with pytest.raises(ValueError, match="is not a"):
        TimeUnit.from_pint(pint_unit)
    with pytest.raises(TypeError, match="not a pint unit"):
        Unit.from_pint("mm")


# ----------------------------------------------------------------------
#   Predicates
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit, index, physical",
    [
        # An instance.
        (Unit("mm"), False, True),
        (Unit("s"), False, True),
        (Unit("%"), False, True),
        (Unit("index"), True, False),
        (Unit("voxel"), True, False),
        (Unit("pixel"), True, False),
        # A class is a kind of unit, and measures nothing.
        (IndexUnit, True, False),
        (Unit, False, False),
        (SpaceUnit, False, False),
        (TimeUnit, False, False),
        # Unspecified.
        (None, False, False),
        # Not a unit.
        ("mm", False, False),
    ],
)
def test_three_state_predicates(
    unit: object, index: bool, physical: bool
) -> None:
    assert is_indexunit(unit) is index
    assert is_physicalunit(unit) is physical


@pytest.mark.parametrize(
    "unit, space, time",
    [
        (Unit("mm"), True, False),
        (SpaceUnit, True, False),
        (Unit("ms"), False, True),
        (TimeUnit, False, True),
        (Unit("Hz"), False, False),
        (Unit("index"), False, False),
        (None, False, False),
    ],
)
def test_kind_predicates(unit: object, space: bool, time: bool) -> None:
    assert is_spaceunit(unit) is space
    assert is_timeunit(unit) is time


def test_predicates_are_exported() -> None:
    for name in ("is_spaceunit", "is_timeunit", "is_indexunit"):
        assert name in units.__all__
    assert "IndexUnit" in units.__all__


# ----------------------------------------------------------------------
#   Deprecated names
# ----------------------------------------------------------------------


def test_sample_unit_is_a_deprecated_alias() -> None:
    with pytest.warns(DeprecationWarning, match="SampleUnit"):
        assert units.SampleUnit is IndexUnit
    with pytest.warns(DeprecationWarning, match="is_sampleunit"):
        assert units.is_sampleunit is is_indexunit
    with pytest.warns(DeprecationWarning, match="SampleUnit"):
        from brainhops.datamodel.units import SampleUnit  # noqa: F401


def test_the_sample_name_is_a_deprecated_alias() -> None:
    with pytest.warns(DeprecationWarning, match="'sample' is deprecated"):
        assert Unit("sample") is IndexUnit()
    with pytest.warns(DeprecationWarning, match="'sample' is deprecated"):
        assert SpaceAxis(name="x", unit="sample").unit is IndexUnit()


@pytest.mark.parametrize(
    "name, unit", [("MilliMeter", "mm"), ("Second", "s"), ("Inch", "in")]
)
def test_the_unit_classes_are_deprecated_constants(
    name: str, unit: str
) -> None:
    with pytest.warns(DeprecationWarning, match=name):
        constant = getattr(units, name)
    assert constant() is Unit(unit)


def test_an_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        units.NotAUnit  # noqa: B018


def test_the_current_names_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Unit("index")
        Unit("mm")
        IndexUnit()
        is_indexunit(None)


# ----------------------------------------------------------------------
#   Lazy import
# ----------------------------------------------------------------------


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_importing_brainhops_does_not_import_pint() -> None:
    out = _run(
        """
        import sys
        import brainhops
        import brainhops.datamodel.units as units
        import brainhops.io
        from brainhops.datamodel.units import IndexUnit, SpaceUnit, Unit
        print("pint" in sys.modules, units._registry_instance is None)
        # Names resolved without pint, and class-level questions.
        Unit("mm"); IndexUnit(); Unit("voxel"); units.is_spaceunit(SpaceUnit)
        print("pint" in sys.modules)
        # The first name pint has to parse imports it.
        Unit("parsec")
        print("pint" in sys.modules)
        """
    )
    assert out.split() == ["False", "True", "False", "True"]
