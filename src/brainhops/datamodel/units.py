"""Units of time and space, and the SI prefixes that scale them.

Each concrete unit, such as [`Meter`][] or [`Hour`][], is a singleton
instance of a [`Unit`][] subclass. An SI unit, such as [`Meter`][], also
exposes a prefixed variant for every SI prefix, so `Millimeter` and
`Kilometer` exist alongside it without being spelled out individually.
"""

# TODO: This is all overly complicated and fiddly.
#       But I am happy with the API for now.
#       I'll revisit the implementation at some point.

__all__ = [
    "PrefixName",
    "UnitName",
    "UnitSIName",
    "SpaceUnitName",
    "TimeUnitName",
    "Unit",
    "UnitSI",
    "TimeUnit",
    "TimeUnitSI",
    "Second",
    "Minute",
    "Hour",
    "Day",
    "SpaceUnit",
    "SpaceUnitSI",
    "Meter",
    "Inch",
    "Foot",
    "Yard",
    "Mile",
    "Angstrom",
    "Parsec",
]

# stdlib
from math import log10

# externals
import typing_extensions as tx
from bagof.magic import ClassVar, Magic, MetaMagic

# core
from brainhops._core.enum import StrEnum


def _make_enum(name: str, d: tx.Dict[str, tx.Tuple]) -> StrEnum:
    return StrEnum(
        name,
        [
            (value, key)
            for key, values in d.items()
            for value in reversed(values)
            if isinstance(value, str) and value
        ],
    )


_MU1 = "\u00b5"
_MU2 = "\u03bc"

PREFIX_SI = {
    "quetta": (30, "Q", "quetta"),
    "ronna": (27, "R", "ronna"),
    "yotta": (24, "Y", "yotta"),
    "zetta": (21, "Z", "zetta"),
    "exa": (18, "E", "exa"),
    "peta": (15, "P", "peta"),
    "tera": (12, "T", "tera"),
    "giga": (9, "G", "giga"),
    "mega": (6, "M", "mega"),
    "kilo": (3, "k", "K", "kilo"),
    "hecto": (2, "h", "H", "hecto"),
    "deca": (1, "da", "D", "deca"),
    "": (0, ""),
    "deci": (-1, "d", "deci"),
    "centi": (-2, "c", "centi"),
    "milli": (-3, "m", "milli"),
    "micro": (-6, _MU2, _MU1, "u", "mc", "mic", "micro"),
    "nano": (-9, "n", "nano"),
    "pico": (-12, "p", "pico"),
    "femto": (-15, "f", "femto"),
    "atto": (-18, "a", "atto"),
    "zepto": (-21, "z", "zepto"),
    "yocto": (-24, "y", "yocto"),
    "ronto": (-27, "r", "ronto"),
    "quecto": (-30, "q", "quecto"),
}
PrefixName = _make_enum("PrefixName", PREFIX_SI)
"""The SI prefixes, from `quetta` (the largest) to `quecto` (the smallest)."""


UNITS_TIME = {
    # SI
    "second": (1, "s", "sec", "second", "seconds"),
    # Other
    "minute": (60, "min", "minute", "minutes"),
    "hour": (60 * 60, "h", "hour", "hours"),
    "day": (24 * 60 * 60, "d", "day", "days"),
    "week": (7 * 24 * 60 * 60, "w", "week", "weeks"),
    "year": (52 * 7 * 24 * 60 * 60, "y", "yr", "year", "years"),
    "decade": (10 * 52 * 7 * 24 * 60 * 60, "decade", "decades"),
    "century": (100 * 52 * 7 * 24 * 60 * 60, "century", "centuries"),
    "millennium": (1000 * 52 * 7 * 24 * 60 * 60, "millennium", "millennia"),
}
TimeUnitName = _make_enum("TimeUnitName", UNITS_TIME)
"""The recognized names of units of time, from `second` to `millennium`."""

UNITS_SPACE = {
    # SI
    "meter": (1, "m", "metre", "meter", "metres", "meters"),
    # Imperial
    "inch": (0.0254, '"', "in", "inch", "inches"),
    "foot": (0.3048, "'", "ft", "foot", "feet"),
    "yard": (0.9144, "yd", "yard", "yards"),
    "mile": (1609.344, "mi", "mile", "miles"),
    # Other
    "angstrom": (1e-10, "Å", "angstrom", "angstroms"),
    "parsec": (3.085677581491367e16, "pc", "parsec", "parsecs"),
    "light year": (
        9.4607304725808e15,
        "ly",
        "lyr",
        "light year",
        "light years",
    ),
}
SpaceUnitName = _make_enum("SpaceUnitName", UNITS_SPACE)
"""The recognized names of units of length, from `meter` to `light year`."""

UNITS = {**UNITS_TIME, **UNITS_SPACE}
UnitName = _make_enum("UnitName", UNITS)
"""The recognized names of every unit of time and length."""

UNITS_SI = dict(
    [next(iter(UNITS_SPACE.items())), next(iter(UNITS_TIME.items()))]
)
UnitSIName = _make_enum("UnitSIName", UNITS_SI)
"""The names of the base SI units of time and length, `second` and `meter`."""


def _parse_unit_name(
    name: str,
) -> tx.Tuple[tx.Optional[PrefixName], UnitName]:  # type: ignore
    if name in UnitName.__members__:
        return None, UnitName[name]
    for prefix in PrefixName:
        if name.startswith(prefix):
            base_name = name[len(prefix) :]
            if base_name in UnitSIName.__members__:
                return PrefixName[prefix], UnitSIName[base_name]
    for _, *prefixes in PREFIX_SI.values():
        for *_, suffixes in UNITS_SI.values():
            for prefix in prefixes:
                for suffix in suffixes:
                    if name == prefix + suffix:
                        return PrefixName[prefix], UnitSIName[suffix]
    return None, name


# ----------------------------------------------------------------------
#   DECORATOR TO REGISTER SINGLETON UNITS
# ----------------------------------------------------------------------
_REGISTERED_UNITS = {}


def register(cls: type) -> type:
    """Register the class as a singleton unit.

    An instance of `cls` is created and stored, and is returned every
    time `cls` is subsequently instantiated. The class itself is
    returned unchanged.
    """
    _REGISTERED_UNITS[cls] = cls()
    return cls


def siunit(globals: dict) -> tx.Callable[[type], type]:
    """Register the base SI unit and generate one prefixed class per prefix.

    The decorated class becomes the singleton unit with no prefix.
    A subclass is also created and registered for every member of
    [`PrefixName`][], such as `MilliMeter` for `Meter`, and each is
    written into `globals` under its own name.
    """

    def decorator(cls: type) -> type:

        base = register(cls)
        if "." in cls.__qualname__:
            qualprefix = cls.__qualname__.rsplit(".")[0] + "."
        else:
            qualprefix = ""

        for prefix in PrefixName:
            name = prefix.capitalize() + base.__name__
            kls = type(
                name,
                cls.__bases__,
                {
                    "prefix": prefix,
                    "base": base.base,
                    "__module__": cls.__module__,
                    "__qualname__": qualprefix + name,
                },
            )
            globals[name] = kls
            register(kls)
            __all__.append(name)

        return base

    return decorator


# ----------------------------------------------------------------------
#   BASE CLASSES
# ----------------------------------------------------------------------


class Unit(
    Magic, convert=True, repr=False, slots=True, init=False, mapping=False
):
    """A unit of time or of length.

    Every recognized unit is a singleton: constructing a unit by name
    returns the same instance every time that name is passed, rather
    than a new object.
    """

    name: ClassVar[tx.Optional[str]] = None
    scale: ClassVar[float] = 1.0
    type: ClassVar[tx.Literal["time", "space"]]

    def __new__(cls, *args, **kwargs) -> tx.Self:
        if cls in _REGISTERED_UNITS:
            return _REGISTERED_UNITS[cls]
        name = kwargs.get("name", args[0] if args else None)
        if name:
            prefix, base = _parse_unit_name(name)
            cls_name = str(base).capitalize()
            if prefix:
                cls_name = str(prefix).capitalize() + cls_name
            if cls_name in globals():
                kls = globals()[cls_name]
                if kls in _REGISTERED_UNITS:
                    return _REGISTERED_UNITS[kls]
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        # Do nothing, so to not trigger Magic.__init__
        pass

    def __str__(self) -> str:
        """The unit's name, such as `"meter"`."""
        return getattr(type(self), "name", "<unknown unit>")

    def __repr__(self) -> str:
        """The unit's name, quoted."""
        return f"'{self.__str__()}'"


class _MetaUnitSI(MetaMagic):
    @property
    def name(cls) -> str:
        prefix = cls.prefix or ""
        return f"{prefix}{cls.base}"

    @property
    def prefixsymbol(cls) -> tx.Optional[str]:
        if cls.prefix is None:
            return None
        return PREFIX_SI[cls.prefix][1]

    @property
    def basesymbol(cls) -> str:
        return UNITS_SI[cls.base][1]

    @property
    def symbol(cls) -> str:
        prefixsymbol = cls.prefixsymbol or ""
        return f"{prefixsymbol}{cls.basesymbol}"

    @property
    def log10_scale(cls) -> int:
        return PREFIX_SI[cls.prefix or ""][0]

    @property
    def scale(cls) -> float:
        return 10 ** (cls.log10_scale)


class UnitSI(Unit, metaclass=_MetaUnitSI):
    """An SI unit, optionally scaled by an SI prefix.

    An instance combines a base unit, such as `second` or `meter`, with
    an optional prefix, such as `milli`. Its name, symbol and scale
    relative to the base unit are derived from the two.
    """

    base: ClassVar[UnitSIName] = "second"
    prefix: ClassVar[tx.Optional[PrefixName]] = None

    @classmethod
    def _parse_name(cls, *args, **kwargs) -> PrefixName:  # type: ignore
        name = cls.name
        if args:
            name = args[0]
        prefix, base = _parse_unit_name(name)
        if "prefix" in kwargs:
            prefix = kwargs["prefix"]
        if prefix:
            prefix = PrefixName[prefix]
        if "base" in kwargs:
            base = kwargs["base"]
        if base in UnitSIName.__members__:
            base = UnitSIName[base]
        return (prefix or "") + base

    def __new__(cls, *args, **kwargs) -> tx.Self:
        name = cls._parse_name(*args, **kwargs)
        args = (name,) + args[1:]
        kwargs.pop("name", None)
        return super().__new__(cls, *args, **kwargs)

    @property
    def name(self) -> str:
        """The unit's name, its prefix followed by its base unit's name."""
        return type(self).name

    @property
    def prefixsymbol(self) -> tx.Optional[str]:
        """The symbol of the unit's SI prefix, or `None` if it has none."""
        return type(self).prefixsymbol

    @property
    def basesymbol(self) -> str:
        """The symbol of the unit's base unit, without its prefix."""
        return type(self).basesymbol

    @property
    def symbol(self) -> str:
        """The unit's symbol, its prefix symbol followed by its base symbol."""
        return type(self).symbol

    @property
    def log10_scale(self) -> int:
        """The base-10 logarithm of the unit's scale relative to its
        base unit."""
        return type(self).log10_scale

    @property
    def scale(self) -> float:
        """The unit's scale relative to its base unit."""
        return type(self).scale


class _MetaKnownUnit(MetaMagic):
    @property
    def prefix(cls) -> None:
        return None

    @property
    def symbol(cls) -> str:
        return UNITS[cls.name][1]

    @property
    def scale(cls) -> float:
        return UNITS[cls.name][0]

    @property
    def log10_scale(cls) -> float:
        return log10(cls.scale)


class KnownUnit(Unit, metaclass=_MetaKnownUnit):
    """A named unit with no SI prefix, such as `inch` or `hour`.

    A unit of this kind has a fixed scale relative to its base unit, and
    is not one of the SI-prefixed units generated for [`UnitSI`][].
    """

    @property
    def prefix(self) -> None:
        """`None`. A known unit never carries an SI prefix."""
        return type(self).prefix

    @property
    def symbol(self) -> str:
        """The unit's symbol, for example `"in"` for `Inch`."""
        return type(self).symbol

    @property
    def scale(self) -> float:
        """The unit's scale relative to the base unit of its kind."""
        return type(self).scale

    @property
    def log10_scale(self) -> float:
        """The base-10 logarithm of the unit's scale."""
        return type(self).log10_scale


# ----------------------------------------------------------------------
#   TIME UNITS
# ----------------------------------------------------------------------


class TimeUnit(Unit):
    """A unit of time."""

    type: ClassVar[tx.Literal["time"]] = "time"


class TimeUnitSI(UnitSI, TimeUnit):
    """A unit of time expressed in seconds, optionally SI-prefixed."""

    base: ClassVar[TimeUnitName] = TimeUnitName.second


@siunit(globals())
class Second(TimeUnitSI):
    """The second, the SI base unit of time.

    Prefixed variants, such as `Millisecond`, are also generated and
    registered under their own names.
    """

    prefix: ClassVar[None] = None


@register
class Minute(TimeUnit):
    """The minute, equal to 60 seconds."""

    name: ClassVar[TimeUnitName] = TimeUnitName.minute


@register
class Hour(TimeUnit):
    """The hour, equal to 60 minutes."""

    name: ClassVar[TimeUnitName] = TimeUnitName.hour


@register
class Day(TimeUnit):
    """The day, equal to 24 hours."""

    name: ClassVar[TimeUnitName] = TimeUnitName.day


@register
class Week(TimeUnit):
    """The week, equal to 7 days."""

    name: ClassVar[TimeUnitName] = TimeUnitName.week


@register
class Year(TimeUnit):
    """The year, equal to 52 weeks."""

    name: ClassVar[TimeUnitName] = TimeUnitName.year


# ----------------------------------------------------------------------
#   SPACE UNITS
# ----------------------------------------------------------------------


class SpaceUnit(Unit):
    """A unit of length."""

    type: ClassVar[tx.Literal["space"]] = "space"


class SpaceUnitSI(UnitSI, SpaceUnit):
    """A unit of length expressed in meters, optionally SI-prefixed."""

    base: ClassVar[SpaceUnitName] = SpaceUnitName.meter


@siunit(globals())
class Meter(SpaceUnitSI):
    """The meter, the SI base unit of length.

    Prefixed variants, such as `Millimeter`, are also generated and
    registered under their own names.
    """

    prefix: ClassVar[None] = None


@register
class Inch(SpaceUnit):
    """The inch, equal to 0.0254 meters."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.inch


@register
class Foot(SpaceUnit):
    """The foot, equal to 12 inches."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.foot


@register
class Yard(SpaceUnit):
    """The yard, equal to 3 feet."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.yard


@register
class Mile(SpaceUnit):
    """The mile, equal to 1760 yards."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.mile


@register
class Angstrom(SpaceUnit):
    """The angstrom, equal to 1e-10 meters."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.angstrom


@register
class Parsec(SpaceUnit):
    """The parsec, a unit of astronomical distance equal to about
    3.09e16 meters."""

    name: ClassVar[SpaceUnitName] = SpaceUnitName.parsec
