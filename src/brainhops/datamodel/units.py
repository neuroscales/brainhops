# TODO: This is all overly complicated and fiddly.
#       But I am happy with the API for now.
#       I'll revisit the implementation at some point.

__all__ = [
    "PrefixName", "UnitName", "UnitSIName", "SpaceUnitName", "TimeUnitName",
    "Unit", "UnitSI",
    "TimeUnit", "TimeUnitSI",
    "Second", "Minute", "Hour", "Day",
    "SpaceUnit", "SpaceUnitSI",
    "Meter", "Inch", "Foot", "Yard", "Mile", "Angstrom", "Parsec",
]

# stdlib
from enum import StrEnum
from math import log10

# externals
import typing_extensions as _tx

# internals
from brainhops._ext.struct import ClassVar, MetaStruct, Struct


def _make_enum(name: str, d: _tx.Dict[str, _tx.Tuple]) -> StrEnum:
    return StrEnum(name, [
        (value, key)
        for key, values in d.items()
        for value in reversed(values)
        if isinstance(value, str) and value
    ])


_MU1 = '\u00B5'
_MU2 = '\u03BC'

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


UNITS_TIME = {
    # SI
    "second": (1, "s", "sec", "second", "seconds"),
    # Other
    "minute": (60, "min", "minute", "minutes"),
    "hour": (60*60, "h", "hour", "hours"),
    "day": (24*60*60, "d", "day", "days"),
    "week": (7*24*60*60, "w", "week", "weeks"),
    "year": (52*7*24*60*60, "y", "yr", "year", "years"),
    "decade": (10*52*7*24*60*60, "decade", "decades"),
    "century": (100*52*7*24*60*60, "century", "centuries"),
    "millennium": (1000*52*7*24*60*60, "millennium", "millennia"),
}
TimeUnitName = _make_enum("TimeUnitName", UNITS_TIME)

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
    "parsec": (3.085677581491367e+16, "pc", "parsec", "parsecs"),
    "light year": (9.4607304725808e+15,
                   "ly", "lyr", "light year", "light years"),
}
SpaceUnitName = _make_enum("SpaceUnitName", UNITS_SPACE)

UNITS = {**UNITS_TIME, **UNITS_SPACE}
UnitName = _make_enum("UnitName", UNITS)

UNITS_SI = dict([
    next(iter(UNITS_SPACE.items())),
    next(iter(UNITS_TIME.items()))
])
UnitSIName = _make_enum("UnitSIName", UNITS_SI)


def _parse_unit_name(name: str) -> _tx.Tuple[_tx.Optional[PrefixName], UnitName]:  # type: ignore # noqa: E501
    if name in UnitName.__members__:
        return None, UnitName[name]
    for prefix in PrefixName:
        if name.startswith(prefix):
            base_name = name[len(prefix):]
            if base_name in UnitSIName.__members__:
                return PrefixName[prefix], UnitSIName[base_name]
    for (_, *prefixes) in PREFIX_SI.values():
        for (*_, suffixes) in UNITS_SI.values():
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
    _REGISTERED_UNITS[cls] = cls()
    return cls


def siunit(globals: dict) -> _tx.Callable[[type], type]:

    def decorator(cls: type) -> type:

        base = register(cls)
        if "." in cls.__qualname__:
            qualprefix = cls.__qualname__.rsplit(".")[0] + "."
        else:
            qualprefix = ""

        for prefix in PrefixName:
            name = prefix.capitalize() + base.__name__
            kls = type(name, cls.__bases__, {
                "prefix": prefix,
                "base": base.base,
                "__module__": cls.__module__,
                "__qualname__": qualprefix + name,
            })
            globals[name] = kls
            register(kls)
            __all__.append(name)

        return base

    return decorator


# ----------------------------------------------------------------------
#   BASE CLASSES
# ----------------------------------------------------------------------


class Unit(Struct,
           convert=True,
           repr=False,
           slots=True,
           init=False,
           mapping=False):
    name: ClassVar[_tx.Optional[str]] = None
    scale: ClassVar[float] = 1.0
    type: ClassVar[_tx.Literal["time", "space"]]

    def __new__(cls, *args, **kwargs) -> _tx.Self:
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
        # Do nothing, so to not trigger Struct.__init__
        pass

    def __str__(self) -> str:
        return getattr(type(self), "name", "<unknown unit>")

    def __repr__(self) -> str:
        return f"'{self.__str__()}'"


class _MetaUnitSI(MetaStruct):

    @property
    def name(cls) -> str:
        prefix = cls.prefix or ""
        return f"{prefix}{cls.base}"

    @property
    def prefixsymbol(cls) -> _tx.Optional[str]:
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
    base: ClassVar[UnitSIName] = "second"
    prefix: ClassVar[_tx.Optional[PrefixName]] = None

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

    def __new__(cls, *args, **kwargs) -> _tx.Self:
        name = cls._parse_name(*args, **kwargs)
        args = (name,) + args[1:]
        kwargs.pop("name", None)
        return super().__new__(cls, *args, **kwargs)

    @property
    def name(self) -> str:
        return type(self).name

    @property
    def prefixsymbol(self) -> _tx.Optional[str]:
        return type(self).prefixsymbol

    @property
    def basesymbol(self) -> str:
        return type(self).basesymbol

    @property
    def symbol(self) -> str:
        return type(self).symbol

    @property
    def log10_scale(self) -> int:
        return type(self).log10_scale

    @property
    def scale(self) -> float:
        return type(self).scale


class _MetaKnownUnit(MetaStruct):

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

    @property
    def prefix(self) -> None:
        return type(self).prefix

    @property
    def symbol(self) -> str:
        return type(self).symbol

    @property
    def scale(self) -> float:
        return type(self).scale

    @property
    def log10_scale(self) -> float:
        return type(self).log10_scale


# ----------------------------------------------------------------------
#   TIME UNITS
# ----------------------------------------------------------------------


class TimeUnit(Unit):
    type: ClassVar[_tx.Literal["time"]] = "time"


class TimeUnitSI(UnitSI, TimeUnit):
    base: ClassVar[TimeUnitName] = TimeUnitName.second


@siunit(globals())
class Second(TimeUnitSI):
    prefix: ClassVar[None] = None


@register
class Minute(TimeUnit):
    name: ClassVar[TimeUnitName] = TimeUnitName.minute


@register
class Hour(TimeUnit):
    name: ClassVar[TimeUnitName] = TimeUnitName.hour


@register
class Day(TimeUnit):
    name: ClassVar[TimeUnitName] = TimeUnitName.day


@register
class Week(TimeUnit):
    name: ClassVar[TimeUnitName] = TimeUnitName.week


@register
class Year(TimeUnit):
    name: ClassVar[TimeUnitName] = TimeUnitName.year


# ----------------------------------------------------------------------
#   SPACE UNITS
# ----------------------------------------------------------------------


class SpaceUnit(Unit):
    type: ClassVar[_tx.Literal["space"]] = "space"


class SpaceUnitSI(UnitSI, SpaceUnit):
    base: ClassVar[SpaceUnitName] = SpaceUnitName.meter


@siunit(globals())
class Meter(SpaceUnitSI):
    prefix: ClassVar[None] = None


@register
class Inch(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.inch


@register
class Foot(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.foot


@register
class Yard(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.yard


@register
class Mile(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.mile


@register
class Angstrom(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.angstrom


@register
class Parsec(SpaceUnit):
    name: ClassVar[SpaceUnitName] = SpaceUnitName.parsec
