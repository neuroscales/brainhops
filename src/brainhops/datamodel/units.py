# TODO/WIP: copied some stuff from ngtools, but I'd like to rework it to
# get someething that fits (in style) with the rest of the data model.

from enum import StrEnum

import typing_extensions as _tx

from .struct import SpecializedStruct
from .typing import HiddenConst

_MU1 = '\u00B5'
_MU2 = '\u03BC'

SI_PREFIX_SHORT2LONG: dict[str, str] = {
    "Q": "quetta",
    "R": "ronna",
    "Y": "yotta",
    "Z": "zetta",
    "E": "exa",
    "P": "peta",
    "T": "tera",
    "G": "giga",
    "M": "mega",
    "K": "kilo",
    "k": "kilo",
    "H": "hecto",
    "h": "hecto",
    "D": "deca",
    "da": "deca",
    "": "",
    "d": "deci",
    "c": "centi",
    "m": "milli",
    "u": "micro",
    _MU1: "micro",
    _MU2: "micro",
    "n": "nano",
    "p": "pico",
    "f": "femto",
    "a": "atto",
    "z": "zepto",
    "y": "yocto",
    "r": "ronto",
    "q": "quecto",
}

SI_PREFIX_LONG2SHORT: dict[str, str] = {
    long: short
    for short, long in SI_PREFIX_SHORT2LONG.items()
}

SI_PREFIX_EXPONENT: dict[str, int] = {
    "Q": 30,
    "R": 27,
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "K": 3,
    "k": 3,
    "H": 2,
    "h": 2,
    "D": 1,
    "da": 1,
    "": 0,
    "d": -1,
    "c": -2,
    "m": -3,
    "u": -6,
    _MU1: -6,
    _MU2: -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
    "r": -27,
    "q": -30,
}

_PREFIX_SI_MEMBERS = dict(SI_PREFIX_SHORT2LONG)
_PREFIX_SI_MEMBERS.pop("")
PrefixSI = StrEnum("PrefixSI", _PREFIX_SI_MEMBERS.items())

UNITS_TIME = {
    "s": "s",
    "sec": "s",
    "second": "s",
    "m": "m",
    "min": "m",
    "minute": "m",
    "h": "h",
    "hour": "h",
    "d": "d",
    "day": "d",
}
TimeUnit = StrEnum("TimeUnit", UNITS_TIME.items())

UNIT_SPACE = {
    # SI
    "m": "m", "meter": "m", "metre": "m",
    # Imperial
    "ft": "ft",
    "foot": "ft",
    "'": "ft",
    "in": "in",
    "inch": "in",
    '"': "in",
    "yd": "yd",
    "yard": "yd",
    "mile": "mi",
    # Other
    "angstrom": "Å",
    "parsec": "pc",
}
SpaceUnit = StrEnum("SpaceUnit", UNIT_SPACE.items())


class SpaceUnitBase(StrEnum):
    meter = metre = "meter"
    inch = "inch"
    foot = "foot"
    yard = "yard"
    mile = "mile"
    angstrom = "angstrom"
    parsec = "parsec"


class TimeUnitBase(StrEnum):
    second = sec = "second"
    minute = min = "minute"
    hour = "hour"
    day = "day"
    year = "year"


UnitBase = StrEnum("UnitBase", { **SpaceUnitBase.__members__, **TimeUnitBase.__members__ })


_UNITS = {}


def register(cls: type) -> type:
    _UNITS[cls] = cls()
    return cls


def siunit(globals: dict) -> _tx.Callable[[type], type]:

    def decorator(cls: type) -> type:

        base = register(cls.base)

        for prefix in PrefixSI:
            name = prefix.capitalize() + base.capitalize()
            kls = type(name, cls.__bases__, { "prefix": prefix})
            globals[name] = kls
            register(kls)

        return base

    return decorator


class Unit(SpecializedStruct):
    name: str
    type: _tx.Literal["time", "space"]
    scale: float = 1.0

    def __new__(cls, *args, **kwargs):
        if cls in _UNITS:
            return _UNITS[cls]
        return super().__new__(cls)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class UnitSI(Unit):
    base: _tx.Literal["second", "meter"] = "second"
    prefix: _tx.Optional[PrefixSI] = None

    @property
    def name(self) -> str:
        prefix = self.prefix or ""
        return f"{prefix}{self.base}"

    @property
    def shortprefix(self) -> _tx.Optional[str]:
        if self.prefix is None:
            return None
        return SI_PREFIX_LONG2SHORT[self.prefix]

    @property
    def shortbase(self) -> str:
        return SI_UNITS[self.base]

    @property
    def shortname(self) -> str:
        shortprefix = self.shortprefix or ""
        return f"{shortprefix}{self.shortbase}"
    
    @property
    def log_scale(self) -> int:
        return SI_PREFIX_EXPONENT[self.shortprefix or ""]

    @property
    def scale(self) -> float:
        return 10 ** (self.log_scale)


class TimeUnit(Unit):
    type: HiddenConst[_tx.Literal["time"]] = "time"


class TimeUnitSI(UnitSI, TimeUnit):
    base: HiddenConst[_tx.Literal["second"]] = "second"


@siunit(globals())
class Second(TimeUnitSI):
    prefix: HiddenConst[str] = ""


@register
class Minute(TimeUnit):
    name: HiddenConst[str] = "minute"
    scale: HiddenConst[float] = 60.0


@register
class Hour(TimeUnit):
    name: HiddenConst[str] = "hour"
    scale: HiddenConst[float] = 3600.0


@register
class Day(TimeUnit):
    name: HiddenConst[str] = "day"
    scale: HiddenConst[float] = 86400.0


@register
class SpaceUnit(Unit):
    type: HiddenConst[_tx.Literal["space"]] = "space"


@register
class SpaceUnitSI(UnitSI, SpaceUnit):
    base: HiddenConst[_tx.Literal["meter"]] = "meter"


@siunit(globals())
class Meter(SpaceUnitSI):
    prefix: HiddenConst[str] = ""


@register
class Inch(SpaceUnit):
    name: HiddenConst[str] = "inch"
    scale: HiddenConst[float] = 0.0254


@register
class Foot(SpaceUnit):
    name: HiddenConst[str] = "foot"
    scale: HiddenConst[float] = 0.3048


@register
class Yard(SpaceUnit):
    name: HiddenConst[str] = "yard"
    scale: HiddenConst[float] = 0.9144


@register
class Mile(SpaceUnit):
    name: HiddenConst[str] = "mile"
    scale: HiddenConst[float] = 1609.344


@register
class Angstrom(SpaceUnit):
    name: HiddenConst[str] = "angstrom"
    scale: HiddenConst[float] = 1e-10


@register
class Parsec(SpaceUnit):
    name: HiddenConst[str] = "parsec"
    scale: HiddenConst[float] = 3.085677581491367e16
