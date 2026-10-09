"""Units of measurement.

A [`Unit`][] is created from a name, for example `Unit("mm")` or
`Unit("s/mm^2")`. Units are immutable, hashable and picklable. They are also
interned: any two spellings of the same unit, such as `"mm"` and
`"millimeter"`, produce the same object.

Names are parsed by a private pint registry, which adds index units
(`index`, `voxel` and `pixel`), the arbitrary unit (`a.u.`) and the
Hounsfield unit (`HU`) to pint's defaults. The registry is built the first
time a name needs pint, not when brainhops is imported. Pint types appear
only through [`Unit.to_pint`][] and [`Unit.from_pint`][].

## Units restricted to a dimension

A subclass of [`Unit`][] can be restricted to a dimension, and then only
builds units of that dimension. [`SpaceUnit`][], [`TimeUnit`][] and
[`IndexUnit`][] are declared in this way:

```python
class SpaceUnit(Unit, dimension="length"): ...
```

Subscripting [`Unit`][] builds and caches the class of any other dimension:

```python
DiffusionUnit = Unit["time / length ** 2"]
DiffusionUnit("s/mm^2")   # 'second / millimeter ** 2'
DiffusionUnit("mm")       # ValueError: 'millimeter' is not a ...
```

These classes are what field type hints name: a field typed
`Optional[Union[SpaceUnit, IndexUnit]]` holds a length unit, an index unit
or nothing. Dimensions use pint's base dimensions plus `index`, with or
without brackets. `Unit(name)` returns an instance of the declared class
for its dimension, so `isinstance(Unit("mm"), SpaceUnit)` is true.

!!! note "Angles"
    As in pint, angles are dimensionless, so a degree is compatible with a
    percent.
"""

# TODO: a dedicated `angle` dimension, so that degrees and percent differ.

__all__ = [
    "Unit",
    "SpaceUnit",
    "TimeUnit",
    "IndexUnit",
    "is_indexunit",
    "is_physicalunit",
    "is_spaceunit",
    "is_timeunit",
]

import math
import re
import threading

import typing_extensions as tx

from brainhops._core import dependencies as _dependencies
from brainhops._core.typing import is_instance_or_subclass

if tx.TYPE_CHECKING:  # pragma: no cover
    import pint

# ----------------------------------------------------------------------
#   THE PRIVATE REGISTRY (built on first use)
# ----------------------------------------------------------------------

_DEFINITIONS = (
    # Index units count array elements.
    "index = [index] = idx = indices",
    "voxel = index = vox = voxels",
    # Replaces pint's `pixel`, a unit of [printing_unit].
    "pixel = index = _ = pixels",
    # Dimensionless imaging units.
    "arbitrary_unit = [] = a.u. = au = arbitrary_units",
    "hounsfield_unit = [] = HU = hounsfield_units",
    # A spelling pint lacks.
    "@alias light_year = lyr",
)
"""Definitions added to pint's default registry."""

_registry_instance: tx.Optional["pint.UnitRegistry"] = None
_registry_lock = threading.Lock()


def _registry() -> "pint.UnitRegistry":
    """Return the private registry, built on first use."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = _build_registry()
    return _registry_instance


def _build_registry() -> "pint.UnitRegistry":
    pint = _dependencies.pint
    if pint is None:
        raise ImportError(
            "Units are parsed by pint, a dependency of brainhops that is "
            "not installed: `pip install pint`."
        )
    registry = pint.UnitRegistry(on_redefinition="ignore")
    for definition in _DEFINITIONS:
        registry.define(definition)
    # A redefined unit (pixel) keeps its cached dimension until rebuilt.
    build_cache = getattr(registry, "_build_cache", None)
    if build_cache is not None:
        build_cache()
    return registry


# ----------------------------------------------------------------------
#   PARSING
# ----------------------------------------------------------------------

_ALIASES: tx.Dict[str, str] = {
    # Pint splits on ".": "a.u." would be atomic mass unit * year.
    "a.u.": "arbitrary_unit",
    "a.u": "arbitrary_unit",
    "A.U.": "arbitrary_unit",
    "A.U": "arbitrary_unit",
    # Pint reads these as unterminated strings.
    "'": "foot",
    '"': "inch",
}
"""Names pint cannot parse, replaced before parsing."""

_EQUIVALENTS: tx.Dict[str, str] = {
    # Pint's micron equals the micrometer in size but not in identity,
    # and cannot be redefined as an alias.
    "micron": "micrometer",
}
"""Pint units replaced after parsing by the unit they equal."""

_PINT_FREE: tx.Dict[str, tx.Tuple[str, str]] = {
    "index": ("index", "[index]"),
    "mm": ("millimeter", "[length]"),
    "millimeter": ("millimeter", "[length]"),
}
"""Names resolved without pint, as (canonical name, dimension).

Default coordinate system axes use these units at import time, so resolving
them here keeps pint out of `import brainhops`. A test checks that each
entry matches the registry.
"""

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPERSCRIPT_RUN = re.compile("[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# Only recent pint versions know "mc" as the micro prefix.
_MIC_PREFIX = re.compile(r"^(?:mic|mc)(?!ro)(?=[a-zA-Z])")


def _normalize(name: str) -> str:
    """Rewrite superscripts and mic/mc prefixes for pint."""
    name = _SUPERSCRIPT_RUN.sub(
        lambda match: "**" + match.group(0).translate(_SUPERSCRIPTS), name
    )
    return _MIC_PREFIX.sub("micro", name)


def _parse(name: str) -> "pint.Unit":
    """Return the pint unit for a name, or raise ValueError."""
    spelled = _ALIASES.get(name, name)
    spelled = _normalize(spelled)
    try:
        unit = _registry().parse_units(spelled)
    except Exception as e:
        # Malformed names can raise TypeError ("ms-1"), AssertionError...
        raise ValueError(
            f"{name!r} is not a unit brainhops recognizes. Spell it as "
            f"pint does, such as 'mm', 'micrometer', 'ms', 's/mm^2' or "
            f"'voxel'."
        ) from e
    factors = dict(unit._units)
    if any(factor in _EQUIVALENTS for factor in factors):
        unit = _registry().parse_units(
            _format_factors(
                {_EQUIVALENTS.get(n, n): e for n, e in factors.items()}
            )
        )
    return unit


def _canonical_name(unit: "pint.Unit") -> str:
    """Return a spelling-independent full name, with sorted factors."""
    return _format_factors(dict(unit._units))


def _format_factors(factors: tx.Mapping[str, tx.Any]) -> str:
    """Write factors in full, as pint does."""

    def term(name: str, exponent: tx.Any) -> str:
        if exponent == 1:
            return name
        if float(exponent).is_integer():
            exponent = int(exponent)
        return f"{name} ** {exponent}"

    factors = sorted(factors.items())
    numerator = [term(n, e) for n, e in factors if e > 0]
    denominator = [term(n, -e) for n, e in factors if e < 0]
    if not numerator and not denominator:
        return "dimensionless"
    name = " * ".join(numerator) if numerator else "1"
    for factor in denominator:
        name += f" / {factor}"
    return name


_MICRO_SIGN = "µ"
_GREEK_MU = "μ"


def _normalize_symbol(symbol: str) -> str:
    """Write micro as mu (U+03BC), whatever the pint release uses."""
    return symbol.replace(_MICRO_SIGN, _GREEK_MU)


_BARE_DIMENSION = re.compile(r"(?<![\[\w])([A-Za-z_]\w*)(?![\]\w])")


def _normalize_dimension(dimension: str) -> str:
    """Normalize `"time / length ** 2"` to `"[time]/[length]**2"`."""
    dimension = "".join(dimension.split())
    if dimension == "dimensionless":
        return ""
    return _BARE_DIMENSION.sub(r"[\1]", dimension)


def _bare_dimension(dimension: tx.Any) -> str:
    """Write a dimension without brackets."""
    return str(dimension).replace("[", "").replace("]", "")


# ----------------------------------------------------------------------
#   UNIT
# ----------------------------------------------------------------------

_INTERNED: tx.Dict[tx.Tuple[type, str], "Unit"] = {}
"""One unit per (class, canonical name)."""

_DECLARED: tx.Dict[str, tx.Type["Unit"]] = {}
"""Declared classes, by normalized dimension."""

_PARAMETRIZED: tx.Dict[str, tx.Type["Unit"]] = {}
"""Classes built by `Unit[dimension]`, by normalized dimension."""

_KIND_NAMES = {"length": "space"}


class Unit:
    """A unit of measurement.

    The name is parsed as pint does, as in `"mm"`, `"µm"`, `"Hz"`,
    `"s/mm^2"` or `"voxel"`. The result is a [`SpaceUnit`][],
    [`TimeUnit`][] or [`IndexUnit`][] for a length, a duration or an
    index, and a plain [`Unit`][] otherwise. Called without a name,
    [`IndexUnit`][] returns `index`.

    Parameters
    ----------
    name : str or Unit
        The name of a unit, or an existing unit.

    Raises
    ------
    TypeError
        If `name` is neither a string nor a unit.
    ValueError
        If `name` is empty, unknown, or of a dimension the class refuses.
    """

    __slots__ = ("_name", "_pint")

    dimension: tx.ClassVar[tx.Optional[str]] = None
    """The dimension the class is restricted to, or `None`."""

    _default: tx.ClassVar[tx.Optional[str]] = None

    def __init_subclass__(
        cls, dimension: tx.Optional[str] = None, **kwargs: tx.Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if dimension is None:
            return
        cls.dimension = dimension
        if not cls.__dict__.get("_parametrized", False):
            _DECLARED.setdefault(_normalize_dimension(dimension), cls)

    def __class_getitem__(cls, dimension: str) -> tx.Type["Unit"]:
        """Return the class of units of a dimension.

        The class is built once per dimension: `Unit["length"]` is
        [`SpaceUnit`][].
        """
        if cls is not Unit:
            raise TypeError(
                f"{cls.__name__} is already restricted to a dimension; "
                f"subscript `Unit` instead."
            )
        if not isinstance(dimension, str):
            raise TypeError(
                f"A dimension is a string such as 'length', not {dimension!r}."
            )
        key = _normalize_dimension(dimension)
        found = _DECLARED.get(key) or _PARAMETRIZED.get(key)
        if found is not None:
            return found
        name = f"Unit[{dimension!r}]"
        kls = type(
            name,
            (Unit,),
            {
                "__slots__": (),
                "__module__": __name__,
                "__qualname__": name,
                "__doc__": f"A unit of dimension `{dimension}`.",
                "_parametrized": True,
            },
            dimension=dimension,
        )
        return _PARAMETRIZED.setdefault(key, kls)

    # ------------------------------------------------------------------
    #   CONSTRUCTION
    # ------------------------------------------------------------------

    def __new__(cls, *args: tx.Any, **kwargs: tx.Any) -> tx.Self:
        if len(args) + len(kwargs) > 1 or (kwargs and "name" not in kwargs):
            raise TypeError(f"{cls.__name__}() takes a single unit name.")
        value = kwargs["name"] if kwargs else args[0] if args else None
        if value is None:
            if cls._default is None:
                raise TypeError(
                    f"{cls.__name__}() needs the name of a unit, such as 'mm'."
                )
            value = cls._default
        if isinstance(value, Unit):
            if isinstance(value, cls):
                return value
            dimension = type(value).dimension
            if dimension is not None and (
                _SIMPLE_DIMENSION.match(_normalize_dimension(dimension))
                and _excludes(cls, _normalize_dimension(dimension))
            ):
                # Decided without pint: an index unit is not a length.
                raise ValueError(
                    f"{value!r} is not a {cls.__name__}: it is a unit of "
                    f"{_bare_dimension(dimension)}, not of "
                    f"{_bare_dimension(cls.dimension)}."
                )
            return cls._intern(value._name, value._pint, value)
        if not isinstance(value, str):
            raise TypeError(
                f"A {cls.__name__} is built from a name or a unit, not "
                f"{type(value).__name__} {value!r}."
            )
        name = value.strip()
        if not name:
            raise ValueError(
                "An empty name is not a unit; leave the unit unspecified "
                "(None) or name it, such as 'mm' or 'dimensionless'."
            )
        known = _PINT_FREE.get(name)
        if known is not None:
            canonical, dimension = known
            target = cls._target_for(dimension)
            if target is not None:
                return target._lookup(canonical, None)
            if _excludes(cls, dimension):
                raise ValueError(
                    f"{value!r} is not a {cls.__name__}: it is a unit of "
                    f"{_bare_dimension(dimension)}, not of "
                    f"{_bare_dimension(cls.dimension)}."
                )
        return cls._intern(None, _parse(name), value)

    def __init__(self, *args: tx.Any, **kwargs: tx.Any) -> None:
        pass

    @classmethod
    def _target_for(cls, dimension: str) -> tx.Optional[tx.Type["Unit"]]:
        """Return the class for a dimension, or `None` if pint must decide."""
        key = _normalize_dimension(dimension)
        if cls is Unit:
            return _DECLARED.get(key, Unit)
        if cls.dimension is not None and (
            _normalize_dimension(cls.dimension) == key
        ):
            return cls
        return None

    @classmethod
    def _lookup(
        cls, canonical: str, unit: tx.Optional["pint.Unit"]
    ) -> tx.Self:
        """Return the interned unit of this class."""
        found = _INTERNED.get((cls, canonical))
        if found is None:
            found = object.__new__(cls)
            object.__setattr__(found, "_name", canonical)
            object.__setattr__(found, "_pint", unit)
            found = _INTERNED.setdefault((cls, canonical), found)
        return found

    @classmethod
    def _intern(
        cls,
        canonical: tx.Optional[str],
        unit: tx.Optional["pint.Unit"],
        source: tx.Any,
    ) -> tx.Self:
        """Return the interned unit, checking its dimension."""
        if unit is None:
            unit = _registry().parse_units(canonical)
        if canonical is None:
            canonical = _canonical_name(unit)
        if cls is Unit:
            return _class_of(unit)._lookup(canonical, unit)
        if cls.dimension is not None:
            expected = _dimensionality(cls.dimension)
            if unit.dimensionality != expected:
                raise ValueError(
                    f"{source!r} is not a {cls.__name__}: it is a unit of "
                    f"{_bare_dimension(unit.dimensionality)}, not of "
                    f"{_bare_dimension(cls.dimension)}."
                )
        return cls._lookup(canonical, unit)

    @classmethod
    def from_pint(cls, unit: tx.Any) -> tx.Self:
        """Build a unit from a pint unit of any registry.

        Raises
        ------
        TypeError
            If `unit` is not a pint unit.
        """
        if not hasattr(unit, "_units") or not hasattr(unit, "dimensionality"):
            raise TypeError(f"{unit!r} is not a pint unit.")
        return cls(_canonical_name(unit))

    # ------------------------------------------------------------------
    #   VALUE SEMANTICS
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._name == other._name
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._name != other._name
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Unit, self._name))

    def __reduce__(self) -> tx.Tuple[tx.Any, ...]:
        kls = type(self)
        if kls.__dict__.get("_parametrized", False):
            return (_restore, (kls.dimension, self._name))
        return (_restore, (kls, self._name))

    def __copy__(self) -> tx.Self:
        return self

    def __deepcopy__(self, memo: tx.Any) -> tx.Self:
        return self

    def __setattr__(self, name: str, value: tx.Any) -> None:
        raise AttributeError(f"A {type(self).__name__} is immutable.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"A {type(self).__name__} is immutable.")

    def __str__(self) -> str:
        """Return the name, such as `"millimeter"`."""
        return self._name

    def __repr__(self) -> str:
        """Return the quoted name."""
        return repr(self._name)

    # ------------------------------------------------------------------
    #   PROPERTIES
    # ------------------------------------------------------------------

    def to_pint(self) -> "pint.Unit":
        """Return the unit in the private pint registry.

        This is the converse of [`from_pint`][].
        """
        unit = self._pint
        if unit is None:
            unit = _registry().parse_units(self._name)
            object.__setattr__(self, "_pint", unit)
        return unit

    @property
    def name(self) -> str:
        """The full name, such as `"millimeter"` or
        `"second / millimeter ** 2"`."""
        return self._name

    @property
    def symbol(self) -> str:
        """The symbol, such as `"mm"`, `"μm"` or `"s / mm ** 2"`."""
        return _normalize_symbol(format(self.to_pint(), "~"))

    @property
    def dimensionality(self) -> str:
        """The dimension without brackets, such as `"length"` or
        `"1 / time"`."""
        return _bare_dimension(self.to_pint().dimensionality)

    @property
    def type(self) -> str:
        """The kind of quantity, such as `"space"`, `"time"` or `"index"`.

        Other units give their dimension. Units of the same type convert
        into one another.
        """
        dimension = self.dimensionality
        return _KIND_NAMES.get(dimension, dimension)

    @property
    def prefix(self) -> tx.Optional[str]:
        """The SI prefix, such as `"milli"`, or `None`."""
        factors = dict(self.to_pint()._units)
        if len(factors) != 1:
            return None
        ((name, exponent),) = factors.items()
        if exponent != 1:
            return None
        parsed = _registry().parse_unit_name(name)
        if not parsed:
            return None
        return parsed[0][0] or None

    @property
    def scale(self) -> float:
        """The size in base units (meters, seconds, or 1.0 for indices).

        Reading it raises a ValueError for a non-multiplicative unit.
        """
        factor, _ = _registry().get_base_units(self.to_pint())
        if factor is None:
            raise ValueError(
                f"{self!r} is not a multiplicative unit, so it has no scale."
            )
        return float(factor)

    @property
    def log10_scale(self) -> tx.Union[int, float]:
        """The base-10 logarithm of [`scale`][], an int for powers of ten."""
        value = math.log10(self.scale)
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return int(rounded)
        return value

    def is_compatible_with(self, other: tx.Union["Unit", str]) -> bool:
        """Return whether the unit converts into `other` (or its name)."""
        if not isinstance(other, Unit):
            other = Unit(other)
        return self.to_pint().dimensionality == other.to_pint().dimensionality


_SIMPLE_DIMENSION = re.compile(r"^\[\w+\]$")


def _excludes(cls: tx.Type[Unit], dimension: str) -> bool:
    """Return whether a class refuses a simple dimension, without pint."""
    if cls.dimension is None:
        return False
    own = _normalize_dimension(cls.dimension)
    return bool(_SIMPLE_DIMENSION.match(own)) and own != dimension


_DIMENSIONALITIES: tx.Dict[str, tx.Any] = {}


def _dimensionality(dimension: str) -> tx.Any:
    """Return the pint dimensionality of a dimension string."""
    found = _DIMENSIONALITIES.get(dimension)
    if found is None:
        try:
            found = _registry().get_dimensionality(
                _normalize_dimension(dimension)
            )
        except Exception as e:
            raise ValueError(f"{dimension!r} is not a dimension.") from e
        _DIMENSIONALITIES[dimension] = found
    return found


def _class_of(unit: "pint.Unit") -> tx.Type[Unit]:
    """Return the declared class for the dimension of a pint unit."""
    dimensionality = unit.dimensionality
    for kls in _DECLARED.values():
        if _dimensionality(kls.dimension) == dimensionality:
            return kls
    return Unit


def _restore(kls: tx.Union[tx.Type[Unit], str], name: str) -> Unit:
    """Rebuild a pickled unit."""
    if isinstance(kls, str):
        kls = Unit[kls]
    return kls(name)


# ----------------------------------------------------------------------
#   UNITS OF A DIMENSION
# ----------------------------------------------------------------------


class SpaceUnit(Unit, dimension="length"):
    """A unit of length, such as the millimeter."""

    __slots__ = ()


class TimeUnit(Unit, dimension="time"):
    """A unit of time, such as the second."""

    __slots__ = ()


class IndexUnit(Unit, dimension="index"):
    """A unit of coordinates that count array elements.

    The index units are `index`, `voxel` and `pixel`, which convert into
    one another with a factor of 1. An axis with an index unit is an array
    axis: its coordinates are positions 0 to N-1, so reversing it shifts
    the origin by N-1 instead of flipping the sign. OME-Zarr gives such an
    axis no unit.

    !!! note
        An index unit differs from `unit=None`, which makes no claim and
        allows no conversion, and from dimensionless physical units such as
        the percent. It never converts into a physical unit.
    """

    __slots__ = ()
    _default = "index"


# ----------------------------------------------------------------------
#   PREDICATES
# ----------------------------------------------------------------------


def is_indexunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Return whether a unit, or a unit class, is an index unit."""
    return is_instance_or_subclass(unit, IndexUnit)


def is_physicalunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Return whether a unit measures a physical quantity.

    The result is false for `None`, for an [`IndexUnit`][] and for a
    class. Two units have a conversion factor exactly when both are
    physical and share a dimension.
    """
    return isinstance(unit, Unit) and not isinstance(unit, IndexUnit)


def is_spaceunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Return whether a unit, or a unit class, is a unit of length."""
    return is_instance_or_subclass(unit, SpaceUnit)


def is_timeunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Return whether a unit, or a unit class, is a unit of time."""
    return is_instance_or_subclass(unit, TimeUnit)
