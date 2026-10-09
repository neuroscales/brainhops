"""
Units of measurement, backed by a private
[pint](https://pint.readthedocs.io) registry.

A [`Unit`][] is built from a name -- `Unit("mm")`, `Unit("millisecond")`,
`Unit("s/mm^2")` -- and is an interned, hashable, picklable value: two
spellings of the same unit give equal units (`Unit("mm") ==
Unit("millimeter")`), and the same spelling gives the same object.

The name is parsed by [pint](https://pint.readthedocs.io), through a
registry that belongs to brainhops and that is extended with the units
imaging data needs: the *index* units (`index`, `voxel`, `pixel`, of
dimension `index`), the arbitrary unit (`a.u.`) and the Hounsfield unit
(`HU`). pint is imported, and the registry built, the first time a name
is parsed, not when `brainhops` is imported. pint types never appear in
the API, apart from the [`Unit.to_pint`][] and [`Unit.from_pint`][]
escape hatches.

## Units restricted to a dimension

A subclass of [`Unit`][] can be restricted to a dimension, and then only
builds units of that dimension. [`SpaceUnit`][] (`length`),
[`TimeUnit`][] (`time`) and [`IndexUnit`][] (`index`) are declared this
way:

```python
class SpaceUnit(Unit, dimension="length"): ...
```

and any other dimension can be asked for with a subscript, which builds
(and caches) such a class:

```python
DiffusionUnit = Unit["time / length ** 2"]
DiffusionUnit("s/mm^2")   # 'second / millimeter ** 2'
DiffusionUnit("mm")       # ValueError: 'millimeter' is not a ...
```

These classes are what a field's type hint names, so that a field typed
`Optional[Union[SpaceUnit, IndexUnit]]` holds a unit of length, an index
unit or nothing, and converts a name to one of these.

A dimension is written with pint's base dimensions (`length`, `time`,
`mass`, ..., and brainhops' `index`), with or without pint's square
brackets: `"time / length ** 2"` is `"[time] / [length] ** 2"`.

`Unit(name)` returns an instance of the declared class of the unit's
dimension, so `isinstance(Unit("mm"), SpaceUnit)` is true.

!!! note "Angles"

    Angles are dimensionless, as pint has them: a degree is compatible
    with a percent. A dedicated `angle` dimension may be introduced
    later.
"""

# TODO: Investigate a dedicated `angle` dimension, so that angles are
#       not interchangeable with every other dimensionless unit (`deg` is
#       currently compatible with `%`).

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

# stdlib
import math
import re
import threading

# externals
import typing_extensions as tx

# core
from brainhops._core import dependencies as _dependencies
from brainhops._core.typing import is_instance_or_subclass

if tx.TYPE_CHECKING:  # pragma: no cover
    import pint

# ----------------------------------------------------------------------
#   THE PRIVATE REGISTRY (built on first use)
# ----------------------------------------------------------------------

_DEFINITIONS = (
    # Index units: coordinates that count the elements of an array.
    "index = [index] = idx = indices",
    "voxel = index = vox = voxels",
    # Replaces pint's `pixel`, a unit of `[printing_unit]`.
    "pixel = index = _ = pixels",
    # Dimensionless units of imaging data.
    "arbitrary_unit = [] = a.u. = au = arbitrary_units",
    "hounsfield_unit = [] = HU = hounsfield_units",
    # A spelling of the light year that pint does not know.
    "@alias light_year = lyr",
)
"""The definitions brainhops adds to pint's default registry."""

_registry_instance: tx.Optional["pint.UnitRegistry"] = None
_registry_lock = threading.Lock()


def _registry() -> "pint.UnitRegistry":
    """The private unit registry, built on first use."""
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
    # The registry caches the dimension of every unit it knows when it is
    # built, so a unit it is given again (`pixel`) keeps its old dimension
    # until the cache is rebuilt.
    build_cache = getattr(registry, "_build_cache", None)
    if build_cache is not None:
        build_cache()
    return registry


# ----------------------------------------------------------------------
#   PARSING
# ----------------------------------------------------------------------

_ALIASES: tx.Dict[str, str] = {
    # pint's tokenizer splits on ".", so `a.u.` would parse as
    # `unified_atomic_mass_unit * year` (and `A.U.` as `ampere *
    # enzyme_unit`) if it reached pint.
    "a.u.": "arbitrary_unit",
    "a.u": "arbitrary_unit",
    "A.U.": "arbitrary_unit",
    "A.U": "arbitrary_unit",
    # The symbols of the foot and the inch, which pint's tokenizer reads
    # as unterminated strings.
    "'": "foot",
    '"': "inch",
}
"""Names matched exactly, before pint parses anything: the ones pint
cannot parse, whatever its definitions."""

_EQUIVALENTS: tx.Dict[str, str] = {
    # pint's `micron` is a unit of its own, equal in size to the
    # micrometer but not equal to it, and it cannot be made an alias of
    # the micrometer since pint already defines it.
    "micron": "micrometer",
}
"""Units of pint replaced, after parsing, by the unit they equal."""

_PINT_FREE: tx.Dict[str, tx.Tuple[str, str]] = {
    "index": ("index", "[index]"),
    "mm": ("millimeter", "[length]"),
    "millimeter": ("millimeter", "[length]"),
}
"""Names resolved without pint, to their canonical name and dimension.

brainhops builds units from these names when it is imported -- the default
axes of the voxel and array coordinate systems are in `"index"`, and those
of the `RASmm`-like systems in `"mm"` -- and resolving them here is what
keeps pint out of `import brainhops`. Every entry is what the registry
gives, which a test checks."""

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPERSCRIPT_RUN = re.compile("[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# "mic" and "mc" for "micro", as in "micm" or "mcs"; pint knows "u" and
# "µ", and only its recent versions know "mc".
_MIC_PREFIX = re.compile(r"^(?:mic|mc)(?!ro)(?=[a-zA-Z])")


def _normalize(name: str) -> str:
    """Spell a unit name the way pint reads it."""
    name = _SUPERSCRIPT_RUN.sub(
        lambda match: "**" + match.group(0).translate(_SUPERSCRIPTS), name
    )
    return _MIC_PREFIX.sub("micro", name)


def _parse(name: str) -> "pint.Unit":
    """The pint unit a name stands for, or a `ValueError`."""
    spelled = _ALIASES.get(name, name)
    spelled = _normalize(spelled)
    try:
        unit = _registry().parse_units(spelled)
    except Exception as e:
        # pint raises its own errors for an unknown name, but a malformed
        # string can also raise a `TypeError` (`"ms-1"`), an
        # `AssertionError` (`"'"`), ...
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
    """A name that two equal pint units share, and that pint parses back.

    It reads as pint writes a unit in full (`"second / millimeter ** 2"`),
    with the factors sorted, so that it does not depend on how the unit
    was spelled.
    """
    return _format_factors(dict(unit._units))


def _format_factors(factors: tx.Mapping[str, tx.Any]) -> str:
    """Write units raised to exponents as pint writes them in full."""

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


_MICRO_SIGN = "\u00b5"
_GREEK_MU = "\u03bc"


def _normalize_symbol(symbol: str) -> str:
    """Write the micro prefix of a symbol with one character, the Greek mu.

    pint's micro prefix lists both the micro sign (U+00B5) and the Greek
    mu (U+03BC), and which of the two is its symbol depends on the pint
    release (the micro sign until 0.25, the Greek mu from 0.26).
    """
    return symbol.replace(_MICRO_SIGN, _GREEK_MU)


_BARE_DIMENSION = re.compile(r"(?<![\[\w])([A-Za-z_]\w*)(?![\]\w])")


def _normalize_dimension(dimension: str) -> str:
    """A dimension in pint's notation, without spaces: `"time / length **
    2"` and `"[time] / [length] ** 2"` are both `"[time]/[length]**2"`."""
    dimension = "".join(dimension.split())
    if dimension == "dimensionless":
        return ""
    return _BARE_DIMENSION.sub(r"[\1]", dimension)


def _bare_dimension(dimension: tx.Any) -> str:
    """A dimension without pint's brackets, such as `"time / length **
    2"`."""
    return str(dimension).replace("[", "").replace("]", "")


# ----------------------------------------------------------------------
#   UNIT
# ----------------------------------------------------------------------

_INTERNED: tx.Dict[tx.Tuple[type, str], "Unit"] = {}
"""One unit per class and canonical name."""

_DECLARED: tx.Dict[str, tx.Type["Unit"]] = {}
"""The class declared for each dimension, by normalized dimension."""

_PARAMETRIZED: tx.Dict[str, tx.Type["Unit"]] = {}
"""The classes built by `Unit[dimension]`, by normalized dimension."""

_KIND_NAMES = {"length": "space"}


class Unit:
    """A unit of measurement.

    `Unit(name)` parses a name, such as `"mm"`, `"micrometer"`, `"µm"`,
    `"ms"`, `"Hz"`, `"s/mm^2"` or `"voxel"`, and returns the unit it
    stands for: an instance of [`SpaceUnit`][], [`TimeUnit`][] or
    [`IndexUnit`][] when the unit is a length, a duration or an index, and
    of [`Unit`][] otherwise.

    Units are values: two spellings of the same unit give equal units with
    equal hashes, and the same unit built twice through the same class is
    the same object. A name that is not a unit raises a `ValueError`.

    A subclass restricted to a dimension (see the module documentation)
    only builds units of that dimension, and raises a `ValueError` for any
    other.

    Parameters
    ----------
    unit : str | Unit
        The name of the unit, or a unit.
    """

    __slots__ = ("_name", "_pint")

    dimension: tx.ClassVar[tx.Optional[str]] = None
    """The dimension that the class restricts its units to, such as
    `"length"`, or `None` for any."""

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
        """The class of the units of a dimension, such as
        `Unit["time / length ** 2"]`. It is built once per dimension, and
        `Unit["length"]` is [`SpaceUnit`][]."""
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
                # Refused without pint: an index unit is not a length.
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
        # Everything is done in `__new__`.
        pass

    @classmethod
    def _target_for(cls, dimension: str) -> tx.Optional[tx.Type["Unit"]]:
        """The class that holds a unit of a known dimension, without
        pint, or `None` if pint is needed to tell."""
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
        """The interned unit of this class with a canonical name."""
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
        """The interned unit of `unit`, as an instance of the class of its
        dimension, which must be one this class allows."""
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
        """The unit of a pint unit, from any registry that names it the
        way brainhops' registry does."""
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
        """The unit's name, such as `"millimeter"`."""
        return self._name

    def __repr__(self) -> str:
        """The unit's name, quoted."""
        return repr(self._name)

    # ------------------------------------------------------------------
    #   PROPERTIES
    # ------------------------------------------------------------------

    def to_pint(self) -> "pint.Unit":
        """The unit in brainhops' private pint registry.

        This is an escape hatch to pint, the converse of
        [`from_pint`][brainhops.datamodel.units.Unit.from_pint]; it is the
        only place where a pint object comes out of brainhops."""
        unit = self._pint
        if unit is None:
            unit = _registry().parse_units(self._name)
            object.__setattr__(self, "_pint", unit)
        return unit

    @property
    def name(self) -> str:
        """The unit's name, in full, such as `"millimeter"` or
        `"second / millimeter ** 2"`."""
        return self._name

    @property
    def symbol(self) -> str:
        """The unit's symbol, such as `"mm"` or `"s / mm ** 2"`. The micro
        prefix is always the Greek mu (U+03BC), as in `"μm"`."""
        return _normalize_symbol(format(self.to_pint(), "~"))

    @property
    def dimensionality(self) -> str:
        """The unit's dimension, such as `"length"` or `"1 / time"`."""
        return _bare_dimension(self.to_pint().dimensionality)

    @property
    def type(self) -> str:
        """The kind of quantity the unit measures: `"space"`, `"time"` or
        `"index"`, or else its dimension (`"1 / time"`, `"dimensionless"`,
        ...).

        Two units with the same `type` can be converted into each other.
        """
        dimension = self.dimensionality
        return _KIND_NAMES.get(dimension, dimension)

    @property
    def prefix(self) -> tx.Optional[str]:
        """The SI prefix of a unit such as the millimeter (`"milli"`), or
        `None` for a unit without one or a compound unit."""
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
        """The size of the unit in the base units of its dimension: in
        meters for a length, in seconds for a duration, and in indices
        for an index unit (`1.0`)."""
        factor, _ = _registry().get_base_units(self.to_pint())
        if factor is None:
            raise ValueError(
                f"{self!r} is not a multiplicative unit, so it has no scale."
            )
        return float(factor)

    @property
    def log10_scale(self) -> tx.Union[int, float]:
        """The base-10 logarithm of the unit's scale: an `int` when the
        scale is a power of ten (the millimeter, `-3`), a `float`
        otherwise (the inch)."""
        value = math.log10(self.scale)
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return int(rounded)
        return value

    def is_compatible_with(self, other: tx.Union["Unit", str]) -> bool:
        """Whether the unit converts into `other`: whether the two have
        the same dimension."""
        if not isinstance(other, Unit):
            other = Unit(other)
        return self.to_pint().dimensionality == other.to_pint().dimensionality


_SIMPLE_DIMENSION = re.compile(r"^\[\w+\]$")


def _excludes(cls: tx.Type[Unit], dimension: str) -> bool:
    """Whether `cls` certainly refuses the units of a base dimension, such
    as `"[index]"`, without asking pint."""
    if cls.dimension is None:
        return False
    own = _normalize_dimension(cls.dimension)
    return bool(_SIMPLE_DIMENSION.match(own)) and own != dimension


_DIMENSIONALITIES: tx.Dict[str, tx.Any] = {}


def _dimensionality(dimension: str) -> tx.Any:
    """The pint dimensionality of a dimension such as `"length"`."""
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
    """The declared class of a pint unit's dimension, or `Unit`."""
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
    """An index unit: coordinates count the elements of an array rather
    than measure a quantity.

    Its units are `index`, `voxel` and `pixel`; they are distinct units of
    the same size, so they convert into each other with a factor of one.
    `IndexUnit()` is the `index`.

    An axis whose unit is an index unit is an *array* axis -- its
    coordinates are positions in the array, so they run `0` to `N-1` over
    a finite number of elements, and reversing it shifts the origin by one
    less than its extent rather than flipping a sign about it. This is the
    convention OME-Zarr writes down by giving such an axis no unit at all,
    and the scale transformation is what converts the indices to a
    physical unit.

    !!! note

        This is deliberately distinct from `unit=None`, which means the
        unit is *unspecified* -- nothing is claimed either way, and no
        conversion and no origin shift follow from it. It is also distinct
        from a dimensionless physical unit, such as the percent, which is
        a real, convertible unit. An index unit never converts into a
        physical unit. See [`is_indexunit`][].
    """

    __slots__ = ()
    _default = "index"


# ----------------------------------------------------------------------
#   PREDICATES
# ----------------------------------------------------------------------


def is_indexunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Whether `unit` is an index unit (an instance or a class), i.e. the
    axis is an array axis."""
    return is_instance_or_subclass(unit, IndexUnit)


def is_physicalunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Whether `unit` measures a physical quantity.

    True for a unit such as a millimetre, a second or a percent. False for
    `None`, which leaves the unit *unspecified*, for an [`IndexUnit`][],
    which says the coordinates count the elements of an array rather than
    measure anything, and for a class, which is a kind of unit rather than
    one -- so a conversion factor to another physical unit exists exactly
    when this is true of both and they have the same dimension.
    """
    return isinstance(unit, Unit) and not isinstance(unit, IndexUnit)


def is_spaceunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Whether `unit` is a unit of space, a length (an instance or a class)."""
    return is_instance_or_subclass(unit, SpaceUnit)


def is_timeunit(unit: tx.Union[Unit, tx.Type[Unit], None]) -> bool:
    """Whether `unit` is a unit of time (an instance or a class)."""
    return is_instance_or_subclass(unit, TimeUnit)
