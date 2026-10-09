r"""The hierarchy of transformation kinds, seen as sets of transformations.

Here, `issubclass(A, B)` means that kind `A` is a subset of kind `B`: every
transformation of kind `A` converts losslessly into kind `B`, as linear
transformations do into affine ones. This differs from the usual type
hierarchy, where subclasses are specialized types. The two hierarchies let the
conceptual question "is this transformation linear?" be asked separately from
the type question "is it a `Linear` instance?".

Transformations map ℝⁿ to ℝᵐ, and two transformations can be composed when
the codomain of one matches the domain of the other. Sets that form a group
under composition are marked with the [`group`][] decorator. Sets that are
also Lie groups, that is, smooth manifolds on which composition and inversion
are smooth, are marked with [`liegroup`][].

The Lie groups form the following lattice:

```
G | > 0 | = 1 | ⋉ T
======================
GL ------------- Aff             General Linear     | Affine
 | \             |
 |   GL+ ------- Aff+            Positive Linear    | Positive Affine
 |    | \        |
 |    |  SL ---- SAff            Special Linear     | Special Affine
 |    |   |      |
CO ------------- Sim             Conformal          | Affine Conformal
 | \  |   |      |
 |   CO+ ------- Sim+            Special Conformal  | Special Affine Conformal
 |    |   |      |
 O ------------- E               Orthogonal         | Euclidean
 | \  | /        |
 S   SO -------- SE              Special Orthogonal | Special Euclidean
 | /             |
 I ------------- T               Identity           | Translation
```

The general groups (GL, CO and O) each have two connected components: the
component that contains the identity, and a flipped component. The positive
groups (GL+, CO+ and SO) are the identity components of these general groups.
Adding translations (⋉ T) extends each linear group into an affine group.

!!! info
    SE contains the rigid-body transformations, which preserve angles and
    volumes. Sim+ contains the similitudes, which only preserve oriented
    angles.

!!! warning
    Python preserves the declared order of base classes in the method
    resolution order (MRO), whereas set inclusion imposes no order between
    unrelated supersets. The two kinds of constraint can therefore conflict,
    and building the MRO can fail even when the inclusions are acyclic.
    Reordering the bases, or explicitly listing an ancestor that is already
    inherited indirectly, resolves such a failure without changing the
    inclusions.
"""

# ======================================================================
#   LIST OF ALL SETS
# ======================================================================
#
# +-------------------------------------+--------+---------+-----------+
# | Name                                | Symbol | Def.    | Property  |
# +-------------------------------------+--------+---------+-----------+
# |                              GENERAL                               |
# +-------------------------------------+--------+---------+-----------+
# | TransformationKind                  |        |         |           |
# | Transformation                      | Trans  |         |           |
# | Injection                           |        |         |           |
# | Surjection                          |        |         |           |
# | Bijection                           |        |         |           |
# | Diffeomorphism                      | Diff   |         |det Df ≠ 0 |
# | VolumePreservingDiffeomorphism      |        |         |det Df =±1 |
# | OrientationPreservingDiffeomorphism |        |         |det Df > 0 |
# | ConformalDiffeomorphism             |        |         |           |
# | SpecialDiffeomorphism               | SDiff  |         |det Df = 1 |
# +-------------------------------------+--------+---------+-----------+
# |                              MATRIX                                |
# +-------------------------------------+--------+---------+-----------+
# | Matrix                              |        |         |           |
# | InvertibleMatrix                    |        |         | det ≠ 0   |
# | OrientationPreservingMatrix         |        |         | det > 0   |
# | VolumePreservingMatrix              |        |         | det =±1   |
# +-------------------------------------+--------+---------+-----------+
# |                              AFFINE                                |
# +-------------------------------------+--------+---------+-----------+
# | Affine                              |        |         | ⊄ Gp      |
# | InvertibleAffine                    | Aff    |         | det ≠ 0   |
# | PositiveAffine                      | Aff+   |         | det > 0   |
# | VolumePreservingAffine              |        |         | det =±1   |
# | SpecialAffine                       | SAff   | SL  ⋉ T | det = 1   |
# | ConformalEuclidean                  | Sim    | CO  ⋉ T | det ≠ 0   |
# | SpecialConformalEuclidean           | Sim+   | CO+ ⋉ T | det > 0   |
# | Euclidean                           | E      | O   ⋉ T | det =±1   |
# | SpecialEuclidean                    | SE     | SO  ⋉ T | det = 1   |
# | Dilation                            |        | ℝ*  ⋉ T | det ≠ 0   |
# | PositiveDilation                    |        | ℝ+  ⋉ T | det > 0   |
# | Translation                         | T      |         | det = 1   |
# +-------------------------------------+--------+---------+-----------+
# |                              LINEAR                                |
# +-------------------------------------+--------+---------+-----------+
# | Linear                              | M      |         | ⊄ Gp      |
# | InvertibleLinear                    | GL     |         | det ≠ 0   |
# | PositiveLinear                      | GL+    |         | det > 0   |
# | SpecialLinear                       | SL     |         | det = 1   |
# | ConformalOrthogonal                 | CO     | O  x ℝ+ | det ≠ 0   |
# | SpecialConformalOrthogonal          | CO+    | SO x ℝ+ | det > 0   |
# | Orthogonal                          | O      |         | det =±1   |
# | SpecialOrthogonal                   | SO     |         | det = 1   |
# | GeneralizedPermutation              |        | Δ ⋊ S   | det ≠ 0   |
# | SignedPermutation                   | B      | C₂ⁿ ⋊ S | det ≠ 0   |
# | Permutation                         | S      |         | det ≠ 0   |
# | Diagonal                            | D      |         | ⊄ Gp      |
# | InvertibleDiagonal                  | Δ      |         | det ≠ 0   |
# | PositiveDiagonal                    | Δ_+    |         | det > 0   |
# | SpecialDiagonal                     | Δ_S    | Δ ∩ SL  | det = 1   |
# | OrthogonalDiagonal                  | Δ_O    | Δ ∩ O   | det =±1   |
# | SpecialOrthogonalDiagonal           | Δ_SO   | Δ ∩ SO  | det = 1   |
# | Multiplicative                      | ℝ      |         | ⊄ Gp      |
# | InvertibleMultiplicative            | ℝ*     |         | det ≠ 0   |
# | PositiveMultiplicative              | ℝ+     |         | det > 0   |
# +-------------------------------------+--------+---------+-----------+
# |                             IDENTITY                               |
# +-------------------------------------+--------+---------+-----------+
# | Identity                            | I      |         | det = 1   |
# +-------------------------------------+--------+---------+-----------+
#
# ======================================================================

__all__ = []

import re
from abc import ABC
from collections.abc import Sequence as AbcSequence
from functools import lru_cache

import typing_extensions as tx
from bagof.magic import Magic, replace

from brainhops._core.compat import PLACEHOLDER, partial

# --- API helpers ------------------------------------------------------


def public(obj: tx.Any) -> tx.Any:
    __all__.append(obj.__name__)
    return obj


# ======================================================================
#
#                           R E G I S T R I E S
#
# ======================================================================

_Type: tx.TypeAlias = tx.Type["TransformationKind"]
_Decorator: tx.TypeAlias = tx.Callable[["_Type"], "_Type"]
_Set: tx.TypeAlias = tx.Set[_Type]
_StrMap: tx.TypeAlias = tx.Dict[str, _Type]
_TypeMap: tx.TypeAlias = tx.Dict[_Type, _Type]

GROUPS: _Set = set()
LIE_GROUPS: _Set = set()
CONNECTED: _Set = set()
SIMPLYCONNECTED: _Set = set()
CLOSEDUNDER: tx.Dict[_Type, _Set] = {}
NONEMBEDDABLE: _Set = set()

NAMETOCLASS: _StrMap = {}
FSYMBOLTOCLASS: _StrMap = {}
SYMBOLTOCLASS: _StrMap = {}
INVERTIBLE_OF: _TypeMap = {}
NONINVERTIBLE_OF: _TypeMap = {}


# --- checks -----------------------------------------------------------


@public
def is_group(cls: _Type) -> bool:
    """Return whether a set is registered as a group."""
    return cls in GROUPS


@public
def is_lie_group(cls: _Type) -> bool:
    """Return whether a set is registered as a Lie group."""
    return cls in LIE_GROUPS


@public
def is_connected(cls: _Type) -> bool:
    """Return whether a set is registered as connected."""
    return cls in CONNECTED


@public
def is_simplyconnected(cls: _Type) -> bool:
    """Return whether a set is registered as simply connected."""
    return cls in SIMPLYCONNECTED


@public
def is_invertible(cls: _Type) -> bool:
    """Return whether every transformation in a set is invertible."""
    return issubclass(cls, Bijection)


@public
@lru_cache(maxsize=None)  # noqa: UP033
def is_closedunder(cls: _Type, subcls: _Type) -> bool:
    """Return whether `cls` is closed under composition with `subcls`.

    This holds when `cls` is a group containing `subcls`, when the pair was
    registered with [`closedunder`][], or when `cls` is closed under a base of
    `subcls`.
    """
    if is_group(cls) and issubclass(subcls, cls):
        return True
    if subcls in CLOSEDUNDER.get(cls, set()):
        return True
    for supcls in subcls.__bases__:
        if is_closedunder(cls, supcls):
            return True
    return False


@public
def is_closed(cls: _Type) -> bool:
    """Return whether a set is closed under composition with itself."""
    return is_closedunder(cls, cls)


@public
def is_transformation_set(cls: tx.Any) -> bool:
    """Return whether an object is one of the transformation sets of this
    hierarchy.

    The result is true for the classes defined in this module and false for
    concrete transformation types. A concrete transformation type is only a
    virtual subclass of [`TransformationKind`][], and virtual registration
    leaves its MRO untouched, so the test can tell the two apart.
    """
    if not isinstance(cls, type):
        return False
    return TransformationKind in cls.__mro__


@public
def is_embeddable(cls: _Type) -> bool:
    """Return whether a set embeds into higher dimensions by identity padding.

    Padding with the identity turns a transformation T into the
    block-diagonal transformation T ⊕ I, which acts as T on the original axes
    and leaves the new axis unchanged. The set is embeddable when the padded
    transformation still belongs to the set in one more dimension. For
    example:

    * Every element of SO(3), once padded, is an element of SO(4).
    * Most elements of ℝ*(3), which scale by the same factor along every
      axis, are not elements of ℝ*(4) once padded. For instance,
      diag(2, 2, 2, 1) is not isotropic in ℝ⁴.
    """
    # Embeddability is registered rather than inferred, because SO(3) has
    # both embeddable and non-embeddable subgroups and supergroups.
    return cls not in NONEMBEDDABLE


# --- generators -------------------------------------------------------


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_sets() -> tx.FrozenSet[type]:
    """Return all registered sets of transformations."""
    return frozenset(NAMETOCLASS.values())


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_groups() -> tx.FrozenSet[type]:
    """Return all registered groups."""
    return frozenset(GROUPS)


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_lie_groups() -> tx.FrozenSet[type]:
    """Return all registered Lie groups."""
    return frozenset(LIE_GROUPS)


@public
def as_invertible(cls: _Type) -> _Type:
    """Return the invertible subset of a set.

    Raises
    ------
    TypeError
        If the set is not invertible and has no known invertible subset.
    """
    node = INVERTIBLE_OF.get(cls, cls)
    if not is_invertible(node):
        raise TypeError(
            f"{cls} is not a known invertible set of transformations"
        )
    return node


@public
def as_unrestricted(cls: _Type) -> _Type:
    """Return the superset that also contains non-invertible maps.

    The set itself is returned when no such superset is registered.
    """
    return NONINVERTIBLE_OF.get(cls, cls)


# --- decorators -------------------------------------------------------


def group(cls: _Type) -> _Type:
    """Mark a set as a
    [group](https://en.wikipedia.org/wiki/Group_(mathematics)) under
    composition.
    """
    GROUPS.add(cls)
    return cls


def liegroup(cls: _Type) -> _Type:
    """Mark a set as a [Lie group](https://en.wikipedia.org/wiki/Lie_group),
    and therefore a group.
    """
    LIE_GROUPS.add(cls)
    return group(cls)


def connected(cls: _Type) -> _Type:
    """Mark a set as
    [connected](https://en.wikipedia.org/wiki/Connected_space).
    """
    CONNECTED.add(cls)
    return cls


def simplyconnected(cls: _Type) -> _Type:
    """Mark a set as [simply
    connected](https://en.wikipedia.org/wiki/Simply_connected_space), and
    therefore connected.
    """
    SIMPLYCONNECTED.add(cls)
    return connected(cls)


def invertible_subset_of(cls: _Type) -> _Decorator:
    """Mark the decorated set as the
    [invertible](https://en.wikipedia.org/wiki/Inverse_element) subset of
    `cls`.
    """

    def decorator(subcls: _Type) -> _Type:
        INVERTIBLE_OF[cls] = subcls
        NONINVERTIBLE_OF[subcls] = cls
        return subcls

    return decorator


def closedunder(cls: _Type) -> _Decorator:
    """Declare `cls`
    [closed](https://en.wikipedia.org/wiki/Closure_(mathematics)) under
    composition with the decorated set.
    """

    def decorator(subcls: _Type) -> _Type:
        CLOSEDUNDER.setdefault(cls, set()).add(subcls)
        return subcls

    return decorator


def closed(cls: _Type) -> _Type:
    """Mark a set as
    [closed](https://en.wikipedia.org/wiki/Closure_(mathematics)) under
    composition with itself.
    """
    return closedunder(cls)(cls)


def nonembeddable(cls: _Type) -> _Type:
    """Mark a set as not embeddable (see [`is_embeddable`][])."""
    NONEMBEDDABLE.add(cls)
    return cls


def alias(cls: _Type, *names: str) -> _Type:
    """Register additional names for a set.

    The function is called either as `alias(cls, "Name", ...)` or as the
    decorator `@alias("Name", ...)`. The names are added to the `ALIAS`
    attribute of the class, to the name registry, to the module namespace and
    to `__all__`.
    """
    if isinstance(cls, str):
        return partial(alias, PLACEHOLDER, cls, *names)

    if cls.__name__ not in __all__:
        __all__.append(cls.__name__)

    if "ALIAS" not in cls.__dict__:
        cls.ALIAS = []  # ALIAS must belong to this class, not be inherited.
    if isinstance(cls.ALIAS, str):
        cls.ALIAS = [cls.ALIAS]
    if isinstance(cls.ALIAS, tuple):
        cls.ALIAS = list(cls.ALIAS)
    cls.ALIAS.extend(names)
    for name in names:
        NAMETOCLASS[name.lower()] = cls
        globals()[name] = cls
        __all__.append(name)
    return cls


# ======================================================================
#
#                           F A M I L I E S
#
# ======================================================================

Kind: tx.TypeAlias = tx.Type["TransformationKind"]
Dim: tx.TypeAlias = tx.Optional[int]
KindLike: tx.TypeAlias = tx.Union[Kind, str]
FamilyTuple: tx.TypeAlias = tx.Tuple[KindLike, Dim, Dim]
FamilyTupleLike: tx.TypeAlias = tx.Union[
    tx.Tuple[KindLike], tx.Tuple[KindLike, Dim], tx.Tuple[KindLike, Dim, Dim]
]
FamilyLike: tx.TypeAlias = tx.Union[tx.Self, FamilyTupleLike, KindLike, int]
MISSING = object()


@public
class TransformationFamily(
    AbcSequence, Magic, eq=True, hash=True, frozen=True
):
    """A kind of transformation with optional input and output dimensions.

    For example, the family `SO(3)` pairs the kind [`SpecialOrthogonal`][]
    with the input and output dimension 3. A family behaves as an immutable
    tuple `(kind, ndim, odim)`.
    """

    # --- attributes ---------------------------------------------------

    kind: Kind
    """The kind of every transformation in the family."""

    ndim: Dim = MISSING
    """The input dimension, or `None` when it is unknown."""

    odim: Dim = MISSING
    """The output dimension, which defaults to the input dimension."""

    def __post_init__(self) -> None:
        if self.ndim is MISSING:
            object.__setattr__(self, "ndim", None)
        if self.odim is MISSING:
            object.__setattr__(self, "odim", self.ndim)

    @property
    def name(self) -> str:
        """The preferred name, which is the class name of the kind."""
        return self.kind.__name__

    @property
    def names(self) -> tx.Tuple[str, ...]:
        """All names of the kind, starting with its class name."""
        name = self.kind.__name__
        alias = self.kind.__dict__.get("ALIAS", ())
        if isinstance(alias, str):
            alias = (alias,)
        return (name, *alias)

    @property
    def symbol(self) -> tx.Optional[str]:
        """The short symbol of the kind, or `None`."""
        return self.kind.__dict__.get("SYMBOL")

    @property
    def fsymbol(self) -> tx.Optional[str]:
        """The symbol with the dimensions filled in, or `None`.

        Unknown dimensions keep the placeholders `{n}` and `{m}`.
        """
        if (fsymbol := self.kind.__dict__.get("FSYMBOL")) is None:
            return None
        ndim = "{n}" if self.ndim is None else self.ndim
        odim = "{m}" if self.odim is None else self.odim
        fsymbol = fsymbol.format(n=ndim, m=odim)
        return fsymbol

    # --- magic --------------------------------------------------------
    # The family implements the sequence protocol, so that it unpacks into
    # three values.

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> tx.Union[Kind, Dim]:
        if index == 0:
            return self.kind
        if index == 1:
            return self.ndim
        if index == 2:
            return self.odim
        raise IndexError(
            f"index {index} out of range for TransformationFamily"
        )

    def __iter__(self) -> tx.Iterator[tx.Union[Kind, Dim]]:
        yield self.kind
        yield self.ndim
        yield self.odim

    # --- to -----------------------------------------------------------

    def to_tuple(self) -> tx.Tuple[Kind, Dim, Dim]:
        """Convert the family into a `(kind, ndim, odim)` tuple."""
        return (self.kind, self.ndim, self.odim)

    # --- from ---------------------------------------------------------

    @classmethod
    def parse(
        cls, repr: FamilyLike, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Build a family from a family, tuple, dimension, type, name or
        symbol.

        Explicit dimensions override those of `repr`, and an explicit `None`
        makes a dimension unknown. An integer gives a generic
        [`Transformation`][] of that dimension. A string is first looked up as
        a name and then, if no kind has that name, as a symbol.

        Raises
        ------
        ValueError
            If `repr` describes no kind or family.
        """
        # --- already a family ---
        if isinstance(repr, TransformationFamily):
            updates = {}
            if ndim is not MISSING and repr.ndim != ndim:
                updates["ndim"] = ndim
            if odim is not MISSING and repr.odim != odim:
                updates["odim"] = odim
            if updates:
                repr = replace(repr, **updates)
            return repr

        # --- a (kind, ndim) pair ---
        if isinstance(repr, tuple):
            if not is_family_tuple(repr):
                raise ValueError(
                    f"Invalid tuple: {repr!r}. A family pairs one kind "
                    f"with zero, one or two dimensions."
                )
            return cls.from_tuple(repr, ndim, odim)

        # --- a bare dimension ---
        # A bool is an int in Python, but it is not a dimension.
        if isinstance(repr, int) and not isinstance(repr, bool):
            if ndim is MISSING:
                ndim = repr
            return cls(Transformation, ndim, odim)

        # --- a kind ---
        if isinstance(repr, type):
            return cls(repr, ndim, odim)

        # --- a name or a symbol ---
        if isinstance(repr, str):
            try:
                return cls.from_name(repr, ndim, odim)
            except ValueError:
                return cls.from_symbol(repr, ndim, odim)

        # --- error ---
        raise ValueError(f"Invalid transformation kind or family: {repr!r}")

    @classmethod
    def from_tuple(
        cls, values: FamilyTupleLike, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Build a family from a tuple of one to three elements.

        Explicit `ndim` and `odim` arguments override the tuple elements.
        """
        values = list(values)
        if len(values) < 3:
            values += [MISSING] * (3 - len(values))
        if ndim is not MISSING:
            values[1] = ndim
        if odim is not MISSING:
            values[2] = odim
        return cls.parse(*values)

    @classmethod
    def from_name(
        cls, name: str, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Build a family from a case-insensitive name.

        Raises
        ------
        ValueError
            If the name is unknown.
        """
        kind = NAMETOCLASS.get(name.lower())
        if kind is None:
            raise ValueError(f"Invalid name for transformation kind: {name}")
        return cls.parse(kind, ndim, odim)

    @classmethod
    def from_symbol(
        cls, symbol: str, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Build a family from a symbol, such as `"SO"` or `"SO(3)"`.

        Raises
        ------
        ValueError
            If the symbol is unknown.
        """
        kind = SYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim, odim)

        kind = FSYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim, odim)

        for fsymbol, kind in FSYMBOLTOCLASS.items():
            if "{n}" not in fsymbol:
                continue
            pattern = _fsymbol_to_pattern(fsymbol)
            if not (match := pattern.match(symbol)):
                continue
            if ndim is MISSING:
                ndim = match.groupdict().get("n", ndim)
                if isinstance(ndim, str):
                    ndim = int(ndim)
            if odim is MISSING:
                odim = match.groupdict().get("m", odim)
                if isinstance(odim, str):
                    odim = int(odim)
            return cls.parse(kind, ndim, odim)

        raise ValueError(
            f"Invalid symbol for transformation kind or family: {symbol}"
        )


def is_family_tuple(value: tuple) -> bool:
    """Return whether a tuple holds a kind and up to two dimensions."""
    if len(value) not in (1, 2, 3):
        return False
    kind = value[0]
    if isinstance(kind, type) and not issubclass(kind, TransformationKind):
        return False
    if not isinstance(kind, (str, type)):
        return False
    if len(value) > 1:
        ndim = value[1]
        if ndim is not None and (
            isinstance(ndim, bool) or not isinstance(ndim, int)
        ):
            return False
    if len(value) > 2:
        ndim = value[2]
        if ndim is not None and (
            isinstance(ndim, bool) or not isinstance(ndim, int)
        ):
            return False
    return True


def _fsymbol_to_pattern(fsymbol: str) -> re.Pattern:
    """Turn a symbol template into a regex with the groups `n` and `m`.

    A template such as `"SO({n})"` may repeat a placeholder, and every
    occurrence of the placeholder must then match the same number.
    """
    pattern = re.escape(fsymbol)
    for name in ("n", "m"):
        slot = re.escape(f"{{{name}}}")
        pattern = pattern.replace(slot, rf"(?P<{name}>\d+)", 1)
        pattern = pattern.replace(slot, rf"(?P={name})")
    return re.compile("^" + pattern + "$")


# ======================================================================
#
#                           H I E R A R C H Y
#
# ======================================================================


@public
class TransformationKind(ABC):
    """The root of the hierarchy of transformation kinds.

    A subclass may declare a short mathematical symbol in `SYMBOL`, such as
    `"SO"`, the same symbol with dimension placeholders in `FSYMBOL`, such as
    `"SO({n})"`, and plain-text names in `ALIAS`. Names can also be added with
    the [`alias`][] decorator. The class name, the symbols and the aliases are
    registered when the subclass is created, so that
    [`TransformationFamily.parse`][] can find the kind from any of them.
    """

    SYMBOL: str
    FSYMBOL: str
    ALIAS: tx.Union[str, tx.List[str], tx.Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        NAMETOCLASS[cls.__name__.lower()] = cls
        if alias := cls.__dict__.get("ALIAS"):
            if isinstance(alias, str):
                alias = (alias,)
            for name in alias:
                NAMETOCLASS[name.lower()] = cls
        if fsymb := cls.__dict__.get("FSYMBOL"):
            FSYMBOLTOCLASS[fsymb] = cls
        if symb := cls.__dict__.get("SYMBOL"):
            SYMBOLTOCLASS[symb] = cls

    def __new__(cls, subcls: type) -> type:
        """Register a concrete type as a virtual subclass of the set.

        Calling a kind on a class, as in `Linear(MyType)`, registers the class
        and returns it, rather than creating an instance of the kind.
        """
        return cls.register(subcls)

    @classmethod
    def parse(cls, repr: KindLike) -> tx.Type[tx.Self]:
        """Return the kind given by a name, a symbol or a type."""
        return TransformationFamily.parse(repr).kind


@closed  # Closure assumes matching domains and codomains.
@alias("Morphism", "Map")
class Transformation(TransformationKind):
    """Any coordinate transformation, also called `Morphism` or `Map`."""

    SYMBOL = "Map"
    FSYMBOL = "Map(ℝ^{n},ℝ^{m})"


@closed
@alias("VolumePreservingTransformation", "VolumePreservingMap")
class VolumePreserving(Transformation):
    """A transformation that preserves volumes locally (|det Df| = 1)."""


@closed
@alias("OrientationPreservingTransformation", "OrientationPreservingMap")
class OrientationPreserving(Transformation):
    """A transformation that preserves orientation (det Df > 0)."""


@closed
@nonembeddable
@alias("Conformal", "ConformalMap", "ConformalTransformation")
@alias("AnglePreservingTransformation", "AnglePreservingMap")
class AnglePreserving(Transformation):
    """A transformation that [preserves
    angles](https://en.wikipedia.org/wiki/Conformal_map) ((Df)ᵀ Df = λ(x) I).
    """


@closed  # Closure assumes matching domains.
@alias("Injective", "InjectiveMap", "InjectiveTransformation")
class Injection(Transformation):
    """A [one-to-one](https://en.wikipedia.org/wiki/Injective_function)
    transformation.
    """


@closed  # Closure assumes matching codomains.
@alias("Surjective", "SurjectiveMap", "SurjectiveTransformation")
class Surjection(Transformation):
    """An [onto](https://en.wikipedia.org/wiki/Surjective_function)
    transformation, which reaches every point of its codomain.
    """


@group
@invertible_subset_of(Transformation)
@alias("Bijective", "BijectiveTransformation", "BijectiveMap")
@alias("Invertible", "InvertibleTransformation", "InvertibleMap")
class Bijection(Injection, Surjection):
    """A [bijection](https://en.wikipedia.org/wiki/Bijection), which is an
    invertible transformation that is both injective and surjective.
    """


@group
@public
class Homeomorphism(Bijection):
    """A [homeomorphism](https://en.wikipedia.org/wiki/Homeomorphism), which is
    a continuous bijection with a continuous inverse.
    """


@group
@public
class Diffeomorphism(Homeomorphism):
    """A [diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism), which
    is a smooth homeomorphism with a smooth inverse.
    """

    SYMBOL = "Diff"
    FSYMBOL = "Diff(ℝ^{n})"


@group
@public
class VolumePreservingDiffeomorphism(VolumePreserving, Diffeomorphism):
    """A diffeomorphism that preserves volumes."""


@group
@public
class OrientationPreservingDiffeomorphism(
    OrientationPreserving, Diffeomorphism
):
    """A diffeomorphism that preserves orientation."""

    SYMBOL = "Diff+"
    FSYMBOL = "Diff+(ℝ^{n})"


@group
@nonembeddable
@alias("AnglePreservingDiffeomorphism")
class ConformalDiffeomorphism(AnglePreserving, Diffeomorphism):
    """A diffeomorphism that preserves angles."""


@group
@alias("VolumeAndOrientationPreservingDiffeomorphism")
class SpecialDiffeomorphism(
    VolumePreservingDiffeomorphism,
    OrientationPreservingDiffeomorphism,
):
    """A diffeomorphism with det Df(x) = 1."""

    SYMBOL = "SDiff"
    FSYMBOL = "SDiff(ℝ^{n})"


# ----------------------------------------------------------------------
#   L I E  /  M A T R I X
# ----------------------------------------------------------------------


@closed
@alias("MatrixTransformation")
class Matrix(Transformation):
    """A transformation given by a
    [matrix](https://en.wikipedia.org/wiki/Transformation_matrix) (on
    homogeneous coordinates).
    """


@group
@invertible_subset_of(Matrix)
@alias("InvertibleMatrixTransformation")
class InvertibleMatrix(Matrix, Diffeomorphism):
    """An [invertible matrix](https://en.wikipedia.org/wiki/Invertible_matrix)
    transformation.

    The matrix kinds join the smooth kinds of the lattice at this class,
    because every invertible affine transformation is a diffeomorphism.
    """


@group
@connected
@public
class OrientationPreservingMatrix(
    OrientationPreservingDiffeomorphism, InvertibleMatrix
):
    """An invertible matrix with a positive determinant."""


@group
@public
class VolumePreservingMatrix(VolumePreservingDiffeomorphism, InvertibleMatrix):
    """An invertible matrix with a determinant of ±1."""


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------


@closed
@alias("AffineTransformation", "AffineMap", "AffineMatrix")
class Affine(Matrix):
    """An [affine
    transformation](https://en.wikipedia.org/wiki/Affine_transformation),
    possibly not invertible.
    """


@liegroup
@invertible_subset_of(Affine)
@alias("InvertibleAffineMatrix", "AffineGroup")
@alias("InvertibleAffineTransformation", "InvertibleAffineMap")
class InvertibleAffine(Affine, InvertibleMatrix):
    """An invertible affine transformation, forming the [affine
    group](https://en.wikipedia.org/wiki/Affine_group) Aff = GL ⋉ T.
    """

    SYMBOL = "Aff"
    FSYMBOL = "Aff(ℝ^{n})"


@liegroup
@connected
@alias("PositiveAffineMatrix", "PositiveAffineGroup")
@alias("PositiveAffineTransformation", "PositiveAffineMap")
class PositiveAffine(InvertibleAffine, OrientationPreservingMatrix):
    """An [affine map](https://en.wikipedia.org/wiki/Affine_group) with a
    positive determinant (Aff+ = GL+ ⋉ T).
    """

    SYMBOL = "Aff+"
    FSYMBOL = "Aff+(ℝ^{n})"


@liegroup
@alias(
    "VolumePreservingAffineMap",
    "VolumePreservingAffineTransformation",
)
class VolumePreservingAffine(InvertibleAffine, VolumePreservingMatrix):
    """An affine transformation with a determinant of ±1."""


@liegroup
@connected
@alias(
    "SpecialAffineMap",
    "SpecialAffineGroup",
    "SpecialAffineTransformation",
)
class SpecialAffine(
    PositiveAffine, VolumePreservingAffine, SpecialDiffeomorphism
):
    """An affine transformation with a determinant of 1, forming the [special
    affine
    group](https://en.wikipedia.org/wiki/Affine_group#Special_affine_group)
    SAff = SL ⋉ T.
    """

    SYMBOL = "SAff"
    FSYMBOL = "SAff({n})"


@liegroup
@nonembeddable
@alias(
    "ConformalEuclideanGroup",
    "ConformalEuclideanMap",
    "ConformalEuclideanTransformation",
    "Similarity",
    "Similitude",
    "AffineSimilarity",
    "AffineSimilitude",
)
class ConformalEuclidean(InvertibleAffine, ConformalDiffeomorphism):
    """An affine transformation that preserves angles, forming the
    [similarity](https://en.wikipedia.org/wiki/Similarity_(geometry)#In_Euclidean_space)
    group Sim = CO ⋉ T.
    """

    SYMBOL = "Sim"
    FSYMBOL = "Sim({n})"


@liegroup
@connected
@nonembeddable
@alias(
    "SpecialConformalEuclideanMap",
    "SpecialConformalEuclideanGroup",
    "SpecialConformalEuclideanTransformation",
    "ConformalSpecialEuclidean",
    "ConformalSpecialEuclideanMap",
    "ConformalSpecialEuclideanGroup",
    "ConformalSpecialEuclideanTransformation",
    "DirectSimilarity",
    "DirectSimilitude",
    "DirectAffineSimilarity",
    "DirectAffineSimilitude",
)
class SpecialConformalEuclidean(ConformalEuclidean, PositiveAffine):
    """A direct
    [similarity](https://en.wikipedia.org/wiki/Similarity_(geometry)#In_Euclidean_space),
    which preserves orientation (Sim+ = CO+ ⋉ T).
    """

    SYMBOL = "Sim+"
    FSYMBOL = "Sim+({n})"


@liegroup
@alias("EuclideanGroup", "EuclideanMap", "EuclideanTransformation")
class Euclidean(ConformalEuclidean, VolumePreservingAffine):
    """A [Euclidean](https://en.wikipedia.org/wiki/Euclidean_group)
    transformation (E = O ⋉ T).
    """

    SYMBOL = "E"
    FSYMBOL = "E({n})"


@liegroup
@connected
@alias(
    "SpecialEuclideanGroup",
    "SpecialEuclideanMap",
    "SpecialEuclideanTransformation",
    "Rigid",
    "RigidTransformation",
)
class SpecialEuclidean(SpecialConformalEuclidean, Euclidean, SpecialAffine):
    """A [rigid
    transformation](https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries),
    with a determinant of 1 (SE = SO ⋉ T).
    """  # noqa: E501

    SYMBOL = "SE"
    FSYMBOL = "SE({n})"


@liegroup
@nonembeddable
@alias(
    "DilationGroup",
    "HomothetyTranslation",
    "HomothecyTranslation",
)
class Dilation(ConformalEuclidean):
    """A [dilation](https://en.wikipedia.org/wiki/Dilation_(metric_space)),
    that is, a [homothety](https://en.wikipedia.org/wiki/Homothety) combined
    with a translation (ℝ* ⋉ T).
    """

    SYMBOL = "ℝ* ⋉ T"
    FSYMBOL = "ℝ* ⋉ T({n})"


@liegroup
@simplyconnected
@nonembeddable
@public
class PositiveDilation(Dilation, SpecialConformalEuclidean):
    """A [dilation](https://en.wikipedia.org/wiki/Dilation_(metric_space)) with
    a positive scale factor (ℝ+ ⋉ T).
    """

    SYMBOL = "ℝ+ ⋉ T"
    FSYMBOL = "ℝ+ ⋉ T({n})"


@liegroup
@simplyconnected
@public
class Translation(PositiveDilation, SpecialEuclidean):
    """A
    [translation](https://en.wikipedia.org/wiki/Translation_(geometry)#As_a_group).
    """

    SYMBOL = "T"
    FSYMBOL = "T({n})"


# ----------------------------------------------------------------------
#   L I N E A R
# ----------------------------------------------------------------------


@closed
@alias("LinearTransformation", "LinearMap")
class Linear(Affine):
    """A linear transformation, possibly not invertible."""


@liegroup
@invertible_subset_of(Linear)
@alias(
    "InvertibleGroup",
    "InvertibleLinearMap",
    "InvertibleLinearTransformation",
)
class InvertibleLinear(Linear, InvertibleAffine):
    """An invertible linear transformation, forming the [general linear
    group](https://en.wikipedia.org/wiki/General_linear_group) GL.
    """

    SYMBOL = "GL"
    FSYMBOL = "GL({n})"


@liegroup
@connected
@alias("PositiveLinearTransformation", "PositiveLinearMap")
class PositiveLinear(InvertibleLinear, PositiveAffine):
    """A linear transformation with a positive determinant (GL+)."""

    SYMBOL = "GL+"
    FSYMBOL = "GL+({n})"


@liegroup
@connected
@alias("SpecialLinearTransformation", "SpecialLinearMap")
class SpecialLinear(PositiveLinear, SpecialAffine):
    """A linear transformation with a determinant of 1, forming the [special
    linear group](https://en.wikipedia.org/wiki/Special_linear_group) SL.

    SL lies inside SAff = SL ⋉ T, and it is through SAff that SL lies inside
    the volume-preserving diffeomorphisms.
    """

    SYMBOL = "SL"
    FSYMBOL = "SL({n})"


@liegroup
@nonembeddable
@alias(
    "ConformalOrthogonalGroup",
    "ConformalOrthogonalMap",
    "ConformalOrthogonalTransformation",
)
class ConformalOrthogonal(InvertibleLinear, ConformalEuclidean):
    """A linear transformation that preserves angles, forming the [conformal
    group](https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group) CO =
    O × ℝ+.
    """

    SYMBOL = "CO"
    FSYMBOL = "CO({n})"


@liegroup
@connected
@nonembeddable
@alias(
    "SpecialConformalOrthogonalGroup",
    "SpecialConformalOrthogonalMap",
    "SpecialConformalOrthogonalTransformation",
    "ConformalSpecialOrthogonal",
    "ConformalSpecialOrthogonalMap",
    "ConformalSpecialOrthogonalTransformation",
)
class SpecialConformalOrthogonal(
    ConformalOrthogonal, PositiveLinear, SpecialConformalEuclidean
):
    """A linear transformation that preserves angles and orientation, forming
    the orientation-preserving part CO+ of the [conformal
    group](https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group).

    !!! note
        A scaled rotation cR has the determinant cⁿ, which is positive but
        differs from 1 unless c = 1. CO+ therefore lies under GL+ but not
        under SL, and its elements do not preserve volumes in general.
    """

    SYMBOL = "CO+"
    FSYMBOL = "CO+({n})"


@liegroup
@alias("OrthogonalTransformation", "OrthogonalMap", "OrthogonalGroup")
class Orthogonal(ConformalOrthogonal, Euclidean):
    """An [orthogonal](https://en.wikipedia.org/wiki/Orthogonal_group) matrix
    (A Aᵀ = I), forming the orthogonal group O.
    """

    SYMBOL = "O"
    FSYMBOL = "O({n})"


@liegroup
@connected
@alias(
    "SpecialOrthogonalGroup",
    "SpecialOrthogonalMap",
    "SpecialOrthogonalTransformation",
    "Rotation",
    "RotationTransformation",
)
class SpecialOrthogonal(
    Orthogonal,
    SpecialConformalOrthogonal,
    SpecialEuclidean,
    SpecialLinear,
):
    """An orthogonal matrix with a determinant of 1, that is, a rotation,
    forming the [special orthogonal
    group](https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group)
    SO.

    !!! note
        SL is listed as a direct base because no ancestor provides that
        inclusion. A rotation has a determinant of 1, but none of the other
        linear groups that contain the rotations, such as O and CO+, is a
        subgroup of SL.
    """  # noqa: E501

    SYMBOL = "SO"
    FSYMBOL = "SO({n})"


@liegroup
@alias(
    "GeneralizedPermutationMatrix",
    "Monomial",
    "MonomialGroup",
    "MonomialMatrix",
    "MonomialTransformation",
)
class GeneralizedPermutation(InvertibleLinear):
    """A [generalized
    permutation](https://en.wikipedia.org/wiki/Generalized_permutation_matrix)
    (monomial) matrix (S ⋉ Δ).
    """

    SYMBOL = "S ⋉ Δ"
    FSYMBOL = "S_{n} ⋉ Δ({n})"


@liegroup
@alias(
    "SignedPermutationMatrix", "SignedPermutationGroup", "HyperoctahedralGroup"
)
class SignedPermutation(GeneralizedPermutation, Orthogonal):
    """A generalized permutation with entries of ±1, forming the [signed
    permutation
    group](https://en.wikipedia.org/wiki/Generalized_permutation_matrix#Signed_permutation_group)
    B = C₂ⁿ ⋊ S.
    """  # noqa: E501

    SYMBOL = "B"
    FSYMBOL = "B_{n}"


@liegroup
@alias("PermutationMatrix", "PermutationGroup", "SymmetricGroup")
class Permutation(SignedPermutation, Orthogonal):
    """A [permutation](https://en.wikipedia.org/wiki/Permutation_group) matrix,
    forming the symmetric group S.
    """

    SYMBOL = "S"
    FSYMBOL = "S_{n}"


@liegroup
@alias("AlternatingGroup", "EvenPermutationGroup")
class EvenPermutation(Permutation, SpecialOrthogonal):
    """An even permutation, forming the [alternating
    group](https://en.wikipedia.org/wiki/Alternating_group) A.

    Even permutations are exactly the permutation matrices in SO(n), so a
    rotation reindexed by an even permutation stays a rotation.
    """

    SYMBOL = "A"
    FSYMBOL = "A_{n}"


@public
class OddPermutation(Permutation):
    r"""An odd [permutation](https://en.wikipedia.org/wiki/Permutation_group),
    with a determinant of -1.

    !!! note
        The odd permutations form the coset S \ A of the [alternating
        group](https://en.wikipedia.org/wiki/Alternating_group) A. This coset
        is not a group, because composing two odd permutations gives an even
        one.
    """

    SYMBOL = "S \\ A"
    FSYMBOL = "S_{n} \\ A_{n}"


@closed
@alias(
    "DiagonalMatrix",
    "DiagonalTransformation",
    "Scaling",
    "ScalingTransformation",
)
class Diagonal(Linear):
    """A [diagonal matrix](https://en.wikipedia.org/wiki/Diagonal_matrix),
    which represents a
    [scaling](https://en.wikipedia.org/wiki/Scaling_(geometry)) that is
    possibly not invertible.
    """


@liegroup
@invertible_subset_of(Diagonal)
@alias(
    "InvertibleDiagonalMatrix",
    "InvertibleDiagonalTransformation",
)
class InvertibleDiagonal(Diagonal, GeneralizedPermutation):
    """An invertible diagonal matrix (Δ), with 2ⁿ connected components."""

    SYMBOL = "Δ"
    FSYMBOL = "Δ({n})"


@liegroup
@simplyconnected
@alias(
    "PositiveDiagonalMatrix",
    "PositiveDiagonalTransformation",
    "PositiveScaling",
    "PositiveScalingTransformation",
)
class PositiveDiagonal(InvertibleDiagonal, PositiveLinear):
    """A diagonal matrix with positive entries (Δ+)."""

    SYMBOL = "Δ+"
    FSYMBOL = "Δ+({n})"


@liegroup
@public
class SpecialDiagonal(InvertibleDiagonal, SpecialLinear):
    """A diagonal matrix with a determinant of 1 (Δ_S)."""

    SYMBOL = "Δ_S"
    FSYMBOL = "Δ_S({n})"


@liegroup
@alias(
    "OrthogonalDiagonalMatrix",
    "OrthogonalDiagonalTransformation",
    "CardinalReflection",
    "CardinalReflectionGroup",
)
class OrthogonalDiagonal(InvertibleDiagonal, SignedPermutation):
    """A diagonal matrix with entries of ±1 (Δ_O)."""

    SYMBOL = "Δ_O"
    FSYMBOL = "Δ_O({n})"


@liegroup
@alias(
    "SpecialOrthogonalDiagonalMatrix",
    "SpecialOrthogonalDiagonalTransformation",
)
class SpecialOrthogonalDiagonal(
    OrthogonalDiagonal,
    SpecialDiagonal,
    SpecialOrthogonal,
):
    """A diagonal matrix with entries of ±1 and a determinant of 1 (Δ_SO)."""

    SYMBOL = "Δ_SO"
    FSYMBOL = "Δ_SO({n})"


@closed
@nonembeddable
@alias(
    "MultiplicativeTransformation",
    "Multiplication",
    "ScaledIdentityMatrix",
)
class Multiplicative(Diagonal):
    """A scaling by the same factor along every axis (ℝ)."""

    SYMBOL = "ℝ"
    FSYMBOL = "ℝ({n})"


@liegroup
@nonembeddable
@invertible_subset_of(Multiplicative)
@alias(
    "MultiplicativeGroup",
    "InvertibleMultiplicativeTransformation",
    "InvertibleMultiplication",
    "Homothety",
    "HomothetyTransformation",
    "Homothecy",
    "HomothecyTransformation",
    "HomogeneousDilation",
)
class InvertibleMultiplicative(
    Multiplicative, InvertibleDiagonal, ConformalOrthogonal
):
    """An isotropic scaling by a non-zero factor, or
    [homothety](https://en.wikipedia.org/wiki/Homothety), forming the
    [multiplicative group](https://en.wikipedia.org/wiki/Multiplicative_group)
    ℝ*.

    Since cI equals |c| times the orthogonal matrix sign(c)I, an isotropic
    scaling preserves angles, and ℝ* therefore lies in CO.
    """

    SYMBOL = "ℝ*"
    FSYMBOL = "ℝ*({n})"


@liegroup
@simplyconnected
@nonembeddable
@alias(
    "PositiveMultiplicativeGroup",
    "PositiveMultiplicativeTransformation",
    "PositiveMultiplication",
    "PositiveHomothety",
    "PositiveHomothetyTransformation",
)
class PositiveMultiplicative(
    InvertibleMultiplicative, PositiveDiagonal, SpecialConformalOrthogonal
):
    """An isotropic [scaling](https://en.wikipedia.org/wiki/Scaling_(geometry))
    by a positive factor (ℝ+), a subgroup of ℝ*.
    """

    SYMBOL = "ℝ+"
    FSYMBOL = "ℝ+({n})"


# ----------------------------------------------------------------------
#   I D E N T I T Y
# ----------------------------------------------------------------------


@liegroup
@simplyconnected
@alias(
    "IdentityMatrix",
    "IdentityMap",
    "IdentityTransformation",
)
class Identity(
    Translation,
    EvenPermutation,
    SpecialOrthogonalDiagonal,
    PositiveMultiplicative,
    SpecialOrthogonal,
):
    """The [identity](https://en.wikipedia.org/wiki/Identity_function)
    transformation.
    """

    SYMBOL = "I"
    FSYMBOL = "I({n})"
