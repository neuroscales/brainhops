"""The compose mode: which kinds of transformation `compute()` fuses.

A *mode* is a list of [`TransformationFamily`][] entries -- a `(kind, ndim)`
pair each. `compute(mode)` composes a run of adjacent transforms only when
every transform in the run is admitted by one of those entries.

The mode says *what gets composed*. It says nothing about how hard a leaf is
looked at, which is the simplify policy's job (see `simplify`). The two are
deliberately independent: `compute(mode=False, simplify="numeric")` reads
values to retype leaves and composes nothing, and `compute(mode=True,
simplify=False)` composes everything without retyping anything.
"""

__all__ = [
    # compute mode
    "ModeLike",
    "mode_admits",
    "mode_children",
    "normalize_modes",
    "normalize_family",
    # kind & family
    "Family",
    "FamilyLike",
    "Kind",
    "KindLike",
    "is_family",
    "is_kind",
    "normalize_family",
]

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel.kinds import Transformation as TransformationSet
from brainhops.datamodel.kinds import TransformationFamily

# The membership predicates, and the single key-normalizing routine, all
# live in `check`, next to the checker registry they dispatch on. They are
# re-exported here because mode resolution is their main caller.
from .compute.check import (
    FamilyLike,
    Kind,
    KindLike,
    is_family,
    is_kind,
    normalize_family,
)

# typing
if tx.TYPE_CHECKING:
    # internals
    from .base import Transformation


Family: tx.TypeAlias = TransformationFamily
"""
A [`TransformationFamily`][] is a [`Kind`][] and, optionally, a
dimensionality.
"""

ModeLike: tx.TypeAlias = tx.Union[
    None, bool, FamilyLike, tx.Iterable[FamilyLike]
]
"""Possible input to the `mode` argument of [`compute()`][]."""


def normalize_modes(mode: ModeLike) -> tx.List[Family]:
    """Convert any mode-like input into a list of [`Family`][] entries.

    | Input                | Meaning                       |
    |----------------------|-------------------------------|
    | `True`               | compose every kind            |
    | `None`               | compose every kind (no restriction given) |
    | `False`              | compose nothing (simplify-only)           |
    | `[]`                 | compose nothing (simplify-only)           |
    | a key, or a list of keys | compose only those kinds  |

    `None` reads as "no restriction was given", not as "nothing": it is the
    absent-argument spelling of `True`. `False` and the empty list are the
    two spellings of "compose nothing", which leaves `compute()` doing only
    what `simplify()` does.
    """
    if mode is True or mode is None:
        return [TransformationFamily(TransformationSet, None)]
    if mode is False:
        return []  # compose nothing (simplify-only)
    if _is_family_like(mode):
        mode = [mode]
    return list(map(normalize_family, mode))


def mode_children(mode: Family) -> tx.List[Family]:
    """The families one step down the hierarchy from `mode`.

    A class kind (a representation rather than a set) has no children in
    the hierarchy, so it yields none.
    """
    children = []
    seen = set()
    subclasses = getattr(mode.kind, "__subclasses__", lambda: [])
    for child in subclasses():
        family = TransformationFamily(child, mode.ndim)
        if family not in seen:
            seen.add(family)
            children.append(family)
    return children


def mode_admits(t: "Transformation", modes: tx.Iterable[Family]) -> bool:
    """Whether any family in `modes` admits a transform."""
    return any(is_family(t, m) for m in modes)


def _is_kind_like(kind: tx.Any) -> bool:
    if isinstance(kind, str):
        return True
    # One kind is one node of the hierarchy -- and a concrete transform is
    # registered into it, so it is one too.
    return isinstance(kind, type) and issubclass(kind, TransformationSet)


def _is_family_like(mode: tx.Any) -> bool:
    if isinstance(mode, TransformationFamily):
        return True
    if _is_kind_like(mode):
        return True
    if isinstance(mode, int) and not isinstance(mode, bool):
        return True
    if not isinstance(mode, tuple):
        return False
    if len(mode) != 2:
        return False
    kind, ndim = mode
    if not _is_kind_like(kind):
        return False
    if isinstance(ndim, bool) or not isinstance(ndim, (int, type(None))):
        return False
    return True
