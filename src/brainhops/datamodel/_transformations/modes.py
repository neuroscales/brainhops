"""Compose modes, which decide what kinds of transformation are composed.

A compose mode is a list of [`Family`][] entries, each pairing a kind of
transformation with an optional number of dimensions. When a sequence is
computed, a run of adjacent transformations is fused only if every
transformation in the run is admitted by some entry of the mode.

The mode is independent of the simplify policy, which decides how closely each
leaf is inspected. For example, `compute(mode=False, simplify="numeric")` reads
parameter values to downcast the leaves but composes nothing, whereas
`compute(mode=True, simplify=False)` composes everything and downcasts nothing.
"""

__all__ = [
    "ModeLike",
    "mode_admits",
    "mode_children",
    "normalize_modes",
    "normalize_family",
    "Family",
    "FamilyLike",
    "Kind",
    "KindLike",
    "is_family",
    "is_kind",
    "normalize_family",
]

import typing_extensions as tx

from brainhops.datamodel.kinds import Transformation as TransformationSet
from brainhops.datamodel.kinds import TransformationFamily

# The membership predicates and the routine that normalizes keys live in
# `check`, beside the checker registry they dispatch on. They are re-exported
# here because mode resolution is their main user.
from .compute.check import (
    FamilyLike,
    Kind,
    KindLike,
    is_family,
    is_kind,
    normalize_family,
)

if tx.TYPE_CHECKING:
    from .base import Transformation


Family: tx.TypeAlias = TransformationFamily
"""A kind of transformation together with an optional number of dimensions.

This is an alias of [`TransformationFamily`][].
"""

ModeLike: tx.TypeAlias = tx.Union[
    None, bool, FamilyLike, tx.Iterable[FamilyLike]
]
"""Values accepted as a compose mode, the `mode` argument of `compute()`."""


def normalize_modes(mode: ModeLike) -> tx.List[Family]:
    """Convert a mode-like value to a list of families.

    | Input                | Meaning                       |
    |----------------------|-------------------------------|
    | `True`               | compose every kind            |
    | `None`               | compose every kind (no restriction given) |
    | `False`              | compose nothing (simplify-only)           |
    | `[]`                 | compose nothing (simplify-only)           |
    | a key, or a list of keys | compose only those kinds  |

    `None` is the spelling of an absent argument, so it means no restriction,
    like `True`. A single family-like value is wrapped in a list first.

    Parameters
    ----------
    mode : ModeLike
        The value to normalize.

    Returns
    -------
    list of Family
        Families that the mode admits.
    """
    if mode is True or mode is None:
        return [TransformationFamily(TransformationSet, None)]
    if mode is False:
        return []
    if _is_family_like(mode):
        mode = [mode]
    return list(map(normalize_family, mode))


def mode_children(mode: Family) -> tx.List[Family]:
    """Return the families one level below `mode` in the kind hierarchy.

    The children are the direct subclasses of the kind, with the number of
    dimensions of `mode` and without duplicates.
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
    """Return whether some family in `modes` admits the transformation `t`."""
    return any(is_family(t, m) for m in modes)


def _is_kind_like(kind: tx.Any) -> bool:
    if isinstance(kind, str):
        return True
    # A concrete transformation class is registered into the kind hierarchy, so
    # it counts as kind-like as well.
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
