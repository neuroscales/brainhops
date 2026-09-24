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
    "ModeLike",
    "mode_admits",
    "mode_children",
    "normalize_modes",
    "normalize_family",
]

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel.hierarchy import Transformation as TransformationSet
from brainhops.datamodel.hierarchy import TransformationFamily

# The single membership predicate, and the single key-lowering routine,
# both live in `check`, next to the checker registry they dispatch on. They
# are re-exported here because mode resolution is their main caller.
from .check import FamilyLike, Kind, KindLike, is_kind, lower_family

# typing
if tx.TYPE_CHECKING:
    # internals
    from .base import Transformation


Family = TransformationFamily
"""
A [`TransformationFamily`][] is a [`Kind`][] and, optionally, a
dimensionality.
"""

ModeLike = tx.Union[None, bool, FamilyLike, tx.Iterable[FamilyLike]]
"""Possible input to the `mode` argument of [`compute()`][]."""

normalize_family = lower_family
"""Normalize any mode/simplify key into a [`TransformationFamily`][]."""

__all__ += ["Family", "FamilyLike", "Kind", "KindLike"]


# ======================================================================
#
#                            M O D E S
#
# ======================================================================


def _is_kind_like(kind: tx.Any) -> bool:
    if isinstance(kind, str):
        return True
    if isinstance(kind, type) and issubclass(kind, TransformationSet):
        return True
    if isinstance(kind, tuple) and all(isinstance(k, type) for k in kind):
        return True
    return False


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
    kind, ndim = mode
    subclasses = getattr(kind, "__subclasses__", lambda: [])
    for child in subclasses():
        if (child, ndim) not in seen:
            seen.add((child, ndim))
            children.append(TransformationFamily(child, ndim))
    return children


def matches_mode(t: "Transformation", mode: Family) -> bool:
    """Whether a single family admits a transform."""
    kind, ndim = mode
    # Resolution always runs at analytic (`compute=False`): it never
    # reads a value to decide which mode admits a leaf or which simplify
    # entry applies.
    if not is_kind(t, kind, compute=False):
        return False
    if ndim is None:
        return True
    # FIXME
    #   In many transforms, the ndim can be guessed from the content of the
    #   xform, even if the input/output spaces are not set (e.g. the shape of
    #   the matrix or the field). The current implementation is a stricter
    #   bound.
    if t.input is None or t.output is None:
        return False
    if t.input.axes is None or t.output.axes is None:
        return False
    return len(t.input.axes) == ndim and len(t.output.axes) == ndim


def mode_admits(t: "Transformation", modes: tx.Iterable[Family]) -> bool:
    """Whether any family in `modes` admits a transform."""
    return any(matches_mode(t, m) for m in modes)
