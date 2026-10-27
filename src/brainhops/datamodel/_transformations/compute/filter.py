"""Replace selected parts of a transformation with the identity.

This module implements `Transformation.without` and
`Transformation.with_only`, which a caller uses to drop the parts of a
transformation that it does not want to measure, for example the
displacement fields of a chain whose scale it needs. The design is
recorded in `spec/design/transformation-filters.md`.
"""

import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel import kinds

from ..base import Transformation
from ..concrete import Identity
from ..inverse import Inverse
from ..sequence import Sequence

__all__ = ["MatchLike", "matches", "replace_parts"]

MatchLike: tx.TypeAlias = tx.Union[
    kinds.Kind,
    str,  # a kind name, symbol or registered alias
    tx.Type["Transformation"],
    tx.Callable[["Transformation"], bool],
    # A collection of the values above.
    tx.List[tx.Any],
    tx.Tuple[tx.Any, ...],
    tx.Set[tx.Any],
    tx.FrozenSet[tx.Any],
]
"""Value that selects some parts of a transformation.

A matcher is a kind or the name of a kind, a concrete class of
transformation, a predicate, or a collection of these values. A
collection is a `list`, a `tuple`, a `set` or a `frozenset`, and it
selects a part when any of its items selects it. No other iterable is
unpacked: a transformation, such as a `Sequence`, is refused even
though it is iterable and callable, and so is a generator, because
either would be read in a way that the caller did not mean. A string
must name a kind.
"""

_COLLECTIONS = (list, tuple, set, frozenset)


def matches(xform: "Transformation", match: MatchLike) -> bool:
    """
    Return whether a matcher selects a transformation.

    Parameters
    ----------
    xform : Transformation
        Transformation to test.
    match : MatchLike
        Kind, kind name, concrete class, predicate, or a `list`,
        `tuple`, `set` or `frozenset` of these values.

    Returns
    -------
    bool
        Whether the matcher selects the transformation.

    Raises
    ------
    TypeError
        If `match` is not a matcher, such as a transformation, a
        generator or another iterable.
    """
    if isinstance(match, str):
        return xform.is_kind(match)
    if isinstance(match, type):
        if issubclass(match, Transformation):
            return isinstance(xform, match)
        return xform.is_kind(match)
    if isinstance(match, _COLLECTIONS):
        return any(matches(xform, item) for item in match)
    # A transformation is callable, but calling it applies it.
    if callable(match) and not isinstance(match, Transformation):
        return bool(match(xform))
    raise TypeError(
        f"Cannot select a transformation with {match!r}. A matcher is a "
        f"kind, the name of a kind, a class of transformation, a "
        f"predicate, or a list, tuple, set or frozenset of these values."
    )


def replace_parts(
    xform: "Transformation", match: MatchLike, keep: bool
) -> "Transformation":
    """
    Replace parts of a transformation with the identity.

    When `keep` is false, the parts that the matcher selects are
    replaced, which is what `Transformation.without` does. When `keep`
    is true, the parts that the matcher does not select are replaced,
    which is what `Transformation.with_only` does.

    The transformation is visited from the outside in. A node that the
    matcher decides is kept or replaced as a whole, and the function
    looks inside a `Sequence` or an `Inverse` only when the node itself
    is not decided. A replaced part becomes an `Identity` with the same
    input and output coordinate systems, so that the elements of a
    chain still connect.

    Parameters
    ----------
    xform : Transformation
        Transformation to filter.
    match : MatchLike
        Matcher that selects parts of the transformation.
    keep : bool
        Whether the selected parts are kept rather than replaced.

    Returns
    -------
    Transformation
        The filtered transformation, or `xform` itself when no part is
        replaced.

    Raises
    ------
    ValueError
        If a part to replace changes the number of axes.
    """
    if matches(xform, match):
        # The node is decided as a whole.
        return xform if keep else _as_identity(xform)
    if not isinstance(xform, (Sequence, Inverse)):
        return _as_identity(xform) if keep else xform
    if isinstance(xform, Inverse):
        forward = xform.forward
        if forward is None:
            return _as_identity(xform) if keep else xform
        resolved = replace_parts(forward, match, keep)
        return xform if resolved is forward else resolved.inverse()
    parts = xform.transformations or []
    resolved = [replace_parts(part, match, keep) for part in parts]
    if any(new is not old for new, old in zip(resolved, parts)):
        return replace(xform, transformations=resolved)
    return xform


def _as_identity(xform: "Transformation") -> "Transformation":
    """
    Return the identity between the systems of a transformation.

    Raises
    ------
    ValueError
        If the input and output systems have different numbers of axes.
    """
    source, target = xform.input, xform.output
    ndim_in = None if source is None else source.ndim
    ndim_out = None if target is None else target.ndim
    if ndim_in is not None and ndim_out is not None and ndim_in != ndim_out:
        raise ValueError(
            f"Cannot replace a {type(xform).__name__} that maps "
            f"{ndim_in} axes to {ndim_out} axes with the identity."
        )
    return Identity(input=source, output=target)
