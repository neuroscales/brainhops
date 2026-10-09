"""Restriction of a transformation to a block of its axes, and embedding back.

The factor pass cuts a chain into independent axis groups with two
dispatched operations, and this module is the single place that knows, type
by type, how to perform them. `restrict(t, rows, cols, ni, no)` returns the
piece of `t` that maps the input axes `cols` to the output axes `rows`.
The block must be decoupled from the other axes, which the factor pass
guarantees. `embed(t, in_axes, out_axes, ni, no)` is the reverse: the
local-space form of `SubspaceTransformation(t, in_axes, out_axes)`, in which
the other axes pass through in order.

Both operations keep the cheaper type (a restricted Scaling stays a
Scaling), keep a lazy inverse lazy by restricting its forward
transformation, and return `t` itself when the block covers everything it
acts on, so that `Sub(warp) . Sub(warp^-1)` still cancels by identity. An
embedding is a plain Affine when `t` has an affine reading. Writing these
operations as compositions with a Projection would lose all of these
properties, and the required composers do not exist.

Rules are [`bagof.dispatchers`][] functions keyed on the type of `t`, and
the built-in ones live in the `restrictors` module. A new class registers
its rules as follows:

```python
@restrictor
def _(t: MyTransform, rows, cols, ni, no) -> tx.Optional[Transformation]:
    ...

@embedder
def _(t: MyTransform, in_axes, out_axes, ni, no) -> Transformation:
    ...
```

A class without its own rule falls back to the Transformation rule, which
reads it as an affine. A typed inverse such as InverseScaling reaches the
Inverse rule, because Inverse comes first among its bases.
"""

__all__ = ["restrict", "restrictor", "embed", "embedder"]

import typing_extensions as tx
from bagof.dispatchers import Function, NoMethodError

from brainhops.errors import RestrictionError

from .utils import axis_counts

if tx.TYPE_CHECKING:
    from ..base import Transformation


_restrict: Function = Function("restrict")
"""Dispatched function that all restriction rules join."""

_embed: Function = Function("embed")
"""Dispatched function that all embedding rules join."""


def restrictor(func: tx.Callable) -> tx.Callable:
    """Register a restriction rule, keyed on the hint of its first parameter.

    A rule takes `(t, rows, cols, ni, no)` and returns the restricted
    transformation, or None for an identity piece.
    """
    _restrict.register(func)
    return func


def embedder(func: tx.Callable) -> tx.Callable:
    """Register an embedding rule, keyed on the hint of its first parameter.

    A rule takes `(t, in_axes, out_axes, ni, no)` and returns the embedded
    transformation.
    """
    _embed.register(func)
    return func


def restrict(
    t: "Transformation",
    rows: tx.List[int],
    cols: tx.List[int],
    ni: tx.Optional[int] = None,
    no: tx.Optional[int] = None,
) -> tx.Optional["Transformation"]:
    """Restrict a transformation to a decoupled block of its axes.

    Parameters
    ----------
    t
        Transformation from `ni` to `no` axes.
    rows
        Sorted positions of the block among the `no` output axes.
    cols
        Sorted positions of the block among the `ni` input axes.
    ni, no
        Axis counts, inferred from `t` when omitted and checked against `t`
        when given.

    Returns
    -------
    Transformation or None
        Transformation from the `cols` axes to the `rows` axes, `t` itself if
        the block covers all of `t`, or None if the piece is the identity.

    Raises
    ------
    RestrictionError
        If no rule applies, if the block cannot be soundly cut, or if a count
        is missing or contradicts the one stated by `t`.
    """
    ni, no = _resolve_counts(t, ni, no)
    try:
        return _restrict(t, rows, cols, ni, no)
    except NoMethodError:
        raise RestrictionError(
            f"Cannot restrict a {type(t).__name__}"
        ) from None


def embed(
    t: tx.Optional["Transformation"],
    in_axes: tx.List[int],
    out_axes: tx.List[int],
    ni: int,
    no: int,
) -> "Transformation":
    """Embed a transformation over some axes into a wider space.

    Parameters
    ----------
    t
        Transformation from `len(in_axes)` to `len(out_axes)` axes. None stands
        for the identity and embeds as a reindex of `in_axes` onto `out_axes`.
    in_axes, out_axes
        Positions that `t` reads among the `ni` axes and writes among the `no`
        axes.
    ni, no
        Axis counts of the wider space, which `t` cannot state.

    Returns
    -------
    Transformation
        An Affine from `ni` to `no` axes if `t` has an affine reading, and
        otherwise a SubspaceTransformation that wraps `t`.

    Raises
    ------
    TypeError
        If no rule applies.
    RestrictionError
        If the counts stated by `t` contradict `in_axes` or `out_axes`.
    """
    if t is not None:
        _resolve_counts(t, len(in_axes), len(out_axes))
    try:
        return _embed(t, in_axes, out_axes, ni, no)
    except NoMethodError:
        raise TypeError(f"Cannot embed a {type(t).__name__}") from None


def _resolve_counts(
    t: tx.Any, ni: tx.Optional[int], no: tx.Optional[int]
) -> tx.Tuple[int, int]:
    # Given counts are checked against those stated by `t`; omitted ones are
    # taken from `t`.
    stated = axis_counts(t)
    resolved = []
    for given, known, side in zip((ni, no), stated, ("input", "output")):
        name = type(t).__name__
        if given is None:
            if known is None:
                raise RestrictionError(
                    f"Cannot infer the number of {side} axes of a {name}, "
                    f"which states none: pass it explicitly"
                )
            given = known
        elif known is not None and int(given) != known:
            raise RestrictionError(
                f"{given} {side} axes were given for a {name} that has {known}"
            )
        resolved.append(int(given))
    return resolved[0], resolved[1]
