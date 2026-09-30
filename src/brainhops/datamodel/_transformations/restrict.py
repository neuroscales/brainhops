"""Restrict a transformation to a block of its axes, and embed one back.

Two dispatched operations that move a transformation between a space and
a block of its axes. They are what the factor pass (the `factor` module)
uses to cut a chain into independent axis groups, and they are the one
place that knows, type by type, how to do so.

`restrict(t, rows, cols, ni, no)` is the piece of `t` that maps the input
axes `cols` to the output axes `rows`, as a transformation over those axes
only. `t` maps `ni` axes to `no` axes, and `rows` / `cols` are sorted
positions among them. The block must be *decoupled* from the rest of `t`:
no output in `rows` reads an input outside `cols`, and no output outside
`rows` reads an input in `cols`. The factor pass guarantees this, since
its axis groups are connected components of the chain's dependency graph.
The result is `None` when the piece is the identity (it maps `cols` to
`rows` in order and changes nothing), which lets the caller drop it.

`embed(t, in_axes, out_axes, ni, no)` goes the other way. It is the
transformation from `ni` to `no` axes that applies `t` from the input
axes `in_axes` to the output axes `out_axes`, and passes every other axis
through in order. This is the local-space form of
`SubspaceTransformation(t, in_axes, out_axes)` over that many axes.

Contract
--------
* **The cheaper type is kept.** A restricted `Scaling` is a `Scaling`,
  and a restricted `Translation` or `Permutation` keeps its type too, not
  a general affine sub-block.
* **A lazy inverse stays lazy.** An `Inverse` is restricted by
  restricting its forward over the swapped block and inverting that
  lazily. Its parameters are never read.
* **Identity is preserved.** When the block covers everything a
  transformation acts on, `restrict` returns that very object (for a
  subspace, its inner), so a `Sub(warp) . Sub(warp^-1)` pair still cancels
  by identity afterwards.
* **An embedding is a plain affine when it can be.** An affine-ish `t`
  embeds as an `Affine`. An endpoint-less subspace cannot be widened to an
  affine by conversion, since its axis count would have to be read from
  systems it does not carry, so the axis counts are passed here instead.
  A lazy inverse, or a transform with no affine reading (a field), stays
  wrapped in a `SubspaceTransformation`.

Extending
---------
Both operations are [`bagof.dispatchers`][] functions keyed on the type of
`t`, like `compose` and `simplify`. The most specific rule wins. The rules
for the built-in types live in `restrictors`. A new transformation class
plugs in by registering its own rules:

    @restrictor
    def _(t: MyTransform, rows, cols, ni, no) -> tx.Optional[Transformation]:
        ...

    @embedder
    def _(t: MyTransform, in_axes, out_axes, ni, no) -> Transformation:
        ...

A class with no rule of its own falls back to the rule for
`Transformation`, which reads it as an affine. A typed inverse such as
`InverseScaling` is also an instance of the family it inverts; it reaches
the `Inverse` rule because `Inverse` comes first among its bases.

Why not composition
-------------------
A restriction could also be written as `project . t . embed`, using the
`Projection` and `SubspaceTransformation` types, and reduced by
`compute()`. That would need a composer for every pair of a `Projection`
and another type (there are none). It would also lose the cheaper type,
since the subspace/affine composers produce a dense `Affine`. It would
read a lazy inverse's parameters, and it would return a new object where
object identity must be kept. A dispatched operation has none of these
problems.
"""

# dependencies
import typing_extensions as tx
from bagof.dispatchers import Function, NoMethodError

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation

__all__ = ["restrict", "restrictor", "embed", "embedder"]


_restrict: Function = Function("restrict")
"""The dispatched function every registered restriction rule joins."""

_embed: Function = Function("embed")
"""The dispatched function every registered embedding rule joins."""


def restrictor(func: tx.Callable) -> tx.Callable:
    """Register a restriction rule, keyed on the hint of its first parameter.

    A rule has the signature `(t, rows, cols, ni, no)` and returns the
    restricted transformation, or `None` for an identity piece (see the
    module docstring).
    """
    _restrict.register(func)
    return func


def embedder(func: tx.Callable) -> tx.Callable:
    """Register an embedding rule, keyed on the hint of its first parameter.

    A rule has the signature `(t, in_axes, out_axes, ni, no)` and returns
    the embedded transformation (see the module docstring).
    """
    _embed.register(func)
    return func


def restrict(
    t: "Transformation",
    rows: tx.List[int],
    cols: tx.List[int],
    ni: int,
    no: int,
) -> tx.Optional["Transformation"]:
    """Restrict a transformation to a decoupled block of its axes.

    Parameters
    ----------
    t : Transformation
        The transformation to restrict, from `ni` to `no` axes.
    rows : list[int]
        Sorted positions of the block among the `no` output axes.
    cols : list[int]
        Sorted positions of the block among the `ni` input axes.
    ni, no : int
        The number of input and output axes of `t`.

    Returns
    -------
    Transformation or None
        The transformation from the `cols` axes to the `rows` axes, of the
        cheapest type the rules know. `t` itself when the block covers all
        of it. `None` when the piece is the identity.

    Raises
    ------
    TypeError
        When no rule applies, as for an object that is not a
        transformation.
    """
    try:
        return _restrict(t, rows, cols, ni, no)
    except NoMethodError:
        raise TypeError(f"Cannot restrict a {type(t).__name__}") from None


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
    t : Transformation or None
        The transformation to embed, from `len(in_axes)` to
        `len(out_axes)` axes. `None` is the identity, which embeds as a
        reindex of `in_axes` onto `out_axes`.
    in_axes : list[int]
        The positions `t` reads among the `ni` input axes.
    out_axes : list[int]
        The positions `t` writes among the `no` output axes.
    ni, no : int
        The number of input and output axes of the wider space.

    Returns
    -------
    Transformation
        An `Affine` from `ni` to `no` axes when `t` has an affine reading,
        else a `SubspaceTransformation` that wraps `t`.

    Raises
    ------
    TypeError
        When no rule applies, as for an object that is not a
        transformation.
    """
    try:
        return _embed(t, in_axes, out_axes, ni, no)
    except NoMethodError:
        raise TypeError(f"Cannot embed a {type(t).__name__}") from None
