# Migrating the transformation registries to `bagof.dispatchers`

This memo records how the `datamodel._transformations` dispatch registries
were moved onto [`bagof.dispatchers`][] (v0.2), which ones moved cleanly,
which one stayed bespoke, and the general enhancements `bagof.dispatchers`
would need to absorb the rest of the family. It closes issue #96.

## Background

Before this change, five dispatch tables lived in
`_transformations/registries.py`, all built on one bespoke `Dispatcher`
base class and a hand-rolled `type_distance()` metric (plus per-pair
`*_FASTMAP` caches):

| registry     | question it answers                          | shape                     |
|--------------|----------------------------------------------|---------------------------|
| `CONVERTERS` | `convert(x, cls)` — turn `x` into a `cls`     | single winner             |
| `COMPOSERS`  | `compose(x1, x2)` — fuse two transforms       | ordered candidates, decline |
| `SIMPLIFIERS`| `simplify(t)` / `simplify(a, b)` — cheapen    | single winner / ordered candidates |
| `is_kind`    | `is_kind(t, kind)` — membership predicate     | all-applicable, combined  |

`bagof.dispatchers` provides *hint-driven, most-specific-single-winner*
multiple dispatch: every applicable overload is found, the single most
specific one is run, an unbreakable tie raises `AmbiguousMethodError`, and
nothing applying raises `NoMethodError`. The migration question is which of
the tables above fit that model.

We answered it empirically, not by eye: for every registry we built the
equivalent `bagof.dispatchers` function from the *actual* registered rules
and compared its winner against the bespoke registry's for every ordered pair
of the 87 concrete `Transformation` subclasses in the package (≈6.7k pairs
each). The results below cite those comparisons.

## What was ported

### `convert` — single winner, dispatching on a value-typed argument

`convert(x, cls)` is the interesting one: it dispatches on `type(x)` **and**
on `cls`, a *type passed as a value*, choosing the converter whose declared
output is the nearest supertype of the requested `cls`. That is expressed
with a [`type`][]`[Out]` parameter — `bagof.dispatchers` reads a `type[...]`
hint as *value-dependent* and scores it covariantly against the class value
handed in, which reproduces the bespoke covariant-target scoring exactly.
The winner matched the bespoke registry on **all 6724 pairs**, with zero
ambiguities and zero missing methods.

The converter implementations take only `(x, **kwargs)` — `cls` is a dispatch
key they never read — so each registration wraps the implementation in a thin
shim `(x, cls, **kwargs)` that drops `cls`. The catch-all
`Transformation -> Transformation` converter matches every request, so the
result is still checked against `cls` (a request for a type nothing produces
would otherwise be answered by the catch-all with the wrong type).

The `Convert(Dispatcher)` subclass and its bespoke machinery were retired.

### `compose` — most-specific wins

`compose` is a two-argument dispatch on the operand types (which may be
unions such as `Union[Linear, Scaling, Permutation]` — `bagof.dispatchers`
reads a union hint natively). The library's position-by-position specificity
matched the bespoke `type_distance` winner on **6667 of 6724 pairs**, with
zero ambiguities.

The 57 differences were all instances of one bug in the bespoke metric: a
class can be *farther* from an ancestor than from a more distant one when the
nearer ancestor sits behind a longer chain of mixins or a generic
subscription. Concretely, `type_distance(OmeZarrField, Sequence) ≈ 4.17`
while `type_distance(OmeZarrField, Transformation) ≈ 3.67`, even though
`Sequence` *is* a `Transformation` — so the bespoke registry picked the
`(Sequence, Transformation)` composer over the strictly more specific
`(Sequence, Sequence)` one. `bagof.dispatchers` is monotonic here (a subtype
is never less specific than its supertype), so it picks the `(Sequence,
Sequence)` composer; both build a sequence and `.compute()` it, and the two
normalise to the same transform. The full test suite passes on both
interpreters with the library's (correct) choice.

A specificity tie is broken by a strictly decreasing `priority` per
registration, reproducing the bespoke registration-order tie-break.

### `simplify` — one function per arity

`simplify` is variadic: one transform (a *total* leaf downcast) or two (a
*partial* pair collapse that returns `None` to decline). It now dispatches
through two `bagof.dispatchers` functions, one per arity, and `simplify`
picks between them by how many transforms it was given.

* **Leaves** are pure single-winner: the winner matched the bespoke registry
  on every type, zero ambiguities. Clean.
* **Pairs** matched the bespoke winner on every pair too, but 255 pairs are a
  *specificity tie* the library reports as `AmbiguousMethodError`
  (`(Identity, Transformation)` versus `(Transformation, Inverse)`, and the
  like — neither is more specific). The bespoke registry broke those ties by
  registration order, so each pair registration carries a decreasing
  `priority` that does the same. The `None`-decline contract is preserved: the
  chosen simplifier returning `None` *is* the decline.

## What stayed bespoke, and why

### `is_kind` — an all-applicable, combined predicate

`is_kind(t, kind, compute=False)` is **not** single-winner, so it was left on
the bespoke `Dispatcher` unchanged. Three things put it outside the library's
model:

1. **It combines every applicable checker, it does not pick one.** Two routes
   into a kind are routinely incomparable — an `Affine` holding a permutation
   is established in the orthogonal set through the permutation node, and one
   holding a rotation through the special-orthogonal node — so *all*
   applicable checkers are asked and their booleans are **OR-ed**. A
   most-specific-single-winner would ask only one and silently drop the
   others, changing the answer.
2. **The kind is matched contravariantly.** A checker registered against node
   `N` answers a query about kind `K` when `N ⊆ K` (establishing membership in
   a subset proves membership in the superset). That is the *reverse* of the
   usual covariant argument direction, and the "subset" relation is the kinds
   lattice, not Python's class hierarchy.
3. **It carries a `compute` level.** `compute=False` (analytic, structure
   only) versus `compute=True` (numeric, reads values) steers each checker's
   answer but must **not** participate in specificity.

The source type is *also* ranked by MRO position (so an `Inverse` wrapper
shadows the concrete base it inherits from), which is single-winner-like, but
only *within* one node's group; across nodes the answers are combined. The
net shape is a group-wise most-specific selection feeding an OR reducer — a
strict superset of what the library offers.

## Proposed enhancements to `bagof.dispatchers`

These are framed for the whole `bagof` family, not for brainhops.

### E1 — an all-applicable / combining dispatch mode with a reducer

A dispatched function should be constructible in a mode where a call gathers
**every** applicable method (optionally one per "group", chosen most-specific
within the group) and combines their results through a caller-supplied
reducer, rather than selecting a single winner. `any`/`all`/`list`/`min`
reducers cover the common cases; `is_kind` is `any` over a group-wise
most-specific selection. This is the single enhancement that would let
membership predicates, validators and multi-handler fan-outs live on
`bagof.dispatchers` at all. It is orthogonal to the sub-hint machinery and
would reuse the existing applicability and specificity passes.

### E2 — an ordered-candidate view / chain-of-responsibility decline protocol

Both `compose` (a composer historically returned `NotImplemented` to hand off
to the next candidate) and the pair simplifiers (a rule returns `None` to
decline) are *chains of responsibility*: try the applicable methods
most-specific-first until one accepts. brainhops no longer needs the fall-
through — no registered composer declines, and the pair rules' declines never
differ from a lower candidate's, so single-winner + `priority` reproduces the
old behaviour — but the pattern is general. The library could expose either

* `Function.candidates(*args) -> Iterator[Method]`, yielding the applicable
  methods in specificity order (ties in registration order), letting a caller
  run its own chain; or
* a first-class decline sentinel, so a method returning it is skipped and the
  next most specific is tried, with `NoMethodError` only if all decline.

The first is the smaller, more composable primitive and also subsumes E1 for
callers that would rather reduce the candidates themselves.

### E3 — per-parameter variance / a pluggable relation

`is_kind` needs a parameter matched *contravariantly* against a domain-
specific "subset" relation (the kinds lattice), not the covariant class
relation. `bagof.dispatchers.core` already isolates the subtype relation;
exposing a per-parameter variance flag, or letting a parameter name a custom
relation from `core`, would make contravariant and lattice-based dispatch
expressible without abandoning the library.

### E4 — a "carried, not dispatched" (context) parameter

`is_kind`'s `compute` flag, and `convert`'s target `cls` before we made it a
real `type[...]` key, are arguments that must be *passed through* to the
implementation (or used to steer it) **without** taking part in specificity.
Today a neutral hint (every overload annotating the parameter identically)
achieves this by accident; a declared "context parameter" would make the
intent explicit and spare implementations the shim `convert` needs to accept
and drop a dispatch-only argument. This is a small ergonomic addition, not a
change to the dispatch model.

## Validation

Full test suite green on CPython 3.11 and on `uv` CPython 3.8.20 (`.[test]`);
`ruff check`, `ruff format --check`, and `codespell` clean. Behaviour is
identical for every registered converter, composer and simplifier; `is_kind`
is unchanged.
