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

Composers carry **no** `priority`: a genuine specificity tie is left to raise
`AmbiguousMethodError`, which propagates out of `compose`. We would rather a
real ambiguity is surfaced than have registration order silently pick a bad
composer. Empirically this raises for no concrete pair -- resolving all 6724
ordered concrete pairs with priority removed yields zero `AmbiguousMethodError`
at call time. The one structural near-tie is the incomparable rule pair
`(Sequence, Transformation)` / `(Transformation, Sequence)`; their only
concrete overlap is a `(Sequence, Sequence)` operand pair, which the strictly
more specific `(Sequence, Sequence)` composer dominates and wins outright.

`bagof` v0.2's *registration-time* pairwise ambiguity check
(`_warn_new_ambiguities`) does not see that dominating third rule, so it emits
a benign `RuntimeWarning` (and lists the pair in `ambiguities()`) for the
incomparable pair at import even though no reachable call is ambiguous. This
is a **false positive**: the check compares two methods in isolation and does
not account for a more specific third method covering their overlap. It does
not break CI (there is no warnings-as-errors filter, and it fires once at
import), so it is left in place rather than papered over with a filter that
could hide a real future ambiguity. **Enhancement note (E5):** the pairwise
registration check should treat a pair as unambiguous when a third registered
method strictly dominates the pair's entire concrete overlap.

**Resolved in bagof-dispatchers 0.3.1** (brainhops now requires it): the
registration-time check is gone, and ambiguity is reported only when a call
hits it, as `AmbiguousMethodError`. `ambiguities()` also runs the full
selection over every registered method, so the dominated `(Sequence,
Transformation)` / `(Transformation, Sequence)` pair is no longer listed and
no import-time warning is emitted. A genuine composer tie still raises at the
call, as intended.

### `simplify` — one function, both arities as overloads

`simplify` is variadic: one transform (a *total* leaf downcast) or two (a
*partial* pair collapse that returns `None` to decline). It dispatches through
a **single** `bagof.dispatchers` function that holds both arities as
overloads: a leaf is `(t, policy)` and a pair is `(first, second, policy)`, so
the two land on different call shapes and `bagof.dispatchers` (v0.2) selects
by argument count -- leaves and pairs never compete. `simplify` calls that one
function with one transform or two accordingly.

* **Leaves** are pure single-winner: the winner matched the bespoke registry
  on every type, zero ambiguities. Clean. `simplify` dispatches a leaf with
  `_simplify(t, policy)`.
* **Pairs** are a chain of responsibility, restored on `bagof.dispatchers`
  v0.3's `Function.candidates()` (enhancement E2 below, now shipped). `simplify`
  walks `_simplify.candidates(first, second, policy)` most-specific-first and
  calls each *partial* rule until one returns non-`None`; all declining is the
  ordinary "these two do not collapse" answer. `candidates()` yields
  specificity ties in registration order, which is the same tie-break the
  bespoke registry used, so no explicit `priority` is needed. We verified this
  empirically: over the 2529 applicable ordered concrete pairs the first
  candidate the chain tries matches the previous single-winner+`priority`
  selection on **every** pair, and the 255 genuine specificity ties
  (`(Identity, Transformation)` versus `(Transformation, Inverse)`, and the
  like — neither is more specific) are exactly the pairs the chain now walks
  instead of picking one arbitrarily.

  Dropping `priority` means bagof's *registration-time* pairwise check
  (`_warn_new_ambiguities`) sees each of those 255 ties as a would-be ambiguous
  single-winner registration and warns. Because `simplify` consumes pairs as an
  all-applicable chain, not a single winner, that warning is not a defect, so it
  is silenced with a `warnings.catch_warnings()` filter scoped to the *pair*
  registrations only — never around leaf registrations, where a real
  single-winner tie must still surface. The scoped silence is removable once
  bagof grows the all-applicable-Function mode (E1), which would let the chain
  register as a combining function rather than a would-be-ambiguous single
  winner.

  **Resolved in bagof-dispatchers 0.3.1:** registration no longer warns
  (ambiguity is reported only at a single-winner call, see E5), and
  `candidates()` still returns tied methods silently, so the scoped filters
  (here and around `is_kind`'s checker registrations) have been removed. A
  real tie between two *leaf* simplifiers now surfaces as
  `AmbiguousMethodError` at the call that hits it.

## What stayed bespoke, and why

### `is_kind` — an all-applicable, combined predicate

> **Update (#101).** `is_kind` has since been rehomed onto `bagof.dispatchers`
> v0.3: `Function.candidates` enumerates every applicable checker and a
> `Type[Super[node]]` lower bound matches the kind contravariantly, while
> brainhops keeps the per-node grouping and the OR reducer (see the `check`
> module docstring). Equivalence with the old `type_distance` dispatch was
> verified at migration time in #101 -- the selected checker set and the
> end-to-end boolean at both `compute` levels, over every concrete instance ×
> kind node -- rather than kept as a regression test, since re-deriving the
> old dispatch over that whole grid is too slow for every CI run. The rest of
> this section records why the first migration (#99) left it bespoke.

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

### E2 — an ordered-candidate view / chain-of-responsibility decline protocol — **shipped, adopted for pairs**

Both `compose` (a composer historically returned `NotImplemented` to hand off
to the next candidate) and the pair simplifiers (a rule returns `None` to
decline) are *chains of responsibility*: try the applicable methods
most-specific-first until one accepts. This was proposed as either

* `Function.candidates(*args) -> tuple[Method, ...]`, yielding the applicable
  methods in specificity order (ties in registration order), letting a caller
  run its own chain; or
* a first-class decline sentinel, so a method returning it is skipped and the
  next most specific is tried, with `NoMethodError` only if all decline.

`bagof.dispatchers` v0.3 shipped the first (with `itercandidates`,
`resolve_candidates`, `bestcandidates` and friends). The pair simplifiers now
use it directly: `simplify` walks `_simplify.candidates(first, second, policy)`
and tries each until one returns non-`None`, which restores the true chain of
responsibility and removed the interim `priority` bookkeeping this branch first
carried. `compose` still runs single-winner (no registered composer declines).
The candidate view is the smaller, more composable primitive and also subsumes
E1 for callers that would rather reduce the candidates themselves.

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

### E5 — registration-time ambiguity check should account for a dominating method — **resolved in 0.3.1**

`Function._warn_new_ambiguities` (RFC 0001 §5) compares a newly registered
method against each existing one *pairwise*. When two methods are incomparable
but a third registered method strictly dominates the whole of their concrete
overlap, no reachable call is actually ambiguous, yet the pairwise check still
warns. `compose` hits this once priority is dropped: `(Sequence,
Transformation)` and `(Transformation, Sequence)` are incomparable, but their
only overlap `(Sequence, Sequence)` is owned by the more specific `(Sequence,
Sequence)` composer, so the import-time `RuntimeWarning` is a false positive.
Both the registration warning and `ambiguities()` should suppress a pair
whose entire concrete overlap region is covered by a strictly more specific
registered method, since no reachable call can then be ambiguous.

bagof-dispatchers 0.3.1 ("report ambiguity at the call, not at
registration") removed the registration-time check altogether: a tie is
reported only when a single-winner call hits it, as `AmbiguousMethodError`,
while `candidates()` / `resolve_candidates()` still return tied methods
silently. `ambiguities()` now runs the full selection over every registered
method, so a pair whose overlap a third method wins is no longer listed.
brainhops requires `bagof-dispatchers >= 0.3.1` and has dropped the scoped
warning filters it carried for this.

## Validation

Full test suite green on CPython 3.11 and on `uv` CPython 3.8.20 (`.[test]`);
`ruff check`, `ruff format --check`, and `codespell` clean. Behaviour is
identical for every registered converter, composer and simplifier; `is_kind`
is unchanged.
