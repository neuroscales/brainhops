# Migrating the transformation registries to bagof.dispatchers

The migration described here is implemented (issues #96, #99 and #101).
brainhops requires `bagof-dispatchers >= 0.3.1`, and no bespoke
dispatcher remains.

This memo records how the dispatch registries of
`datamodel._transformations` were moved onto [`bagof.dispatchers`][], why
each registry ended up with the shape it has, and which enhancements to
`bagof.dispatchers` the migration motivated.

## Background

The registries used to share a bespoke `Dispatcher` base class, a
hand-written `type_distance()` metric and per-pair caches. Four questions
were answered by dispatch:

| registry      | question it answers                         | shape                              |
|---------------|---------------------------------------------|------------------------------------|
| `CONVERTERS`  | `convert(x, cls)`: turn `x` into a `cls`    | single winner                      |
| `COMPOSERS`   | `compose(x1, x2)`: fuse two transforms      | ordered candidates, decline        |
| `SIMPLIFIERS` | `simplify(t)` / `simplify(a, b)`: cheapen   | single winner / ordered candidates |
| `is_kind`     | `is_kind(t, kind)`: membership predicate    | all-applicable, combined           |

`bagof.dispatchers` implements multiple dispatch driven by type hints. All
applicable overloads are found and the single most specific one runs. A
tie that cannot be broken raises `AmbiguousMethodError`, and a call to
which no overload applies raises `NoMethodError`. The question was which
of the registries fit that model.

The migration was validated empirically. For each registry, an equivalent
`bagof.dispatchers` function was built from the rules actually registered,
and its choice was compared with the bespoke one for every ordered pair of
concrete `Transformation` subclasses (6724 pairs at the time).

## convert

`convert` dispatches on `type(x)` and on the requested class `cls`, and
picks the converter whose declared output is the nearest supertype of
`cls`. The requested class is expressed as a `type[Out]` parameter, which
`bagof.dispatchers` treats as value-dependent and matches covariantly
against the class passed in. The new dispatch matched the old one on all
6724 pairs, with no ambiguity and no missing method.

The converter implementations take `(x, **kwargs)`, while `cls` is only a
dispatch key, so each registration is wrapped in a shim with signature
`(x, cls, **kwargs)` that drops `cls`. A catch-all converter from
`Transformation` to `Transformation` matches every request, so the result
is still checked against `cls`; otherwise a request for a type that no
converter produces would return an object of the wrong type.

## compose

`compose` dispatches on the types of its two operands, and unions such as
`Union[Linear, Scaling, Permutation]` are read natively. The new dispatch
matched the old one on 6667 of 6724 pairs. The 57 differences all came
from one bug in the bespoke metric, which was not monotonic: a class could
be farther from an ancestor than from a more distant one when the nearer
ancestor sat behind a longer chain of mixins or generic subscriptions. For
example, `type_distance(OmeZarrField, Sequence)` was about 4.17 while
`type_distance(OmeZarrField, Transformation)` was about 3.67, so the
bespoke dispatch picked the `(Sequence, Transformation)` composer over the
more specific `(Sequence, Sequence)` one. `bagof.dispatchers` picks
`(Sequence, Sequence)`. Both composers build a sequence and compute it, so
the results are the same transformation.

Composers carry no priority. A genuine tie raises `AmbiguousMethodError`
out of `compose`, so that a real ambiguity surfaces instead of being
resolved by registration order. No tie occurs over the 6724 pairs. The
only structural near-tie is the incomparable pair of composers
`(Sequence, Transformation)` and `(Transformation, Sequence)`, whose only
concrete overlap, `(Sequence, Sequence)`, is dominated by a third
composer.

## simplify

`simplify` takes one or two transformations. With one, it is a total
downcast of a leaf; with two, it is a partial collapse of a pair that
returns `None` to decline. Both arities are overloads of a single
function, a leaf rule `(t, policy)` and a pair rule
`(first, second, policy)`, and `bagof.dispatchers` selects by the number
of arguments, so the two never compete.

- Leaf rules are a pure single-winner dispatch, and the new dispatch
  matched the old one on every type.
- Pair rules form a chain of responsibility. `simplify` walks
  `_simplify.candidates(first, second, policy)` from the most specific
  candidate down, and calls each rule until one returns something other
  than `None`. If every rule declines, the pair is not collapsed.
  `candidates()` yields ties in registration order, which is the
  tie-break the bespoke dispatch used, so no priority is needed. Over the
  2529 applicable ordered pairs, the first candidate tried matched the
  rule that the previous single-winner dispatch with priorities selected,
  and the 255 genuine ties, such as `(Identity, Transformation)` against
  `(Transformation, Inverse)`, are exactly the pairs that the chain now
  walks.

A genuine tie between two leaf rules raises `AmbiguousMethodError` at the
call.

## is_kind

`is_kind(t, kind, compute=False)` does not fit single-winner dispatch, for
three reasons:

1. It combines every applicable checker with a logical OR. An `Affine`
   that holds a permutation, for instance, is established as orthogonal
   through the permutation node, while one that holds a rotation is
   established through the special-orthogonal node. These routes are
   incomparable, and a single winner would drop all but one of them.
2. The kind is matched contravariantly. A checker registered on node `N`
   answers a query for kind `K` when `N` is a subset of `K`, which is the
   reverse of the usual covariant direction, and the subset relation is
   that of the lattice of kinds, not of the class hierarchy.
3. It carries the `compute` level, either `False` for an analytic answer
   from structure alone or `True` for a numeric answer from the values.
   The level steers the answer but must not take part in specificity.

The source type is also ranked by its position in the MRO, so that an
`Inverse` wrapper shadows its concrete base, but only within the group of
checkers of one node; the answers of different nodes are combined. The
whole is a group-wise most-specific selection that feeds an OR reducer.

The first migration (#99) therefore left `is_kind` on the bespoke
dispatcher. It was moved onto `bagof.dispatchers` 0.3 in #101:
`Function.candidates` enumerates every applicable checker, and a
`Type[Super[node]]` lower bound matches the kind contravariantly, while
brainhops keeps the grouping by node and the OR reducer (see the
docstring of the `check` module). At migration time, the set of selected
checkers and the end-to-end answer at both `compute` levels were verified
to be unchanged for every concrete instance and every kind node. The
comparison was not kept as a regression test because it is too slow for
CI.

## Enhancements to bagof.dispatchers

The migration motivated five enhancements, framed for the whole bagof
family.

### E1. All-applicable mode with a reducer (proposed)

A combining mode would gather every applicable method, optionally only
the most specific one of each group, and combine their results with a
reducer supplied by the caller (`any`, `all`, `list`, `min`). `is_kind`
would be `any` over the group-wise most specific checkers. The same mode would host
membership predicates, validators and fan-outs.

### E2. Ordered candidates and a decline protocol (shipped in 0.3)

Composers, which historically returned `NotImplemented` to hand off,
and pair simplifiers, which return `None`, are both chains of
responsibility. `bagof.dispatchers` 0.3 added
`Function.candidates(*args)`, which returns the applicable methods in
order of specificity with ties in registration order, together with
related helpers. `simplify` uses it, which removed the interim
priorities. `compose` remains single-winner because no registered
composer declines. A candidate view is smaller and more composable
than a decline sentinel, and it subsumes E1 for callers that reduce
the results themselves.

### E3. Per-parameter variance or a pluggable relation (proposed)

`is_kind` needs a contravariant match under a domain-specific subset
relation. `bagof.dispatchers.core` already isolates the subtype
relation, which could be exposed as a per-parameter variance flag or
as a custom relation.

### E4. Context parameters that are carried but not dispatched on (proposed)

The `compute` level of `is_kind`, and the target class of `convert`
before it became a real `type[...]` key, are passed through without
affecting specificity. Today, a neutral hint that every overload
annotates identically achieves this by accident. A declared context
parameter would make the intent explicit and remove the shim.

### E5. Registration-time ambiguity check (resolved in 0.3.1)

Version 0.2 compared methods pairwise at registration and did not see
that a third method dominates the overlap of two others. It therefore
emitted a spurious `RuntimeWarning` for the composer near-tie and for
the 255 pair-simplifier ties, which brainhops silenced with scoped
warning filters. Version 0.3.1 removed the registration check:
ambiguity is reported only when a single-winner call meets it,
`candidates()` returns ties silently, and `ambiguities()` runs the full
selection. brainhops then dropped its warning filters.

## Validation

At the time of the migration, the full test suite passed on CPython 3.11
and 3.8, and `ruff check`, `ruff format --check` and `codespell` were
clean. The behaviour of every registered converter, composer, simplifier
and kind checker was unchanged.
