# Design: the compute API and open coordinate systems

**Status:** proposal, under review; the first round of decisions is
recorded under [Decisions](#decisions). Nothing here is implemented yet.

This document proposes two connected changes:

1. **Part A.** A small, public set of operations (`compute`, `simplify`,
   `factor`, `restrict`) with precise contracts. `compute` becomes a single
   pass built from the other three.
2. **Part B.** *Open* coordinate systems: `CoordinateSystem.axes` may contain
   `...`, meaning "an unknown number of axes that we know nothing about".
   `None` means the same thing as `[...]`.

The guiding rule for both: an operation that becomes public must have a
contract that makes sense on its own, can be documented in a few
sentences, and does exactly what it says, with no silent surprises.
Passing tests is not enough.

---

## Part A. Public operations

### A.1 Principles

- **Few operations, each with one job.**
  - `simplify` rewrites a transformation into a cheaper, equivalent one
    without composing anything.
  - `factor` rewrites it into independent axis groups without composing
    anything.
  - `restrict` cuts out one block of axes.
  - `compute` is the only operation that composes. It is built from the
    other three.
- **Every operation returns a transformation equivalent to its input**, or
  raises. None of them silently drops a piece, changes a value, or returns
  an approximation.
- **One policy argument decides how much an operation may look at
  parameter values**, and it is the same for every operation:
  - `"analytic"` reads types and structure only (axes, kinds, the identity
    of lazy-inverse links);
  - `"numeric"` may also read parameter values, with exact tests only
    (no tolerance);
  - `"none"` does nothing.
- **Identity is preserved.** When an operation has nothing to do, it
  returns its input (the same object), and every leaf it does not rewrite
  keeps its identity. This is what lets a transform and its lazy inverse
  keep cancelling.
- **Per-type rules by registration.** Each operation is a
  `bagof.dispatchers` Function with a private decorator (`@simplifier`,
  `@restrictor`, …), so supporting another of the library's classes means
  registering one rule, not editing the operation itself.

### A.2 `simplify(t, policy="analytic")`

*Exists today as `Transformation.simplify`. The contract below states what
it already does.*

> Return a transformation equivalent to `t`, of the cheapest type that the
> policy can prove, without composing anything: no matrix is multiplied,
> no field is sampled, and no lazy inverse is materialized.

- Under `"analytic"` it downcasts by structure, and cancels a transform
  next to its own lazy inverse (by object identity).
- Under `"numeric"` it also downcasts on exact parameter tests, e.g. an
  `Affine` whose matrix is exactly diagonal becomes a `Scaling`.
- It never makes a sequence longer.

### A.3 `factor(t, policy="analytic")` (new public operation)

> Rewrite `t` into its axis-group normal form
> `[grid?, F_1, …, F_m, Π?]` without composing anything. Each
> `F_i = SubspaceTransformation(chain_i, A_i, A_i)` acts only on the axis
> group `A_i`, and `chain_i` is `t` restricted to that group. When `t` does
> not split into two or more groups, `t` itself is returned.

**Changes from what #104 does today:**

- **No composition inside `factor`.** Today the factor pass composes each
  group's chain itself (it takes `mode` and `simplify`). Under this
  proposal, `chain_i` is the *uncomposed*, restricted chain, and `compute`
  composes it.
  - `factor` loses its `mode` parameter.
  - A group whose pieces cannot be composed no longer forces the whole
    sequence to stay unfactored. Its chain just stays uncomposed.
- **The partition follows the policy.**
  - Under `"analytic"`, a dependency pattern is read from structure only:
    - `Scaling`, `Translation` and `Identity` are diagonal;
    - a `Permutation` is read from its index vector, which counts as
      structure, like axes;
    - a subspace is read from its axes;
    - fields couple all their axes;
    - `Affine`/`Linear`/`Rotation` are treated as fully coupled.
  - Under `"numeric"`, the exact zero pattern of affine matrices is read as
    well. That is what #104 does today, whatever the policy.
- **Each factor declares its full-space systems.**
  - Between factors, the space is mixed: axes in groups already applied
    carry the data system's axes, mapped back through `Π`, and the
    remaining axes carry the grid system's axes.
  - With the chain's end systems known, `factor` builds these systems and
    sets them on each `F_i`. A factor therefore never relies on
    `SubspaceTransformation` deriving a full-space system from its inner
    (see B.1).
- **Name.** `factor` is the verb that matches `simplify` and `restrict`.
  The module `factor.py` would be renamed (e.g. to `factoring.py`) so the
  function and the module do not share a name. The `factor=` keyword of
  `compute` stays.

**Contract details:**

- **Idempotent:** `factor(factor(t)) is factor(t)`.
- **Never materializes a lazy inverse.** It is read through the transpose
  of its forward's pattern, and restricted through its forward.
- **Equivalent to `t`**, so `compute(factor(t))` gives the same values as
  `compute(t)`.

**Consumer:** the reslice executor reads the normal form. It asks for
`compute(mode=Affine, factor=True, simplify="numeric")` explicitly, since
resampling reads every value anyway. This makes the value-reading visible.
It is no longer hidden inside `factor`.

### A.4 `restrict(t, input_axes, output_axes)` (new public operation)

> Return the transformation `t` performs from its input axes `input_axes`
> to its output axes `output_axes`, as a transformation over those axes
> alone. This is only defined when that block is *decoupled*: no output in
> `output_axes` reads an input outside `input_axes`, and no other output
> reads an input in `input_axes`. A block that is not decoupled raises
> `RestrictionError`; it never gives an approximation.

**Changes from #104 today:**

- **Keyword names** match `SubspaceTransformation`: `input_axes` and
  `output_axes`, instead of `rows` and `cols`.
- **No `ni`/`no` arguments.**
  - Restriction never needs the total number of axes. Every rule indexes
    into what `t` already holds: matrix rows and columns, scale entries,
    permutation entries, or a subspace's absolute axis positions.
  - An `Identity` restricts to an `Identity`.
  - With open systems (Part B), the restricted systems are well defined
    even when the full count is unknown.
- **Decoupling is checked, not assumed.** Today the factor pass guarantees
  it, but a public function has to protect itself. The check reuses the
  dependency-pattern reader under the call's policy. A block that cannot
  be shown to be decoupled raises.
  - Restricting an `Affine` reads its sub-block, i.e. its values, so
    checking the off-block zeros costs nothing extra.
  - **Decided: `restrict` takes no `policy` argument.** Cutting out a
    block is inherently a read of that block, so `restrict` may always
    read the values it extracts, and the off-block zeros that make the
    extraction exact.
- **The result carries restricted systems.**
  - `input` is `t.input` restricted to `input_axes`, and `output` is
    `t.output` restricted to `output_axes`.
  - Adjacent pieces of a chain therefore agree by construction. Element
    *k*'s output, restricted to a group, is exactly element *k+1*'s input
    restricted to the same group.
- **A block that covers everything `t` acts on returns `t` itself, or its
  inner for a subspace**, so object identity is kept.
- **Duality with `SubspaceTransformation`:**
  `restrict(SubspaceTransformation(t, a, b), a, b) is t`.

#### Relation to `Projection`

`Projection` does not wrap a transformation. It is a map of its own that
drops axes (`dropped`), or, inverted, an embedding that creates axes
(`created`). The composition that matches a restriction is therefore

    π_O ∘ t ∘ ι_I

where `ι_I` embeds the input axes `I` into the full input space and `π_O`
drops every output axis outside `O`. On a decoupled block, this
composition *is* `restrict(t, I, O)`. `restrict` still earns its place:

- **Its precondition is exactly what makes the composition well defined.**
  `ι_I` has to give the axes it creates *some* value, and the model does
  not say which.
  - On a decoupled block, the outputs in `O` never read those axes, so the
    value does not matter.
  - On a coupled block, the composition is a *slice* of `t` at an
    arbitrary point. That is a different operation, and silently returning
    it would be a bad surprise. `restrict` checks decoupling and raises
    instead.
- **Cheaper type, no materialization, identity kept.**
  - No `Projection` composers exist, and the subspace/affine composers
    would turn a `Scaling` into a dense `Affine`.
  - Composing would read a lazy inverse's parameters.
  - Composing returns new objects, which breaks cancellation by object
    identity.
- **One rule per type.** Getting the same through composition would need a
  composer for every `(Projection, T)` pair and every `(T, Projection⁻¹)`
  pair. That is twice as many registrations, each restating the
  restriction.

So the projection composition is how `restrict` is *specified*, and
`restrict` is how it is *computed*. The docstring states the identity
`restrict(t, I, O) == compute(π_O ∘ t ∘ ι_I)` for decoupled blocks, and a
test checks it numerically.

### A.5 No separate `embed`

#104 introduced `embed` as the inverse of `restrict`. The embedding
already exists: it is `SubspaceTransformation(t, input_axes, output_axes)`.
Lowering that embedding to a plain `Affine` is the job of `convert`/
`simplify`, and it needs only one thing: knowing how many axes the full
space has. Part B makes "we don't know" representable. With it:

- the subspace's derived systems no longer claim a wrong count;
- the `Subspace → Affine` converter raises "axis count unknown" instead of
  building a matrix that is too small;
- `factor` no longer needs `embed` at all, because it sets explicit
  systems on its factors (A.3) and leaves pieces uncomposed when they
  cannot be lowered.

So `embed` and its `ni`/`no` arguments are removed, which shrinks the API
surface.

### A.6 `compute(t, mode=True, *, simplify="analytic", factor=False)`

> Return a transformation equivalent to `t` in which every run of
> transforms that `mode` admits has been composed.

A single pass, with no fixpoint loop:

1. **Bridge** every boundary where adjacent systems disagree. This is done
   once and already recurses into nested sequences.
2. **Flatten.**
3. **`simplify(·, policy)`.**
4. **If `factor`: `factor(·, policy)`.**
5. **Compose once.**
   - Outside the factors, compose runs of transforms `mode` admits; the
     factor gates are kept.
   - Inside each factor, compose its chain recursively.
   - Simplify each product as it is produced. If a product collapses to
     `Identity`, drop it, and keep composing across the gap.

**Why no loop:**

- Measured on main: every bridge is inserted in round 1. 37 out of 37
  second rounds changed nothing, and so did 4,000 random chains and 4,500
  numeric scanner-affine chains.
- Under `factor=True`, 132 of 133 calls ran exactly two rounds, the second
  only confirming that nothing changed.
- The one case a second round helps (a product that simplifies away,
  leaving a composable pair) is handled locally by step 5.

**Contract additions:**

- `compute` is **idempotent**: `compute(compute(t))` is structurally
  identical to `compute(t)`. This is enforced by a test over the parity
  sweep and the test suite, and replaces the iteration cap.
- `compute(t, factor=True)` never fails where `compute(t)` succeeds.

### A.7 What stays internal

- **`compose` and `convert`.** Their public faces are `compute` and
  `Transformation.to`.
- **The registration decorators** (`@composer`, `@converter`,
  `@simplifier`, `@restrictor`). **Decided: they stay private.**
  Registration is how the library's own transformation classes plug in.
  It is not a public extension point.
- **Bridging and adaptation.**

---

## Part B. Open coordinate systems (`...`)

### B.1 The problem

A `SubspaceTransformation` that declares no full-space systems derives
them from its inner system (`meta._subsystem`). That function places the
inner axes at their positions, fills the gaps with `Axis()`, and stops at
`max(axes) + 1`. The *positions* are right, but the *count* is invented.
This causes two problems:

- **On main:** `Sub(Scaling([2], input=CS([x]), output=CS([x])), axes=[0])`
  inside a 3-D chain reports a 1-axis system. `to(Affine)` then builds a
  1×2 matrix, and composition raises a matmul `ValueError`.
- **In #104:** restricted pieces with systems produced factors whose
  systems were too short. This broke composition and the reslice planner,
  and silently skipped the discrete-axis check.

The count is genuinely unknown, and nothing in the current types can say
so. `None` throws away the positions we do know, and the
`composers.py` discrete-axis check relies on them.

### B.2 Semantics

- `CoordinateSystem.axes` is a list whose items are `Axis` instances, plus
  **at most one `...` (`Ellipsis`)**, which stands for *zero or more axes
  about which nothing is known*.
- **`axes=None` is equivalent to `axes=[...]`**, and a transformation
  endpoint `input=None`/`output=None` is equivalent to a system with
  `axes=[...]`. **Decided: no normalization for now.** `None` stays an
  accepted spelling and is stored as given. Every place that interprets
  axes must treat the two spellings identically:
  - All readers go through one accessor, which returns `[...]` for a
    `None` system or `None` axes. No code inspects `system.axes` or
    `t.input` for `None` on its own.
  - Equality and compatibility of systems treat `None` and `[...]` as the
    same.
  - A test parametrizes the open-system cases over both spellings, so the
    two cannot drift apart.
- **A system with `...` is *open*; one without is *closed*.**
  - `ndim` is `None` for an open system.
  - `min_ndim` counts its explicit axes.
- **Fixed-dimension classes cannot be open.** `CoordinateSystem2D`/`3D`
  and their subclasses reject `...`.
- **Where `...` may appear. Decided: anywhere, at most once**, as in
  numpy. This expresses things like `[..., TimeAxis()]` ("the last axis is
  time"), as well as the subspace case, which needs a trailing `...`
  because subspace positions are absolute, counted from 0. Positional
  access goes through the accessor:
  - a non-negative index resolves when it falls in the explicit prefix
    (before `...`);
  - a negative index resolves when it falls in the explicit suffix (after
    `...`);
  - any other index of an open system resolves to an unknown `Axis()`;
  - an index past the end of a closed system is an error.

### B.3 Operations on systems

These are small, documented methods on `CoordinateSystem`. They replace
the ad-hoc list manipulation currently scattered across modules.

- **`take(positions)`:** the system restricted to those positions, in
  order. Positions that resolve into `...` become `Axis()`. `restrict`
  uses this (A.4).
- **`place(positions, ndim=None)`:** the inverse of `take`. It is the
  full-space system in which this system's axes sit at `positions`.
  - Gaps are filled with `Axis()`.
  - With `ndim=None`, the result ends with `...`, because the count is
    unknown.
  - This replaces `meta._subsystem`'s padding.
- **`compatible(other)`:** whether some expansion of the `...` makes the
  two systems match axis by axis, under `Axis.compatible`. Equality stays
  strict and structural.

### B.4 The `Axis` class

- **`Axis()`, with every field `None`, is "an axis about which nothing is
  known".** It is already the placeholder; we make that official.
- **`Axis.compatible(other)`:** every field set on both sides must be
  equal. A `None` field matches anything. This lets
  `[RAS axes, ...]` be compatible with a closed 4-D RAS+time system.
- **`Axis.merge(other)`:** combines what both sides know, and raises if
  they conflict. It is used when two descriptions of the same axis meet,
  e.g. a subspace's derived system against a declared neighbour.
- **Subclasses with defaults** (`SpatialAxis` defaults to millimetres) are
  *not* unknown. Only plain `Axis()` is.

### B.5 Consumers that must handle open systems

Every reader of an axis count or position is updated in the same PR, not
just the one that crashes:

- `converters.py`, `Subspace → Affine`: an open system raises
  `ConversionError("axis count unknown")`.
- `adaptors.py`, the bridging checks (`len(source.axes) !=
  len(target.axes)`): compare using `compatible`. A bridge that would need
  to reorder axes it cannot see raises `AdaptationError`.
- `composers.py`, the discrete-axis check: positional access (B.2) keeps
  working on the known positions.
- `utils.axis_counts`, and the `separable`/`factor` readers: `ndim is
  None` means unknown, never a guess.
- **I/O writers** (NGFF/OME-Zarr, NIfTI). No format can store `...`.
  **Decided: writers close an open system from the data shape.**
  - They expand `...` to as many axes as the data has, beyond the explicit
    ones.
  - The expanded axes are filled the way the format fills an axis it
    knows nothing about.
  - A closed system whose length disagrees with the data still raises, as
    it does today.

---

## Sequencing

1. **Merge #104 as it is.** Its helpers (`factor_sequence`, `restrict`,
   `embed`) are not exported, so merging commits to nothing public.
2. **PR 1, against main: open coordinate systems (Part B).**
   - `...` semantics, `Axis.compatible`/`merge`, and
     `CoordinateSystem.take`/`place`/`compatible`.
   - All the consumers listed in B.5.
   - `_subsystem` becomes `place(…, ndim=None)`, which fixes the
     `Subspace → Affine` crash.
3. **PR 2: the operation API (Part A).**
   - `factor` and `restrict` become public with the contracts above, and
     `embed` is removed.
   - `factor` becomes a pure rewrite, and `restrict` carries systems.
   - Factors declare their full-space systems.
   - `compute` becomes a single pass.
   - Reslice asks for `simplify="numeric"`.
   - Checked by the parity sweep, the idempotence test, and the reslice
     timings.

## Decisions

1. `restrict` takes no `policy` argument.
2. The registration decorators stay private.
3. `axes=None` (and `input`/`output=None`) is not normalized to `[...]`,
   but every reader, equality and compatibility treats the two spellings
   identically, through one accessor.
4. `...` may appear anywhere, at most once.
5. Writers close an open system from the data shape.
