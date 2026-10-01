# Design: the compute API and open coordinate systems

**Status:** proposal, under review; decisions so far are recorded under
[Decisions](#decisions), remaining questions under
[Open questions](#open-questions). Nothing here is implemented yet.

This document proposes two connected changes:

1. **Part A.** A small, public set of operations (`compute`, `simplify`,
   `factor`, and `restrict` as a lazy method like `inverse`) with precise
   contracts. `compute` becomes a single pass.
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
  - `t.restrict(...)`, like `t.inverse()`, only *describes* a
    transformation lazily. Here it is a projection composition, which
    `compute` reduces.
  - `compute` is the only operation that composes.
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
  `@composer`, …), so supporting another of the library's classes means
  registering one rule, not editing the operation itself.
- **A missing rule is something to add, never a design constraint.** When
  an expression does not reduce well today because a composer or
  simplifier is not registered, the fix is to register it, not to route
  around it.

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

*If option 1 of A.8 is adopted, the normal form becomes
`[grid?, SubspaceTransformation(blocks)]`: one block per axis group, with
reordering absorbed into each block's `output_axes`, so `Π` disappears.
Restriction to whole blocks is then structural. The contract below is
unchanged otherwise.*

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

### A.4 Restriction: a lazy projection composition

*Revised after review. The first draft made `restrict` a free-standing
operation with its own per-type rules. It also argued that composition
could not deliver the cheap types and laziness. That argument was wrong:
it described the composers that happen to be registered today, not
anything the approach requires.*

#### Decoupled and coupled blocks

A transformation `t` maps input axes to output axes. Its *dependency
pattern* says which outputs read which inputs. Take a set of input axes
`I` and a set of output axes `O`.

- **`O` is *closed over* `I`** when every output in `O` reads only inputs
  in `I`. This is exactly what a restriction needs. The outputs in `O` are
  then a function of the inputs in `I` alone, whatever values the other
  inputs take.
- **The block `(I, O)` is *decoupled*** when, in addition, no output
  outside `O` reads an input in `I`. Then `t` is a direct product of a map
  `I → O` and a map on the rest. This stronger condition is what the
  factor pass needs, because both parts must be restrictable.
- **A block that is not closed is *coupled*.**

For an affine matrix `M`, closure is `M[O, ∁I] = 0`. Decoupling adds
`M[∁O, I] = 0`.

Example: a rotation in the `x`–`y` plane that leaves `z` alone.
- `({x, y}, {x, y})` is decoupled.
- `({z}, {z})` is decoupled.
- `({x}, {x})` is coupled, because output `x` reads input `y`.

Another example: `(x, y) → (x, x + y)`.
- `({x}, {x})` is closed but not decoupled, because output `y` reads `x`.
  Its restriction is the identity on `x`, and that is well defined.

The earlier draft required decoupling for `restrict`. Closure is enough.

#### Definition

Following the pattern of `t.inverse()`, which returns the lazy
`Inverse(t)` and leaves the work to `compute`:

> `t.restrict(input_axes, output_axes)` returns the lazy composition
> `π_O @ t @ ι_I`, where `ι_I` is a `Projection` that creates the input
> axes outside `I`, and `π_O` is a `Projection` that drops the output
> axes outside `O`. Nothing is evaluated until `compute`.

- **No new transformation type.** A restriction is an ordinary sequence
  of projections around `t`, so every rule that reduces it also applies to
  projections that users write themselves.
- **The keyword names match `SubspaceTransformation`** (`input_axes`,
  `output_axes`). No axis counts are needed: the projections index into
  `t`'s own axes, and open systems (Part B) describe unknown widths.

#### Reduction: rewrites, not composition

Reducing the sandwich is *analytic simplification*, run under
`compute(mode=False, simplify="analytic")`. It uses two kinds of
cost-free rewrite.

- **Commutation.** `π ∘ T → T' ∘ π'`, where `T'` is `T` cut down to the
  axes `π` keeps, and `π'` drops the corresponding input axes. The rule
  applies when the kept outputs are closed over some set of inputs.
  Repeating it moves the projection towards the input side, one element
  at a time.
- **Cancellation.** `π ∘ ι → Identity` when one exactly undoes the other.
  More generally, a drop followed by a create reduces to a single
  projection.

These rewrites never compose anything, which is what gives the
properties we want:

- **Type is kept.** `π ∘ Scaling` commutes to a smaller `Scaling`. It is
  not folded into an `Affine`, because nothing is folded.
- **Lazy inverses stay lazy.** `π ∘ Inverse(f)` commutes through `f`'s own
  rule (over the swapped axes) and re-inverts lazily.
- **Identity is kept.** When `I` and `O` cover all of `t`'s axes, both
  projections are identities, simplify drops them, and the result is `t`
  itself.

Under a mode that composes, `compute` reduces the sandwich the same way,
then composes what is left.

#### What has to be added

These are additions, not blockers.

1. **The meaning of created axes.** Settled by the spec: created axes
   are `0` (A.8).
2. **A third kind of simplification rule**, next to leaf rules and pair
   collapses: a pair *rewrite* (2 → 2). Every commutation moves a
   projection one step towards the input side, so the rewriting always
   terminates.
3. **One commutation rule per type.** These replace #104's `restrictors`
   one for one, so the number of registered rules does not grow.
4. **`Projection` with coordinate systems.** Which axes are dropped and
   created must be readable from, and written to, systems, including open
   ones (Part B).
5. **Composers for `Projection`**, so that a sandwich that does not fully
   reduce can still be composed, e.g. with an `Affine` under a mode that
   admits both.

#### Contract

- `t.restrict(I, O).compute()` is equivalent to `t` read from `I` to `O`,
  whenever `O` is closed over `I`.
- Under `mode=False, simplify="analytic"`, the result composes nothing.
  It has the cheapest type the rules know, keeps lazy inverses lazy, and
  is `t` itself when the block covers everything.
- `restrict(SubspaceTransformation(t, a, b), a, b)` reduces to `t`
  (through `SubspaceTransformation`'s commutation rule).
- **Coupled blocks** are resolved in A.8: `t.project(I, O)` gives the
  section at `0`, and `t.restrict(I, O)` raises.

The factor pass uses `restrict` exactly this way. Its partition only
yields decoupled blocks, which reduce fully.

### A.5 No separate `embed`

#104 introduced `embed` as a function. It is not needed:
- **Subspace embedding** (act on some axes, pass the rest through) is
  `SubspaceTransformation(t, input_axes, output_axes)`.
- **Adding axes** is `Projection(created=...)`.

Lowering either to a plain `Affine` is the job of `convert`/`simplify`,
and needs the full axis count, which open systems (Part B) can state, or
say is unknown. `embed` and its `ni`/`no` arguments are removed.

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
  `@simplifier`). **Decided: they stay private.**
  Registration is how the library's own transformation classes plug in.
  It is not a public extension point.
- **Bridging and adaptation.**

---

### A.8 Alignment with OME-NGFF 0.6

The meta transformations follow the OME-NGFF 0.6 coordinate-transformation
spec (`ome/ngff-spec`, tag `0.6`, `index.md`). Checked against it:

#### `projectAxis` and `Projection`

- **Spec:** `droppedInputs` lists indices of the *input* vector to drop.
  `createdOutputs` lists indices of the *output* vector where axes are
  added. "The value added to the coordinate vector defaults to zero."
  `projectAxis` "is not invertible in general if a dimension is dropped".
  When the dropped axis is discrete, it "MAY be applied along the dropped
  dimension by iterating over all possible values".
- **Our `Projection(dropped, created)`** has the same structure under
  shorter names.
- **This settles the created-value question in A.4: created axes are
  `0`.** That is what the spec says, so it is what `Projection` means. No
  `fill` field is needed, and the coupled-block section in A.4 is
  well defined: it is the section of `t` at `0`.
- **Correctness issue, independent of this design:**
  `Projection.inverse()` swaps `dropped` and `created`, but a projection
  that drops axes has no inverse.
  - Creating at `0` is only a *right* inverse: `drop ∘ create = id`, while
    `create ∘ drop ≠ id`.
  - Today, `~Projection(dropped=[2])` is treated as an inverse, which is
    wrong for any point not at `0` on that axis.
  - **Decided: `inverse()` raises when anything is dropped.** It keeps
    the swap only for a projection that creates without dropping, whose
    left inverse is the drop.
  - The cancellation rule in A.4 must be one-sided accordingly: a drop
    after a create of the same axes cancels, and the reverse order does
    not.

#### `mapAxis` and `Permutation`

- **Spec (0.6):** `mapAxis` is strictly a permutation. The array has one
  entry per axis, and each input index appears exactly once.
- **History.**
  - In 0.6.dev1, `mapAxis` was an object mapping output axis *names* to
    input axis names. It could drop axes (`{"x": "b"}`, "projection
    down") and duplicate them (`{"z": "b", "y": "b", "x": "a"}`,
    "projection up").
  - From 0.6.dev2 onwards (dev2, dev3, dev4, rc0 and 0.6), it is an
    integer array that MUST be a permutation.
  - The integer form was never allowed to be a subset. The name-object
    form was allowed to drop and duplicate, but `abczarr`'s 0.6 model
    (`mapAxis: List[int]`) cannot parse it.
- **Our reader** (`io/transformations/zarr/_map.py`) accepts a strictly
  increasing integer *subset* and reads it as a dropping `Projection`.
  That form was never valid in any version. **Decided:** following the
  rule "be liberal with formerly valid forms, strict with forms that were
  never valid", the reader raises on it. The writer only ever emits
  permutations; `projectAxis` is the 0.6 spelling of a drop. Accepting
  the dev1 name-object form would first need `abczarr` to parse it (out
  of scope here).

#### `byDimension`, the missing meta transformation

- **Spec:** a `byDimension` holds a list of children, each with
  `inputAxes`, `outputAxes` and a `transformation`.
  - "Every axis index in the parent byDimension's `output` coordinate
    system MUST appear in exactly one child transformation's
    `outputAxes`."
  - Children may read any input axes, in any order, and need not use all
    of them. For example, `byDimension2` maps a 4-D input to a 3-D output,
    reads input axes `[3, 2]` into outputs `[1, 2]`, and never reads
    input `0`.
  - There is no implicit pass-through.
- **We have no equivalent.** `SubspaceTransformation` is a *single* child
  with an implicit in-order pass-through of every other axis. It equals a
  `byDimension` whose other children are identities. The reverse does not
  hold: a `byDimension` with several non-identity children, unused inputs
  or reordered outputs is not a `SubspaceTransformation`.
- **History.** In every 0.6 version, `byDimension` is a *product* of
  children that together cover every output axis exactly once, with no
  implicit pass-through:
  - dev1: children addressed by axis *name* (`input`/`output`);
  - dev2: `input_axes`/`output_axes`, still by name;
  - 0.6: `inputAxes`/`outputAxes`, by index.

  `SubspaceTransformation`'s one-block-plus-in-order-pass-through
  semantics therefore does not come from any version of the spec. It is
  our own extension.
- **Can a `byDimension` be a sequence of `SubspaceTransformation`s?**
  Only partly. When every child is square (as many input as output axes),
  the product equals

      [P, Sub(t_1, o_1, o_1), …, Sub(t_m, o_m, o_m)]

  where `P` is a permutation (plus a projection dropping unread inputs)
  that moves each child's input axes to that child's output positions.
  The subspaces then act on disjoint axes in place, so they commute.
  The limits:
  - A child that changes the number of axes cannot act in place, so this
    form cannot represent it.
  - Writing the sequence back as a `byDimension` means recognizing the
    pattern, which is fragile.
  - It is exactly the `[F_1, …, F_m, Π]` chain that #104's factor pass
    juggles, with the composition gates that come with it.
- **Options.**
  - **(1) Generalize `SubspaceTransformation` into the product
    (recommended).** It keeps your name, gives one class instead of two,
    and becomes the factor normal form and the reader/writer target:
    - It holds a list of blocks `(transformation, input_axes,
      output_axes)`, with the spec's semantics: blocks may read any input
      axes in any order, and an input axis no block reads is dropped.
    - It **keeps our pass-through extension**: output axes that no block
      writes are fed, in order, from input axes that no block reads. This
      is what lets a subspace act on `x, y, z` of a space whose other
      axes are unknown (open systems, Part B), which the spec's
      "every output covered" rule cannot express.
    - Today's single-block call, `SubspaceTransformation(t, input_axes,
      output_axes)`, stays valid and means one block plus pass-through.
    - The writer closes the systems (`expand`) and emits a `byDimension`
      with explicit identity children for the passed-through axes. The
      reader maps a `byDimension` to blocks with nothing passed through.
    - **To settle:** the exact pass-through rule when the number of
      uncovered inputs and outputs differ. My proposal: the surplus
      uncovered *inputs* are dropped (as in the spec), and surplus
      uncovered *outputs* are an error, raised as soon as both counts are
      known.
  - **(2) Keep `SubspaceTransformation` as it is, and add a separate
    `ByDimension`.** Two classes with overlapping meaning, and a larger
    API surface.
  - **(3) No product class.** Read a `byDimension` into the sequence
    above. It is limited to square children, round-trips poorly, and
    keeps the factor normal form as a gated chain.

- **Other 0.6 types are already covered:** `identity`, `scale`,
  `translation`, `affine`, `rotation`, `sequence`, `bijection`,
  `displacements` and `coordinates`. 0.6 has no `inverseOf`, so our lazy
  `Inverse` is internal, and writers must emit it either as the closed-form
  inverse or inside a `bijection`.

#### `restrict` and `project`

The two behaviours in A.4 become two methods with distinct contracts. A
boolean flag that changes the meaning of the result would be a worse
interface.

- **`t.project(input_axes, output_axes)`** is the lazy
  `π_O @ t @ ι_I`, with `ι_I` creating axes at `0` as the spec defines.
  It is always defined.
  - On a closed block, it reduces by commutation.
  - On a coupled block, it is the section of `t` with the other inputs at
    `0`, computed by composition.
  - The name follows the spec's `projectAxis`.
- **`t.restrict(input_axes, output_axes)`** is the same expression, with
  the guarantee that the block is closed. When it is not, it raises
  `RestrictionError`.
  - **Decided: it raises lazily**, at `compute`, when a commutation
    cannot proceed. Like `inverse`, `restrict` only builds an expression.

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
  - **`CoordinateSystem.ndim`** (new property) is the number of axes of a
    closed system, and `None` for an open one, including `axes=None`.
  - The count of *explicit* axes never includes the `...` entry, even
    though `...` is an item of the `axes` list.
- **Fixed-dimension classes cannot be open.** `CoordinateSystem2D`/`3D`
  and their subclasses reject `...` (and `None`) at validation. Their
  `ndim` is always their fixed count.
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

- **`expand(ndim)`:** the closed system obtained by replacing `...` with
  as many `Axis()` as needed to reach `ndim` axes.
  - It raises if `ndim` is less than the number of explicit axes.
  - On a closed system, it returns the system itself when `ndim` matches,
    and raises otherwise.
  - Writers (B.5) and anything that learns the true width from data use
    it.
- **`take(positions)`:** the system restricted to those positions, in
  order. Positions that resolve into `...` become `Axis()`. Projections
  use this to describe the systems on either side (A.4).
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
   `embed`, the restrictors) are not exported, so merging commits to
   nothing public. PR 2 replaces them.
2. **PR 1, against main: open coordinate systems (Part B).**
   - `...` semantics, `CoordinateSystem.ndim`, `Axis.compatible`/`merge`,
     and `CoordinateSystem.expand`/`take`/`place`/`compatible`.
   - All the consumers listed in B.5.
   - `_subsystem` becomes `place(…, ndim=None)`, which fixes the
     `Subspace → Affine` crash.
3. **PR 2: the operation API (Part A).**
   - `Projection` gains coordinate systems and a defined meaning for
     created axes (or explicitly none; see the open question in A.4).
   - Pair-rewrite simplification rules, with commutation rules per type
     and projection cancellation. These replace #104's restrictors.
   - `Transformation.restrict` returns the lazy projection composition,
     and `embed` is removed.
   - `factor` becomes a public, pure rewrite built on `restrict`.
   - Factors declare their full-space systems.
   - `compute` becomes a single pass.
   - Reslice asks for `simplify="numeric"`.
   - Checked by the parity sweep, the idempotence test, and the reslice
     timings.

## Decisions

1. Restriction takes no `policy` argument. (Now moot: restriction is
   `t.restrict(I, O)`, and policy belongs to `compute`/`simplify`.)
2. The registration decorators stay private.
3. `axes=None` (and `input`/`output=None`) is not normalized to `[...]`,
   but every reader, equality and compatibility treats the two spellings
   identically, through one accessor.
4. `...` may appear anywhere, at most once.
5. Writers close an open system from the data shape (`expand`).
6. `CoordinateSystem.ndim` is `None` for an open system. Fixed-dimension
   classes reject `...`, and `...` never counts towards the number of
   explicit axes.
7. Restriction follows `inverse`: `t.restrict(I, O)` returns the lazy
   composition `π_O @ t @ ι_I`, reduced by `compute` through analytic
   rewrite rules. Missing composers or simplifiers are added, not worked
   around.
8. Created axes are `0`, as the OME-NGFF 0.6 `projectAxis` spec defines.
9. `t.project(I, O)` (always defined; the section at `0` on a coupled
   block) and `t.restrict(I, O)` (raises on a coupled block) are separate
   methods.
10. `restrict` raises lazily, at `compute`. Like `inverse`, it only
    builds an expression.
11. `Projection.inverse()` raises when the projection drops axes.
12. A subset integer `mapAxis`, never valid in any 0.6 version, is
    refused on read.

## Open questions

1. A.8: generalize `SubspaceTransformation` into the multi-block product
   (option 1, recommended)? If so, confirm the pass-through rule
   (surplus uncovered inputs dropped, surplus uncovered outputs an
   error).
