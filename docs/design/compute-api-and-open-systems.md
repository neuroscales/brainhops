# Design: the compute API and open coordinate systems

**Status:** Part B is implemented and merged (#112, with #115). Part A is
an agreed design that is not implemented yet. Decisions are recorded under
[Decisions](#decisions).

This document covers two connected changes:

1. **Part A (to do).** A small, public set of operations (`compute`,
   `simplify`, `factor`, and `restrict`/`project` as lazy methods like
   `inverse`) with precise contracts. `compute` becomes a single pass.
2. **Part B (done).** *Open* coordinate systems: `CoordinateSystem.axes` may
   contain `...`, meaning "zero or more axes that we know nothing about".
   Part B describes what was built, which differs in places from the first
   proposal. In particular, `axes=None` reads as `[...]`, but a `None`
   *system* is not the same as a system with `axes=[...]`.

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
    `SubspaceTransformation` deriving a full-space system from its inner,
    which gives an open system (B.1, B.3).
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
   ones. Part B provides the system side: `CoordinateSystem.restrict`
   gives the system that remains after a drop, and
   `CoordinateSystem.embed` the one after a create (B.3). `Projection`
   itself is unchanged by Part B.
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

This is #104's transformation-level `embed` function. It is unrelated to
`CoordinateSystem.embed` and `AxisSequence.embed` (B.2, B.3), which place a
system's axes in a larger space and stay.

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
    - **Decided: option 1.** The pass-through rule is in A.9: it is
      decided by axis *names* where they exist, and never by guessing
      which positional axes to drop.
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

### A.9 Axis references: positions or names

OME-NGFF 0.6 refers to axes by integer position, and it can, because an
OME transformation always sits between coordinate systems that list
*every* axis. Brainhops has no such guarantee. Systems may be missing, or
open (Part B). A positional reference into a space of unknown width is
therefore often a guess, and the pass-through rule is where it hurts. If
a block leaves two inputs and one output uncovered, which input is
dropped?

**Proposal: wherever brainhops takes an axis reference, it accepts a
position (`int`) or a name (`str`).** This covers a block's
`input_axes`/`output_axes`, `Permutation`, `Projection`, and the axes of
`t.restrict`/`t.project`.

- **References are stored as given and resolved lazily**, against the
  coordinate system on the relevant side, when the transformation is
  computed, applied or written.
  - A name resolves when that system has exactly one axis with that name
    (`axes["x"]`). An open system may hold it among its explicit axes. It
    never matches one of the axes that `...` stands for.
  - Otherwise it raises: unknown or ambiguous name, or no system.
  - Names survive reordering and bridging, which positions do not.
- **Validation against the transformation's own systems. Decided:** when
  a transformation carries an input or output system, every axis
  reference on that side is checked against it, whether the reference is
  a name or a position.
  - A name must exist, and be unique.
  - A position must be within range of a closed system. In an open one,
    every position is valid: it falls in the explicit part or inside `...`
    (`axes.at(position)`, B.2).
  - A transformation *carries* a system when the endpoint is not `None`,
    including an explicit open one such as `CoordinateSystem()`, which is
    kept as given (B.3). Such a system checks only the axes it states.

  The check runs when the transformation is built, and again when a system
  is attached or replaced (e.g. through `.to(input=...)`). A reference is
  only left unchecked while there is no system to check it against.
  Resolution needs one public function that maps a reference to a
  position, which Part B only partly provides: `AxisSequence.restrict`
  accepts positions or names, but its name-to-position step is private
  (see Open questions).
- **Writers resolve everything to positions**, because OME 0.6 only has
  positions. A writer closes the systems first (`expand`).
- **Mapping-type transformations also accept a name mapping**, as the
  0.6.dev1 `mapAxis` did. **Decided: added now**, in the operation-API PR:
  - `Permutation({"x": "j", "y": "i"})`;
  - `Projection` that drops `["c"]`.

  A mapping that duplicates an axis (dev1's "projection up",
  `{"z": "b", "y": "b"}`) is neither a permutation nor a projection. It
  stays out of scope unless we add a dedicated type.

**Pass-through rule of the multi-block `SubspaceTransformation`**, with
no guessing:

1. **Every output written by a block:** inputs that no block reads are
   dropped. This is the spec's own semantics, and it is explicit by
   construction.
2. **Some outputs unwritten, and the axes are named:** each unwritten
   output is fed from the unread input *with the same name*.
   - An unread input with no same-named output is dropped.
   - An unwritten output with no same-named input raises.
3. **Some outputs unwritten, and the axes are not named:** positional
   in-order pairing applies only when the number of unread inputs equals
   the number of unwritten outputs. Otherwise it raises and asks for
   names (or explicit blocks). It never picks axes to drop.

An open system's unknown `...` axes on both sides pass through one to
one. This is the case a subspace acting on `x, y, z` of an unknown space
needs, and no axis is ever dropped from inside `...`.

---

## Part B. Open coordinate systems (`...`)

*Implemented in #112 (open systems, axis containers, physical systems,
memory order) and #115 (sample units). The sections below describe the
code on `main`.*

### B.1 The problem

A `SubspaceTransformation` that declares no full-space systems derives
them from its inner system (`meta._subsystem`). The old code placed the
inner axes at their positions, filled the gaps with `Axis()`, and stopped
at `max(axes) + 1`. The *positions* were right, but the *count* was
invented. This caused two problems:

- **On main, before Part B:**
  `Sub(Scaling([2], input=CS([x]), output=CS([x])), axes=[0])` inside a 3-D
  chain reported a 1-axis system. `to(Affine)` then built a 1×2 matrix, and
  composition raised a matmul `ValueError`.
- **In #104:** restricted pieces with systems produced factors whose
  systems were too short. This broke composition and the reslice planner,
  and silently skipped the discrete-axis check.

The count is genuinely unknown, and no type could say so. `None` throws
away the positions that are known, and the `composers.py` discrete-axis
check relies on them. Now `_subsystem` returns a declared system as is, and
otherwise derives an open one with `CoordinateSystem.embed` (B.3).

### B.2 Axis containers

The axes of a system are stored in one of three related types, all in
`brainhops.datamodel.systems`.

- **`AxisSequence(collections.abc.Sequence)`** is the shared read-only API.
  It holds `Axis` items and at most one `...`, anywhere. `...` stands for
  *zero or more axes about which nothing is known*, and never counts as an
  axis. A sequence with two `...` describes no axes: every method that
  reads the axes raises `ValueError`.
  - **`is_open`** is true when it holds `...`. **`ndim`** is the number of
    axes, or `None` when it is open. The empty sequence is closed, with
    `ndim == 0`.
  - **`index(Axis | str)`** is `list.index` with a looser test. An entry
    matches when it is an instance of the query's class and has every field
    that the query sets, with the same value. A name stands for
    `Axis(name=...)`. `...` matches nothing.
  - **`["name"]`**, **`"name" in`** and **`names`** read explicit axes by
    name. A name never matches one of the axes that `...` stands for. A
    missing name raises `KeyError`, a shared one `ValueError`. There is no
    `keys()`, `values()`, `items()`, `update()` or `pop()`.
  - **`at(position)`**: the axis at a *position* in the space.
  - **`expand(ndim)`**, **`restrict(refs)`**, **`embed(positions, ndim)`**
    and **`compatible_with(other)`**: see B.3.
- **`AxisTuple(tuple, AxisSequence)`** is immutable and typed per position:
  `AxisTuple[SpaceAxis, SpaceAxis]` is two spatial axes. A field of that
  type converts each item to the type of its position, and refuses a wrong
  number of items or `...`. Fixed-dimension systems store it.
- **`AxisList(AxisSequence, list)`** is mutable. Systems whose number of
  axes is not fixed (the open-capable ones) store it.

Slices and every method that builds a sequence return the sequence's own
type. A plain list or tuple given as `axes` is converted.

**Entries and positions are different things.**
`len()`, iteration, `==`, `repr`, `[i]` and `index` are about *entries*,
`...` included, as in any list. `ndim`, `at`, `expand`, `restrict`,
`embed` and `compatible_with` are about *positions in the space*: a
non-negative position counts from the first axis, and a negative one from
the last. In a closed sequence the two coincide. In `[x, ..., t]`, entry 2
is `t`, but position 2 is one of the axes `...` stands for.

**`at(position)`** reads the axis at a position:

- in a closed sequence, the position must lie in `[-ndim, ndim)`, else
  `IndexError`;
- in an open one, every position is valid. A non-negative position in the
  explicit prefix, or a negative one in the explicit suffix, gives that
  axis, and any other position gives a new unknown `Axis()`.

There is no `AxisList.of` and no public `position` method. The `AxisList`
and `AxisTuple` constructors are the standard `list` and `tuple` ones, and
the position-to-entry arithmetic is private.

### B.3 Coordinate systems

#### The `axes` field

- **`CoordinateSystem.axes` is not optional.** `axes=None` is not "no axes".
  It reads as not giving them, so the class's default takes its place:
  `[...]` for an open-capable class, and the class's own default axes for a
  fixed-dimension one. The system stores that default, never `None`:
  `CoordinateSystem(axes=None)` and `CoordinateSystem3D(axes=None)` are
  `CoordinateSystem()` and `CoordinateSystem3D()`.
- **Equality is plain field-wise**: same class, and equal fields.
  `CS() == CS(axes=None) == CS(axes=[...])`, and no system equals `None`.
  There is no second spelling to keep in sync, and no accessor for it.
- **Which classes are open-capable.** `CoordinateSystem`,
  `SpatialCoordinateSystem`, `PhysicalCoordinateSystem` and
  `ArrayCoordinateSystem` (with its C- and F-ordered bases) store an
  `AxisList` and accept `...`. A `SpatialCoordinateSystem` can hold only
  spatial axes and `...`.
- **Fixed-dimension classes cannot be open.** `CoordinateSystem2D`/`3D` and
  every subclass (spatial, array, pixel, voxel, RAS, ...) store an
  `AxisTuple` and refuse `...`. Their `ndim` is always their fixed count.
- **`...` may appear anywhere, at most once**, as in numpy. This expresses
  `[..., TimeAxis()]` ("the last axis is time"), and the subspace case,
  which needs a trailing `...` because subspace positions are absolute,
  counted from 0.
- **`CoordinateSystem.ndim`** is the `ndim` of the axes: the number of axes
  of a closed system, and `None` for an open one.

#### A `None` system is not an open system

`CoordinateSystem(axes=[...])` and `axes=None` are the same, but a
transformation endpoint `input=None`/`output=None` is **not** the same as
`input=CoordinateSystem(axes=[...])`.

- A **`None` endpoint** is no system. It *defers* to its context or is
  *derived*: a subspace derives it from its inner transformation, an
  `Inverse` takes it from its forward, a `Sequence` from its members, a
  `Bijection` from either side, `Geometry` propagates onto it.
- An **explicit system is always kept** as given, even `CoordinateSystem()`.
  It does not defer.
- Code that needs the axes of a possibly missing endpoint reads them with
  `_axes_or_unknown(system)`, which maps `None` to `[...]` and nothing
  else.

An earlier iteration treated an endpoint that "says nothing" as `None`
(`_says_nothing`). It was reverted: it silently discarded systems the
user had set.

#### Dispatch

Calling `CoordinateSystem(...)` builds the most specific class its axes
describe (bagof polymorphism). Every predicate on the axes (how many, or
what each one is) says something about *all* of them. **An open axis list
matches no such predicate**, so an open system is built as the class it was
called as: `CoordinateSystem(axes=[x, ...])` is not two-dimensional, and
`[R(), A(), S(), ...]` is not an `RASCoordinateSystem`.
**`expand` re-dispatches** the closed result, so
`CoordinateSystem().expand(2)` is a `CoordinateSystem2D`. A predicate on
another field, `order`, still applies (B.5).

#### Operations

These are methods of `CoordinateSystem`, and of `AxisSequence` for the
axes alone. A system's version calls the axes' version and builds a system
from the result. They replace the ad-hoc list manipulation that was
scattered across modules. (The first proposal named them `take`, `place`
and `compatible`.)

- **`expand(ndim)`:** the closed system obtained by replacing `...` with as
  many unknown `Axis()` as needed to reach `ndim` axes.
  - It raises `ValueError` if `ndim` is less than the number of explicit
    axes, and, on a closed system, if it differs from the axis count.
  - A closed system is returned as itself. An open one is rebuilt through
    its class, with its other fields kept, so the result may be a subclass
    (see Dispatch).
  - Writers (B.6) and anything that learns the true width from data use it.
- **`restrict(refs)`:** the system of the axes at some positions or names, in
  the order of `refs`. A position that falls inside `...` gives an unknown
  `Axis()`. A position past the end of a closed system raises, and so does
  a repeated axis. The result describes a different space, so the class and
  name are not carried over: it is what `CoordinateSystem(axes=...)` builds
  from the restricted axes. Projections use this to describe the system
  after a drop (A.4).
- **`embed(positions, ndim=None)`:** the inverse of `restrict`. It is the
  system in which this system's axes sit at `positions` (non-negative,
  absolute, so never names).
  - Gaps are filled with `Axis()`.
  - With `ndim=None` the result ends with `...`, because the count is
    unknown.
  - This is what `_subsystem` uses.
- **`compatible_with(other)`:** whether some choice of the axes that each
  `...` stands for makes the two match axis by axis, under
  `Axis.compatible_with`. Names of the systems are not compared. `None` is
  compatible with every system. Equality stays strict.

### B.4 Axes, and units

- **`Axis()`, with every field `None`, is "an axis about which nothing is
  known".** It is the placeholder that fills any position no description
  covers. Subclasses that set a field (`SpaceAxis` sets the type) are not
  unknown. Only a plain `Axis()` is.
- **`Axis.compatible_with(other)`:** every field set on both sides must be
  equal. A `None` field matches anything. The relation is symmetric but not
  transitive. It lets `[RAS axes, ...]` be compatible with a closed 4-D
  RAS+time system. (The first proposal named it `compatible`.)
- **`Axis.merge_with(other)`:** combines what both sides know, as an
  instance of the more derived class, and raises `ValueError` if they
  conflict. It is implemented but has **no consumer yet**.
- **`axes.R`, `L`, `A`, `P`, `S`, `I` are classes, not instances.** An axis
  is mutable, so a module-level instance would be shared by every system
  that took it. Build one where needed (`R()`, `R(unit="mm")`) and test with
  `isinstance`. `R` is the left-to-right axis, `A` posterior-to-anterior,
  `S` inferior-to-superior, and `L`, `P`, `I` their opposites.
- **Units (#115).**
  - `Unit("sample")` (`SampleUnit`) marks an *array* axis: its coordinates
    count samples. (Since #289 these are the index units, `Unit("index")`,
    `"voxel"` or `"pixel"` (`IndexUnit`), tested with `is_indexunit`.) `unit=None` means *unspecified*: nothing is claimed.
    These are three different things: a sample, a physical unit, and no
    claim.
  - Each unit class refuses the other kinds: `SpaceUnit("s")` and
    `SampleUnit("mm")` raise. `SpaceAxis.unit` accepts a space unit, the
    sample or `None`, and `TimeAxis.unit` a time unit, the sample or `None`.
  - `is_sampleunit` and `is_physicalunit` ask the two questions.
    `is_physicalunit(None)` is false.
  - The bridge never converts between a sample and a physical unit, since
    the size of a sample is not known.

### B.5 Physical systems and memory order

- **`PhysicalCoordinateSystem`** has axes measured in a physical unit or in
  an unspecified one (`None`), never in samples. It accepts open axes and
  `None` units, since neither claims anything non-physical, and refuses the
  sample unit. Voxel, pixel and array systems are not physical: their axes
  count samples. It is a base to inherit, not a dispatch target.
- **`RASmm`, `LPSmm` and `RSAmm`** are mm-only. They are dispatched only
  when every axis is in mm, and built by name they refuse any other unit,
  including `None`. RAS axes in another unit, or with no unit, build an
  `RASCoordinateSystem`.
- **`order: Optional[Literal["C", "F"]]` is declared on `CoordinateSystem`**,
  because bagof-magic registers a subclass only with polymorphic classes
  that have the field. Only array systems keep a non-None order, and any
  other system refuses one.
  - It selects the C- and F-ordered classes from every class above them:
    `CoordinateSystem(axes=<RAS axes>, order="F")` is an
    `FRASCoordinateSystem`.
  - A system the axes and the order do not make an array of
    (`RASmm(order="F")`) raises.
  - The memory order is not written on the axes, which is why it is a field
    of the system.

### B.6 Consumers

Every reader of an axis count or position was updated in the same change:

- `converters.py`, `Subspace → Affine`: a missing or open system raises
  `ConversionError` ("axis count unknown").
- `adaptors.py` and `utils.systems_disagree`: two systems disagree when they
  are not equal (both closed) or not `compatible_with` (either open). An
  open system compatible with its neighbour bridges as the identity, and an
  incompatible one raises `AdaptationError`: a bridge would have to
  reorder axes it cannot see. Embedding a smaller transform in a fuller
  space needs closed systems.
- `composers.py`, the discrete-axis check: `at(position)` reads the known
  positions of an open system, and an unknown `Axis()` is not discrete.
- **`meta._close_subspace`**: a composer closes an open subspace system from
  its neighbour. A subspace passes every axis it does not act on through, so
  a count on one side gives the count on the other. The open systems are
  closed with `expand`. A count the subspace cannot fit raises
  `CompositionError`. The composers that fold a subspace into an affine use
  it.
- `utils.axis_counts`, `get_ndim`, and the `separable`/`factor` readers:
  `ndim is None` means unknown, never a guess.
- **I/O writers** (NIfTI, OME-Zarr): no format can store `...`, so writers
  close an open system from the data shape with `expand`. The axes `...`
  stood for are written as any axis the format knows nothing about. A
  system that states more axes than the data has raises `WriterError`, and
  a closed system whose length disagrees with the data raises, as before.

### B.7 Smaller changes

- **`smartproperty` and `lazyproperty` have a single `unset=` option.** It
  says when a stored value reads as "not set", so that the getter computes
  it instead: `None` (the default), `"empty"` (an empty list, tuple, dict or
  set), a predicate `(value) -> bool`, or a tuple of them. The setter stores
  the value as given, whatever `unset` says. This replaced the earlier pair
  of options.

### Considered and rejected

- **Normalizing a `None` system to `CoordinateSystem(axes=[...])`.** A `None`
  system defers; an explicit one is kept (B.3). Treating them alike
  silently dropped what the user set. Only the `axes` field reads `None` as
  the default.
- **An accessor and a "both spellings" test** to keep `axes=None` and
  `axes=[...]` aligned. The field now stores one spelling, and plain
  equality is enough.
- **A plain `list` plus free functions** (`take`, `place`) for axes. The
  containers carry the operations, and the system methods are thin wrappers.
- **Module-level axis instances** (`R = RightAxis()`). Shared mutable state.
- **Open fixed-dimension systems.** The count is part of the class, so they
  refuse `...`.

---

## Sequencing

1. **#104 merged as it was.** Its helpers (`factor_sequence`, `restrict`,
   `embed`, the restrictors) are not exported, so merging committed to
   nothing public. PR 2 replaces them.
2. **#112 merged: open coordinate systems (Part B).**
   - `...` semantics, `AxisSequence`/`AxisTuple`/`AxisList`,
     `CoordinateSystem.ndim`/`expand`/`restrict`/`embed`/`compatible_with`,
     `Axis.compatible_with`/`merge_with`.
   - All the consumers listed in B.6, including `_close_subspace`.
   - `_subsystem` embeds with `ndim=None`, which fixes the
     `Subspace → Affine` crash.
   - Physical systems, `order`, and the `unset=` option (B.5, B.7).
3. **#115 merged: sample units (B.4)**, with bagof-magic 0.3 dispatch.
4. **PR 2, still to do: the operation API (Part A).**
   - `Projection` gains coordinate systems (using `CoordinateSystem.restrict`
     and `embed`) and a defined meaning for created axes (`0`, A.8).
   - Pair-rewrite simplification rules, with commutation rules per type
     and projection cancellation. These replace #104's restrictors.
   - `Transformation.restrict` and `project` return the lazy projection
     composition, and the transformation-level `embed` is removed.
   - `factor` becomes a public, pure rewrite built on `restrict`.
   - Factors declare their full-space systems.
   - `compute` becomes a single pass.
   - Reslice asks for `simplify="numeric"`.
   - The multi-block `SubspaceTransformation` and axis references by name
     (A.8, A.9), including a public reference-to-position function.
   - Checked by the parity sweep, the idempotence test, and the reslice
     timings.

## Decisions

**Part A (agreed, to do)**

1. Restriction takes no `policy` argument. (Now moot: restriction is
   `t.restrict(I, O)`, and policy belongs to `compute`/`simplify`.)
2. The registration decorators stay private.
3. Restriction follows `inverse`: `t.restrict(I, O)` returns the lazy
   composition `π_O @ t @ ι_I`, reduced by `compute` through analytic
   rewrite rules. Missing composers or simplifiers are added, not worked
   around.
4. Created axes are `0`, as the OME-NGFF 0.6 `projectAxis` spec defines.
5. `t.project(I, O)` (always defined; the section at `0` on a coupled
   block) and `t.restrict(I, O)` (raises on a coupled block) are separate
   methods.
6. `restrict` raises lazily, at `compute`. Like `inverse`, it only builds an
   expression.
7. `Projection.inverse()` raises when the projection drops axes.
8. A subset integer `mapAxis`, never valid in any 0.6 version, is refused on
   read.
9. `SubspaceTransformation` becomes the multi-block product (A.8,
   option 1), keeping its name and the pass-through extension.
10. Axis references may be positions or names everywhere (A.9), with the
    pass-through rule as written. A transformation that carries systems
    validates its references against them, whether names or positions.
11. Name mappings for `Permutation`/`Projection` are added now.

**Part B (implemented)**

12. `CoordinateSystem.axes` is never `None`. `axes=None` reads as the
    class's default (`[...]`, or the fixed axes). Equality is plain
    field-wise.
13. `system=None` is not `CoordinateSystem(axes=[...])`. A `None` endpoint
    defers or is derived. An explicit system is always kept.
14. `...` may appear anywhere, at most once. `ndim` is `None` for an open
    system. Fixed-dimension classes reject `...`, and `...` never counts
    towards the number of explicit axes.
15. Axes are `AxisTuple` (immutable, per-position typed; fixed-dimension
    systems) or `AxisList` (mutable; open-capable systems), sharing the
    `AxisSequence` API. Entries (`[i]`) and positions (`at`) are distinct.
16. An open axis list matches no dispatch predicate. `expand` re-dispatches.
17. Writers close an open system from the data shape (`expand`).
18. `Unit("sample")` marks array axes, and `None` is unspecified. Each unit
    class refuses the other kinds.
19. `PhysicalCoordinateSystem` accepts open axes and `None` units, and
    refuses the sample. `RASmm`, `LPSmm` and `RSAmm` are mm-only.
20. `order` is declared on `CoordinateSystem`. Only array systems keep one,
    and it selects the C- and F-ordered classes.
21. `axes.R`, `L`, `A`, `P`, `S`, `I` are classes.
22. `smartproperty` and `lazyproperty` take one `unset=` option.

## Open questions

Part B left two things for Part A:

1. **A public reference-to-position function.** A.9 needs to resolve an
   axis reference (a position or a name) to a position against a system.
   `AxisSequence.restrict` does it internally, while `index` and `[name]`
   give entries, which are positions only in a closed sequence (a name
   after `...` is a negative position). Either PR 2 makes the internal step
   public, or it resolves through `restrict`.
2. **`Axis.merge_with` has no consumer.** It was meant for combining a
   subspace's derived system with a declared neighbour. Today a declared
   system wins and is never merged. PR 2 either uses it (e.g. when factors
   declare full-space systems) or drops it.
