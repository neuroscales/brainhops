# The compute API and open coordinate systems

Part B of this memo, on open coordinate systems, is implemented in #112
and #115. Part A, on the public operations `compute`, `simplify`,
`factor`, `restrict` and `project`, is agreed but not implemented yet.

Part A defines a small set of public operations on transformations. Part B
lets a coordinate system state that some of its axes are unknown, which
several operations of Part A rely on. The guiding rule for both is that a
public operation needs a contract that makes sense on its own, can be
documented in a few sentences, and does exactly what it says without
silent surprises. Passing tests is not sufficient.

---

## Part A. Public operations

### A.1 Principles

- Each operation has one job. `simplify` rewrites a transformation into a
  cheaper equivalent and `factor` rewrites it into independent groups of
  axes, neither of them composing anything. `t.restrict(...)`, like
  `t.inverse()`, only describes its result lazily. `compute` is the only
  operation that composes.
- Every operation returns an equivalent transformation or raises. It never
  silently drops a piece, changes a value or approximates.
- One policy argument, with the same meaning everywhere, governs how much
  an operation may read parameter values. `"analytic"` reads only types
  and structure (axes, kinds, and the identity of lazy-inverse links),
  `"numeric"` may also read values for exact tests without tolerance, and
  `"none"` does nothing.
- When there is nothing to do, the same object is returned, and leaves
  that are not rewritten keep their identity, so that a transformation and
  its lazy inverse keep cancelling.
- Per-type rules are added by registration, through private decorators
  (`@simplifier`, `@composer`, and so on) on `bagof.dispatchers`
  functions. A missing rule is something to add, never a constraint on the
  design.

### A.2 `simplify(t, policy="analytic")`

`simplify` already exists as `Transformation.simplify`. Its contract is to
return a transformation equivalent to `t`, of the cheapest type that the
policy can prove, without composing: no matrix is multiplied, no field is
sampled and no lazy inverse is materialised. Under the analytic policy it
downcasts by structure and cancels a transformation that is adjacent to
its own lazy inverse. Under the numeric policy it also downcasts on exact
tests, so that an exactly diagonal `Affine` becomes a `Scaling`. It never
lengthens a sequence.

### A.3 `factor(t, policy="analytic")`

`factor` becomes public. It rewrites `t`, without composing, into the
axis-group normal form `[grid?, F_1, …, F_m, Π?]`, in which each
`F_i = SubspaceTransformation(chain_i, A_i, A_i)` acts on the group of
axes `A_i` and `chain_i` is `t` restricted to that group. A transformation
that does not split into at least two groups is returned unchanged. With
the multi-block `SubspaceTransformation` of A.8, the normal form becomes
`[grid?, SubspaceTransformation(blocks)]`, the reordering `Π` is absorbed
into `output_axes`, and restriction to whole blocks is structural.

The implementation from #104 changes in four ways. First, `factor` no
longer composes and loses its `mode` argument, so a group whose pieces
cannot be composed no longer prevents the rest from being factored.
Second, the partition follows the policy: the analytic policy reads the
structure of each type (diagonal scalings and translations, permutation
vectors, subspace axes, fully coupled fields and matrices), and only the
numeric policy reads the exact zero pattern of affine matrices, which #104
does today whatever the policy. Third, each factor declares full-space
coordinate systems, built by `factor` itself, because deriving them from
the inner transformation would yield an open system (B.1). Fourth, the
module `factor.py` will be renamed (for example to `factoring.py`), while
the `factor=` keyword of `compute` stays.

`factor` is idempotent (`factor(factor(t)) is factor(t)`), never
materialises a lazy inverse, and preserves the map. The reslice executor
reads the normal form and asks explicitly for
`compute(mode=Affine, factor=True, simplify="numeric")`, since resampling
reads every value anyway.

### A.4 Restriction as a lazy composition with projections

For a set of input axes `I` and output axes `O`, `O` is closed over `I` if
every output in `O` reads only inputs in `I`; this is what restriction
needs. The block `(I, O)` is decoupled if, in addition, no output outside
`O` reads an input in `I`; this is what factoring needs. For an affine
matrix `M`, closure means `M[O, ∁I] = 0`, and decoupling adds
`M[∁O, I] = 0`. For the map `(x, y) → (x, x + y)`, the block `({x}, {x})`
is closed but not decoupled, and its restriction, the identity on x, is
well defined.

`t.restrict(input_axes, output_axes)` returns the lazy composition
`π_O @ t @ ι_I`, where the `Projection` `ι_I` creates the input axes
outside `I` and the `Projection` `π_O` drops the output axes outside `O`.
No new type is needed, and the keywords match those of
`SubspaceTransformation`. `compute(mode=False, simplify="analytic")`
reduces the composition with two rewrites that cost nothing. Commutation,
`π ∘ T → T' ∘ π'`, cuts `T` down to the axes that `π` keeps and moves the
projection towards the input side. Cancellation, `π ∘ ι → Identity`,
applies when one projection exactly undoes the other. The reduction keeps
the type (`π ∘ Scaling` is a smaller `Scaling`), keeps lazy inverses
lazy, and returns `t` itself when the block covers all axes. Under a
composing mode, `compute` reduces the projections first and then composes
the rest.

This requires five additions: created axes with value `0` (settled in
A.8); a third kind of rule, the terminating pair rewrite; one commutation
rule per type, replacing the restrictors of #104; coordinate systems on
`Projection`, through `CoordinateSystem.restrict` and `embed`; and
composers for `Projection`. The contract is that
`t.restrict(I, O).compute()` is equivalent to `t` read from `I` to `O`
whenever `O` is closed over `I`, and that
`restrict(SubspaceTransformation(t, a, b), a, b)` reduces to `t`. Coupled
blocks are handled by `project` (A.8).

### A.5 No separate `embed`

The transformation-level `embed` of #104, with its `ni` and `no`
arguments, is removed. Embedding into a subspace is
`SubspaceTransformation(t, input_axes, output_axes)`, adding axes is
`Projection(created=...)`, and lowering to a plain `Affine` is the job of
`convert` and `simplify`, which need the full axis count. The methods
`CoordinateSystem.embed` and `AxisSequence.embed` are unrelated and stay.

### A.6 `compute(t, mode=True, *, simplify="analytic", factor=False)`

`compute` returns a transformation equivalent to `t` in which every run of
transformations that `mode` admits has been composed. It makes a single
pass:

1. Bridge every boundary at which adjacent systems disagree.
2. Flatten.
3. Apply `simplify(·, policy)`.
4. If `factor` is set, apply `factor(·, policy)`.
5. Compose once: outside the factors, compose the runs that `mode` admits;
   inside each factor, compose its chain recursively. Each product is
   simplified as it is produced, and a product that collapses to
   `Identity` is dropped, after which composition continues across the
   gap.

A fixed-point loop is unnecessary. In measurements at the time, every
bridge was inserted in the first round, no second round changed anything
over the test suite or over thousands of random chains, and the one case
in which a second round helped is handled locally by step 5. `compute` is
idempotent, which a test over the parity sweep and the test suite
enforces in place of an iteration cap, and `compute(t, factor=True)` never
fails where `compute(t)` succeeds.

### A.7 What stays internal

`compose` and `convert` stay internal, behind `compute` and
`Transformation.to`, and so do bridging and adaptation. The registration
decorators are private, because registration is how the library's own
classes plug in, not a public extension point.

### A.8 Alignment with OME-NGFF 0.6

#### `projectAxis` and `Projection`

The `projectAxis` transformation of OME-NGFF 0.6 has the structure of
`Projection(dropped, created)`: `droppedInputs` are the inputs it drops,
and `createdOutputs` are the outputs it adds, with value zero. Created
axes are therefore `0`, with no `fill` field, and a coupled block is the
section of `t` at `0`.

The specification also exposes a correctness issue. `Projection.inverse()`
currently swaps `dropped` and `created`, but a projection that drops an
axis has no inverse: creating the axis at `0` is only a right inverse
(`drop ∘ create = id`, while `create ∘ drop ≠ id`). It is decided that
`inverse()` raises whenever anything is dropped, and that the cancellation
of A.4 is one-sided: a drop after a creation of the same axes cancels, but
the reverse order does not.

#### `mapAxis` and `Permutation`

In OME-NGFF 0.6, `mapAxis` is strictly a permutation. Version 0.6.dev1
used an object of axis names that could drop or duplicate axes, but a
subset of integers was never allowed. The brainhops reader
(`io/transformations/zarr/_map.py`) nevertheless accepts a strictly
increasing subset of integers as a dropping `Projection`. Following the
rule of being liberal with forms that were once valid and strict with
forms that never were, it is decided that the reader raises on such a
subset, that the writer emits only permutations, and that `projectAxis` is
the spelling of a drop.

#### `byDimension`

A `byDimension` is a list of children, each with `inputAxes`,
`outputAxes` and a transformation. Every output axis belongs to exactly
one child, children may read any inputs in any order, and nothing passes
through implicitly. `SubspaceTransformation`, by contrast, is a single
child with an implicit in-order pass-through of the other axes, which is a
brainhops extension. A `byDimension` can be written as a sequence of
subspaces only when every child is square, and writing it back would then
require fragile pattern recognition.

It is decided to generalise `SubspaceTransformation` into the product: a
list of blocks `(transformation, input_axes, output_axes)` with the
semantics of the specification, plus the pass-through extension, by which
outputs that no block writes are fed from inputs that no block reads. This
lets a subspace act on x, y and z of a space whose other axes are unknown,
and today's single-block call remains valid. The writer closes the systems
and emits explicit identity children for the axes passed through. A
separate `ByDimension` class was rejected because it would overlap
`SubspaceTransformation`, and reading `byDimension` into a sequence was
rejected because it only handles square children and round-trips poorly.

The other 0.6 types (`identity`, `scale`, `translation`, `affine`,
`rotation`, `sequence`, `bijection`, `displacements` and `coordinates`)
are already covered. Version 0.6 has no `inverseOf`, so writers emit a
closed-form inverse or wrap it in a `bijection`.

#### `restrict` and `project`

Two methods with distinct contracts are clearer than a boolean flag that
changes the meaning of one method. `t.project(input_axes, output_axes)`
is always defined: a closed block reduces by commutation, and a coupled
block is computed as the section of `t` with the other inputs at `0`. The
name follows `projectAxis`. `t.restrict(...)` builds the same expression
with the guarantee that the block is closed. It is decided that it raises
`RestrictionError` lazily, at `compute`, when commutation cannot proceed.

### A.9 Axis references: positions or names

OME-NGFF refers to axes by position, because its transformations sit
between systems that list every axis. In brainhops, systems may be missing
or open, so a position in a space of unknown width is a guess, notably
when a pass-through has to decide which input to drop. Every axis
reference (block axes, `Permutation`, `Projection`, and the axes of
`restrict` and `project`) will therefore accept a position or a name.
References are stored as given and resolved lazily against the system on
the relevant side. A name resolves when the system has exactly one axis
with that name, and never matches the axes that `...` stands for. Names
survive reordering and bridging, whereas positions do not. Writers resolve
every reference to a position after closing the systems.

It is decided that a transformation that carries a system on a side, even
an explicit open one, validates the references on that side at
construction and whenever the system is replaced. A name must exist and be
unique, and a position must be in range in a closed system, while in an
open system every position is valid. `Permutation` and `Projection` will
also accept mappings of names, such as `Permutation({"x": "j", "y": "i"})`.

The pass-through rule of a multi-block `SubspaceTransformation` is:

1. If every output is written by a block, inputs that no block reads are
   dropped.
2. Otherwise, if the axes are named, each unwritten output is fed from the
   unread input with the same name. Other unread inputs are dropped, and
   an unwritten output without a matching input raises.
3. Otherwise, unwritten outputs are paired in order with unread inputs
   when their numbers are equal, and the rule raises if they are not. It
   never chooses axes to drop.

The unknown axes behind `...` pass through one to one, and no axis is ever
dropped from inside `...`.

---

## Part B. Open coordinate systems

Part B describes the code as implemented in #112 and #115.

### B.1 The problem

A `SubspaceTransformation` without full-space systems used to derive them
from its inner system by placing the inner axes at their positions and
stopping at `max(axes) + 1`. The positions were right but the count was
invented, so a 1-D scaling embedded in a 3-D chain reported a 1-axis
system, converted to a 1x2 matrix, and failed to compose. The count is
genuinely unknown in such cases, but no type could say so, and a `None`
system would discard the known positions that the discrete-axis check
uses. `_subsystem` now returns the declared system, or else derives an
open one with `CoordinateSystem.embed`.

### B.2 Axis containers

`AxisSequence`, defined in `datamodel/_axes_list.py` and exported from
`brainhops.datamodel.systems`, holds `Axis` items and at most one `...`,
which stands for zero or more unknown axes and never counts as an axis.
`ndim` is the number of axes, or `None` when the sequence is open. Two
concrete containers share this interface: the immutable `AxisTuple`,
typed per position and used by fixed-dimension systems, and the mutable
`AxisList`, used by systems that may be open.

Entries and positions are kept distinct. `len()`, iteration, `==`, `[i]`
and `index` are about entries, including `...`, whereas `ndim`,
`at(position)`, `expand`, `restrict`, `embed` and `compatible_with` are
about positions in the space. In `[x, ..., t]`, entry 2 is `t`, while
position 2 is one of the axes that `...` stands for, which `at` returns as
an unknown `Axis()`. Lookup by name (`axes["x"]`, `"x" in axes`) reads only
the explicit axes.

### B.3 Coordinate systems

`axes` is never `None`. `axes=None` selects the class default, which is
`[...]` for classes that may be open, and equality is plain field-wise
equality, so `CoordinateSystem() == CoordinateSystem(axes=[...])`. The
classes that may be open (`CoordinateSystem`, `SpatialCoordinateSystem`,
`PhysicalCoordinateSystem`, `ArrayCoordinateSystem` and their ordered
variants) accept `...` anywhere, at most once. Fixed-dimension classes
refuse it, because their count is part of the class.

A `None` endpoint is not an open system. It means that there is no
system, which defers to the context or is derived (a subspace from its
inner transformation, an `Inverse` from its forward transformation, a
`Sequence` from its members). An explicit system, even
`CoordinateSystem()`, is always kept as given. Treating an endpoint that
says nothing as `None` was tried and reverted, because it discarded
systems set by users. Code that needs the axes of a possibly missing
endpoint uses `get_axes(system)`.

Calling `CoordinateSystem(...)` dispatches to the most specific class that
its axes describe. An open list of axes matches no predicate on the axes,
because every such predicate makes a claim about all of them, so
`CoordinateSystem(axes=[R(), A(), S(), ...])` is not a
`RASCoordinateSystem`. `expand(ndim)` closes a system and re-dispatches
it. `restrict(refs)` returns the system of the selected axes,
`embed(positions, ndim=None)` is its inverse, and `compatible_with(other)`
is true when some choice of the axes behind `...` makes the two systems
match axis by axis. Equality stays strict.

### B.4 Axes and units

An `Axis()` with every field `None` is an axis about which nothing is
known. `Axis.compatible_with` treats `None` fields as matching anything,
and `Axis.merge_with` combines two axes; the latter has no consumer yet.
`axes.R`, `L`, `A`, `P`, `S` and `I` are classes rather than instances,
because axes are mutable and module-level instances would be shared
state.

Array axes are marked with an index unit, `Unit("index")`, `"voxel"` or
`"pixel"`, which builds an `IndexUnit` and is tested with `is_indexunit`.
The unit was introduced in #115 as a sample unit and renamed in #289.
`unit=None` claims nothing, so an index unit, a physical unit and `None`
are three different statements. Each unit class refuses units of other
kinds, and a bridge never converts between index and physical units,
since the size of a sample is unknown.

### B.5 Physical systems and memory order

`PhysicalCoordinateSystem` accepts physical or unspecified units and open
axes, and refuses index units. `RASmm`, `LPSmm` and `RSAmm` are
millimetre-only, and RAS axes in another unit, or without one, build a
`RASCoordinateSystem`. The memory order, `order: Optional[Literal["C",
"F"]]`, is a field of `CoordinateSystem`, because bagof-magic registers a
subclass only with classes that have the field and because the order is
not recorded on the axes. Only array systems accept a non-`None` order.

### B.6 Consumers

Converting a subspace with a missing or open system to an `Affine` raises
`ConversionError`. Adjacent systems disagree when they are unequal and
closed, or incompatible when either is open; a compatible open neighbour
is bridged as the identity, and an incompatible one raises
`AdaptationError`. A composer closes the open system of a subspace from
its neighbour (`meta._close_subspace`), and raises `CompositionError` when
the count does not fit. Code that reads axis counts treats `None` as
unknown and never guesses. Since no file format stores `...`, the NIfTI
and OME-Zarr writers close open systems from the shape of the data, and
raise `WriterError` when more axes are stated than the data has.

### B.7 Smaller changes

`smartproperty` and `lazyproperty` take a single `unset=` option, which
says when a stored value reads as "not set": `None` (the default),
`"empty"`, a predicate, or a tuple of these.

### Considered and rejected

- Normalising a `None` system to `CoordinateSystem(axes=[...])` was
  rejected because it silently dropped systems that users had set.
- Plain lists manipulated by free functions `take` and `place` were
  rejected in favour of the axis containers of B.2.
- Module-level axis instances were rejected because they would be shared
  mutable state.
- Open fixed-dimension systems were rejected because the number of axes
  is part of their class.

---

## Remaining work

Part A will land in one pull request. It gives `Projection` coordinate
systems and creation at `0`; replaces the restrictors of #104 with pair
rewrite rules; adds `Transformation.restrict` and `project` and removes
the transformation-level `embed`; makes `factor` a public pure rewrite and
`compute` a single pass; makes reslice ask for `simplify="numeric"`; and
adds the multi-block `SubspaceTransformation` and axis references by name.
It will be checked by the parity sweep, the idempotence test and reslice
timings.

## Decisions

Part A, agreed and not yet implemented:

1. Restriction takes no `policy`; the policy belongs to `compute` and
   `simplify`.
2. The registration decorators are private.
3. Restriction follows `inverse`: it is a lazy composition with
   projections, reduced by `compute` through analytic rewrite rules.
4. Created axes are `0`, as in `projectAxis`.
5. `project`, which is always defined, and `restrict`, which raises on a
   coupled block, are separate methods.
6. `restrict` raises lazily, at `compute`.
7. `Projection.inverse()` raises when the projection drops axes.
8. A subset of integers in `mapAxis` is refused on reading.
9. `SubspaceTransformation` becomes a multi-block product and keeps its
   name and its pass-through.
10. Axes are referred to by position or by name everywhere, and a
    transformation that carries systems validates its references.
11. `Permutation` and `Projection` accept mappings of names.

Part B, implemented:

12. `axes` is never `None`, and equality is field-wise.
13. A `None` system is not an open system.
14. `...` appears at most once, anywhere; `ndim` is `None` when open; and
    fixed-dimension systems reject `...`.
15. `AxisTuple` and `AxisList` share `AxisSequence`, which distinguishes
    entries from positions.
16. An open list of axes matches no dispatch predicate, and `expand`
    re-dispatches.
17. Writers close open systems with `expand`.
18. Index units mark array axes, and `None` means unspecified.
19. `PhysicalCoordinateSystem` refuses index units, and `RASmm`, `LPSmm`
    and `RSAmm` are millimetre-only.
20. `order` is declared on `CoordinateSystem`.
21. `axes.R`, `L`, `A`, `P`, `S` and `I` are classes.
22. `smartproperty` and `lazyproperty` take a single `unset=` option.

## Open questions

1. A.9 needs a public function that maps an axis reference to a position.
   `AxisSequence.restrict` performs that step internally, while `index` and
   `[name]` return entries, which are positions only in closed sequences.
2. `Axis.merge_with` has no consumer. Part A will either use it, for
   example when factors declare full-space systems, or remove it.
