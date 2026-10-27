# The compute API and open coordinate systems

Part B of this memo, on open coordinate systems, is implemented in #112
and #115. Part A, on the public operations `compute`, `simplify`,
`factor` and `restrict`, is agreed. Its first part, which lets `compute`
read, compose and cancel projections, is implemented in #570, and the
values of created axes, the free axes of `restrict`, the reduction of
projections in `simplify` and the two ends of the normal form of
`factor` are implemented for #582. The rest is not implemented yet.

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
  cheaper equivalent without composing anything, and `factor` rewrites it
  into independent groups of axes. `factor` composes nothing by default,
  and its `compute=` mode composes inside each group, as a convenience
  that is equivalent to a call of `compute` with `factor=True`.
  `t.restrict(...)`, like `t.inverse()`, only describes its result
  lazily. `compute` is the operation that composes.
- Every operation returns an equivalent transformation or raises. It never
  silently drops a piece, changes a value or approximates.
- One policy argument, with the same meaning everywhere, governs how
  much an operation may read parameter values. `"analytic"` reads only
  types and structure (axes, kinds, and the identity of lazy-inverse
  links), `"numeric"` may also read values for exact tests without
  tolerance, and `"none"` does nothing. A bare `"numeric"` reads the
  values of matrix transformations only, because reading a dense field
  costs a scan of the whole array. The table `{None: "numeric"}` extends
  it to every transformation, fields included (#584).
- When there is nothing to do, the same object is returned, and leaves
  that are not rewritten keep their identity, so that a transformation and
  its lazy inverse keep cancelling.
- Per-type rules are added by registration, through private decorators
  (`@simplifier`, `@composer`, and so on) on `bagof.dispatchers`
  functions. A missing rule is something to add, never a constraint on the
  design.

### A.2 `simplify(t, policy="analytic")`

`simplify` already exists as `Transformation.simplify`. Its contract is
to return a transformation equivalent to `t`, of the cheapest type that
the policy can prove, without composing: no matrix is multiplied and no
field is sampled. Under the analytic policy it downcasts by structure
and cancels a transformation that is adjacent to its own lazy inverse.
Under the numeric policy it also downcasts on exact tests, so that an
exactly diagonal `Affine` becomes a `Scaling`. It never lengthens a
sequence.

`simplify` also removes the axes that a chain does not need (A.4), which
requires knowing which outputs of each element read which inputs. A lazy
inverse is never materialised to answer a question about its type,
because the forward transformation answers it exactly: the sets that a
downcast targets, such as scalings, rotations and permutations, are
closed under inversion. The numeric policy may read the coefficients of
the lazy inverse of a matrix transformation to find which of them are
exactly zero. The lazy inverse then stays in place, and it is replaced
by a concrete transformation only when it is cut down to fewer axes,
after the pass that cancels each transformation against its own lazy
inverse, so that no cancellation is lost.

### A.3 `factor(t, policy="analytic", *, compute=False)`

`factor` is public. It takes the policy of `simplify`, with the same
values and the same meaning, and rewrites `t` into the
axis-group normal form `[grid?, δ?, F_1, …, F_m, Π?, κ?]`, in which each
`F_i = SubspaceTransformation(chain_i, A_i, A_i)` acts on the group of
axes `A_i` and `chain_i` is `t` restricted to that group. The two
optional ends come from the reduction of A.4. The projection `δ` drops
the inputs that no output reads, and the projection `κ` creates the
outputs that read no input, at their constant values. A chain whose two
ends have different numbers of axes can therefore be factored. A
transformation that does not split into at least two groups and has
neither end is returned unchanged. By default nothing is composed, so
each factor holds the pieces of its group. The `compute=` mode composes
inside each group, never across groups, and is a convenience: a call of
`compute` with `factor=True` and the same mode and policy gives the same
factors. With the multi-block
`SubspaceTransformation` of A.8, the normal form becomes
`[grid?, δ?, SubspaceTransformation(blocks), κ?]`, the reordering `Π` is
absorbed into `output_axes`, and restriction to whole blocks is
structural.

The two ends let the resampler avoid work along whole axes. A 3D mask
resampled onto the grid of a 4D series `(x, y, z, t)` gives
`[grid, δ_t, F_xyz]`, so the resampler interpolates once in 3D and
repeats the result along `t`, instead of interpolating in 4D. The
axis-aligned slice `k = 40` of a volume, shown on a 2D grid, gives
`[grid, F_ij, κ_k]`, so the resampler reads that slice and interpolates
in 2D. Volume 4 of a series with one affine per volume (A.10) is
reduced in the same way: the embedding of `t` stays before each element
that reads it, a composing `compute` folds the value of `t` into that
element, and `κ_t` selects the volume.

The groups are the connected components of the graph that links two axes
when one reads the other, whatever the direction of the dependence. The
pass reads the patterns with the pattern readers of A.4, so the
partition follows the policy. The analytic policy reads the structure of
each type (diagonal scalings and translations, permutation vectors,
subspace axes, fully coupled fields and matrices), and a matrix couples
all its axes. The numeric policy also reads the exact zeros of matrices,
so that an affine with independent axes splits. For the lazy inverse of
a matrix transformation, the numeric policy reads the exact zeros of the
coefficients of the inverse, and otherwise the reader returns the
closure of the pattern of the forward transformation. The closure links
every pair of axes that the inverse links, so a group is never split
wrongly, and no inverse is computed for it.

Two changes remain. Each factor will declare full-space coordinate
systems, built by `factor` itself, because deriving them from the inner
transformation would yield an open system (B.1). The module `factor.py`
will be renamed (for example to `factoring.py`), while the `factor=`
keyword of `compute` stays.

`factor` is idempotent (`factor(factor(t)) is factor(t)`), never
computes a lazy inverse only under the numeric policy to read its
coefficients, and preserves the map. The reslice executor reads the
normal form and asks explicitly for
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

`t.restrict(input_axes, output_axes, at=None)` returns the lazy
composition `π_O @ t @ ι_I`, where the `Projection` `ι_I` creates the
input axes outside `I` and the `Projection` `π_O` drops the output axes
outside `O`. A created axis is placed at the value that `at` gives it,
and it is free when `at` does not name it, which means that the result
must not depend on it. No new type is needed, and the keywords match
those of `SubspaceTransformation`.

`simplify` reduces the projections, so that `compute`, which simplifies
before it composes (A.6), never builds a transformation over axes that
are dropped afterwards. The reduction needs no rule per pair of types.
Each element of a flattened chain has a dependency pattern, which says
which of its outputs read which of its inputs. A projection is an
element like any other: the axes that it creates are outputs that read
no input, and the axes that it drops are inputs that no output reads.
Two passes read these patterns.

1. The backward pass starts from the outputs that the end of the chain
   keeps. At each element, going towards the input, an input is needed
   when a needed output reads it. An input that is not needed is dropped
   as early as possible.
2. The forward pass follows each created axis, and its value, from its
   embedding towards the output, through the elements that carry it
   through unchanged, such as a permutation, an identity or a subspace
   that does not act on it. An output that an element creates at a
   value, such as a constant row of a matrix under the numeric policy,
   becomes a created axis too. When a needed output reads a created
   axis, whether its value is a number or NaN, the embedding stays
   immediately before that element, and a composing `compute` folds the
   value into the element with the matrix product in which zero cancels
   NaN (see below). `simplify` computes no section: it multiplies no
   matrix and only copies the values of created axes.

Each element is then cut down to its needed outputs and to the inputs
that they read. A cut only requires the block to be closed, which the
backward pass guarantees. A decoupled block is also closed, so the
factor pass cuts its groups with the same restrictors, and the
restrictors of decoupled blocks of #104 are removed. An embedding followed by the projection that drops the
same axes cancels, and the pair rule of #570 merges two projections that
end up next to each other. The reduction keeps the type (a cut `Scaling`
is a smaller `Scaling`), and it returns `t` itself when nothing is
dropped. Uncoupling that appears only in a product, such as an affine
that adds `y` to `x` followed by another that subtracts it, is found by
the same passes when `compute` simplifies each product as it is produced
(A.6).

The knowledge about each type is registered through two dispatched
functions. A pattern reader returns the dependency pattern of a
transformation under a policy, and a restrictor cuts a transformation
down to a closed block under a policy, or declines. An element whose pattern cannot be
read, such as a transformation of an unknown type, counts as fully
coupled. An element whose restrictor declines keeps all its axes, so
that a drop happens after it and an embedding stays before it. A
projection that stays next to an element in this way is multiplied into
it as a matrix by a composing `compute`, through the composers of #570.

The policy decides what a pattern shows. The analytic policy reads the
structure of each type, and the numeric policy also reads the exact
zeros of matrices. A lazy inverse needs the pattern of its own map, in
which an output reads an input, and the transpose of the pattern of its
forward transformation does not give it (A.3). Under the analytic
policy, the pattern of the inverse of a matrix is bounded by the
reflexive and transitive closure of the forward pattern. By the
Cayley-Hamilton theorem, the inverse of a matrix is a polynomial in that
matrix, so an output of the inverse can read an input only along a path
of the forward pattern. Under the numeric policy, the exact zeros of the
coefficients of the inverse are read, for matrix transformations only
(A.2). Rounding can leave a residue where the true coefficient is zero,
which keeps an axis that could be dropped. It can also, more rarely,
round a small coefficient to an exact zero, which is the same risk that
the matrix product of free axes takes (see below).

A pattern does not depend on how the chain was built. An affine whose
linear part has a zero column reads the same as an affine after a
projection that drops that input, and an affine whose linear part has a
zero row reads the same as an embedding of a constant output. The passes
therefore also split such affines. This recovers the structure of a
projection after it has been multiplied into a matrix, and that of a
rectangular affine, which OME-NGFF 0.6 allows. The passes run only when
there may be something to drop: when some element has different numbers
of input and output axes, or, under the numeric policy, when the linear
part of a matrix transformation has a zero row or a zero column.

`simplify` never lengthens a chain (A.2), so a reduction that would add
an element is not applied by `simplify`. This happens when the first
element does not read an input, or when the last element has a constant
output, and no projection is there to absorb the drop or the creation.
Inside a chain, a drop is absorbed by the element before it, which is
cut down to fewer outputs, and a creation by the element after it, which
passes it through unchanged. The factor pass runs the same reduction,
after the pairs that cancel at no cost have collapsed, and allows it to
add the two ends of its normal form and nothing else. When a lazy inverse is cut, the
pattern of the forward transformation under the policy decides whether
the cut can stay lazy. When the forward map is block diagonal or block
triangular with respect to the block, the cut is the lazy inverse of
the cut of the forward transformation. Otherwise, under the numeric
policy, the inverse of a matrix transformation is computed and then
cut, and under the analytic policy the restrictor declines, so that the
inverse is never computed.

A field is read as fully coupled, and its restrictor declines for now,
because a field cannot yet hold a number of components different from
the dimension of its grid (#588). Two reductions therefore wait for that
work. A drop after a field does not cut the field, so the field computes
outputs that are discarded afterwards. An embedding before a field is
not absorbed into the field, so the resampler interpolates the field at
the points of the section instead of reading one slice of it. Both
results are correct. A field inside a subspace is not affected, because
its pattern shows that it carries the other axes through unchanged.

A free axis is a created axis whose value is NaN (see below). A kept
output that reads a free axis has no defined value, and `compute` raises
`RestrictionError` (decision 6). `simplify` never raises. A structural
pattern can claim a dependence that the values cancel, so `simplify`
drops an axis only when it can prove that nothing reads it, and
otherwise it leaves the embedding in place. The outermost composing
`compute` performs the test, and the NaN carries it through the matrix
products. Composers call `compute` on partial products, in which a free
axis may still reach an output that a later projection drops, so only
the outermost call checks its result.

`Projection` has a field `at`, which holds one value per created axis.
Its default is `0`, the value of `projectAxis`. A number places the
created axis at that value, which is the same map as the embedding at `0`
followed by a translation, and NaN makes the axis free. In the matrix of
a projection, `at` is the translation of the rows of the created axes, so
NaN appears only in the translation column, and every product keeps it
there. Matrix products let an exact zero coefficient cancel a NaN:
`0 · NaN = 0`, while `c · NaN = NaN` for `c ≠ 0` and `NaN + x = NaN`.
An output that truly depends on a free axis is reached by at least one
path of non-zero coefficients, so it always ends with a NaN, and the
result is never wrong. A product can still keep a NaN that two paths
would cancel exactly, or that a coefficient left by rounding lets
through, and `restrict` then refuses a block that is in fact closed.

This requires the following additions: the field `at` of `Projection`
and the matrix product that lets zero cancel NaN; the pattern readers
and the restrictors of closed blocks, which replace the restrictors of
#104, and the two passes in `simplify`; coordinate systems on
`Projection`, through `CoordinateSystem.restrict` and `embed`; and
composers for `Projection`, of which the last two are implemented in
#570. The contract is that `t.restrict(I, O, at).compute()` is
equivalent to `t` read from `I` to `O` with the inputs that `at` names
at their values, that it raises when an output in `O` reads an input
outside `I` that `at` does not name, and that
`restrict(SubspaceTransformation(t, a, b), a, b)` reduces to `t`.

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
3. Apply `simplify(·, policy)`, which also drops the axes that are not
   needed (A.4).
4. If `factor` is set, apply `factor(·, policy)`, which groups what the
   reduction leaves and has no pass of its own.
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
and `createdOutputs` are the outputs it adds, with value zero. The
default of `Projection.at` is therefore `0`. The OME-NGFF writer of
brainhops writes a projection through its affine matrix, whose
translation holds the values of `at`, and it refuses a free axis, which
has no value to write and never survives `compute` (A.4).

The specification also exposes a correctness issue. Swapping `dropped`
and `created` does not invert a projection that drops an axis, which has
no inverse: creating the axis at `0` is only a right inverse
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

#### `restrict` with fixed and free axes

A single method, `restrict`, removes axes, and its `at` argument tells
apart the two reasons for removing an input. When the caller means that
the kept outputs do not depend on the input, the input is left out of
`at`, so that it is free and `compute` refuses a block that is not
closed. When the caller means the transformation at a value of the
input, such as one volume of a batch (A.10) or one slice of a volume,
`at` gives that value, and the result is the section of `t` there. A
flag such as `allow_coupled=True` was rejected, because it would supply
the value implicitly, as `0`, which is the silent behaviour that the
free axes prevent, and because the caller could not choose the value. A
separate method `project`, with removed inputs at `0` by default, was
rejected for the same reason, and because "projection" also names the
creation of axes, whereas `restrict` only removes them.

For an affine of `(x, y, t)` restricted to `x` and `y` on both sides:

| Rows of the affine                         | `at` omitted        | `at={"t": 4}`       |
| ------------------------------------------ | ------------------- | ------------------- |
| `x' = 2x + 1`, `y' = 3y + 2`, `t' = t + 5` | `[[2,0,1],[0,3,2]]` | `[[2,0,1],[0,3,2]]` |
| `x' = 2x + t/2 + 1`, `y' = 3y + 2`         | `RestrictionError`  | `[[2,0,3],[0,3,2]]` |
| `x' = 2x + 1`, `y' = 3y + 2`, `t' = t + y` | `[[2,0,1],[0,3,2]]` | `[[2,0,1],[0,3,2]]` |

In the second row, `x'` reads `t`. In the third row, only the output
`t'`, which is dropped, reads a kept input.

### A.9 Axis references: positions or names

OME-NGFF refers to axes by position, because its transformations sit
between systems that list every axis. In brainhops, systems may be missing
or open, so a position in a space of unknown width is a guess, notably
when a pass-through has to decide which input to drop. Every axis
reference (block axes, `Permutation`, `Projection`, and the axes of
`restrict`) will therefore accept a position or a name.
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

### A.10 Batched transformations

Some transformations are a family of transformations indexed by an axis,
such as one spatial affine per volume of a time series, as estimated by
motion correction, or the batches of transforms that X5 and ITK files
store (#251). Batched transformations are not implemented yet (#576), but
the operations of Part A are defined so that they treat them
consistently.

A batch axis is an axis that a transformation carries through unchanged,
while its other outputs may depend on it. With one affine per volume, the
spatial outputs are `A(t) x + b(t)` and the time output is `t`, so time
is a batch axis. With slice timing, the time output is `t + δ(z)`, so
time is not a batch axis, and the spatial outputs do not depend on `t`.

The following rules hold for batched transformations:

- Restricting one affine per volume to its spatial axes, with time left
  free, raises when the result is computed, because the spatial outputs
  read `t` and no single spatial map exists. Restricting slice timing to
  its spatial axes is defined, because the outputs that are kept do not
  read `t`, although the time output, which is dropped, reads `z`.
- The transformation of one element is the restriction with `at` set on
  the batch axis, as in `t.restrict(SpaceAxis, SpaceAxis, at={"t": k})`.
- A batched transformation reports that its other outputs depend on the
  batch axis. The rules of `compute` and the factor pass read which
  outputs depend on which inputs, and a batched affine that looked like
  a plain affine would be cut without an error.
- A batch axis is discrete. Along a discrete axis, the resampler already
  gathers whole samples and refuses a map that would interpolate between
  two of them, so it selects elements of the batch and never blends them.
  Re-expressing vectors in another frame already evaluates a frame at
  every position of such an axis without differentiating along it, which
  gives one Jacobian per element.

How a batched transformation is stored, for example as an affine whose
matrix has a leading batch dimension, is decided when it is implemented.

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

Entries and positions are kept distinct. `len()`, iteration, `==` and
`[i]` are about entries, including `...`, whereas `ndim`,
`at(position)`, `expand`, `restrict`, `embed` and `compatible_with` are
about positions in the space. In `[x, ..., t]`, entry 2 is `t`, while
position 2 is one of the axes that `...` stands for, which `at` returns as
an unknown `Axis()`. Lookup by name (`axes["x"]`, `"x" in axes`) reads only
the explicit axes.

`index` and `indices` find the axes that match a query, which is a name, a
template axis, an axis class, a dictionary of axis fields or a function
(#574). They return numbers that are entries and positions at once: an
axis before `...` is counted from the start and an axis after `...` from
the end, so that in `[x, ..., t]` the index of `t` is `-1`, and `axes[-1]`
and `axes.at(-1)` are both `t`. `restrict` accepts the same queries,
alone or among positions and names.

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

`smartproperty` computes its value when the stored value is `None`, and
has no option to read any other value as "not set". A property over an
inherited field whose default is an empty container takes the field
over under a private name with `Inherit("x", default=None)`, so that an
empty container is a value. `lazyproperty` computes its value when
nothing is stored and caches it. Every setter of either
property deletes the cached value of the property before it runs, and
the cached values named by `invalidates=` after it runs.

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

Part A lands in several pull requests. The first one, #570, wires
`Projection` into `compute`:

- created axes are `0`, and a projection that drops axes has no
  inverse;
- a projection derives the system on one side from a closed system on
  the other side;
- a projection whose number of axes is known converts to `Linear` and
  `Affine`, so its kind is linear, as well as surjective or injective;
- composers combine a projection with an affine-like neighbour, or with
  a subspace that wraps an affine, and take the number of axes from that
  neighbour when the projection does not state it;
- a pair rule merges two adjacent projections into one, so that a drop
  after the creation of the same axes cancels.


The next pull requests, in order (#582):

1. add the field `at` of `Projection`, the matrix product that lets
   zero cancel NaN and the check of `compute` for free axes, and make the
   OME-NGFF writer refuse a free axis, which is implemented;
2. add the pattern readers, the restrictors of closed blocks and the
   two passes of A.4 in `simplify`, add the two ends of the normal form
   to the factor pass, and remove the restrictors of #104, which is
   implemented. The factor pass reads its patterns with the pattern
   readers, so that its partition follows the policy, and cuts its
   groups with the restrictors of closed blocks. `factor` takes the
   policy of `simplify`, and reslice asks for the numeric policy;
3. add `Transformation.restrict`, which accepts the axis queries of #574,
   and remove the transformation-level `embed`.

The remaining work makes `compute` a single pass, and adds the
multi-block `SubspaceTransformation` and axis references by name. It
will be checked by the parity sweep, the idempotence test and reslice
timings. Cutting a field down to some of its own outputs, and taking the
section of a field, need fields whose number of components differs from
the dimension of their grid (#588). Until then, a projection that would
cut a field stays next to it. Batched transformations (A.10) come after
Part A, together with the readers and writers of the formats that
store them (#576).

## Decisions

Part A, agreed. Decisions 4 and 7 are implemented in #570, and the
others are not implemented yet:

1. Restriction takes no `policy`; the policy belongs to `compute` and
   `simplify`.
2. The registration decorators are private.
3. Restriction follows `inverse`: it is a lazy composition with
   projections, reduced by `simplify` and therefore by `compute`.
4. Created axes are `0`, as in `projectAxis`.
5. `restrict` is the only method that removes axes. Its `at` argument
   fixes removed inputs at values, and an input that `at` does not name
   is free.
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
22. `smartproperty` computes its value when the stored value is `None`,
    and every setter of a cached property deletes the cached value of
    that property.
23. `index` and `indices` return numbers that are entries and positions
    at once, counted from the end after `...`, and `index`, `indices`
    and `restrict` accept queries (#574).

Part A, agreed with `restrict` and batched transformations. Decisions
24, 25 and 26 are implemented (#582), and the others are not implemented
yet:

24. `Projection.at` gives the value of each created axis. It is `0` by
    default, and NaN makes the axis free.
25. Matrix products let an exact zero coefficient cancel a NaN.
26. `simplify` reduces projections with two passes over the dependency
    patterns of the elements of a chain. A backward pass drops the
    inputs that no needed output reads, and a forward pass carries
    created axes and their values. A projection that cannot be reduced
    is composed as a matrix by `compute`.
27. A batch axis is an axis that a transformation carries through
    unchanged while its other outputs may depend on it, and a batched
    transformation reports that dependence.
28. One element of a batched transformation is the restriction with `at`
    set on the batch axis.

Part A, agreed with the reduction of projections. Decisions 29 to 33
and 35 to 37 are implemented (#582), and decision 34 is implemented
(#587):

29. The knowledge about each type is registered through two dispatched
    functions, a pattern reader and a restrictor of closed blocks, which
    both take the policy, and there are no commutation rules for pairs
    of types.
30. The normal form of `factor` has two optional ends, a projection that
    drops the unread inputs and a projection that creates the constant
    outputs.
31. `simplify` never raises. It drops an axis only when it can prove
    that nothing reads it, and only the outermost composing `compute`
    raises for a free axis.
32. The passes run only when an element has different numbers of input
    and output axes or, under the numeric policy, when a matrix has a
    zero row or column, and they split such affines as they would
    projections.
33. Under the analytic policy, the pattern of the lazy inverse of a
    matrix is bounded by the closure of the forward pattern, and the
    inverse is never computed. Under the numeric policy, the
    coefficients of the inverse of a matrix transformation may be read,
    and the inverse is materialised only to be cut, after the
    cancellation pass.
34. A bare `"numeric"` policy reads the values of matrix transformations
    only, and `{None: "numeric"}` reads every transformation (#584).
35. `simplify` computes no section. A created axis is carried only
    through elements that pass it through unchanged, and an embedding
    that an element reads stays immediately before that element, for a
    composing `compute` to fold in.
36. The restrictors of decoupled blocks of #104 are removed. The factor
    pass reads its patterns with the pattern readers and cuts its groups
    with the restrictors of closed blocks.
37. `factor` takes the policy of `simplify`, and its partition follows
    that policy. It keeps its `compute=` mode, which composes inside the
    groups as a convenience equivalent to a call of `compute` with
    `factor=True`. The reslice executor asks for the numeric policy.

## Open questions

1. `Axis.merge_with` has no consumer. Part A will either use it, for
   example when factors declare full-space systems, or remove it.
