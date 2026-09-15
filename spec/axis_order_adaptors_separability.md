# Axis order, adaptors, and separability

Status: draft for maintainer review. Decisions marked **[DECISION]** need a
maintainer sign-off before implementation. This document audits the current
code, then specifies a unified design.

## 0. Thesis: three requests, one mechanism

Three separate roadmap items turn out to be facets of a single missing piece,
the **coordinate-system adaptor**:

- **Axis-order reconciliation** (OME/zarr C-order `(t, c, z, y, x)` versus
  nibabel F-order `(x, y, z, t, c)`) is the adaptor's *same axes, different
  order* case. It produces a `Permutation`.
- **Adaptor audit (#15)** is the adaptor itself.
- **Separability ("by-dimension" transforms)** is the adaptor's *per-dimension*
  case, plus a detection pass that recognises when any transform factors into
  independent per-axis work.

The design below builds the adaptor once and lets the other two fall out of it.

**Related tracking issues.** This spec is the design deliverable for the second
and third steps of #7 ("compute() on sequences": compatible compose is done,
then adaptor bridges, then separability), for #10 (bridge non-compatible
input/output spaces) and #11 (separable transforms). It supersedes the ad-hoc
direction of PR #15 (james/adaptor1), the existing partial adaptor that handles
only permutation, direction, and scale. It also touches #54 (collapsing the
axes/coordinate-system subclasses), whose outcome changes how axis keys in §3
are compared but not the matching algorithm itself.

## 1. Current state (audit)

### 1.1 Coordinate systems already encode order
`datamodel/systems.py` defines named systems whose axis TUPLES carry the order
explicitly. C-order and F-order are distinct types with reversed tuples:

- `FVoxelCoordinateSystem` axes are `(i, j, k)`; `CVoxelCoordinateSystem` axes
  are `(k, j, i)`.
- `fRAS` axes are `(x, y, z)` pointing RAS; `cRAS` axes are `(z, y, x)`.

So order is not implicit in an array flag — it is spelled out per system. The
consequence: reconciling order between two systems is a data transformation
between two fully-described axis lists, not a guess.

### 1.2 Axis vocabulary
`datamodel/axes.py` gives each `Axis` a `name`, `type` (`"space"`, `"time"`,
`"channel"`), `unit`, `discrete` flag, and `orientation` (for spatial axes:
`LeftToRight`, `PosteriorToAnterior`, …). Anatomical axes are singletons
(`R`, `A`, `S`, `L`, `P`, `I`). This vocabulary is what an adaptor matches on.

Note: `DisplacementAxis` / `CoordinateAxis` and `vector_axis()` are added by the
zarr redesign (#57) on its branch, not yet on `main`. This spec assumes they
land.

### 1.3 The building-block transforms exist, with exact inverses
`Permutation` (element `i` = the input dim feeding output dim `i`), `Scaling`,
`Translation`, `Identity`, and `Bijection` are all defined with closed-form
`inverse()`. These are exactly the transforms an adaptor emits, and they are the
"stay eager" set for lazy inverse (#31).

### 1.4 The adaptor is a stub, and unwired
`datamodel/_xform_adaptors.py` is a single `@_adaptor` function that raises
`NotImplementedError` ("not working at all yet"). Its TODO enumerates the
intended behaviour (same axes different order → `Permutation`; same axes
different units → `Scaling`; matched names/types/orientations; subset → add
axes + per-dimension transform).

`_adapt(s1, s2)` in `transformations.py` is a working dispatcher (type-distance
over `_ADAPTORS`, cached in `_ADAPTORS_FASTMAP`). But grepping the datamodel
shows `_adapt(` is **never called**. So the adaptor is both unimplemented and
uninvoked. #15 is greenfield.

## 2. What the adaptor must compute

Given a source system `S1` (the `output` system of transform `A`) and a target
system `S2` (the `input` system of transform `B`), produce a transformation
`T : S1 → S2` so that `B ∘ T ∘ A` is well defined. Cases, made precise from the
TODO:

- (a) **Same axis set, different order** → `Permutation`.
- (b) **Same axes, different units** → `Scaling` by the per-axis unit ratio.
- (c) **Same axes, different orientation** (e.g. RAS↔LPS sign flips) → a `±1`
  `Scaling` (a flip), and — only between array-index systems — the accompanying
  origin shift (see corner case 2).
- (d) **Names differ, but type and orientation match** → match on
  `(type, orientation)`, then apply (a)–(c).
- (e) **Subset / differing dimensionality** → embed missing axes as identity,
  and return a per-dimension (separable) transform over the matched axes.
- (f) **No match** → `AdaptationError` naming the two systems and the unmatched
  axes.

## 3. The matching algorithm (core routine)

1. Compute an **axis key** for each axis. Primary key: `(type, orientation)`.
   Secondary, for tie-breaks and for unclassified axes: `name`, then `unit`,
   then position.
2. Build a **bijection** between `S1` and `S2` axes by matching keys. A unique
   key match is unambiguous. Multiple axes with the same key (e.g. two
   unoriented spatial axes) tie-break by name, then by position.
3. From the bijection derive, in order:
   - a `Permutation` (the reordering), then
   - a per-axis `Scaling` combining the **unit ratio** and the **orientation
     sign** (`-1` where the matched axes point in opposite directions).
4. If any axis is unmatched, apply the subset policy (§5, corner case 5).

This routine is the shared kernel: axis-order reconciliation is step 3's
`Permutation`; unit/orientation reconciliation is step 3's `Scaling`.

## 4. Axis-order reconciliation (the OME/zarr ↔ nibabel problem)

The data model already lets any system state any order, so order only has to be
reconciled at I/O boundaries. A reader emits a `CoordinateSystem` with the
file's TRUE axis order; `_adapt` then inserts the `Permutation` to the working
order. No special-casing in the readers.

**Transforms act on any axes.** (Maintainer clarification.) The OME model — and
therefore ours — lets a transform act on an arbitrary subset of axes, not only
spatial ones. Consequences:

- The reconciling `Permutation` applies to the **full** axis list, not just
  `x/y/z`.
- For a transformation **field**, the vector (component) axis enumerates one
  component per acted-on axis, in that axis order. So when the acted-on axes are
  permuted, the field's **component axis must be permuted in lockstep** — and
  that lockstep spans non-spatial axes too when the field acts on them.

**(c, t) versus (t, c): do we care?** Only for a field whose vector axis spans
both. For a plain image, `(c, t)` vs `(t, c)` is a storage convention that
changes no transformation math; track it and move on. For a field acting on both
`c` and `t`, the k-th vector component *is* the k-th acted-on axis, so swapping
`c` and `t` forces the matching swap of vector components.

**[DECISION] Canonical working order.** Recommend the nibabel convention:
F-order `(x, y, z, t, c)` — spatial first in `x, y, z`, then time, then channel.
This makes the non-spatial order `(t, c)` (nibabel's dim-4 is time, the vector /
component dimension is last). Adopting it answers "(c,t) vs (t,c)" as **(t, c)**.

**[DECISION] Vector/component axis position.** Recommend **last**, matching
OME RFC-5 and the NIfTI vector dimension (dim-5). A format that stores it first
is reconciled to last at the boundary.

## 5. Separability ("by-dimension" transforms)

**Definition.** A transform `T` over axes `A` is *separable* if it factors as
independent sub-transforms over a partition of `A`: `T = ⊕ₖ Tₖ`, each `Tₖ`
acting on a disjoint axis group with no cross-coupling. Then `T` applies
group-by-group, which for resampling means a product of low-dimensional
interpolations instead of one high-dimensional one — the main efficiency win.

**Representation.** A `Separable` (a.k.a. `PerDimension`) transform holding an
ordered mapping `{axis-group → sub-transform}`, identity on any unlisted axis.
**[DECISION]** approve the type and its name.

**Detection.**
- Trivially separable: `Permutation`, `Scaling`, per-axis `Translation`, and any
  `Affine` whose linear block is diagonal.
- General `Affine`: separable iff the linear block is **block-diagonal under some
  axis partition**. Detect by taking the graph whose edges are non-zero
  off-diagonal entries and reading off its connected components; each component
  is an inseparable group.
- Fields: a displacement field is separable only if each component depends on its
  own axis alone (rare); assume **not** separable unless the producer says so.
- Closure: `Separable ∘ Separable` with compatible partitions stays separable;
  a `Permutation` relabels groups and is free.

**Payoff.** The adaptor's own output (permutation + per-axis scaling) is always
separable, so axis-order reconciliation costs a transpose/view, never a resample.

## 6. Wiring the adaptor into composition

`_adapt` must be invoked where adjacent transforms disagree on the system at
their shared boundary:

- In `Sequence` flattening/compute, at each adjacent pair `A, B` where
  `A.output` and `B.input` describe different systems, insert
  `_adapt(A.output, B.input)`.
- At the reslice boundary, between a transform's output system and the target
  grid's system.

Cache via the existing `_ADAPTORS_FASTMAP`. An inserted adaptor that is the
identity (systems already agree) must simplify away, so this ties into the
identity-dropping and lazy-inverse cancellation work: `adapt(S1,S2)` followed by
`adapt(S2,S1)` should cancel.

## 7. Corner-case audit

1. **Anisotropic + rotated data.** Order reconciliation is a `Permutation`;
   it is exact and never resamples. Keep it distinct from any resampling step.
2. **Orientation flips (RAS↔LPS).** Between *world* systems a flip is a `±1`
   `Scaling`. Between *array-index* systems a reversed axis also shifts the
   origin by `(n-1)` along that axis. **[DECISION]** the adaptor must know
   whether it is reconciling world or array-index systems to decide whether the
   flip carries that offset.
3. **Unknown axes** (`Axis()` with `type=None`). Matching falls back to
   position. Policy: positional match only when the axis counts are equal;
   otherwise `AdaptationError`. Do not silently pair mismatched unknowns.
4. **Duplicate keys** (two axes sharing `(type, orientation)`). Tie-break by
   name, then position; if still ambiguous, `AdaptationError`.
5. **Subset / superset** (a 2D transform in a 3D pipeline, or a field acting on
   a subset). Embed identity on the missing axes; drop an axis only when the
   transform is provably identity there. Never drop a non-identity axis.
6. **Non-spatial vector components.** As in §4: lockstep-permute the field's
   component axis with the acted-on axes; `(c, t)` ordering matters here.
7. **Discrete (channel) axes.** Never interpolate across them; separability must
   isolate a discrete axis into its own group.
8. **Unitless vs metric** (an unmatched unit, e.g. array vs millimetre). No
   scale factor exists. **[DECISION]** error, or pass through unconverted.
   Recommend error, since a silent unit mismatch is a correctness trap.
9. **Vector-axis position.** If a format stores the component axis first, the
   boundary reconciles it to last (§4 decision).
10. **Round-trip.** `adapt(S1,S2)` and `adapt(S2,S1)` must compose to identity
    and simplify away, so the adaptor emits only exactly-invertible pieces
    (`Permutation`, `Scaling`, `Translation`).

## 8. Decisions needed from the maintainer

- **[D1]** Canonical working order = nibabel F-order `(x, y, z, t, c)`, hence
  non-spatial order `(t, c)`?
- **[D2]** Vector/component axis canonical position = last?
- **[D3]** Approve a `Separable` / `PerDimension` transform type (and its name).
- **[D4]** Wire `_adapt` into composition boundaries now, or land the adaptor
  standalone first and wire later?
- **[D5]** Scope of the first #15 cut: full matcher (cases a–f) or a first cut
  covering a–c (permutation / scale / flip between known systems)?
- **[D6]** Corner cases 2 and 8: flip-offset semantics, and unitless-vs-metric
  policy.

## 9. Suggested build order

1. Matching kernel (§3) returning `Permutation` + `Scaling`, unit-tested against
   the named systems in `systems.py` (fRAS↔cRAS, fVoxel↔cVoxel, RAS↔LPS).
2. Replace the `NotImplementedError` adaptor with cases (a)–(d).
3. Wire `_adapt` into `Sequence` composition, with identity simplification.
4. `Separable` type + the block-diagonal detector (case (e) and B5).
5. Field component-axis lockstep permutation at the OME/zarr boundary (feeds B1).

## 10. What to route to Fable

Most of this is mechanical. Two parts warrant `[Fable Scope: Review Only]`
before merge: the **separability detector** (block-diagonal partition and its
composition closure) and the **orientation-flip offset semantics** between
array-index systems (corner case 2). The matching kernel, wiring, and
corner-case policy do not need it.
