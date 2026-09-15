# The coordinate-system adaptor: contract and semantics

Status: draft for maintainer review. Written to answer three concerns raised in
review of the current adaptor: what it dispatches on, what it returns, and a
robust definition of what "adapting" means. Supersedes the `_adapt` /
`_ADAPTORS` sketch and the partial PR #15. Companion to
`axis_order_adaptors_separability.md`.

## 1. What "adapting" means

Adaptation bridges a **mismatch between two coordinate systems that meet at a
composition boundary**: the output system of one transform and the input system
of the next, or a transform's output system and a target grid's system at
reslice.

Adapting a source system to a target system produces a transformation, the
**bridge**, that carries coordinates expressed in the source frame to
coordinates expressed in the target frame, using only the information the two
systems' axis descriptions carry, and it is **exact and exactly invertible** —
no resampling, no interpolation, no information loss.

Concretely, adapting is: (1) identify a correspondence between the axes of the
two systems from their metadata, then (2) construct the exact, invertible
transformation that carries one frame to the other under that correspondence.
It is bookkeeping — reorder, rescale, flip, embed, drop, localize — never a
heuristic geometric alignment and never a resample.

Adaptation is **defined** (succeeds) exactly when a correspondence can be
established for every axis that must be matched (§3). When it cannot, it
**fails explicitly** (§6) rather than guessing.

## 2. What it dispatches on

**On axis values, not on Python types.** Whether and how two systems need
bridging depends entirely on their axes — name, type, unit, orientation,
discreteness — not on the systems' classes. Two systems of the *same* class can
need a flip; two of *different* classes can need nothing. So a registry keyed by
`(source type, target type)` is the wrong backbone.

The adaptor is therefore **one value-based routine over the two axis lists**.
The current `_adapt` / `_ADAPTORS` type-distance dispatch is retired. If
per-format customization is ever needed, it is registered against a **predicate
over axis properties**, never against a Python type; the default is the single
built-in routine.

## 3. Matching, and the pieces of a bridge

Input: two coordinate systems, each an ordered list of axes carrying `name`,
`type`, `unit`, `orientation`, `discrete`.

**Matching** establishes a partial bijection between source and target axes,
comparing in priority order:

1. `(type, orientation)` — the strongest signal (a right-to-left spatial axis
   matches a right-to-left spatial axis).
2. `name` — when types/orientations tie or are absent.
3. `unit` compatibility — spatial with spatial, temporal with temporal.
4. position — last-resort fallback, subject to the failure policy (§6).

From the bijection the bridge is built from these primitives, all exactly
invertible:

- **`Scaling`** — the per-axis unit ratio, and an orientation sign (`-1` where
  matched axes point oppositely). Between array-index systems an orientation
  flip also carries the extent-dependent origin offset (§6 / corner cases).
- **`Permutation`** — reorder matched axes into the target order.
- **`Projection`** — drop a source axis with no target match, or create a target
  axis with no source match (embedding), where that is meaningful.
- **`SubspaceTransformation`** — localize any of the above to the affected axis
  subset, or wrap a transform that acts on a subspace (§4).

## 4. What it returns

**The adaptor returns a `Sequence` — the bridge — never a bare single
transform.** Two reasons, the second of which is the substantive one:

1. A bridge is generally several pieces (`Permutation` after `Scaling` after
   `Projection`); a `Sequence` expresses that uniformly, and an empty/identity
   `Sequence` expresses "no adaptation needed" and simplifies away.
2. Sometimes the correct result is not a transform to **insert** between the two
   neighbours, but a restructuring that **wraps** the affected axes in a meta
   transform. Returning a `Sequence` lets the adaptor emit, for example, a
   `SubspaceTransformation(inner, input_axes=…, output_axes=…)` when only a
   subset of axes participates, or a `Projection` when a lower-dimensional
   transform must be embedded into a higher-dimensional space to meet its
   neighbour.

So the adaptor may return either shape, both as a `Sequence`:

- **an inserted bridge** — `Sequence([Permutation, Scaling, …])` spliced at the
  boundary (the common case); or
- **a wrapping** — a `Sequence` whose elements localize the adaptation to a
  subspace, e.g. `Sequence([SubspaceTransformation(Sequence([…]),
  input_axes=…, output_axes=…)])`, used when the mismatch is confined to a
  subset of axes, or when neighbours act on different subspaces that must be
  lifted to a common space before they compose.

Because the bridge is built only from exactly-invertible primitives,
`adapt(source, target)` and `adapt(target, source)` compose to identity and
simplify away. This is required so an inserted bridge that is later removed
leaves no residue, and so the #31 adjacent-inverse cancellation applies to
bridges.

## 5. Where it is invoked

Two candidate call sites:

- **(a)** at every composition boundary in `compute()` where adjacent systems
  disagree; or
- **(b)** only at I/O boundaries — adapt each object read or written to and from
  the canonical order — so that mid-sequence systems already agree.

With the canonical order now fixed (F-order `(x, y, z, t, c)`, vector axis
last), option (b) makes mid-sequence adaptation rare. **Recommend (b) as the
default**, with an explicit adapt available on request. **[Open — D4.]**

**Implicit vs explicit.** Recommend adaptation is **explicit** by default, and
that where `compute()` inserts one automatically it **warns**, because a silent
flip or permutation can mask a genuine data mismatch (the wrong file paired with
the wrong reference). **[Open.]**

## 6. Failure policy

When matching cannot establish a correspondence for an axis that must be matched
— an underspecified axis, or incompatible units with no conversion — adaptation
**fails with a clear error** naming the two systems and the unmatched axes,
rather than silently pairing by position. Positional fallback is used only when
explicitly permitted by an option, and then it warns. **Recommend
error-by-default. [Open — D6.]**

The orientation-flip origin offset between array-index systems (a reversed axis
shifts the origin by `extent − 1`) applies only when reconciling array-index
systems, not world systems; the adaptor must know which it is bridging.
**[Open — D6.]**

## 7. Relationship to other work

- Separability (B5) is expressed with the same `SubspaceTransformation` the
  adaptor emits.
- The adaptor emits only exact primitives, so it never resamples; a permutation
  is a transpose or a view.
- Retiring the type-keyed `_ADAPTORS` registry removes `_adapt`'s type-distance
  dispatch, and supersedes the partial adaptor of PR #15.

## 8. Open decisions

- Confirm the value-based single routine, no type dispatch (§2).
- Confirm the return-a-`Sequence` contract with `SubspaceTransformation` /
  `Projection` wrapping (§4).
- **D4** — invoke at all boundaries, or only at I/O boundaries to canonical (§5).
- Implicit-with-warning vs explicit-only (§5).
- **D6** — error-by-default vs positional-fallback-with-warning; and the
  array-index flip-offset rule (§6).
- The matching priority `(type, orientation) > name > unit > position` — is that
  the right order (§3)?
