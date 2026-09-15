# Do not fold an affine into an interpolated field

Status: implementation plan, verified, for maintainer review. Produced by a
Fable design pass. Answers the maintainer principle: folding an affine into a
displacement/coordinates field changes the encoded transform once the field is
interpolated, so `compute()` must not combine an affine with an interpolated
field, and reslice must apply the composition first-to-last.

## Premise (verified numerically)

Real datamodel objects, `Sequence([D, A]).compute()` through the actual
composers, affine = anisotropic scale + shear + shift, versus the in-order
result `A(y + d̃(y))`, max abs error over 200 query points:

| order | bound | coeff | where | fold (engine) | fold + D's metadata | in-order `[C, D, A]` |
|---|---|---|---|---|---|---|
| 1 | nearest | F | interior | 7e-15 | 7e-15 | 4e-15 |
| 1 | nearest | F | outside FOV | **3.6** | **3.6** | 2e-15 |
| 1 | reflect | F | outside | **4.0** | **5.4** | 2e-15 |
| 3 | nearest | F | interior | **8.6e-2** | 2.1e-3 | 4e-15 |
| 3 | nearest | T | interior | **9.2e-2** | 7e-15 | 4e-15 |
| 3 | reflect | T | outside | **4.6** | **6.2** | 2e-15 |

Folding is exact only for order-1 value fields strictly inside the FOV (linear
precision). It is wrong outside the FOV for any order, wrong near borders for
order ≥ 2, and wrong even in the interior today because the composers reset
`order`/`bound`/`coeff` to defaults and treat coefficients as values. The
in-order path with a leading grid/coordinates field (the shape `reslice` builds)
is exact to ~1e-15 everywhere. `[A, D]` (affine first) is never folded.

## Engine facts that shape the design

The root compose pass walks left to right, so the accumulated `item` is always
the right operand (evaluated, never interpolated) while only raw elements become
the left operand interpolated by `pull_field`. `CompositionError` already keeps
elements separate. So the lossy fold happens exactly when the accumulated item
is a **stored field with no leading domain**: `[D, A]`, the sandwich
`[W⁻¹, D, W]`, `Geometry.compute()`, the NIfTI writer's `xform.compute()`.
Reslice instrumentation confirms `Affine ∘ DisplacementField` fires 0×;
`Affine ∘ CoordinatesField` fires on the terminal query coordinates.

## Design — distinguish the terminal query grid from an intermediate field

The crux: an affine folded into the **terminal query grid** is exact and
required by reslice; an affine folded into an **intermediate stored field** is
the corruption. Distinguish them by type.

- Add a private marker `_Evaluated(CoordinatesField)`: coordinates evaluated on a
  sampling domain, never interpolated, type-transparent. The root pass promotes
  a leading raw `CoordinatesField` (a point set or query grid) to `_Evaluated`,
  and demotes the final result back to a plain `CoordinatesField`.
- `_xform_composers.py`: **delete** the ten generic
  `{Translation, Scaling, Permutation, Linear, Affine} ∘ {CoordinatesField,
  DisplacementField}` composers. **Add** `Affine-ish ∘ CartesianField → _Evaluated`
  and `Affine-ish ∘ _Evaluated → _Evaluated` (pointwise, exact), so terminal
  folding is kept. **Narrow** `DisplacementField ∘ <domain>` and
  `CoordinatesField ∘ <domain>` to a `CartesianField`/`_Evaluated` left operand,
  returning `_Evaluated` — the one legitimate interpolation, in the field's own
  frame. A raw stored field then matches no affine composer, raises
  `CompositionError`, and is kept separate in application order. Affine∘affine,
  `Sequence`, and `Identity` composers are untouched.
- Reslice needs **no ordering change**: with a leading grid the root pass applies
  first-to-last and only `D ∘ domain` interpolates. The only local change: after
  `compute()`, if the result is still a `CoordinatesField` (an un-appliable
  element remained), raise a clear error naming it, instead of the current
  `AttributeError` on `.field`.
- New `Sequence.compute` contract: collapses to one `CoordinatesField` only when
  a sampling domain leads; otherwise affine runs merge and stored fields stay
  separate elements in order. `[D, A] → Sequence([D, A])`.

## Interactions

- **#31 cancellation** is complementary (more adjacency survives); it must run
  after `_drop_interior_grids` and to a fixpoint. `level⁻¹ @ level` still needs a
  lazy affine inverse or tolerance-based "numerically identity affine" dropping.
- **#59** unchanged; the `Identity` composer positional-reconstruction crash
  (#60) becomes more visible — fix first.
- **OME/FSL sandwich** `[W⁻¹, D, W]` standalone-computes to a `Sequence` (today
  already a Sequence, but with a mislabelled fold); in reslice it is exact and in
  order. Zarr level transforms are unaffected.
- **coeff** is safe once folds are refused (only `pull_field` consumes them), but
  is blocked by the broken bsplines conversions and by the undefined coeff-inverse.
- **`SubspaceTransformation`** needs a `SubspaceTransformation ∘ _Evaluated`
  composer applying the inner transform to `field[..., input_axes]` (a one-composer
  addition once the marker exists); its `inverse()` bug must be fixed first.

## Prerequisites and steps

Prereqs: the bsplines conversion fix, the `Identity` composer fix (#60), and the
#31 rework (with cancellation moved after grid-dropping). Then: (1)
`tests/test_fold_ordering.py` from the verification script (no-fold shape,
in-order exactness including outside-FOV for orders 1/3 × bounds × coeff, affine
pre-merge, end-to-end reslice vs scipy, the clear reslice error); (2) composers;
(3) marker + promotion/demotion + docstrings; (4) reslice check; (5) re-run
multiscale/NIfTI tests; (6) gate on 3.8 + current. Roughly 150 lines composers,
40 engine, 30 reslice/docs, 200 tests.

## Open questions for the maintainer

- **Q1** Refuse `DisplacementField ∘ DisplacementField` for two raw elements too
  (same principle; reslice loses nothing)?
- **Q2** Treat a leading raw `CoordinatesField` as a sampling domain (needed by
  `points.py` and the multiscale tests)?
- **Q3** The inverse of a coefficient field: carry `coeff=True` (re-fit) or
  `coeff=False`? (Shared with #31.)
- **Q4** Lazy affine inverse now (in #31) or later?
- **Q5** Tolerance-based identity detection for numerically-identity affines (so
  the `level⁻¹ @ level` sandwich cancels)?
- **Q6** Name and visibility of the evaluated-domain marker (private, or reuse
  `points.py`)?
