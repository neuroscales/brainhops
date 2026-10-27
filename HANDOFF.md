# Handoff: restriction of transformations, projections and axis queries

This branch exists only to hand the work over to another agent. It is cut
from the top of the current stack (`claude/feat/582-projection-at`, pull
request #583) and adds this file alone. It is not meant to be merged.
Delete the branch once the work is picked up.

The work started from the frames of vector and tensor images (#536,
#539). It moved to the compute API of transformations, because the frame
code needed to read part of a transformation, and that operation did not
exist in a principled form. The design memo for all of it is
`spec/design/compute-api-and-open-systems.md`, Part A.

## 1. State of the pull requests and issues

Merged into `main`:

- #571 (closes #570) lets `compute` read, compose and cancel
  projections. Created axes are zero, a `Projection` derives the system
  on one side from a closed system on the other, it converts to `Linear`
  and `Affine` when its number of axes is known, composers combine it
  with affine-like neighbours and with subspaces that wrap an affine, a
  pair rule merges two adjacent projections, and `Projection.inverse()`
  raises when the projection drops axes.
- #575 (closes #574) adds axis queries. `AxisSequence.index`, the new
  `indices` and `restrict` accept a name, a template axis, an axis class,
  a dict of axis fields or a predicate (`AxisQuery`, exported from
  `brainhops.datamodel.axes`). `index` and `indices` return numbers that
  are entries and positions at once: an axis after `...` is counted from
  the end, so that `index("t")` in `[x, ..., t]` is `-1`.

Open, in this order of the stack:

- #577, branch `claude/docs/576-batched-transforms-memo`, based on
  `main`. Documentation only. It settles the design of section 2 below in
  the memo: `restrict(input_axes, output_axes, at=None)` as the only
  method that removes axes, `Projection.at`, the product in which zero
  cancels NaN, the reduction of projections before composition, and
  batched transformations (new section A.10, decisions 24 to 28).
- #583, branch `claude/feat/582-projection-at`, based on the branch of
  #577. Pull request 1 of the three of #582. It implements
  `Projection.at`, the product in which zero cancels NaN, the check of the
  outermost composing `compute`, and the refusal of a free axis by the
  OME-NGFF writer. The gate passed: 10676 passed, 371 skipped, 1 xpassed.

Closed without merging: #573, a refactor of the frames of images that
stored frames as given. The user put it on hold. Issue #572 records the
idea and stays open.

Issues of this thread that are still open: #576 (batched
transformations), #582 (implement `restrict` with fixed and free axes,
in three pull requests), #572 (frames stored as given, on hold).

Nobody watches the pull requests of this thread, and no check-in is
scheduled.

Other open pull requests touch nearby code but belong to other sessions
or to the user: #548 (NIfTI frames, the base of the frames stack), #564,
#565 and #566 (MRtrix, OME-Zarr and NRRD frames, stacked on #548), and
#569 (the user's draft rework of the frames, stacked on #548). Do not
change them as part of this work.

## 2. The agreed design

These points are agreed with the user and written in the memo of #577.

`t.restrict(input_axes, output_axes, at=None)` returns the lazy
composition `π_O ∘ t ∘ ι_I`. The embedding `ι_I` creates the inputs that
are removed, and the projection `π_O` drops the outputs that are
removed. An input that `at` names is created at that value, which gives
the section of `t` there, for example one volume of a batch or one slice
of a volume. An input that `at` does not name is free: the result must
not depend on it, and `compute` raises `RestrictionError` when a kept
output reads it. A separate `project` method and a flag such as
`allow_coupled=` were rejected, because both would supply the value
implicitly, as zero.

`Projection.at` holds one value per created axis. It is empty by
default, which means zero, as in the `projectAxis` of OME-NGFF. A number
is the same map as the embedding followed by a translation. NaN makes the
axis free. In the matrix of a projection, `at` is the translation of the
created rows.

In a product of affine matrices, an exact zero coefficient cancels a
NaN, while a non-zero coefficient carries it. NaN therefore stays in the
translation column. An output that truly reads a free axis always ends
with a NaN, so the result is never wrong. An exact cancellation between
two paths, or a rounding residue, can keep a NaN that should vanish, and
`restrict` then refuses a block that is in fact closed.

The example that the user and the memo use is an affine of `(x, y, t)`
read from `(x, y)` to `(x, y)`:

| Affine                                        | `t` free            | `at={"t": 4}`       |
| --------------------------------------------- | ------------------- | ------------------- |
| `x' = 2x + 1`, `y' = 3y + 2`, `t' = t + 5`     | `[[2,0,1],[0,3,2]]` | `[[2,0,1],[0,3,2]]` |
| `x' = 2x + t/2 + 1`, `y' = 3y + 2`             | `RestrictionError`  | `[[2,0,3],[0,3,2]]` |
| `x' = 2x + 1`, `y' = 3y + 2`, `t' = t + y/10`  | `[[2,0,1],[0,3,2]]` | `[[2,0,1],[0,3,2]]` |

A batch axis is an axis that a transformation carries through unchanged
while its other outputs may depend on it, such as time for one affine
per volume. Slice timing is not batched over time, since its time output
reads `z`. One element of a batch is the restriction with `at` set on the
batch axis.

## 3. The proposed design that is not agreed yet

The next step is pull request 2 of #582. The memo in #577 still
describes it as commutation rules, pair rewrites such as
`π ∘ T → T' ∘ π'`. The user then asked for the dropping and the
reordering of axes to happen inside the compute and factor chain, written
as generally and as principled as possible. The last proposal, which the
user has not answered yet, is the following.

The chain is read as a graph of which axis depends on which. The factor
pass already reads the dependency pattern of every element
(`_read_pattern` in `compute/factor.py`), links coupled axes with a
union-find structure, and cuts every element into the piece of each group
(`_restrict_group`, through `compute/restrict.py`). Projections then need
no special rules: they are elements whose pattern has empty rows, for the
created axes, or empty columns, for the dropped axes. Two passes are
added over the same patterns:

1. A backward pass finds the needed axes. It starts from the outputs
   that the end of the chain keeps, and at each element an input is
   needed when a needed output reads it. The other inputs are dropped as
   early as possible.
2. A forward pass follows the created axes and their values from the
   embeddings, through the elements that carry them unchanged. A free
   axis that a needed output reads raises `RestrictionError`, and an axis
   with a value that a needed output reads becomes a constant of that
   element, which is the translation of the section.
3. Coupled axes are grouped as today, among the needed axes only.
4. Every element is cut to its needed outputs and to the inputs that
   they read. The current restrictors need a decoupled block, whereas this
   cut only needs a closed block, which the backward pass guarantees.

The normal form gains two optional ends:
`[drop the inputs that nothing reads] [grid] F_1 ... F_m [Π] [create the
constant outputs]`. A chain whose two ends have different numbers of
axes can then be factored, which the factor pass refuses today. The
passes would run in every composing `compute` whose chain contains a
projection, and the grouping into the normal form only with
`factor=True`. `simplify` would stay free of them.

The merged rules of #571 and #583 (the merge of two projections, the
projection composers and the product in which zero cancels NaN) remain
as the fallback for elements whose pattern cannot be read, which are
composed as matrices.

Limits that were stated to the user:

- An element whose pattern cannot be read, such as an unknown type or the
  lazy inverse of a general affine, counts as fully coupled. The factor
  pass reads the pattern of an inverse as the transpose of the pattern of
  its forward transformation, which is valid for independent groups but
  not for the one-sided closure, so the backward pass cannot reuse it.
- A field cannot be cut down to some of its own outputs until fields can
  have input and output spaces with different numbers of axes. The user
  plans that work separately. Until then the field keeps all its outputs,
  and the drop happens after it. A field inside a subspace is handled,
  because its pattern shows that it carries the other axes through.
- Patterns come from exact zeros. As section A.3 of the memo states, the
  analytic policy reads only the structure of types, and the numeric
  policy also reads which matrix entries are zero.

The suggested order was to write this design into the memo of #577
(sections A.3 and A.4), to get the review of the user, and only then to
implement it as pull request 2 of #582, replacing the commutation rules
that the memo currently describes. Pull request 3 of #582 then adds
`Transformation.restrict`, which accepts the axis queries of #575 and
builds the lazy expression, and removes the transformation-level
`embed` of `compute/restrict.py`.

## 4. Where the code is

| Topic | Location |
| --- | --- |
| `Projection`, `at`, derived systems, `_close_projection`, `_created_values`, `_has_free_axes` | `src/brainhops/datamodel/_transformations/meta.py` |
| Kinds of a projection (`_projection_member`) | `src/brainhops/datamodel/_transformations/compute/checkers.py` |
| Conversion of a projection to a matrix (`_projection_to_matrix`) | `src/brainhops/datamodel/_transformations/compute/converters.py` |
| Composers of projections, and the `Affine ∘ Affine` composer | `src/brainhops/datamodel/_transformations/compute/composers.py` |
| Merge of two projections (`_merge_projections`) | `src/brainhops/datamodel/_transformations/compute/simplifiers.py` |
| `affine_product`, `outermost_compute`, `refuse_free_axes`, `axis_counts` | `src/brainhops/datamodel/_transformations/compute/utils.py` |
| The stages of `compute` (`_compute_sequence`, `_compose_mode`) | `src/brainhops/datamodel/_transformations/sequence.py` |
| The factor pass and its normal form | `src/brainhops/datamodel/_transformations/compute/factor.py` |
| The current restrictors (decoupled blocks) | `src/brainhops/datamodel/_transformations/compute/restrict.py`, `restrictors.py` |
| The separable resampler, which reads the normal form | `src/brainhops/datamodel/_transformations/compute/separable.py` |
| Axis queries (`AxisQuery`, `index`, `indices`, `restrict`) | `src/brainhops/datamodel/_axes/list.py` |
| The OME-NGFF writer of transformations | `src/brainhops/io/transformations/zarr/_map.py` (`to_ome`, `_affine_to_ome`) |
| Image indexing, which fixes an axis like a section | `src/brainhops/datamodel/geometry.py` (`_index2transform`) |
| Tests of projections | `tests/test_datamodel_projection.py` |

## 5. Things that were learned the hard way

- `Transformation.simplify(policy)` is `compute(mode=False)`. A check
  that must not run in `simplify` has to test that `compute` composes.
- Composers call `compute` on partial products, for example
  `(To.to(Affine) @ Ti.to(Affine)).compute()`. A free axis may still
  reach an output there that a later projection drops, so the free-axis
  check runs only in the outermost call, which a context variable tracks.
- A `Projection` does not need to know how many axes the space has. It
  takes that number from its systems or from a neighbour in a
  composition. Converting it alone, without systems, raises
  `ConversionError`.
- The merge of two projections is computed in a space just wide enough
  to hold every position. It was checked against explicit matrix
  products on 5000 random pairs.
- The positional parameters of every public transformation are pinned
  by `tests/test_datamodel_constructor_signatures.py`. A new field must
  be keyword-only, as `Projection.at` is, unless the pin is meant to
  change.
- Plain `doctest` reads the closing fence of a Markdown code block as
  expected output. The examples of `_axes/list.py` were checked with the
  axis names in scope, ignoring that artefact.
- The OME-NGFF writer of brainhops has no `projectAxis` output. A
  projection is written through its affine matrix.
- A dict query such as `{"type": "vector"}` builds a `VectorAxis`, which
  does not match a `DisplacementAxis`. The class `ComponentAxis` matches
  every component axis.
- `AxisList([{"name": "x"}])` stores the dict as an entry without
  converting it to an `Axis`. This looks like a bug and has no issue yet.

## 6. Working with the user

- The user drives the design and wants a discussion before code. When a
  design point is open, explain it with a concrete example and ask,
  rather than implementing a guess. When the user says to stop, stop
  and change nothing.
- The user wants general and principled mechanisms, not special cases,
  and prefers separate pull requests based on `main`, unless a pull
  request depends on another one, in which case it is stacked.
- Before acting on GitHub (closing, commenting, opening pull requests),
  make sure the user asked for it.

## 7. Conventions and gate

The rules of `CLAUDE.md` in this repository and of
`bagofseeds/.github` (`CLAUDE.md` and `CONTRIBUTING.md`) apply. In short:

- The flow is an issue, then a branch, then a pull request. Branches are
  named `claude/<type>/<issue>-<description>`, without generated
  nicknames, and must not clash with an existing branch.
- Pull request bodies say what changed, why, and how it was verified,
  with the test line and the clean results of ruff and codespell pasted,
  and end with `Closes #NN` or `Part of #NN`. Pull requests are squash
  merged.
- Documentation follows the rules of `CLAUDE.md`: complete sentences, no
  bold text and no "Note:" callouts, NumPy docstrings, text wrapped at 72
  columns, every statement checked against the code, and no reference to
  an issue, a pull request or a file under `spec/` in a docstring or a
  message. Design memos record the decision, its reason and its status,
  not the history.
- No model identifier goes into a commit, a pull request or any file.

The gate before a pull request, run from a neutral directory so that the
installed package is not shadowed:

```
pip install '.[test]'
cd /tmp && python -m pytest <repo>/tests -q -n auto -p no:cacheprovider
cd <repo> && ruff check . && ruff format --check .
codespell -I codespell-ignore-words.txt src tests spec
```
