# Images of vectors and tensors

Status: not implemented (see section 8).

Some images hold, at each voxel, a vector or a tensor, whose components
change when the coordinate system changes. Displacement fields, fibre
directions, diffusion tensors and image gradients are common examples. When such an image is moved or
resampled, its values must also be re-expressed, using the Jacobian of
the transformation or a rotation derived from it. This memo records how
brainhops describes these values and how it re-expresses them. It keeps
the data model generic: nothing in it is specific to diffusion imaging.

---

## 1. What has to be recorded

Two separate facts describe a vector image.

- What the entries along an axis are. An axis may hold the components of
  a vector, of a covector or of a symmetric matrix, rather than
  independent channels such as colours.
- In which coordinate system the components are expressed. This system
  is called the frame of the values. Diffusion software often uses the
  voxel axes of the image, and other software uses a world space such as
  scanner RAS.

The first fact describes positions along an axis, so it belongs to the
axis. The second fact describes the values, in the same way as their
unit, which brainhops already keeps out of the axes (`data_unit` is part
of the metadata). It therefore belongs to the image.

## 2. The content of an axis

An axis gains one field, `content`. It is `None` for an axis whose
entries are independent values, which is the default for every axis.
Otherwise it holds an instance of a small, open hierarchy:

| Class             | How the values change under a linear map `J` |
| ----------------- | -------------------------------------------- |
| `Vector`          | `v ↦ J v`                                    |
| `Covector`        | `g ↦ J⁻ᵀ g`                                  |
| `SymmetricMatrix` | `T ↦ J T Jᵀ`, on the packed components       |

The entries of a `Vector` or `Covector` axis follow axes of the system
of the frame, in order. This is also the rule that OME-NGFF RFC-3 states
for vector axes. Section 5 explains how these axes are found. A `SymmetricMatrix` axis stores the packed upper
triangle, and its `order` field gives the pair of indices of each entry,
because the packing differs between software (FSL, MRtrix and NRRD each
use a different order). A full matrix is stored as two `Vector` or
`Covector` axes, and the actions on the two axes combine into `J T Jᵀ`.
A two-point tensor, such as the Jacobian of a warp, has its two axes in
different frames.

The content holds no array, no system and no transformation, so an axis
stays a small value that can be copied into every system that passes it
through. No axis type gets a default content. A reader sets the content
explicitly, because a `displacement` axis read from a file whose frame is
unknown must not be taken to be in the voxel frame.

## 3. The frames of an image

A single-scale image gains one field, `frames`, which holds one slot per
data axis, aligned with the axes as the shape is. A slot is one of three
things.

- `None` means the array system of the image. This is the default.
- A transformation from the array system to the frame of the values. In
  most cases it is the same object as an entry of `transformations`.
- A coordinate system names a foreign frame, which belongs to another
  space and which no transformation of the image reaches. The
  coordinates held by a coordinates field and the moving index of a
  two-point tensor are in a foreign frame. Such a slot is a label, and
  no operation changes it.

A slot may be set only on an axis whose `content` is not `None`. The
constructor also accepts a mapping from axis names or indices to slots.

The frames are never found by searching `transformations` for a system
name. Neuroimaging formats give the same names to different spaces, for
example "RAS" for the world of every NIfTI file, and a lazily applied
transformation from RAS to RAS adds a second entry that ends in "RAS".
A slot instead holds the path to its frame, which no name can confuse.

## 4. What each operation does

Let `F` be the slot of a content axis, with `None` read as the identity
of the array system.

| Operation                    | Data                 | Slot                    |
| ---------------------------- | -------------------- | ----------------------- |
| `X(U)`                       | unchanged            | unchanged               |
| `X.transformation = ...`     | unchanged            | unchanged               |
| `X[index]`                   | unchanged (a view)   | `F ∘ sub2full`          |
| `X.reslice(G)`               | sampled, not rotated | `F ∘ ψ`                 |
| `X.reframe(F′)`              | re-expressed         | `F′`                    |

Here `sub2full` maps the indexed array to the original array, and `ψ`
maps the voxels of the resliced image to the voxels of `X`. Both are
already computed by these operations. A `None` slot stays `None` when the
linear part of `sub2full` or `ψ` is the identity on the axes that are not
content axes, because the voxel components of a vector do not change
then. A foreign
slot is never changed.

Indexing a content axis with anything else than its full range in order
clears its content, because the remaining entries are no longer the
components of the same object. Dropping the axis drops its slot.

`reslice` takes the target geometry, whose array system may describe its
channel axes without content. The result therefore copies the content of
each content axis of the source onto the axis at the same position in its
array system.

## 5. Re-expressing the values

`reframe(frame=-1, *, method="jacobian")` re-expresses the values that
are in the frame `F` in a new frame `F′`. At each voxel `x`, the matrix

```
J(x) = D(F′ ∘ F⁻¹)(F(x)) = D F′(x) · (D F(x))⁻¹
```

is applied to the values according to their content, and the slot
becomes `F′`. `D M (x)` is the Jacobian matrix of a map `M` at `x`,
restricted to the axes described below, and it is the linear part of `M`
when `M` is affine. `F` must be a diffeomorphism, so that `D F(x)` is
invertible. `J` is computed from two Jacobians at the voxels and never
needs the inverse of a map.

The axes that the Jacobian is restricted to are read from the structure
of the frames, and not from the types of the axes, so that the rules do
not assume that the vectors are spatial. The structure of a frame is its
linear part, or the linear part of its affine part, in which every field
is taken as the identity. An output of the frame is coupled to an input
when the entry between them is not negligible.

- The components follow the outputs of the frame that are not coupled to
  a content axis of the array, in order.
- When there are more of these outputs than components, as when the
  frame carries a time axis along, the components follow the outputs
  that share one axis type, provided that exactly one type has as many
  outputs as there are components. Otherwise the frame is ambiguous and
  is refused. A frame whose system has exactly as many axes as there are
  components, besides the content axes, is never ambiguous.
- The Jacobian is taken along the axes of the array, other than content
  axes, that are coupled to these outputs and to no other output. A time
  axis that the frame carries along is left out, even when a motion
  changes with time.
- The two frames of a change must take the components from the same axes
  of the array, since their Jacobians are only comparable in the same
  basis. The array system has no structure of its own, so it takes the
  axes of the other frame.

These rules let the components live in a subspace: a frame that maps the
first three of four axes onto its three axes describes vectors of three
components in a four-dimensional image.

The `frame` argument accepts:

- an integer, which selects an entry of `transformations` without
  reordering it, with `-1` (the preferred transformation) as default;
- a string, which selects the most preferred entry of `transformations`
  whose output system has that name, as the `transformation` setter
  does;
- `None`, the array system;
- a transformation from the array system;
- a mapping from axis names or indices to any of the above.

A single value applies to every content axis whose frame is not foreign.
A mapping applies to the axes that it names, and the other axes keep
their frame. Naming an axis without content, or an axis in a foreign
frame, raises an error. A list is not accepted, because it would be
unclear whether it follows all the data axes or only the content axes,
and because a string and a `Sequence` transformation are both Python
sequences.

The `method` argument chooses what is applied.

- `"jacobian"` applies `J`. It changes the frame exactly, so the values
  describe the same objects as before.
- `"rotation"` applies the rotation `R` of the polar decomposition
  `J = R P`, where `P` is symmetric positive definite. It keeps the
  length of vectors and the eigenvalues of tensors, which diffusion
  software calls the finite-strain reorientation.
- A callable receives `J` and returns the matrix to apply.

Diffusion software calls these policies "reorientation". The word is
not used for the method itself, because in neuroimaging it already means
permuting and flipping the voxel axes of an image.

`reslice` accepts the same choice through two keyword arguments.
`reframe` is `False` by default, which keeps the values in their frame;
`True` stands for `"jacobian"`, and a method name or a callable chooses
the method. `frame` takes the same values as in `reframe` and is resolved
on the result, whose only transformation is that of the target geometry.
The values are re-expressed after they are sampled, so that `J` is
evaluated at the output voxels. The two arguments are separate because
`bool` is a subclass of `int` in Python, so a single argument would make
`reframe=0` select the first transformation rather than turn reframing
off.

## 6. A worked example

`X = load(f)` reads a NIfTI displacement field whose vectors are in RAS.
`V` is its array system, `W` its world and `A : V → W` its
voxel-to-world map. `U : W′ → W` is a lazily applied transformation, and
`G` is a geometry with array system `V′` and voxel-to-world map
`B : V′ → W′`, so that `ψ = A⁻¹ ∘ U ∘ B`.

| Expression        | `data` | `transformations` | Slot of `c`    |
| ----------------- | ------ | ----------------- | -------------- |
| `X`               | `c`    | `[A]`             | `A`            |
| `X(U)`            | `c`    | `[A, U⁻¹ ∘ A]`    | `A`            |
| `X(U).reslice()`  | `c`    | `[U⁻¹ ∘ A]`       | `A`            |
| `X(U).reslice(G)` | `c ∘ ψ` | `[B]`            | `A ∘ ψ = U ∘ B` |

After `X(U).reslice()`, no entry of `transformations` ends in `W`, but
the slot still reaches it. The values are re-expressed as follows.

| Call                             | `F`     | `F′`      | `J`                        |
| -------------------------------- | ------- | --------- | -------------------------- |
| `X.reframe(None)`                | `A`     | identity  | `(D A)⁻¹`                  |
| `X(U).reframe()`                 | `A`     | `U⁻¹ ∘ A` | `D U⁻¹ (A(x))`             |
| `X(U).reframe(None)`             | `A`     | identity  | `(D A)⁻¹`                  |
| `X(U).reslice(G).reframe()`      | `U ∘ B` | `B`       | `(D U (B(y)))⁻¹`           |
| `X(U).reslice(G).reframe(None)`  | `U ∘ B` | identity  | `(D B)⁻¹ · (D U (B(y)))⁻¹` |

The fourth row is the usual result after a registration: the vectors are
rotated into the world of the target. The third row shows that the voxel
components of a vector do not depend on where the image is placed.

## 7. Relation to OME-NGFF

OME-NGFF 0.6 defines `displacement` and `coordinate` axes for the arrays
of displacement and coordinate fields, and states that the `i`th entry
along such an axis refers to the `i`th output axis of the transformation
that uses the array. RFC-3, which is under review, lets an image have any
axes and requires that the entries of a vector axis follow the order of
the spatial axes. It gives diffusion tensors as a motivating case but
defines no layout for them.

Everything that these documents let a file state can be represented
here. A vector axis of an OME-Zarr image is a `Vector` axis whose frame
is the transformation from the array to the intrinsic coordinate system.
The reverse is lossy: NGFF has no standard way to store a covector, the
packing of a symmetric matrix, a two-point tensor or a foreign frame, so
a writer would store them as custom axis types and report the loss.

## 8. Status

Nothing in this memo is implemented yet. The work is tracked in #536 and
lands in separate pull requests: the data model and the operations
(#538), the readers and the writers (#539), and the metadata under a
change of frame (#540).
