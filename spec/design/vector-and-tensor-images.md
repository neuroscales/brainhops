# Images of vectors and tensors

Status: the data model and the operations are implemented; the readers,
the writers and the metadata are not (see section 8).

Some images hold, at each voxel, a vector or a tensor, whose components
change when the coordinate system changes. Displacement fields, fibre
directions, diffusion tensors and image gradients are common examples.
When such an image is moved or resampled, its values must also be
re-expressed, using the Jacobian of the transformation or a rotation
derived from it. This memo records how brainhops describes these values
and how it re-expresses them. It keeps the data model generic: nothing
in it is specific to diffusion imaging.

---

## 1. What has to be recorded

Two separate facts describe a vector image.

- What the entries along an axis are. An axis may hold the components of
  a point, a vector, a covector or a tensor, rather than independent
  channels such as colours.
- In which coordinate system the components are expressed. This system
  is called the frame of the values. Diffusion software often uses the
  voxel axes of the image, and other software uses a world space such as
  scanner RAS.

The first fact describes positions along an axis, so it belongs to the
axis, and the type of the axis records it (section 2). The second fact
describes the values, in the same way as their unit, which brainhops
already keeps out of the axes (`data_unit` is part of the metadata). It
therefore belongs to the image (section 3).

## 2. Component axes

`AxisType` has four types for the axes whose entries are the components
of one object, besides the `displacement` and `coordinate` types of
OME-NGFF. These six types are called component types, and an axis of
one of these types is a component axis. Calling `Axis` with a component
type builds a subclass of `ComponentAxis`, which says how the entries
change under a linear map `J` of the frame:

| Type           | Class              | How the entries change under `J` |
| -------------- | ------------------ | -------------------------------- |
| `point`        | `PointAxis`        | not by `J` (see below)           |
| `coordinate`   | `CoordinateAxis`   | as for `point`                   |
| `vector`       | `VectorAxis`       | `v ↦ J v`                        |
| `displacement` | `DisplacementAxis` | as for `vector`                  |
| `covector`     | `CovectorAxis`     | `g ↦ J⁻ᵀ g`                      |
| `tensor`       | `TensorAxis`       | `J` or `J⁻ᵀ` on each index       |

The type is used, rather than a separate field of the axis, because the
entries of an axis are either independent values or the components of
one object, never both, so a single field states it without allowing
contradictory combinations. OME-NGFF already describes the components of
its fields with axis types. `displacement` and `coordinate` are kept so
that the fields of OME-NGFF keep their types. A subclass of `Axis` cannot
pin a different type than its parent, so `DisplacementAxis` and
`VectorAxis` are siblings that share their behaviour through a table of
`ComponentAxis`, and so are `CoordinateAxis` and `PointAxis`. The
function `is_component_type` tells whether a type is a component type,
and the readers and the writers use it to store component axes with the
channels.

The entries of a vector or a covector axis follow axes of the system of
the frame, in order. This is also the rule that OME-NGFF RFC-3 states
for vector axes, and section 5 explains how these axes are found.

A tensor axis has three fields of its own.

- `indices` lists the kind of each index of the tensor, `"vector"` or
  `"covector"`, so that its length is the rank `k` of the tensor. The
  default is `("vector", "vector")`.
- `layout` is `"full"`, which stores the `n**k` components of a tensor
  in a frame of `n` axes, or `"symmetric"`, which stores only the
  `C(n + k - 1, k)` components whose indices do not decrease. A
  symmetric tensor must have indices of a single kind, because the
  symmetry of a tensor whose indices change differently depends on the
  frame. The default is `"full"`.
- `order` gives, for each entry along the axis, the frame axis of each
  index. By default, the full layout lists the components in row-major
  order and the symmetric layout lists the non-decreasing indices in
  lexicographic order, which is the order of FSL for a matrix. The
  order is a field because software packs a symmetric matrix in
  different orders, as MRtrix does.

When the frame changes by `J`, each vector index is multiplied by `J`
and each covector index by `J⁻ᵀ`. A diffusion tensor, with two vector
indices, becomes `J T Jᵀ`, a Hessian, with two covector indices, becomes
`J⁻ᵀ H J⁻¹`, and a linear map, with one index of each kind, becomes
`J A J⁻¹`. A two-point tensor, such as the Jacobian of a warp between
two spaces, has indices in two different frames, so it is stored as two
axes, a vector axis and a covector axis, each with its own frame.

The coordinates of a point change with the map itself rather than with
its Jacobian. A point axis can have a frame, which records the space of
the points, but `reframe` raises `NotImplementedError` on it, because
re-expressing points needs the inverse of a map, which is out of scope.

A component axis holds no array, no system and no transformation, so it
stays a small value that can be copied into every system that passes it
through.

## 3. The frames of an image

A single-scale image has a field `frames`, which holds one slot per data
axis, aligned with the axes as the shape is. A slot is one of three
things.

- `None` means the array system of the image. This is the default.
- A transformation from the array system to the frame of the values. In
  most cases it is an entry of `transformations`.
- A coordinate system names a foreign frame, which belongs to another
  space and which no transformation of the image reaches. The moving
  index of a two-point tensor is in a foreign frame. Such a slot is a
  label, and no operation changes it.

A slot may be set only on a component axis. The constructor and the
setter also accept a mapping from axis names or indices to slots.

A stored transformation corresponds exactly to the components: the
outputs of the transformation that no component axis of the array feeds
are as many as the components, and the components follow them in
order. Nothing else is guessed when a stored slot is read, and in
particular the types of the output axes are not consulted. A
transformation that is given to the constructor, to the setter, or as
the `frame` argument of `reframe` and `reslice` may have more such
outputs, as a map to a world that also has a time axis does. The
components then follow the outputs that share one axis type, provided
that exactly one type has as many outputs as there are components, and
a `Projection` that removes the other outputs is composed onto the
transformation before it is stored. The guess is limited to these
arguments because it serves the common case of a time axis carried
along, and storing its result keeps every slot unambiguous for the
operations that follow. A frame may end with projections and
permutations, which select and reorder the outputs of the map before
them, so a user can also restrict a frame explicitly.

The frames are never found by searching `transformations` for a system
name. Neuroimaging formats give the same names to different spaces, for
example "RAS" for the world of every NIfTI file, and a lazily applied
transformation from RAS to RAS adds a second entry that ends in "RAS".
A slot instead holds the path to its frame, which no name can confuse.

## 4. What each operation does

Let `F` be the slot of a component axis, with `None` read as the
identity of the array system.

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
component axes, because the voxel components of a vector do not change
then. The voxel coordinates of a point also need the map to have no
translation. Otherwise the slot becomes `sub2full` or `ψ`, restricted as
in section 3 when the types of its axes tell which axes the components
follow. A foreign slot is never changed.

Indexing a component axis with anything else than its full range in
order turns it into a `channel` axis, because the remaining entries are
no longer the components of one object. The axis then loses its frame.
Dropping the axis drops its slot.

`reslice` takes the target geometry, whose array system may describe the
components as channels. The result therefore copies each component axis
of the source onto the axis at the same position in its array system,
and keeps the name of that axis.

## 5. Re-expressing the values

`reframe(frame=-1, *, method="jacobian")` re-expresses the values that
are in the frame `F` in a new frame `F′`. At each voxel `x`, the matrix

```
J(x) = D(F′ ∘ F⁻¹)(F(x)) = D F′(x) · (D F(x))⁻¹
```

is applied to the values according to the type of their axis, and the
slot becomes `F′`. `D M (x)` is the Jacobian matrix of a map `M` at `x`,
restricted to the axes described below, and it is the linear part of `M`
when `M` is affine. `F` must be a diffeomorphism, so that `D F(x)` is
invertible. `J` is computed from two Jacobians at the voxels and never
needs the inverse of a map.

The axes that the Jacobian is restricted to are read from the structure
of the frames, and not from the types of the axes, so that the rules do
not assume that the vectors are spatial. The structure of a frame is the
linear part of the frame without its final projections and permutations,
or the linear part of its affine part, in which every field is taken as
the identity, followed by the matrix that selects the outputs. An output
is coupled to an input when the entry between them is not negligible.

- The components follow the outputs of the frame that no component axis
  of the array feeds, in order. A stored frame has exactly as many of
  them as there are components (section 3).
- The Jacobian is taken along the axes of the array, other than
  component axes, that are coupled to these outputs and to no other
  output that no component axis feeds. The outputs that the final
  projections remove count among these other outputs, so a time axis
  that the frame carries along is left out, even when a motion changes
  with time.
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

A transformation that the argument selects is restricted as in section
3 before it is used. A single value applies to every component axis
whose frame is not foreign. A mapping applies to the axes that it names,
and the other axes keep their frame. Naming an axis that is not a
component axis, or an axis in a foreign frame, raises an error. A list
is not accepted, because it would be unclear whether it follows all the
data axes or only the component axes, and because a string and a
`Sequence` transformation are both Python sequences.

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
here. A vector axis of an OME-Zarr image is a `vector` axis whose frame
is the transformation from the array to the intrinsic coordinate system.
The Zarr reader maps the types `point`, `vector`, `covector` and
`tensor` to their classes, and a tensor axis that it reads has the
default `indices`, `layout` and `order`, since OME-NGFF does not state
them. The reverse is lossy: NGFF has no standard way to store a
covector, the layout of a tensor, a two-point tensor or a foreign frame,
so a writer would store them as custom axis types and report the loss.

## 8. Status

Implemented:

- the component types of `AxisType` and their classes `ComponentAxis`,
  `PointAxis`, `CoordinateAxis`, `VectorAxis`, `DisplacementAxis`,
  `CovectorAxis` and `TensorAxis` (`brainhops.datamodel.axes`), and the
  predicate `is_component_type`, which the code that places the axes of
  an image in the order of a file format uses to store these axes with
  the channels;
- the `frames` field of single-scale images, the restriction of the
  frames that a user gives, and the way lazy transformations, indexing
  and reslicing keep the frames;
- `reframe`, and the `reframe` and `frame` arguments of `reslice`, on
  single-scale and multi-scale images.

The Jacobian of a frame that is not affine is computed with the
`gradient` function of the array backend, along the axes of section 5
only: the differences are central inside the grid and one-sided at its
edges, which is exact for a linear field. Along such an axis of one
voxel, the displacements of the fields are taken to be constant, so that
the column of the Jacobian is that of the frame with every field taken as
the identity. Along the other axes, such as time, the frame is evaluated
at every position but not differentiated, so a frame that changes with
time gives one Jacobian per time point. An exact Jacobian, computed from
the derivatives of the splines of the fields, is left for later.

Not implemented:

- readers that set the frames of the component axes they read, and
  writers that re-express the values in the frame of their format (NIfTI
  displacement fields, ITK, MRtrix, FSL, NRRD and OME-Zarr), which #539
  tracks. Until then, the frame of a component axis that a reader
  creates, such as the `displacement` axis of a NIfTI displacement
  field, is the array system by default, even when the file stores the
  vectors in another frame;
- the update of the metadata by `reframe`, which keeps the metadata as it
  is for now, although `data_unit`, `display_range` and the descriptions
  of the channels can describe the values in their old frame (#540);
- the descriptions of the channels in the metadata, which run along the
  axes of type `channel` only (`Along(AxisType.channel)`), and therefore
  not along component axes. Whether they should also run along
  component axes is left to #540;
- re-expressing the coordinates of points, which needs the inverse of a
  map rather than its Jacobian;
- the Jacobian of a spline field computed from the derivatives of its
  spline, which a port of `pull_grad` from torch-interpol would provide;
- a `reframe` method on multi-scale images;
- the use of frames by the directions held in the metadata, such as
  `bvectors`.
