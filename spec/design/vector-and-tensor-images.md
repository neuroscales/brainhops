# Images of vectors and tensors

Status: the data model, the operations and the frames of NIfTI files are
implemented; the readers and the writers of the other formats and the
metadata are not (see section 9).

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
  a point, a vector or a tensor, rather than independent
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

`AxisType` has three types for the axes whose entries are the components
of one object, besides the `displacement` and `coordinate` types of
OME-NGFF. These five types are called component types, and an axis of
one of these types is a component axis. Calling `Axis` with a component
type builds a subclass of `ComponentAxis`, which says how the entries
change under a linear map `J` of the frame:

| Type           | Class              | How the entries change under `J` |
| -------------- | ------------------ | -------------------------------- |
| `point`        | `PointAxis`        | not by `J` (see below)           |
| `coordinate`   | `CoordinateAxis`   | as for `point`                   |
| `vector`       | `VectorAxis`       | `v ↦ J v`, or `g ↦ J⁻ᵀ g`        |
| `displacement` | `DisplacementAxis` | `v ↦ J v`                        |
| `tensor`       | `TensorAxis`       | `J` or `J⁻ᵀ` on each index       |

The valence of a vector or a tensor says, for each index of the object,
whether the index changes with `J` (a `"vector"` index) or with `J⁻ᵀ` (a
`"covector"` index). A vector axis has a field `valence`, which is
`"vector"` by default and `"covector"` for an object such as the
gradient of an image, whose pairing with a vector does not depend on the
frame. A covector is therefore a vector axis rather than a type of its
own, so that vectors and tensors describe the valences of their indices
in the same way, and one rule applies the map to both: each vector index is
multiplied by `J` and each covector index by `J⁻ᵀ`. A displacement axis
has no `valence` field, because a displacement is always of valence
`"vector"`, and a point has no valence.

The type is used, rather than a separate field of the axis, because the
entries of an axis are either independent values or the components of
one object, never both, so a single field states it without allowing
contradictory combinations. OME-NGFF already describes the components of
its fields with axis types. `displacement` and `coordinate` are kept so
that the fields of OME-NGFF keep their types. A subclass of `Axis` cannot
pin a different type than its parent, so `DisplacementAxis` and
`VectorAxis` are siblings, and so are `CoordinateAxis` and `PointAxis`.
Each class gives the valence of its axis, and `ComponentAxis` applies
the map from the valence, so the siblings behave in the same way. The
function `is_component_type` tells whether a type is a component type,
and the readers and the writers use it to store component axes with the
channels.

The entries of a vector axis follow axes of the system of the frame, in
order. This is also the rule that OME-NGFF RFC-3 states
for vector axes, and section 5 explains how these axes are found.

The valences are the members of the enumeration `Valence`, and a field
also accepts their names as strings. A tensor axis has two fields of its
own.

- `valence` lists the valence of each index of the tensor, `"vector"`
  or `"covector"`, so that its length is the rank `k` of the tensor. The
  default is `("vector", "vector")`.
- `layout` says which components the axis stores, and in which order,
  for a frame of `n` axes. It is a name or an explicit tuple that gives,
  for each entry, the frame axis of each index. A single field holds
  both, because a named layout is only a short way to write a common
  tuple, and because software stores a symmetric matrix in different
  orders.

The names are the values of the enumeration `TensorLayout`, which a
field also accepts as strings. They are generic rather than the names of
software, so that they also describe tensors of other ranks. Their
orders for `k = 2` and `n = 3` are given below, where `ij` stands for
the entry `(i, j)`:

| `layout`                     | Order for `k = 2`, `n = 3`   | Used by                |
| ---------------------------- | ---------------------------- | ---------------------- |
| `"C"`                        | `00 01 02 10 11 12 20 21 22` |                        |
| `"F"`                        | `00 10 20 01 11 21 02 12 22` |                        |
| `"symmetric"`                | `00 01 02 11 12 22`          | FSL, ITK, NRRD, Camino |
| `"symmetric-lower"`          | `00 10 11 20 21 22`          | DIPY                   |
| `"symmetric-diagonal-first"` | `00 11 22 01 02 12`          | MRtrix                 |
| `"diagonal"`                 | `00 11 22`                   |                        |

`"C"` and `"F"` store all the `n**k` components, with the last and the
first index varying fastest. `"symmetric"` and `"symmetric-lower"` store
the `C(n + k - 1, k)` components whose indices do not decrease, or do
not increase, in lexicographic order, which is defined for every rank.
`"symmetric-diagonal-first"` has no natural order beyond rank 2, so it
is refused for other ranks. `"diagonal"` stores the `n` components whose
indices are all equal, and the other components are zero.

An explicit tuple lists each stored component once. The listed
components are independent, so a full tensor can be stored in any
order. A component that the tuple does not list takes the value of the
listed component whose indices are a permutation of its own, if there
is one, and is zero otherwise. A tuple that leaves out a component with
several listed permutations is refused, because their values may
differ. Taking a value from a permutation makes the tensor symmetric,
which needs all its indices to be of the same valence, because the
symmetry of a tensor whose indices have different valences depends on the
frame. The symmetric layouts, and the tuples that take a value from a
permutation, are therefore refused for a mixed valence.

`None`, the default, reads the layout from the number of entries once
`n` is known: `n**k` entries are `"C"` and `C(n + k - 1, k)` entries are
`"symmetric"`. These numbers differ when `n` and `k` are both at least
2, and any other number of entries is refused. `"diagonal"` is never
inferred. The operations take `n` from the frames of the axis: it is
the number of outputs of a frame that no component axis feeds, when
that number agrees with the number of entries. A number of entries that
two frames could give, such as 36 for a matrix, and an explicit tuple
that leaves the last axes of the frame out, are therefore read
correctly. Only when no frame is known is `n` read from the number of
entries alone, and an ambiguous number is then refused.

A layout with components that are zero by construction, such as
`"diagonal"` or an explicit tuple that leaves components out without a
permutation to take them from, cannot hold the tensor after a general
change of frame. `reframe` then stores the result in the smallest layout
that can hold it, `"symmetric"` when all the indices have the same
valence and the layout stores no two components whose indices are
permutations of each other, and `"C"` otherwise. The axis changes its
layout and its length, and the transformations and the frames of the
image get the new axis in their input system.

When the frame changes by `J`, each vector index is multiplied by `J`
and each covector index by `J⁻ᵀ`. A diffusion tensor, with two vector
indices, becomes `J T Jᵀ`, a Hessian, with two covector indices, becomes
`J⁻ᵀ H J⁻¹`, and a linear map, with one index of each valence, becomes
`J A J⁻¹`. A two-point tensor, such as the Jacobian of a warp between
two spaces, has indices in two different frames, so it is stored as two
vector axes, one of valence `"vector"` and one of valence `"covector"`,
each with its own frame.

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

The `method` argument chooses what is applied. It is a member of the
enumeration `ReframeMethod`, its name, or a callable.

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
The Zarr reader maps the types `point`, `vector` and `tensor` to their
classes, and a vector or a tensor axis that it reads has the default
`valence` and `layout`, since OME-NGFF does not state them. The reverse
is lossy: NGFF has no standard way to store the valence of a vector, the
valence and the layout of a tensor, a two-point tensor or a foreign
frame, so a writer would store them as custom axis types and report the
loss.

## 8. The frames of NIfTI files

A NIfTI file states the frame of its vectors only through its intent,
and only for displacement fields. `NiftiImage` reads the frame of the
displacement axis from the intent and keeps the values as the file
stores them:

| Intent                         | Code | Frame of the displacement axis      |
| ------------------------------ | ---- | ----------------------------------- |
| `DISPVECT`                     | 1006 | the best affine (RAS millimetres)   |
| FNIRT deformation field        | 2006 | the FSL scaled millimetres          |
| FNIRT cubic coefficients       | 2007 | the same, after the initial affine  |
| FNIRT quadratic coefficients   | 2009 | the same, after the initial affine  |
| FNIRT DCT coefficients         | 2008 | a foreign `FslCoordinateSystem`     |

A `DISPVECT` file holds RAS vectors in millimetres, in the world of its
best affine, as ITK 5.4 and later read it and as the NIfTI displacement
fields of brainhops do. Its frame is the last transformation that the
header decodes to, which is the best affine, restricted to its spatial
axes. A field with fewer components than spatial axes, such as a
two-dimensional field, has components along the first spatial axes, as
ITK reads it. The restriction by axis types of section 3 cannot choose
these axes, because all three spatial axes of the world have the same
type, so the reader composes the projection itself.

FSL writes its fields as 4-D files whose fourth axis holds the
components, rather than on the fifth axis as the NIfTI standard places
them. The reader therefore reads the fourth axis of a 4-D file with an
FSL intent as a displacement axis, and the writer stores the components
of a field that keeps an FSL intent along the fourth axis, because FSL
reads only the first volume of a 5-D file whose fourth axis has one
entry. Before #539, the reader read that axis as a time axis.

The FSL files hold vectors in FSL scaled millimetres, which are the
voxel indices multiplied by the voxel sizes, with the first axis flipped
when the best affine has a positive determinant. The frame of a
deformation field is the affine map from its voxels to an
`FslCoordinateSystem`. A coefficient field stores the coefficients of a
spline on a grid of knots, and its frame maps that grid to the scaled
millimetres of the reference, moved by the initial affine that the sform
stores as its inverse. The header gives this map without the reference
image, because FSL places the knots from the scaled millimetres of the
reference, so the geometry of the reference cancels out. Re-expressing
the coefficients in the array system gives the coefficients in units of
the grid, which is what the FNIRT reader of brainhops stores in its
displacement field, and the tests check that the two agree. A field of
DCT coefficients is not a grid of positions, and brainhops does not
decode it, so its frame is foreign.

Two properties of the FSL files are assumptions rather than facts that
the header states.

- FSL adds a vector to a position in the scaled millimetres of the
  reference and reads the sum as a position in the scaled millimetres of
  the moving image, so a vector is a difference between positions in two
  coordinate systems. The reader takes the vectors to be in the scaled
  millimetres of the reference, which identifies the axes of the two
  systems as FSL does when it adds them. The frame describes the
  components as FSL stores them, but the vectors re-expressed in RAS are
  the displacements of the map in RAS only when the scaled millimetres
  of both images have the same orientation in RAS. Otherwise the map
  needs the moving image, which `FnirtWarpField` takes and the image
  reader does not. The reader also assumes, as `FnirtWarpField` does
  when it is given no reference image, that a deformation field lies on
  the grid of its reference, as FNIRT and `convertwarp` write it.
- An FSL deformation field may hold absolute positions rather than
  displacements, and the header does not tell which. `FnirtWarpField`
  guesses from the values, but the image reader would have to read the
  values every time it builds its array system, so it reads every such
  field as displacements in the frame above. Absolute positions
  re-expressed in another frame are then wrong, because positions also
  change with the translation of the frame.

The frames of a file are decoded from its header each time they are
read, and they are not stored in the image, so that an image that is
consolidated gets the frames of its new header. Frames that are given to
the image take precedence over those of the header.

The writer of `NiftiImage` follows the same table in reverse.

- An image whose frames are those of its own header, and whose geometry
  is kept, is written with its values and its intent as they are,
  including an FSL intent, because its values are already in the frame
  that the header states.
- Otherwise, the vectors of a displacement axis, or of a vector axis of
  valence `"vector"`, are re-expressed in the world of the preferred
  transformation, which the writer stores as the best affine, and the
  header gets the DISPVECT intent. A vector axis therefore reads back as
  a displacement axis. A field read from an FSL file whose geometry has
  changed no longer lies on the grid of its reference, so it is written
  in this way too. DISPVECT is chosen rather than an FSL intent because
  every reader of NIfTI fields reads it, and because an FSL field is
  written by `FnirtWarpField`, which also stores what the image does not
  hold, such as the parameters of a spline.
- NIfTI has no frame for covectors, tensors and the coordinates of
  points. Its `SYMMATRIX` intent stores the lower triangle of a
  symmetric matrix, which is the `"symmetric-lower"` layout, but it says
  nothing of the frame, so tensors are left for later. The coefficients
  of an FSL spline field cannot be re-expressed without the header that
  holds the parameters of the spline. These values are written as they
  are, and only with `lossy=True`, as the next section describes.

## 9. Status

Implemented:

- the component types of `AxisType` and their classes `ComponentAxis`,
  `PointAxis`, `CoordinateAxis`, `VectorAxis`, `DisplacementAxis` and
  `TensorAxis` (`brainhops.datamodel.axes`), with the `valence` of
  vectors and tensors and the `layout` of tensors, and the predicate
  `is_component_type`, which the code that places the axes of an image
  in the order of a file format uses to store these axes with the
  channels;
- the `frames` field of single-scale images, the restriction of the
  frames that a user gives, and the way lazy transformations, indexing
  and reslicing keep the frames;
- `reframe`, and the `reframe` and `frame` arguments of `reslice`, on
  single-scale and multi-scale images;
- the frames of NIfTI displacement fields, which the reader sets from
  the intent and the writer stores as section 8 describes.

A writer applies a change of frame through affine maps silently, because
it is exact and part of the encoding of the format. It refuses, with a
`LossyConversionError`, a change through a map that is not affine, which
is computed by finite differences, a foreign frame, and a component
axis that the format cannot store in a frame of its own, such as a
covector axis in NIfTI. The `lossy=True` argument of `save` and of
`consolidate` allows these cases, as it allows a lossy conversion of a
transformation, and the values are then written approximately or as
they are.

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

- readers and writers of the frames of the component axes in the other
  formats (ITK, MRtrix, the FSL transformation formats, NRRD and
  OME-Zarr), which #539 tracks. Until then, the frame of a component
  axis that such a reader creates is the array system by default, even
  when the file stores the vectors in another frame;
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
