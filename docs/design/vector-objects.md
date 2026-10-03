# Design: vector objects (points, polylines, meshes)

**Status:** design only, no code. Second draft, rewritten after the
first review round on #249 around an nd data model (raster and index
axes, elements and labels as nd relations), reviewed by Fable and
corrected. Relates to #130 (general
vector formats), #175 (zarr-vectors) and the per-category format
issues. Decisions still open are listed under
[Open questions](#open-questions).

This memo proposes a data model for *vector objects*: points (point
clouds), polylines (streamlines), graphs and skeletons, surface meshes
and volume meshes, including the less common layouts that chunked
stores (zarr-vectors, neuroglancer) and time-resolved data produce. It
aims for four things:

1. **One API shape shared with images.** A vector object stores its
   vertices in their native coordinate system and carries a list of
   native-to-world transformations, exactly as an image stores voxels
   and carries voxel-to-world transformations. `transformations`,
   `transformation`, `__call__`, `__getitem__`, `to_singlescale` and
   `load`/`save` mean the same thing for both, and `pull` / `push` name
   the two directions in which data meets a transformation (§4).
2. **nd everywhere it makes sense.** Vertices may be laid out on any
   number of array axes, some of which are *raster* coordinates (an
   implicit, image-like grid, such as frames in time) and some plain
   *index* axes. Elements and labels address vertices with nd indices,
   never with linearised ones, and broadcast over the axes they do not
   address.
3. **Region queries.** `vectors[...]` selects the part of the object
   inside a box of its native space: discrete, image-like indexing on
   raster axes and *continuous* bounds on explicit coordinates. On a
   chunked store it only reads the chunks that intersect the box.
4. **Multi-scale.** A pyramid of levels, where a level is a geometric
   simplification, a sub-sample of the objects, or both.

---

## 0. Vocabulary

The same handful of concepts goes by different names in every field and
format. This memo uses the names in the first column.

| Concept (this memo) | Meaning | zarr-vectors | neuroglancer | TRX / nibabel | VTK / meshio | trimesh / GIFTI |
|---|---|---|---|---|---|---|
| **vertex** | a point of the native space | vertex | vertex / position | position / point | point | vertex / pointset entry |
| **component** | one explicit coordinate stored per vertex (`x`, `y`, `z`, …) | vertex column | — | xyz | point coordinate | coordinate |
| **raster axis** | an array axis whose index *is* a coordinate (time frames, slices, channels) | — | — | — | — | — |
| **index axis** | an array axis that only enumerates vertices (no coordinate meaning) | row in a chunk; chunk grid | — | point index | point id | vertex index |
| **element** | an atomic, ordered label of `E` vertices: point, edge, triangle, quad, tetrahedron | link (`link_width = E`) | edge (skeleton), triangle (mesh) | implicit edge | cell (with a cell type) | face / triangle |
| **element width** `E` | number of vertices in an element | `link_width` | — | — | cell size | 3 |
| **topological dimension** `dim` | 0 point, 1 edge, 2 surface element, 3 volume element | — | — | — | cell dimension | — |
| **directed** | the order of an element's vertices carries meaning | `directed` | — | — | — | winding |
| **label** | a set of members of a lower level (vertices or other labels), possibly ordered; elements, fragments, objects and groups are labels | — | — | — | — | label |
| **fragment** | a label over vertices: a run of vertices belonging to one object | fragment | fragment (mesh octree node) | one streamline | — | — |
| **object** | a label over fragments: one streamline, one neuron, one surface | object | segment / annotation | streamline | — | one mesh |
| **group** | a label over objects | group | — | group | — | label |
| **attribute** | data attached to each vertex, element or label | vertex / link / fragment / object attribute | vertex attribute, annotation property | dpv / dps / dpg | point_data / cell_data / field_data | vertex / face attributes |
| **native space** | the coordinate system of the raster axes plus the components | level coordinates | model space (before `transform`) | RAS mm (TRX), voxmm (TRK) | — | `coords` |
| **chunk** | a box of the native space, the unit of storage | chunk | chunk / octree node / spatial cell | — | — | — |
| **level** | one resolution of a pyramid | level | level of detail / spatial index level | — | — | — |

**CSR** ("compressed sparse row", the sparse-matrix layout) stores a list
of variable-length lists as two flat arrays: the members of every list
end to end, and the start of each list. TRX `offsets`, nibabel's
`ArraySequence`, VTK's `connectivity` + `offsets` and zarr-vectors'
fragment ranges are all this layout. Labels (§2.3) generalise it to nd:
a start nd-index and an nd extent per label.

---

## 1. What the image API does today

The parts the vector API copies, from `datamodel/images.py` and
`datamodel/geometry.py`:

| Image API | Meaning |
|---|---|
| `data` | array in the image's native (voxel) space |
| `transformations` | list of voxel-to-world transformations; last is preferred |
| `transformation` | the preferred one; the setter reorders the list |
| `geometry` | `Geometry((CartesianField(shape), transformation))` |
| `img(T)` | lazy: new preferred transformation `T.inverse() @ img.transformation` |
| `img.reslice(geom)` | resample data onto `geom`; result carries `geom.transformation` (renamed `pull`, §4.4) |
| `img[index]` | crop/stride in voxel space; transformations are updated so the world placement is unchanged |
| `MultiScaleImage` | `images` (levels) + transformations of the pyramid; `reslice` picks the level nearest to the target resolution |
| `io.images.load/sniff` | format dispatch through `FileBasedImage` |

The observation that lets the vector model reuse the existing machinery:
**a point array can be evaluated through the transformation composers.**
The composers that apply a transformation to a `CoordinatesField`
(`Affine @ CoordinatesField` is matrix arithmetic; the field composers
call `pull_field(field, coords=...)`, which accepts any `(..., D)`
coordinates) already compute `T(x)` for arbitrary points. A private
helper, `_apply_to_points(T, points) -> points`, wraps a `(..., D)`
array in a throw-away `CoordinatesField` (`coeff=False` forced),
composes, computes, and returns `.field`.

The vertices are **never** stored or exposed as a `Transformation`. Much
of the transformation code assumes a field is defined on a grid with as
many axes as components (`_fields_as_identity`, `InverseCoordinatesField`
mesh inversion, `order`/`bound`/`coeff` metadata, and spline prefiltering
when `coeff` is set, which would run *along the vertex index*). Treating
vertices as a transformation would leak all of that.

---

## 2. Data model

Shapes are written as tuples of named groups: `(*A, D)` is "the array
axes `A`, then `D` components". `X[fields]` means a structured
alternative with one field per entry, used wherever columns may have
different dtypes (§2.6).

### 2.1 Overview

```
Vectors[E]
├── vertices: Vertices          the points, on nd array axes
├── labels: {name: Labels}      nd relations over vertices or over other labels:
│                                 elements (E), fragments, objects, groups, …
└── transformations: [...]      native -> world, last preferred
```

| Class | Elements | Typical use |
|---|---|---|
| `Points` | `Singular` (implicit, `dim = 0`) | point clouds, centroids, annotations |
| `Graphs` | `Edges` (`dim = 1`, `E = 2`) | skeletons, trees, connectomes, lineages |
| `Polylines` (`Streamlines`) | `SequentialEdges` (implicit) | streamlines, tracks, contours |
| `SurfaceMeshes` | `SurfaceElements` (`dim = 2`) | general surface meshes |
| `TriangleMeshes`, `QuadMeshes` | `Triangles` (`E = 3`), `Quads` (`E = 4`) | cortical surfaces, structured surfaces |
| `VolumeMeshes` | `VolumeElements` (`dim = 3`) | general volume meshes |
| `TetrahedralMeshes` | `Tetrahedra` (`E = 4`) | FEM meshes, deformation meshes |

`Polylines` is a `Graphs` whose edges are implicit; `TriangleMeshes` is a
`SurfaceMeshes` whose elements are triangles, and so on. A quad and a
tetrahedron both have `E = 4`; they are told apart by `dim`, which is
also what the class hierarchy is organised by. `MultiScaleVectors[T]`
(§6) holds levels of one of these types.

Parametric shapes (neuroglancer `AXIS_ALIGNED_BOUNDING_BOX`,
`ELLIPSOID`) are **out of scope**: they are not closed under nonlinear
transformations, so they do not fit "store natively, transform
lazily". A later `Annotations` class can hold them and offer
`to_mesh()`; neuroglancer `POINT`, `LINE` and `POLYLINE` annotations map
onto `Points`, `Graphs` and `Polylines`.

### 2.2 `Vertices`

```
class Vertices(Magic):
    data:        (*A, D) | (*A,)[components]   explicit coordinates
    axes:        AxisList, len(A)              one per array axis: raster or index
    components:  AxisList, D                   one per explicit coordinate
    attributes:  (*A, AV) | (*A,)[fields]      optional
    valid:       (*A,) bool | lengths          optional, for padded layouts

    system   -> CoordinateSystem(raster axes of `axes` + components)
    ncomponents -> D
```

Each **array axis** is one of:

- a **raster axis**: its index is a coordinate, exactly like an image
  axis. It is a `SpaceAxis`, `TimeAxis` or `ChannelAxis` (the
  categorical case, as in images) whose unit is the sample. A vertex at
  array index `(t, n)` with raster axis `t` lives at native coordinate
  `(t, *data[t, n])`.
- an **index axis**: it only enumerates vertices and is not a
  coordinate. This needs a new axis type, `IndexAxis` (`type="index"`),
  which is never part of a coordinate system. The usual `(N,)` axis of a
  point list is one, and so is a chunk-grid axis of a store whose
  vertices carry their full position (§2.7, example K).

The **native coordinate system** is the raster axes (in array order)
followed by the components. Transformations take it as input. A plain
point cloud has no raster axis, so its system is just the components,
as before.

**Padding.** nd layouts are often ragged: a different number of cells
per frame, of points per slice or per chunk. They are padded, and
`valid` says which entries exist: either a boolean mask of shape `(*A,)`,
or `lengths` of shape `(*A[:-1],)`, the number of valid entries along the
last array axis (the compact form for "padded at the end", which is what
per-chunk stores produce). Invalid vertices are never inside a region,
never moved, and never members of an element or label.

**Components and attributes** may be a homogeneous array (`(*A, D)`,
the common, fast case) or structured (one field per component, for
mixed dtypes such as a categorical coordinate stored as integer codes;
§2.6).

### 2.3 `Labels`: one relation for elements, fragments, objects, groups

Everything above the vertices is the same kind of thing: a **label** is
a set of members of a lower level, possibly ordered. An element is a
label of `E` vertices; a fragment is a label of vertices; an object is a
label of fragments; a group is a label of objects. They differ only in
how their members are stored and in a few flags, so one class defines
them all.

```
class Labels(Magic, Generic[T]):
    of:          str = "vertices"        what is labelled
    axes:        names of the target axes the member indices address
    batch:       names of the target free axes that G lines up with = ()
    own_axes:    names of the label axes L = ("label",)

    # members, in one of three forms (below)
    members:     dense: (*G, *L, W[, len(axes)]) | boxed: (*K[, len(axes)]) | None
    indices:     boxed only: (*G, *L[, ndim(members)])  start of each label
    lengths:     boxed only: (*G, *L[, ndim(members)])  extent; optional iff ndim(members) == 1

    ordered:     bool = False
    directed:    bool = False
    exclusive:   bool = False
    atomic:      bool = False
    names:       (*L,) | None
    attributes:  (*G, *L, A) | (*G, *L)[fields]   optional
    valid:       (*G, *L) bool | lengths (*G, *L[:-1])   optional
```

**Three storage forms.**

| Form | Stored | Label `p` is | Used by |
|---|---|---|---|
| **implicit** | nothing | generated from the target's layout | `Singular`, `SequentialEdges`, `GridElements`, `Objects.identity(n)` |
| **dense** | `members (*G, *L, W[, len(axes)])` | `members[p, :]`, exactly `W` members | elements (`W = E`), fixed-length fragments |
| **boxed** | `indices`, `lengths`, optional `members (*K, ...)` | `members[indices[p] : indices[p] + lengths[p]]`, an nd box | fragments, objects, groups, polygons of varying size |

- **Dense** is the `(M, 3)` triangle array. The width `W` is the axis
  after the label axes, and nothing else is stored.
- **Boxed** generalises CSR to nd. When `members` is `None` the box is
  taken in the target's own index space (`ndim(members) = len(axes)`),
  so a track over `("t", "n")` is `indices = (0, n)`,
  `lengths = (T, 1)`. When `members` is given, `ndim(members) = len(K)`.
  `lengths` is optional only when `members` is 1-D: labels are then CSR
  runs and each length is the distance to the next start, as in TRX
  `offsets`. `members` is an indirection: it may reorder, repeat (a
  vertex shared by two fragments, or a fragment by two objects, as
  zarr-vectors allows) or gather non-contiguous members. A label that is
  not a box of the target is a run of a 1-D `members`.
- A box is read in C order when `ordered`. Invalid members inside a box
  are skipped; they do not break the order or split the label (a track
  with a missing frame stays one fragment).

**nd indices.** Each member reference is an nd index over `axes`. The
trailing index dimension is present only when `len(axes) > 1`, so the
common cases stay `(M, 3)` and `(P,)`; with a structured dtype the
fields are named after `axes` instead. Indices are never linearised: an
index into a 2-D layout is a pair.

**Named axes.** `axes` names the target axes the indices address: vertex
array axes for elements and fragments (default: the last index axis,
which covers every flat layout), the target label's `own_axes` for
objects and groups (`Objects.axes = ("fragment",)`). The target axes
*not* in `axes` are the label's **free axes**; `batch` names the free
axes that the leading dimensions `G` line up with, one to one and by
name, and every free axis not in `batch` is shared (as is a `G`
dimension of size 1). With `batch` and `own_axes` named, the split of
every shape into `G`, `L`, `W` and the index dimension is determined.

- One `(M, 3)` triangle array with `batch = ()` serves every frame of a
  `(T, N, 3)` time series (shared topology); a `(Z, M, 2)` edge array
  with `batch = ("z",)` gives every slice its own edges.
- A label that joins members from different frames or slices includes
  that raster axis in `axes`: a division edge from `(t, n)` to
  `(t + 1, m)` is `axes = ("t", "n")`, one row `[[t, n], [t + 1, m]]`.
- The label axes `L` are usually one axis, but may be nd when the labels
  themselves lie on a grid, for example a stored quad mesh of a `H × W`
  vertex grid, `members (H - 1, W - 1, 4, 2)` with `axes = ("i", "j")`
  and `own_axes = ("ci", "cj")`.
- In the dense form `members` carries the batch axes, so dense labels may
  differ per frame. In the boxed form only `indices` and `lengths` do;
  an explicit member list shared by all frames is enough for every
  example here (open question 7).

**Flags.**

- `ordered`: the order of the members is structural. A polyline follows
  its fragment; an element's vertex order defines the cell (a quad is a
  loop through its vertices in order). Groups are unordered.
- `directed` (implies `ordered`, the zarr-vectors name): reversing the
  order changes the meaning. An edge has a direction (parent → child); a
  face has a winding (which side its normal points to); a tetrahedron
  has a handedness, the sign of `det[v1 − v0, v2 − v0, v3 − v0]`. A
  directed `VolumeMeshes` promises every tet has positive signed volume
  in native space (the VTK / Gmsh convention), which makes folding
  visible after a deformation and gives boundary triangles an outward
  winding.
- `exclusive`: every member has exactly one label (a partition).
  Shared fragments and overlapping TRX groups mean it cannot be assumed.
- `atomic`: the label exists whole or not at all. An element is atomic
  (a triangle that lost a vertex is gone); a fragment is not (it loses
  members, and splits when compacted). This is the only thing cropping
  needs to know (§5.2).

**Lookup and composition.** The stored form is label → members. The
reverse map, member → label(s), is derived and cached, and it is what
makes `v.objects[17]` or `v.groups["CST"]` a lookup. Object → vertices is
fragments composed with objects, a product of two relations; group →
vertices adds one more. `Objects.identity(n)` is "one object per
fragment"; `Objects.from_ids(ids)` builds a labeling from a plain column
of ids.

**Attributes** live on the label they belong to: per element (VTK
`cell_data`), per fragment (duplicated when a crop splits a fragment),
per object (TRX `dps`, zarr-vectors `object_attributes`), per group (TRX
`dpg`). Neuroglancer `segment_properties` are object attributes and
`relationships` are a label over objects of another store.

### 2.4 The standard labels

```
Labels[T]
├── Elements(Labels[Vertices])     dense, ordered, atomic; own_axes = ("element",); + dim
│   ├── Singular                   implicit, W = 1, dim 0
│   ├── Edges                      W = 2, dim 1
│   │   └── SequentialEdges        implicit: i → i+1 along an axis or a fragment
│   ├── SurfaceElements            dim 2
│   │   ├── Triangles, Quads       W = 3, 4
│   │   └── Polygons               boxed: varying size (VTK_POLYGON)
│   ├── VolumeElements             dim 3
│   │   └── Tetrahedra             W = 4
│   └── GridElements               implicit: quads / hexahedra / Kuhn tets of a grid
├── Fragments(Labels[Vertices])    boxed, ordered; own_axes = ("fragment",)
├── Objects(Labels[Fragments])     own_axes = ("object",)
└── Groups(Labels[Objects])        own_axes = ("group",)
```

`Elements` adds one field to `Labels`, `dim`, and fixes `ordered` and
`atomic` to `True`. Dispatch to `Edges`, `Triangles`, `Quads`,
`Tetrahedra` is on `(dim, W)`: a quad and a tetrahedron both have
`W = 4` and differ in `dim`. Because elements are labels, a label over
elements (Gmsh physical groups, zarr-vectors `link_fragments`) is just
`Labels(of="elements", axes=("element",))`, and a mesh of polygons of
varying size is the boxed form of `SurfaceElements`, not a new concept.

`Vectors.labels` is a dict holding all of them. `elements`, `fragments`,
`objects` and `groups` are well-known keys exposed as properties
(`edges` on `Graphs` and `faces` on `SurfaceMeshes` alias `elements`);
any other labeling (a parcellation, FreeSurfer annotation labels on
surface vertices) is another entry. Several element blocks (tracks plus
divisions in example F, intra- plus cross-chunk links in example K,
triangles plus quads) are a tuple under `elements`, all of the same
`dim`. A categorical attribute column and a label are two forms of the
same information: the column is the cheap forward form, the label adds
names, per-label attributes and the reverse index. Either converts into
the other.

### 2.5 `Vectors`

```
class Vectors(Magic, Generic[E]):
    vertices:         Vertices
    labels:           dict[str, Labels] = {}   "elements" is an E (or a tuple of E)
    transformations:  list[Transformation] = ()

    elements, fragments, objects, groups -> labels[...]
```

`vertices` is lazy for the same reason `SingleScaleImage.data` is: a
reader derives it on first access. Arrays follow `ArrayProtocol`
(numpy, dask, cupy, torch), like image data.

| Image | Vectors | Note |
|---|---|---|
| `data` | `vertices.data` | |
| `shape` | `shape` | the vertex array axes `A` |
| `ndim` | `ndim` | dimension of the native space: raster axes + components (so `len(v.shape) != v.ndim` in general; open question 1) |
| `dtype` | `dtype` | component dtype |
| `__array__` | `__array__` | `vertices.data`, native coordinates; the docstring says so |
| `geometry` | `bounds` | box of the native space covered (§5.4) |

### 2.6 Structured columns

Coordinates that transformations act on (space, time) must be a
homogeneous float block. Two cases need more:

- a **categorical coordinate**: a channel or label that is sparse (each
  vertex has one value) rather than raster. It is a component whose axis
  is a `ChannelAxis`, stored as integer codes with the category names on
  the axis.
- **heterogeneous attributes**: mixed dtypes (a float radius, an int
  label, a bool flag).

The structured form is accepted for both, and is the natural on-disk
form for some formats. Internally the data are held as one array per
column group (continuous components / categorical components /
attributes), so that a transformation only ever sees the homogeneous
continuous block, every backend works (torch and cupy have no structured
dtypes), and a single attribute can stay lazy while another is loaded.
`to_structured()` / `from_structured()` convert at the edges.
Transformations are the identity on categorical axes.

### 2.7 Examples

The common cases use one index axis and no nd indices at all; the nd
features only appear when the data are genuinely nd.

**A. Point cloud.** `Points`.
`vertices.data (N, 3)`, `axes = [IndexAxis("n")]`, components RAS mm.
Elements are `Singular`; no labels.

**B. Streamlines (TRX).** `Streamlines`.
`vertices.data (N, 3)`; `fragments`: `indices (P,)` = TRX `offsets`,
`lengths` derived (1-D, CSR); elements `SequentialEdges(along="fragments")`;
`objects` = one per fragment; `groups`: TRX groups, `members (K,)`
object indices, `indices (NG,)`, overlapping, `exclusive = False`. dpv,
dps, dpg are vertex, object and group attributes.

**C. Cortical surface (GIFTI, FreeSurfer).** `TriangleMeshes`.
`vertices.data (N, 3)`, `elements.members (M, 3)`, `dim = 2`,
`directed = True` (winding). A FreeSurfer annotation is a label over
vertices with names and colours as label attributes.

**D. Tetrahedral mesh (Gmsh, meshio).** `TetrahedralMeshes`.
`elements.members (M, 4)`, `dim = 3`, `directed = True`. Physical groups
are labels over elements: `Labels(of="elements", axes=("element",))`.

**E. Skeletons (neuroglancer).** `Graphs`.
`vertices.data (N, 3)`, `elements.members (M, 2)`, `directed = True`
(parent → child); one fragment per segment, `objects` named by segment
id; `radius` a vertex attribute.

**F. Cell tracking: raster time, sparse space.** `Polylines` / `Graphs`.
Centroids of cells followed over `T` frames:
`vertices.data (T, N, 3)`, `axes = [TimeAxis("t") raster, IndexAxis("n")]`,
components `(x, y, z)`, `valid (T, N)` for cells that do not exist in a
frame. The native system is `(t, x, y, z)`.
- If slot `n` is a stable identity, track `n` is the box
  `indices = (0, n)`, `lengths = (T, 1)` of fragments over `("t", "n")`:
  `fragments.indices (N, 2)`, `lengths (N, 2)`, `ordered` along `t`.
  Track edges are `SequentialEdges(along="t")`, shared over `n`; an edge
  exists only between two valid vertices.
- Cell divisions add explicit edges across frames, as a second element
  block: `Edges.members (M, 2, 2)` with `axes = ("t", "n")`, one row
  `[[t, n], [t + 1, m]]` per division. Lineage trees are objects over the
  track fragments.
- A time-varying motion correction `T(t, x)` is a transformation of
  `(t, x, y, z)` that leaves `t` unchanged: `push` (§4.3) applies it frame
  by frame.

**G. Histological contours: raster slices.** `Polylines`, `TriangleMeshes`.
Contours drawn on `Z` sections: `vertices.data (Z, N, 2)`,
`axes = [SpaceAxis("z") raster, IndexAxis("n")]`, components `(x, y)`,
`valid` given as `lengths (Z,)`.
- Per-slice contour edges: `Edges.members (Z, M, 2)`, `axes = ("n",)`,
  `batch = ("z",)`, `valid lengths (Z,)`.
- A surface lofted between consecutive sections: `Triangles.members (M, 3, 2)`,
  `axes = ("z", "n")`.
- An oblique-sectioning affine that mixes `z` into `(x, y)` is pushed
  without leaving the raster; one that mixes `(x, y)` into `z` needs
  `sparsify("z")` first (§4.3).

**H. A deforming surface: shared topology.** `TriangleMeshes`.
`vertices.data (T, N, 3)` with raster `t`, one `elements.members (M, 3)`
with `axes = ("n",)` and `batch = ()`, shared by every frame.

**I. Multi-channel localisation microscopy: a categorical raster axis.**
`Points`.
`vertices.data (C, N, 2)`, `axes = [ChannelAxis("c", names=[...]) raster,
IndexAxis("n")]`, `valid lengths (C,)`. `v["GFP", 0:10, 0:10]` selects one
channel and a continuous box. `ChannelAxis` has no `names` today; adding
them is part of the `IndexAxis` work (open question 2).

**J. Raster space, sparse time.** `Points`.
Spike times on a `X × Y` electrode array: `vertices.data (X, Y, K, 1)`,
`axes = [SpaceAxis("x") raster, SpaceAxis("y") raster, IndexAxis("k")]`,
one component `t`, `valid lengths (X, Y)`. The mirror image of example F.

**K. A zarr-vectors store, chunk-native.** Any type.
The store layout itself, exposed lazily without stitching:
`vertices.data (Cx, Cy, Cz, K, 3)`,
`axes = [IndexAxis("cx"), IndexAxis("cy"), IndexAxis("cz"), IndexAxis("k")]`
(chunk axes are index axes: the vertices carry their full position),
`valid lengths (Cx, Cy, Cz)` from the per-chunk row counts.
- Fragments in range mode are boxes inside one chunk:
  `indices (F, 4) = (cx, cy, cz, k0)`, `lengths (F, 4) = (1, 1, 1, count)`.
  Explicit-list fragments use `members (R, 4)`, nd indices that may
  repeat (shared vertices). `members` is then 1-D (`K = (R,)`), so
  `indices (F,)` are plain CSR starts and `lengths` is optional: the
  `ndim(members)` rule at work. A store that mixes both modes is read in
  the explicit form, ranges becoming runs of `members`.
- Intra-chunk links: `elements.members (Cx, Cy, Cz, M, E)`, `axes = ("k",)`,
  `batch = ("cx", "cy", "cz")`, `valid lengths (Cx, Cy, Cz)`: one link
  block per chunk, as in `links/0/0.0.0`.
- Cross-chunk links: a second element block with `axes = ("cx", "cy",
  "cz", "k")` and `members (M, E, 4)`.
- Objects are labels over fragments (the object manifests), groups are
  labels over objects.
- A region read is first a slice of the chunk axes (cheap, like an image
  crop), then the exact vertex test. `stitch()` turns this into the flat
  form of examples A–E.

**L. A deformation field as a mesh.** `VolumeMeshes`.
A coordinates field of shape `(X, Y, Z, 3)` is a vertex array with three
index axes (the grid indices are not coordinates: the positions are in
the data). `GridElements(axes=("x", "y", "z"), cell="kuhn")` gives six
tetrahedra per grid cube without storing them; their signed volumes show
folding. The same field read with *raster* axes and no components would
be an image: points and images are the two ends of one model (§9).

### 2.8 Ragged layouts and backends

**Logical vs physical.** Many axes are ragged: points per frame, vertices
per chunk, vertices per streamline. The model describes them logically
(an index axis with `valid`, or a boxed label), and three physical
layouts carry them:

| Layout | Storage | Good for |
|---|---|---|
| **padded** | `(*outer, Lmax, ...)` + `valid` (mask or `lengths`) | nd raster layouts (examples F, G, I, J); fully vectorised on every backend |
| **packed** | flat `content (N, ...)` + `offsets (P + 1,)`, or nd `indices` + `lengths` | flat collections (TRX, meshes); compact, one buffer, memory-mappable |
| **nested** | an outer array of inner arrays | user code and per-chunk readers; converted on the way in |

Packed is the default in memory for flat data and padded for nd raster
data. Nested is accepted from user code and from chunk-native readers
and converted (pad ↔ pack ↔ nest are cheap and lossless). The packed
form is already in the model: a boxed label over a flat axis is exactly
`offsets` (§2.3), and stacked boxed labels are awkward-array's nested
`ListOffsetArray`.

**No storage wrapper; a view for convenience.** The datamodel stores the
raw arrays as fields (`members`, `indices`, `lengths`, `valid`, the
vertex `data`), never an "array of arrays" object. Plain arrays
serialise, load lazily, chunk in dask and move between backends without
a second source of truth, and multi-level raggedness is already the
composition of labels. For users, a label exposes a thin, read-only
view over its target:

```python
tracts = v.fragments.view(v.vertices)  # nibabel-ArraySequence-like
len(tracts)  # number of fragments
tracts[17]  # (L_17, D) array of vertices
tracts[mask]  # a smaller view (same buffers, new offsets)
tracts.lengths, tracts.pad(fill=nan), tracts.reduce("mean")
```

The view owns nothing: it holds references to the arrays and delegates
to private functional helpers (`_ragged.py`), which are also what the
datamodel itself calls.

**Helpers.** A handful of operations cover everything: `lengths`,
`segment_ids` (`repeat(arange(P), lengths)`), `segment_reduce` (segment
ids + `bincount` / scatter-add), `take` (gather whole lists, rebuild
offsets with `cumsum`), `mask_items` (filter, recount), `pad` / `pack`.
Each is written once against the array module returned by
`get_array_backend`, so numpy and cupy share the code. Data-dependent
sizes (masking, `Lmax`) force a host sync on cupy; operations that
produce them are grouped.

**dask: chunks aligned with lists.** `offsets` (or `indices` and
`lengths`) stay eager, on the host: they describe the graph and hold one
integer per list. `content` is chunked along the item axis so that **no
list straddles a chunk boundary**, with chunks
`((n_0, n_1, ...), (D,))`, each `n_b` the length of a run of whole lists.
Readers produce this directly (one chunk per stored chunk, or per batch
of streamlines); other arrays are rechunked once to list boundaries.
Then element-wise operations (`push`) stay element-wise, per-list
operations are `map_blocks` over local offsets with results chunked by
the number of lists per block, and item masks give unknown (`nan`) chunk
sizes until `compute_chunk_sizes()` is called once.

**dask over cupy** is the same graph with cupy blocks
(`meta=cupy.empty((0, D))`). Offsets stay numpy on the host; each block
moves its local offsets to the device, and its kernels take their module
from the block. This needs one addition to `backends.py`: the block
module of a dask array (read from its `_meta`), since
`get_array_backend` returns `dask.array` for both dask-over-numpy and
dask-over-cupy today.

---

## 3. Coordinate systems and transformations

### 3.1 Native space

The native space is the raster axes plus the components (§2.2). Vertices
are stored as the format stores them, never converted on read:

| Format | Native space | First transformation |
|---|---|---|
| zarr-vectors | the level's vertex coordinates (NGFF axes) | the level's `coordinateTransformations` (`scale`, `translation`) |
| neuroglancer multilod mesh | dequantized model space (see below) | `transform` (3×4, to the `info` dimensions) |
| neuroglancer skeleton | stored vertex positions | `transform` (3×4, to nm) |
| neuroglancer annotations | stored float32 positions | `dimensions` scales/units |
| TRX | positions | `VOXEL_TO_RASMM` is *not* applied: TRX positions are already RAS mm, so the list is `[Identity → RASmm]`; `DIMENSIONS`/`VOXEL_TO_RASMM` are kept as metadata describing a reference grid |
| TRK | voxmm (TrackVis) | voxmm → voxel → RAS mm (two transforms) |
| TCK | RAS mm | identity to RASmm |
| FreeSurfer surface | tkr-RAS | tkr-RAS → scanner RAS (from `c_ras` / volume info) |
| GIFTI | `coords` | the darray's `CoordinateSystemTransformMatrix` entries |

The neuroglancer multilod format quantizes vertices per octree node
(`vertex_quantization_bits`). The integer codes are an encoding detail,
not a coordinate system a user would want to keep, so the reader
dequantizes into the mesh's model space and that is the native space.
This is the one place where "store natively" bends, and it bends at the
codec boundary, like a NIfTI `scl_slope` would.

### 3.2 The transformation list

Identical semantics to images, and implemented once. Today the
`transformations` / `transformation` property and setter are duplicated
in `SingleScaleImage` and `MultiScaleImage`; a third and fourth copy are
not acceptable. **Proposal:** move them to a private mixin
(`datamodel/_placed.py`, name to bikeshed) used by images and vectors
alike. That refactor is a prerequisite and can land on its own.

---

## 4. Transforming: `__call__`, `push` and `pull`

### 4.1 Same lazy call, same convention as images

```python
# lazy; img2.transformation == T.inverse() @ img.transformation
img2 = img(T)
# lazy; v2.transformation == T.inverse() @ v.transformation
v2 = v(T)
```

`T` maps *new world → current world* (its output matches the current
world), in both cases. This is the registration convention: a
registration that reslices a moving image onto a fixed one yields `T:
fixed → moving`, and `moving_image(T)` and `moving_vectors(T)` both put
the object in fixed space. The same `T` that moves the image moves the
tracts that live with it.

`@` is composition with the right operand applied first, as in the
code. The current `SingleScaleImage.__call__` / `MultiScaleImage.__call__`
docstrings state the product the other way round
(`self.transformation @ transform.inverse()`); the code is right, and the
mixin refactor of §3.2 fixes the docstring.

### 4.2 Pull and push

Data meets a transformation in one of two directions:

- **pull**: for every sample of a *target*, evaluate the transformation
  there and fetch the value it points to. This is how images are
  resampled.
- **push**: move every sample of the *source* through the transformation
  to where it lands. This is how vertices are moved.

The cost lands on different transformations. Pulling an image into
fixed space evaluates `T` on the fixed grid. Pushing vertices into fixed
space evaluates `T.inverse()` at the vertices. For an affine both are
free. For a dense field the inverse is the expensive part: `T.inverse()`
is a lazy `Inverse`, and what exists today is the grid inversion
(`Inverse.field` materializes the whole inverse field by mesh inversion,
and the points are then sampled from it; `InverseCoordinatesField`
assumes grid units, so the world-to-voxel affine is composed in first).
A per-point fixed-point / Newton solve, cheaper for few points, is **new
work**. The docstrings say plainly that a field transform makes pushing
costlier than pulling, and the other way round for a field defined in
the other direction.

### 4.3 `Vectors.push`

```
def push(self, target=None, *, copy=False) -> Self    # new object
def push_(self, target=None) -> Self                  # in place
```

`push` moves the vertices into the native space of `target`:

```
x = (raster coordinates, continuous components)    native coordinates
y = (target.inverse() @ self.transformation)(x)
```

| `target` | Result's native space |
|---|---|
| `None` | the current world (transformation `Identity`) |
| `Transformation` | its input space |
| `Geometry` / `Image` | the image's voxel space (e.g. tracts in voxel indices for a TRK writer); only the transformation is used |
| `Vectors` | the other object's native space |

There is no vector `Geometry`: an image geometry carries a grid because
an image must know where to resample, and vertices are not resampled.

**Components are moved eagerly, raster axes lazily.** The raster axes
stay a grid only if the transformation keeps them on one. `push` factors
the transformation into axis groups (`compute(factor=True)`) and requires
that the output raster axes depend on the input raster axes alone,
through an axis-aligned affine `A` (a permutation, scaling and shift).
The array index of a raster axis does not change, so `A` is not applied
to anything: it stays in the result's transformation, which is

```
result.transformations = [target @ (A ⊕ Id_components)]
```

and is `target` exactly when `A` is the identity. The components are
computed per vertex, and may depend on the raster coordinates (example
F: a time-varying motion correction). A transformation that mixes
components into a raster axis raises, naming the axis; the user calls
`v.sparsify("z")` first, which turns the raster axis into an index axis
plus one more component without changing any index (example G).
`densify(component)` is the inverse for a component whose values lie on
a grid: it adds a new leading raster axis, pads the vertices into it
(`valid` lengths), and rewrites element and label indices to address it.

**Categorical axes are left out.** A `ChannelAxis`, raster or
component, is removed from `x` before composing and reattached after; a
transformation that names such an axis must be the identity on it.

`push_` writes into `vertices.data` and raises when it cannot: the
output dimension differs, the component dtype is not floating point, the
data are lazy, file-backed or shared with another object, or `A`
permutes or flips raster axes (which would transpose or reverse the
array).

Topology is carried over unchanged. Geometric attributes are transformed
with the local Jacobian `J` of the pushed transformation: attributes
tagged `vector` by `J`, `normal` by `J^{-T}` (renormalized); untagged
attributes are copied. When the pushed transformation is affine with a
negative determinant, directed elements with `dim >= 2` have their
vertex order flipped, so that windings and tet volumes keep their sign.
Nothing is flipped for edges, whose direction is semantic
(parent → child), nor per element under a nonlinear transformation:
there a local reversal is a fold, which must stay visible.

`Vectors.pull` is left out for now. Its natural meaning, sampling a
target image at the vertices (image → vertex attribute), is listed in
§9.

### 4.4 Images: `pull`, and later `push`

The same two words apply to images:

- `img.pull(geometry)` is today's `reslice`: it resamples the image on
  the target grid. `reslice` stays as a deprecated alias.
- `img.push(geometry)` is the adjoint: it splats the image values onto a
  target grid (rasterisation, or resampling under a transformation that
  is only known in the forward direction). It is a follow-up, not part
  of this design.

So images naturally pull and vectors naturally push, each under the
same `T` and the same lazy `__call__`.

---

## 5. Region selection: `__getitem__` and `crop`

### 5.1 Indexing the native space

```python
v[10.0:20.5, :, 3:7]  # three components: continuous box
v[0:5, 10.0:20.0, ...]  # raster t (frames 0-4), then components
v["GFP", 0:10, 0:10]  # categorical raster axis, then components
v[BoundingBox(lower, upper)]
```

`__getitem__` indexes the native coordinate system, in its order: raster
axes first, then components.

**On raster axes** it extends `Image.__getitem__`: integers (which drop
the axis), slices with steps, negative indices from the end, and, new,
names on a categorical axis. The vertex array is sliced along that axis
and the transformations are updated through `_index2transform` (which
takes `int | slice | None` today; name lookup resolves to an int first),
so the world placement is unchanged.

Slicing a raster axis also slices every relation that refers to it:

- **Integer index `t = i`.** In a relation whose `axes` include `t`, only
  rows whose `t` index equals `i` in *every* reference are kept, and the
  `t` column is dropped from the indices (`axes` shrinks; a `(M, 2, 2)`
  block over `("t", "n")` becomes `(M', 2)` over `("n",)`). A relation
  with `t` in `batch` is indexed at `i` along that batch dimension
  (`(Z, M, 2)` with `batch = ("z",)` becomes `(M, 2)` under `v[3]`).
  Shared relations are unchanged. In example F, `v[5]` keeps the
  centroids of frame 5, every division edge disappears (none lies
  within one frame), every track becomes a one-vertex fragment and
  `SequentialEdges(along="t")` becomes empty.
- **Slice `start:stop:step`.** References along `t` are renumbered
  `(i − start) // step`, and rows referring to a frame that is not kept
  are dropped (elements) or skip that member (ordered labels, which are
  not split by it). Batch dimensions are sliced in lockstep.
- **`isel` on index axes** follows the same two rules.

**On components** the space is continuous:

- each slice is a half-open interval `[start, stop)` in native units;
  integers are just numbers (`3:7` ≡ `3.0:7.0`); `None` bounds are
  unbounded;
- `step` must be `None`, and a bare scalar raises `TypeError`: a
  hyperplane selects nothing for a point set, and dropping an axis is a
  projection, a different operation (§9);
- negative bounds are coordinates, never "from the end";
- `start >= stop` gives an empty result; `NaN` bounds raise; vertices
  with a `NaN` coordinate are never inside;
- vertices are not shifted: the result keeps the same native space and
  transformations.

`None` (newaxis) raises; `...` fills unindexed axes with `:`; more
indices than `ndim` raises. Index axes are not coordinates and are not
addressed by `__getitem__`; `v.isel(n=slice(0, 10))` selects along them
by name.

> **Half-voxel caution.** In voxel space an image slice `a:b` keeps voxel
> *centres* `a … b-1`, i.e. the continuous interval `[a-½, b-½)`. A
> component bound `a:b` is the interval `[a, b)`. If components share
> their space with an image, the two differ by half a voxel. To crop
> vectors to exactly what an image crop covers, use `v.crop(img[a:b])`.
> The `__getitem__` docstring repeats this.

### 5.2 What "inside" means for elements and labels

`__getitem__` is `crop(box)` with the default mode:

```python
def crop(self, region, *, mode="inner", space=None, compact=None): ...
```

| `mode` | Kept vertices | Kept elements | Polylines effect |
|---|---|---|---|
| `"inner"` (default) | inside the region | elements whose vertices are **all** kept | streamlines are split at the boundary into fragments of the same object |
| `"outer"` | inside, plus every vertex of an element with **any** vertex inside | elements with any vertex inside | fragments extend one vertex past the boundary |
| `"object"` | all vertices of every object touching the region | all elements of those objects | whole streamlines through the region (tractography "ROI include") |
| `"exact"` | inside, plus new vertices on the boundary | elements clipped by the region, attributes interpolated | fragments end exactly on the boundary |

The rule is the same for every type: cropping decides which vertices are
kept, and that decision **propagates up the labels**, level by level,
with one rule that only looks at the `atomic` flag:

- an **atomic** label (an element) is kept when all its members are
  (`"inner"`) or when any is (`"outer"`, which then brings its other
  members back);
- a **non-atomic** label (a fragment, object or group) loses the members
  that were dropped, disappears when it has none left, and, when
  `ordered` and compacted, splits into several labels with the same
  parent (a split streamline stays one object).
`"exact"` is the only mode that creates vertices; it can come later.
The modes apply to the component bounds; raster indices always slice
exactly, as for images, so `"outer"` and `"object"` never bring back a
frame that the raster index left out.

**Masking vs compacting.** With `compact=False` the crop only updates
the `valid` masks: the nd layout, the boxes of the labels and the
indices of the elements are untouched, which is what a chunk-native or
raster layout wants (examples F, K). Holes in a box never split a label.
With `compact=True` dropped vertices are removed, element indices are
renumbered, an `ordered` label is split into its maximal runs (in C
order), and a surviving label that is no longer a box becomes a run of
an explicit 1-D `members`. The pieces of a split label keep the same
parent in the level above (a split streamline is still one object);
their `names` are copied and a `part` attribute numbers them.
`compact=None` (the default) compacts when every vertex array axis is
an index axis and there is only one of them (flat layouts), and masks
otherwise. On an nd layout, compaction only removes hyperslabs that are
entirely invalid: removing one cell from a single frame of `(T, N, 3)`
would shift the slots of that frame only and break the identity of
slot `n` across frames.

### 5.3 Regions in other spaces

`region` may be:

- a tuple of indices or a `BoundingBox` in **native** space (the
  `__getitem__` path);
- a `BoundingBox` with a coordinate system, or `space=` naming a
  transformation in the list: the box is in that **world** space;
- a `Geometry` / `Image`: the image field of view, i.e. the continuous
  voxel box `[-½, shape-½)` of its grid, placed by its transformation.

A world-space region is not an axis-aligned box in native space. The
test is done on pushed vertices. For chunked stores the chunk pre-filter
needs a native-space bound of the region: exact (box of the mapped
corners) when native-to-world is affine, and *all chunks* when it
contains a field, unless the field declares a displacement bound. The
exact per-vertex test is always done afterwards, so the pre-filter only
affects speed.

### 5.4 `BoundingBox` and `bounds`

A small immutable datamodel object, `BoundingBox(lower, upper,
system=None)`, half-open; without a system it is in native space, with
`__and__` (intersection), `contains`, and `to_slices()`.
`Vectors.bounds` is the box of the native space covered (raster extents
and component extents), read from metadata when the format stores it
(zarr-vectors `bounds`, neuroglancer `lower_bound`/`upper_bound`) so it
costs no vertex read.

---

## 6. Multi-scale

### 6.1 The container

```
class MultiScaleVectors(Vectors, Generic[T]):
    levels:           list[T]                finest first
    transformations:  list[Transformation]   pyramid -> world, as MultiScaleImage
```

Each level is a single-scale object with its own transformations (level
native → pyramid space), exactly as each `MultiScaleImage.images[i]`
carries its voxel-to-pyramid transformation. The pyramid space is the
finest level's native space (level 0 carries `Identity`), matching
`MultiScaleImage.geometry`, which is built from `images[0]`.

The semantics follow `MultiScaleImage`, not the `Multiscale` mixin, where
the two disagree: `to_singlescale(i)` composes the pyramid transformation
into the level (the mixin returns the raw stored scale), and `scales` is
a property yielding the composed levels (the mixin's `scales` is the
stored list). The stored list is therefore named `levels` here.
`nscales`, `transformation` (setter included) and `__call__` behave as
for `MultiScaleImage`; `push` pushes the level it selects (§6.2).

### 6.2 Two ways a level can be reduced

| Reduction | Example | Recorded per level |
|---|---|---|
| **coarsened**: every object is kept, with a simplified geometry | zarr-vectors coarsening (`bin_ratio`, metavertices); neuroglancer multilod mesh LODs | `resolution`: per-axis bin size (zarr-vectors `base_bin_shape × reduction_factor^l`; neuroglancer `lod_scales × lod_scale_multiplier`) |
| **subsampled**: a subset of the objects is kept | neuroglancer annotation `spatial` levels; zarr-vectors `object_sparsity < 1` | `sparsity`: fraction of the objects kept (finest = 1) |

A zarr-vectors level can do both at once, so every level carries a
`resolution` (or `None` when it is not coarsened) and a `sparsity`
(default 1). Level selection takes either criterion, or both:

- `resolution=` (or a target `Geometry`) picks, among the levels that
  record a resolution, the nearest one with `_nearest_resolution_index`.
  Levels with `resolution=None` are filtered out first, because that
  helper falls back to the finest level as soon as any resolution is
  `None`.
- `max_count=` or `sparsity=` restricts the choice to levels holding at
  most that many (or at least that fraction of) objects.

When both are given, the sparsity constraint filters and the resolution
picks among the levels left. With neither, the finest level is used.

Neuroglancer's annotation pyramid is *cumulative*: a coarse level holds
a random subset, and each finer level holds only what its parents did
not. The reader exposes level `i` as the union of levels coarse … `i`,
so that every level is self-contained and the finest one is complete.
The `sparsity` of an exposed level is its cumulative count over the
total. Users never see the "residual" encoding.

### 6.3 Cropping a pyramid

`MultiScaleVectors.__getitem__` / `crop` crop every level (lazily, when
file-backed) and return a `MultiScaleVectors`. The box is in the
pyramid's space and is mapped into each level's native space through
that level's transformation, which is affine for every format considered
here. `MultiScaleImage` has no `__getitem__` today; adding one with the
same "crop every level" rule is a natural follow-up for parity.

---

## 7. Laziness and chunked stores

A file-backed object can expose a store in two ways:

- **chunk-native** (example K): the vertex array keeps the chunk axes as
  index axes. A region read slices the chunk axes, which reads only the
  intersecting chunks, then masks (`compact=False`). Nothing is
  stitched, so this is cheap and lazy, and it is the natural view for
  processing chunk by chunk.
- **stitched** (examples A–E): `stitch()` (or the reader, on request)
  assembles fragments across chunks into flat runs, using the object
  manifests and the store's cross-chunk strategy (zarr-vectors explicit
  cross-chunk links, or boundary deduplication by coordinate match).

The chunk pre-filter maps the region to a native-space box (§5.3), then
to a range of the store's grid (zarr-vectors `chunk_shape` + grid origin;
neuroglancer spatial index `grid_shape`/`chunk_size`; multilod octree
nodes `chunk_shape × 2^level` from `grid_origin`). Objects that cross
chunk boundaries are the hard part, and the reason the `"object"` mode
exists: it needs the manifest of every touching object, which
zarr-vectors' `object_index/manifests` provides without reading the
other chunks' vertices first.

---

## 8. I/O

`brainhops.io.vectors` follows `brainhops.io.images`:

```python
from brainhops.io import vectors

# dispatch via FileBasedVectors registry
v = vectors.load("tracts.trx")
cls = vectors.sniff("brain.zv")
vectors.save(v, "tracts.zv")
```

`FileBasedVectors` is a `@format_registry` dispatcher registered with
`@register_parser(Vectors)`, with `PRIORITY` lower than images so that a
NIfTI is never read as vectors by accident (GIFTI and NIfTI pointsets
are the only overlap). A reader builds the plain data model: a
zarr-vectors store becomes a `Vectors` whose `labels` hold fragments,
objects and groups, not a store-specific subclass. Suggested order of
work, matching #130 / #175:

1. zarr-vectors (read/write, multiscale, chunk-native and stitched) — the
   reference for the design, since it exercises every feature here.
2. TRX, TRK, TCK (streamlines; via nibabel where possible).
3. GIFTI and FreeSurfer surfaces.
4. neuroglancer precomputed: skeletons, multilod meshes, annotations
   (read first; Draco decoding is an optional dependency).
5. meshio-backed volume meshes.

The zarr-vectors spec is a draft and its package is alpha; readers pin
a version and keep the spec-to-model mapping in one module.

---

## 9. What is deliberately *not* in this design

- **Projection / slicing to a lower dimension** (a 2-D section of a 3-D
  mesh, a scalar index on a component). It changes the type, so it gets
  its own method later (`section(axis, value)`), not an indexing
  overload.
- **`Vectors.pull`**: sampling an image at the vertices (image → vertex
  attribute), and **`Image.push`**: splatting (vectors or images → image,
  density maps, label volumes).
- **Parametric annotations** (boxes, ellipsoids), see §2.1.
- **Fields as vectors**: `Points.from_field(f)` and
  `TetrahedralMeshes.from_field(f)` (example L) as constructors.
- **Images as vectors.** An image is the limit of this model with only
  raster axes and no components; a point cloud is the other limit. The
  shared machinery (`__getitem__` on raster axes, the transformation
  list, `pull`/`push`) is designed so that a later unification is
  possible, but the classes are not merged now.
- **Writing multiscale pyramids** (building LODs). Reading pyramids is
  in scope; generating them is a separate tool.

---

## Open questions

1. **Name of the component count.** `Vertices.ncomponents` is used here
   for `D` (the explicit coordinates), and `ndim` for the full native
   dimension (raster axes + components), which differs from
   `len(shape)` (the array axes). Alternatives: `embedding_dim`,
   `ncoords`.
2. **`IndexAxis`, and names on `ChannelAxis`.** `IndexAxis` is a new
   axis type that is never part of a coordinate system; `ChannelAxis`
   needs category `names` (example I). Do both belong in `axes.py`, or
   should vertex array axes be described by a lighter structure?
3. **Default `crop` mode.** `"inner"` (proposed) vs `"object"`, which is
   what tractography users usually mean by "streamlines in a ROI".
4. **`__array__` returning native components.** Convenient, but
   `np.asarray(v)` silently ignoring the transformations and the raster
   axes may surprise. Alternative: no `__array__`.
5. **Optional awkward-array backend.** awkward (and dask-awkward) work
   on the packed layout natively. Conversions to and from them are
   nearly copy-free; is it worth an optional backend, or only
   `to_awkward()` / `from_awkward()`?
6. **Mixed-type stores.** A zarr-vectors store may declare several
   `geometry_types`. Load as a dict of objects by type, or require
   `load(..., type=...)`?
7. **Per-frame explicit members.** In the boxed form `members` has no
   batch axes, so an explicit member list cannot differ per frame
   (§2.3); the dense form can. Nothing in the examples needs it; adding
   `batch` to boxed `members` is possible later.
