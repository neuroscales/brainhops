# Design: vector objects (points, polylines, meshes)

**Status:** design only, no code. Reviewed once (Fable); review
findings are folded in. Relates to #130 (general vector
formats), #175 (zarr-vectors) and the per-category format issues
(points, streamlines, meshes). Decisions still open are listed under
[Open questions](#open-questions).

This memo proposes a data model for *vector objects*: point clouds,
polylines (streamlines), skeletons, triangle surface meshes and
tetrahedral volume meshes. It aims for three things:

1. **One API shape shared with images.** A vector object stores its
   vertices in their native coordinate system and carries a list of
   native-to-world transformations, exactly as an image stores voxels
   and carries voxel-to-world transformations. The same names
   (`transformations`, `transformation`, `__call__`, `reslice`,
   `__getitem__`, `to_singlescale`, `load`/`save`) mean the same thing
   wherever that is possible, and differ in a documented way where the
   geometry forces it.
2. **Region queries.** `vectors[x0:x1, y0:y1, z0:z1]` selects the part of
   the object inside a box, with *continuous* bounds in the native space
   of the vertices. On a file-backed, chunked store (zarr-vectors,
   neuroglancer precomputed) it only reads the chunks that intersect the
   box.
3. **Multi-scale.** A pyramid of levels, where a level is either a
   geometric simplification (zarr-vectors coarsening, neuroglancer mesh
   LODs) or a sub-sample of the objects (neuroglancer annotation spatial
   levels).

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
| `img.reslice(geom)` | resample data onto `geom`; result carries `geom.transformation` |
| `img[index]` | crop/stride in voxel space; transformations are updated so the world placement is unchanged |
| `MultiScaleImage` | `images` (levels) + transformations of the pyramid; `reslice` picks the level nearest to the target resolution |
| `io.images.load/sniff` | format dispatch through `FileBasedImage` |

The observation that lets the vector model reuse the existing machinery:
**a point array can be evaluated through the transformation composers.**
The composers that apply a transformation to a `CoordinatesField`
(`Affine @ CoordinatesField` is matrix arithmetic; the field composers
call `pull_field(field, coords=...)`, which accepts any `(..., D)`
coordinates) already compute `T(x)` for arbitrary points. A private
helper, `_apply_to_points(T, points) -> points`, wraps the `(N, D)` array
in a throw-away `CoordinatesField` (one-axis input system, `coeff=False`
forced), composes, computes, and returns `.field`.

The vertices are **never** stored or exposed as a `Transformation`. Much
of the transformation code assumes a field is defined on a grid with as
many axes as components (`_fields_as_identity`, `InverseCoordinatesField`
mesh inversion, `order`/`bound`/`coeff` metadata, and spline prefiltering
when `coeff` is set, which would run *along the vertex index*). Treating
vertices as a transformation would leak all of that.

---

## 2. Class hierarchy

```
Vectors                          (abstract; ≈ Image)
├── SingleScaleVectors           (abstract; ≈ SingleScaleImage)
│   ├── PointCloud               cells: none (each vertex is its own object)
│   ├── Polylines                cells: implicit edges i → i+1 inside a piece
│   │   └── Streamlines          alias / thin subclass, tractography vocabulary
│   ├── Skeleton                 cells: explicit edges (M, 2); trees / graphs
│   └── Mesh                     cells: explicit simplices (M, k)
│       ├── SurfaceMesh          k = 3 (triangles)
│       └── VolumeMesh           k = 4 (tetrahedra)
└── MultiScaleVectors[T]         (≈ MultiScaleImage; levels of one type T)
```

Every single-scale class is a *simplicial complex*: vertices plus cells
of a fixed width `k` (1 = point, 2 = edge, 3 = triangle, 4 = tet). This
is the same notion as zarr-vectors' `link_width` and lets one
implementation of cropping, attribute handling and topology bookkeeping
serve all types. A subclass only says how its cells are stored
(implicit for points and polylines, explicit for the rest) and what its
intrinsic dimension is.

Parametric shapes (neuroglancer `AXIS_ALIGNED_BOUNDING_BOX`,
`ELLIPSOID`) are **out of scope** for this memo: they are not closed
under nonlinear transformations, so they do not fit "store natively,
transform lazily". A later `Annotations` class can hold them and offer
`to_mesh()`; neuroglancer `POINT`, `LINE` and `POLYLINE` annotations map
onto `PointCloud`, `Skeleton` (or `Polylines`) here.

### 2.1 Fields of a single-scale object

Every type uses the same layout: vertices grouped into **pieces** by a
CSR `offsets` array, an object id per piece, and (for types with
explicit cells) cells grouped by piece the same way.

```python
class SingleScaleVectors(Vectors):
    # (N, D) float, native coordinates
    vertices: Optional[ArrayProtocol]
    # (P + 1,) CSR pieces over vertices
    offsets: Optional[ArrayProtocol] = None
    # (P,) object id of each piece
    ids: Optional[ArrayProtocol] = None
    # native -> world, last preferred
    transformations: List[Transformation] = ()
    # each (N, ...), (P, ...); kinds: see 4.4
    vertex_attributes: Dict[str, ArrayProtocol] = {}
    piece_attributes: Dict[str, ArrayProtocol] = {}
    attribute_kinds: Dict[str, AttributeKind] = {}


# base of Skeleton and Mesh
class _ExplicitCells(SingleScaleVectors):
    # (M, k) int, indices into the global vertex array
    cells: Optional[ArrayProtocol] = None
    # (P + 1,) CSR pieces over cells
    cell_offsets: Optional[ArrayProtocol] = None
    # each (M, ...)
    cell_attributes: Dict[str, ArrayProtocol] = {}
```

| Class | A piece is | Cells |
|---|---|---|
| `PointCloud` | one point (`offsets = arange(N + 1)` by default) | none |
| `Polylines` | one polyline run | implicit edges `i → i+1` inside the piece |
| `Skeleton` | one connected component / stored fragment | `cells: (M, 2)` |
| `SurfaceMesh`, `VolumeMesh` | one connected component / stored fragment | `cells: (M, 3)` / `(M, 4)` |

Defaults keep simple cases simple: `offsets=None` means one piece holding
every vertex (one point per piece for `PointCloud`), and `ids=None` means
`arange(P)`. Cell indices refer to the global vertex array, zero-based;
their dtype is preserved from the store, default `int64`. Vertex dtype is
preserved too (`float32` for neuroglancer after dequantization).

`ids` separate *pieces* from *objects*. A streamline cut in two by a
crop becomes two pieces with the same id, so per-object data (`dps` in
TRX vocabulary) still refers to the right object. `piece_attributes` is
indexed by piece and duplicated on split, so it stays a plain array
aligned with `offsets`; `unique(ids)` gives the objects. This is
zarr-vectors' model: a *fragment* is a run of an object's vertices, and
the object manifest lists fragments. Neuroglancer annotation
`relationships` become piece attributes.

`vertices` is optional for the same reason `SingleScaleImage.data` is:
a reader derives it lazily. Arrays follow `ArrayProtocol` (numpy, dask,
cupy, torch), like image data.

**Dimensions.** `D` (native dimension, `ndim`) may differ from the
world dimension. 2-D data (histology ROIs, slice contours) is supported;
a cell width must satisfy `k <= D + 1`, so `VolumeMesh` needs `D >= 3`.

### 2.2 Array-like API (parity with `Image`)

| Image | Vectors | Note |
|---|---|---|
| `data` | `vertices` | not renamed to `data`: vectors have several arrays |
| `ndim` | `ndim` | dimension of the native space (`D`) in both cases |
| `dtype` | `dtype` | vertex dtype |
| `shape` | — | no single shape; use `nvertices`, `ncells`, `npieces`, `nobjects` |
| `__array__` | `__array__` | returns `vertices` (native coordinates); the docstring says so |
| `grid` | — | see `bounds` |
| `geometry` | `bounds` | axis-aligned box of the vertices, native space (§5) |

`len(vectors)` is the number of pieces (points for a `PointCloud`), and
iteration yields pieces, matching nibabel's `ArraySequence`.

---

## 3. Coordinate systems and transformations

### 3.1 Native space

Vertices are stored as the format stores them, never converted on read:

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

### 3.3 World coordinates

```python
# (N, D') vertices in the preferred world space
v.world()
# in the output space of another transformation in the list
v.world(space="RASmm")
```

`world()` is `_apply_to_points(v.transformation, v.vertices)` (§1), so
nonlinear transformations are applied by the composers that already
exist. There is deliberately no `world(T)`: a transformation argument
would run *native → X*, the opposite of `v(T)` in §4.1. Moving to
another space is `v(T).world()`.

---

## 4. Transforming: `__call__` and `reslice`

### 4.1 Same call, same convention as images

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
the object in fixed space. Users never have to think about direction:
the same `T` that moves the image moves the tracts that live with it.

`@` is composition with the right operand applied first, as in the
code. The current `SingleScaleImage.__call__` / `MultiScaleImage.__call__`
docstrings state the product the other way round
(`self.transformation @ transform.inverse()`); the code is right, and the
mixin refactor of §3.2 fixes the docstring.

### 4.2 Why "reslicing goes the other way"

The *cost* is what differs. To resample an image in fixed space one
evaluates `T` at fixed-space grid points (pull). To move vertices into
fixed space one evaluates `T.inverse()` at the vertices (push). For an
affine this is free. For a dense field it is the expensive part:
`T.inverse()` is a lazy `Inverse`. What exists today is the grid
inversion: `Inverse.field` materializes the whole inverse field (mesh
inversion, `InverseDisplacementField` / `InverseCoordinatesField`) and
the points are then sampled from it. `InverseCoordinatesField` assumes
coordinates in grid units, so the world-to-voxel affine must be composed
in first. A per-point fixed-point / Newton solve, cheaper for few
points, is **new work** (a composer for `Inverse*Field` evaluated at
points), not an existing path. The API requirement is only that
`v(T).world()` works when `T` contains a field; the docstrings say
plainly that a field transform makes vectors costlier than images (and
the other way round for a field defined in the other direction).

### 4.3 `reslice`: re-express vertices in another native space

```python
def reslice(self, target=None, *, copy=False) -> Self
```

Images: `reslice(geometry)` produces data whose native space is the
target grid, and whose transformation is `geometry.transformation`.
Vectors do the same with vertices:

```
new_vertices = _apply_to_points(target.inverse() @ self.transformation, vertices)
result.transformations = [target]
```

`target` accepts what `Image.reslice` accepts, read the same way:

| `target` | Result's native space |
|---|---|
| `None` | the current world: vertices "baked" into world coordinates, transformation `Identity` to that world |
| `Transformation` | its input space |
| `Geometry` / `Image` | the image's voxel space (e.g. tracts in voxel indices for a TRK writer) |
| `Vectors` | the other object's native space |

So `reslice` is the eager counterpart of `__call__`, exactly as for
images, and "baking" a transform is `v(T).reslice()`. Like
`Image.reslice`, it does not crop: tracts in an image's voxel space and
restricted to its field of view are `v.crop(img).reslice(img)`.
Topology and attributes are carried over unchanged, except as in §4.4.

When `self.transformation` holds a multiscale field, `_at_resolution`
picks the level matching the target grid only if `target` is a
`Geometry` (or `Image`); otherwise the finest level is used, as for
images.

### 4.4 Attributes that are geometric

Some vertex attributes are not scalars: tangents, normals. In a first
version `attribute_kinds` tags them as `scalar` (default), `vector` or
`normal`, and `reslice` transforms them with the local Jacobian of the
native-to-new transformation: vectors by `J`, normals by `J^{-T}`
(renormalized). Untagged attributes are copied. Tensors, radii/lengths
and fixing triangle winding under orientation-reversing transformations
are left for later.

---

## 5. Region selection: `__getitem__` and `crop`

### 5.1 Continuous indices in native space

```python
v[10.0:20.5, :, 3:7]
v[..., 0:100]
v[BoundingBox(lower, upper)]
```

`Image.__getitem__` indexes the voxel (native) space; `Vectors.__getitem__`
indexes the native space too, but the space is continuous:

- each slice is a half-open interval `[start, stop)` in native units;
  integers are just numbers (`3:7` ≡ `3.0:7.0`); `None` bounds are
  unbounded;
- `step` must be `None` (a step has no meaning for a region; raising
  is better than guessing);
- a bare scalar raises `TypeError`: a hyperplane selects nothing for a
  point set, and dropping an axis (what an integer does to an image) is
  a projection, which is a different operation (§9);
- `None` (newaxis) raises; `...` fills unindexed axes with `:`; more
  indices than `D` raises;
- negative bounds are coordinates, **never** "from the end" (an explicit
  divergence from `_index2transform`);
- `start >= stop` gives an empty result, not an error; `NaN` bounds
  raise; vertices with a `NaN` coordinate are never inside, in every
  mode;
- an empty result has `vertices.shape == (0, D)`, `offsets == [0]`, and
  `bounds is None`.

The `__getitem__` docstring repeats the half-voxel caution below, so it
is seen where it bites, not only in this memo.

Unlike `Image.__getitem__`, the result keeps **the same native space and
the same transformations**: vertices are not shifted. An image crop has
to re-index because array positions are its coordinates; vertices carry
their own coordinates, so nothing needs re-basing.

> **Half-voxel caution.** In voxel space an image slice `a:b` keeps voxel
> *centres* `a … b-1`, i.e. the continuous interval `[a-½, b-½)`. If a
> vector object shares its native space with an image, `v[a:b]` and
> `img[a:b]` therefore differ by half a voxel. That is deliberate: a
> vector index is a coordinate, not a voxel. To crop vectors to exactly
> what an image crop covers, use `v.crop(img[a:b])` (§5.3).

### 5.2 What "inside" means for cells

`__getitem__` is `crop(box)` with the default mode. `crop` exposes the
policy:

```python
def crop(self, region, *, mode="inner", space=None) -> Self
```

| `mode` | Kept vertices | Kept cells | Polylines effect |
|---|---|---|---|
| `"inner"` (default) | inside the region | cells whose vertices are **all** kept (induced sub-complex) | streamlines are split at the boundary into pieces sharing an id |
| `"outer"` | inside, plus every vertex of a cell that has **any** vertex inside | cells with any vertex inside (closure) | pieces extend one vertex past the boundary |
| `"object"` | all vertices of every object touching the region | all cells of those objects | whole streamlines that pass through the region (tractography "ROI include") |
| `"exact"` | inside, plus new vertices on the boundary | cells clipped by the region, attributes interpolated | pieces end exactly on the boundary |

`"inner"` is the default because, like an image crop, it never returns
geometry outside the requested region and it is cheap (a vertex mask
plus a cell mask). `"exact"` is the only mode that creates vertices; it
can come later. Vertex, cell and piece attributes are subset with the
same masks, kept cells are **renumbered** to the kept vertices, `offsets`
and `cell_offsets` are rebuilt (a piece split by the region becomes
several pieces), and `ids` keep provenance.

### 5.3 Regions in other spaces

`region` may be:

- a tuple of slices or a `BoundingBox` in **native** space (the
  `__getitem__` path);
- a `BoundingBox` with a coordinate system, or `space=` naming a
  transformation in the list: the box is in that **world** space;
- a `Geometry` / `Image`: the image field of view, i.e. the continuous
  voxel box `[-½, shape-½)` of its grid, placed by its transformation.

A world-space region is not an axis-aligned box in native space. The
test is done on transformed vertices (`world(space)` then box test, or,
for a `Geometry`, `geometry.transformation.inverse()` applied to world
vertices then voxel-box test). For chunked stores the chunk pre-filter
needs a native-space bound of the region: exact (box of the mapped
corners) when native-to-world is affine, and *all chunks* (with a
warning-free fallback) when it contains a field, unless the field
declares a displacement bound. The exact per-vertex test is always done
afterwards, so the pre-filter only affects speed.

### 5.4 `BoundingBox` and `bounds`

A small immutable datamodel object, `BoundingBox(lower, upper,
system=None)`, half-open; without a system it is `D`-dimensional and in
native space, with `__and__` (intersection), `contains`,
and `to_slices()`. `Vectors.bounds` returns the box of the vertices in
native space, read from metadata when the format stores it
(zarr-vectors `bounds`, neuroglancer `lower_bound`/`upper_bound`) so it
costs no vertex read. `Geometry` could gain a matching `bounds` (the
field-of-view box) later; that is not required here.

---

## 6. Multi-scale

### 6.1 The container

```python
class MultiScaleVectors(Vectors, Generic[T]):
    # finest first
    levels: List[T] = ()
    kind: Literal["geometric", "sparse"] = "geometric"
    # pyramid -> world, as MultiScaleImage
    transformations: List[Transformation] = ()
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
stored list). The stored list is therefore named `levels` here, which
avoids both the clash and `MultiScaleImage`'s type-specific `images`.
`nscales`, `transformation` (setter included), `__call__` and `reslice`
behave as for `MultiScaleImage`.

### 6.2 Two kinds of levels

| Kind | Example | A coarse level is | Resolution of a level |
|---|---|---|---|
| **geometric** | zarr-vectors coarsening (`bin_ratio`, metavertices); neuroglancer multilod mesh LODs | a simplified version of *every* object | per-axis bin size (zarr-vectors `base_bin_shape × reduction_factor^l`; neuroglancer `lod_scales × lod_scale_multiplier`) |
| **sparse** | neuroglancer annotation `spatial` levels; zarr-vectors `object_sparsity < 1` | a *subset* of the objects, at full precision | none; carries `sparsity` (fraction kept) instead |

A pyramid has **one** kind, stated by `kind`; levels of both kinds are
never mixed. This matters for selection: `_nearest_resolution_index`
falls back to the finest level as soon as any resolution is `None`, so a
mixed list would always select level 0.

- **geometric:** each level records `resolution`. Given a target
  resolution (from a `Geometry`, as in `MultiScaleImage.reslice`, or an
  explicit `resolution=`), the level whose resolution is nearest is
  picked with `_nearest_resolution_index`.
- **sparse:** each level records `sparsity` (fraction of the objects it
  holds, finest = 1). The level is chosen by `max_count=` or
  `sparsity=`.

A zarr-vectors store whose levels both coarsen and drop objects is read
as geometric; `sparsity` is still reported per level as metadata.

Neuroglancer's annotation pyramid is *cumulative*: a coarse level holds
a random subset, and each finer level holds only what its parents did
not, so the complete set is the union of all levels. The reader exposes
level `i` as the union of levels coarse … `i`, so that every level is a
self-contained object and the finest one is complete. The `sparsity`
of an exposed level is its cumulative count over the total. Reading
level `i` also reads every coarser level, which is cheap by design
(coarse levels are small). Users never see the "residual" encoding.

### 6.3 Cropping a pyramid

`MultiScaleVectors.__getitem__` / `crop` crop every level (lazily, when
file-backed) and return a `MultiScaleVectors`. The box is in the
pyramid's space (the space all levels map into), and is mapped into
each level's native space through that level's transformation, which is
affine for every format considered here. `MultiScaleImage` has no
`__getitem__` today; adding one with the same "crop every level" rule
is a natural follow-up for parity, not part of this design.

---

## 7. Laziness and chunked stores

The in-memory classes implement `crop` with masks. The file-backed
classes (`FileBasedVectors` subclasses) override the region path so that
nothing outside the region is read:

1. map the region to a native-space box (§5.3), and to a range of
   chunks of the store's grid (zarr-vectors `chunk_shape` + grid origin;
   neuroglancer spatial index `grid_shape`/`chunk_size`; multilod octree
   nodes `chunk_shape × 2^level` from `grid_origin`);
2. read only those chunks and assemble fragments into pieces using the
   object manifests and the cross-chunk strategy (zarr-vectors explicit
   cross-chunk links, or boundary deduplication by coordinate match);
3. apply the exact per-vertex test and the cell `mode`.

The result of `file_vectors[box]` is an in-memory single-scale object
(or a lazy one backed by dask arrays when the reader supports it), the
same way `SingleScaleImage.__getitem__` returns a plain image.

Objects that cross chunk boundaries are the hard part, and the reason
the `"object"` mode exists: it needs the manifest of every touching
object, which zarr-vectors' `object_index/manifests` provides without
reading the other chunks' vertices first. For neuroglancer meshes,
`"object"` is natural (one mesh per segment id), while `"inner"` /
`"outer"` operate on whole octree fragments plus the vertex test.

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
are the only overlap). Suggested order of work, matching #130 / #175:

1. zarr-vectors (read/write, multiscale, chunked region reads) — the
   reference for the design, since it covers every type here.
2. TRX, TRK, TCK (streamlines; via nibabel where possible).
3. GIFTI and FreeSurfer surfaces (`SurfaceMesh`).
4. neuroglancer precomputed: skeletons, multilod meshes, annotations
   (read first; Draco decoding is an optional dependency).
5. meshio-backed volume meshes (`VolumeMesh`).

The zarr-vectors spec is a draft and its package is alpha; readers pin
a version and keep the spec-to-model mapping in one module.

---

## 9. What is deliberately *not* in this design

- **Projection / slicing to a lower dimension** (a 2-D section of a 3-D
  mesh, a scalar index in `__getitem__`). It is a well-defined operation
  (intersection with a hyperplane gives points from polylines, polylines
  from surfaces), but it changes the type, so it gets its own method
  later (`section(axis, value)`), not an indexing overload.
- **Rasterization** (vectors → image: density maps, label volumes) and
  **sampling an image at vertices** (image → vertex attribute). Both are
  natural next steps and both reduce to operations this design provides
  (`reslice` onto an image's geometry, then gather/scatter).
- **Parametric annotations** (boxes, ellipsoids), see §2.
- **Writing multiscale pyramids** (building LODs). Reading pyramids is
  in scope; generating them is a separate tool.

---

## Open questions

1. **`reslice` name for vectors.** Keeping the name gives parity
   (`obj.reslice(target)` works for any object); its effect on vertices
   is a re-expression, not a resampling. Alternative: `to_space`, with
   `reslice` as an alias. Recommendation: keep `reslice`.
2. **`__array__` returning native vertices.** Convenient, but
   `np.asarray(v)` silently ignoring the transformations may surprise.
   Alternative: no `__array__`, explicit `v.vertices` / `v.world()`.
   Recommendation: keep it, as images do the same with `data`.
3. **Default `crop` mode.** `"inner"` (proposed) vs `"object"`, which is
   what tractography users usually mean by "streamlines in a ROI".
4. **Piece vs object attributes.** Indexing per-object data by piece
   (`piece_attributes`, duplicated on split) keeps arrays aligned;
   indexing by object id avoids duplication but needs an id → row map.
   A dense `object_attributes` indexed by id could be added beside it.
5. **Integer-bound warning.** `v[3:7]` is continuous `[3, 7)`, which
   differs from `img[3:7]` by half a voxel. Should `__getitem__` warn
   once when every bound is an integer and the native space is a voxel
   space? Proposed: no warning, docstring only.
6. **Mixed-type stores.** A zarr-vectors store may declare several
   `geometry_types`. Load as a dict of objects by type, or require
   `load(..., type=...)`?
