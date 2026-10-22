# Design: conversions between the classes of the data model

Status: decided and implemented, except for the open question at the
end. This memo records how an object of the data model is turned into
an object of another class: an axis into a more specific axis, a
linear map into a rotation, a transformation into a file format. It
fixes what `to()`, `from_instance()` and `from_any()` mean, and how
they relate to each other. Decisions are tagged `C1` to `C9`.

## 1. The problem

Two mechanisms existed side by side. Transformations had a graph of
converters, run by `convert()` and reached through `to()`, which knows
how each class stores its map and checks what each class promises.
Every class also had a `from_instance()`, inherited from
`DataModelBase`, which copied the fields of any object by name. The
copy is correct when two fields with the same name mean the same
thing, which holds for axes and coordinate systems, but not across
transformations. `Scaling.from_instance(Translation([1, 2, 3]))` built
the scaling `[1, 2, 3]`, because both classes call their parameter
`data`, and `Rotation.from_instance()` accepted a matrix that is not
orthogonal.

## 2. Principles

C1. One operation, three spellings. For an object `x` of the data
model, `x.to(C)`, `C.from_instance(x)` and `C.from_any(x)` give the
same result or raise the same error. `from_any()` also reads a file,
its content, or a mapping, and passes any other value to the
constructor. `from_instance()` takes an object of the data model only,
and raises `TypeError` for anything else.

C2. A conversion is chosen by the pair of classes. Transformations
dispatch on the pair (class of the source, requested class) with the
converters of `compute/converters.py`, as before. The other families
have no converters, because their classes differ only by fixed fields.
Their conversion is the copy of fields by name that
`DataModelBase.from_instance()` makes. The two dispatchers stay
separate because no conversion crosses families: an image never
becomes a transformation through `to()`.

C3. A copy by name is made between related classes only. Two classes
are related when one is a parent of the other, or when both belong to
the same polymorphic family, that is, to the subclasses of a class
such as `Axis` whose calls build the subclass that the arguments
select. For a file format, the class that is compared is the data
model that it stores, such as `SingleScaleImage` for `NiftiImage`, so
two formats of the same data model are related. Copying between
unrelated classes raises `ConversionError` immediately, without a
deprecation period. The copy also checks the fields that the target
class fixes, so `RASmm.from_instance(LPSmm())` and
`TimeAxis.from_instance(LeftToRightAxis())` raise `ValueError`.

C4. A transformation is converted by its converters. A transformation
that is not an instance of the requested class, or of the data model
of a requested file format, goes through `to()`. The converters check
what the target class promises, so a non-orthogonal matrix is not a
rotation. An instance of the class, or of the data model of a format,
is copied, because the converters of a format build the format that
way (`io/transformations/base/_converters.py`).

C5. The result is an instance of the requested class, not necessarily
of exactly that class. A polymorphic class may build one of its
subclasses, as `Affine(..., log=True)` builds an `AffineExponential`,
and `SpaceAxis.from_instance(LeftToRightAxis())` keeps the left to
right axis. `convert()` checks `isinstance(result, cls)`.

C6. Conversions are exact by default, and a loss is opted into. A
transformation raises `LossyConversionError` unless `lossy=True` is
given, and the error carries the lossy result. Metadata follows its
loss policy, which warns by default, because a lost metadata field is
minor where a lost parameter changes the geometry. A converter that
goes through intermediate classes reports a loss with a result of the
requested class, not of an intermediate one.

C7. Converting an object to its own class is not a copy. `x.to()` and
`x.to(type(x))` without any field return `x` itself, and
`x.to(copy=True)` returns a new object. `C.from_instance(x)` always
builds a new object. Within a transformation format, a copy keeps the
stored data, so a record that was not decoded stays so. One rule in
`TransformationFormat.from_instance()` does this for every format that
stores its data in `_data`. The NIfTI classes keep their own override,
because their data model comes first in their method resolution order.

C8. Lazy wrappers are not conversion targets. `InverseAffine` and the
other classes that `inverse()` and `sqrt()` return hold the
transformation they derive from, not the parameters of a map.
`x.to(InverseAffine)` raises `ConversionError` and says to call
`x.inverse()`. A wrapper converted to its own class is returned as it
is (C7).

C9. The graph of conversions is tested for completeness.
`tests/test_transformation_conversion_graph.py` converts every pair of
concrete transformation classes, from an identity and from a general
transformation. Each pair converts exactly, reports a loss with a
result of the requested class, or is listed as refused with its
reason. A missing converter therefore fails the test.

## 3. What each family does

| family | how its classes differ | conversion |
|---|---|---|
| axes, systems, orientations | by fixed fields | copy by name with checks (C3) |
| metadata | by the fields a format stores | copy by name with a loss report |
| images | by the file record, or single scale against pyramid | copy by name between formats; `to()` between a single-scale image and a pyramid |
| transformations | by representation and by promises on values | converters (C4) |

A requested tangent class implies `log=True`, so `x.to(AffineExponential)`
is `x.to(Affine, log=True)`. `convert()` adds the flag at every step of
a multi-step conversion, unless the source already holds a tangent.

A transformation converted to a chain, as in `t.to(Sequence)`, becomes a
chain of one element. A file format that stores a chain decides with its
own converters what it holds, so a single affine is not written as a
chain file of one element, which would make `save()` ambiguous between
chain formats and single-affine formats that share an extension.
Whether a chain belongs to a kind is decided in three steps, for every
kind. The chain is bridged first, because two neighbours that do not
share a coordinate system are joined by a flip or a permutation, which
is not an identity. If every element of the bridged chain is an
identity, asked at the analytic level and then by value with
`compute=True`, the chain is the identity and nothing is composed.
Otherwise, with `compute=True`, the chain is composed and the result is
asked about the kind. A chain that reduces to the identity therefore
converts exactly to `Identity`.

Images follow the pattern of transformations: the conversions that
change how an image is stored are made by `Image.to()`, and
`from_instance()` hands them to it, both in `Image` and in
`ImageFormat`, which comes before the data model in the method
resolution order of every image format. A single-scale image becomes a
pyramid of one level. The level keeps its own transformations and the
pyramid adds none, so the image keeps its geometry. A pyramid becomes
the single-scale image of its highest-resolution level, placed in the
world space of the pyramid, and its other levels are left out.

Every affine file format has an exact converter from each family with
an affine form, and from a chain of affines. The converter carries the
transformation into the coordinate systems of the format: each endpoint
is bridged as composition bridges two neighbours, by reordering,
flipping and rescaling axes, which is exact, and the chain is composed
into one matrix. A voxel system is not bridged to a world system,
because that needs the geometry of a volume. A format that places its
matrix with the geometries of two volumes (FLIRT, `register.dat`, and
LTA between voxels) takes them as the conversion options `reference=`,
the volume of the input, and `moving=`, the volume of the output, and
refuses the conversion without them. `save()` writes through the same
conversion, so an affine between LPS systems is saved in an `.xfm` file
as the equivalent affine between RAS systems. Status: implemented.

A field is a B-spline of some degree, whether a format stores the values
at its nodes (a B-spline of degree 1) or the coefficients of a cubic
B-spline (FNIRT coefficient fields, NiftyReg grids of control points
and velocity grids, elastix B-spline transforms, X5 B-splines). Every
field format is therefore converted to by the same helpers of
`io/transformations/base/_conversions.py`, which take the degree of the
format. A field of that degree keeps its coefficients. The affine that
a field does not undo, in a chain `A o (id + d) o B`, is stored in the
header when the format has a place for it, as the tools write their own
files: the affine that NiftyReg applies before its grid, the initial
affine of an FSL coefficient field, another transform of an elastix,
ITK, X5 or MINC chain. The field then holds the displacement alone,
placed by `B` or by the inverse of `A`, and nothing is lost
(`split_affine`). A format without such a place, such as a NIfTI
displacement field, carries the affine into the coefficients of the
field, which is also exact, since a B-spline is linear in its
coefficients and reproduces affine maps. A
field of another degree is evaluated at its nodes, and its values are
converted to coefficients by the existing conversion
`field.to(store="coefficients")`, which is a loss. A format that imposes
its own grid, such as the reference of an FSL warp or the knots of
FNIRT, has the transformation evaluated at the nodes of that grid and
interpolated with its degree, which is exact only when the
transformation is already a B-spline of that degree on that grid. No
spline is fitted by least squares. How a field is extended beyond its
grid, and the boundary condition of the spline, are not counted as
losses. FNIRT and FLIRT need the geometry of their two images, which a
conversion takes as the options `reference=` and `moving=`. Status:
implemented.

## 4. Refusals that are deliberate

The completeness test lists them. A field cannot be built from a
transformation without a grid. A matrix is not read from a field,
because fitting one approximates. The logarithm of a displacement is
not computed, and `to(log=True)` is the way to read a field that holds
a velocity. A matrix with no real logarithm, such as a permutation that
swaps two axes, raises `DomainError` when a tangent class is requested.

A file format lists its format base before its data model, so that
`TransformationFormat.from_instance()` converts a foreign transformation
before `ConcreteTransformation.from_instance()` copies any field.
Otherwise the data model would copy the map of the source into the
arguments of the conversion, and that copy would replace the map that
an exact converter builds. The format alone can decide, because two
formats of the same data model may mean different maps: an LTA affine
between voxel systems and an MNI affine between RAS systems are both
`Affine`. Bases that only read or write, such as `BinaryFileReader`,
may come first. `_FileBasedModelMixin` raises `TypeError` when a class
that breaks the rule is defined, and a test checks that no format
passes copied fields to its converter.

## 5. Open questions

1. A pyramid could be built from an array or a single-scale image with
   its levels computed, through an option such as `nlevels`, so that
   `image.to(MultiScaleImage, nlevels=5)` builds five levels. This is
   not implemented. It needs a method that computes the levels, which
   `Image.to()` would then call.
