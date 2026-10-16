# Design: conversions between the classes of the data model

Status: decided and implemented, except for the open questions at the
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
| images | by the file record, or single scale against pyramid | copy by name between formats; refused between single scale and pyramid |
| transformations | by representation and by promises on values | converters (C4) |

A requested tangent class implies `log=True`, so `x.to(AffineExponential)`
is `x.to(Affine, log=True)`. `convert()` adds the flag at every step of
a multi-step conversion, unless the source already holds a tangent.

## 4. Refusals that are deliberate

The completeness test lists them. A field cannot be built from a
transformation without a grid. A matrix is not read from a field,
because fitting one approximates. The logarithm of a displacement is
not computed, and `to(log=True)` is the way to read a field that holds
a velocity. A matrix with no real logarithm, such as a permutation that
swaps two axes, raises `DomainError` when a tangent class is requested.

## 5. Open questions

1. Wrapping a transformation in a sequence of one, as `t.to(Sequence)`,
   is refused. It is exact and cheap, but it changes the kind of the
   object, so it may belong to the constructor rather than to `to()`.
2. `Sequence.is_identity(compute=True)` does not compute the chain, so
   a chain that reduces to the identity does not convert exactly to
   `Identity`. The completeness test marks this pair as an expected
   failure. A checker that computes the chain must not be used by
   `compute()` itself, which relies on the checkers.
3. A single-scale image and a pyramid are unrelated, so converting one
   into the other is refused. A converter could build a pyramid of one
   level, and take the only level of a pyramid.
4. `TrakEM2PatchAffine` has its data model first in its method
   resolution order, like NIfTI, but it does not convert in its own
   override. A transformation that it converts reaches
   `TransformationFormat` with the map that its data model has already
   copied into the arguments. This works today but depends on the
   converters accepting those arguments.
