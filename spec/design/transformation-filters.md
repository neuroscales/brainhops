# Design: filtering a transformation and converting it to a kind

Status: implemented (#589). This memo records two additions to the
public methods of `Transformation`. The first is a pair of methods,
`without()` and `with_only()`, that replace some parts of a
transformation by the identity. The second is a family of methods,
`to_affine()`, `to_linear()` and the others, that convert a
transformation to the class that matches a kind. Decisions are tagged
`F1` to `F7`.

## 1. The problem

Several modules need the affine part of a transformation. They import
`_as_affine` and `_as_affine_ignoring_fields` from
`datamodel/_transformations/compute/multiscale.py`, which are private
functions, and the importers include the readers and writers of Zarr
files and two test files. `_as_affine` computes the transformation,
converts it to `Affine`, and returns `None` when the conversion fails.
`_as_affine_ignoring_fields` first calls `_fields_as_identity`, which
replaces every field with the identity, so that a chain that contains
a warp still has an affine part that measures its scale.

Each of these functions answers a general question with a private
name and a fixed choice. A user who wants the linear part of a chain,
or who wants to drop a different class of transformation, has no
public way to ask. The `is_*` methods (`is_affine()`, `is_linear()`
and the others) have no matching conversion methods either.

## 2. Matching parts of a transformation

F1. A matcher says which parts of a transformation a filter applies
to. A matcher is one of the following values:

- a kind or the name of a kind, which is anything that
  `Transformation.is_kind` accepts, and which matches a part that
  has the properties of that kind;
- a concrete class of transformation, which matches its instances;
- a predicate, which is a function that takes a transformation and
  returns a boolean;
- a collection of the values above, which matches a part when any
  item of the collection matches it.

A collection is a `list`, a `tuple`, a `set` or a `frozenset`, and no
other iterable is unpacked. A transformation such as a `Sequence` is
both iterable and callable, and a generator can be read only once, so
either would be read in a way that the caller did not mean. Such a
value, and any other value that is not a matcher, raises a
`TypeError`. A string that names no kind raises the `ValueError` of
the lookup of kinds.

Classes are needed because kinds cannot express every set of
transformations. A kind is a set of properties, and the kinds form a
hierarchy in which a more specific kind has all the properties of a
more general one. A displacement field and an affine are both
diffeomorphisms, so no kind contains the fields without also
containing the affines. The class `TransformationField` names the
fields directly.

## 3. Removing parts of a transformation

F2. `without(match)` returns a transformation in which every part that
the matcher selects is replaced by the identity. `with_only(match)`
replaces every part that the matcher does not select. The second
method is not there only for symmetry. Because a kind can only select
the transformations that have a property, "remove everything that is
not affine" can be written with a kind only as
`with_only("affine")`. That call is what `_fields_as_identity` does
today.

F3. A removed part becomes an `Identity` with the same input and
output coordinate systems as the part it replaces. If the part were
dropped from a chain instead, the output system of the element before
it would no longer match the input system of the element after it,
and bridging might insert a transformation that it guesses to close
the gap. The simplifier removes the identities when the result is
computed. A part that changes the number of axes cannot be replaced by
an identity, so removing such a part raises an error.

F4. Both methods visit a transformation from the outside in. They test
a node first and keep or replace it as a whole when the matcher
decides it. They look inside a `Sequence` or an `Inverse` only when the
node itself is not decided, so a chain that is affine as a whole is
kept whole by `with_only("affine")`. The inverse of a removed part is
also removed, because the inverse of an identity is an identity. When
nothing is replaced, the methods return the transformation itself.
The methods do not look inside the other wrappers, which are the lazy
square root `Sqrt`, `Bijection` and `SubspaceTransformation`, so such a
wrapper is kept or replaced as a whole. Looking inside them is not
implemented yet, and the limit is documented in the docstrings of both
methods.

## 4. Converting to a kind

F5. `to_kind(kind, ...)` converts a transformation to the concrete
class that represents a kind. It computes the transformation first,
because a chain must be composed before it can be converted, and then
calls `Transformation.to` with the same keyword arguments as
`to()`. The kinds that the `is_*` methods test map to concrete classes
through a single table, which `is_*` and `to_*` share.

The result of `to_kind` always has its parameters. A result that has
no matrix, or, for a translation, a scaling or a permutation, no
`data`, raises a `ConversionError`, because its dimension is unknown,
as for `Identity()` or for the inverse of an affine whose matrix is not
set. The error goes through the error policy of the call, so that
`to_affine(on_error=None)` returns `None` and a caller tests only for
`None`, rather than also for a missing matrix. Plain `to()` makes no
such check, because a transformation without parameters is a valid
object of its class.

| Method             | Kind tested by `is_*`     | Concrete class |
| ------------------ | ------------------------- | -------------- |
| `to_translation()` | `kinds.Translation`       | `Translation`  |
| `to_scaling()`     | `kinds.Diagonal`          | `Scaling`      |
| `to_permutation()` | `kinds.Permutation`       | `Permutation`  |
| `to_rotation()`    | `kinds.SpecialOrthogonal` | `Rotation`     |
| `to_linear()`      | `kinds.Linear`            | `Linear`       |
| `to_affine()`      | `kinds.Affine`            | `Affine`       |

Each `to_*` method is a one-line call to `to_kind()`. There is no
`to_identity()`, because a transformation that is the identity has
nothing to convert, and a transformation that is not the identity
cannot become one without losing all of its content.

F6. `to_kind()` and every `to_*` method take two matchers,
`include=` and `exclude=`. The method applies `with_only(include)` and
then `without(exclude)` before it computes and converts the
transformation, and either or both may be given. The two names say
which way the matcher works, where a single `ignore=` argument had to
take the value `True` to mean every part that is not of the target
kind. The affine part of a chain is therefore
`to_affine(include="affine")`. That call replaces
`_as_affine_ignoring_fields`, and `to_affine(on_error=None)` replaces
`_as_affine` (see [error-policy.md](error-policy.md) for `on_error=`).

## 5. What is removed

F7. `_as_affine`, `_as_affine_ignoring_fields` and
`_fields_as_identity` are removed once their callers use the public
methods. The decision of `multiscale.py` about which level of a
multiscale object to read keeps using the affine part, through
`to_affine(include="affine", on_error=None)`.
