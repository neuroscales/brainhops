Every concrete transformation stores its parameter as one array, `data`,
together with the encoding flags that say what that array holds. What a
transformation *means* is read through named, read-only views, which are
always the map, as values, whatever the flags:

| Class                                   | `data` holds                             | Flags                             | Views                                   |
|-----------------------------------------|------------------------------------------|-----------------------------------|-----------------------------------------|
| `DisplacementField`                     | the values, or their spline coefficients | `store`, `degree`, `bound`, `log` | `field`, `values`, `coefficients`       |
| `CoordinatesField`                      | the values, or their spline coefficients | `store`, `degree`, `bound`        | `field`, `values`, `coefficients`       |
| `Affine`                                | the `(No, Ni + 1)` matrix                | `log`                             | `matrix`, `homogeneous_matrix`          |
| `Linear`, `Rotation`                    | the `(No, Ni)` matrix                    | `log`                             | `matrix`                                |
| `Scaling`                               | the scaling factors                      | `log`                             | `scale`                                 |
| `Translation`                           | the translation vector                   |                                   | `translation`                           |
| `Permutation`                           | the permutation vector                   |                                   | `permutation`                           |
| `CartesianField`                        | the grid, derived from `shape`           | `store`, `degree`, `bound`        | `field`, `values`, `coefficients`       |
| `Identity`                              | nothing: always `None`                   |                                   |                                         |

A field whose `store` flag is `"coefficients"` stores the coefficients
of the spline of degree `degree` (with boundary condition `bound`) that
interpolates its values; `"values"`, which is what an unset flag means,
stores the values themselves. Its `field` view decodes them, once, and
caches the result on the instance, so code that reads a field never has
to know how it is stored. Both readings are views: `t.values` is the map
and `t.coefficients` its spline coefficients, whichever of the two
`data` holds, and `t.field` is `t.values`.

A `CartesianField` stores its `shape` rather than an array, and derives
its `data` from it, encoded under its flags. A lazy inverse, such as an
`InverseDisplacementField`, derives its `data` from the transformation
it inverts, in that transformation's encoding.

!!! note "Inverting a field is approximate"
    The inverse of a displacement or coordinates field is computed from
    the field's values at its grid nodes only: they define a
    piecewise-affine map, which is inverted exactly (Ashburner's mesh
    inversion), and the result is interpolated with the field's
    `degree`. A field of degree 3 is therefore inverted as its
    piecewise-linear interpolant, about as accurately as a field of
    degree 1. On smooth fields of a few voxels' amplitude,
    `fwd(inv(x)) - x` is typically a few hundredths of a voxel in the
    interior and a few tenths near the border.

    A `StationaryVelocityField` is the exception: its inverse is
    `exp(-v)`, exact in the tangent, and integrated as accurately as the
    field itself; no mesh is inverted.

`data` is the one positional parameter of a constructor; every other
parameter, the flags and the endpoints included, is keyword-only. The
view's name is also a keyword, a convenience meaning "the map, as
values". One rule holds everywhere: a convenience keyword is the map, as
values, and the flags describe how it is stored -- except
`coefficients=`, which is the map as coefficients, and sets `store`
when nothing else did.

```python
DisplacementField(u)  # u is displacement values
DisplacementField(field=u)  # the same
DisplacementField(values=u)  # the same
DisplacementField(data=c, degree=3, store="coefficients")  # c is coefficients
DisplacementField(coefficients=c, degree=3)  # the same
DisplacementField(field=u, degree=3, store="coefficients")  # stores u's
Affine(m), Affine(data=m), Affine(matrix=m)  # the same affine, thrice
```

So `DisplacementField(field=u, degree=3, store="coefficients")` holds the
same `data` as
`DisplacementField(field=u, degree=3).to(store="coefficients")`: the
constructor encodes `u` the way `.to(...)` does. A convenience keyword
cannot be combined with `data=`, which already is the stored array.

`data` and the flags can be assigned in place (`t.data = d`,
`t.store = "coefficients"`, `t.steps = 6`). A cached view (`field`, or
the `matrix` and `scale` of a tangent) is cleared when they are, so the
next read reflects them. `log` selects the class, which an assignment
cannot change: a class that holds the map refuses `log = True`, and says
to use `t.to(log=True)` instead. Any other assignment stores what it is
given: `t.store = "coefficients"` says that the array already in `data`
holds coefficients, and reinterprets it. To change the map of an
existing transformation, or how it is stored, use `.to(...)`. Within a
type, it re-encodes rather than reinterprets: `t.to(field=u)` stores `u`
in the encoding of `t`, `t.to(store="coefficients")` fits coefficients
to the values, and `t.to(degree=3)` on a field of coefficients refits
them. Passing `data=` to `.to(...)` stores the array as given, under the
flags of the result. `bagof.magic.replace` is not the way to do it: it
carries `data` over, so a convenience keyword passed through it (as in
`replace(t, field=u)`) meets that `data` and raises whenever `t` has
one, and a flag passed through it (as in
`replace(t, store="coefficients")`) reinterprets the stored array
instead of re-encoding it.

Transformations compare, and hash, by identity: `a == b` is `a is b`,
so two distinct transformations are never equal, whatever their `data`
and flags (see [Comparing transformations and
images](../../start/python.md#comparing-transformations-and-images)).
The same map stored as values and as coefficients is two different
objects either way. To test whether two transformations are the same
map, use `is_identity((a.inverse() @ b).compute(), compute=True)`; to
compare how they are stored, compare their `data` and flags explicitly.

## Tangents: the `log` flag

The `log` flag says which function `data` describes: the map itself, or
its tangent about the identity, whose exponential is the map. A tangent
is always about the identity, so unset or zero `data` is the identity
whatever the flag. `log=True` builds a subclass whose views read `data`
as a tangent:

| Class                                          | `data` holds, with `log=True`            | Views                                                |
|------------------------------------------------|------------------------------------------|------------------------------------------------------|
| `StationaryVelocityField(DisplacementField)`  | the velocity: values, or coefficients    | `field`: the displacement of its flow at time one    |
| `AffineExponential(Affine)`                    | the `(N, N + 1)` tangent `[L, l]`        | `matrix`: `expm([[L, l], [0, ..., 0]])[:-1]`        |
| `LinearExponential(Linear)`                    | the `(N, N)` tangent `L`                 | `matrix`: `expm(L)`                                  |
| `RotationExponential(Rotation)`                | the antisymmetric `(N, N)` tangent `L`   | `matrix`: `expm(L)`                                  |
| `ScalingExponential(Scaling)`                  | the logarithms `s` of the factors        | `scale`: `exp(s)`                                    |

So `DisplacementField(data=v, log=True)` and
`StationaryVelocityField(data=v)` are one object, and so are
`Affine(data=L, log=True)` and `AffineExponential(data=L)`. A tangent is
never a matrix: `LinearExponential(data=I)` is the scaling by `e`, not
the identity. `Translation`, `Permutation`, `CoordinatesField`,
`CartesianField` and `Identity` take no `log` flag. A convenience
keyword is still the map: `Affine(matrix=M, log=True)` stores the
principal logarithm of `M`, while `DisplacementField(field=u, log=True)`
raises, since a field has no logarithm that brainhops computes.

A field has the two flags, which combine; `data` is decoded from
coefficients first, and integrated second:

| `log`   | `store`          | `data` holds                                       |
|---------|------------------|----------------------------------------------------|
| `False` | `"values"`       | the displacement, as values                        |
| `False` | `"coefficients"` | the displacement's spline coefficients             |
| `True`  | `"values"`       | the velocity, as values                            |
| `True`  | `"coefficients"` | the velocity's spline coefficients (NiftyReg `-vel -cpp`) |

The `field` view of a `StationaryVelocityField` is always the
displacement, as values. The velocity is integrated by scaling and
squaring: it is divided by `2 ** steps`, which is its own flow to first
order, and composed with itself `steps` times. `steps` exists only on a
`StationaryVelocityField`, and reads as the number the field integrates
with: the one it was built with, or, when it was given none, the
smallest for which the first step moves no point by more than an eighth
of a voxel. A velocity of coefficients is refitted at each step. The
velocity's coefficients are `t.coefficients`, and its displacement is
`t.to(log=False)`.

The encoding changes with `.to(log=...)`:

- `.to(log=True)` takes the principal logarithm of a matrix (of the
  scaling factors), and raises `DomainError` when it has an eigenvalue
  on the closed negative real axis (a factor that is not positive). A
  field has no logarithm that brainhops computes: it raises
  `NotImplementedError`, unless the field is unset.
- `.to(log=False)` builds the base class (`Affine`, `DisplacementField`,
  ...) from the exponential: the matrix, or the integrated displacement,
  which keeps its `store`, `degree` and `bound`.
- The flags combine: `.to(store="values")` on a velocity decodes its
  coefficients and keeps `log`, and `.to(log=False)` keeps `store`.
- A `SubspaceTransformation` converts its inner transformation (the axes
  it does not act on are the identity, whose tangent is zero).
- A `Sequence` converts the transformation it reduces to: a chain that
  simplifies to one transformation; the middle of a change of
  coordinates `[P, *X, P^-1]`, whose ends are kept (the flow commutes with
  them, so a velocity read between a world-to-voxel affine and its
  inverse becomes its displacement exactly); or the affine a chain of
  affines composes to. Any other chain is refused before anything is
  computed.

A tangent makes some operations exact: `inverse()` is a lazy `Inverse`
whose `data` is the negated tangent, which still cancels against its
forward in a `Sequence`; `sqrt()` halves the tangent (a velocity
integrates with one squaring fewer), and `square()` doubles it.

# ::: brainhops.datamodel.transformations
