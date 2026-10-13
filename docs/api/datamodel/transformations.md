Every concrete transformation stores its parameters as a single array,
`data`, together with encoding flags that say what the array holds. The
meaning of a transformation is read through named, read-only views, and
those views always return the map as values, whatever the flags say.

| Class                | `data` holds                             | Flags                             | Views                             |
|----------------------|------------------------------------------|-----------------------------------|-----------------------------------|
| `DisplacementField`  | the values, or their spline coefficients | `store`, `degree`, `bound`, `log` | `field`, `values`, `coefficients` |
| `CoordinatesField`   | the values, or their spline coefficients | `store`, `degree`, `bound`        | `field`, `values`, `coefficients` |
| `Affine`             | the `(No, Ni + 1)` matrix                | `log`                             | `matrix`, `homogeneous_matrix`    |
| `Linear`, `Rotation` | the `(No, Ni)` matrix                    | `log`                             | `matrix`                          |
| `Scaling`            | the scaling factors                      | `log`                             | `scale`                           |
| `Translation`        | the translation vector                   |                                   | `translation`                     |
| `Permutation`        | the permutation vector                   |                                   | `permutation`                     |
| `CartesianField`     | the grid, derived from `shape`           | `store`, `degree`, `bound`        | `field`, `values`, `coefficients` |
| `Identity`           | nothing: always `None`                   |                                   |                                   |

When `store="coefficients"`, the array holds the coefficients of the spline
of degree `degree`, with boundary condition `bound`, that interpolates the
values. When `store="values"`, which is also what an unset `store` means,
the array holds the values themselves. Both encodings are exposed as
views: `t.values` is the map and `t.coefficients` is its spline
coefficients, whichever of the two `data` holds, and `t.field` is the same
as `t.values`. The `field` view is decoded once and cached on the instance,
so code that reads a field does not need to know how it is stored.

A `CartesianField` stores its `shape` and derives `data` from it, encoded
under its own flags. A lazy inverse, such as `InverseDisplacementField`,
derives its `data` from the transformation it inverts, in that
transformation's encoding.

!!! note "Inverting a field is approximate"
    The inverse of a displacement or coordinates field is computed from the
    values at the grid nodes only. These values define a piecewise-affine
    map, which is inverted exactly with Ashburner's mesh inversion, and the
    inverse is then interpolated with the field's `degree`. A field of
    degree 3 is therefore inverted as its piecewise-linear interpolant, and
    the result is about as accurate as for a field of degree 1. On smooth
    fields whose amplitude is a few voxels, `fwd(inv(x)) - x` is typically a
    few hundredths of a voxel in the interior of the field of view and a few
    tenths of a voxel near its border. A `StationaryVelocityField` is the
    exception: its inverse is `exp(-v)`, which is exact in the tangent
    space and is integrated as accurately as the field itself, without
    inverting any mesh.

`data` is the only positional parameter of a constructor. Every other
parameter, flags and endpoints alike, is keyword-only. The name of a view
is also accepted as a keyword, as a convenience that means "the map, as
values". A convenience keyword therefore always gives the map as values,
while the flags describe how it is stored. The one exception is
`coefficients=`, which gives the map as coefficients and sets `store` when
nothing else does.

```python
DisplacementField(u)  # u is displacement values
DisplacementField(field=u)  # the same
DisplacementField(values=u)  # the same
DisplacementField(data=c, degree=3, store="coefficients")  # c is coefficients
DisplacementField(coefficients=c, degree=3)  # the same
DisplacementField(field=u, degree=3, store="coefficients")  # stores u's
Affine(m), Affine(data=m), Affine(matrix=m)  # the same affine, thrice
```

The constructor encodes a convenience keyword in the same way as
`.to(...)` does, so `DisplacementField(field=u, degree=3, store="coefficients")`
has the same `data` as
`DisplacementField(field=u, degree=3).to(store="coefficients")`. A
convenience keyword cannot be combined with `data=`, since `data` already
is the stored array.

`data` and most flags can be assigned in place (`t.data = d`,
`t.store = "coefficients"`, or `t.steps = 6` on a velocity), and the cached
views, such as `field` or the `matrix` and `scale` of a tangent, are cleared
on assignment. An assignment stores what it is given: `t.store =
"coefficients"` declares that the array already in `data` holds
coefficients, so it reinterprets the array rather than re-encoding it. The
`log` flag selects the class of a transformation and cannot be assigned at
all.

To change the map or its storage, use `.to(...)`. Within one type,
`.to(...)` re-encodes: `t.to(field=u)` stores `u` in the encoding of `t`,
`t.to(store="coefficients")` fits coefficients to the values, and
`t.to(degree=3)` refits coefficients of another degree. Passing `data=` to
`.to(...)` stores the array as given, under the flags of the result.
`bagof.magic.replace` is not a substitute. It carries `data` over, so a
convenience keyword such as `replace(t, field=u)` collides with that `data`
and raises whenever `t` has one, and a flag such as
`replace(t, store="coefficients")` reinterprets the array instead of
re-encoding it.

Transformations compare and hash by identity: `a == b` is the same as
`a is b`, so two transformations are never equal, even when they hold the
same data and flags (see
[Comparing transformations and images](../../start/python.md#comparing-transformations-and-images)).
In particular, the same map stored once as values and once as coefficients
gives two distinct objects. Whether two transformations describe the same
map is tested with `is_identity((a.inverse() @ b).compute(), compute=True)`.
Whether they are stored in the same way is tested by comparing `data` and
the flags explicitly.

## Tangents: the `log` flag

The `log` flag says which function `data` describes: the map itself, or
its tangent about the identity, whose exponential is the map. Since the
tangent is always taken about the identity, an unset or zero `data` is the
identity whatever the flag. Passing `log=True` builds a subclass whose
views read `data` as a tangent.

| Class                                         | `data` holds, with `log=True`          | Views                                             |
|-----------------------------------------------|----------------------------------------|---------------------------------------------------|
| `StationaryVelocityField(DisplacementField)`  | the velocity: values, or coefficients  | `field`: the displacement of its flow at time one |
| `AffineExponential(Affine)`                   | the `(N, N + 1)` tangent `[L, l]`      | `matrix`: `expm([[L, l], [0, ..., 0]])[:-1]`      |
| `LinearExponential(Linear)`                   | the `(N, N)` tangent `L`               | `matrix`: `expm(L)`                               |
| `RotationExponential(Rotation)`               | the antisymmetric `(N, N)` tangent `L` | `matrix`: `expm(L)`                               |
| `ScalingExponential(Scaling)`                 | the logarithms `s` of the factors      | `scale`: `exp(s)`                                 |

A tangent only exists for a map from a space to itself, which is why its
matrices are square. `DisplacementField(data=v, log=True)` is a
`StationaryVelocityField(data=v)`, and `Affine(data=L, log=True)` is an
`AffineExponential(data=L)`. A tangent is never read as a matrix:
`LinearExponential(data=I)` is a scaling by `e`, not the identity.
`Translation`, `Permutation`, `CoordinatesField`, `CartesianField` and
`Identity` take no `log` flag. A convenience keyword still gives the map,
so `Affine(matrix=M, log=True)` stores the principal logarithm of `M`,
whereas `DisplacementField(field=u, log=True)` raises
`NotImplementedError`, because the logarithm of a field is not computed.

On a field, the two flags combine. The stored `data` is first decoded from
its coefficients, and then integrated.

| `log`   | `store`          | `data` holds                                              |
|---------|------------------|-----------------------------------------------------------|
| `False` | `"values"`       | the displacement, as values                               |
| `False` | `"coefficients"` | the displacement's spline coefficients                    |
| `True`  | `"values"`       | the velocity, as values                                   |
| `True`  | `"coefficients"` | the velocity's spline coefficients (NiftyReg `-vel -cpp`) |

The `field` of a `StationaryVelocityField` is always the displacement, as
values. It is integrated by scaling and squaring: the velocity is divided
by `2 ** steps`, which approximates its own flow to first order, and the
result is composed with itself `steps` times. The `steps` flag exists only
on `StationaryVelocityField`. It holds the number the field was built with
or, if none was given, the smallest number for which the first step moves
no point by more than an eighth of a voxel. A velocity stored as
coefficients is refitted at each step. The coefficients of the velocity are
`t.coefficients`, and its displacement is `t.to(log=False)`.

Converting with `.to(log=...)` follows these rules:

- `.to(log=True)` takes the principal logarithm of a matrix, or of the
  scaling factors. It raises `DomainError` when the matrix has an
  eigenvalue on the closed negative real axis, or when a factor is not
  positive. The logarithm of a field is not computed, so a field raises
  `NotImplementedError` unless it is unset.
- `.to(log=False)` builds the base class from the exponential: the matrix,
  or the integrated displacement with the same `store`, `degree` and
  `bound`.
- The flags convert independently. `.to(store="values")` on a velocity
  decodes its coefficients and keeps `log`, and `.to(log=False)` keeps
  `store`.
- A `SubspaceTransformation` converts its inner transformation. The axes it
  does not act on are the identity, whose tangent is zero.
- A `Sequence` converts the transformation it reduces to: a chain that
  simplifies to a single transformation; the middle of a change of
  coordinates `[P, *X, P^-1]`, whose ends are kept; or the affine that a
  chain of affines composes to. A flow commutes with a change of
  coordinates, so a velocity placed between a world-to-voxel affine and
  its inverse becomes its displacement exactly. Any other chain is refused
  before anything is computed.

The operators are exact on a tangent. `inverse()` returns a lazy `Inverse`
whose `data` is the negated tangent, and which still cancels against its
forward transformation in a `Sequence`. `sqrt()` halves the tangent, so a
velocity is integrated with one squaring step fewer, and `square()` doubles
it.

# ::: brainhops.datamodel.transformations
