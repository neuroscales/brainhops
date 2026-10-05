Every concrete transformation stores its parameter as one array, `data`,
together with the encoding flags that say what that array holds. What a
transformation *means* is read through named, read-only views, which are
always the map, as values, whatever the flags:

| Class                                   | `data` holds                             | Flags                      | Views                          |
|-----------------------------------------|------------------------------------------|----------------------------|--------------------------------|
| `DisplacementField`, `CoordinatesField` | the values, or their spline coefficients | `coeff`, `degree`, `bound` | `field`                        |
| `Affine`                                | the `(No, Ni + 1)` matrix                |                            | `matrix`, `homogeneous_matrix` |
| `Linear`, `Rotation`                    | the `(No, Ni)` matrix                    |                            | `matrix`                       |
| `Scaling`                               | the scaling factors                      |                            | `scale`                        |
| `Translation`                           | the translation vector                   |                            | `translation`                  |
| `Permutation`                           | the permutation vector                   |                            | `permutation`                  |
| `CartesianField`                        | the grid, derived from `shape`           | `coeff`, `degree`, `bound` | `field`                        |
| `Identity`                              | nothing: always `None`                   |                            |                                |

A field whose `coeff` flag is set stores the coefficients of the spline
of degree `degree` (with boundary condition `bound`) that interpolates
its values. Its `field` view decodes them, once, and caches the result
on the instance, so code that reads a field never has to know how it is
stored. Any other encoding is reached by conversion, never by a view:
the coefficients of a field are `t.to(coeff=True).data`, and its values
are `t.field` (or `t.to(coeff=False).data`).

A `CartesianField` stores its `shape` rather than an array, and derives
its `data` from it, encoded under its flags. A lazy inverse, such as an
`InverseDisplacementField`, derives its `data` from the transformation
it inverts, in that transformation's encoding.

Constructors take `data` (positionally, as the first argument) and the
flags. The view's name is also a keyword, a convenience meaning "the
map, as values". One rule holds everywhere: a convenience keyword is the
map, as values, and the flags describe how it is stored.

```python
DisplacementField(u)  # u is displacement values
DisplacementField(field=u)  # the same
DisplacementField(data=c, degree=3, coeff=True)  # c is coefficients
DisplacementField(field=u, degree=3, coeff=True)  # stores u's coefficients
Affine(m), Affine(data=m), Affine(matrix=m)  # the same affine, thrice
```

So `DisplacementField(field=u, degree=3, coeff=True)` holds the same
`data` as `DisplacementField(field=u, degree=3).to(coeff=True)`: the
constructor encodes `u` the way `.to(...)` does. A convenience keyword
cannot be combined with `data=`, which already is the stored array.

To change the map of an existing transformation, use `.to(...)`. Within
a type, it re-encodes rather than reinterprets: `t.to(field=u)` stores
`u` in the encoding of `t`, `t.to(coeff=True)` fits coefficients to the
values, and `t.to(degree=3)` on a field of coefficients refits them.
Passing `data=` to `.to(...)` stores the array as given, under the flags
of the result. `bagof.magic.replace` is not the way to do it: it carries
`data` over, so a convenience keyword passed through it (as in
`replace(t, field=u)`) meets that `data` and raises whenever `t` has
one, and a flag passed
through it (as in `replace(t, coeff=True)`) reinterprets the stored
array instead of re-encoding it.

Transformations compare, and hash, by identity: `a == b` is `a is b`,
so two distinct transformations are never equal, whatever their `data`
and flags (see [Comparing transformations and
images](../../start/python.md#comparing-transformations-and-images)).
The same map stored as values and as coefficients is two different
objects either way. To test whether two transformations are the same
map, use `is_identity((a.inverse() @ b).compute(), compute=True)`; to
compare how they are stored, compare their `data` and flags explicitly.

# ::: brainhops.datamodel.transformations
