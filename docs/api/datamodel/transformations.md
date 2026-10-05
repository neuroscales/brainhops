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
map, as values":

```python
DisplacementField(u)  # u is displacement values
DisplacementField(field=u)  # the same
DisplacementField(data=c, degree=3, coeff=True)  # c is coefficients
DisplacementField(field=u, coeff=True)  # TypeError: contradictory
Affine(m), Affine(data=m), Affine(matrix=m)  # three equal affines
```

A convenience keyword cannot be combined with `data=`, nor with a flag
that says `data` holds another encoding. Within a type, `.to(...)`
re-encodes rather than reinterprets: `t.to(coeff=True)` fits
coefficients to the values, `t.to(degree=3)` on a field of coefficients
refits them, and `t.to(field=u)` stores `u` in the encoding of `t`.
Passing `data=` to `.to(...)` stores the array as given, under the flags
of the result.

Equality compares the class, `data` and the flags, so the same map
stored as values and as coefficients does not compare equal.

# ::: brainhops.datamodel.transformations
