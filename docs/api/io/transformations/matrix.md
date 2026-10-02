A plain matrix file stores an affine and nothing else. Its conventions
are given when reading it:

| Keyword      | Default     | Meaning                                                                 |
|--------------|-------------|-------------------------------------------------------------------------|
| `vector`     | `"column"`  | `"row"` if the file maps row vectors (`y = x @ A`); it is transposed.   |
| `ndim`       | `None`      | `2` to read a `(3, 3)` matrix as a 2-D affine (else 3-D linear).        |
| `direction`  | `"forward"` | `"inverse"` if the file maps `output` to `input`; it is inverted.       |
| `input`      | `None`      | `"voxel"`, `"ras"`, `"lps"`, a `CoordinateSystem`, or unnamed.         |
| `output`     | `None`      | Same as `input`.                                                        |
| `index_base` | `0`         | `1` for 1-based voxel indices (MATLAB, SPM), or an `(input, output)` pair. |
| `source`     | `None`      | Image whose voxel-to-world affine maps a voxel `input` to world.       |
| `target`     | `None`      | Image whose voxel-to-world affine maps a voxel `output` to world.      |
| `variable`   | `None`      | `.npz` key or `.mat` variable; by default the only 2-D numeric array.  |

They are applied in the order: transposition, inversion, index shift,
image placement. The result is a column-vector, 0-based affine (#201:
integer index = voxel centre).

Because any small numeric table reads as a matrix, this reader scores
low and never takes a file away from FLIRT (`.mat` text) or ITK (`.mat`
v4, `.tfm`); use `hint="matrix"` to force it.

# ::: brainhops.io.transformations.matrix
