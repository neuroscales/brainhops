A plain matrix file stores an affine and nothing else. There is one
reader per container, each built on that container's generic array
parser from [`brainhops.io.base.arrays`][brainhops.io.base.arrays], and
all deriving from the abstract `MatrixAffine`, which is never registered:

| Class                   | Container                   | Extensions             | Hints                                 |
|-------------------------|-----------------------------|------------------------|---------------------------------------|
| `TxtMatrixAffine`       | whitespace-separated text   | `.txt`, `.dat`, `.1D`  | `"matrix.txt"`                        |
| `CsvMatrixAffine`       | comma-separated text        | `.csv`                 | `"matrix.csv"`                        |
| `TsvMatrixAffine`       | tab-separated text          | `.tsv`                 | `"matrix.tsv"`                        |
| `NpyMatrixAffine`       | NumPy `.npy`                | `.npy`                 | `"matrix.npy"`                        |
| `NpzMatrixAffine`       | NumPy `.npz`                | `.npz`                 | `"matrix.npz"`                        |
| `MatLegacyMatrixAffine` | MATLAB v4, v5-v7 (`scipy`)  | `.mat`                 | `"matrix.mat"`                        |
| `Mat73MatrixAffine`     | MATLAB v7.3 (`h5py`)        | `.mat`                 | `"matrix.mat"`, `"mat.73"`, `"matrix.mat.73"` |

`MatMatrixAffine` (hint `"matrix.mat"`) is the parent of the two MATLAB
readers. It is not registered (its variants are), but it can be used
directly: it reads any MATLAB version and returns an object of the
variant that matches the file. AFNI `.1D` and generic `.dat` files are
whitespace-separated columns with `#` comments, so they are read by
`TxtMatrixAffine`.

`hint="matrix"` selects among all of them by content; a container hint
selects one. The conventions are given when reading:

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
| `variable`   | `None`      | `.npz` key or `.mat` variable (alias `key`); by default the only 2-D numeric array. |

They are applied in the order: transposition, inversion, index shift,
image placement. The result is a column-vector, 0-based affine (#201:
integer index = voxel centre).

Because any small numeric table reads as a matrix, these readers score
low and never take a file away from FLIRT (`.mat` text) or ITK (`.mat`
v4, `.tfm`, `.h5`), except that a text matrix named `.txt`, `.csv`,
`.tsv`, `.dat` or `.1D` (with that file's separator) is preferred to
FLIRT. Without a telling name, text content goes to one reader only: a
comma makes it CSV, tab-separated values TSV, anything else TXT. Use `hint="matrix"` (or a
container hint) to force them.

# ::: brainhops.io.transformations.matrix
