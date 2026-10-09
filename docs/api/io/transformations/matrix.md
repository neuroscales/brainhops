A plain matrix file stores an affine transformation and nothing else: no
coordinate systems, no index base and no direction. brainhops has one
reader for each kind of container. Each reader is built on the generic
array parser of its container, from `brainhops.io.common._arrays`, and all
of them derive from the abstract [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine], which is never
registered itself.

| Class                   | Container                  | Extensions            | Hints                                          |
|-------------------------|----------------------------|-----------------------|------------------------------------------------|
| `TxtMatrixAffine`       | whitespace-separated text  | `.txt`, `.dat`, `.1D` | `"matrix.txt"`                                 |
| `CsvMatrixAffine`       | comma-separated text       | `.csv`                | `"matrix.csv"`                                 |
| `TsvMatrixAffine`       | tab-separated text         | `.tsv`                | `"matrix.tsv"`                                 |
| `NpyMatrixAffine`       | NumPy `.npy`               | `.npy`                | `"matrix.npy"`                                 |
| `NpzMatrixAffine`       | NumPy `.npz`               | `.npz`                | `"matrix.npz"`                                 |
| `MatLegacyMatrixAffine` | MATLAB v4, v5-v7 (`scipy`) | `.mat`                | `"matrix.mat"`                                 |
| `Mat73MatrixAffine`     | MATLAB v7.3 (`h5py`)       | `.mat`                | `"matrix.mat"`, `"mat.73"`, `"matrix.mat.73"` |

[`MatMatrixAffine`][brainhops.io.transformations.matrix.MatMatrixAffine] (hint `"matrix.mat"`) is the parent of the two
MATLAB readers. It is not registered, since its two variants are, but it
can be used directly: it reads a file of any MATLAB version and returns an
object of the matching variant. AFNI `.1D` files and generic `.dat` files
hold whitespace-separated columns with `#` comments, so they are read by
`TxtMatrixAffine`.

The hint `"matrix"` selects among all of these readers by content, while a
container hint such as `"matrix.csv"` selects one of them. Since the file
records nothing but the matrix, its conventions are given as keyword
arguments when it is read:

| Keyword      | Default     | Meaning                                                                             |
|--------------|-------------|-------------------------------------------------------------------------------------|
| `vector`     | `"column"`  | `"row"` if the file maps row vectors (`y = x @ A`); it is transposed.               |
| `ndim`       | `None`      | `2` to read a `(3, 3)` matrix as a 2-D affine (else 3-D linear).                    |
| `direction`  | `"forward"` | `"inverse"` if the file maps `output` to `input`; it is inverted.                   |
| `input`      | `None`      | `"voxel"`, `"ras"`, `"lps"`, a `CoordinateSystem`, or unnamed.                      |
| `output`     | `None`      | Same as `input`.                                                                    |
| `index_base` | `0`         | `1` for 1-based voxel indices (MATLAB, SPM), or an `(input, output)` pair.          |
| `source`     | `None`      | Image whose voxel-to-world affine maps a voxel `input` to world.                    |
| `target`     | `None`      | Image whose voxel-to-world affine maps a voxel `output` to world.                   |
| `variable`   | `None`      | `.npz` key or `.mat` variable (alias `key`); by default the only 2-D numeric array. |

The images given as `source` and `target` may be `nibabel` images or
headers, or brainhops images. The conventions are applied in a fixed
order: transposition, then inversion, then the index shift, and finally
the placement in image space. The result is always an affine that maps
column vectors and uses 0-based indices, in which an integer index is the
centre of a voxel (see #201). The raw matrix and the conventions it was
read with are kept on the object, as `raw_matrix`, `variable`, `vector`,
`direction` and `index_base`.

```python
from brainhops import io

affine = io.transformations.load("affine.txt", input="ras", output="ras")
rows = io.transformations.load("rows.csv", hint="matrix", vector="row")
```

Any small numeric table can be read as a matrix, so these readers give
themselves a low score and never take a file away from a format that
recognises it positively, such as a FLIRT `.mat` (text) or an ITK `.mat`
(MATLAB v4), `.tfm` or `.h5` file. The exception is a text matrix whose
file name has a text-array extension (`.txt`, `.csv`, `.tsv`, `.dat` or
`.1D`) that matches its separator: such a file is preferred to the FLIRT
reader. Without an informative name, text content is offered to one reader
only, chosen by its separator: commas select the CSV reader, tabs the TSV
reader, and anything else the whitespace reader. Files larger than 1 MiB
are not inspected at all. To force these readers, pass `hint="matrix"` or a
container hint.

# ::: brainhops.io.transformations.matrix
