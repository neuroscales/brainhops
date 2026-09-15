# All brainhops commands

##

### reslice

=== ":octicons-terminal-24:"
    ```shell
    brainhops reslice <input> --reference IMAGE [--transform SOURCE ...] --output IMAGE [--order ORDER] [--bound BOUND]
    ```

Resample an image onto the grid of a reference image, after applying a
chain of transformations to it.

Each `--transform` is a forward (push) map that moves the input image
through world space, in the same direction the image itself travels.
Transforms are applied in the order they are given on the command line,
so the first `--transform` is applied first. The command inverts the
composed map internally and pulls the reference grid from the input, so
the arguments are written in the forward direction while the resampling
runs in the opposite one.

#### Positional arguments

| Name    | Type   | Description                     |
| ------- | ------ | -------------------------------- |
| `input` | `path` | Path to the image to resample. |

#### Options

| Flag                 | Type   | Description                                                                            | Default    |
| --------------------- | ------ | ---------------------------------------------------------------------------------------- | ---------- |
| `-h`, `--help`         |        | Show the help message and exit.                                                          |            |
| `-r`, `--reference`    | `path` | Reference image whose geometry defines the output grid and its placement in world space. | *required* |
| `-t`, `--transform`    | `path` | A forward (push) transformation to apply to the input image. Repeat the option to apply several, in the order given. | |
| `-o`, `--output`       | `path` | Path to write the resampled image to.                                                    | *required* |
| `--order`              | `int`  | Spline interpolation order (`0` = nearest, `1` = linear).                                | `1`        |
| `--bound`              | `str`  | Boundary condition used outside the field of view.                                       | `reflect`  |

#### Transform operators

A `--transform` value is a path, optionally followed by pipe-separated
operators applied in written order, for example `warp.nii.gz|inv`. The
`|` usually needs shell quoting, for example `--transform "warp.nii.gz|inv"`.

Many warp files are stored as pull maps that run from the reference
image to the moving image, rather than as the push maps `reslice`
expects. ANTs, SPM and FSL warps are usually of this kind, and the `inv`
operator inverts such a file before it is composed.

| Operator | Description                          | Status                     |
| -------- | -------------------------------------- | --------------------------- |
| `inv`    | Inverts the transform.                 | Implemented                 |
| `sqrt`   | Matrix square root of the transform.   | Not implemented (issue #47) |
| `square` | Matrix square of the transform.        | Not implemented (issue #47) |
| `exp`    | Matrix exponential of the transform.   | Not implemented (issue #47) |
| `log`    | Matrix logarithm of the transform.     | Not implemented (issue #47) |

### compose

=== ":octicons-terminal-24:"
    ```shell
    brainhops compose [FILE ...] [--output FILE]
    ```

Combine several transformations into one and write the result. The
transformations are combined following the composition operator: given
transformations `A` and `B`, the composite maps a coordinate through `B`
first and then through `A`, matching `A @ B` in the data model. The
files are listed in this composition order, so the last file listed is
applied first.

#### Positional arguments

| Name         | Type          | Description                                                    |
| ------------ | ------------- | ------------------------------------------------------------------ |
| `transforms` | `list[path]`  | Transformation files to combine, in composition order.        |

#### Options

| Flag             | Type   | Description                                    | Default |
| ----------------- | ------ | ------------------------------------------------ | ------- |
| `-h`, `--help`     |        | Show the help message and exit.                  |         |
| `-o`, `--output`   | `path` | Path to write the combined transformation to.    |         |

!!! warning "Not implemented yet"
    Running this command raises an error rather than combining anything.
    It depends on a writable transformation format (issue #41), and on
    the still-undecided command-line syntax for per-operand operations
    such as inversion and the matrix square root.

### convert

=== ":octicons-terminal-24:"
    ```shell
    brainhops convert <input> --output FILE
    ```

Read an object from one file format and write it back in another, for
example an ITK transform to a NIfTI displacement field. The kind of
object is detected from the input, and the target format is chosen from
the output file's extension.

#### Positional arguments

| Name    | Type   | Description                  |
| ------- | ------ | ------------------------------- |
| `input` | `path` | Path to the object to convert. |

#### Options

| Flag             | Type   | Description                                                          | Default    |
| ----------------- | ------ | ----------------------------------------------------------------------- | ---------- |
| `-h`, `--help`     |        | Show the help message and exit.                                         |            |
| `-o`, `--output`   | `path` | Path to write the converted object to. Its extension selects the target format. | *required* |

!!! warning "Not implemented yet"
    Running this command raises an error rather than writing anything.
    It depends on writable formats (issue #41), and on the policy for a
    feature that the source format records but the target format cannot
    represent.
