# reslice

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

## Positional arguments

| Name    | Type   | Description                     |
| ------- | ------ | -------------------------------- |
| `input` | `path` | Path to the image to resample. |

## Options

| Flag                 | Type   | Description                                                                            | Default    |
| --------------------- | ------ | ---------------------------------------------------------------------------------------- | ---------- |
| `-h`, `--help`         |        | Show the help message and exit.                                                          |            |
| `-r`, `--reference`    | `path` | Reference image whose geometry defines the output grid and its placement in world space. | *required* |
| `-t`, `--transform`    | `path` | A forward (push) transformation to apply to the input image. Repeat the option to apply several, in the order given. | |
| `-o`, `--output`       | `path` | Path to write the resampled image to.                                                    | *required* |
| `--order`              | `int`  | Spline interpolation order (`0` = nearest, `1` = linear).                                | `1`        |
| `--bound`              | `str`  | Boundary condition used outside the field of view.                                       | `reflect`  |

## Transform operators

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
