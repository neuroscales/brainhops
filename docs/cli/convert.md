# convert

=== ":octicons-terminal-24:"

    ```shell
    brainhops convert SOURCE --output FILE [--to HINT] [--kind KIND] [--reference IMAGE] [--moving IMAGE] [--lossy]
    ```

The `convert` command reads an image or a transformation stored in one
format and writes it in another, for example to turn an ITK transform into
a NIfTI displacement field. The kind of object is detected from the input
file, and the target format is chosen from the name of the output file, as
`brainhops.io.save` chooses it.

A transformation is written in a format that holds the same map, with its
coordinate systems adapted to those of the format: an affine between LPS
spaces is written in a RAS format with its first two axes flipped, and a
field of displacements is written as a field of coordinates when the format
stores coordinates. How a field is extended outside its grid is decided by
each format, so the written field may differ from the source there.

A format that holds a transformation only approximately is used only with
`--lossy`. This is the case, for example, when a field interpolated with
cubic splines is written as the values at its nodes, which the target
format reads back with linear interpolation. A format that holds the
transformation exactly is still preferred when one is available.

## Positional arguments

| Name    | Type     | Description                                                                              |
| ------- | -------- | ---------------------------------------------------------------------------------------- |
| `input` | `source` | The object to convert, given as a [source](reslice.md#source-specifications). |

## Options

| Flag             | Type    | Description                                                                                                              | Default    |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `-h`, `--help`   |         | Show the help message and exit.                                                                                          |            |
| `-o`, `--output` | `path`  | Path to write the converted object to. Its name selects the target format.                                              | *required* |
| `--to`           | `str`   | Format hint of the target, such as `itk` or `niftyreg`, for a name that several formats claim or that none claims.      |            |
| `--kind`         | `str`   | Kind of the input object: `auto`, `image` or `transform`.                                                                | `auto`     |
| `--reference`    | `path`  | Reference (fixed) image, for a target format that needs its geometry, such as an FSL warp or a FLIRT matrix.            |            |
| `--moving`       | `path`  | Moving (source) image, for a target format that needs its geometry, such as an FSL warp or a FLIRT matrix.              |            |
| `--lossy`        |         | Allow a target format that holds the transformation only approximately.                                                 |            |

## Examples

The input is a source specification, so its hints select the reader and its
options give the reader what the file does not hold. An FSL warp, for
example, needs the moving image it was estimated for:

```shell
brainhops convert 'warp.nii.gz|fnirt|moving:[source.nii.gz]' -o warp.mif
```

A transformation may be followed by operators, which are applied before it
is written. The inverse of an ANTs affine is written as an AFNI matrix with

```shell
brainhops convert 'affine0GenericAffine.mat|inv' -o inverse.aff12.1D
```

The images that a target format needs are given with `--reference` and
`--moving`, and formats that do not need them ignore them:

```shell
brainhops convert warp.nii.gz -o fnirt_warp.nii.gz --to fnirt \
    --reference ref.nii.gz --moving source.nii.gz
```

## Exit status

The command exits with `0` when the object is written, `1` when the input
cannot be read, and `3` when no format can write the object to the output
name, with the reason that each candidate format gives.
