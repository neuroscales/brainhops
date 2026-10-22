# convert

=== ":octicons-terminal-24:"

    ```shell
    brainhops convert SOURCE --output TARGET [--kind KIND] [--lossy]
    ```

The `convert` command reads an image or a transformation stored in one
format and writes it in another, for example to turn an ITK transform into
a NIfTI displacement field. The kind of object is detected from the input
file, and the target format is chosen from the name of the output file, as
`brainhops.io.save` chooses it. Both the input and the output are written
as [source specifications](reslice.md#source-specifications), with pipes.

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

| Flag             | Type     | Description                                                                                  | Default    |
| ---------------- | -------- | -------------------------------------------------------------------------------------------- | ---------- |
| `-h`, `--help`   |          | Show the help message and exit.                                                              |            |
| `-o`, `--output` | `target` | The file to write, given as a [target specification](#target-specifications).                | *required* |
| `--kind`         | `str`    | Kind of the input object: `auto`, `image` or `transform`.                                    | `auto`     |
| `--lossy`        |          | Allow a target format that holds the transformation only approximately.                     |            |

## Target specifications

The output is written in the syntax of a source: a path, optionally
followed by pipe-separated segments. A format hint, such as `itk` or
`niftyreg`, selects the target format for a name that several formats
claim, such as `.nii.gz`, or for a name that no format claims. A named
option, written `name:value`, is an option of the conversion. The value of
an option may be a source specification in square brackets, which is then
read as an image. This is how a target format is given the images that it
needs and that the object does not hold, such as the reference and the
moving images of an FSL warp:
`warp.nii.gz|fnirt|reference:[ref.nii.gz]|moving:[source.nii.gz]`. A
format that does not take an option ignores it. The values `true` and
`false` are read as booleans, and numbers as numbers, so that
`velocity.nii.gz|log:true` writes the velocity that a field holds.

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

The images that a target format needs are given in the target
specification:

```shell
brainhops convert warp.nii.gz \
    -o 'fnirt_warp.nii.gz|fnirt|reference:[ref.nii.gz]|moving:[source.nii.gz]'
```

## Exit status

The command exits with `0` when the object is written, `1` when the input
cannot be read, and `3` when no format can write the object to the output
name, with the reason that each candidate format gives.
