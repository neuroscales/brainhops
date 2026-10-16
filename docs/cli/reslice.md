# reslice

=== ":octicons-terminal-24:"

    ```shell
    brainhops reslice SOURCE --reference SOURCE [--transform SOURCE ...] --output IMAGE [--degree DEGREE] [--bound BOUND] [--clip MARGIN] [--dtype DTYPE]
    ```

The `reslice` command resamples an image onto the grid of a reference image,
after applying a chain of transformations to it.

Each `--transform` is a forward (push) map. It moves the input image through
world space, in the same direction as the image itself travels, and the
transformations are applied in the order they appear on the command line,
so the first `--transform` is applied first. Resampling, however, works by
pulling values: the command inverts the composed map internally and samples
the input image at the locations that the reference grid maps to. The
arguments are therefore written in the forward direction, while the
resampling runs in the opposite direction.

## Positional arguments

| Name    | Type     | Description                                                   |
| ------- | -------- | ------------------------------------------------------------- |
| `input` | `source` | The image to resample, given as a [source](#source-specifications). |

## Options

| Flag                | Type     | Description                                                                                                                       | Default    |
| ------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `-h`, `--help`      |          | Show the help message and exit.                                                                                                   |            |
| `-r`, `--reference` | `source` | Reference image whose geometry defines the output grid and its placement in world space.                                         | *required* |
| `-t`, `--transform` | `source` | A forward (push) transformation to apply to the input image. Repeat the option to apply several, in the order given.             |            |
| `-o`, `--output`    | `path`   | Path to write the resampled image to.                                                                                             | *required* |
| `--degree`          | `int`    | Spline degree (`0` = nearest, `1` = linear, `3` = cubic).                                                                         | `1`        |
| `--bound`           | `str`    | Boundary condition used outside the field of view.                                                                                | `reflect`  |
| `--clip`            | `literal` | Scalar margin, asymmetric pair, or list of pairs; zero samples outside the resulting closed voxel box. | `None` |

## Source specifications

The input image, the reference image and each transformation are given as
source specifications. A source specification is a path, optionally
followed by pipe-separated segments that guide how the file is read.
Each segment is one of three things:

- A format hint, such as `nifti` or `flirt`, restricts the readers that
  are tried.
- A named option, written `name:value`, is passed to the reader.
- An operator, such as `inv`, is applied to a transformation once it has
  been read. Operators are only accepted after a transformation.

The value of an option may itself be a source specification, enclosed in
square brackets. A FLIRT matrix, for example, needs both images it was
estimated from, and the reference image can be given inline:
`affine.mat|flirt|reference:[ref.nii.gz]|inv`. The `|` character usually
needs shell quoting, and a literal `|` in a path is written `%7C`.

## Transform operators

The operators of a transformation are applied in the order they are
written, so `warp.nii.gz|inv` reads `warp.nii.gz` and then inverts it.

| Operator | Description                          | Status                      |
| -------- | ------------------------------------ | --------------------------- |
| `inv`    | Inverts the transform.               | Implemented                 |
| `sqrt`   | Matrix square root of the transform. | Not implemented (issue #47) |
| `square` | Matrix square of the transform.      | Not implemented (issue #47) |

Many warp files are stored as pull maps, which run from the reference image
to the moving image, rather than as the push maps that `reslice` expects.
ANTs, SPM and FSL warps are usually of this kind, and the `inv` operator
inverts such a file before it is composed with the other transformations.

A warp may also be stored as a stationary velocity field, whose flow at
time one is the map. Such a file is read with the `svf` hint, which is an
alias of `displacements|log:true`. The number of squaring steps used to
integrate the velocity is an option of that hint:

```shell
--transform "velocity.nii.gz|svf|steps:6"
```

A NiftyReg velocity field or velocity control-point grid (written by
`reg_f3d -vel`) identifies itself as a velocity and needs no hint. The
exponential and the logarithm are therefore not operators: whether a file
holds a velocity is a property of how it is read. The Python interface
describes the same encoding in
[Operators](../start/python.md#operators).

### Clipping and extrapolation

`--clip` sets a voxel margin beyond which output samples are zero. It is
independent of `--bound`, which extends the interpolation stencil near an edge.
By default, no cutoff is applied and the existing extrapolation is preserved.

For example, `--bound nearest --clip 0.5` extends edge values up to the outer
edges of the border voxels, then returns zero beyond them. `--clip 0` cuts at
border voxel centres; `--clip -0.5` cuts at their inner edges. Both endpoints
are included: the accepted interval on an axis of length `n` is
`[-clip, n - 1 + clip]`. Coordinates are never clamped by this option.

In Python, the same option is available as
`image.reslice(reference, bound="nearest", clip=0.5)` and on the spline
`pull`/`pull_axes` functions. Clipped samples are zero even with a nonzero
numeric boundary value. Dask clipping remains lazy; separable reslicing builds
masks per axis group instead of materialising a full coordinates field.


For asymmetric clipping, pass a quoted pair such as `--clip '(0, 0.5)'`.
A list such as `--clip '[(0, 0.5), (0.5, 0), (-0.5, -0.5)]'` sets
margins for three input axes. Python accepts these tuples/lists directly.
A pair `(lower, upper)` accepts `-lower <= x <= n - 1 + upper`;
per-dimension pairs follow **input** axes even if the output permutes them.
For `pull_axes`, they follow the order of its sampled `axes` argument.
Spline degree remains a scalar shared by the sampled axes.

The cutoff box is closed. Some native packages use half-open (ITK/ANTs) or
open (MRtrix) boxes, so their exact endpoint samples can still differ from
`clip=0.5`. Away from those endpoints, nearest-edge extrapolation plus
half-voxel clipping reproduces their linear image boundary rule. Elastix's
linear B-spline image interpolation instead uses `bound="mirror"` within
the half-voxel box. Clipping is independent of a nonlinear transform field's
own boundary convention.
