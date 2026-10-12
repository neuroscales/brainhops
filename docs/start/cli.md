# Getting started with the command line interface

The command-line interface is a single program, `brainhops`, whose
subcommands each perform one task. The list of subcommands and their
options is printed by:

```shell
brainhops --help
```

The only subcommand implemented so far is [`reslice`](../cli/reslice.md).
It resamples an image onto the grid of a reference image after moving it
through a chain of transformations. The following command applies an
affine transformation and then a nonlinear warp to `moving.nii.gz`, and
writes the result on the grid of `fixed.nii.gz`:

```shell
brainhops reslice moving.nii.gz \
    --reference fixed.nii.gz \
    --transform affine.lta \
    --transform "warp.nii.gz|inv" \
    --output moved.nii.gz
```

Each `--transform` is a forward map that moves the input image, and the
transformations are applied in the order they are given. The `|inv`
operator inverts the warp, which is needed because most registration tools
store their warps in the opposite direction. The
[`reslice`](../cli/reslice.md) page describes these conventions in detail,
and the [command-line reference](../cli/index.md) lists every subcommand.
