---
icon: octicons/rocket-24
---

<p align="center">
  <img src="images/logo.png" alt="brainhops logo" width="50%" />
</p>

# Getting started

## Installation

```shell
pip install brainhops
```

!!! warning "Early development"
    brainhops is in a very early stage of development, and its interfaces
    may change at any time.

## Description

brainhops is a library for applying spatial transformations to images. It
aims to support most of the image and transformation formats used in
neuroimaging and microscopy.

A central aim of brainhops is to scale to very large images. Arrays can
live in several backends (`numpy`, `cupy` and `dask.array`), so the same
code can run on a GPU or in parallel over chunks of an image that does not
fit in memory.

brainhops has two interfaces:

- The [command-line interface](start/cli.md), invoked as `brainhops`,
  exposes a subset of the library. Its `reslice` command applies a chain of
  transformations to an image and resamples the result onto a reference
  grid. Commands for combining transformations and for converting between
  formats are planned but not implemented yet.
- The [Python interface](start/python.md), imported as `brainhops`, models
  many kinds of spatial transformation and reads and writes most
  neuroimaging and microscopy formats.
