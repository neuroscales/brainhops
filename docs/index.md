---
icon: octicons/rocket-24
---

<center><img src="images/logo.png" alt="brainhops logo" width="50%" /></center>

# Getting started

## Installation

```shell
pip install brainhops
```

## Description

`brainhops` is a python library to apply spatial transformations to
images. It aims to support most image and transformation formats used in
neuroimaging and in microscopy. Most importantly, it aims to scale to
very large images. To this end, it supports multiple array backends
(`numpy`, `cupy`, `dask.array`), which allows user to benefit from their
acceleration and parallelization capabilities.

`brainhops` can be used through two different interfaces:

- a [**command-line interface**](/start/cli/) (`brainhops --help`),
  which exposes a subset of functionalities such as

    * applying chains of transformations to images, meshes or point clouds;
    * converting between different images and transformations formats.

- a [**python API**](/start/python/) (`import brainhops`) that abstracts
  away many types of spatial transformations and interfaces with most
  neuroimaging and microscopy formats.
