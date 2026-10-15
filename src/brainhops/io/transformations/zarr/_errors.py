class OmeFieldError(ValueError):
    """An OME-Zarr field that cannot be read.

    The error is raised, in particular, for a displacement field placed
    by a transformation that is not affine, since displacements in world
    units cannot be rescaled to voxels without a linear part. A field
    whose axes cannot be read as one vector field raises an
    [`AxisError`][brainhops.errors.AxisError] instead.
    """
