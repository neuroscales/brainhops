"""Readers and writers of FreeSurfer `register.dat` registrations.

A `register.dat` file is the registration that `tkregister2` and
`bbregister` write with `--reg`. It holds a 4x4 matrix from the
tkregister RAS space of the reference (target) volume to the tkregister
RAS space of the movable volume, together with the name of the subject,
the voxel size of the movable volume and two settings of `tkregister2`.
It holds no geometry of either volume, so the scanner RAS affine that it
describes is known only once the geometries of both volumes are given.

A `register.dat` file is short and is read in one pass, as an LTA file
is. Its whole content is held as a [`RegisterDatRaw`][] record by
[`RegisterDatMetadata`][], the metadata of the file, and a
[`RegisterDatTransformation`][] holds that metadata. The matrix lives in
the record, so a transformation that is read and written again without
changes is written as its file stored it.

!!! example "Reading a registration in scanner RAS"
    ```python
    from brainhops.io.transformations.freesurfer.lta import LtaRaw
    from brainhops.io.transformations.freesurfer.regdat import (
        RegisterDatTransformationRASToRAS,
    )

    reference = LtaRaw.VolumeInfo(volume=(256, 256, 256))
    moving = LtaRaw.VolumeInfo(volume=(64, 64, 36), voxelsize=(3, 3, 4))
    xform = RegisterDatTransformationRASToRAS.load(
        "register.dat", reference=reference, moving=moving
    )
    ```
"""

__all__ = [
    "RegisterDatRaw",
    "RegisterDatMetadata",
    "RegisterDatTkrSystem",
    "RegisterDatTransformation",
    "RegisterDatTransformationRASToRAS",
]

from ._metadata import RegisterDatMetadata
from ._raw import RegisterDatRaw
from ._systems import RegisterDatTkrSystem
from ._xforms import (
    RegisterDatTransformation,
    RegisterDatTransformationRASToRAS,
)

# Importing the private converters registers them with the data model,
# so that a conversion into one of these formats, such as
# `t.to(Format)`, fails with a reason instead of building a wrong
# object.
from . import _converters  # noqa: E402, F401  isort: skip
