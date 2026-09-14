# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import Unit
from brainhops.io.base.nifti import NiftiParser, _nifti_to_axes


class NiftiImage(SingleScaleImage, NiftiParser):
    """
    An image that is encoded by a NIfTI file.
    """

    @property
    def transformations(self) -> tx.List[Transformation]:
        if getattr(self, "_transformations", None) is not None:
            return self._transformations
        return _nifti_to_transformations(self.header)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value


_NIFTI_XCODES = {
    0: "unknown",
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}


def _nifti_to_transformations(
    header: nb.Nifti1Header,
) -> tx.List[Transformation]:
    """
    Convert a NIfTI header to a list of transformations.

    The output list contains, in order:

    1. A transformation from voxel to scaled voxel space, named "physical";
    2. The qform voxel-to-RAS rigid transformation, named "qform";
    3. The sform voxel-to-RAS affine transformation, named "sform";
    4. The qform voxel-to-RAS rigid transformation, named after its code.
    5. The sform voxel-to-RAS affine transformation, named after its code.

    Code names are one of
    {"unknown", "scanner", "aligned", "talairach", "mni", "template"}.

    The order is slightly different in two cases:
    - If the qform and sform are identical, transform 4 (the qform named
      after its code) is omitted.
    - If the sform is zero, transforms 4 and 5 are inverted (the qform
      named after its code comes after the sform named after its code).

    This is so the last transformation in the list matches nibabel's
    `get_best_affine()` function, while being named after its code.
    """

    # Allocate output
    xforms = []

    # --- preliminaries ------------------------------------------------

    axes = _nifti_to_axes(header)
    units = {}
    units["space"], units["time"] = header.get_xyzt_units()
    orientation = {
        "x": "left-to-right",
        "y": "posterior-to-anterior",
        "z": "inferior-to-superior",
    }
    keep_dims = [
        i
        for i, axis in enumerate(axes)
        if axis.name is not None and axis.type == "space"
    ]

    # --- coordinate systems -------------------------------------------

    # >> Voxel space
    voxel_axes = [axis for axis in axes if axis.name is not None]
    voxel_space = CoordinateSystem(name="voxel", axes=voxel_axes)

    # >> Physical space
    phys_axes = [
        replace(axis, unit=Unit(units.get(axis.type, axis.unit)))
        for axis in voxel_axes
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

    # >>> RAS space
    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=orientation[axis.name]
            ),
        )
        if axis.name in orientation
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    # --- voxel-to-physical --------------------------------------------
    zooms = header.get_zooms()
    zooms = [zooms[i] for i, axis in enumerate(axes) if axis.name is not None]
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=zooms)
    xforms.append(vox2phys)

    # --- qform --------------------------------------------------------
    qform, qcode = header.get_qform(coded=True)
    qform = qform[keep_dims + [-1], :][:, keep_dims + [-1]]
    qname = _NIFTI_XCODES.get(qcode, "unknown")
    qform = Affine(
        input=voxel_space,
        output=replace(ras_space, name="qform"),
        matrix=qform[:-1],
    )
    xforms.append(qform)

    # --- sform --------------------------------------------------------
    sform, scode = header.get_sform(coded=True)
    keep_dims = [
        i
        for i, axis in enumerate(axes)
        if axis.name is not None and axis.type == "space"
    ]
    sform = sform[keep_dims + [-1], :][:, keep_dims + [-1]]
    sname = _NIFTI_XCODES.get(scode, "unknown")
    sform = Affine(
        input=voxel_space,
        output=replace(ras_space, name="sform"),
        matrix=sform[:-1],
    )
    xforms.append(sform)

    # --- named & best affines -----------------------------------------
    if scode == qcode:
        xforms.append(replace(sform, name=sname))
    elif sform != 0:
        xforms.append(replace(qform, name=qname))
        xforms.append(replace(sform, name=sname))
    elif qform != 0:
        xforms.append(replace(sform, name=sname))
        xforms.append(replace(qform, name=qname))

    return xforms
