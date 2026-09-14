# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import Unit
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_FIELD_INTENTS,
    _NIFTI_INTENT_NONE,
    _NIFTI_XCODES,
    NiftiParser,
    _image_with_geometry,
    _nifti_intent,
    _nifti_shape,
    _nifti_to_axes,
    _NiftiObject,
)
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.images.base import WritableFileBasedImage


@register_format
class NiftiImage(NiftiParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a NIfTI file.

    !!! note "Why the bases are in this order"
        `SingleScaleImage.data` has no default, while `NiftiParser`
        contributes `image` and `_header`, which do. Struct fields are
        collected in reverse MRO order, so `SingleScaleImage` has to come
        *last* or `data` ends up behind a defaulted field and `bagof`
        rejects the signature. Leading with `NiftiParser` also lets its
        lazy `data`/`system` properties -- which pull from the nibabel
        image on demand -- take precedence over the plain fields.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        float a NIfTI header as a plain image.

        Any NIfTI can be read as an image, so this is never zero -- but
        a file whose intent code says "displacement field" is very
        probably wanted as a transformation, not as an image.
        """
        intent = _nifti_intent(header)
        if intent is None:
            return Confidence.MAYBE
        if intent in _NIFTI_FIELD_INTENTS:
            return Confidence.WEAK
        # An intent code is often left unset, so the shape has to be
        # read too: a trailing axis of length 3 on a 4D-or-more volume
        # may be a deformation field, and reading it as a plain image
        # would be technically valid but almost never what was wanted.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.WEAK
        if intent == _NIFTI_INTENT_NONE:
            return Confidence.LIKELY
        return Confidence.MAYBE

    @property
    def transformations(self) -> tx.List[Transformation]:
        if getattr(self, "_transformations", None):
            return self._transformations
        return _nifti_to_transformations(self.header)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this image.

        The image data becomes the NIfTI data array. The preferred
        transformation becomes the sform, and a rigid transformation among
        the others, or the rigid part of the sform, becomes the qform. A
        large array is written as NIfTI-2, and a smaller one as NIfTI-1.

        A preferred transformation that is not an affine, such as a
        displacement field, cannot describe NIfTI geometry, and raises
        `UnrepresentableTransformationError`.

        When `like` is given, non-encoding header fields such as the
        description and the intent are copied from it. The template may be a
        path to a NIfTI file, a `nibabel` image or header, or another object
        read from NIfTI. The geometry always comes from this image, never
        from the template.

        Keyword arguments override header fields after the derived values
        and after `like`, so an explicit value always wins. `dtype` sets the
        stored data type, `intent` the intent code, and `descrip` the
        description. The array's own data type is kept unless `dtype` is
        given.
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        return _image_with_geometry(
            data,
            self.transformation,
            self.transformations,
            like=like,
            overrides=overrides,
        )


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

    def _coded_affine(matrix: np.ndarray, label: str) -> Affine:
        """Build the voxel-to-RAS affine for a qform/sform matrix."""
        matrix = matrix[keep_dims + [-1], :][:, keep_dims + [-1]]
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=label),
            matrix=matrix[:-1],
        )

    # --- qform --------------------------------------------------------
    # `get_qform`/`get_sform` return `None` when the corresponding code
    # is 0, i.e. the form is simply not set. Only one of the two is
    # required to be present.
    qmatrix, qcode = header.get_qform(coded=True)
    qform = None
    if qmatrix is not None:
        qname = _NIFTI_XCODES.get(qcode, "unknown")
        qform = _coded_affine(qmatrix, "qform")
        xforms.append(qform)

    # --- sform --------------------------------------------------------
    smatrix, scode = header.get_sform(coded=True)
    sform = None
    if smatrix is not None:
        sname = _NIFTI_XCODES.get(scode, "unknown")
        sform = _coded_affine(smatrix, "sform")
        xforms.append(sform)

    # --- named & best affines -----------------------------------------
    # The last transformation in the list must be the one nibabel calls
    # the "best" affine, but named after its code rather than its form.
    if sform is not None and qform is not None:
        if scode == qcode:
            xforms.append(
                replace(sform, output=replace(ras_space, name=sname))
            )
        else:
            xforms.append(
                replace(qform, output=replace(ras_space, name=qname))
            )
            xforms.append(
                replace(sform, output=replace(ras_space, name=sname))
            )
    elif sform is not None:
        xforms.append(replace(sform, output=replace(ras_space, name=sname)))
    elif qform is not None:
        xforms.append(replace(qform, output=replace(ras_space, name=qname)))

    return xforms
