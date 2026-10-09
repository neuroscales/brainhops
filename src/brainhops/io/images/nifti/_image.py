import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.common._nifti_units import nifti_to_unit
from brainhops.io.common.nifti import (
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
from brainhops.io.images.base import WritableFileBasedImage


@register_format
class NiftiImage(NiftiParser, WritableFileBasedImage, SingleScaleImage):
    """An image stored in a NIfTI file.

    !!! note "Why the bases are in this order"
        `SingleScaleImage.data` has no default, while [`NiftiParser`][]
        contributes the defaulted fields `image` and `_header`. Fields are
        collected in reverse method resolution order, so
        [`SingleScaleImage`][] must come last; otherwise `data` would
        follow a defaulted field and `bagof` would reject the signature.
        Leading with [`NiftiParser`][] also lets its lazy `data` and
        `system` properties, which are read from the nibabel image on
        demand, take precedence over plain fields.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a plain image.

        Any NIfTI file can be read as an image, but a displacement-field
        intent suggests that the file is meant as a transformation, so it
        is only a weak match.
        """
        intent = _nifti_intent(header)
        if intent is None:
            return Confidence.MAYBE
        if intent in _NIFTI_FIELD_INTENTS:
            return Confidence.WEAK
        # The intent is often unset, so the shape is checked too: a trailing
        # axis of length 3 on a 4D or larger image may be a deformation.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.WEAK
        if intent == _NIFTI_INTENT_NONE:
            return Confidence.LIKELY
        return Confidence.MAYBE

    @property
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        The decoded transformations are cached while the image holds the
        same header object, so `img.transformation is img.transformation`.
        A header modified in place is not noticed; a modified copy should
        be assigned instead.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        header = self.header
        if header is None:
            return list(getattr(self, "_transformations", None) or [])
        decoded = getattr(self, "_decoded", None)
        if decoded is None or decoded[0] is not header:
            decoded = (header, _nifti_to_transformations(header))
            self._decoded = decoded
        return list(decoded[1])

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """Build the nibabel image that encodes this image.

        The data become the NIfTI array. The preferred transformation
        becomes the sform, and the qform is a rigid transformation among
        the others or the rigid part of the sform. Large arrays are
        written as NIfTI-2 and smaller ones as NIfTI-1.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose description and intent are copied. The geometry
            always comes from this image.
        **overrides
            Values of `dtype`, `intent` and `descrip`, which take
            precedence over the derived values and `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.

        Raises
        ------
        WriterError
            If the image has no data.
        UnrepresentableTransformationError
            If the preferred transformation is not affine, as for a
            displacement field.
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
    """Convert a NIfTI header to transformations.

    The transformations are, in order:

    1. A transformation from voxel to scaled voxel space, named "physical";
    2. The qform voxel-to-RAS rigid transformation, named "qform";
    3. The sform voxel-to-RAS affine transformation, named "sform";
    4. The qform voxel-to-RAS rigid transformation, named after its code.
    5. The sform voxel-to-RAS affine transformation, named after its code.

    The code names are `"unknown"`, `"scanner"`, `"aligned"`,
    `"talairach"`, `"mni"` and `"template"`. A form whose code is zero is
    not set and is omitted, along with its entry named after the code.
    When the qform and the sform have the same code, the fourth entry is
    omitted. As a result, the last transformation is the affine that
    nibabel's `get_best_affine()` returns, named after its code.

    The NIfTI affine applies to the spatial axes. When there are other
    axes, each voxel-to-RAS transformation is a `Sequence` made of a
    `SubspaceTransformation` of the spatial axes and, for a time axis, one
    that scales by `pixdim[4]` and translates by `toffset`. Other axes pass
    through, and a time axis without a valid spacing stays a frame index
    in every space. An image with only spatial axes gets a plain `Affine`.
    """

    xforms = []

    axes = _nifti_to_axes(header)
    # Units go through the single NIfTI-to-brainhops converter, which reads
    # an unknown spatial unit as millimeters and leaves an unknown time
    # unit unspecified (see `brainhops.io.common._nifti_units`).
    space, time = header.get_xyzt_units()
    units = {
        "space": nifti_to_unit(space, "space"),
        "time": nifti_to_unit(time, "time"),
    }
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

    named_axes = [axis for axis in axes if axis.name is not None]

    # A time axis whose spacing is zero or non-finite (pixdim[4] = 0 when
    # the step is missing) has no TR. It is not mapped to time and stays a
    # frame index in every space.
    zooms = header.get_zooms()
    zooms = [zooms[i] for i, axis in enumerate(axes) if axis.name is not None]
    untimed = {
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and not (np.isfinite(zooms[j]) and zooms[j])
    }
    zooms = [1.0 if j in untimed else z for j, z in enumerate(zooms)]

    # The axes count samples, in Fortran order as nibabel reads them.
    voxel_space = CoordinateSystem(name="voxel", axes=named_axes, order="F")

    # Axes that are neither space nor time, such as channels, have no unit
    # rather than inheriting the index unit.
    phys_axes = [
        axis if j in untimed else replace(axis, unit=units.get(axis.type))
        for j, axis in enumerate(named_axes)
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

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

    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=zooms)
    xforms.append(vox2phys)

    # The voxel space holds only the named axes, so these positions are not
    # always the positions in the header.
    named = [i for i, axis in enumerate(axes) if axis.name is not None]
    space_pos = [named.index(i) for i in keep_dims]
    time_pos = [
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and j not in untimed
    ]
    product = bool(space_pos) and len(space_pos) < len(named_axes)

    def _coded_affine(matrix: np.ndarray, label: str) -> Transformation:
        """Build the voxel-to-RAS transformation of a qform or sform."""
        matrix = matrix[keep_dims + [-1], :][:, keep_dims + [-1]]
        if product:
            return _coded_product(matrix, label)
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=label),
            matrix=matrix[:-1],
        )

    def _coded_product(matrix: np.ndarray, label: str) -> Sequence:
        """Build the voxel-to-RAS transformation of an image with more axes.

        The affine maps the spatial axes and the time axis is mapped by its
        own spacing and origin, as a sequence of subspace transformations.
        """
        world = replace(ras_space, name=label)
        # Between the two steps, the spatial axes are in world coordinates
        # while the others are still voxel indices. That space is neither
        # the voxel nor the world space, so it has no name.
        middle_axes = list(voxel_space.axes)
        for j in space_pos:
            middle_axes[j] = world.axes[j]
        middle = CoordinateSystem(axes=middle_axes) if time_pos else world
        spatial = SubspaceTransformation(
            transformation=Affine(
                matrix=matrix[:-1],
                input=voxel_space.restrict(space_pos),
                output=world.restrict(space_pos),
            ),
            input_axes=space_pos,
            output_axes=space_pos,
            input=voxel_space,
            output=middle,
        )
        steps = [spatial]
        if time_pos:
            # t_world = toffset + dt * t_index: the spacing gives the time
            # since the first frame, and `toffset` shifts it to world time.
            (it,) = time_pos
            dt = zooms[it]
            toffset = float(header["toffset"])
            temporal = Sequence(
                transformations=[
                    Scaling(
                        scale=[dt],
                        input=voxel_space.restrict([it]),
                        output=phys_space.restrict([it]),
                    ),
                    Translation(
                        translation=[toffset],
                        input=phys_space.restrict([it]),
                        output=world.restrict([it]),
                    ),
                ],
                input=voxel_space.restrict([it]),
                output=world.restrict([it]),
            )
            steps.append(
                SubspaceTransformation(
                    transformation=temporal,
                    input_axes=[it],
                    output_axes=[it],
                    input=middle,
                    output=world,
                )
            )
        return Sequence(transformations=steps, input=voxel_space, output=world)

    # get_qform and get_sform return None for a form whose code is 0, and
    # only one of the two forms is required.
    qmatrix, qcode = header.get_qform(coded=True)
    qform = None
    if qmatrix is not None:
        qname = _NIFTI_XCODES.get(qcode, "unknown")
        qform = _coded_affine(qmatrix, "qform")
        xforms.append(qform)

    smatrix, scode = header.get_sform(coded=True)
    sform = None
    if smatrix is not None:
        sname = _NIFTI_XCODES.get(scode, "unknown")
        sform = _coded_affine(smatrix, "sform")
        xforms.append(sform)

    # The last transformation must be nibabel's best affine, named after
    # its code. It is rebuilt under that name rather than renamed, so that
    # the world space is named consistently in every step of a sequence.
    if sform is not None and qform is not None:
        if scode == qcode:
            xforms.append(_coded_affine(smatrix, sname))
        else:
            xforms.append(_coded_affine(qmatrix, qname))
            xforms.append(_coded_affine(smatrix, sname))
    elif sform is not None:
        xforms.append(_coded_affine(smatrix, sname))
    elif qform is not None:
        xforms.append(_coded_affine(qmatrix, qname))

    return xforms
