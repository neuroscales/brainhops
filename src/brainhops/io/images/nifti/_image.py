# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
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
from brainhops.io.common.nifti import NiftiParser
from brainhops.io.common.nifti._constants import (
    _NIFTI_FIELD_INTENTS,
    _NIFTI_INTENT_NONE,
    _NIFTI_XCODES,
)
from brainhops.io.common.nifti._geometry import _image_with_geometry
from brainhops.io.common.nifti._header import (
    _nifti_intent,
    _nifti_shape,
    _nifti_to_axes,
    _NiftiObject,
)
from brainhops.io.common.nifti._units import nifti_to_unit
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
        """The voxel-to-world transformations recorded by the header,
        decoded on first access unless set explicitly.

        An image built from data alone has no header, so it records no
        transformation and the list is empty.

        The decoded transformations are kept for as long as the image holds
        the same header object, so each access hands back the same
        transformations (in a new list): `img.transformation is
        img.transformation`, and a transformation composed with the inverse
        of the very same one cancels without being computed. Setting
        another header decodes it anew. A header changed in place is not
        noticed: assign a new header object (a copy, say) to decode it
        anew.
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

    The NIfTI affine applies to the spatial axes. In an image with other
    axes (time, ...), each voxel-to-RAS transformation is therefore a
    `Sequence` of a `SubspaceTransformation` that applies the affine to the
    spatial axes and, when there is a time axis, one that maps the time
    axis by its spacing and offset (`pixdim[4]`, `toffset`), as a `Scaling`
    then a `Translation`. Every other axis passes through. In an image with
    spatial axes only, it is a plain `Affine`.

    A time spacing of zero (or not finite) means that the repetition time
    is missing. The time axis is then not mapped: the sequence holds the
    spatial subspace transform only, and the time axis stays a frame index
    (unit `index`) in every space, including the physical one.
    """

    # Allocate output
    xforms = []

    # --- preliminaries ------------------------------------------------

    axes = _nifti_to_axes(header)
    # The header's units, through the one NIfTI <-> brainhops converter.
    # An unknown spatial unit reads as millimetres (the ecosystem's
    # convention), an unknown temporal one as unspecified; see
    # `brainhops.io.common.nifti._units` for every policy.
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

    # --- coordinate systems -------------------------------------------

    named_axes = [axis for axis in axes if axis.name is not None]

    # The spacing of each named axis. A time axis whose spacing is zero
    # (or not finite) has no repetition time: NIfTI writes `pixdim[4] = 0`
    # when the time step is missing. Such an axis is not mapped to time at
    # all: it stays a frame index in every space, and only the spatial
    # axes are mapped to the world.
    zooms = header.get_zooms()
    zooms = [zooms[i] for i, axis in enumerate(axes) if axis.name is not None]
    untimed = {
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and not (np.isfinite(zooms[j]) and zooms[j])
    }
    zooms = [1.0 if j in untimed else z for j, z in enumerate(zooms)]

    # >> Voxel space
    # `_nifti_to_axes` gives the axes of the voxel space: they count samples.
    # A NIfTI array is stored, and read by nibabel, in F order: the first
    # axis changes fastest.
    voxel_space = CoordinateSystem(name="voxel", axes=named_axes, order="F")

    # >> Physical space
    # The same axes, measured in the header's units. An axis of another
    # type (a channel, a vector component) has no physical unit: its unit
    # is left unspecified rather than inherit "index" from the voxel space.
    phys_axes = [
        axis if j in untimed else replace(axis, unit=units.get(axis.type))
        for j, axis in enumerate(named_axes)
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
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=zooms)
    xforms.append(vox2phys)

    # The positions, in the voxel space, of its spatial axes and of its
    # time axis. The voxel space holds the named axes only, so a position
    # in it is not always a position in the header.
    named = [i for i, axis in enumerate(axes) if axis.name is not None]
    space_pos = [named.index(i) for i in keep_dims]
    time_pos = [
        j
        for j, axis in enumerate(named_axes)
        if axis.type == "time" and j not in untimed
    ]
    product = bool(space_pos) and len(space_pos) < len(named_axes)

    def _coded_affine(matrix: np.ndarray, label: str) -> Transformation:
        """Build the voxel-to-RAS transformation for a qform/sform matrix."""
        matrix = matrix[keep_dims + [-1], :][:, keep_dims + [-1]]
        if product:
            return _coded_product(matrix, label)
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=label),
            matrix=matrix[:-1],
        )

    def _coded_product(matrix: np.ndarray, label: str) -> Sequence:
        """
        Build the voxel-to-RAS transformation of an image with more than
        spatial axes.

        The NIfTI affine applies to the spatial axes, and the time axis is
        mapped by its own spacing (`pixdim[4]`) and origin (`toffset`).
        The two act on disjoint axes, so the transformation is the
        sequence of a subspace transform over the spatial axes and, when
        there is a time axis, one over the time axis. Every other axis
        passes through.
        """
        world = replace(ras_space, name=label)
        # >> Between the two steps, the spatial axes are in world (RAS)
        #    coordinates and every other axis is still a voxel index. It is
        #    neither the voxel space nor the world space, so it is unnamed.
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
            # >> t_world = toffset + dt * t_index. The spacing scales the
            #    index into the physical time since the first frame (the
            #    time axis of the physical space), and `toffset`, which
            #    NIfTI states in the time unit, shifts it to the world
            #    time.
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
    # Each is built again under its code's name rather than renamed, so the
    # world space is named consistently at every step of a transformation
    # that has several.
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
