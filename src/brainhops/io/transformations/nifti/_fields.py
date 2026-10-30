"""
RAS coordinate fields and RAS displacement fields stored in NIfTI files.

The intent codes and the reasons for them are described in
[`brainhops.io.transformations.nifti`][].
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Inherit, NoRepr

# internals
from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import BoundaryCondition, StoreEnum
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._internal_api import (
    NIFTI_INTENT_DISPVECT,
    NIFTI_INTENT_NAME_ITK_COORDINATES,
    NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
    NIFTI_INTENT_NAME_NIFTYREG,
    NIFTI_INTENT_VECTOR,
    NiftiObject,
    header_of,
    nifti_intent,
    nifti_intent_name,
    nifti_shape,
)
from brainhops.io.images.nifti import NiftiImage
from brainhops.io.transformations.base.fields import (
    COORDINATE_HINTS,
    DISPLACEMENT_HINTS,
    RASCoordinateField,
    ras_displacement_chain,
    split_ras_displacement_chain,
)

# this format
from ._base import (
    NiftiImageBackedTransformation,
    consolidate_through,
    coordinates_intent,
    data_image,
    image_with,
    intent_overrides,
)
from ._views import field_to_disk, field_to_model

_NDIM = 3
"""Spatial dimension supported by the displacement reader."""


_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


@register_format
class NiftiRASCoordinateField(
    NiftiImageBackedTransformation, RASCoordinateField
):
    """Field of RAS coordinates stored in a NIfTI file.

    Each voxel of the file holds the RAS position, in millimetres, to
    which the field maps the centre of that voxel. The field keeps its
    file in a [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], as
    it was read or as the field was last consolidated, and `rawdata`
    and `metadata` read that image. [`from_image`][] builds a field
    from such an image, and [`to_image`][] returns the image that a
    file of the field holds. Assigned coordinates are pending until the
    field is consolidated or written, which encodes them through the
    image, so a field that is read and written again without changes is
    copied byte for byte.
    """

    HINTS = COORDINATE_HINTS

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as a field of RAS coordinates.

        A `VECTOR` image (1007), the intent that this reader and SPM write,
        scores `Confidence.CERTAIN`. A `DISPVECT` image (1006) is left to
        [`NiftiRASDisplacementField`][], and a file named `"NREG_TRANS"` to
        [`brainhops.io.transformations.niftyreg`][]. A file named
        `"ITKDisplacement"` or `"ITKCoordinates"`, which brainhops writes
        on ITK fields, is left to the ITK readers. The ITK displacement
        reader ties on any other `VECTOR` image in the ITK layout unless its
        intent name is `"Mapping"`, so a hint decides (see
        [`brainhops.io.transformations.itk.nifti`][]). Any other file whose
        trailing axis has length 3 scores `Confidence.MAYBE`.
        """
        intent = nifti_intent(header)
        if nifti_intent_name(header) in (
            NIFTI_INTENT_NAME_NIFTYREG,
            NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
            NIFTI_INTENT_NAME_ITK_COORDINATES,
        ):
            # Only the NiftyReg readers decode what intent_p1 says the file
            # holds, and only the ITK readers read a field that brainhops
            # wrote as one.
            return Confidence.NO
        if intent == NIFTI_INTENT_VECTOR:
            return Confidence.CERTAIN
        if intent == NIFTI_INTENT_DISPVECT:
            return Confidence.NO
        # The intent is often unset, so fall back on the shape. A plain image
        # scores zero rather than compete with the affine readers.
        shape = nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    @smartproperty(
        invalidates=_FORGET_VIEWS,
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The field as an (X, Y, Z, 3) array, which its `field` view reads.

        An array that is assigned is returned as it is, and stays pending
        until [`consolidate`][] or a write encodes it. Otherwise, the data
        is decoded from `rawdata`. NIfTI stores a vector field with
        shape (X, Y, Z, 1, 3), and the data is decoded without the
        singleton axis, since the array would otherwise be sampled as a
        4-D grid of vectors. A file whose fourth axis is longer than one
        holds a field per volume, which is not supported yet, so reading
        its data raises `ParserContentError`.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    def _consolidate(self, **options: tx.Any) -> tx.Self:
        """Encode the pending coordinates through the stored image.

        The coordinates are written under the `VECTOR` intent (1007)
        with the SPM intent name `"Mapping"`, which marks a coordinate
        field, rather than under `DISPVECT` (1006), which ITK and
        brainhops read as displacements. A header that already has this
        intent keeps its intent parameters. The field holds no grid of
        its own, so the image keeps the geometry of its header, and a
        field without metadata gets the identity. NIfTI stores sampled
        coordinates, so a field that stores spline coefficients is
        converted to its values, and the copy then stores values.
        """
        if StoreEnum(self.store) is not StoreEnum.values:
            data = self.to(store="values").data
            image = image_with(self, field_to_disk(data))
        else:
            image = data_image(self, field_to_disk)
        intent = coordinates_intent(header_of(self))
        changes = {"data": None, "store": StoreEnum.values}
        return consolidate_through(self, image, intent, changes, **options)


@register_format
class NiftiRASDisplacementField(
    NiftiImageBackedTransformation, _xforms.DerivedSequence
):
    """
    Field of RAS displacements stored in a NIfTI file.

    The file holds a `DISPVECT` field (1006), in which each voxel holds the
    displacement, in RAS millimetres, of the point at its centre. The field
    maps RAS to RAS as `x -> x + u(x)`, which is how ITK 5.4 and later read and
    write `DISPVECT`. A displacement field adds its values in the units of its
    own grid, so this field is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three named slots:

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the field's voxels |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the field's voxels back to RAS world        |

    The displacements are interpolated linearly and extrapolated with the
    nearest value outside the grid, as by the default
    `DisplacementFieldTransform` of ITK.

    The file may hold a stationary velocity instead, whose flow at time
    one is the map: the standard has no code for one, so it is said with
    the `log` option (`warp.nii.gz|displacements|log:true`, or its alias
    `warp.nii.gz|svf`; in Python, `load(path, log=True)`). The
    `displacement` slot is then a
    [`StationaryVelocityField`][brainhops.datamodel.transformations.\
StationaryVelocityField], integrated with `steps` squaring steps
    (`|svf|steps:6`). The field is written in the encoding the options
    say: a velocity when `log` is set, and otherwise the displacement,
    which a velocity is integrated into.

    Another transformation is converted to this format by its converters
    ([`brainhops.io.transformations.nifti._converters`][]), exactly or not
    at all. Its chain is carried over, rather than re-read from a NIfTI
    header that comes with it and says something else (a NiftyReg file
    holds positions, say). Its encoding is not: `log` and `steps` are this
    format's options, so a velocity converted here is written as its
    displacement unless `log=True` is given -- to the conversion, or to
    `save`.

    The field keeps its file in a
    [`NiftiImage`][brainhops.io.images.nifti.NiftiImage], as it was
    read or as the field was last consolidated, and `rawdata` and
    `metadata` read that image. [`from_image`][] builds a field from
    such an image, and [`to_image`][] returns the image that a file of
    the field holds. Assigned vectors and an assigned chain are pending
    until the field is consolidated or written, which encodes them
    through the image. A chain that is encoded gives its grid to the
    image as its voxel-to-world transformation, so the sform and the
    qform are written as for any NIfTI image.
    """

    HINTS = DISPLACEMENT_HINTS

    degree: tx.ClassVar[int] = 1
    """Spline degree of the interpolation."""

    bound: tx.ClassVar[BoundaryCondition] = BoundaryCondition.nearest
    """Boundary condition outside the field of view."""

    log: bool = False
    """
    Whether the file holds the stationary velocity whose flow is
    the map, rather than its displacement.
    """

    steps: tx.Optional[int] = None
    """
    The number of squaring steps that integrate a velocity (`log` only).
    If `None`, the default of [`StationaryVelocityField`][].
    """

    _data: NoRepr[tx.Optional[ArrayProtocol]] = None
    """
    The vectors given to the field, or `None` to decode them from `rawdata`.

    Vectors stored here are pending until [`consolidate`][] or a write
    encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The stored vectors as an (X, Y, Z, 3) array.

        Vectors that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata`: NIfTI stores the field with
        shape (X, Y, Z, 1, 3), and the singleton axis is dropped. A file
        whose fourth axis is longer than one holds a field per volume,
        which is not supported yet, so reading its data raises
        `ParserContentError`. The vectors are the RAS displacements of
        the file, or its velocity with `log`. Assigning the data drops
        the chain decoded from the previous data.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    # --- reading ------------------------------------------------------

    @classmethod
    def from_image(cls, image: NiftiImage, **options: tx.Any) -> tx.Self:
        """Build a displacement field from the image that stores it.

        The readers call this method, so the encoding options that they
        are given are checked here: `steps` is the number of squaring
        steps of a velocity, so it is refused without `log`.

        Parameters
        ----------
        image : NiftiImage
            The image, which is consolidated first if it has pending
            data or transformations.
        **options : Any
            The encoding options `log` and `steps`, and the other fields
            of the field.

        Returns
        -------
        NiftiRASDisplacementField
            The field.

        Raises
        ------
        TypeError
            If the image is not a `NiftiImage`, or if `steps` is given
            without `log`.
        """
        if options.get("steps") is not None and not options.get("log"):
            raise TypeError(
                "steps is the number of squaring steps of a velocity, so "
                "it is given with log=true (or the hint svf)."
            )
        return super().from_image(image, **options)

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as a field of RAS displacements.

        Only a `DISPVECT` image (1006) is claimed, with `Confidence.CERTAIN`.
        Other images may hold coordinates and are left to
        [`NiftiRASCoordinateField`][].
        """
        if nifti_intent(header) == NIFTI_INTENT_DISPVECT:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- endpoints ----------------------------------------------------
    # The endpoints are declared rather than read off the chain, which would
    # decode the data.

    @smartproperty(cache=True)
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """
        The anatomical space that the field maps from, in RAS millimetres.
        """
        return _systems.RASmm()

    @smartproperty(cache=True)
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The anatomical space that the field maps to, in RAS millimetres."""
        return _systems.RASmm()

    # --- decoding -----------------------------------------------------

    def _vox2ras(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS affine of the grid.

        The affine is the grid that the stored image decodes from its
        header, which is the sform, else the qform, else the default
        grid of a header whose codes are both zero. A field without
        metadata has the identity.
        """
        header = header_of(self)
        if header is None:
            return np.eye(4)
        return np.asarray(header.get_best_affine(), dtype=np.float64)

    def _ras_vectors(self) -> ArrayProtocol:
        """
        Return the stored displacements, in RAS millimetres.

        A field whose array is not a grid of vectors with three components
        is refused.
        """
        data = self.data
        if data is None:
            raise ParserContentError("This field has no data to read.")
        if len(data.shape) != 4:
            shape = tuple(int(d) for d in data.shape)
            raise ParserContentError(
                f"A NIfTI displacement field is stored as a (X, Y, Z, 1, 3) "
                f"array, and its vectors have the shape {shape}."
            )
        if int(data.shape[-1]) != _NDIM:
            raise ParserContentError(
                f"Only three-dimensional displacement fields are "
                f"supported, and this one has {data.shape[-1]} components."
            )
        return data

    # --- chain --------------------------------------------------------

    # The chain is decoded from the header, so it is decoded again when
    # the stored image is replaced.
    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations that the field encodes.

        The chain is built lazily from the header and the data, and then
        cached. Assigning a chain overrides the derived one, which is how
        fields that do not come from a file are built. Since the slots name
        fixed positions, a copy with other slots is made with `replace` rather
        than by editing the chain in place.
        """
        return ras_displacement_chain(
            self._ras_vectors(),
            self._vox2ras(),
            degree=self.degree,
            bound=self.bound,
            log=self.log,
            steps=self.steps,
        )

    @property
    def ras2voxel(self) -> tx.Optional[_xforms.Transformation]:
        """The affine from RAS world coordinates to the voxels of the field."""
        return self.transformations[0]

    @property
    def displacement(self) -> tx.Optional[_xforms.Transformation]:
        """
        The displacement field in voxel units, or the velocity with `log`.
        """
        return self.transformations[1]

    @property
    def voxel2ras(self) -> tx.Optional[_xforms.Transformation]:
        """
        The affine from the voxels of the field back to RAS world coordinates.
        """
        return self.transformations[2]

    # --- writing ------------------------------------------------------

    def _consolidate(
        self, *, log: tx.Optional[bool] = None, **options: tx.Any
    ) -> tx.Self:
        """Encode the pending vectors or chain through the stored image.

        The displacements are written in RAS millimetres under the
        `DISPVECT` intent (1006), with shape (X, Y, Z, 1, 3). With the
        option `log`, which defaults to the option of the field, the
        velocity is written instead, and otherwise a velocity is
        integrated first. The copy then has the `log` option that it was
        encoded with.

        A field whose chain is decoded from its stored vectors, in the
        encoding that it is written in, keeps its vectors, assigned or
        read, and the geometry of its header. A header that already has
        the `DISPVECT` intent keeps its intent name and parameters. A
        field whose chain was assigned is encoded from the chain, and
        the grid of the chain replaces the geometry of the header.
        """
        log = self.log if log is None else log
        changes = {"data": None, "transformations": None, "log": log}
        intent = intent_overrides(NIFTI_INTENT_DISPVECT)
        if self._transformations is None and log == self.log:
            image = data_image(self, field_to_disk)
            if nifti_intent(header_of(self)) == NIFTI_INTENT_DISPVECT:
                intent = {}
        else:
            vox2ras, vectors = split_ras_displacement_chain(
                self.transformations,
                "A NIfTI displacement field",
                ndim=_NDIM,
                log=log,
            )
            image = image_with(self, field_to_disk(vectors), vox2ras)
        return consolidate_through(self, image, intent, changes, **options)

    def to_nibabel(
        self,
        like: tx.Any = None,
        log: tx.Optional[bool] = None,
        **overrides,
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image of the displacement field.

        The field is consolidated with the options, as by
        [`consolidate`][], into a copy, and the nibabel image holds the
        `rawdata` array and the header of that copy. The field itself is
        not changed.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        log : bool, optional
            Whether to write the velocity rather than the displacements.
            It defaults to the `log` option of the field.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.
        """
        return super().to_nibabel(like=like, log=log, **overrides)
