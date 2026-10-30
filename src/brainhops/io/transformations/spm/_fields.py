"""
SPM deformation fields, stored in NIfTI files prefixed with `y_` or `iy_`.
"""

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol
from bagof.magic import Inherit, KwOnly, NoRepr

# internals
from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.common.nifti import NiftiMetadata
from brainhops.io.common.nifti._internal_api import NiftiObject, header_of
from brainhops.io.transformations.base._internal_api import (
    data_image,
    image_with,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinateField,
    homogeneous_matrix,
)
from brainhops.io.transformations.nifti import (
    NiftiRASCoordinateField,
    NiftiRASToVoxel,
)
from brainhops.io.transformations.nifti._internal_api import (
    NiftiImageBackedTransformation,
    consolidate_nifti,
    coordinates_intent,
    field_to_disk,
    field_to_model,
)


@register_format
class SpmCoordinateField(
    NiftiImageBackedTransformation, _xforms.DerivedSequence
):
    """
    Field of RAS coordinates written by SPM.

    The file names of these fields often start with `y_` or `iy_`. The field is
    an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of two named slots, `ras2voxel` and `rasfield`: editing it in place
    raises `TypeError`. It maps RAS to RAS: a point is carried to the
    voxels of the field's grid, where the field holds the RAS position it
    maps to.

    It is written as SPM writes it: the RAS coordinates, as values, in a
    NIfTI file whose voxel-to-RAS affine is the inverse of `ras2voxel`.
    Another transformation is converted to it exactly, or not at all (see
    [`brainhops.io.transformations.spm._converters`][]).
    """

    HINTS = ("spm",)

    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ("y_", "iy_")
    """
    File name prefixes of SPM fields.

    An SPM field is a plain NIfTI field with a naming convention, so the prefix
    is the only thing that distinguishes it from a generic field. A parser that
    constrains the prefix is more specific and wins ties.
    """

    @classmethod
    def score_header(cls, header: NiftiObject) -> float:
        """
        Score a header as an SPM field.

        The score is that of [`NiftiRASCoordinateField`][], since the content
        of the file cannot distinguish the two readers, and pretending
        otherwise would invent evidence. The `y_` or `iy_` prefix decides the
        tie instead, through
        [`PREFIXES`][brainhops.io.transformations.spm.SpmCoordinateField.PREFIXES].
        """
        return NiftiRASCoordinateField.score_header(header)

    # --- data -----------------------------------------------------------

    _data: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The coordinates given to the field, or `None` to decode them.

    Coordinates stored here are pending until [`consolidate`][] or a
    write encodes them into `rawdata`.
    """

    @smartproperty(
        invalidates="transformations",
        cache=True,
        invalidated_by="rawdata",
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The RAS coordinates as an (X, Y, Z, 3) array.

        Coordinates that are assigned are returned as they are, and stay
        pending until [`consolidate`][] or a write encodes them. Otherwise
        they are decoded from `rawdata`: NIfTI stores the field with
        shape (X, Y, Z, 1, 3), and the singleton axis is dropped. A file
        whose fourth axis is longer than one holds a field per volume,
        which is not supported yet, so reading its data raises
        `ParserContentError`. Assigning the data drops the chain decoded
        from the previous data.
        """
        if self.rawdata is None:
            return None
        return field_to_model(self.rawdata)

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None, var=True
    )

    # --- chain --------------------------------------------------------
    # The stored chain is a tuple, as `ImmutableSequence` declares it, so
    # the derived and the assigned chain are both immutable, and in-place
    # edits such as `del field[0]` are refused.

    @smartproperty(cache=True, invalidated_by="_image")
    def transformations(
        self,
    ) -> tx.Tuple[RASToVoxel, RASCoordinateField]:
        """The transformations, decoded from the NIfTI file unless stored.

        The decoded chain is made of the RAS-to-voxel affine of the header
        and the field of RAS coordinates of the file. Both hold the metadata
        of this field, and the field holds its `rawdata` array and its
        pending
        coordinates too.
        """
        return (
            NiftiRASToVoxel(metadata=self.metadata),
            NiftiRASCoordinateField(
                data=self._data, rawdata=self.rawdata, metadata=self.metadata
            ),
        )

    @transformations.setter
    def transformations(
        self,
        value: tx.Optional[
            tx.Tuple[
                tx.Optional[RASToVoxel],
                tx.Optional[RASCoordinateField],
            ]
        ],
    ) -> None:
        # None means that the chain is decoded from the NIfTI file.
        self._transformations = None if value is None else tuple(value)

    @property
    def ras2voxel(self) -> tx.Optional[RASToVoxel]:
        """The transformation from RAS to voxel space."""
        xform = self.transformations[0]
        if xform is None:
            xform = NiftiRASToVoxel(metadata=self.metadata)
        return xform

    @property
    def rasfield(self) -> tx.Optional[RASCoordinateField]:
        """The field of RAS coordinates."""
        xform = self.transformations[1]
        if xform is None:
            xform = NiftiRASCoordinateField(
                data=self._data, rawdata=self.rawdata, metadata=self.metadata
            )
        return xform

    @ras2voxel.setter
    def ras2voxel(self, value: RASToVoxel) -> None:
        self._transformations = (value, self.transformations[1])

    @rasfield.setter
    def rasfield(self, value: RASCoordinateField) -> None:
        self._transformations = (self.transformations[0], value)

    # --- writing ------------------------------------------------------

    def _consolidate(self, **options: tx.Any) -> tx.Self:
        """Encode the pending coordinates or chain through the image.

        The field is written as SPM writes it: the RAS coordinates, as
        values, under the `VECTOR` intent (1007) with the intent name
        `"Mapping"`. A header that already has this intent keeps its
        intent parameters.

        A field whose chain was not assigned keeps its coordinates,
        assigned or read, and the geometry of its header. A field whose
        chain was assigned is encoded from the chain: the coordinates of
        `rasfield` become the array, and the inverse of `ras2voxel`
        becomes the voxel-to-RAS matrix of the grid, which replaces the
        geometry of the header.

        Raises
        ------
        WriterError
            If the field has no coordinates.
        """
        intent = coordinates_intent(header_of(self))
        changes = {"data": None, "transformations": None}
        stored = self._data is not None or self.rawdata is not None
        if self._transformations is None and stored:
            return consolidate_nifti(
                self,
                data_image(self, field_to_disk),
                intent,
                changes,
                **options,
            )
        what = "An SPM deformation field"
        # The inverse of a `NiftiRASToVoxel` read from a header is that
        # header's affine itself, rather than an inverse of its inverse.
        vox2ras = homogeneous_matrix(self.ras2voxel.inverse(), what, ndim=3)
        # SPM stores sampled coordinates.
        coordinates = self.rasfield.values
        if coordinates is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        image = image_with(self, field_to_disk(coordinates), vox2ras)
        return consolidate_nifti(self, image, intent, changes, **options)
