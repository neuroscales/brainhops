"""
SPM deformation fields, stored in NIfTI files prefixed with `y_` or `iy_`.
"""

# dependencies
import nibabel as nb
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
from brainhops._core.properties import always_unset, smartproperty
from brainhops.backends import get_array_backend
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _NiftiObject,
    _record_image,
    _set_coordinates_intent,
)
from brainhops.io.common.nifti._views import _field_to_disk, _field_to_model
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
)
from brainhops.io.transformations.nifti import (
    NiftiBasedTransformation,
    NiftiRASCoordinatesField,
    NiftiRASToVoxel,
)


def _set_field_data(
    self: "SpmCoordinatesField", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the array of an SPM field in `raw`, as NIfTI stores it.

    The cached data is dropped, and the chain decoded from it is dropped
    by the invalidation of the property.
    """
    self.raw = None if value is None else _field_to_disk(value)
    self.__dict__.pop("_cache_data", None)


@register_format
class SpmCoordinatesField(_xforms.ImmutableSequence, NiftiBasedTransformation):
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
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as an SPM field.

        The score is that of [`NiftiRASCoordinatesField`][], since the content
        of the file cannot distinguish the two readers, and pretending
        otherwise would invent evidence. The `y_` or `iy_` prefix decides the
        tie instead, through
        [`PREFIXES`][brainhops.io.transformations.spm.SpmCoordinatesField.PREFIXES].
        """
        return NiftiRASCoordinatesField._score_nibabel(header)

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    # --- data -----------------------------------------------------------

    @smartproperty(
        cache=True,
        unset=always_unset,
        fset=_set_field_data,
        invalidates=("transformations",),
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The RAS coordinates as an (X, Y, Z, 3) array, decoded from `raw`.

        NIfTI stores the field with shape (X, Y, Z, 1, 3), and the singleton
        axis is dropped. Setting the data stores it in `raw` and drops the
        chain decoded from the previous data.
        """
        if self.raw is None:
            return None
        return _field_to_model(self.raw)

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the chain decoded from the previous
    header.
    """

    # --- chain --------------------------------------------------------
    # The stored chain is a tuple, as `ImmutableSequence` declares it, so
    # the derived and the assigned chain are both immutable, and in-place
    # edits such as `del field[0]` are refused.

    @smartproperty(cache=True)
    def transformations(
        self,
    ) -> tx.Tuple[RASToVoxel, RASCoordinatesField]:
        """The transformations, decoded from the NIfTI file unless stored.

        The decoded chain is made of the RAS-to-voxel affine of the header
        and the field of RAS coordinates of the file. Both hold the metadata
        of this field, and the field holds its `raw` array too.
        """
        return (
            NiftiRASToVoxel(metadata=self.metadata),
            NiftiRASCoordinatesField(raw=self.raw, metadata=self.metadata),
        )

    @transformations.setter
    def transformations(
        self,
        value: tx.Optional[
            tx.Tuple[
                tx.Optional[RASToVoxel],
                tx.Optional[RASCoordinatesField],
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
    def rasfield(self) -> tx.Optional[RASCoordinatesField]:
        """The field of RAS coordinates."""
        xform = self.transformations[1]
        if xform is None:
            xform = NiftiRASCoordinatesField(
                raw=self.raw, metadata=self.metadata
            )
        return xform

    @ras2voxel.setter
    def ras2voxel(self, value: RASToVoxel) -> None:
        self._transformations = (value, self.transformations[1])

    @rasfield.setter
    def rasfield(self, value: RASCoordinatesField) -> None:
        self._transformations = (self.transformations[0], value)

    # --- writing ------------------------------------------------------

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this deformation field.

        The coordinates of `rasfield`, as values, become the data array,
        with the `VECTOR` (1007) intent code and the intent name
        `"Mapping"` that SPM writes. The voxel-to-RAS affine of the grid
        is the inverse of `ras2voxel`.

        A field whose chain is decoded from its file is written back as it
        is stored, under a copy of the header of its metadata, whose
        geometry is kept. A field whose chain was assigned is encoded from
        the chain over that header, whose geometry is replaced. When `like`
        is given, non-encoding header fields are copied from it. Keyword
        arguments override header fields last.
        """
        stored = (
            self.raw is not None
            and getattr(self, "_transformations", None) is None
        )
        if stored:
            image = _record_image(
                self.raw, self._record(), overrides=overrides
            )
        else:
            what = "An SPM deformation field"
            # The inverse of a `NiftiRASToVoxel` read from a header is that
            # header's affine itself, rather than an inverse of its inverse.
            vox2ras = homogeneous_matrix(
                self.ras2voxel.inverse(), what, ndim=3
            )
            # SPM stores sampled coordinates.
            coordinates = self.rasfield.values
            if coordinates is None:
                raise WriterError(
                    "This field has no coordinates, so there is nothing to "
                    "write."
                )
            backend = get_array_backend(coordinates)
            coordinates = _field_to_disk(backend.asarray(coordinates))
            image = _record_image(
                coordinates, self._record(), vox2ras, overrides
            )
        _set_coordinates_intent(image.header)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image
