"""
SPM deformation fields, stored in NIfTI files prefixed with `y_` or `iy_`.
"""

# externals
import nibabel as nb
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _NiftiObject,
)
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.fields import (
    RASCoordinatesField,
    homogeneous_matrix,
)
from brainhops.io.transformations.nifti.affines import NiftiRASToVoxel
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation
from brainhops.io.transformations.nifti.fields import (
    NiftiRASCoordinatesField,
    _ras_coordinates_nifti,
)


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
        [`PREFIXES`][brainhops.io.transformations.spm.y.SpmCoordinatesField.PREFIXES].
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

    # The stored chain is a tuple, as `ImmutableSequence` declares it, so
    # the derived and the assigned chain are both immutable, and in-place
    # edits such as `del field[0]` are refused.

    @property
    def transformations(
        self,
    ) -> tx.Tuple[
        tx.Optional[RASToVoxel],
        tx.Optional[RASCoordinatesField],
    ]:
        """The transformations, derived from the NIfTI file unless stored."""
        _transformations = getattr(self, "_transformations", None)
        if _transformations is not None:
            return _transformations
        return (
            NiftiRASToVoxel(image=self.image, header=self.header),
            NiftiRASCoordinatesField(image=self.image, header=self.header),
        )

    @transformations.setter
    def transformations(
        self,
        value: tx.Tuple[
            tx.Optional[RASToVoxel],
            tx.Optional[RASCoordinatesField],
        ],
    ) -> None:
        # None means that the chain is derived from the NIfTI file.
        self._transformations = None if value is None else tuple(value)

    @property
    def ras2voxel(self) -> tx.Optional[RASToVoxel]:
        """The transformation from RAS to voxel space."""
        xform = self.transformations[0]
        if xform is None:
            xform = NiftiRASToVoxel(image=self.image, header=self.header)
        return xform

    @property
    def rasfield(self) -> tx.Optional[RASCoordinatesField]:
        """The field of RAS coordinates."""
        xform = self.transformations[1]
        if xform is None:
            xform = NiftiRASCoordinatesField(
                image=self.image, header=self.header
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

        When `like` is given, non-encoding header fields are copied from
        it. Keyword arguments override header fields last.
        """
        what = "An SPM deformation field"
        # The inverse of a `NiftiRASToVoxel` read from a header is that
        # header's affine itself, rather than an inverse of its inverse.
        vox2ras = homogeneous_matrix(self.ras2voxel.inverse(), what, ndim=3)
        # SPM stores sampled coordinates.
        image = _ras_coordinates_nifti(self.rasfield.values, vox2ras)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image
