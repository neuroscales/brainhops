"""
SPM deformation fields, stored in NIfTI files prefixed with `y_` or `iy_`.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.common.nifti import _NiftiObject
from brainhops.io.transformations.base.affines import RASToVoxel
from brainhops.io.transformations.base.fields import RASCoordinatesField
from brainhops.io.transformations.nifti.affines import NiftiRASToVoxel
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation
from brainhops.io.transformations.nifti.fields import NiftiRASCoordinatesField


@register_format
class SpmCoordinatesField(_xforms.ImmutableSequence, NiftiBasedTransformation):
    """
    Field of RAS coordinates written by SPM.

    The file names of these fields often start with `y_` or `iy_`. The field is
    an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of two named slots, `ras2voxel` and `rasfield`, and editing it in place
    raises a `TypeError`.
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

    # The stored chain is a tuple, as ImmutableSequence declares, so in-place
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
