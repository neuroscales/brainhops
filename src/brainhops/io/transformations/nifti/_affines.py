"""Affines between voxel and RAS space, stored in the header of a NIfTI
file."""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.properties import (
    InvalidatorInAttribute,
    always_unset,
    smartproperty,
)
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserExistsError
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw
from brainhops.io.common.nifti._geometry import _voxel_to_ras
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _header,
    _NiftiObject,
    _record_image,
)
from brainhops.io.common.nifti._views import (
    _affine_to_disk,
    _affine_to_model,
)
from brainhops.io.transformations.base import AffineTransformationFormat
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

# this format
from ._base import NiftiBasedTransformation, _nibabel_parts

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


def _set_affine_data(
    self: "_NiftiAffine", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the matrix of an affine in `raw` as a homogeneous matrix.

    The private field in which the data model stores its data is emptied,
    so that a copy made with `replace` takes the matrix from `raw`.
    """
    self.raw = None if value is None else _affine_to_disk(value)
    self._data = None
    self.__dict__.pop("_cache_data", None)


def _set_affine_metadata(
    self: "_NiftiAffine", value: tx.Optional[NiftiMetadata]
) -> None:
    """Store the metadata of an affine and drop the matrix decoded from it.

    Without a matrix in `raw`, the matrix is decoded from the header of
    the metadata, so other metadata gives another matrix.
    """
    self._metadata = value
    self.__dict__.pop("_cache_data", None)


class _NiftiAffine(AffineTransformationFormat, NiftiBasedTransformation):
    """Shared reading, scoring and writing of the affines of a header.

    An affine read from a file keeps its matrix in the header of its
    metadata and has no `raw` array, because the array of the file is only
    a placeholder. An affine whose matrix is set holds the homogeneous
    matrix in `raw`, which then takes precedence over the header.
    """

    # --- reading ------------------------------------------------------
    # Only the header is read, since the array of the file is a placeholder
    # that carries the geometry.

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an affine from the header of a NIfTI file.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        return cls(metadata=NiftiMetadata.from_filename(filename, **kwargs))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> tx.Self:
        """Read an affine from the header at the start of a NIfTI stream."""
        return cls(metadata=NiftiMetadata.from_fileobj(file, **kwargs))

    @classmethod
    def from_nibabel(cls, nifti: _NiftiObject, **kwargs) -> tx.Self:
        """Build an affine from the header of a nibabel image or header.

        Parameters
        ----------
        nifti : nibabel.Nifti1Image or nibabel.Nifti1Header
            The nibabel image or header. NIfTI-2 images and headers are
            accepted too.
        **kwargs : Any
            Other fields of the affine.

        Returns
        -------
        _NiftiAffine
            The affine.

        Raises
        ------
        TypeError
            If `nifti` is neither a NIfTI image nor a NIfTI header.
        """
        _, header = _nibabel_parts(nifti)
        metadata = NiftiMetadata.from_raw(NiftiRaw(header=header.copy()))
        return cls(metadata=metadata, **kwargs)

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as a bare affine, always `Confidence.WEAK`.

        Every NIfTI file carries an affine, but loading a `.nii` file rarely
        means loading its voxel-to-RAS matrix, which is reached through the
        image instead. The weak score keeps these readers as a last resort that
        never outranks an image or a field.
        """
        return Confidence.WEAK

    # --- writing ------------------------------------------------------

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS matrix that this affine encodes."""
        raise NotImplementedError

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build a `nibabel` image whose affine is this transformation.

        NIfTI stores an affine only as the geometry of a data array, so
        a single-voxel placeholder volume carries it. A copy of the
        header of the metadata is the base of the new header, and its
        fields that describe the layout and the geometry are encoded
        again. The voxel-to-RAS matrix becomes both the sform and the
        qform, and each form keeps the code that the header records. The
        sform takes the qform code when its own code is zero, and both
        forms take the code 2 when the header records none. Header
        fields are copied from `like`, and `overrides` are applied last,
        but the geometry always comes from this transformation.
        """
        matrix = self._voxel_to_ras_matrix()
        scode, qcode = _xform_codes(_header(self))
        data = np.zeros((1, 1, 1), dtype="float32")
        image = _record_image(data, self._record(), matrix, overrides)
        _apply_like(image, like)
        image.header.set_sform(matrix, code=scode)
        image.header.set_qform(matrix, code=qcode)
        _apply_overrides(image, overrides)
        return image


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Return a matrix decoded from a header, made read-only.

    The matrix is a view of the header that the writer does not read back,
    so an edit in place would be lost. A new matrix is set as data instead.
    """
    matrix.flags.writeable = False
    return matrix


def _xform_codes(header: tx.Optional[nb.Nifti1Header]) -> tx.Tuple[int, int]:
    """Return the sform and qform codes to write over a header.

    Each form keeps the code that the header records, so that the codes
    of a file read and written again do not change. The sform always
    holds the matrix, so it takes the qform code when its own code is
    zero. A header that records no code, or no header, gives the code 2
    to both forms.
    """
    if header is not None:
        _, scode = header.get_sform(coded=True)
        _, qcode = header.get_qform(coded=True)
        if scode or qcode:
            return int(scode or qcode), int(qcode)
    return 2, 2


class NiftiRASToVoxel(RASToVoxel, _NiftiAffine):
    """
    Affine from RAS to voxel space, stored in the header of a NIfTI file.

    !!! note "Not a registered format"
        The header encodes the voxel-to-RAS affine, and the RAS-to-voxel affine
        is its computed inverse. The two cannot be told apart by content, since
        they share the container, the extension and the confidence, so
        registering both would make every `.nii` file ambiguous. This affine is
        reached through `NiftiVoxelToRAS.inverse()`.
    """

    @smartproperty(
        cache=True,
        unset=always_unset,
        fset=_set_affine_data,
        invalidates=_FORGET_VIEWS,
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The (3, 4) matrix of the affine, which the `matrix` view reads.

        A matrix that is set is stored in `raw` as a homogeneous matrix.
        Otherwise, the matrix is the inverse of the voxel-to-RAS affine of
        the header, which is read-only, or `None` without metadata.
        """
        if self.raw is not None:
            return _affine_to_model(self.raw)
        header = _header(self)
        if header is None:
            return None
        return _read_only(np.linalg.inv(header.get_best_affine())[:-1])

    metadata = smartproperty(
        "metadata", fset=_set_affine_metadata, invalidates=_FORGET_VIEWS
    )
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the matrix decoded from the header.
    """

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
        if isinstance(other, cls):
            # Within the format, `raw` and the metadata carry the data, so
            # the data is not read, and a proxy stays lazy in the copy.
            kwargs.setdefault("data", None)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> VoxelToRAS:
        """Return the inverse transformation, from voxel to RAS space.

        The inverse of an affine read from a header is the affine of the
        same header, rather than an inverse of its inverse.
        """
        if self.raw is None:
            return NiftiVoxelToRAS(metadata=self.metadata)
        return super().inverse(compute=compute, **kwargs).to(VoxelToRAS)

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        # NIfTI stores the voxel-to-RAS map, which is the inverse of this one.
        return _voxel_to_ras(self.inverse())


@register_format
class NiftiVoxelToRAS(VoxelToRAS, _NiftiAffine):
    """Affine from voxel to RAS space, stored in the header of a NIfTI file.

    The matrix of an affine read from a file is the best affine of the
    header, which is the sform when its code is set and the qform
    otherwise. Setting the matrix stores it in `raw`, which then takes
    precedence over the header.
    """

    HINTS = ("nifti",)

    @smartproperty(
        cache=True,
        unset=always_unset,
        fset=_set_affine_data,
        invalidates=_FORGET_VIEWS,
    )
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The (3, 4) matrix of the affine, which the `matrix` view reads.

        A matrix that is set is stored in `raw` as a homogeneous matrix.
        Otherwise, the matrix is the voxel-to-RAS affine of the header,
        which is read-only, or `None` without metadata.
        """
        if self.raw is not None:
            return _affine_to_model(self.raw)
        header = _header(self)
        if header is None:
            return None
        return _read_only(header.get_best_affine()[:-1])

    metadata = smartproperty(
        "metadata", fset=_set_affine_metadata, invalidates=_FORGET_VIEWS
    )
    """The metadata of the file, which holds its header, or `None`.

    Assigning other metadata drops the matrix decoded from the header.
    """

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
        if isinstance(other, cls):
            # Within the format, `raw` and the metadata carry the data, so
            # the data is not read, and a proxy stays lazy in the copy.
            kwargs.setdefault("data", None)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> RASToVoxel:
        """Return the inverse transformation, from RAS to voxel space.

        The inverse of an affine read from a header is the RAS-to-voxel
        affine of the same header.
        """
        if self.raw is None:
            return NiftiRASToVoxel(metadata=self.metadata)
        return super().inverse(compute=compute, **kwargs).to(RASToVoxel)

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        return _voxel_to_ras(self)
