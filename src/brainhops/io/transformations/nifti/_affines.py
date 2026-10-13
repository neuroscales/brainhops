"""Affines between voxel and RAS space, stored in the header of a NIfTI
file."""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import (
    InvalidatorInAttribute,
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
    _written_nifti,
)
from brainhops.io.transformations.base import AffineTransformationFormat
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

# this format
from ._base import (
    NiftiBasedTransformation,
    _consolidated_metadata,
    _nibabel_parts,
)
from ._views import _affine_to_model

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


class _NiftiAffine(AffineTransformationFormat, NiftiBasedTransformation):
    """Shared reading, scoring and writing of the affines of a header.

    An affine read from a file keeps its matrix in the header of its
    metadata and has no `raw` array, because the array of the file is only
    a placeholder. A matrix that is assigned is pending: the data model
    stores it, and the affine reads it first, until [`consolidate`][]
    encodes it into a new header. An affine built with a homogeneous
    matrix in `raw` reads that matrix before the header.
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

    def consolidate(
        self, *, inplace: bool = False, like: tx.Any = None, **overrides
    ) -> tx.Self:
        """Encode the pending matrix into a new header.

        NIfTI stores an affine only as the geometry of a data array, so
        the header of a single-voxel placeholder volume carries it. A copy
        of the header of the metadata is the base of the new header, and
        its fields that describe the layout and the geometry are encoded
        again. The voxel-to-RAS matrix becomes both the sform and the
        qform, and each form keeps the code that the header records. The
        sform takes the qform code when its own code is zero, and both
        forms take the code 2 when the header records none. Header fields
        are copied from `like`, and `overrides` are applied last, but the
        geometry always comes from this transformation.

        Afterwards the matrix is decoded from the new header, with the
        precision of the file, which stores it in single precision, and
        the affine has no `raw` array. An affine that has a header and no
        pending matrix, consolidated without options, is left as it is.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this affine rather than a copy.
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        Self
            The consolidated copy, or this affine with `inplace=True`.
        """
        if (
            self._data is None
            and like is None
            and not overrides
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            return self if inplace else replace(self)
        image = _affine_encoded(self, like, overrides)
        record = NiftiRaw(header=image.header)
        metadata = _consolidated_metadata(self, record)
        if not inplace:
            return replace(
                self,
                raw=None,
                metadata=metadata,
                data=None,
            )
        self.raw = None
        self.metadata = metadata
        self.data = None
        return self

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image whose affine is this transformation.

        The affine is consolidated with the options, as by
        [`consolidate`][], into a copy, and the header of that copy is
        written over a single-voxel placeholder volume. The affine itself
        is not changed.

        Parameters
        ----------
        like : path, nibabel image or header, or NIfTI object, optional
            Template whose header fields that do not encode the
            transformation are copied.
        **overrides : Any
            Header fields that take precedence over the derived values and
            `like`.

        Returns
        -------
        nibabel.Nifti1Image or nibabel.Nifti2Image
            The encoded image.
        """
        consolidated = self.consolidate(like=like, **overrides)
        placeholder = np.zeros((1, 1, 1), dtype="float32")
        return _written_nifti(placeholder, consolidated.metadata.raw)


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

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The (3, 4) matrix of the affine, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] or a write encodes it. Otherwise, the
        matrix is decoded from a homogeneous matrix in `raw`, or else it is
        the inverse of the voxel-to-RAS affine of the header, which is
        read-only, or `None` without metadata.
        """
        if self.raw is not None:
            return _affine_to_model(self.raw)
        header = _header(self)
        if header is None:
            return None
        return _read_only(np.linalg.inv(header.get_best_affine())[:-1])

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds its header, or `None`.

    Without an assigned matrix, the matrix is decoded from the header, so
    assigning other metadata drops the matrix decoded from the previous
    header.
    """

    # --- conversions --------------------------------------------------
    # `TransformationFormat.from_any` converts another transformation, as
    # `t.to(cls)` converts it. The data model comes before that class in
    # the method resolution order, and its `from_instance` would copy the
    # map of an affine first, so `from_instance` converts here, with the
    # same helpers.

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an instance, or convert a transformation to this format.

        A transformation that is not of this format is converted exactly
        as `other.to(cls)` converts it. An instance of this format is
        copied with its metadata and its stored matrix, which is `None`
        unless a matrix was assigned, so the header is not decoded. Any
        other instance is copied as
        [`Format.from_instance`][brainhops.io.base.Format.from_instance]
        copies it.

        Parameters
        ----------
        other : object
            The instance to copy or convert.
        *args : Any
            Constructor arguments. A transformation that is converted
            takes keyword options only.
        **kwargs : Any
            Options of the conversion, or field values of the copy.

        Returns
        -------
        Self
            The new object.

        Raises
        ------
        TypeError
            If positional arguments accompany a transformation that is
            converted.
        ConversionError
            If the format cannot hold the transformation.
        """
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        if isinstance(other, cls):
            # The data model copies the decoded view of another instance,
            # so the stored value is passed instead: `None` keeps a proxy
            # lazy, and an assigned array stays pending in the copy.
            kwargs.setdefault("data", other._data)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> VoxelToRAS:
        """Return the inverse transformation, from voxel to RAS space.

        The inverse of an affine read from a header is the affine of the
        same header, rather than an inverse of its inverse.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the inverse now rather than lazily. It is
            ignored for an affine read from a header.
        **kwargs : Any
            Arguments passed to the inverse of the data model, which are
            ignored for an affine read from a header.

        Returns
        -------
        VoxelToRAS
            The inverse transformation.
        """
        if self._data is None and self.raw is None:
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

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The (3, 4) matrix of the affine, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] or a write encodes it. Otherwise, the
        matrix is decoded from a homogeneous matrix in `raw`, or else it is
        the voxel-to-RAS affine of the header, which is read-only, or
        `None` without metadata.
        """
        if self.raw is not None:
            return _affine_to_model(self.raw)
        header = _header(self)
        if header is None:
            return None
        return _read_only(header.get_best_affine()[:-1])

    _metadata: NoRepr[tx.Optional[NiftiMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds its header, or `None`.

    Without an assigned matrix, the matrix is decoded from the header, so
    assigning other metadata drops the matrix decoded from the previous
    header.
    """

    # --- conversions --------------------------------------------------
    # `TransformationFormat.from_any` converts another transformation, as
    # `t.to(cls)` converts it. The data model comes before that class in
    # the method resolution order, and its `from_instance` would copy the
    # map of an affine first, so `from_instance` converts here, with the
    # same helpers.

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an instance, or convert a transformation to this format.

        A transformation that is not of this format is converted exactly
        as `other.to(cls)` converts it. An instance of this format is
        copied with its metadata and its stored matrix, which is `None`
        unless a matrix was assigned, so the header is not decoded. Any
        other instance is copied as
        [`Format.from_instance`][brainhops.io.base.Format.from_instance]
        copies it.

        Parameters
        ----------
        other : object
            The instance to copy or convert.
        *args : Any
            Constructor arguments. A transformation that is converted
            takes keyword options only.
        **kwargs : Any
            Options of the conversion, or field values of the copy.

        Returns
        -------
        Self
            The new object.

        Raises
        ------
        TypeError
            If positional arguments accompany a transformation that is
            converted.
        ConversionError
            If the format cannot hold the transformation.
        """
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        if isinstance(other, cls):
            # The data model copies the decoded view of another instance,
            # so the stored value is passed instead: `None` keeps a proxy
            # lazy, and an assigned array stays pending in the copy.
            kwargs.setdefault("data", other._data)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> RASToVoxel:
        """Return the inverse transformation, from RAS to voxel space.

        The inverse of an affine read from a header is the RAS-to-voxel
        affine of the same header.

        Parameters
        ----------
        compute : bool, default=False
            Whether to compute the inverse now rather than lazily. It is
            ignored for an affine read from a header.
        **kwargs : Any
            Arguments passed to the inverse of the data model, which are
            ignored for an affine read from a header.

        Returns
        -------
        RASToVoxel
            The inverse transformation.
        """
        if self._data is None and self.raw is None:
            return NiftiRASToVoxel(metadata=self.metadata)
        return super().inverse(compute=compute, **kwargs).to(RASToVoxel)

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        return _voxel_to_ras(self)


def _affine_encoded(
    xform: _NiftiAffine, like: tx.Any, overrides: tx.Mapping[str, tx.Any]
) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
    """Encode an affine as the header of a single-voxel placeholder volume.

    The header is built over a copy of the record, as
    [`_NiftiAffine.consolidate`][] describes.
    """
    matrix = xform._voxel_to_ras_matrix()
    scode, qcode = _xform_codes(_header(xform))
    data = np.zeros((1, 1, 1), dtype="float32")
    image = _record_image(data, xform._record(), matrix, overrides)
    _apply_like(image, like)
    image.header.set_sform(matrix, code=scode)
    image.header.set_qform(matrix, code=qcode)
    _apply_overrides(image, overrides)
    return image
