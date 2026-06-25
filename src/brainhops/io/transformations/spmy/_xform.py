# stdlib
from os import PathLike

# externals
import typing_extensions as _tx
import numpy as np
import dask.array as da

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

# typing + optionals
if _tx.TYPE_CHECKING:
    import nibabel as nib
else:
    try:
        import nibabel as nib
    except ImportError:
        nib = None


_NiftiLike = _tx.Union[
    nib.Nifti1Header,
    nib.Nifti1Image,
    str,
    PathLike,
    _tx.BinaryIO
]


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel space to RAS space."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class RASToVoxel(_xforms.Affine):
    """Affine transformation from RAS space to voxel space."""

    input: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()


class NiftiBasedTransformation(_xforms.Transformation):

    header: _tx.Optional[nib.Nifti1Header] = None
    image: _tx.Optional[nib.Nifti1Image] = None

    @property
    def header(self) -> _tx.Optional[nib.Nifti1Header]:
        """The NIfTI header from which the transformation was derived."""
        if getattr(self, "_header", None) is not None:
            return self._header
        if getattr(self, "_image", None) is not None:
            return self._image.header
        return None

    @property
    def image(self) -> _tx.Optional[nib.Nifti1Image]:
        """The NIfTI image from which the transformation was derived."""
        if getattr(self, "_image", None) is not None:
            return self._image
        return None

    @classmethod
    def from_(cls, other: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        return cls.from_nifti(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file."""
        return cls.from_nifti(nib.load(fileobj))

    @classmethod
    def from_bytes(cls, data: bytes) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file in bytes."""
        return cls.from_nifti(nib.Nifti1Image.from_bytes(data))

    @classmethod
    def from_nifti(cls, nifti: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        if isinstance(nifti, nib.Nifti1Header):
            return cls(header=nifti)
        if isinstance(nifti, nib.Nifti1Image):
            return cls(image=nifti)
        return cls.from_nifti(nib.load(nifti))


class NiftiVoxelToRAS(VoxelToRAS, NiftiBasedTransformation):
    """
    Affine transformation from voxel space to RAS space, derived from a
    NIfTI header.
    """

    @property
    def matrix(self) -> _tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if self.header is not None:
            return self.header.get_best_affine()[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray):
        self._matrix = value

    def inverse(self) -> RASToVoxel:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiRASToVoxel(image=self.image, header=self.header)
        return super().inverse().to(RASToVoxel)


class NiftiRASToVoxel(RASToVoxel, NiftiBasedTransformation):
    """
    Affine transformation from RAS space to voxel space, derived from a
    NIfTI header.
    """

    @property
    def matrix(self) -> _tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if self.header is not None:
            return np.linalg.inverse(self.header.get_best_affine())[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray):
        self._matrix = value

    def inverse(self) -> VoxelToRAS:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiVoxelToRAS(image=self.image, header=self.header)
        return super().inverse().to(VoxelToRAS)


class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    @property
    def field(self) -> _tx.Optional[np.ndarray]:
        """The field of RAS coordinates."""
        if self.image is not None:
            return da.from_array(self.image.dataobj,
                                 fancy=False,
                                 name=self.image)
        return None


class SPMCoordinatesField(_xforms.Sequence, NiftiBasedTransformation):
    """Field of RAS coordinates.

    Transformations of this type are generated by the SPM software, and
    their filenames are often prefixed with `y_` or `iy_`.
    """

    @property
    def transformations(self) -> _tx.Tuple[
        _tx.Optional[RASToVoxel],
        _tx.Optional[RASCoordinatesField],
    ]:
        """The transformations that make up the sequence."""
        _transformations = getattr(self, "_transformations", None)
        if _transformations is not None:
            return _transformations
        return (
            NiftiRASToVoxel(image=self.image, header=self.header),
            NiftiRASCoordinatesField(image=self.image, header=self.header),
        )

    @transformations.setter
    def transformations(self, value: _tx.Tuple[
        _tx.Optional[RASToVoxel],
        _tx.Optional[RASCoordinatesField],
    ]) -> None:
        self._transformations = tuple(value)

    @property
    def ras2voxel(self) -> _tx.Optional[RASToVoxel]:
        """The RAS-to-voxel transformation."""
        xform = self.transformations[0]
        if xform is None:
            xform = NiftiRASToVoxel(image=self.image, header=self.header)
        return xform

    @property
    def rasfield(self) -> _tx.Optional[RASCoordinatesField]:
        """The field of RAS coordinates."""
        xform = self.transformations[1]
        if xform is None:
            xform = NiftiRASCoordinatesField(image=self.image, header=self.header)
        return xform

    @ras2voxel.setter
    def ras2voxel(self, value: RASToVoxel):
        self._transformations = (value, self.transformations[1])

    @rasfield.setter
    def rasfield(self, value: RASCoordinatesField):
        self._transformations = (self.transformations[0], value)
