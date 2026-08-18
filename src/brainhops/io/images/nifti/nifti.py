from os import PathLike

import numpy as np
import typing_extensions as _tx

from brainhops._core.backends import get_array_backend
from brainhops._core.typing import ArrayProtocol
from brainhops._ext.struct.constants import HIDE_IF_NONE
from brainhops.datamodel.images import Image, MultiImage
from brainhops.datamodel.transformations import Affine, Transformation

# optionals
if _tx.TYPE_CHECKING:
    import nibabel as nb
else:
    try:
        import nibabel as nb
    except ImportError:
        nb = None

# typing
_NiftiLike = _tx.Union[
    nb.Nifti1Header,
    nb.Nifti1Image,
    str,
    PathLike,
    _tx.BinaryIO
]


class NiftiImage(Image):

    header: _tx.Optional[nb.Nifti1Header] = None
    image: _tx.Optional[nb.Nifti1Image] = None

    @classmethod
    def from_(cls, other: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        return cls.from_nifti(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file."""
        return cls.from_nifti(nb.load(fileobj))

    @classmethod
    def from_bytes(cls, data: bytes) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI file in bytes."""
        return cls.from_nifti(nb.Nifti1Image.from_bytes(data))

    @classmethod
    def from_nifti(cls, nifti: _NiftiLike) -> _tx.Self:
        """Create a NiftiVoxelToRAS transformation from a NIfTI image."""
        if isinstance(nifti, nb.Nifti1Header):
            raise ValueError("can not make value from only header")
        if isinstance(nifti, nb.Nifti1Image):
            return cls(image=nifti)
        return cls.from_nifti(nb.load(nifti))

    @property
    def coordinateTransforms(self) -> _tx.Optional[_tx.List[Transformation]]:
        """The affine matrix of the transformation."""
        if getattr(self, "_coordinateTransforms", None) is None:
            self._coordinateTransforms = []
            if self.header is not None:
                self._coordinateTransforms.append(
                    Affine(matrix=self.header.get_best_affine()[:-1]))
        return self._coordinateTransforms

    @coordinateTransforms.setter
    def coordinateTransforms(self, value) -> None:
        self._coordinateTransforms = value

    @property
    def data(self) -> ArrayProtocol:
        """The affine matrix of the transformation."""
        if getattr(self, "_data", None) is None:
            # TODO: use dask if dask is installed
            self._data = np.asarray(self.image.dataobj)
        return self._data

    @data.setter
    def data(self, value: ArrayProtocol) -> None:
        self._data = value
