from os import PathLike

import dask.array as da
import numpy as np
import typing_extensions as _tx

from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.base.nifti import NiftiBasedParser

# optionals
if _tx.TYPE_CHECKING:
    import nibabel as nb

    # typing
    _NiftiLike = _tx.Union[
        nb.Nifti1Header, nb.Nifti1Image, str, PathLike, _tx.BinaryIO
    ]
else:
    try:
        import nibabel as nb

        _NiftiLike = _tx.Union[
            nb.Nifti1Header, nb.Nifti1Image, str, PathLike, _tx.BinaryIO
        ]
    except ImportError:
        nb = None
        _NiftiLike = _tx.Any


class NiftiImage(SingleScaleImage, NiftiBasedParser):
    """
    Parse a Nifti file into an Image
    """

    @property
    def transformations(self) -> _tx.List[Transformation]:
        """The affine matrix of the transformation."""
        if getattr(self, "_transformations", None) is None:
            self._transformations = []
            if self.header is not None:
                self._transformations.append(
                    Affine(matrix=self.header.get_best_affine()[:-1])
                )
        return self._transformations

    @transformations.setter
    def transformations(self, value: _tx.Optional[_tx.List[Transformation]]) -> None:
        """Update transformations to the given file."""
        self._transformations = value

    @property
    def data(self) -> ArrayProtocol:
        """The affine matrix of the transformation."""
        if getattr(self, "_data", None) is None:
            # TODO: This feels janky. I am unsure how I should wrap nifit
            # files so its lazily loaded in chunks
            self._data = da.from_array(self.image.dataobj, chunks=256)
        return self._data

    @data.setter
    def data(self, value: _tx.Optional[ArrayProtocol]) -> None:
        """Update data to be the given value."""
        self._data = value
