from os import PathLike

import dask.array as da
import typing_extensions as _tx

from brainhops.datamodel.images import Image
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


class NiftiImage(Image, NiftiBasedParser):
    """
    Parse a Nifti file into an Image
    """

    @property
    def transformations(self) -> _tx.Optional[_tx.List[Transformation]]:
        """The affine matrix of the transformation."""
        if getattr(self, "_file_transformations", None) is None:
            self._file_transformations = []
            if self.header is not None:
                self._file_transformations.append(
                    Affine(matrix=self.header.get_best_affine()[:-1])
                )
        return self._file_transformations

    @transformations.setter
    def transformations(self, value: _tx.List[Transformation]) -> None:
        """Update transformations to the given file."""
        self._file_transformations = value
        super(NiftiImage, type(self)).transformations.fset(self, value)

    @property
    def data(self) -> da.Array:
        """The affine matrix of the transformation."""
        if getattr(self, "_data", None) is None:
            # TODO: use dask if dask is installed
            self._data = da.from_array(self.image.dataobj, chunks=256)
        return self._data

    @data.setter
    def data(self, value: da.Array) -> None:
        """Update data to be the given value."""
        self._data = value
