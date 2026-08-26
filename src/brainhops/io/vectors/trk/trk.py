from os import PathLike

import typing_extensions as _tx

from brainhops.datamodel.points import Points
from brainhops.datamodel.transformations import Affine, Transformation

# optionals
if _tx.TYPE_CHECKING:
    import nibabel as nb
    # typing
    _TractogramLike = _tx.Union[
        nb.streamlines.tractogram_file.TractogramFile,
        str,
        PathLike,
    ]
else:
    try:
        import nibabel as nb
        # typing
        _TractogramLike = _tx.Union[
            nb.streamlines.tractogram_file.TractogramFile,
            str,
            PathLike,
        ]
    except ImportError:
        nb = None
        # typing
        _TractogramLike = _tx.Union[
            str,
            PathLike,
        ]


class TrkPoints(Points):

    header: _tx.Optional[nb.Nifti1Header] = None
    image: _tx.Optional[nb.Nifti1Image] = None

    @property
    def header(self) -> _tx.Optional[nb.Nifti1Header]:
        """The NIfTI header from which the transformation was derived."""
        if getattr(self, "_header", None) is not None:
            return self._header
        if getattr(self, "_image", None) is not None:
            return self._image.header
        return None

    @property
    def image(self) -> _tx.Optional[nb.Nifti1Image]:
        """The NIfTI image from which the transformation was derived."""
        if getattr(self, "_image", None) is not None:
            return self._image
        return None

    _coordinateTransforms: _tx.Optional[_tx.List[Transformation]] = None
    _data: _tx.Optional[_tx.List[object]] = None

    @classmethod
    def from_(cls, other: _TractogramLike) -> _tx.Self:
        return cls.from_trk(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        return nb.streamlines.load(fileobj, lazy_load=True)

    @classmethod
    def from_trk(cls, trkobj: _TractogramLike) -> _tx.Self:
        if isinstance(trkobj,  _tx.Union[str, PathLike]):
            trkobj = cls.from_file(trkobj)
        return trkobj

    @property
    def coordinateTransforms(self) -> _tx.List[Transformation]:
        """The affine matrix of the transformation."""
        if self._coordinateTransforms is None:
            self._coordinateTransforms = []
            if self.header is not None:
                self._coordinateTransforms.append(
                    Affine(matrix=self.header.get_best_affine()[:-1]))
        return self._coordinateTransforms

    @coordinateTransforms.setter
    def coordinateTransforms(self, value: _tx.List[Transformation]) -> None:
        self._coordinateTransforms = value

    @property
    def data(self) -> _tx.List[object]:
        """The affine matrix of the transformation."""
        if self._data is None:
            self._data = [self.image]
        return self._data

    @data.setter
    def data(self, value: object) -> None:
        self._data = value
