import typing_extensions as _tx

from brainhops._core.backends import get_array_backend
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base.omezarr import OmeZarrParser


class OmeZarrImage(OmeZarrParser, MultiScaleImage):
    @property
    def images(self) -> _tx.List[SingleScaleImage]:
        """
        Convert each layer into an image for the MultiScaleImage.
        Store it in the cached variable _images

        Returns
        -------
        list[SingleScaleImage]
        """
        if not getattr(self, "_images", None):
            if self.group is None or self._metadata is None:
                self._images = None
            else:
                self._images = [
                    SingleScaleImage(
                        data=get_array_backend().from_array(
                            self.group[ds["path"]]
                        ),
                        transformations=self._transform_from_metadata(
                            self._metadata, i
                        ),
                    )
                    for i, ds in enumerate(self._metadata["datasets"])
                ]
        return self._images

    @images.setter
    def images(self, value: _tx.Optional[_tx.List[SingleScaleImage]]) -> None:
        """
        set images into the cache

        Parameters
        ----------
        value: list[SingleScaleImage]
            the list of images that should be set
        """
        self._images = value

    @property
    def transformations(self) -> _tx.List[Transformation]:
        if not self._transformations:
            self._transformations = self._transform_from_metadata(
                self._metadata,
            )
        return self._transformations

    @transformations.setter
    def transformations(self, value: _tx.List[Transformation]) -> None:
        self._transformations = value
