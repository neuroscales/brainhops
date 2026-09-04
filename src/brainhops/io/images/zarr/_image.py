import typing_extensions as _tx

from brainhops._core.backends import get_ndimage_backend
from brainhops.datamodel.images import Image, MultiImage
from brainhops.io.base.omezarr import OmeZarrParser


class OmeZarrImage(OmeZarrParser, MultiImage):
    @property
    def images(self) -> _tx.List[Image]:
        """
        Convert each layer into an image for the MultiImage.
        Store it in the cached variable _images

        Returns
        -------
        list[Image]
        """
        if getattr(self, "_images", None) is None:
            if self.group is None or self._multiscale is None:
                self._images = None
            else:
                self._images = [
                    Image(
                        data=get_ndimage_backend().from_array(
                            self.group[ds["path"]]
                        ),
                        transformations=self._transform_from_multiscale(
                            self._multiscale, i
                        ),
                    )
                    for i, ds in enumerate(self._multiscale["datasets"])
                ]
        return self._images

    @images.setter
    def images(self, value: _tx.Optional[_tx.List[Image]]) -> None:
        """
        set images into the cache

        Parameters
        ----------
        value: list[Image]
            the list of images that should be set
        """
        self._images = value
