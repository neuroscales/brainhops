# externals
from warnings import warn

import dask.array as da
import typing_extensions as _tx

from brainhops._core.bsplines import (
    pull,
)

# internals
from .base import DataModelBase
from .systems import CoordinateSystem
from .transformations import (
    CartesianField,
    CoordinatesField,
    Identity,
    Sequence,
    Transformation,
)


class Image(DataModelBase):
    data: _tx.Optional[da.Array] = None
    _transformation: _tx.Optional[CoordinateSystem] = None
    _transformations: _tx.Optional[_tx.List[Transformation]] = None

    @property
    def transformation(self) -> Transformation:
        """compute transformation if not already computed"""
        if self._transformation is None:
            if len(self.transformations) == 0:
                self._transformation = Identity()
                return self._transformation

            transformation = Sequence(
                transformations=self.transformations,
                input=self.transformations[0].input,
                output=self.transformations[-1].output,
            ).compute()
            self._transformation = transformation.compute()
        return self._transformation

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        """set transformation and transformations"""
        self._transformation = value
        self._transformations = [value]

    @property
    def transformations(self) -> _tx.List[Transformation]:
        """compute transformations, use identity if not set"""
        if self._transformations is None:
            return [Identity()]
        return self._transformations

    @transformations.setter
    def transformations(self, value: _tx.List[Transformation]):
        """if transformations is updated transformation needs to be recomputed"""
        self._transformation = None
        self._transformations = value

    @property
    def geometry(self):
        """get the CaresianField for the image data"""
        if self.data is None:
            return None
        return CartesianField(shape=self.data.shape)

    @geometry.setter
    def geometry(self, value: CartesianField):
        """geometry is not something that can be updated without updating data"""
        raise NotImplementedError("can not set geometry")

    def reslice(self, geometry: _tx.Optional[_tx.Union[CartesianField, _tx.Literal["preserve"], _tx.Tuple[int]]] = None) -> "Image":
        """
        Apply transformations to current data and return new image

        Parameters
        ----------
        geomerty: CartesianField | "preserve" | tuple[int] | None
            what the shape of data should be after the reslice.
            if none this is based on self.transformations.

        Returns
        -------
        Image
            The resliced image.
        """

        transformation = self.transformation
        if geometry is not None:
            if geometry == "preserve":
                geometry = self.geometry
            elif isinstance(geometry, _tx.Tuple):
                geometry = CartesianField(shape=geometry)
            transformation = (geometry @ self.transformation)
        if isinstance(transformation, Sequence):
            coord_transform = transformation[1].to(CoordinatesField)
            affine_transform = transformation[0]
        else:
            coord_transform = transformation.to(CoordinatesField)
            affine_transform = Identity()
        new_data = pull(self.data, coord_transform.field,
                        0, 0.0, coeff=coord_transform.coeff)
        return Image(data=new_data, transformation=affine_transform)

    def __call__(self, transform: Transformation, reslice: _tx.Optional[_tx.Union[CartesianField, _tx.Literal["preserve"], _tx.Tuple[int]]] = None) -> "Image":
        """
        Apply transformation to Image but does not compute.

        Parameters
        ----------
        transform: Transformation
            The transformation to add.

        reslice: CartesianField | "preserve" | tuple[int] | None
            what the shape of data should be once it gets resliced.
            if none this is based on self.transformations.

        Returns
        -------
        Image
            The updated image.
        """

        transformations = [*self.transformations, transform.inverse()]
        if reslice == "preserve":
            transformations = [*transformations, self.geometry]
        elif isinstance(reslice, CartesianField):
            transformations = [*transformations, reslice]
        elif isinstance(reslice, _tx.Tuple):
            transformations = [*transformations, CartesianField(shape=reslice)]
        return Image(data=self.data, transformations=transformations)


class MultiImage(Image):
    images: _tx.Optional[_tx.List[Image]] = None

    @property
    def data(self) -> da.Array:
        return self.images[0].data

    @data.setter
    def data(self, value: da.Array) -> None:
        warn("setting data for MultiImages is not recommended as it only effects the first layer")
        if self.images is not None and len(self.images) > 0:
            self.images[0].data = value

    @property
    def transformations(self) -> _tx.List[Transformation]:
        return self.images[0].transformations

    @transformations.setter
    def transformations(self, value: _tx.List[Transformation]) -> None:
        warn("setting transformations for MultiImages is not recommended as it only effects the first layer")
        if self.images is not None and len(self.images) > 0:
            self.images[0].transformations = value

    @property
    def transformation(self) -> Transformation:
        return self.images[0].transformation

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        warn("setting transformation for MultiImages is not recommended as it only effects the first layer")
        if self.images is not None and len(self.images) > 0:
            self.images[0].transformation = value

    @property
    def geometry(self) -> CartesianField:
        return self.images[0].geometry

    @geometry.setter
    def geometry(self, value: CartesianField) -> None:
        raise NotImplementedError("can not set geometry")

    def reslice(self, geometry: _tx.Optional[_tx.Union[CartesianField, _tx.Literal["preserve"], _tx.Tuple[int]]] = None) -> "MultiImage":
        """
        reslice each image

        Parameters
        ----------
        geomerty: CartesianField | "preserve" | tuple[int] | None
            what the shape of data should be after the reslice.
            if none this is based on self.transformations.

        Returns
        -------
        MultiImage
            The resliced images.
        """
        new_images = [img.reslice(geometry) for img in self.images]
        return MultiImage(images=new_images)

    def __call__(self, transform: Transformation, reslice: _tx.Optional[_tx.Union[CartesianField, _tx.Literal["preserve"], _tx.Tuple[int]]] = None) -> "MultiImage":
        """
        call each image.

        Parameters
        ----------
        transform: Transformation
            The transformation to add.

        reslice: CartesianField | "preserve" | tuple[int] | None
            what the shape of data should be once it gets resliced.
            if none this is based on self.transformations.

        Returns
        -------
        MultiImage
            The updated images.
        """

        return MultiImage(images=[img(transform, reslice) for img in self.images])
