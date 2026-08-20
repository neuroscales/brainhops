# externals
from warnings import warn

import dask.array as da
import typing_extensions as _tx

from brainhops._core.bsplines import (
    pull,
)
from brainhops.datamodel import hierarchy
from brainhops.datamodel.hierarchy import AffineTransformation
from brainhops.datamodel.transformations import is_identity

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

# Type alias for the recurring reslice/geometry parameter shape used across
# Image.reslice, Image.__call__, and their MultiImage overrides.
GeometryLike = _tx.Optional[
    _tx.Union[CartesianField, _tx.Literal["preserve"], _tx.Tuple[int, ...]]
]


class Image(DataModelBase):
    data: _tx.Optional[da.Array] = None
    transformations: _tx.Optional[_tx.List[Transformation]] = None
    transformation: _tx.Optional[Transformation] = None

    @property
    def transformation(self) -> Transformation:
        """
        Compute the composed voxel -> preferred-space transformation.

        The result is cached in `_transformation` until `transformations`
        is reassigned (see the `transformations` setter, which invalidates
        this cache).
        """

        self._transformation = getattr(self, "_transformation", None)
        self._transformations = getattr(self, "_transformations", None)
        if self._transformation is None:
            if len(self.transformations) == 0:
                self._transformation = Identity()
            else:
                self._transformation = Sequence(
                    transformations=self.transformations,
                    input=self.transformations[0].input,
                    output=self.transformations[-1].output,
                ).compute()
        return self._transformation

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        """
        Set transformation directly.
        """
        self._transformation = value
        self._transformations = [value]

    @property
    def transformations(self) -> _tx.List[Transformation]:
        """Ordered list of transformations; defaults to [Identity()] if unset."""
        self._transformations = getattr(self, "_transformations", None)
        if self._transformations is None:
            return [Identity()]
        return self._transformations

    @transformations.setter
    def transformations(self, value: _tx.List[Transformation]) -> None:
        """Update transformations and invalidate the cached composed transformation."""
        self._transformation = None
        self._transformations = value

    @property
    def geometry(self) -> _tx.Optional[CartesianField]:
        """
        Shape-only CartesianField matching the data's shape.
        """
        if self.data is None:
            return None
        return CartesianField(shape=self.data.shape)

    @geometry.setter
    def geometry(self, value: CartesianField) -> None:
        """Geometry is derived from data + transformation; it cannot be set directly."""
        raise NotImplementedError("can not set geometry")

    def reslice(self, geometry: GeometryLike = None) -> "Image":
        """
        Apply transformations to current data and return new image.

        Parameters
        ----------
        geometry : CartesianField | "preserve" | tuple[int] | None
            The shape/grid the output data should be resampled onto.
            - None: based on `self.transformations` alone (no extra grid change).
            - "preserve": keep the current data's shape.
            - tuple[int]: resample onto a grid of this shape.
            - CartesianField: resample onto this explicit grid.

        Returns
        -------
        Image
            The resliced image.
        """
        transformation = self.transformation
        if geometry is not None:
            if geometry == "preserve":
                geometry = self.geometry
            elif isinstance(geometry, tuple):
                geometry = CartesianField(shape=geometry)
            transformation = geometry @ self.transformation

        if isinstance(transformation, Sequence):
            coord_transform = transformation[1].to(CoordinatesField)
            affine_transform = transformation[0]
        else:
            if not isinstance(hierarchy.parseType(type(transformation)), hierarchy.AffineTransformation):
                coord_transform = transformation.to(CoordinatesField)
                affine_transform = Identity()
            else:
                affine_transform = transformation
                coord_transform = Identity()

        if is_identity(coord_transform):
            new_data = self.data
        else:
            new_data = pull(
                self.data,
                coord_transform.field,
                0,
                0.0,
                coeff=coord_transform.coeff,
            )
        return Image(data=new_data, transformation=affine_transform)

    def __call__(self, transform: Transformation, reslice: GeometryLike = None) -> "Image":
        """
        Append a transformation to the image without computing/resampling yet.

        Parameters
        ----------
        transform : Transformation
            The transformation to add

        reslice : CartesianField | "preserve" | tuple[int] | None
            What the shape of data should be once it gets resliced.
            If None, this is based on `self.transformations` alone.

        Returns
        -------
        Image
            The updated (not-yet-resliced) image.
        """
        transformations = [*self.transformations, transform.inverse()]
        if reslice == "preserve":
            transformations = [*transformations, self.geometry]
        elif isinstance(reslice, CartesianField):
            transformations = [*transformations, reslice]
        elif isinstance(reslice, tuple):
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
    def geometry(self) -> _tx.Optional[CartesianField]:
        return self.images[0].geometry

    @geometry.setter
    def geometry(self, value: CartesianField) -> None:
        raise NotImplementedError("can not set geometry")

    def reslice(self, geometry: GeometryLike = None) -> "MultiImage":
        """
        Reslice each image.

        Parameters
        ----------
        geometry : CartesianField | "preserve" | tuple[int] | None
            What the shape of data should be after the reslice.
            If None, this is based on `self.transformations`.

        Returns
        -------
        MultiImage
            The resliced images.
        """
        new_images = [img.reslice(geometry) for img in self.images]
        return MultiImage(images=new_images)

    def __call__(self, transform: Transformation, reslice: GeometryLike = None) -> "MultiImage":
        """
        Call each image.

        Parameters
        ----------
        transform : Transformation
            The transformation to add.

        reslice : CartesianField | "preserve" | tuple[int] | None
            What the shape of data should be once it gets resliced.
            If None, this is based on `self.transformations`.

        Returns
        -------
        MultiImage
            The updated images.
        """
        return MultiImage(images=[img(transform, reslice) for img in self.images])
