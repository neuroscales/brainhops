# dependencies
import dask.array as da
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
from brainhops._core.bsplines import pull

# internals
from .base import DataModelBase
from .geometry import Geometry
from .transformations import CartesianField, Identity, Transformation


class Image(DataModelBase):
    """Base class for all images."""


class SingleScaleImage(Image):
    """Base class for all single-resolution images."""

    data: tx.Annotated[
        ArrayProtocol,
        tx.Doc(
            "The image data. Must be an array-like object that supports "
            "the array protocol (e.g. numpy, cupy or dask array)."
        ),
    ]

    transformations: tx.Annotated[
        tx.List[Transformation],
        tx.Doc(
            "A list of transformations from voxel space (i.e., in terms"
            "of data's axes) to different world spaces. The last "
            "transformation in the list is the preferred one."
        ),
    ]

    @property
    def transformation(self) -> Transformation:
        """
        The preferred transformation.

        It is always the last transformation in the list.
        Changing it appends the new transformation to the list (or
        reorders the list if the value is an integer or a string).
        """
        if self.transformations:
            return self.transformations[-1]
        return Identity()

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        if isinstance(value, int):
            value = self.transformations.pop(value)
        elif isinstance(value, str):
            for i, x in enumerate(self.transformations):
                if getattr(x.output, "name", None) == value:
                    value = self.transformations.pop(i)
                    break
        self.transformations.append(value)

    @property
    def geometry(self) -> Geometry:
        """
        A transformation that is the composition of the preferred
        voxel-to-world transformation and the cartesian field corresponding
        to the image's shape.

        This transformation can be used to reslice any image onto the same
        grid as this image.
        """
        return Geometry(
            (
                CartesianField(
                    shape=self.data.shape,
                    input=self.transformation.input,
                    output=self.transformation.input,
                ),
                self.transformation,
            )
        )

    def reslice(
        self,
        geometry: tx.Union[tx.Self, Geometry, Transformation],
        order: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
    ) -> tx.Self:
        """
        Apply transformations to current data and return new image.

        Parameters
        ----------
        geometry : Image | Geometry | Transformation
            Geometry of the output image.

            The geometry is a voxel-to-world transformation that defines
            the grid onto which the image will be resliced.

            If it is a `Geometry`, then it also defines the shape of the
            output image. Otherwise, the current shape of the image is used.
        order : {0..5}
            The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
        bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
            The boundary condition. If a string, one of:
            - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
            - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
            - 'mirror': mirror at edge        (d c b | a b c d | c b a)
            - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
            - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
            If a float, the constant value to use beyond the edge.
        coeff : bool
            If True, the input image is assumed to already contain spline
            coefficients. If False, the input image is prefiltered
            before interpolation.

        Returns
        -------
        Image
            The resliced image.
        """
        opt = dict(order=order, bound=bound, coeff=coeff)

        # Guess geometry of output image
        if isinstance(geometry, Image):
            geometry = geometry.geometry
        if not isinstance(geometry, Geometry):
            geometry = Geometry((self.geometry.grid, geometry))

        # Compute voxel-to-voxel transformation and apply it to the data
        transformation = self.transformation.inverse() @ geometry
        transformation = transformation.compute()
        new_data = pull(self.data, transformation.field, **opt)
        return Image(data=new_data, transformation=geometry.transformation)

    def __call__(self, transform: Transformation) -> "Image":
        """
        Apply a transformation to the image, but does not compute.

        Parameters
        ----------
        transform: Transformation
            The transformation to apply.

            The **output** space of this transformation should match
            (or be compatible with) the **output** space of the preferred
            transformation. That is, the new "voxel-to-world" transformation
            is defined as `self.transformation @ transform.inverse()`.


        Returns
        -------
        Image
            The updated (not-yet-resliced) image.
        """
        transform = transform.inverse() @ self.transformation
        if self.transformations:
            transformations = self.transformations.copy()
            transformations.append(transform)
        else:
            transformations = [transform]
        return Image(data=self.data, transformations=transformations)

    def __getitem__(
        self, index: tx.Tuple[tx.Union[int, slice, None], ...]
    ) -> "Image":
        """
        Index into the image data while preserving the geometry of the image.
        """
        data = self.data[index]
        transformations = [
            Geometry((self.grid, xform))[index].transformation
            for xform in self.transformations
        ]
        return Image(data=data, transformations=transformations)


class MultiScaleImage(Image):
    """Base class for all multi-scale images."""

    images: tx.Annotated[
        tx.List[SingleScaleImage],
        tx.Doc(
            "Each image in the multi-resolution pyramid, ordered from "
            "highest to lowest resolution."
        ),
    ] = ()

    transformations: tx.Annotated[
        tx.List[Transformation],
        tx.Doc(
            "A list of transformations from each level's preferred space "
            "to different world spaces. The last transformation in the list "
            "is the preferred one."
        ),
    ] = []

    def to_singlescale(self, index: int = 0) -> SingleScaleImage:
        """
        Return one of the levels as a single-resolution image.
        """
        return self.images[index](self.transformation.inverse())

    @property
    def nscales(self) -> int:
        """
        Return the number of scales in the multi-resolution pyramid.
        """
        return len(self.images)

    @property
    def scales(self) -> tx.Iterator[SingleScaleImage]:
        """
        Yield all levels as single-resolution images.
        """
        for i in range(len(self.images)):
            yield self.to_singlescale(i)

    @property
    def transformation(self) -> Transformation:
        """
        The preferred transformation.

        It is always the last transformation in the list.
        Changing it appends the new transformation to the list (or
        reorders the list if the value is an integer or a string).
        """
        if self.transformations:
            return self.transformations[-1]
        return Identity()

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        if isinstance(value, int):
            value = self.transformations.pop(value)
        elif isinstance(value, str):
            for i, x in enumerate(self.transformations):
                if getattr(x.output, "name", None) == value:
                    value = self.transformations.pop(i)
                    break
        self.transformations.append(value)

    @property
    def data(self) -> da.Array:
        return self.images[0].data

    @property
    def geometry(self) -> Geometry:
        """
        The geometry of the highest-resolution image in the pyramid.

        A transformation that is the composition of the preferred
        voxel-to-world transformation and the cartesian field corresponding
        to the image's shape.

        This transformation can be used to reslice any image onto the same
        grid as this image.
        """
        return Geometry(
            (
                CartesianField(
                    shape=self.data.shape,
                    input=self.transformation.input,
                    output=self.transformation.input,
                ),
                self.transformation @ self.images[0].transformation,
            )
        )

    def reslice(
        self,
        geometry: tx.Union[Image, Geometry, Transformation],
        order: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
    ) -> tx.Self:
        """
        Apply transformations to current data and return new image

        Parameters
        ----------
        geometry : Image | Geometry | Transformation
            Geometry of the highest-resolution level of the output image.

            The geometry is a voxel-to-world transformation that defines
            the grid onto which the image will be resliced.

            If it is a `Geometry`, then it also defines the shape of the
            output image. Otherwise, the current shape of the image is used.
        intrinsic : Geometry | Transformation | None
            An optional transformation that defines the intrinsic geometry
            of the highest-resolution image in the output pyramid.
            If provided, it is used to compute the geometry of each
            level in the output pyramid. If not provided, this function
            returns a single-scale image instead.
        order : {0..5}
            The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
        bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
            The boundary condition. If a string, one of:
            - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
            - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
            - 'mirror': mirror at edge        (d c b | a b c d | c b a)
            - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
            - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
            If a float, the constant value to use beyond the edge.
        coeff : bool
            If True, the input image is assumed to already contain spline
            coefficients. If False, the input image is prefiltered
            before interpolation.

        Returns
        -------
        SingleScaleImage
            The resliced image.
        """
        opt = dict(order=order, bound=bound, coeff=coeff)
        return self.to_singlescale().reslice(geometry, **opt)

        # TODO:
        #   * find level closest to the output geometry (in terms of
        #     resolution) for computational efficiency.
        #   * decide on an API that triggers the whole pyramid to be
        #     resliced, not just the highest-resolution level.
        #     It requires a way to specify the intrinsic geometry of the
        #     output pyramid. Simplest (intermediate) step is to accept
        #     a MultiScaleImage as the geometry argument.

    def __call__(self, transform: Transformation) -> tx.Self:
        """
        Apply a transformation to the multi-scale image.

        Parameters
        ----------
        transform: Transformation
            The transformation to apply.

            The **output** space of this transformation should match
            (or be compatible with) the **output** space of the preferred
            transformation. That is, the new "intrinsic-to-world"
            transformation is defined as
            `self.transformation @ transform.inverse()`.

        Returns
        -------
        MultiScaleImage
            The transformed image.
        """
        transform = transform.inverse() @ self.transformation
        if self.transformations:
            transformations = self.transformations.copy()
            transformations.append(transform)
        else:
            transformations = [transform]
        return MultiScaleImage(
            images=self.images, transformations=transformations
        )
