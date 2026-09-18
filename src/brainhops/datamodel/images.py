"""Single-resolution and multi-resolution images, and how they are resliced
onto a new geometry."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike

# core
from brainhops._core.affines import axis_scales
from brainhops._core.typing import ArrayProtocol

# internals
from ._transformations.multiscale import (
    _as_affine_ignoring_fields,
    _at_resolution,
    _nearest_resolution_index,
)
from .base import DataModelBase
from .geometry import Geometry
from .transformations import (
    CartesianField,
    Identity,
    Transformation,
)


class Image(DataModelBase):
    """Base class for all images."""

    # --- array API ----------------------------------------------------

    def __array__(self, dtype: tx.Optional[DTypeLike] = None) -> np.ndarray:
        """Return the image data as an array."""
        return np.asarray(self.data, dtype=dtype)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the image data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the image data."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the image data."""
        return self.data.dtype

    @property
    def grid(self) -> CartesianField:
        """
        The Cartesian field that defines the sampling grid of the image.

        This is the grid of the image's geometry.
        """
        return self.geometry.grid


class SingleScaleImage(Image):
    """Base class for all single-resolution images."""

    data: tx.Annotated[
        tx.Optional[ArrayProtocol],
        tx.Doc(
            """
            The image data. Must be an array-like object that supports
            the array protocol (e.g. numpy, cupy or dask array).

            It defaults to `None` so that subclasses can derive it
            lazily -- a format reader typically holds a handle to the
            file and materializes the array on first access, and cannot
            supply it at construction time. This mirrors
            `Affine.matrix`, which is optional for the same reason.
            """
        ),
    ] = None

    transformations: tx.Annotated[
        tx.List[Transformation],
        tx.Doc(
            "A list of transformations from voxel space (i.e., in terms"
            "of data's axes) to different world spaces. The last "
            "transformation in the list is the preferred one."
        ),
    ] = ()

    @property
    def transformation(self) -> Transformation:
        """
        The preferred transformation.

        It is always the last transformation in the list.

        Assigning a transformation appends it as the new preferred
        transformation. Assigning an integer or a string selects an
        existing transformation by position or by output-space name and
        moves it to the end. Assigning a transformation that is already
        in the list moves it to the end instead of adding a copy.

        A transformation is recognized as already present by identity. A
        distinct transformation that merely compares equal to one in the
        list is appended as a new preferred transformation.
        """
        if self.transformations:
            return self.transformations[-1]
        return Identity()

    @transformation.setter
    def transformation(
        self, value: tx.Union[Transformation, int, str]
    ) -> None:
        transformations = list(self.transformations)
        if isinstance(value, int):
            value = transformations.pop(value)
        elif isinstance(value, str):
            for i, x in enumerate(transformations):
                if getattr(x.output, "name", None) == value:
                    value = transformations.pop(i)
                    break
            else:
                raise KeyError(
                    f"no transformation with output space named {value!r}"
                )
        else:
            for i, x in enumerate(transformations):
                if x is value:
                    del transformations[i]
                    break
        transformations.append(value)
        self.transformations = transformations

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

    # --- methods ------------------------------------------------------

    def reslice(
        self,
        geometry: tx.Optional[
            tx.Union[tx.Self, Geometry, Transformation]
        ] = None,
        order: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
    ) -> tx.Self:
        """
        Apply transformations to current data and return new image.

        Parameters
        ----------
        geometry : Image | Geometry | Transformation, optional
            Geometry of the output image.

            The geometry is a voxel-to-world transformation that defines
            the grid onto which the image will be resliced.

            If it is a `Geometry`, then it also defines the shape of the
            output image. Otherwise, the current shape of the image is used.

            If it is `None`, the image is resampled onto its own grid.
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
        if geometry is None:
            geometry = self.geometry
        if isinstance(geometry, Image):
            geometry = geometry.geometry
        if not isinstance(geometry, Geometry):
            geometry = Geometry((self.geometry.grid, geometry))

        # When the preferred transformation carries a multiscale field,
        # select the level whose resolution matches the output grid. A
        # transformation without such a field is returned unchanged, so
        # its finest level is used.
        preferred = _at_resolution(
            self.transformation, geometry.transformation
        )

        # Compute voxel-to-voxel transformation and apply it to the data.
        # The transformation is factored into independent per-axis groups,
        # so an axis that is only rescaled, flipped, or permuted is handled
        # cheaply and only the coupled group keeps the N-dimensional pull.
        # The factoring is imported lazily to avoid an import cycle.
        from ._transformations.separable import pull_separable

        transformation = (
            preferred.inverse() @ geometry.transformation @ geometry.grid
        )
        new_data = pull_separable(self.data, transformation, **opt)
        return SingleScaleImage(
            data=new_data, transformations=[geometry.transformation]
        )

    def __call__(self, transform: Transformation) -> "SingleScaleImage":
        """
        Apply a transformation to the image, but do not compute.

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
            transformations = list(self.transformations)
            transformations.append(transform)
        else:
            transformations = [transform]
        return SingleScaleImage(
            data=self.data, transformations=transformations
        )

    def __getitem__(
        self, index: tx.Tuple[tx.Union[int, slice, None], ...]
    ) -> "SingleScaleImage":
        """
        Index into the image data while preserving the geometry of the image.
        """
        data = self.data[index]
        grid = self.grid
        transformations = [
            Geometry((grid, xform))[index].transformation
            for xform in self.transformations
        ]
        return SingleScaleImage(data=data, transformations=transformations)


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
    ] = ()

    @property
    def data(self) -> ArrayProtocol:
        """The data of the highest-resolution level of the pyramid."""
        return self.images[0].data

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

        Assigning a transformation appends it as the new preferred
        transformation. Assigning an integer or a string selects an
        existing transformation by position or by output-space name and
        moves it to the end. Assigning a transformation that is already
        in the list moves it to the end instead of adding a copy.

        A transformation is recognized as already present by identity. A
        distinct transformation that merely compares equal to one in the
        list is appended as a new preferred transformation.
        """
        if self.transformations:
            return self.transformations[-1]
        return Identity()

    @transformation.setter
    def transformation(
        self, value: tx.Union[Transformation, int, str]
    ) -> None:
        transformations = list(self.transformations)
        if isinstance(value, int):
            value = transformations.pop(value)
        elif isinstance(value, str):
            for i, x in enumerate(transformations):
                if getattr(x.output, "name", None) == value:
                    value = transformations.pop(i)
                    break
            else:
                raise KeyError(
                    f"no transformation with output space named {value!r}"
                )
        else:
            for i, x in enumerate(transformations):
                if x is value:
                    del transformations[i]
                    break
        transformations.append(value)
        self.transformations = transformations

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
                self.images[0].geometry.grid,
                self.transformation @ self.images[0].transformation,
            )
        )

    def reslice(
        self,
        geometry: tx.Optional[
            tx.Union[Image, Geometry, Transformation]
        ] = None,
        order: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
    ) -> tx.Self:
        """
        Apply transformations to current data and return new image

        Parameters
        ----------
        geometry : Image | Geometry | Transformation, optional
            Geometry of the highest-resolution level of the output image.

            The geometry is a voxel-to-world transformation that defines
            the grid onto which the image will be resliced.

            If it is a `Geometry`, then it also defines the shape of the
            output image. Otherwise, the current shape of the image is used.

            If it is `None`, the image is resampled onto its own grid.
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
        level = _nearest_resolution_index(
            _level_voxel_sizes(self),
            _as_affine_ignoring_fields(_reslice_voxel2world(self, geometry)),
        )
        return self.to_singlescale(level).reslice(geometry, **opt)

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


def _reslice_voxel2world(
    image: "MultiScaleImage",
    geometry: tx.Optional[tx.Union[Image, Geometry, Transformation]] = None,
) -> Transformation:
    """The voxel-to-world transformation of the grid `image` is resliced onto.

    `geometry` is accepted in each of the forms
    [reslice][brainhops.datamodel.images.MultiScaleImage.reslice] takes, and
    read the same way
    [SingleScaleImage.reslice][brainhops.datamodel.images.SingleScaleImage.reslice]
    reads its own.
    """
    if geometry is None:
        return image.geometry.transformation
    if isinstance(geometry, Image):
        return geometry.geometry.transformation
    if isinstance(geometry, Geometry):
        return geometry.transformation
    return geometry


def _level_voxel_sizes(
    image: "MultiScaleImage",
) -> tx.List[tx.Optional[ArrayProtocol]]:
    """The voxel size of every level of `image`, finest first.

    Each is a per-axis vector in world units, so it can be compared with
    the grid an image is resliced onto. A level whose placement does not
    reduce to an affine, even with its fields discarded, has an unknown
    voxel size and is reported as `None`.

    The pyramid's own transformation is included, so a pyramid whose
    placement rescales its levels -- a unit conversion, say -- is measured
    in the same units as the target grid.
    """
    sizes = []  # type: tx.List[tx.Optional[ArrayProtocol]]
    for level in image.images or ():
        affine = _as_affine_ignoring_fields(
            image.transformation @ level.transformation
        )
        sizes.append(None if affine is None else axis_scales(affine.matrix))
    return sizes
