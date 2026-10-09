"""Single-scale and multi-scale images, and reslicing them."""

import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike

from brainhops._core.affines import axis_scales
from brainhops._core.typing import ArrayProtocol

from ._transformations.multiscale import (
    _as_affine_ignoring_fields,
    _at_resolution,
    _nearest_resolution_index,
)
from .base import DataModelBase, IdentityComparison
from .geometry import Geometry
from .transformations import (
    CartesianField,
    Identity,
    Transformation,
)


class Image(IdentityComparison, DataModelBase, eq=False):
    """The base class of all images.

    !!! note
        Images compare and hash by identity: `a == b` means `a is b`, even for
        images with the same data and transformations, and `==` never raises.
        Images can therefore be used in sets and as dictionary keys. To compare
        the contents of two images, compare their data explicitly, for example
        with `numpy.array_equal(a, b)`, and compare their geometries.
    """

    # --- array API ----------------------------------------------------

    def __array__(self, dtype: tx.Optional[DTypeLike] = None) -> np.ndarray:
        """Convert the data to a NumPy array, optionally of a given type."""
        return np.asarray(self.data, dtype=dtype)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the data."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the data."""
        return self.data.dtype

    @property
    def grid(self) -> CartesianField:
        """The Cartesian field that defines the sampling grid."""
        return self.geometry.grid


class SingleScaleImage(Image):
    """The base class of single-resolution images."""

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
        """The preferred transformation, which is always the last in the list.

        The getter returns an [`Identity`][] when the list is empty. Assigning
        a transformation makes it the preferred one: a new transformation is
        appended, and one already in the list (compared by identity) is moved
        to the end. An integer or a string selects an existing transformation
        by position or by the name of its output space, and an unknown name
        raises a KeyError.
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
        """The geometry of the image.

        The geometry pairs the Cartesian field matching the data shape with the
        preferred voxel-to-world transformation. Any image can be resliced onto
        this geometry.
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
        degree: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
        copy: bool = False,
    ) -> tx.Self:
        """Resample the data onto another geometry.

        Parameters
        ----------
        geometry : Image or Geometry or Transformation, optional
            The target voxel-to-world geometry. An image or a geometry also
            determines the output shape, whereas a bare transformation keeps
            the current shape. By default, the image is resampled onto its own
            geometry.
        degree : int, default=1
            Spline degree, from 0 (nearest neighbour) up to 5. A degree of 1
            gives linear interpolation.
        bound : str or float, default="reflect"
            Boundary condition. A float is used as a constant value beyond
            the edge.
            The strings are:

                - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
                - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
                - 'mirror': mirror at edge        (d c b | a b c d | c b a)
                - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
                - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        coeff : bool, default=False
            Whether the data already holds spline coefficients. If false, the
            data is prefiltered before interpolation.
        copy : bool, default=False
            If true, the output never shares memory with the input. If false,
            a reslicing that only rearranges existing voxels, such as a flip,
            a permutation or a slice with a unit step, may return a view, as
            `torch.Tensor.to` does. Dask arrays are immutable and are never
            copied.

        Returns
        -------
        SingleScaleImage
            The resliced image.
        """
        opt = dict(degree=degree, bound=bound, coeff=coeff, copy=copy)

        if geometry is None:
            geometry = self.geometry
        if isinstance(geometry, Image):
            geometry = geometry.geometry
        if not isinstance(geometry, Geometry):
            geometry = Geometry((self.geometry.grid, geometry))

        # Replace each multiscale field by the level that best matches the
        # output grid.
        preferred = _at_resolution(
            self.transformation, geometry.transformation
        )

        # Each group of independent axes is resampled separately, so that
        # rescaling, flipping or permuting axes stays cheap. The function is
        # imported here to avoid an import cycle.
        from ._transformations.compute.separable import pull_separable

        transformation = (
            preferred.inverse() @ geometry.transformation @ geometry.grid
        )
        new_data = pull_separable(self.data, transformation, **opt)
        return SingleScaleImage(
            data=new_data, transformations=[geometry.transformation]
        )

    def __call__(self, transform: Transformation) -> "SingleScaleImage":
        """Apply a transformation lazily, without reslicing.

        The output space of `transform` should match the output space of the
        preferred transformation. The new preferred transformation,
        `transform.inverse() @ self.transformation`, is appended to the
        existing ones.
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
        """Index the data, keeping every transformation consistent with it.

        The index may contain integers, slices and `None`.
        """
        if not isinstance(index, tuple):
            index = (index,)
        data = self.data[index]
        grid = self.grid
        transformations = [
            Geometry((grid, xform))[index].transformation
            for xform in self.transformations
        ]
        return SingleScaleImage(data=data, transformations=transformations)


class MultiScaleImage(Image):
    """The base class of multi-scale (pyramid) images."""

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
        """The data of the highest-resolution level."""
        return self.images[0].data

    def to_singlescale(self, index: int = 0) -> SingleScaleImage:
        """Return a level as a single-scale image in world space."""
        return self.images[index](self.transformation.inverse())

    @property
    def nscales(self) -> int:
        """The number of pyramid levels."""
        return len(self.images)

    @property
    def scales(self) -> tx.Iterator[SingleScaleImage]:
        """Every level, as a single-scale image."""
        for i in range(len(self.images)):
            yield self.to_singlescale(i)

    @property
    def transformation(self) -> Transformation:
        """The preferred transformation, which is always the last in the list.

        The getter returns an [`Identity`][] when the list is empty. Assigning
        a transformation makes it the preferred one: a new transformation is
        appended, and one already in the list (compared by identity) is moved
        to the end. An integer or a string selects an existing transformation
        by position or by the name of its output space, and an unknown name
        raises a KeyError.
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
        """The geometry of the highest-resolution level.

        The geometry pairs the grid of the highest-resolution level with the
        preferred transformation of the pyramid, composed with that level's
        own transformation.
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
        degree: int = 1,
        bound: str = "reflect",
        coeff: bool = False,
        copy: bool = False,
    ) -> tx.Self:
        """Reslice the level whose voxel size best matches the target grid.

        The method always returns a single-scale image.

        Parameters
        ----------
        geometry : Image or Geometry or Transformation, optional
            The target geometry, interpreted as in
            [`SingleScaleImage.reslice`][].
        degree : int, default=1
            Spline degree, from 0 (nearest neighbour) up to 5. A degree of 1
            gives linear interpolation.
        bound : str or float, default="reflect"
            Boundary condition. A float is used as a constant value beyond
            the edge.
            The strings are:

                - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
                - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
                - 'mirror': mirror at edge        (d c b | a b c d | c b a)
                - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
                - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        coeff : bool, default=False
            Whether the data already holds spline coefficients. If false, the
            data is prefiltered before interpolation.
        copy : bool, default=False
            If true, the output never shares memory with the input. If false,
            a reslicing that only rearranges existing voxels, such as a flip,
            a permutation or a slice with a unit step, may return a view, as
            `torch.Tensor.to` does. Dask arrays are immutable and are never
            copied.

        Returns
        -------
        SingleScaleImage
            The resliced image.
        """
        opt = dict(degree=degree, bound=bound, coeff=coeff, copy=copy)
        level = _nearest_resolution_index(
            _level_voxel_sizes(self),
            _as_affine_ignoring_fields(_reslice_voxel2world(self, geometry)),
        )
        return self.to_singlescale(level).reslice(geometry, **opt)

    def __call__(self, transform: Transformation) -> tx.Self:
        """Apply a transformation to the pyramid lazily, without reslicing.

        The transformation is handled as in [`SingleScaleImage.__call__`][].
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
    """Return the voxel-to-world transformation of the reslicing target."""
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
    """Return the voxel size of each level in world units, finest first.

    The preferred transformation of the pyramid is included, so that the sizes
    are measured in the units of the target. A level whose transformation is
    not affine, even when its fields are ignored, gets `None`.
    """
    sizes = []  # type: tx.List[tx.Optional[ArrayProtocol]]
    for level in image.images or ():
        affine = _as_affine_ignoring_fields(
            image.transformation @ level.transformation
        )
        sizes.append(None if affine is None else axis_scales(affine.matrix))
    return sizes
