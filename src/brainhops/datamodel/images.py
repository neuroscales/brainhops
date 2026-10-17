"""Single-scale and multi-scale images, and reslicing them."""

import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike
from bagof.magic import NoRepr, NotKwOnly, replace

from brainhops._core.affines import axis_scales
from brainhops._core.properties import alias
from brainhops._core.typing import ArrayProtocol, Clip
from brainhops.backends import _CURRENT

from ._backends import (
    _as_backend_data,
    _BackendsMixin,
    _stored_data,
    _with_fields,
    as_backend_all,
    computing_with,
    dataproperty,
    voxel_arrays_of,
)
from ._transformations.multiscale import (
    _as_affine_ignoring_fields,
    _at_resolution,
    _nearest_resolution_index,
)
from .base import DataModelBase, IdentityComparison
from .geometry import Geometry
from .metadata import Metadata
from .metadata._derive import Indexed, Resampled
from .transformations import (
    CartesianField,
    Identity,
    Transformation,
)


class Image(
    _BackendsMixin,
    IdentityComparison,
    DataModelBase,
    eq=False,
    kw_only=True,
):
    """The base class of all images.

    !!! note
        Images compare and hash by identity: `a == b` means `a is b`,
        even for images with the same data and transformations, and
        `==` never raises. Images can therefore be used in sets and as
        dictionary keys. To compare the contents of two images, compare
        their data explicitly, for example with `numpy.array_equal(a, b)`,
        and compare their geometries.
    """

    metadata: NoRepr[tx.Optional[Metadata]] = None
    """What describes the image beyond its data and geometry, or `None`.

    The metadata holds the common vocabulary of
    [`Metadata`][brainhops.datamodel.metadata.Metadata], such as a
    repetition time, a description or a slice timing. An image built
    from data has none unless it is given some. A file format takes the
    field over to hold the metadata of its own class, which also holds
    the header of the file, and converts the metadata it is given into
    that class. The field is keyword-only and is left out of the
    representation of the image.
    """

    # --- conversions --------------------------------------------------

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy an image, or convert a single-scale image or a pyramid.

        A single-scale image copied into a pyramid, or a pyramid copied
        into a single-scale image, changes how the image is stored, so it
        is converted as `other.to(cls)` converts it. Any other image is
        copied as
        [`DataModelBase.from_instance`][brainhops.datamodel.base.DataModelBase.from_instance]
        copies it. A single-scale image of another class, such as an
        image read from a file, gives its data as it holds them, without
        converting them with the backend setting.

        Parameters
        ----------
        other : DataModelBase
            The image to copy or convert.
        *args : Any
            Constructor arguments of a copy. A conversion takes keyword
            arguments only.
        **kwargs : Any
            Fields of the result, which take precedence over those of
            `other`.

        Returns
        -------
        Self
            The new image.

        Raises
        ------
        TypeError
            If positional arguments accompany a conversion.
        """
        if _changes_storage(other, cls):
            if args:
                raise TypeError(
                    f"{cls.__name__}.from_instance() converts a "
                    f"{type(other).__name__} with keyword arguments only."
                )
            return other.to(cls, **kwargs)
        # The data of a single-scale image are stored as `_data`, which a
        # format leaves empty when it decodes its data from its file, so
        # an image of another class gives its data as it holds them.
        if (
            not args
            and type(other) is not cls
            and isinstance(other, SingleScaleImage)
        ):
            data = _stored_data(other)
            if data is not None:
                kwargs.setdefault("data", data)
        return super().from_instance(other, *args, **kwargs)

    def to(
        self, cls: tx.Optional[type] = None, *, copy: bool = False, **kwargs
    ) -> tx.Any:
        """Convert the image to another class of image.

        A single-scale image becomes a pyramid of one level. The level
        keeps its own transformations and the pyramid adds none, so the
        image keeps its geometry. A pyramid becomes the single-scale image
        of its highest-resolution level, in the world space of the
        pyramid, and its other levels are left out. Any other conversion
        is as in
        [`DataModelBase.to`][brainhops.datamodel.base.DataModelBase.to].

        Parameters
        ----------
        cls : type, optional
            The class to convert to. By default, the class of the image.
        copy : bool, default=False
            Whether to return a new image even when nothing changes.
        **kwargs : Any
            Fields of the result.

        Returns
        -------
        Image
            The converted image, an instance of `cls`.
        """
        target = type(self) if cls is None else cls
        if isinstance(self, SingleScaleImage) and issubclass(
            target, MultiScaleImage
        ):
            pyramid = MultiScaleImage(images=[self], metadata=self.metadata)
            return target.from_instance(pyramid, **kwargs)
        if isinstance(self, MultiScaleImage) and issubclass(
            target, SingleScaleImage
        ):
            return target.from_instance(self.to_singlescale(0), **kwargs)
        return super().to(cls, copy=copy, **kwargs)

    # --- array API ----------------------------------------------------

    def __array__(self, dtype: tx.Optional[DTypeLike] = None) -> np.ndarray:
        """Convert the data to a NumPy array, optionally of a given type."""
        return np.asarray(_stored_data(self), dtype=dtype)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data."""
        return tuple(self.dataobj.shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the data."""
        return len(self.shape)

    dtype = alias("dataobj.dtype")
    """The data type of the data."""

    grid = alias("geometry.grid")
    """The Cartesian field that defines the sampling grid."""

    @property
    def transformation(self) -> Transformation:
        """
        The preferred transformation, which is always the last in the
        `transformations` list.

        When the list is empty, this property return an [`Identity`][].

        Assigning a transformation makes it the preferred one:

        * if the assigned value is an integer or a string, the corresponding
          transformation (by index of by output system name) is selected
          and moved to the end of the list;
        * if the assigned value is a transformation that is already in
          the list, it is moved to the end of the list;
        * if it is a new transformation, it is appended to the list.

        An unknown index or system name raises a `KeyError`.

        !!! example
            ```pycon
            >>> voxel = CoordinateSystem(["i", "j", "k"], name="voxel")
            >>> phys = CoordinateSystem(["x", "y", "z"], name="phys")
            >>> world = CoordinateSystem(["x", "y", "z"], name="world")
            >>> aligned = CoordinateSystem(["x", "y", "z"], name="aligned")
            >>>
            >>> vox2scanner = Affine(input=voxel, output=phys)
            >>> vox2world = Affine(input=voxel, output=world)
            >>> vox2aligned = Affine(input=voxel, output=aligned)
            >>>
            >>> img = SingleScaleImage(data, [vox2scanner, vox2world])
            >>> img.transformation.output
            CoordinateSystem(name='world', axes=['x', 'y', 'z'])
            >>>
            >>> img.transformation = 0
            >>> img.transformation.output
            CoordinateSystem(name='phys', axes=['x', 'y', 'z'])
            >>>
            >>> img.transformation = "world"
            >>> img.transformation.output
            CoordinateSystem(name='world', axes=['x', 'y', 'z'])
            >>>
            >>> img.transformation = vox2aligned
            >>> img.transformation.output
            CoordinateSystem(name='aligned', axes=['x', 'y', 'z'])
            >>>
            >>> [t.output.name for t in img.transformations]
            ['phys', 'world', 'aligned']
            ```

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


class SingleScaleImage(Image):
    """The base class of single-resolution images.

    `data` and `transformations` are the parameters most often given
    to the constructor, and they can be passed by position in this order.
    The other parameters are keyword-only, except those that a file
    reader from `brainhops.io.common` adds, such as the opened Zarr
    `node` of a plain Zarr array, which follow `transformations`.
    """

    _data: tx.Annotated[
        tx.Optional[ArrayProtocol],
        NotKwOnly(),
        NoRepr(),  # leave out from repr to avoid loading a whole file
    ] = None
    """
    The image data.

    Must be an array-like object that supports the array protocol (e.g.
    numpy, cupy or dask array).

    It defaults to `None` so that subclasses can derive it lazily -- a
    format reader typically holds a handle to the file and materializes
    the array on first access, and cannot supply it at construction time.
    This mirrors `Affine.matrix`, which is optional for the same reason.
    """

    data = dataproperty()
    """The image data, in the backend of the setting in effect.

    Under the default setting, the stored array is returned as it is.
    Under any other setting, such as `with backend("dask"):`, a new array
    of that backend is returned, and the stored array does not change.
    """

    dataobj = alias("_data", readonly=True)
    """The array of the image, without reading it.

    An image built from an array returns that array. An image read
    from a file returns a lazy array of the file in the layout of
    `data`, which reads nothing until it is converted, unless an array
    was assigned as `data`.
    """

    transformations: NotKwOnly[tx.List[Transformation]] = ()
    """
    A list of transformations from voxel space (i.e., in terms of data's
    axes) to different world spaces.

    The last transformation in the list is the preferred one.
    """

    # --- conversion to another backend -------------------------------

    def _voxel_arrays(self) -> tx.Iterator[tx.Any]:
        # The voxels of the image decide its backend.
        dataobj = self.dataobj
        if dataobj is not None:
            yield dataobj

    def as_backend(
        self, backend: tx.Any = _CURRENT, *, inplace: bool = False
    ) -> tx.Self:
        """Return the image with its data converted as a setting says.

        The data are converted, and so are the voxels of the
        transformations of the image. Small parameters, such as affine
        matrices, stay NumPy arrays. An image read from a file holds the
        converted array as assigned data, which a writer encodes again.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.
        inplace : bool, default=False
            Whether this image holds the converted data and is returned,
            instead of a new image.

        Returns
        -------
        SingleScaleImage
            An image of the same class.
        """
        obj = _as_backend_data(self, backend, inplace)
        transformations = obj.transformations
        converted = as_backend_all(transformations, backend)
        if converted is transformations:
            return obj
        return _with_fields(obj, inplace, transformations=converted)

    @property
    def geometry(self) -> Geometry:
        """The geometry of the image.

        The geometry pairs the Cartesian field matching the data shape
        with the preferred voxel-to-world transformation. Any image can
        be resliced onto this geometry.
        """
        return Geometry(
            (
                CartesianField(
                    shape=tuple(self.dataobj.shape),
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
        bound: tx.Union[str, float] = "reflect",
        coeff: bool = False,
        copy: bool = False,
        *,
        clip: Clip = None,
        dtype: tx.Optional[DTypeLike] = None,
    ) -> "SingleScaleImage":
        """Resample the data onto another geometry.

        Parameters
        ----------
        geometry : Image or Geometry or Transformation, optional
            The target voxel-to-world geometry. An image or a geometry
            also determines the output shape, whereas a bare
            transformation keeps the current shape.
            By default, the image is resampled onto its ow geometry.
        degree : int, default=1
            Spline degree, from 0 (nearest neighbour) up to 5.
            A degree of 1 gives linear interpolation.
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
            Whether the data already holds spline coefficients.
            If false, the data is prefiltered before interpolation.
        copy : bool, default=False
            If true, the output never shares memory with the input.
            If false, a reslicing that only rearranges existing voxels,
            such as a flip, a permutation or a slice with a unit step,
            may return a view, as `torch.Tensor.to` does.
            Dask arrays are immutable and are never copied.

        clip : float, pair, sequence of pairs, or None, default=None
            A scalar sets symmetric margins. A `(lower, upper)` pair sets
            asymmetric margins on every input axis; a sequence of pairs
            follows the input data axes, independently of output permutations.
            Samples outside the closed interval
            ``[-lower, size - 1 + upper]`` on any axis are zero. None preserves
            extrapolation with `bound`. Use 0.5 for the outer voxel edges,
            0 for border voxel centres, or -0.5 for inner voxel edges.
            Inside this interval, `bound` still extends interpolation
            stencils; clipping never clamps the coordinates.

        dtype : DTypeLike, optional
            Output dtype. None preserves the input dtype. A floating
            dtype retains interpolated fractions of integer input.
            An explicit dtype promotes the input and output dtypes to
            choose working precision, with a minimum of float32.
            Integer-only combinations use float64.
            Integer output keeps rounding half away from zero and
            saturation. Dask arrays remain lazy.

        Returns
        -------
        SingleScaleImage
            The resliced image. Its metadata is derived from the metadata
            of this image with a [`Resampled`][] record (see
            [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive]):
            the fields tied to the spatial sampling are cleared, an
            encoding direction given in voxel axes is mapped through the
            map of the voxels, and `"reslice"` is appended to `history`.
            The metadata keeps its class and its record.
        """
        opt = dict(
            degree=degree,
            bound=bound,
            coeff=coeff,
            copy=copy,
            clip=clip,
            dtype=dtype,
        )

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

        # The backend of the result is chosen once, from the voxels of the
        # image and of the fields, so that the voxels are read and the
        # coordinates are created in that backend.
        with computing_with(self, preferred, geometry.transformation):
            new2old = preferred.inverse() @ geometry.transformation
            transformation = new2old @ geometry.grid
            new_data = pull_separable(self.data, transformation, **opt)
        metadata = self.metadata
        if metadata is not None:
            # The grid is left out of the map of the voxels: it is the
            # identity on voxel coordinates, but a field, which has no
            # affine form.
            metadata = metadata.derive(
                Resampled(new2old, geometry), history="reslice"
            )
            if dtype is not None:
                metadata = replace(metadata, dtype=new_data.dtype)
        return SingleScaleImage(
            data=new_data,
            transformations=[geometry.transformation],
            metadata=metadata,
        )

    def __call__(self, transform: Transformation) -> "SingleScaleImage":
        """Apply a transformation lazily, without reslicing.

        The output space of `transform` should match the output space of
        the preferred transformation. The new preferred transformation,
        `transform.inverse() @ self.transformation`, is appended to the
        existing ones. No voxel moves, so the new image shares the
        metadata of this image.
        """
        transform = transform.inverse() @ self.transformation
        if self.transformations:
            transformations = list(self.transformations)
            transformations.append(transform)
        else:
            transformations = [transform]
        return SingleScaleImage(
            data=_stored_data(self),
            transformations=transformations,
            metadata=self.metadata,
        )

    def __getitem__(
        self, index: tx.Tuple[tx.Union[int, slice, None], ...]
    ) -> "SingleScaleImage":
        """Index the data, keeping every transformation consistent with it.

        The metadata is derived from the metadata of this image with an
        [`Indexed`][] record (see
        [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive]).
        An index along a time or a channel axis keeps the entries of the
        fields that run along that axis at the positions that it keeps,
        and an index of a spatial axis, or of an axis of unknown type,
        clears the fields tied to the spatial sampling and maps an
        encoding direction given in voxel axes. `"getitem"` is appended
        to `history`, and the metadata keeps its class and its record.

        Parameters
        ----------
        index : int, slice, None or tuple
            The index, as for the data array. A tuple may contain
            integers, slices, `None` and `...`.

        Returns
        -------
        SingleScaleImage
            The indexed image.
        """
        if not isinstance(index, tuple):
            index = (index,)
        data = self.data[index]
        grid = self.grid
        transformations = [
            Geometry((grid, xform))[index].transformation
            for xform in self.transformations
        ]
        metadata = self.metadata
        if metadata is not None:
            metadata = metadata.derive(
                Indexed(index, self.shape, grid.input), history="getitem"
            )
        return SingleScaleImage(
            data=data, transformations=transformations, metadata=metadata
        )


class MultiScaleImage(Image):
    """The base class of multi-scale (pyramid) images.

    `images` and `transformations` are the parameters most often given
    to the constructor, and they can be passed by position in this order.
    The other parameters are keyword-only, except those that a file
    reader from `brainhops.io.common` adds, such as the opened Zarr
    `node` of an OME-Zarr pyramid, which follow `transformations`.
    """

    images: NotKwOnly[tx.List[SingleScaleImage]] = ()
    """
    Each image in the multi-resolution pyramid, ordered from highest to
    lowest resolution.
    """

    transformations: NotKwOnly[tx.List[Transformation]] = ()
    """
    A list of transformations from each level's preferred space to
    different world spaces.

    The last transformation in the list is the preferred one.
    """

    @property
    def data(self) -> ArrayProtocol:
        """The data of the highest-resolution level."""
        return self.images[0].data

    @property
    def dataobj(self) -> tx.Optional[ArrayProtocol]:
        """The array of the highest-resolution level, without reading it."""
        return self.images[0].dataobj

    # --- conversion to another backend -------------------------------

    def _voxel_arrays(self) -> tx.Iterator[tx.Any]:
        # Every level of the image takes part in the choice.
        return voxel_arrays_of(self.images)

    def as_backend(
        self, backend: tx.Any = _CURRENT, *, inplace: bool = False
    ) -> tx.Self:
        """Return the pyramid with the data of every level converted.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.
        inplace : bool, default=False
            Whether this pyramid holds the converted levels and is
            returned, instead of a new pyramid.

        Returns
        -------
        MultiScaleImage
            A pyramid of the same class.
        """
        changes = {}
        for name in ("images", "transformations"):
            value = getattr(self, name)
            converted = as_backend_all(value, backend)
            if converted is not value:
                changes[name] = converted
        return _with_fields(self, inplace, **changes)

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
        bound: tx.Union[str, float] = "reflect",
        coeff: bool = False,
        copy: bool = False,
        *,
        clip: Clip = None,
        dtype: tx.Optional[DTypeLike] = None,
    ) -> "SingleScaleImage":
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

        clip : float, pair, sequence of pairs, or None, default=None
            A scalar sets symmetric margins. A `(lower, upper)` pair sets
            asymmetric margins on every input axis; a sequence of pairs
            follows the input data axes, independently of output permutations.
            Samples outside the closed interval
            ``[-lower, size - 1 + upper]`` on any axis are zero. None preserves
            extrapolation with `bound`. Use 0.5 for the outer voxel edges,
            0 for border voxel centres, or -0.5 for inner voxel edges.
            Inside this interval, `bound` still extends interpolation
            stencils; clipping never clamps the coordinates.

        dtype : DTypeLike, optional
            Output dtype, as in [`SingleScaleImage.reslice`][].
            None preserves the selected level's dtype.

        Returns
        -------
        SingleScaleImage
            The resliced image.
        """
        opt = dict(
            degree=degree,
            bound=bound,
            coeff=coeff,
            copy=copy,
            clip=clip,
            dtype=dtype,
        )
        level = _nearest_resolution_index(
            _level_voxel_sizes(self),
            _as_affine_ignoring_fields(_reslice_voxel2world(self, geometry)),
        )
        return self.to_singlescale(level).reslice(geometry, **opt)

    def __call__(self, transform: Transformation) -> "MultiScaleImage":
        """Apply a transformation to the pyramid lazily, without reslicing.

        The transformation is handled as in [`SingleScaleImage.__call__`][],
        and the new pyramid shares the metadata of this one.
        """
        transform = transform.inverse() @ self.transformation
        if self.transformations:
            transformations = self.transformations.copy()
            transformations.append(transform)
        else:
            transformations = [transform]
        return MultiScaleImage(
            images=self.images,
            transformations=transformations,
            metadata=self.metadata,
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


def _changes_storage(other: tx.Any, cls: type) -> bool:
    """Return whether copying `other` into `cls` changes how it is stored.

    A single-scale image holds its data, and a pyramid holds its levels,
    so one is converted into the other rather than copied by name.
    """
    return (
        isinstance(other, SingleScaleImage)
        and issubclass(cls, MultiScaleImage)
    ) or (
        isinstance(other, MultiScaleImage)
        and issubclass(cls, SingleScaleImage)
    )
