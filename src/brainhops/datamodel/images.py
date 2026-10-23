"""Single-scale and multi-scale images, and reslicing them."""

import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike
from bagof.magic import Forget, NoRepr, NotKwOnly, replace

from brainhops._core.affines import axis_scales
from brainhops._core.magic import smart_replace
from brainhops._core.properties import alias
from brainhops._core.typing import ArrayProtocol, CartesianIndex, Clip, Shape
from brainhops.backends import BackendLike, DeviceLike

from ._backends import (
    _BackendsMixin,
    _stored_data,
    as_backend,
    dataobjproperty,
    dataproperty,
)
from ._transformations.compute.multiscale import (
    _as_affine_ignoring_fields,
    _nearest_resolution_index,
    nearest_level,
)
from ._transformations.compute.separable import pull_separable
from .base import DataModelBase, _ComparesByIdMixin
from .enums import BoundaryLike, InterpolationLike
from .geometry import Geometry
from .metadata import Metadata
from .metadata._derive import Indexed, Resampled
from .multiscale import Multiscale
from .transformations import (
    CartesianField,
    Identity,
    Sequence,
    Transformation,
)


class Image(
    _BackendsMixin,
    _ComparesByIdMixin,
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
    """What describes the image beyond its data and geometry."""

    # --- conversions --------------------------------------------------

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if _single_multi_swap(other, cls):
            # single-scale to multi-scale or multi-scale to single-scale
            # > convert via `to`
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
        self,
        cls: tx.Optional[tx.Type["Image"]] = None,
        *,
        copy: bool = False,
        **fields,
    ) -> tx.Any:
        target = type(self) if cls is None else cls

        # --- single-scale to multi-scale ------------------------------
        if isinstance(self, SingleScaleImage) and issubclass(
            target, MultiScaleImage
        ):
            pyramid = MultiScaleImage(images=[self], metadata=self.metadata)
            return target.from_instance(pyramid, **fields)

        # --- multi-scale to single-scale ------------------------------
        if isinstance(self, MultiScaleImage) and issubclass(
            target, SingleScaleImage
        ):
            return target.from_instance(self.to_singlescale(0), **fields)

        # --- edit fields only -----------------------------------------
        return super().to(cls, copy=copy, **fields)

    # --- array API ----------------------------------------------------

    def __array__(self, dtype: tx.Optional[DTypeLike] = None) -> np.ndarray:
        """Convert the data to a NumPy array, optionally of a given type."""
        return np.asarray(_stored_data(self), dtype=dtype)

    @property
    def shape(self) -> Shape:
        """The shape of the data."""
        return tuple(self.dataobj.shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the data."""
        return len(self.shape)

    dtype = alias("dataobj.dtype", readonly=True)
    """The data type of the data."""

    grid = alias("geometry.grid", readonly=True)
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


GeometryLike: tx.TypeAlias = tx.Union[Image, Geometry, Transformation]


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

    Must be an array-like object that supports the array protocol
    (e.g. numpy, cupy or dask array).
    """
    # It defaults to `None` so that subclasses can derive it lazily -- a
    # format reader typically holds a handle to the file and materializes
    # the array on first access, and cannot supply it at construction time.
    # This mirrors `Affine.matrix`, which is optional for the same reason.

    data = dataproperty()
    """The image data, in the backend of the setting in effect.

    Under the default setting, the stored array is returned as it is.
    Under any other setting, such as `with backend("dask"):`, a new array
    of that backend is returned, and the stored array does not change.
    """

    dataobj = dataobjproperty()
    """The array of the image, without reading it.

    An image built from an array returns that array. An image read
    from a file returns a lazy array of the file in the layout of
    `data`, which reads nothing until it is converted, unless an array
    was assigned as `data` or `dataobj`. Assigning `dataobj` is the same
    as assigning `data`.
    """

    transformations: NotKwOnly[tx.List[Transformation]] = ()
    """
    A list of transformations from voxel space (i.e., in terms of data's
    axes) to different world spaces.

    The last transformation in the list is the preferred one.
    """

    # --- conversion to another backend -------------------------------

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        # Overrides _BackendsMixin to convert transformations as well.
        opt = dict(backend=backend, device=device, inplace=inplace)
        obj = super().as_backend(**opt)
        transformations = obj.transformations
        converted = as_backend(transformations, **opt)
        if converted is transformations:
            return obj
        return smart_replace(obj, "inplace", transformations=converted)

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
        geometry: tx.Optional[GeometryLike] = None,
        degree: InterpolationLike = 1,
        bound: BoundaryLike = "reflect",
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

        geometry: Geometry = _get_geometry(self, geometry)

        # Replace each multiscale transformation field by the level that
        # best matches the output grid.
        transformation = self.transformation
        transformation = _match_resolution(transformation, geometry)

        # Build the voxel-to-voxel transformation, and reslice
        vox2vox = transformation.inverse() @ geometry.transformation
        transformation = vox2vox @ geometry.grid
        new_data = pull_separable(self.data, transformation, **opt)

        metadata = self.metadata
        if metadata is not None:
            # The grid is left out of the map of the voxels: it is the
            # identity on voxel coordinates, but a field, which has no
            # affine form.
            metadata = metadata.derive(
                Resampled(vox2vox, geometry), history="reslice"
            )
            if dtype is not None:
                metadata = replace(metadata, dtype=new_data.dtype)
        return SingleScaleImage(
            data=new_data,
            transformations=[geometry.transformation],
            metadata=metadata,
        )

    def __call__(self, transformation: Transformation) -> "SingleScaleImage":
        """Apply a transformation lazily, without reslicing.

        The output space of `transform` should match the output space of
        the preferred transformation. The new preferred transformation,
        `transform.inverse() @ self.transformation`, is appended to the
        existing ones. No voxel moves, so the new image shares the
        metadata of this image.
        """
        transformation = transformation.inverse() @ self.transformation
        transformations = list(self.transformations or [])
        transformations.append(transformation)
        return SingleScaleImage(
            data=_stored_data(self),
            transformations=transformations,
            metadata=self.metadata,
        )

    def __getitem__(self, index: CartesianIndex) -> "SingleScaleImage":
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
        index : [tuple of] (int | slice | None)
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


class MultiScaleImage(Image, Multiscale[SingleScaleImage]):
    """The base class of multi-scale (pyramid) images.

    `images` and `transformations` are the parameters most often given
    to the constructor, and they can be passed by position in this order.
    The other parameters are keyword-only, except those that a file
    reader from `brainhops.io.common` adds, such as the opened Zarr
    `node` of an OME-Zarr pyramid, which follow `transformations`.
    """

    scales: Forget

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

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        # Overrides _BackendsMixin to convert transformations as well.
        changes = {}
        for name in ("images", "transformations"):
            value = getattr(self, name)
            converted = as_backend(value, backend, device=device)
            if converted is not value:
                changes[name] = converted
        return smart_replace(self, "inplace" if inplace else None, **changes)

    def to_singlescale(self, index: int = 0) -> SingleScaleImage:
        """Return a level as a single-scale image in world space."""
        return self.images[index](self.transformation.inverse())

    @property
    def nscales(self) -> int:
        """The number of pyramid levels."""
        return len(self.images)

    @property
    def scales(self) -> tx.List[SingleScaleImage]:
        """The pyramid levels, from highest to lowest resolution."""
        return list(self.iterscales)

    @property
    def iterscales(self) -> tx.Iterator[SingleScaleImage]:
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
        geometry: tx.Optional[GeometryLike] = None,
        degree: InterpolationLike = 1,
        bound: BoundaryLike = "reflect",
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
            _as_affine_ignoring_fields(_get_voxel2world(self, geometry)),
        )
        return self.to_singlescale(level).reslice(geometry, **opt)

    def __call__(self, transform: Transformation) -> "MultiScaleImage":
        """Apply a transformation to the pyramid lazily, without reslicing.

        The transformation is handled as in [`SingleScaleImage`][].
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


def _get_geometry(
    image: "MultiScaleImage", geometry: tx.Optional[GeometryLike] = None
) -> Geometry:
    if geometry is None:
        geometry = image.geometry
    if isinstance(geometry, Image):
        geometry = geometry.geometry
    if not isinstance(geometry, Geometry):
        geometry = Geometry((image.geometry.grid, geometry))
    return geometry


def _get_voxel2world(
    image: "MultiScaleImage",
    geometry: tx.Optional[GeometryLike] = None,
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
    multiscale: MultiScaleImage,
) -> tx.List[tx.Optional[ArrayProtocol]]:
    """Return the voxel size of each level in world units, finest first.

    The preferred transformation of the pyramid is included, so that the sizes
    are measured in the units of the target. A level whose transformation is
    not affine, even when its fields are ignored, gets `None`.
    """
    sizes = []  # type: tx.List[tx.Optional[ArrayProtocol]]
    for level in multiscale.images or ():
        affine = _as_affine_ignoring_fields(
            multiscale.transformation @ level.transformation
        )
        sizes.append(None if affine is None else axis_scales(affine.matrix))
    return sizes


def _single_multi_swap(other: Image, cls: tx.Type[Image]) -> bool:
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


def _match_resolution(
    transformation: Transformation, voxel2world: Transformation
) -> Transformation:
    """
    Replace every multiscale field inside the transformation by its
    scale that best matches the grid of `voxel2world`, the voxel-to-world t
    ransformation of the grid an image is resliced onto.

    This functions is applied recursively to sequences.
    A transformation without a multiscale field is returned unchanged.
    """
    if isinstance(voxel2world, Geometry):
        voxel2world = voxel2world.transformation
    if isinstance(transformation, Multiscale):
        return transformation.to_singlescale(
            nearest_level(transformation, voxel2world)
        )
    if isinstance(transformation, Sequence):
        parts = transformation.transformations or []
        resolved = [_match_resolution(part, voxel2world) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(transformation, transformations=resolved)
        return transformation
    return transformation
