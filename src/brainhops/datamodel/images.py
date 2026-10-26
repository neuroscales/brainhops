"""Single-scale and multi-scale images, and reslicing them."""

from collections.abc import Mapping

import numpy as np
import typing_extensions as tx
from bagof.hints.numpy import DTypeLike
from bagof.magic import Forget, NoRepr, NotKwOnly, replace

from brainhops._core.affines import axis_scales
from brainhops._core.magic import smart_replace
from brainhops._core.properties import alias, smartproperty
from brainhops._core.typing import ArrayProtocol, CartesianIndex, Clip, Shape
from brainhops.backends import BackendLike, DeviceLike

from ._backends import (
    _BackendsMixin,
    _stored_data,
    as_backend,
    dataobjproperty,
    dataproperty,
)
from ._frames import (
    FrameSlot,
    FrameTargets,
    ReframeMethodOrCallable,
    array_axes,
    axis_index,
    change_matrix,
    component_axes,
    component_count,
    compose_frames,
    explicit_frame,
    frames_as_backend,
    jacobian,
    normalize_frames,
    pair_axes,
    resolve_target,
    squeeze_axis,
    with_components,
)
from ._transformations.compute.multiscale import (
    _as_affine_ignoring_fields,
    _nearest_resolution_index,
    nearest_level,
)
from ._transformations.compute.separable import pull_separable
from .axes import AxisSequence, TensorAxis
from .base import DataModelBase, _ComparesByIdMixin
from .enums import BoundaryLike, InterpolationLike
from .geometry import Geometry, _index2transform
from .metadata import Metadata
from .metadata._derive import Indexed, Reframed, Resampled
from .multiscale import Multiscale
from .systems import CoordinateSystem
from .transformations import (
    CartesianField,
    Identity,
    Sequence,
    Transformation,
)


def _always_unset(value: tx.Any) -> bool:
    """Return true, so that a property always computes its value.

    The value is then cached, and an assignment deletes the cache.
    """
    return True


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
        # an image of another class gives its data as it holds them. Its
        # frames are read through the public property for the same
        # reason, unless they are all the array system, the default.
        if (
            not args
            and type(other) is not cls
            and isinstance(other, SingleScaleImage)
        ):
            data = _stored_data(other)
            if data is not None:
                kwargs.setdefault("data", data)
            frames = other.frames
            if any(slot is not None for slot in frames):
                kwargs.setdefault("frames", frames)
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

        * if the assigned value is an integer, the transformation at that
          index is selected and moved to the end of the list;
        * if the assigned value is a string, the transformation whose
          output system has that name is selected and moved to the end of
          the list. When several output systems have that name, the
          transformation closest to the end of the list is selected, so
          that the preferred transformation stays preferred if it
          matches;
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
            # Several transformations may end in systems of the same name,
            # so the search starts from the preferred one and keeps it when
            # it matches.
            for i in reversed(range(len(transformations))):
                x = transformations[i]
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

    data = dataproperty(invalidates="frames")
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

    _frames: NoRepr[tx.Any] = ()
    """The frames of the values, as they were given (see `frames`)."""

    # The stored value of `frames` is what was given, which may be a
    # mapping or a frame that must still be restricted, so the property
    # always reads it through its getter, which makes the slots explicit
    # once and caches them. A subclass that decodes frames from a file
    # declares the property again with its own getter.
    @smartproperty(cache=True, unset=_always_unset)
    def frames(self) -> tx.Tuple[FrameSlot, ...]:
        """The frame of the values along each axis.

        Some axes hold the components of points, vectors or tensors, and
        the type of such an axis says which (see
        [`ComponentAxis`][brainhops.datamodel.axes.ComponentAxis]). The
        components are expressed in a coordinate system, called the frame
        of the values, and each axis has one slot that records it:

        - `None` is the array system of the image, so that the components
          are in voxel units along the axes of the data. This is the
          default.
        - A transformation maps the array system to the frame. It is
          often an entry of `transformations`, such as the map to scanner
          RAS for a file that stores its vectors in RAS.
        - A coordinate system names a foreign frame, which belongs to
          another space that no transformation of the image reaches. The
          values along such an axis are never re-expressed.

        Only a component axis can have a frame. The constructor and the
        setter also accept a mapping from axis names or indices to slots,
        and an empty tuple or `None`, which means that nothing is given,
        so that every slot is `None`. A subclass that decodes frames from
        a file returns those frames instead when nothing is given. The
        slots are made explicit once and cached, so the same objects are
        returned at every read, and assigning frames or data drops the
        cache.

        A stored transformation corresponds exactly to the components:
        the outputs that no component axis of the array feeds are as
        many as the components, which follow them in order. A
        transformation that is given with more such outputs, such as a
        map that also carries a time axis along, is restricted to the
        outputs that share one axis type, when exactly one type has as
        many outputs as there are components. A projection that removes
        the other outputs is then composed onto the transformation.

        Lazily applying a transformation keeps the frames, because the
        values do not change. Indexing and reslicing compose each frame
        with the map of the voxels, so that the values keep their meaning
        without being rotated. [`reframe`][] re-expresses the values in
        another frame.

        Raises
        ------
        TypeError
            If a slot is neither a transformation, a coordinate system nor
            `None`.
        ValueError
            If there is not one slot per axis, if an axis that is not a
            component axis has a frame, or if a transformation does not
            tell which of its axes the components follow.
        KeyError
            If a mapping names an axis that does not exist.
        """
        return normalize_frames(
            self._frames, self._array_axes(), tuple(self.shape)
        )

    def _array_axes(self) -> tx.Optional[AxisSequence]:
        """Return the axes of the array system, or `None` if unknown."""
        return array_axes(self.transformation.input, self.ndim)

    # --- conversion to another backend -------------------------------

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        # Overrides _BackendsMixin to convert transformations and frames as
        # well.
        opt = dict(backend=backend, device=device, inplace=inplace)
        obj = super().as_backend(**opt)
        changes = {}
        transformations = obj.transformations
        converted = as_backend(transformations, **opt)
        if converted is not transformations:
            changes["transformations"] = converted
        frames = obj.frames
        converted = frames_as_backend(frames, **opt)
        if converted is not frames:
            changes["frames"] = converted
        if not changes:
            return obj
        return smart_replace(obj, "inplace", **changes)

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
        reframe: tx.Union[bool, ReframeMethodOrCallable] = False,
        frame: FrameTargets = -1,
    ) -> "SingleScaleImage":
        """Resample the data onto another geometry.

        The values along an axis that holds vectors or tensors are
        sampled without being rotated: each of their
        [`frames`][brainhops.datamodel.images.SingleScaleImage.frames] is
        composed with the map of the voxels, so that the values keep
        their meaning. With `reframe`, the values are then re-expressed
        in the frame `frame` of the result, as by [`reframe`][].

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

        reframe : bool | ReframeMethod | str | callable, default=False
            Whether and how the values of vector and tensor axes are
            re-expressed after they are sampled. `False` keeps them in
            their frame, `True` is the same as `"jacobian"`, and the other
            values are the methods of [`reframe`][].

        frame : int or str or Transformation or None or dict, default=-1
            The frame in which the values are re-expressed when `reframe`
            is set, as in [`reframe`][]. It is read on the result, whose
            only transformation is that of `geometry`, so `-1` is the
            world of `geometry`.

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

        # The array system of the target may describe the components of this
        # image as channels, so the component axes of this image are copied
        # onto it, and each frame is composed with the map of the voxels.
        target = geometry.transformation
        frames = ()
        source_axes = self._array_axes()
        if component_axes(source_axes):
            system = with_components(
                target.input, self.transformation.input, source_axes, self.ndim
            )
            if system is not target.input:
                target = target.to(input=system)
                vox2vox = vox2vox.to(input=system)
            frames = compose_frames(
                self.frames,
                list(range(self.ndim)),
                vox2vox,
                array_axes(system, self.ndim),
                tuple(new_data.shape),
            )
        resliced = SingleScaleImage(
            data=new_data,
            transformations=[target],
            metadata=metadata,
            frames=frames,
        )
        if reframe is False:
            return resliced
        method = "jacobian" if reframe is True else reframe
        return resliced.reframe(frame, method=method)

    def reframe(
        self,
        frame: FrameTargets = -1,
        *,
        method: ReframeMethodOrCallable = "jacobian",
    ) -> "SingleScaleImage":
        """Re-express the values of vectors and tensors in another frame.

        Some axes hold the components of vectors or tensors, expressed in
        a frame that
        [`frames`][brainhops.datamodel.images.SingleScaleImage.frames]
        records. This method changes that frame. With the default
        arguments, the values are re-expressed in the world of the
        preferred transformation, which is what a writer or a viewer of
        the image expects. No voxel moves.

        If the values are in the frame `F` and are re-expressed in the
        frame `F′`, the components at each voxel `x` are changed by

        ```text
        J(x) = D F′(x) · (D F(x))⁻¹
        ```

        where `D` is the Jacobian, which is the linear part of a map that
        is affine. The type and the valence of the axis say how the
        components change (see
        [`ComponentAxis`][brainhops.datamodel.axes.ComponentAxis]): each
        vector index is multiplied by `J` and each covector index by
        `J⁻ᵀ`, so that a vector becomes `J v`, a gradient becomes
        `J⁻ᵀ g` and a diffusion tensor becomes `J T Jᵀ`. A tensor stored
        in a layout with components that are zero by construction is
        stored in a closed layout afterwards (see
        [`TensorAxis`][brainhops.datamodel.axes.TensorAxis]), which
        changes the length of its axis. The coordinates of points cannot
        be re-expressed this way.

        The components follow some axes of the system of their frame, in
        order, and these axes are read from the structure of the frame,
        not from the types of the axes. They are the outputs of the frame
        that no component axis of the array feeds, and a stored frame has
        exactly one such output per component. A frame given as `frame`
        that has more of them is first restricted to the outputs that
        share one axis type, as the setter of
        [`frames`][brainhops.datamodel.images.SingleScaleImage.frames]
        does. The Jacobian is taken along the axes of the array that the
        frame maps onto these outputs, and the two frames of a change
        must map the same axes of the array. The array system takes the
        axes of the other frame.

        The Jacobian of a map that is not affine is computed by finite
        differences between the voxels: central inside the image and
        one-sided at its edges. Along an axis of one voxel, the fields of
        the map are taken to be constant.

        Parameters
        ----------
        frame : int or str or Transformation or None or dict, default=-1
            The new frame. An integer selects an entry of
            `transformations`, without reordering it, so that `-1` is the
            preferred transformation. A string selects the most preferred
            transformation whose output system has that name. `None` is
            the array system, in which the components are in voxel units.
            A transformation maps the array system to the new frame. A
            single value applies to every component axis, except the axes
            whose frame is foreign. A dictionary maps axis names or
            indices to frames, and the axes that it leaves out keep their
            frame.

        method : ReframeMethod | str | callable, default="jacobian"
            What is applied to the values. `"jacobian"` applies `J`, which
            changes the frame exactly. `"rotation"` applies the rotation
            `R` of the polar decomposition `J = R P`, which keeps the
            length of vectors and the eigenvalues of tensors, as the
            finite-strain reorientation of diffusion tensors does. A
            callable receives `J`, of shape `(n, n)` or `(*shape, n, n)`,
            and returns the matrix to apply.

        Returns
        -------
        SingleScaleImage
            The image with its values re-expressed, the same
            transformations, and its new frames. The image itself is
            returned when no value changes frame. Its metadata is derived
            from the metadata of this image with a [`Reframed`][] record
            (see
            [`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive]):
            the fields that describe the values, such as `data_unit` and
            `display_range`, are cleared, and so are the fields with one
            entry per index along an axis whose values were re-expressed.
            `"reframe"` is appended to `history`.

        Raises
        ------
        KeyError
            If a frame names a system that no transformation ends in, or
            if a dictionary names an axis that does not exist.
        ValueError
            If a dictionary names an axis that is not a component axis or
            that has a foreign frame, if the frames do not tell which of
            their axes the components follow, or if they map different
            axes of the array onto them.
        NotImplementedError
            If the values of an axis whose frame changes are the
            coordinates of points.
        """
        axes = self._array_axes()
        slots = self.frames
        targets = _frame_targets(self, frame, axes, slots)
        targets = {i: t for i, t in targets.items() if t is not slots[i]}
        if not targets:
            return self
        for i in targets:
            if axes[i].valences is None:
                raise NotImplementedError(
                    f"The values along axis {i} are the coordinates of "
                    f"points, which change with the map itself rather than "
                    f"with its Jacobian, so they cannot be re-expressed in "
                    f"another frame."
                )

        data = self.data
        shape = tuple(data.shape)
        components = component_axes(axes)
        # The frames do not depend on the position along a component axis,
        # so their Jacobians are computed with these axes reduced to one
        # entry.
        reduced = tuple(
            1 if i in components else s for i, s in enumerate(shape)
        )
        cache: tx.Dict[tx.Any, tx.Any] = {}

        def jacobian_of(
            slot: tx.Optional[Transformation],
            info: tx.Any,
            cols: tx.List[int],
        ) -> tx.Any:
            key = (id(slot), tuple(cols))
            if key not in cache:
                cache[key] = jacobian(slot, info, cols, reduced, data)
            return cache[key]

        new_slots = list(slots)
        new_axes = list(axes)
        for i, target in targets.items():
            axis = axes[i]
            source = slots[i]
            ncomponents = component_count(axes, i, shape[i], [source, target])
            source_info, target_info, cols = pair_axes(
                source, target, axes, ncomponents
            )
            matrix = change_matrix(
                jacobian_of(source, source_info, cols),
                jacobian_of(target, target_info, cols),
                method,
                ncomponents,
            )
            data = axis.apply(data, squeeze_axis(matrix, i, len(shape)), i)
            new_slots[i] = target
            if isinstance(axis, TensorAxis):
                # A layout with components that are zero by construction is
                # replaced by a layout that holds the tensor in any frame.
                new_axes[i] = axis.reframed(ncomponents, shape[i])
        transformations = self.transformations
        if any(new is not old for new, old in zip(new_axes, axes)):
            transformations = [
                _with_input_axes(xform, new_axes) for xform in transformations
            ]
            new_slots = [
                _with_input_axes(slot, new_axes)
                if isinstance(slot, Transformation)
                else slot
                for slot in new_slots
            ]
        metadata = self.metadata
        if metadata is not None:
            types = tuple(axes[i].type for i in targets)
            metadata = metadata.derive(Reframed(types), history="reframe")
        return SingleScaleImage(
            data=data,
            transformations=transformations,
            metadata=metadata,
            frames=tuple(new_slots),
        )

    def __call__(self, transformation: Transformation) -> "SingleScaleImage":
        """Apply a transformation lazily, without reslicing.

        The output space of `transform` should match the output space of
        the preferred transformation. The new preferred transformation,
        `transform.inverse() @ self.transformation`, is appended to the
        existing ones. No voxel moves, so the new image shares the
        metadata and the frames of this image.
        """
        transformation = transformation.inverse() @ self.transformation
        transformations = list(self.transformations or [])
        transformations.append(transformation)
        return SingleScaleImage(
            data=_stored_data(self),
            transformations=transformations,
            metadata=self.metadata,
            frames=self.frames,
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

        The [`frames`][brainhops.datamodel.images.SingleScaleImage.frames]
        of the values are composed with the map from the indexed voxels
        to the original ones, so the values are not changed. A component
        axis stays a component axis only when it is indexed with its full
        range in order. Otherwise its remaining entries are no longer the
        components of one object, so it becomes a channel axis and loses
        its frame.

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
        frames = ()
        if component_axes(self._array_axes()):
            sub2full, shape = _index2transform(index, self.shape, grid.input)
            frames = compose_frames(
                self.frames,
                _source_axes(sub2full),
                sub2full,
                array_axes(sub2full.input, len(shape)),
                tuple(shape),
            )
        return SingleScaleImage(
            data=data,
            transformations=transformations,
            metadata=metadata,
            frames=frames,
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
        reframe: tx.Union[bool, ReframeMethodOrCallable] = False,
        frame: FrameTargets = -1,
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

        reframe : bool | ReframeMethod | str | callable, default=False
            Whether and how the values of vector and tensor axes are
            re-expressed, as in [`SingleScaleImage.reslice`][].

        frame : int or str or Transformation or None or dict, default=-1
            The frame in which the values are re-expressed, as in
            [`SingleScaleImage.reslice`][].

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
            reframe=reframe,
            frame=frame,
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


def _frame_targets(
    image: SingleScaleImage,
    frame: FrameTargets,
    axes: tx.Optional[tx.Sequence[tx.Any]],
    slots: tx.Sequence[FrameSlot],
) -> tx.Dict[int, tx.Optional[Transformation]]:
    """Return the new frame of each axis whose values change frame.

    A single frame applies to every component axis whose frame is not
    foreign. A mapping applies to the axes that it names. Each frame is
    restricted to the axes that the components follow, as
    `explicit_frame` does.

    Raises
    ------
    ValueError
        If a mapping names an axis that is not a component axis or that
        has a foreign frame, or if a frame does not tell which of its
        axes the components follow.
    """
    transformations = list(image.transformations or [])
    shape = tuple(image.shape)
    components = component_axes(axes)
    if not isinstance(frame, Mapping):
        target = resolve_target(frame, transformations)
        chosen = {
            i: target
            for i in components
            if not isinstance(slots[i], CoordinateSystem)
        }
    else:
        chosen = {}
        for key, value in frame.items():
            i = axis_index(key, axes, image.ndim)
            if i not in components:
                raise ValueError(
                    f"Axis {key!r} is not a component axis, so its values "
                    f"have no frame."
                )
            if isinstance(slots[i], CoordinateSystem):
                raise ValueError(
                    f"The values along axis {key!r} are in a foreign frame, "
                    f"which no transformation of the image reaches."
                )
            chosen[i] = resolve_target(value, transformations)
    # A frame that is restricted to the same number of components is the
    # same projection, so it is built once for all the axes that share it.
    restricted: tx.Dict[tx.Any, tx.Optional[Transformation]] = {}
    targets = {}
    for i, target in chosen.items():
        if target is None:
            targets[i] = None
            continue
        known = [target, slots[i]]
        ncomponents = component_count(axes, i, shape[i], known)
        key = (id(target), ncomponents)
        if key not in restricted:
            restricted[key] = explicit_frame(target, axes, ncomponents)
        targets[i] = restricted[key]
    return targets


def _with_input_axes(
    xform: Transformation, axes: tx.Sequence[tx.Any]
) -> Transformation:
    """Return a transformation whose input system has other axes.

    The axes replace those of the input system when the system has as
    many axes, and the transformation is returned as it is otherwise.
    """
    system = xform.input
    if array_axes(system, len(axes)) is None:
        return xform
    return xform.to(input=replace(system, axes=list(axes)))


def _source_axes(sub2full: Transformation) -> tx.List[tx.Optional[int]]:
    """Return the original axis of each axis of an indexed array.

    `sub2full` is the affine that indexing builds. Each of its columns
    belongs to an axis of the indexed array, and has one nonzero entry, in
    the row of the original axis, unless the axis was inserted.
    """
    matrix = np.asarray(sub2full.matrix)[:, :-1]
    sources = []
    for column in matrix.T:
        rows = np.flatnonzero(column)
        sources.append(int(rows[0]) if len(rows) else None)
    return sources


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
