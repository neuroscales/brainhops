"""Geometry of an image: its sampling grid and voxel-to-world mapping."""

__all__ = ["Geometry"]

import numbers

import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, NoRepr, NotKwOnly

from brainhops._core.typing import CartesianIndex, Shape

from ._sugar import get_axes
from .axes import Axis, ChannelAxis, is_component_type
from .base import DataModelBase
from .systems import (
    AxisSequence,
    CoordinateSystem,
)
from .transformations import (
    Affine,
    CartesianField,
    Identity,
    ImmutableSequence,
    ModeLike,
    Sequence,
    SimplifyLike,
    Transformation,
)

if tx.TYPE_CHECKING:
    from ._transformations.restriction import AxisRefs


def _geometry_factory() -> tx.Tuple[CartesianField, Transformation]:
    return (CartesianField(), Identity())


class _GeometryFields(DataModelBase):
    # --- attributes ---------------------------------------------------

    # The `Sequence` base exposes this slot as `transformations`, which is
    # also the name of the constructor argument.
    _transformations: tx.Annotated[
        tx.Tuple[CartesianField, Transformation],
        Factory(_geometry_factory),
        NoRepr(),
        NotKwOnly(),
    ]
    """A cartesian field and a voxel-to-world transformation."""

    shape: NoRepr[tx.Optional[Shape]] = None
    """The shape of the image data."""

    grid: tx.Optional[CartesianField] = None
    """The Cartesian field that defines the grid of the image."""

    transformation: tx.Optional[Transformation] = None
    """The voxel-to-world transformation."""


class Geometry(_GeometryFields, ImmutableSequence):
    """Geometry of an image.

    A geometry pairs the Cartesian field on which the image is sampled
    with the transformation from voxel to world coordinates.
    """

    # --- properties ---------------------------------------------------

    @property
    def transformation(self) -> Transformation:
        """Voxel-to-world transformation, the second element of the pair."""
        return self.transformations[1]

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        if value is not None:
            self.transformations = (self.grid, value)

    @property
    def grid(self) -> CartesianField:
        """Cartesian field of the image grid, the first element of the pair."""
        return self.transformations[0]

    @grid.setter
    def grid(self, value: CartesianField) -> None:
        if value is not None:
            self.transformations = (value, self.transformation)

    @property
    def shape(self) -> Shape:
        """Shape of the image data, which is the shape of the grid."""
        return self.grid.shape

    @shape.setter
    def shape(self, value: tx.Optional[tx.Iterable[int]]) -> None:
        if value is not None:
            self.grid = CartesianField(
                shape=tuple(value),
                input=self.grid.input,
                output=self.grid.output,
            )

    # --- operators ----------------------------------------------------

    def __rmatmul__(self, other: Transformation) -> tx.Self:
        """Return the geometry with `other` applied after its transformation.

        The grid is kept, and the output system is that of `other`.
        """
        return Geometry(
            (self.grid, other @ self.transformation),
            input=self.grid.input,
            output=other.output,
        )

    def __getitem__(self, index: CartesianIndex) -> tx.Self:
        """Return the geometry of a sub-image.

        The index is the one that would be applied to the image data: a
        tuple of integers, slices, `None` and possibly an ellipsis.
        """
        sub2full, shape = _index2transform(index, self.shape, self.grid.input)
        return Geometry(
            (
                CartesianField(
                    shape=shape,
                    input=sub2full.input,
                    output=sub2full.input,
                ),
                self.transformation @ sub2full,
            )
        )

    # --- methods ------------------------------------------------------

    def restrict(
        self,
        input_axes: "AxisRefs",
        output_axes: "AxisRefs",
        at: tx.Optional[tx.Mapping[tx.Any, float]] = None,
    ) -> Transformation:
        """Return the geometry restricted to some of its axes.

        Restricting a geometry gives the geometry of a part of the
        image, such as one volume of a time series or one slice of a
        volume. The input axes of a geometry are the axes of its grid,
        and each sample of the restricted grid must be a sample of the
        original grid. A removed input therefore needs a value in `at`
        that is the index of a sample along its axis, which is an
        integer between zero and the size of the axis. The result is
        then a geometry, as for indexing with an integer: its grid has
        the kept input axes, with the shape of the image along them, and
        its transformation is the restriction of the voxel-to-world
        transformation, as
        [`Transformation.restrict`][brainhops.datamodel.transformations.Transformation.restrict]
        builds it. Removing outputs only keeps the grid. The
        restriction is lazy, and [`compute`][] keeps the grid.

        A removed input that `at` does not name, or that `at` places
        between two samples or outside the grid, has no grid to sample.
        The result is then the restriction that
        [`Transformation.restrict`][brainhops.datamodel.transformations.Transformation.restrict]
        returns for any transformation, which is a map and not a
        geometry. The geometry keeps the contract of that method in this
        way, so that a free input that no kept output reads still gives
        a valid map. The same happens when the shape of the grid is
        unknown.

        !!! example
            ```pycon
            >>> grid = CartesianField(shape=(4, 5, 6, 3))
            >>> geometry = Geometry((grid, Affine(matrix=np.eye(4, 5))))
            >>> keep = [0, 1, 2]
            >>> volume = geometry.restrict(keep, keep, at={3: 2})
            >>> type(volume).__name__, volume.shape
            ('Geometry', (4, 5, 6))
            ```

        Parameters
        ----------
        input_axes : AxisQuery or iterable of (int or AxisQuery)
            Axes of the grid to keep, given by position, by name or by
            query.
        output_axes : AxisQuery or iterable of (int or AxisQuery)
            Output axes to keep, given by position, by name or by query.
        at : mapping of (int or AxisQuery) to float, optional
            Index of each removed axis of the grid. Each key designates
            input axes as an item of `input_axes` does.

        Returns
        -------
        Transformation
            A geometry when every removed input is at the index of a
            sample, the geometry itself when both sides keep every axis,
            and otherwise the lazy restriction of the geometry as a
            transformation.

        Raises
        ------
        RestrictionError
            If the number of input or output axes is unknown.
        KeyError
            If no axis has a name that a reference gives.
        IndexError
            If a position lies outside the axes of the geometry.
        ValueError
            If a query matches no axis, a name is shared by several
            axes, an axis is designated twice, or `at` gives a value to
            an input that is kept.
        TypeError
            If a reference is neither a position nor a query, or if `at`
            is not a mapping of real numbers.
        """
        from ._transformations.restriction import (
            require_counts,
            resolve_block,
            restricted_chain,
        )

        block = resolve_block(self, input_axes, output_axes, at)
        require_counts(self, block)
        shape = self.shape
        if shape is None or not all(
            _is_sample(block.values.get(p), shape[p]) for p in block.removed_in
        ):
            return super().restrict(input_axes, output_axes, at)
        if not block.removed_in and not block.removed_out:
            return self
        # The transformation reads the coordinates that the grid gives
        # it, so the projection that creates the removed axes of the grid
        # faces the output system of the grid.
        transformation = restricted_chain(
            self.transformation, block, self.grid.output, self.output
        )
        if not block.removed_in:
            return Geometry((self.grid, transformation))
        voxels = self.grid.input
        if voxels is not None:
            voxels = voxels.expand(len(shape)).restrict(block.kept_in)
        grid = CartesianField(
            shape=tuple(shape[p] for p in block.kept_in),
            input=voxels,
            output=voxels,
        )
        return Geometry((grid, transformation))

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: SimplifyLike = "analytic",
        factor: bool = False,
    ) -> tx.Self:
        """Return the geometry with its transformation simplified.

        The arguments are forwarded to the `compute` method of the
        transformation. The grid is kept, so the sampling domain of the
        image is never lost.
        """
        flat = self.flatten()
        transformation = flat.transformation.compute(
            mode, simplify=simplify, factor=factor
        )
        return Geometry(
            (flat.grid, transformation),
            input=self.input,
            output=self.output,
        )

    def flatten(self, endpoints: bool = True) -> tx.Self:
        """Return the geometry with its transformation flattened.

        A geometry keeps its shape, so the result is a geometry: the grid
        and the voxel-to-world transformation stay two separate elements,
        and only the transformation is flattened.
        """
        # Only the transformation is flattened, and the grid is never merged
        # into it. The systems of the geometry are propagated onto both parts,
        # as in `Sequence.flatten`.
        grid, transformation = self.grid, self.transformation
        if endpoints:
            if grid.input is None and self.input is not None:
                grid = grid.to(input=self.input)
            if transformation.output is None and self.output is not None:
                transformation = transformation.to(output=self.output)
        if isinstance(transformation, Sequence):
            flat_seq = transformation.flatten(endpoints)
            flat = flat_seq.transformations or []
            transformation = flat[0] if len(flat) == 1 else flat_seq
        return Geometry(
            (grid, transformation),
            input=self.input,
            output=self.output,
        )


def _is_sample(value: tx.Optional[float], size: int) -> bool:
    """Return whether a coordinate is the index of a sample of an axis."""
    return (
        value is not None and float(value).is_integer() and 0 <= value < size
    )


def _index2transform(
    index: CartesianIndex,
    shape: Shape,
    system: tx.Optional[CoordinateSystem] = None,
) -> tx.Tuple[Affine, Shape]:
    """Convert an array index into an affine from sub-array to array.

    Parameters
    ----------
    index : tuple of int or slice or None or ellipsis
        Tuple of integers (NumPy integers included), slices, `None` and
        at most one ellipsis, which is appended when missing. Booleans
        are rejected, because NumPy reads them as masks and not as
        positions.
    shape : tuple of int
        Shape of the original array.
    system : CoordinateSystem, optional
        Coordinate system of the original array.

    Returns
    -------
    affine : Affine
        Transformation from the indexed array to the original array.
    shape : tuple of int
        Shape of the indexed array.

    Raises
    ------
    ValueError
        If an index element is not an integer, a slice, `None` or an
        ellipsis, or if the index contains more than one ellipsis.
    """
    # Validate every element before anything else, so that an invalid
    # element raises an error whether or not the axes of `system` are
    # known. Integers are converted to built-in integers, which the code
    # below expects.
    if not isinstance(index, tuple):
        index = (index,)
    checked_index = []
    for idx in index:
        if isinstance(idx, numbers.Integral) and not isinstance(idx, bool):
            checked_index.append(int(idx))
        elif idx is None or idx is ... or isinstance(idx, slice):
            checked_index.append(idx)
        else:
            raise ValueError(f"Invalid index: {idx}")
    index = tuple(checked_index)
    if sum(1 for idx in index if idx is ...) > 1:
        raise ValueError(f"Invalid index: {index} has more than one ellipsis")

    if ... not in index:
        index = (*index, ...)
    index_ellipsis = index.index(...)
    nb_indexed_dims = sum(1 for idx in index if idx not in (None, ...))
    nb_implicit_dims = len(shape) - nb_indexed_dims
    fill = (slice(None),) * nb_implicit_dims
    index = index[:index_ellipsis] + fill + index[(index_ellipsis + 1) :]

    nb_output_dims = sum(1 for idx in index if not isinstance(idx, int))
    nb_input_dims = len(shape)

    input_axes: tx.Optional[AxisSequence] = get_axes(system)
    if input_axes == [...]:
        input_axes = None
    elif input_axes.is_variadic:
        input_axes = input_axes.expand(nb_input_dims)
    output_axes = None
    if input_axes:
        output_axes = []
        axis_walker = list(input_axes)
        dim = 0
        for idx in index:
            if isinstance(idx, int):
                axis_walker.pop(0)
                dim += 1
            elif isinstance(idx, slice):
                axis = axis_walker.pop(0)
                # The entries of a component axis are the components of one
                # object only when the axis keeps all of them, in order.
                # Otherwise they are independent values, as on a channel
                # axis.
                kept = range(*idx.indices(shape[dim]))
                if is_component_type(getattr(axis, "type", None)) and (
                    kept != range(shape[dim])
                ):
                    axis = ChannelAxis(
                        name=axis.name,
                        unit=axis.unit,
                        discrete=axis.discrete,
                    )
                output_axes.append(axis)
                dim += 1
            else:
                # The element is `None`, which inserts a new axis.
                output_axes.append(Axis())

    # Start from zeros and set every entry explicitly. An identity seed
    # would leave a spurious 1 in the row of a dropped (integer) axis and
    # in the column of an inserted (None) axis.
    # The matrix is a small parameter, so it is built with NumPy whatever
    # the backend setting.
    affine = np.zeros((nb_input_dims, nb_output_dims + 1))

    input_dim_walker = 0
    output_dim_walker = 0
    output_shape = []
    for idx in index:
        if idx is None:
            output_dim_walker += 1
            output_shape.append(1)

        elif isinstance(idx, int):
            affine[input_dim_walker, -1] = idx
            input_dim_walker += 1

        elif isinstance(idx, slice):
            shp = shape[input_dim_walker]
            step = idx.step or 1
            if idx.start is None:
                start = 0 if step > 0 else (shp - 1)
            else:
                start = (shp + idx.start) if idx.start < 0 else idx.start
            if idx.stop is None:
                stop = shp if step > 0 else -1
            else:
                stop = (shp + idx.stop) if idx.stop < 0 else idx.stop

            # Round the length up, with a bias that depends on the sign of
            # the step, to match `len(range(start, stop, step))`.
            if step > 0:
                stop = min(stop, shp)
                oshp = max(0, (stop - start + (step - 1)) // step)
            else:
                stop = max(stop, -1)
                oshp = max(0, (stop - start + (step + 1)) // step)
            output_shape.append(oshp)

            affine[input_dim_walker, output_dim_walker] = step
            affine[input_dim_walker, -1] = start

            input_dim_walker += 1
            output_dim_walker += 1

    # The input system is that of the indexed array and the output system
    # is that of the original array.
    return Affine(
        matrix=affine,
        input=CoordinateSystem(axes=output_axes) if output_axes else None,
        output=system,
    ), tuple(output_shape)
