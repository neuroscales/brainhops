__all__ = ["Geometry"]

# dependencies
import typing_extensions as tx

# core
from brainhops._core.backends import get_array_backend

# internals
from .axes import Axis
from .systems import CoordinateSystem
from .transformations import Affine, CartesianField, Sequence, Transformation


class Geometry(Sequence):
    """
    A Cartesian field and a voxel-to-world transformation that, together,
    define the geometry of an image.

    The Cartesian field defines the grid onto which the image is defined.
    The transformation maps the voxel coordinates to world coordinates.
    """

    transformations: tx.Tuple[CartesianField, Transformation]

    _shape: tx.Optional[tx.Tuple[int, ...]] = None
    """The shape of the image data."""

    _grid: tx.Optional[CartesianField] = None
    """The Cartesian field that defines the grid of the image."""

    _transformation: tx.Optional[Transformation] = None
    """The voxel-to-world transformation that defines the image geometry."""

    @property
    def transformation(self) -> Transformation:
        """
        The voxel-to-world transformation that defines the image geometry.
        """
        return self.transformations[1]

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        self.transformations = (self.grid, value)

    @property
    def grid(self) -> CartesianField:
        """The Cartesian field that defines the grid of the image."""
        return self.transformations[0]

    @grid.setter
    def grid(self, value: CartesianField) -> None:
        self.transformations = (value, self.transformation)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the image data."""
        return self.grid.shape

    @shape.setter
    def shape(self, value: tx.Tuple[int, ...]) -> None:
        self.grid = CartesianField(
            shape=value,
            input=self.grid.input,
            output=self.grid.output,
        )

    def __rmatmul__(self, other: Transformation) -> tx.Self:
        return Geometry(
            (self.grid, other @ self.transformation),
            input=self.grid.input,
            output=other.output,
        )

    def __getitem__(
        self, index: tx.Tuple[tx.Union[int, slice, None], ...]
    ) -> tx.Self:
        """
        This mimics indexing into the data array of an image and returns
        the geometry of the resulting sub-image.
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


def _index2transform(
    index: tx.Tuple[tx.Union[int, slice, None], ...],
    shape: tx.Tuple[int, ...],
    system: tx.Optional[CoordinateSystem] = None,
) -> Transformation:
    """
    Convert an ND index into a transformation that maps the coordinates
    of the indexed array to the coordinates of the original array.

    Parameters
    ----------
    index : tuple[int | slice | Ellipsis | None, ...]
        The index to convert.
    shape : tuple[int, ...]
        The shape of the original array.
    system : CoordinateSystem, optional
        The coordinate system of the original array. If provided, the
        transformation will be returned in the same coordinate system.

    Returns
    -------
    Transformation
        The transformation that maps the coordinates of the indexed array to
        the coordinates of the original array.
    shape
        The shape of the resulting array.
    """

    # Convert ellipsis to slices
    if ... not in index:
        index = (*index, ...)
    index_ellipsis = index.index(...)
    nb_indexed_dims = sum(1 for idx in index if idx not in (None, ...))
    nb_implicit_dims = len(shape) - nb_indexed_dims
    fill = (slice(None),) * nb_implicit_dims
    index = index[:index_ellipsis] + fill + index[index_ellipsis + 1 :]

    # Compute number of output dimensions after indexing
    # (some may be dropped, some may be added)
    nb_output_dims = sum(1 for idx in index if not isinstance(idx, int))
    nb_input_dims = len(shape)

    # Compute output axes
    input_axes = getattr(system, "axes", None)
    output_axes = None
    if input_axes:
        output_axes = []
        axis_walker = list(input_axes)
        for idx in index:
            if isinstance(idx, int):
                # If it's an integer index, we need to remove this axis
                axis_walker.pop(0)
            elif isinstance(idx, slice):
                # If it's a slice, we keep the axis but adjust its properties
                output_axes.append(axis_walker.pop(0))
            elif idx is None:
                # If it's None, we are adding a new axis (dimension)
                output_axes.append(Axis())
            else:
                raise ValueError(f"Invalid index: {idx}")

    # Create an affine matrix
    backend = get_array_backend()
    affine = backend.eye(nb_input_dims + 1, nb_output_dims + 1)[:-1]

    # Iterate over each dimension and apply the index
    input_dim_walker = 0
    output_dim_walker = 0
    output_shape = []
    for idx in index:
        if idx is None:
            # The new axis does not map to any input dimension
            output_dim_walker += 1
            output_shape.append(1)

        elif isinstance(idx, int):
            # If it's an integer index, save it as an offset
            affine[input_dim_walker, -1] = idx
            input_dim_walker += 1

        elif isinstance(idx, slice):
            # If it's a slice, we need to adjust the transformation accordingly
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

            # Compute output shape
            if step > 0:
                stop = min(stop, shp)
            else:
                stop = max(stop, -1)
            oshp = max(0, (stop - start + (step - 1)) // step)
            output_shape.append(oshp)

            # Fill affine matrix
            affine[input_dim_walker, output_dim_walker] = step
            affine[input_dim_walker, -1] = start

            input_dim_walker += 1
            output_dim_walker += 1

    # Create the transformation object.
    # NOTE: the input coordinate system is the sub-array, and the
    # output coordinate system is the original (input) array.
    return Affine(
        matrix=affine,
        input=CoordinateSystem(axes=output_axes) if output_axes else None,
        output=system,
    ), tuple(output_shape)
