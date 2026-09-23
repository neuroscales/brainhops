"""The geometry of an image: its sampling grid and voxel-to-world
transformation."""

__all__ = ["Geometry"]

# dependencies
import typing_extensions as tx
from bagof.magic import Factory, NoRepr

# core
from brainhops.backends import get_array_backend
from brainhops.datamodel.base import DataModelBase

# internals
from .axes import Axis
from .systems import CoordinateSystem
from .transformations import (
    Affine,
    CartesianField,
    Identity,
    ImmutableSequence,
    ModeLike,
    Sequence,
    Transformation,
)


def _geometry_factory() -> tx.Tuple[CartesianField, Transformation]:
    return (CartesianField(), Identity())


class _GeometryFields(DataModelBase):
    # --- attributes ---------------------------------------------------

    # Named `_transformations`, the storage slot that `Sequence` declares
    # and serves through its `transformations` property. The constructor
    # argument is still `transformations=`.
    _transformations: tx.Annotated[
        tx.Tuple[CartesianField, Transformation],
        tx.Doc("A cartesian field and a voxel-to-world transformation."),
        Factory(_geometry_factory),
        NoRepr(),
    ]

    shape: tx.Annotated[
        tx.Optional[tx.Tuple[int, ...]],
        tx.Doc("The shape of the image data."),
        NoRepr(),
    ] = None

    grid: tx.Annotated[
        tx.Optional[CartesianField],
        tx.Doc("The Cartesian field that defines the grid of the image."),
    ] = None

    transformation: tx.Annotated[
        tx.Optional[Transformation],
        tx.Doc("The voxel-to-world transformation."),
    ] = None


class Geometry(_GeometryFields, ImmutableSequence):
    """
    A Cartesian field and a voxel-to-world transformation that, together,
    define the geometry of an image.

    The Cartesian field defines the grid onto which the image is defined.
    The transformation maps the voxel coordinates to world coordinates.
    """

    # --- properties ---------------------------------------------------

    @property
    def transformation(self) -> Transformation:
        """
        The voxel-to-world transformation that defines the image geometry.
        """
        return self.transformations[1]

    @transformation.setter
    def transformation(self, value: Transformation) -> None:
        if value is not None:
            self.transformations = (self.grid, value)

    @property
    def grid(self) -> CartesianField:
        """The Cartesian field that defines the grid of the image."""
        return self.transformations[0]

    @grid.setter
    def grid(self, value: CartesianField) -> None:
        if value is not None:
            self.transformations = (value, self.transformation)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the image data."""
        return self.grid.shape

    @shape.setter
    def shape(self, value: tx.Tuple[int, ...]) -> None:
        if value is not None:
            self.grid = CartesianField(
                shape=value,
                input=self.grid.input,
                output=self.grid.output,
            )

    # --- operators ----------------------------------------------------

    def __rmatmul__(self, other: Transformation) -> tx.Self:
        """Compose `other` with this geometry's transformation.

        Returns a new `Geometry` with the same grid, whose transformation
        is the composition of `other` and this geometry's transformation.
        """
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

    # --- methods ------------------------------------------------------

    def compute(
        self,
        mode: tx.Optional[ModeLike] = None,
        *,
        simplify: bool = False,
    ) -> tx.Self:
        """
        Compute the geometry by simplifying its transformation.

        A `Geometry` holds a grid and a voxel-to-world transformation. The
        transformation part is computed, and the result is returned as a
        `Geometry` with the same grid and the simplified transformation.
        The grid is preserved, so the geometry keeps its grid-and-
        transformation pair and the domain it defines is never lost.
        """
        # Every transformation exposes the same
        # `compute(mode, *, simplify=...)`, so the voxel-to-world part is
        # computed uniformly whether it is a sequence or a single leaf. The
        # grid is kept as is, so the sampling domain is never lost.
        flat = self._flattened()
        transformation = flat.transformation.compute(mode, simplify=simplify)
        return Geometry(
            (flat.grid, transformation),
            input=self.input,
            output=self.output,
        )

    # --- helpers ------------------------------------------------------

    def _flattened(self) -> tx.Self:
        # A `Geometry` keeps its (grid, transformation) pair. Only the
        # transformation part is flattened, so the grid that restricts the
        # domain is never merged into the surrounding sequence. The
        # geometry's own input and output are propagated onto the grid and
        # the transformation, matching `Sequence._flattened`.
        grid, transformation = self.grid, self.transformation
        if grid.input is None and self.input is not None:
            grid = grid.to(input=self.input)
        if transformation.output is None and self.output is not None:
            transformation = transformation.to(output=self.output)
        if isinstance(transformation, Sequence):
            flat_seq = transformation._flattened()
            flat = flat_seq.transformations or []
            transformation = flat[0] if len(flat) == 1 else flat_seq
        return Geometry(
            (grid, transformation),
            input=self.input,
            output=self.output,
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
    index = index[:index_ellipsis] + fill + index[(index_ellipsis + 1) :]

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

    # Create an affine matrix.
    # Seed it with zeros and set every entry explicitly in the loop below.
    # An identity seed would leave a spurious diagonal 1 in the row of a
    # dropped (integer) axis and in the column of an inserted (`None`) axis,
    # corrupting the coordinate mapping.
    backend = get_array_backend()
    affine = backend.zeros((nb_input_dims, nb_output_dims + 1))

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

            # Compute output shape. The ceiling bias depends on the sign of
            # the step: `step - 1` for a forward slice, `step + 1` for a
            # reversed one, matching `len(range(start, stop, step))`.
            if step > 0:
                stop = min(stop, shp)
                oshp = max(0, (stop - start + (step - 1)) // step)
            else:
                stop = max(stop, -1)
                oshp = max(0, (stop - start + (step + 1)) // step)
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
