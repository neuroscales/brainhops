"""Geometry of an image: its sampling grid and voxel-to-world mapping."""

__all__ = ["Geometry"]

import typing_extensions as tx
from bagof.magic import Factory, NoRepr

from brainhops.backends import get_array_backend

from ._sugar import get_axes
from .axes import Axis
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


def _geometry_factory() -> tx.Tuple[CartesianField, Transformation]:
    return (CartesianField(), Identity())


class _GeometryFields(DataModelBase):
    # The `Sequence` base exposes this slot as `transformations`, which is
    # also the name of the constructor argument.
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
    """Geometry of an image.

    A geometry pairs the Cartesian field on which the image is sampled
    with the transformation from voxel to world coordinates.
    """

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
    def shape(self) -> tx.Tuple[int, ...]:
        """Shape of the image data, which is the shape of the grid."""
        return self.grid.shape

    @shape.setter
    def shape(self, value: tx.Tuple[int, ...]) -> None:
        if value is not None:
            self.grid = CartesianField(
                shape=value,
                input=self.grid.input,
                output=self.grid.output,
            )

    def __rmatmul__(self, other: Transformation) -> tx.Self:
        """Return the geometry with `other` applied after its transformation.

        The grid is kept, and the output system is that of `other`.
        """
        return Geometry(
            (self.grid, other @ self.transformation),
            input=self.grid.input,
            output=other.output,
        )

    def __getitem__(
        self, index: tx.Tuple[tx.Union[int, slice, None], ...]
    ) -> tx.Self:
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
        flat = self._flattened()
        transformation = flat.transformation.compute(
            mode, simplify=simplify, factor=factor
        )
        return Geometry(
            (flat.grid, transformation),
            input=self.input,
            output=self.output,
        )

    def _flattened(self) -> tx.Self:
        # Only the transformation is flattened, and the grid is never merged
        # into it. The systems of the geometry are propagated onto both parts,
        # as in `Sequence._flattened`.
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
    """Convert an array index into an affine from sub-array to array.

    Parameters
    ----------
    index : tuple of int or slice or None
        Tuple of integers, slices, `None` and at most one ellipsis, which
        is appended when missing.
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
        If an index element is invalid and the axes of `system` are known.
    """
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
    elif input_axes.is_open:
        input_axes = input_axes.expand(nb_input_dims)
    output_axes = None
    if input_axes:
        output_axes = []
        axis_walker = list(input_axes)
        for idx in index:
            if isinstance(idx, int):
                axis_walker.pop(0)
            elif isinstance(idx, slice):
                output_axes.append(axis_walker.pop(0))
            elif idx is None:
                output_axes.append(Axis())
            else:
                raise ValueError(f"Invalid index: {idx}")

    # Start from zeros and set every entry explicitly. An identity seed
    # would leave a spurious 1 in the row of a dropped (integer) axis and
    # in the column of an inserted (None) axis.
    backend = get_array_backend()
    affine = backend.zeros((nb_input_dims, nb_output_dims + 1))

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
