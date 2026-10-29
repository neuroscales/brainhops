"""Geometry of an image: its sampling grid and voxel-to-world mapping."""

__all__ = ["Geometry"]


import typing_extensions as tx
from bagof.magic import Factory, NoRepr, NotKwOnly

from brainhops._core.typing import CartesianIndex, Shape

from ._indexing import index2transform
from .base import DataModelBase
from .transformations import (
    CartesianField,
    Identity,
    ImmutableSequence,
    ModeLike,
    Sequence,
    SimplifyLike,
    Transformation,
)

if tx.TYPE_CHECKING:
    from ._transformations.restriction import AxisReferences, InputValues


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
        sub2full, shape = index2transform(index, self.shape, self.grid.input)
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
        input_axes: "AxisReferences",
        output_axes: "AxisReferences",
        at: tx.Optional["InputValues"] = None,
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
            build_restriction,
            require_axis_counts,
            resolve_restriction,
        )

        restriction = resolve_restriction(self, input_axes, output_axes, at)
        require_axis_counts(self, restriction)
        inputs, outputs = restriction.inputs, restriction.outputs
        values = restriction.removed_input_values
        shape = self.shape
        if shape is None or not all(
            _is_index(values.get(p), shape[p])
            for p in inputs.removed_positions
        ):
            return super().restrict(input_axes, output_axes, at)
        if not inputs.removed_positions and not outputs.removed_positions:
            return self
        # The transformation reads the coordinates that the grid gives
        # it, so the projection that creates the removed axes of the grid
        # faces the output system of the grid.
        transformation = build_restriction(
            self.transformation, restriction, self.grid.output, self.output
        )
        if not inputs.removed_positions:
            return Geometry((self.grid, transformation))
        voxels = self.grid.input
        if voxels is not None:
            voxels = voxels.expand(len(shape)).restrict(inputs.positions)
        grid = CartesianField(
            shape=tuple(shape[p] for p in inputs.positions),
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


def _is_index(value: tx.Optional[float], size: int) -> bool:
    """Return whether a coordinate is the index of a sample of an axis."""
    return (
        value is not None and float(value).is_integer() and 0 <= value < size
    )
