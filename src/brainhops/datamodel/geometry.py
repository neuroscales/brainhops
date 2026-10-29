"""Geometry of an image: its sampling grid and voxel-to-world mapping."""

__all__ = ["Geometry"]


import typing_extensions as tx
from bagof.magic import Factory, NoRepr, NotKwOnly

from brainhops._core._internal_api import CartesianIndex, Shape

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
