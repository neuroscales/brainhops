# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly

# core
from brainhops._core._internal_api import ArrayProtocol

# datamodel
from .base import DataModelBase

SINGLE_SCALE = tx.TypeVar("SINGLE_SCALE", covariant=True)


class Multiscale(DataModelBase, tx.Generic[SINGLE_SCALE]):
    """Mixin for a pyramid of scales, finest first.

    Index 0 is the scale with the highest resolution.

    This base class provides the list of scales and the operations that
    select one of these scales.

    It does not define what a scale is: a subclass supplies the scales
    and, through the `_level_resolution` hook, the physical grid size of
    each one, from which `nearest_level` picks the scale whose resolution
    is closest to a target grid.
    """

    # --- attributes ---------------------------------------------------

    scales: NotKwOnly[tx.List[SINGLE_SCALE]] = ()
    """Scales, from finest to coarsest."""

    @property
    def nscales(self) -> int:
        """Number of scales."""
        return len(self.scales)

    @property
    def _finest(self) -> SINGLE_SCALE:
        # Return the finest scale, or `None` if there are no scales.
        scales = self.scales
        return scales[0] if scales else None

    # --- methods ------------------------------------------------------

    def to_singlescale(self, index: int = 0) -> SINGLE_SCALE:
        """Return the scale at `index`, where 0 is the finest.

        The scale is returned as it is stored: for a field, it is the plain
        transformation of that scale, not a multiscale field.
        """
        return self.scales[int(index)]

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # Return the physical grid size of a level, as a vector with one entry
        # per axis in the input units, or `None` if the size is unknown. This
        # hook returns `None` here, and subclasses override it.
        return None
