import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel.base import DataModelBase


class Raw(DataModelBase, frozen=True, convert=False):
    """Base class for raw metadata read from a file."""

    def copy(self) -> tx.Self:
        """Return a copy of the object.

        Returns
        -------
        self
            The copy.
        """
        return replace(self)
