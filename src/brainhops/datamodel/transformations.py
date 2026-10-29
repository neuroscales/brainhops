# -- deprecated names ----------------------------------------------------
# The old spellings of the coordinate fields (#547) still import, with a
# DeprecationWarning, for one release.
from brainhops._core._internal_api import (  # noqa: E402
    renamed_attributes as _renamed_attributes,
)

from ._transformations import *  # noqa: F401, F403
from ._transformations import __all__  # noqa: F401

__getattr__ = _renamed_attributes(
    __name__,
    {
        "CoordinatesField": "CoordinateField",
        "InverseCoordinatesField": "InverseCoordinateField",
    },
)
