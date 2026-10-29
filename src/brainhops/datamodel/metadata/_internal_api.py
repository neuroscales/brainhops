"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.datamodel.metadata` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "apply_loss_policy",
    "BIDS_KEYS",
    "call_under_policy",
    "FIELDS",
    "GROUPS",
    "Indexed",
    "Reframed",
    "Resampled",
    "short",
    "takes_error_policy",
    "VOCABULARY",
]

from ._derive import Indexed, Reframed, Resampled
from ._report import (
    apply_loss_policy,
    call_under_policy,
    short,
    takes_error_policy,
)
from ._vocabulary import BIDS_KEYS, FIELDS, GROUPS, VOCABULARY
