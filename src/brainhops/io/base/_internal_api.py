"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.base` from this module, and not from the modules that
define them. This module only re-exports these names. No module of the
package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "checked_score",
    "declined",
    "delegated_from_fileobj",
    "FileBasedModelMixin",
    "no_options",
    "not_text",
    "passthrough_from_fileobj",
    "Raw",
    "register_missing_format",
    "sniffed_raw",
]

from ._base import FileBasedModelMixin
from ._dispatch import register_missing_format
from ._parsers import (
    delegated_from_fileobj,
    not_text,
    passthrough_from_fileobj,
)
from ._raw import Raw
from ._sniffing import checked_score, declined, no_options, sniffed_raw
