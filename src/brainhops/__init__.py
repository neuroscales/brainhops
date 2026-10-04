__all__ = [
    "datamodel",
    "io",
]

# dependencies
import typing_extensions as tx

# internals
from ._core.lazy import lazy_exports as _lazy_exports

# The subpackages are imported on first access, so that `import
# brainhops` (and the command line) does not pay for all of them.
__getattr__, __dir__ = _lazy_exports(
    __name__,
    globals(),
    {
        "datamodel": ".datamodel",
        "io": ".io",
    },
)

if tx.TYPE_CHECKING:
    from . import datamodel, io

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
