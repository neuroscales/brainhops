__all__ = ["PathLike", "Path", "LocalPath", "UPath", "AnyPath"]

from os import PathLike
from pathlib import Path as LocalPath

# optionals
try:
    from upath import UPath
except ImportError:
    UPath = None
try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None

Path = UPath or AnyPath or LocalPath
