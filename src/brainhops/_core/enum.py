__all__ = ["StrEnum", "IntEnum", "Enum"]

import sys

if sys.version_info.minor >= 11:
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

from enum import Enum, IntEnum
