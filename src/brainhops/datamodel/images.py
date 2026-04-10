import typing_extensions as _tx

from .struct import SpecializedStruct
from .systems import CoordinateSystem
from .base import Transform


class Image(SpecializedStruct):
    data: object
    coordinateSystem: CoordinateSystem
    coordinateTransforms: _tx.List[Transform] = []