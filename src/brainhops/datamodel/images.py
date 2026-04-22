import typing_extensions as _tx

from .struct import DataModelBase
from .systems import CoordinateSystem
from .transforms import Transform


class Image(DataModelBase):
    data: object
    coordinateSystem: CoordinateSystem
    coordinateTransforms: _tx.List[Transform] = ()