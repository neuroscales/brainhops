# externals
import typing_extensions as _tx

# internals
from .struct import DataModelBase
from .systems import CoordinateSystem
from .transformations import Transformation


class Image(DataModelBase):
    data: object
    coordinateSystem: CoordinateSystem
    coordinateTransforms: _tx.List[Transform] = ()