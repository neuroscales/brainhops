# dependencies
import typing_extensions as tx

# internals
from .base import DataModelBase
from .systems import CoordinateSystem
from .transformations import Transformation


class Image(DataModelBase):
    data: object
    coordinateSystem: CoordinateSystem
    coordinateTransforms: tx.List[Transformation] = ()
