__all__ = [
    "CoordinateSystem",
    "SpatialCoordinateSystem",
    "Spatial3dCoordinateSystem",
    "RASCoordinateSystem",
    "LPSCoordinateSystem"
]
import typing_extensions as _tx

from .struct import DataModelBase
from .axes import Axis, SpatialAxis
from . import axes as _axes


class CoordinateSystem(DataModelBase):
    name: _tx.Optional[str] = None
    axes: _tx.Optional[_tx.List[Axis]] = None


class SpatialCoordinateSystem(CoordinateSystem):
    axes: _tx.Optional[_tx.List[SpatialAxis]] = None


class Spatial3dCoordinateSystem(CoordinateSystem):
    axes: _tx.Optional[_tx.Tuple[SpatialAxis, SpatialAxis, SpatialAxis]] = None


class RASCoordinateSystem(Spatial3dCoordinateSystem):
    name: str = "RAS"
    axes: _tx.Tuple[
        _axes.LeftToRightAxis, 
        _axes.PosteriorToAnteriorAxis, 
        _axes.InferiorToSuperiorAxis
    ] = (_axes.R, _axes.A, _axes.S)


class LPSCoordinateSystem(Spatial3dCoordinateSystem):
    name: str = "LPS"
    axes: _tx.Tuple[
        _axes.RightToLeftAxis, 
        _axes.AnteriorToPosteriorAxis, 
        _axes.SuperiorToInferiorAxis
    ] = (_axes.L, _axes.P, _axes.S)

