import type_extensions as _tx

from .struct import SpecializedStruct
from .axes import Axis, SpatialAxis
from . import axes as _axes


class CoordinateSystem(SpecializedStruct):
    name: _tx.Optional[str] = None
    axes: _tx.List[Axis] = tuple()


class SpatialCoordinateSystem(CoordinateSystem):
    axes: _tx.List[SpatialAxis] = tuple()


class Spatial3dCoordinateSystem(CoordinateSystem):
    axes: _tx.Tuple[SpatialAxis, SpatialAxis, SpatialAxis] = tuple()


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
        _axes.LeftToRightAxis, 
        _axes.PosteriorToAnteriorAxis, 
        _axes.InferiorToSuperiorAxis
    ] = (_axes.L, _axes.P, _axes.S)

