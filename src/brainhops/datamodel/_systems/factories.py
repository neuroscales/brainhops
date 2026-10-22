import typing_extensions as tx
from bagof.magic import Factory

from ..axes import Axis, SpaceAxis
from .base import AXIS

# --- Array ---


def _index_axis(i: int) -> Axis:
    return Axis(f"dim{i}", unit="index")


class AxesFactory(Factory):
    _AxisFactory = tx.Callable[..., AXIS]

    def __init__(
        self, ndim: int, axis_factory: tx.Optional[_AxisFactory] = None
    ) -> None:
        self.ndim = ndim
        self.axis_factory = axis_factory or (lambda *a: Axis())
        super().__init__(self._factory)

    def _factory(self) -> tx.Tuple[AXIS, ...]:
        return tuple(map(self.axis_factory, range(self.ndim)))


class ArrayAxesFactory(AxesFactory):
    def __init__(self, ndim: int) -> None:
        super().__init__(ndim, _index_axis)


# --- Spatial ---


def SpaceAxes(*names) -> tx.Tuple[SpaceAxis, ...]:
    return tuple(map(SpaceAxis, names))


def SpaceArrayAxis(name: str) -> SpaceAxis:
    return SpaceAxis(name, unit="index")


def SpaceArrayAxes(*names) -> tx.Tuple[SpaceAxis, ...]:
    return tuple(map(SpaceArrayAxis, names))


def SpaceAxesFactory(*names) -> Factory:
    return Factory(lambda: SpaceAxes(*names))


def SpaceArrayAxesFactory(*names) -> Factory:
    return Factory(lambda: SpaceArrayAxes(*names))
