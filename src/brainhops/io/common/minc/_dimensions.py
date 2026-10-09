"""MINC dimensions, and how they are read."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic
from nibabel import minc1 as _minc1

# this format
from ._constants import (
    _DEFAULT_COSINES,
    _MINC2_ROOT,
    SPATIAL_DIMENSIONS,
)


class MincDimension(Magic, frozen=True):
    """
    One dimension of a MINC volume, as its header describes it.

    The attributes the file does not record are `None`; the properties
    give MINC's defaults instead.
    """

    name: str
    """The MINC name of the dimension (`"xspace"`, `"time"`, ...)."""

    length: int
    """The number of samples along the dimension."""

    start: tx.Optional[float] = None
    """The world coordinate of the first sample, along the direction
    cosines (MINC's default is 0)."""

    step: tx.Optional[float] = None
    """The distance between two samples, possibly negative (MINC's
    default is 1)."""

    direction_cosines: tx.Optional[tx.Tuple[float, ...]] = None
    """The world direction of the dimension, for a spatial one (MINC's
    default is its own world axis)."""

    units: tx.Optional[str] = None
    """The unit of `start` and `step`, as written in the file."""

    @property
    def is_spatial(self) -> bool:
        """Whether the dimension is one of `xspace`, `yspace`, `zspace`."""
        return self.name in SPATIAL_DIMENSIONS

    @property
    def origin(self) -> float:
        """`start`, or 0 when the file does not record it."""
        return 0.0 if self.start is None else float(self.start)

    @property
    def spacing(self) -> float:
        """`step`, or 1 when the file does not record it."""
        return 1.0 if self.step is None else float(self.step)

    @property
    def cosines(self) -> tx.Tuple[float, float, float]:
        """`direction_cosines`, or the dimension's own world axis."""
        if self.direction_cosines is not None:
            return tuple(float(c) for c in self.direction_cosines)
        return _DEFAULT_COSINES.get(self.name, (0.0, 0.0, 0.0))


def _read_dimensions(
    container: tx.Any, mfile: _minc1.Minc1File, version: int
) -> tx.Tuple[MincDimension, ...]:
    """The dimensions of the image of an open MINC file (its NetCDF or
    HDF5 container), slowest first."""
    if version == 1:
        image = container.variables["image"]
        names = list(image.dimensions)
        describe = [container.variables[name] for name in names]

        def attr(var: tx.Any, key: str) -> tx.Any:
            return getattr(var, key, None)

    else:
        image = container[_MINC2_ROOT]["image"]["0"]["image"]
        dimorder = _string(image.attrs.get("dimorder")) or ""
        names = dimorder.split(",")[: len(image.shape)]
        group = container[_MINC2_ROOT]["dimensions"]
        describe = [group[name] for name in names]

        def attr(var: tx.Any, key: str) -> tx.Any:
            return var.attrs.get(key, None)

    shape = tuple(int(n) for n in image.shape)
    return tuple(
        MincDimension(
            name=name,
            length=length,
            start=_float(attr(var, "start")),
            step=_float(attr(var, "step")),
            direction_cosines=_floats(attr(var, "direction_cosines")),
            units=_string(attr(var, "units")),
        )
        for name, length, var in zip(names, shape, describe)
    )


def _float(value: tx.Any) -> tx.Optional[float]:
    """A scalar attribute as a float (NetCDF stores it as an array)."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(value[0]) if value.size else None


def _floats(value: tx.Any) -> tx.Optional[tx.Tuple[float, ...]]:
    """A vector attribute as a tuple of floats."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in value) if value.size == 3 else None


def _string(value: tx.Any) -> tx.Optional[str]:
    """A string attribute as a `str`."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else b""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("latin-1")
    return str(value).strip("\x00 ") or None
