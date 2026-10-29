"""MINC dimensions, and how they are read."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

# this format
from ._constants import (
    DEFAULT_COSINES,
    SPATIAL_DIMENSIONS,
)
from ._netcdf import NetcdfHeader


class MincDimension(Magic, frozen=True):
    """
    One dimension of a MINC volume, as the header describes it.

    Unrecorded attributes are `None`; the properties give the defaults.
    """

    name: str
    """The MINC name of the dimension, such as `"xspace"` or `"time"`."""

    length: int
    """The number of samples."""

    start: tx.Optional[float] = None
    """The world coordinate of the first sample."""

    step: tx.Optional[float] = None
    """The distance between samples, which may be negative."""

    direction_cosines: tx.Optional[tx.Tuple[float, ...]] = None
    """The world direction of a spatial dimension."""

    units: tx.Optional[str] = None
    """The unit of `start` and `step`, as written."""

    @property
    def is_spatial(self) -> bool:
        """Whether the dimension is `xspace`, `yspace` or `zspace`."""
        return self.name in SPATIAL_DIMENSIONS

    @property
    def origin(self) -> float:
        """The `start`, or 0 if it is not recorded."""
        return 0.0 if self.start is None else float(self.start)

    @property
    def spacing(self) -> float:
        """The `step`, or 1 if it is not recorded."""
        return 1.0 if self.step is None else float(self.step)

    @property
    def cosines(self) -> tx.Tuple[float, float, float]:
        """The `direction_cosines`, or else the world axis of the dimension."""
        if self.direction_cosines is not None:
            return tuple(float(c) for c in self.direction_cosines)
        return DEFAULT_COSINES.get(self.name, (0.0, 0.0, 0.0))


def minc1_dimensions(
    header: NetcdfHeader, names: tx.Sequence[str]
) -> tx.Tuple[MincDimension, ...]:
    """Return the dimensions of a MINC1 header, slowest first.

    Each dimension is described by the variable of the same name, and a
    record dimension has the length of the records.
    """
    dimensions = []
    for name in names:
        length = header.dimensions[name] or header.records
        attributes = header.variables[name].attributes
        dimensions.append(_dimension(name, length, attributes.get))
    return tuple(dimensions)


def minc2_dimensions(
    image: tx.Any, group: tx.Any, names: tx.Sequence[str]
) -> tx.Tuple[MincDimension, ...]:
    """Return the dimensions of a MINC2 image dataset, slowest first.

    Each dimension is described by the dataset of the same name in the
    `dimensions` group of the file.
    """
    return tuple(
        _dimension(name, int(length), group[name].attrs.get)
        for name, length in zip(names, image.shape)
    )


def _dimension(
    name: str, length: int, attribute: tx.Callable[[str], tx.Any]
) -> MincDimension:
    """Build a dimension from a function that reads its attributes."""
    return MincDimension(
        name=name,
        length=int(length),
        start=_float(attribute("start")),
        step=_float(attribute("step")),
        direction_cosines=_floats(attribute("direction_cosines")),
        units=string(attribute("units")),
    )


def _float(value: tx.Any) -> tx.Optional[float]:
    """Read a scalar attribute, which NetCDF stores as an array."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(value[0]) if value.size else None


def _floats(value: tx.Any) -> tx.Optional[tx.Tuple[float, ...]]:
    """Read a vector attribute of three floats, or `None`."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in value) if value.size == 3 else None


def string(value: tx.Any) -> tx.Optional[str]:
    """Read a string attribute, or `None` if it is empty."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else b""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("latin-1")
    return str(value).strip("\x00 ") or None
