"""The structured values of the vocabulary, and the field converters."""

__all__ = ["Channel", "EncodingDirection", "GeneratedBy"]

# stdlib
import math

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo

# internals
from ..base import DataModelBase
from ..enums import Space
from ..units import Unit
from ._sentinel import UNSUPPORTED


class GeneratedBy(DataModelBase):
    """One entry of BIDS `GeneratedBy`: a program that made the data."""

    name: tx.Annotated[str, tx.Doc("Name of the program (BIDS `Name`).")]
    version: tx.Annotated[
        tx.Optional[str], tx.Doc("Its version (BIDS `Version`).")
    ] = None
    description: tx.Annotated[
        tx.Optional[str], tx.Doc("What it did (BIDS `Description`).")
    ] = None
    code_url: tx.Annotated[
        tx.Optional[str], tx.Doc("Where its code lives (BIDS `CodeURL`).")
    ] = None


class Channel(DataModelBase):
    """The description of one channel (or volume) of an image."""

    name: tx.Annotated[tx.Optional[str], tx.Doc("The channel label.")] = None
    color: tx.Annotated[
        tx.Optional[str], tx.Doc("Display color, as an RGBA hex string.")
    ] = None
    display_range: tx.Annotated[
        tx.Optional[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
    ] = None
    unit: tx.Annotated[
        tx.Optional[str], tx.Doc("Unit of the channel's values.")
    ] = None


# ----------------------------------------------------------------------
#   CONVERTERS THAT THE DECLARATION OF `EncodingDirection` USES
# ----------------------------------------------------------------------


class _Term:
    """
    The converter of a free-text field with a list of known terms (an
    enum): a known term becomes the enum member, any other string stays
    a string. `bagof` does not do this for a `Union[Enum, str]` by
    itself (a string already satisfies the union).
    """

    def __init__(self, enum: type) -> None:
        self.enum = enum

    def __call__(self, value: tx.Any) -> tx.Any:
        if _passes(value) or isinstance(value, self.enum):
            return value
        if isinstance(value, str):
            try:
                return self.enum(value)
            except ValueError:
                return str(value)
        raise TypeError(
            f"Expected a {self.enum.__name__} or a str, not "
            f"{type(value).__name__}."
        )


def _vector(value: tx.Any) -> tx.Tuple[float, ...]:
    if isinstance(value, str):
        return _bids_vector(value)
    return tuple(float(v) for v in np.ravel(np.asarray(value, dtype=float)))


class EncodingDirection(DataModelBase):
    """
    The direction of an encoding axis (phase or slice encoding): a unit
    vector in a named coordinate system.

    `space=None` means the image's own voxel (array) axes, the frame of
    BIDS `PhaseEncodingDirection`: `EncodingDirection("j-")` is
    `EncodingDirection((0, -1, 0))`, and compares equal to `"j-"`. A
    direction in voxel axes that is aligned with one of them reads and
    writes as a BIDS string (`to_bids()`); any other one (an oblique
    direction, after a resampling) is kept exactly, and the formats that
    can only store an axis report it as lost.
    """

    vector: tx.Annotated[
        tx.Tuple[float, ...],
        tx.Doc(
            "The direction, a unit vector (normalised on construction); "
            "a BIDS string (`'j-'`) is accepted."
        ),
        ConvertTo(_vector),
    ]
    space: tx.Annotated[
        tx.Optional[tx.Union[Space, str]],
        tx.Doc(
            "The coordinate system of `vector`: `None` for the image's "
            "voxel axes, or the label of a world space."
        ),
        ConvertTo(_Term(Space)),
    ] = None

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        vector = np.asarray(self.vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if not vector.size or not norm or not math.isfinite(norm):
            raise ValueError(
                f"A direction is a non-zero vector, not {self.vector!r}."
            )
        if not math.isclose(norm, 1.0, rel_tol=1e-12):
            self.vector = tuple(float(v) for v in vector / norm)

    @classmethod
    def from_bids(cls, value: str) -> "EncodingDirection":
        """The direction of a BIDS string (`"i"`, `"j-"`, `"k"`), in
        voxel axes."""
        return cls(_bids_vector(value))

    def axis(self) -> tx.Optional[tx.Tuple[int, int]]:
        """`(index, sign)` of the voxel axis this direction is aligned
        with, or `None` (another space, or an oblique direction)."""
        if self.space is not None:
            return None
        vector = np.asarray(self.vector, dtype=float)
        index = int(np.argmax(np.abs(vector)))
        if not math.isclose(abs(vector[index]), 1.0, abs_tol=1e-6):
            return None
        return index, (1 if vector[index] > 0 else -1)

    def to_bids(self) -> tx.Optional[str]:
        """The BIDS string of this direction (`"j-"`), or `None` when it
        is not aligned with one of the first three voxel axes."""
        axis = self.axis()
        if axis is None or axis[0] >= len(_AXES):
            return None
        index, sign = axis
        return _AXES[index] + ("-" if sign < 0 else "")

    def transform(self, linear: tx.Any) -> "EncodingDirection":
        """The direction after a linear map of its space (`linear`, a
        matrix from the old axes to the new ones)."""
        matrix = np.asarray(linear, dtype=float)
        vector = matrix @ np.asarray(self.vector, dtype=float)
        return EncodingDirection(tuple(vector), space=self.space)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = EncodingDirection(other)
            except ValueError:
                return False
        if not isinstance(other, EncodingDirection):
            return NotImplemented
        if self.space != other.space or len(self.vector) != len(other.vector):
            return False
        return bool(np.allclose(self.vector, other.vector, atol=1e-9))

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        bids = self.to_bids()
        if bids is not None:
            return f"EncodingDirection({bids!r})"
        vector = tuple(round(v, 6) for v in self.vector)
        if self.space is None:
            return f"EncodingDirection({vector!r})"
        return f"EncodingDirection({vector!r}, space={str(self.space)!r})"


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------


def _passes(value: tx.Any) -> bool:
    """`None` and `UNSUPPORTED` go through every vocabulary converter."""
    return value is None or value is UNSUPPORTED


def _unit(value: tx.Any) -> tx.Any:
    """A unit name the units module parses, as a `Unit`; a name it
    cannot parse stays a string, so that a file's own spelling survives."""
    if _passes(value) or isinstance(value, Unit):
        return value
    if isinstance(value, str):
        try:
            return Unit(value)
        except ValueError:
            return value
    raise TypeError(f"Expected a Unit or a str, not {type(value).__name__}.")


def _dtype(value: tx.Any) -> tx.Any:
    """A numpy data type, in native byte order: the byte order is
    storage encoding, never metadata (M1)."""
    if _passes(value):
        return value
    return np.dtype(value).newbyteorder("=")


def _direction(value: tx.Any) -> tx.Any:
    """The converter of an encoding direction field: a BIDS string, a
    vector, a mapping (`vector`/`space`, or the JSON `Vector`/`Space`)."""
    if _passes(value) or isinstance(value, EncodingDirection):
        return value
    if isinstance(value, tx.Mapping):
        vector = value.get("vector", value.get("Vector"))
        space = value.get("space", value.get("Space"))
        return EncodingDirection(vector, space=space)
    return EncodingDirection(value)


def _bids_vector(value: str) -> tx.Tuple[float, ...]:
    """The unit vector of a BIDS direction (`"i"`, `"j-"`, `"k"`)."""
    axis = value[:-1] if value.endswith("-") else value
    if len(axis) != 1 or axis not in _AXES:
        raise ValueError(
            f"A BIDS direction is one of 'i', 'j', 'k', optionally "
            f"followed by '-', not {value!r}."
        )
    vector = [0.0] * len(_AXES)
    vector[_AXES.index(axis)] = -1.0 if value.endswith("-") else 1.0
    return tuple(vector)


_AXES = "ijk"
