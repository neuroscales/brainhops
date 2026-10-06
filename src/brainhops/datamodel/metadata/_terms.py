"""The structured values of the vocabulary, and the field converters."""

__all__ = ["Channel", "EncodingDirection", "GeneratedBy"]

# stdlib
import math

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo, replace

# internals
from brainhops._core.enum import EnumConverter
from brainhops._core.typing import ArrayLike

from ..base import DataModelBase
from ..enums import SpaceEnum
from ..units import Unit
from ._sentinel import UNSUPPORTED

AXES = "ijk"
"""The voxel axes of a BIDS direction, in order."""


class GeneratedBy(DataModelBase):
    """One entry of BIDS `GeneratedBy`: a program that produced the data."""

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


# The converter of `Channel.color`; above the class, which evaluates it.
def _rgba(value: tx.Any) -> tx.Any:
    """A color as an RGBA hex string: upper case, without `#`, with an
    opaque alpha when it has none."""
    if not isinstance(value, str):
        return value
    value = value.lstrip("#").upper()
    if len(value) == 6:
        value += "FF"
    return value or None


class Channel(DataModelBase):
    """The description of one channel of an image."""

    name: tx.Annotated[tx.Optional[str], tx.Doc("The channel label.")] = None
    color: tx.Annotated[
        tx.Optional[str],
        tx.Doc(
            """
            Display color, as an RGBA hex string. It is held in upper
            case, without `#`, and an RGB string (6 digits) is given an
            opaque alpha (`"0000ff"` is `"0000FFFF"`).
            """
        ),
        ConvertTo(_rgba),
    ] = None
    display_range: tx.Annotated[
        tx.Optional[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
    ] = None
    unit: tx.Annotated[
        tx.Optional[str], tx.Doc("Unit of the channel's values.")
    ] = None


# ----------------------------------------------------------------------
#   THE CONVERTER THAT THE DECLARATION OF `EncodingDirection` USES
# ----------------------------------------------------------------------


def _vector(value: tx.Union[str, ArrayLike]) -> tx.Tuple[float, ...]:
    if isinstance(value, str):
        return _bids_vector(value)
    return tuple(float(v) for v in np.ravel(np.asarray(value, dtype=float)))


# A component this close to 0 or to +-1, once normalised, is snapped to
# it: a direction through a permutation or a flip of the axes, or a
# rotation by a multiple of 90 degrees, is then exactly an axis, and
# compares equal to it with the default (field by field) equality.
_SNAP = 1e-9


def _snapped(value: float) -> float:
    for exact in (0.0, 1.0, -1.0):
        if abs(value - exact) <= _SNAP:
            return exact
    return value


class EncodingDirection(DataModelBase):
    """
    The direction of an encoding axis, such as the phase-encoding or the
    slice-encoding axis, as a unit vector in a named coordinate system.

    When `space` is `None`, the vector is expressed in the voxel axes of
    the image, which is the frame BIDS uses for `PhaseEncodingDirection`.
    In that frame, `EncodingDirection("j-")` is the same direction as
    `EncodingDirection((0, -1, 0))`. A direction that is aligned with a
    voxel axis reads and writes as a BIDS string (see `to_bids`). Any
    other direction, such as an oblique direction obtained after a
    resampling, is kept exactly, and a format that can only store an axis
    reports it as lost.

    Two directions are equal when their vectors and their spaces are
    equal. The vector is normalised on construction, and a component
    within `1e-9` of 0, 1 or -1 is snapped to that value, so that
    `(0, 0, 2)`, `"k"`, and `"k"` mapped through a permutation of the
    axes are the same vector.
    """

    vector: tx.Annotated[
        ArrayLike,
        tx.Doc(
            "The direction: an array-like vector (or a BIDS string, "
            "`'j-'`), stored as a tuple of floats, normalised and with "
            "its near-axis components snapped on construction."
        ),
        ConvertTo(_vector),
    ]
    space: tx.Annotated[
        tx.Optional[tx.Union[SpaceEnum, str]],
        tx.Doc(
            "The coordinate system of `vector`: `None` for the image's "
            "voxel axes, or the label of a world space."
        ),
        ConvertTo(EnumConverter(SpaceEnum)),
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
        self.vector = tuple(_snapped(float(v)) for v in vector / norm)

    def to_bids(self) -> tx.Optional[str]:
        """
        The BIDS string of this direction, such as `"j-"`.

        Returns
        -------
        str or None
            The BIDS string, or `None` when the direction is not along one
            of the first three voxel axes (because it is expressed in a
            world space, or because it is oblique).
        """
        if self.space is not None:
            return None
        vector = np.asarray(self.vector, dtype=float)
        index = int(np.argmax(np.abs(vector)))
        if index >= len(AXES) or not math.isclose(
            abs(vector[index]), 1.0, abs_tol=1e-6
        ):
            return None
        return AXES[index] + ("-" if vector[index] < 0 else "")

    def transform(self, linear: ArrayLike) -> tx.Self:
        """
        The direction after a linear map of its coordinate system.

        Parameters
        ----------
        linear : array-like
            The matrix that maps the old axes to the new ones.

        Returns
        -------
        EncodingDirection
            The mapped direction, normalised, in the same `space`.
        """
        matrix = np.asarray(linear, dtype=float)
        return replace(self, vector=tuple(matrix @ np.asarray(self.vector)))

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


def _is_absent(value: tx.Any) -> bool:
    """Whether a value is absent, so that there is nothing to convert:
    `None` (unknown) or `UNSUPPORTED` (no slot). Every vocabulary
    converter lets an absent value through unchanged."""
    return value is None or value is UNSUPPORTED


class MaybeEnumConverter(EnumConverter):
    """
    The converter of a vocabulary field that has a list of known terms.

    It converts as [`EnumConverter`][brainhops._core.enum.EnumConverter]
    does, and lets `UNSUPPORTED` through, as every vocabulary converter
    does.
    """

    __slots__ = ()

    def __call__(self, value: tx.Any) -> tx.Any:
        """
        Convert a value.

        Parameters
        ----------
        value : Enum, str, None or UNSUPPORTED
            The value to convert.

        Returns
        -------
        Enum, str, None or UNSUPPORTED
            The converted value.

        Raises
        ------
        TypeError
            If `value` is neither a member, a string, `None` nor
            `UNSUPPORTED`.
        """
        if value is UNSUPPORTED:
            return value
        return super().__call__(value)


def unit(value: tx.Any) -> tx.Any:
    """
    Convert the value of a unit field.

    A unit name that the units module parses becomes a `Unit`. A name
    that it cannot parse stays a string, so that the spelling of a file
    survives.

    Parameters
    ----------
    value : Unit, str, None or UNSUPPORTED
        The value to convert.

    Returns
    -------
    Unit, str, None or UNSUPPORTED
        The converted value.

    Raises
    ------
    TypeError
        If `value` is neither a unit nor a string.
    """
    if _is_absent(value) or isinstance(value, Unit):
        return value
    if isinstance(value, str):
        try:
            return Unit(value)
        except ValueError:
            return value
    raise TypeError(f"Expected a Unit or a str, not {type(value).__name__}.")


def dtype(value: tx.Any) -> tx.Any:
    """
    Convert the value of a data type field to a numpy data type in native
    byte order.

    The byte order is a detail of how a file encodes its values, never
    metadata, so it is dropped.

    Parameters
    ----------
    value : dtype-like, None or UNSUPPORTED
        The value to convert.

    Returns
    -------
    numpy.dtype, None or UNSUPPORTED
        The converted value.
    """
    if _is_absent(value):
        return value
    return np.dtype(value).newbyteorder("=")


def direction(value: tx.Any) -> tx.Any:
    """
    Convert the value of an encoding direction field.

    Parameters
    ----------
    value : EncodingDirection, str, array-like, mapping, None or UNSUPPORTED
        A direction, a BIDS string such as `"j-"`, a vector, or a mapping
        with the keys `vector` and `space` (or `Vector` and `Space`, as
        JSON spells them).

    Returns
    -------
    EncodingDirection, None or UNSUPPORTED
        The converted value.

    Raises
    ------
    ValueError
        If the value does not describe a direction.
    """
    if _is_absent(value) or isinstance(value, EncodingDirection):
        return value
    if isinstance(value, tx.Mapping):
        vector = value.get("vector", value.get("Vector"))
        space = value.get("space", value.get("Space"))
        return EncodingDirection(vector, space=space)
    return EncodingDirection(value)


def _bids_vector(value: str) -> tx.Tuple[float, ...]:
    """The unit vector of a BIDS direction (`"i"`, `"j-"`, `"k"`)."""
    axis = value[:-1] if value.endswith("-") else value
    if len(axis) != 1 or axis not in AXES:
        raise ValueError(
            f"A BIDS direction is one of 'i', 'j', 'k', optionally "
            f"followed by '-', not {value!r}."
        )
    vector = [0.0] * len(AXES)
    vector[AXES.index(axis)] = -1.0 if value.endswith("-") else 1.0
    return tuple(vector)
