"""Units of NIfTI headers and their brainhops equivalents.

A NIfTI header stores its units in the single byte `xyzt_units`: the spatial
unit in bits 0 to 2 and the temporal unit in bits 3 to 5. nibabel reads and
writes them as labels, with `header.get_xyzt_units()` and
`header.set_xyzt_units(space, time)`.

| Slot  | Code | Label       | brainhops unit             |
| ----- | ---- | ----------- | -------------------------- |
| space | 0    | `"unknown"` | millimetre (see below)     |
| space | 1    | `"meter"`   | `Unit("meter")`            |
| space | 2    | `"mm"`      | `Unit("millimeter")`       |
| space | 3    | `"micron"`  | `Unit("micrometer")`       |
| time  | 0    | `"unknown"` | `None` (unspecified)       |
| time  | 8    | `"sec"`     | `Unit("second")`           |
| time  | 16   | `"msec"`    | `Unit("millisecond")`      |
| time  | 24   | `"usec"`    | `Unit("microsecond")`      |
| time  | 32   | `"hz"`      | none -- `None`, and a warning |
| time  | 40   | `"ppm"`     | none -- `None`, and a warning |
| time  | 48   | `"rads"`    | none -- `None`, and a warning |

Every NIfTI reader and writer goes through [`nifti_to_unit`][] and
[`unit_to_nifti`][], which own the policies below.

An unknown spatial unit is read as millimetres. The format calls it unknown,
but FSL, SPM, FreeSurfer, AFNI, ITK and viewers all treat the affine as
millimetres, and a header meaning metres or microns says so. The reading is
lossy, since the unit is written back as `"mm"`. An unknown temporal unit is
read as `None`, because nothing depends on it and `pixdim[4]` is less
consistently in seconds. The temporal codes that are not units of time, `hz`,
`ppm` and `rads`, are read as `None` with a [`NiftiUnitWarning`][].

When writing, `None` becomes `"unknown"`. An index unit, or a unit of the wrong
kind for its slot, raises `ValueError`. A unit without a NIfTI code is written
as `"unknown"` with a warning rather than refused, since the unit is only
metadata. A spatial unit can instead ask for the nearest code, as the image
writer does before rescaling the affine to keep the physical size.
"""

__all__ = [
    "NiftiUnitWarning",
    "NIFTI_SPACE_CODES",
    "NIFTI_TIME_CODES",
    "nifti_to_unit",
    "unit_to_nifti",
]

import warnings
from math import log10

import typing_extensions as tx

from brainhops.datamodel.units import (
    Unit,
    is_indexunit,
    is_physicalunit,
    is_spaceunit,
    is_timeunit,
)

_Kind = tx.Literal["space", "time"]

NIFTI_SPACE_CODES: tx.Dict[str, int] = {
    "unknown": 0,
    "meter": 1,
    "mm": 2,
    "micron": 3,
}
"""The NIfTI code of each spatial unit label."""

NIFTI_TIME_CODES: tx.Dict[str, int] = {
    "unknown": 0,
    "sec": 8,
    "msec": 16,
    "usec": 24,
    "hz": 32,
    "ppm": 40,
    "rads": 48,
}
"""The NIfTI code of each temporal unit label."""

_CODES = {"space": NIFTI_SPACE_CODES, "time": NIFTI_TIME_CODES}

# The brainhops unit name of each label that has one; "unknown" is a
# policy, handled apart. Units are built lazily, since building one may
# import pint.
_UNITS = {
    "space": {
        "meter": "meter",
        "mm": "millimeter",
        "micron": "micrometer",
    },
    "time": {
        "sec": "second",
        "msec": "millisecond",
        "usec": "microsecond",
    },
}

_LABELS = {
    kind: {name: label for label, name in table.items()}
    for kind, table in _UNITS.items()
}

_IS_KIND = {"space": is_spaceunit, "time": is_timeunit}

_UNKNOWN = "unknown"


class NiftiUnitWarning(UserWarning):
    """Warning that a unit cannot be carried between NIfTI and brainhops."""


def _label(value: tx.Union[str, int, None], kind: _Kind) -> str:
    """Return the NIfTI label of a label or code.

    `None` and `""` read as `"unknown"`; anything else unknown raises
    `ValueError`.
    """
    codes = _CODES[kind]
    if value is None or value == "":
        return _UNKNOWN
    if isinstance(value, str):
        if value in codes:
            return value
    else:
        for label, code in codes.items():
            if int(value) == code:
                return label
    raise ValueError(
        f"{value!r} is not a NIfTI {kind} unit. The {kind} units are "
        f"{', '.join(repr(label) for label in codes)}."
    )


def nifti_to_unit(
    value: tx.Union[str, int, None], kind: _Kind
) -> tx.Optional[Unit]:
    """Return the brainhops unit of a NIfTI unit label or code.

    Parameters
    ----------
    value : str, int or None
        A label as nibabel writes it, such as `"mm"` or `"sec"`, or its code.
        `None` and `""` read as `"unknown"`.
    kind : {"space", "time"}
        The slot of `xyzt_units`.

    Returns
    -------
    Unit or None
        The unit. An unknown spatial unit is a millimetre, and an unknown
        temporal unit or a temporal code that is not a time is `None`, with a
        [`NiftiUnitWarning`][] in the latter case.

    Raises
    ------
    ValueError
        If `value` is not a NIfTI unit of that kind.
    """
    label = _label(value, kind)
    if label == _UNKNOWN:
        return Unit("millimeter") if kind == "space" else None
    unit = _UNITS[kind].get(label)
    if unit is None:
        warnings.warn(
            f"The NIfTI {kind} unit {label!r} is not a unit brainhops can "
            f"represent, so it is read as unspecified.",
            NiftiUnitWarning,
            stacklevel=2,
        )
        return None
    return Unit(unit)


def unit_to_nifti(
    unit: tx.Optional[Unit], kind: _Kind, *, nearest: bool = False
) -> str:
    """Return the NIfTI label under which a brainhops unit is written.

    Parameters
    ----------
    unit : Unit or None
        The unit to write.
    kind : {"space", "time"}
        The slot of `xyzt_units`.
    nearest : bool, default=False
        For a spatial unit without a NIfTI code, return the nearest label on a
        logarithmic scale instead of `"unknown"`. The caller must then rescale
        what the unit measures.

    Returns
    -------
    str
        The label for `set_xyzt_units`: the unit's own, the nearest one, or
        `"unknown"` for `None`, for a unit that is not physical, and, with a
        [`NiftiUnitWarning`][], for a unit without a code.

    Raises
    ------
    ValueError
        For an index unit, or a unit of the wrong kind.
    """
    if unit is None:
        return _UNKNOWN
    if is_indexunit(unit):
        raise ValueError(
            "An index unit is never written to a NIfTI header: it says "
            "that an axis indexes an array, which has no NIfTI unit."
        )
    if not is_physicalunit(unit):
        return _UNKNOWN
    if not _IS_KIND[kind](unit):
        raise ValueError(
            f"{unit!r} is not a unit of {kind}, so it cannot be written as "
            f"the {kind} unit of a NIfTI header."
        )
    label = _LABELS[kind].get(unit.name)
    if label is not None:
        return label
    if nearest and kind == "space":
        meters = float(unit.scale)
        return min(
            _LABELS["space"].values(),
            key=lambda label: abs(
                log10(meters)
                - log10(float(Unit(_UNITS["space"][label]).scale))
            ),
        )
    warnings.warn(
        f"{unit!r} has no NIfTI {kind} code, so it is written as 'unknown'.",
        NiftiUnitWarning,
        stacklevel=2,
    )
    return _UNKNOWN


def nifti_unit_meters(label: str) -> tx.Optional[float]:
    """Return the size in metres of a NIfTI spatial label, or `None`."""
    name = _UNITS["space"].get(label)
    return None if name is None else float(Unit(name).scale)
