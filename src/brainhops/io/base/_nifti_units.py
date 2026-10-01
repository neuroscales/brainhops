"""Units in NIfTI headers, and how they map onto brainhops units.

A NIfTI header stores its units in one byte, `xyzt_units`: the spatial
unit in bits 0-2 and the temporal unit in bits 3-5. `nibabel` reads them
back as labels (`header.get_xyzt_units()`) and writes them from labels
(`header.set_xyzt_units(space, time)`):

| Slot  | Code | Label       | brainhops unit             |
| ----- | ---- | ----------- | -------------------------- |
| space | 0    | `"unknown"` | millimetre (see below)     |
| space | 1    | `"meter"`   | [`Meter`][]                |
| space | 2    | `"mm"`      | `MilliMeter`               |
| space | 3    | `"micron"`  | `MicroMeter`               |
| time  | 0    | `"unknown"` | `None` (unspecified)       |
| time  | 8    | `"sec"`     | [`Second`][]               |
| time  | 16   | `"msec"`    | `MilliSecond`              |
| time  | 24   | `"usec"`    | `MicroSecond`              |
| time  | 32   | `"hz"`      | none -- `None`, and a warning |
| time  | 40   | `"ppm"`     | none -- `None`, and a warning |
| time  | 48   | `"rads"`    | none -- `None`, and a warning |

[`nifti_to_unit`][] reads a code, and [`unit_to_nifti`][] writes one. Every
reader and writer of NIfTI headers goes through them, and the policies
below are theirs.

Reading
-------
* An **unknown spatial unit is read as millimetres.** The format calls the
  field "unknown", but every reader in the ecosystem -- FSL, SPM,
  FreeSurfer, AFNI, ITK and the viewers -- treats the affine as
  millimetres, and a header that means metres or microns says so. This is
  a lossy normalisation: such a header is written back with `"mm"`, not
  `"unknown"`.
* An **unknown temporal unit is read as `None`**, an unspecified unit.
  Nothing places an image in space by its temporal unit, and `pixdim[4]`
  is less consistently seconds than the spatial units are millimetres.
* A **temporal code that is not a unit of time** -- `hz`, `ppm` and `rads`
  are frequencies and spectral offsets, stored in the temporal slot --
  has no brainhops unit. It is read as `None`, with a
  [`NiftiUnitWarning`][].

Writing
-------
* `None` (an unspecified unit) is written as `"unknown"`, and so is a
  unit that measures nothing (a name [`Unit`][] did not recognise).
* The **sample is never written**: it says that an axis indexes an array,
  which a NIfTI voxel space always does, and has no code. Asking for one
  raises a `ValueError`.
* A unit of the wrong kind for its slot (a second for the spatial unit)
  raises a `ValueError`.
* A **unit with no NIfTI code** is written as `"unknown"`, with a
  [`NiftiUnitWarning`][]: a unit is metadata, and refusing to write the
  image over it would lose the data along with it. A *spatial* unit can be
  asked for its nearest code instead (`nearest=True`), which is what the
  image writer does: it then rescales the affine by the ratio between the
  two units, so the stored geometry keeps its physical size.
"""

__all__ = [
    "NiftiUnitWarning",
    "NIFTI_SPACE_CODES",
    "NIFTI_TIME_CODES",
    "nifti_to_unit",
    "unit_to_nifti",
]

# stdlib
import warnings
from math import log10

# externals
import typing_extensions as tx

# internals
from brainhops.datamodel import units as _units
from brainhops.datamodel.units import (
    Unit,
    is_physicalunit,
    is_sampleunit,
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
"""The NIfTI code of each spatial label."""

NIFTI_TIME_CODES: tx.Dict[str, int] = {
    "unknown": 0,
    "sec": 8,
    "msec": 16,
    "usec": 24,
    "hz": 32,
    "ppm": 40,
    "rads": 48,
}
"""The NIfTI code of each temporal label."""

_CODES = {"space": NIFTI_SPACE_CODES, "time": NIFTI_TIME_CODES}

# The brainhops unit of each label that has one. `"unknown"` is a policy,
# not a unit, and is handled on its own.
_UNITS = {
    "space": {
        "meter": _units.Meter,
        "mm": _units.MilliMeter,
        "micron": _units.MicroMeter,
    },
    "time": {
        "sec": _units.Second,
        "msec": _units.MilliSecond,
        "usec": _units.MicroSecond,
    },
}

# The label of each brainhops unit that has one, by unit class.
_LABELS = {
    kind: {cls: label for label, cls in table.items()}
    for kind, table in _UNITS.items()
}

_IS_KIND = {"space": is_spaceunit, "time": is_timeunit}

_UNKNOWN = "unknown"


class NiftiUnitWarning(UserWarning):
    """A unit could not be carried between a NIfTI header and brainhops."""


def _label(value: tx.Union[str, int, None], kind: _Kind) -> str:
    """The NIfTI label of a label or a code, in the slot `kind`."""
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
    """
    The brainhops unit of a NIfTI unit label or code.

    Parameters
    ----------
    value : str | int | None
        A label as `nibabel` writes it (`"mm"`, `"sec"`, ...), or the code
        of one. `None` and `""` read as `"unknown"`.
    kind : {"space", "time"}
        The slot of `xyzt_units` the value comes from.

    Returns
    -------
    Unit | None
        The unit. An unknown spatial unit is a millimetre, an unknown
        temporal unit is `None`, and a temporal code that is not a unit of
        time is `None`, with a [`NiftiUnitWarning`][].

    Raises
    ------
    ValueError
        If `value` is not a NIfTI unit of that kind.
    """
    label = _label(value, kind)
    if label == _UNKNOWN:
        return _units.MilliMeter() if kind == "space" else None
    unit = _UNITS[kind].get(label)
    if unit is None:
        warnings.warn(
            f"The NIfTI {kind} unit {label!r} is not a unit brainhops can "
            f"represent, so it is read as unspecified.",
            NiftiUnitWarning,
            stacklevel=2,
        )
        return None
    return unit()


def unit_to_nifti(
    unit: tx.Optional[Unit], kind: _Kind, *, nearest: bool = False
) -> str:
    """
    The NIfTI label a brainhops unit is written under.

    Parameters
    ----------
    unit : Unit | None
        The unit to write.
    kind : {"space", "time"}
        The slot of `xyzt_units` the unit goes to.
    nearest : bool
        For a spatial unit with no NIfTI code, return the label of the
        nearest one NIfTI has (in log scale) rather than `"unknown"`. The
        caller is then responsible for rescaling what is measured in it.

    Returns
    -------
    str
        A label for `nibabel`'s `set_xyzt_units`: the unit's own, the
        nearest one (`nearest=True`, spatial units), or `"unknown"` for an
        unspecified unit or -- with a [`NiftiUnitWarning`][] -- a unit with
        no NIfTI code.

    Raises
    ------
    ValueError
        If `unit` is the sample, which is never written, or a unit of
        another kind than `kind`.
    """
    if unit is None:
        return _UNKNOWN
    if is_sampleunit(unit):
        raise ValueError(
            "The sample unit is never written to a NIfTI header: it says "
            "that an axis indexes an array, which has no NIfTI unit."
        )
    if not is_physicalunit(unit):
        return _UNKNOWN
    if not _IS_KIND[kind](unit):
        raise ValueError(
            f"{unit!r} is not a unit of {kind}, so it cannot be written as "
            f"the {kind} unit of a NIfTI header."
        )
    label = _LABELS[kind].get(type(unit))
    if label is not None:
        return label
    if nearest and kind == "space":
        meters = float(unit.scale)
        return min(
            _LABELS["space"].values(),
            key=lambda label: abs(
                log10(meters) - log10(float(_UNITS["space"][label].scale))
            ),
        )
    warnings.warn(
        f"{unit!r} has no NIfTI {kind} code, so it is written as 'unknown'.",
        NiftiUnitWarning,
        stacklevel=2,
    )
    return _UNKNOWN


def nifti_unit_meters(label: str) -> tx.Optional[float]:
    """The size in metres of a NIfTI spatial label, or `None`."""
    unit = _UNITS["space"].get(label)
    return None if unit is None else float(unit.scale)
