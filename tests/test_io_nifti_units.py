"""Tests for the NIfTI <-> brainhops unit converter, and its policies."""

import warnings

import nibabel as nb
import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel import units
from brainhops.datamodel.units import SampleUnit, Unit
from brainhops.io.base._nifti_units import (
    NIFTI_SPACE_CODES,
    NIFTI_TIME_CODES,
    NiftiUnitWarning,
    nifti_to_unit,
    unit_to_nifti,
)

_REPRESENTABLE = [
    ("space", "meter", units.Meter),
    ("space", "mm", units.MilliMeter),
    ("space", "micron", units.MicroMeter),
    ("time", "sec", units.Second),
    ("time", "msec", units.MilliSecond),
    ("time", "usec", units.MicroSecond),
]

# ----------------------------------------------------------------------
#   Every code, both ways
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kind, label, cls", _REPRESENTABLE)
def test_every_representable_code_round_trips(
    kind: str, label: str, cls: type
) -> None:
    codes = NIFTI_SPACE_CODES if kind == "space" else NIFTI_TIME_CODES
    for value in (label, codes[label]):
        unit = nifti_to_unit(value, kind)
        assert type(unit) is cls
        assert unit_to_nifti(unit, kind) == label


def test_an_unknown_spatial_unit_is_read_as_millimetres() -> None:
    # The ecosystem's convention, and a lossy normalisation: it is written
    # back as "mm", not as "unknown".
    for value in ("unknown", 0, None, ""):
        unit = nifti_to_unit(value, "space")
        assert type(unit) is units.MilliMeter
        assert unit_to_nifti(unit, "space") == "mm"


def test_an_unknown_temporal_unit_is_read_as_unspecified() -> None:
    for value in ("unknown", 0, None, ""):
        assert nifti_to_unit(value, "time") is None
    assert unit_to_nifti(None, "time") == "unknown"


@pytest.mark.parametrize("label", ["hz", "ppm", "rads"])
def test_a_temporal_code_that_is_not_a_unit_warns(label: str) -> None:
    for value in (label, NIFTI_TIME_CODES[label]):
        with pytest.warns(NiftiUnitWarning, match=label):
            assert nifti_to_unit(value, "time") is None
    # ... and it does not come back: unspecified is written as unknown.
    assert unit_to_nifti(None, "time") == "unknown"


def test_every_code_is_covered() -> None:
    covered = {(kind, label) for kind, label, _ in _REPRESENTABLE}
    covered |= {("space", "unknown"), ("time", "unknown")}
    covered |= {("time", label) for label in ("hz", "ppm", "rads")}
    every = {("space", label) for label in NIFTI_SPACE_CODES}
    every |= {("time", label) for label in NIFTI_TIME_CODES}
    assert covered == every


@pytest.mark.parametrize(
    "value, kind",
    [("sec", "space"), ("mm", "time"), ("furlong", "space"), (7, "space")],
)
def test_a_value_that_is_not_a_nifti_unit_of_that_kind_raises(
    value: object, kind: str
) -> None:
    with pytest.raises(ValueError, match="is not a NIfTI"):
        nifti_to_unit(value, kind)


# ----------------------------------------------------------------------
#   Writing policies
# ----------------------------------------------------------------------


def test_unspecified_and_nameless_units_are_written_as_unknown() -> None:
    assert unit_to_nifti(None, "space") == "unknown"
    assert unit_to_nifti(Unit("not-a-unit"), "space") == "unknown"


def test_the_sample_is_never_written() -> None:
    for kind in ("space", "time"):
        with pytest.raises(ValueError, match="never written"):
            unit_to_nifti(SampleUnit(), kind)


def test_a_unit_of_the_wrong_kind_raises() -> None:
    with pytest.raises(ValueError, match="not a unit of space"):
        unit_to_nifti(Unit("s"), "space")
    with pytest.raises(ValueError, match="not a unit of time"):
        unit_to_nifti(Unit("mm"), "time")


def test_a_unit_with_no_code_warns_and_is_written_as_unknown() -> None:
    with pytest.warns(NiftiUnitWarning, match="no NIfTI time code"):
        assert unit_to_nifti(Unit("minute"), "time") == "unknown"
    with pytest.warns(NiftiUnitWarning, match="no NIfTI space code"):
        assert unit_to_nifti(Unit("cm"), "space") == "unknown"


@pytest.mark.parametrize(
    "name, label",
    [("cm", "mm"), ("km", "meter"), ("nm", "micron"), ("dm", "meter")],
)
def test_a_spatial_unit_can_be_written_as_the_nearest_code(
    name: str, label: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert unit_to_nifti(Unit(name), "space", nearest=True) == label


# ----------------------------------------------------------------------
#   Through files
# ----------------------------------------------------------------------


def _round_trip(tmp_path, space: str, time: str) -> tuple:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros((2, 3, 4, 2), dtype="float32"), np.eye(4))
    img.header.set_xyzt_units(space, time)
    source = tmp_path / "source.nii"
    nb.save(img, str(source))
    image = io.images.load(source)
    target = tmp_path / "target.nii"
    image.save(target)
    return image, nb.load(str(target)).header.get_xyzt_units()


@pytest.mark.parametrize("space", ["meter", "mm", "micron"])
@pytest.mark.parametrize("time", ["sec", "msec", "usec"])
def test_every_code_survives_a_file_round_trip(
    tmp_path,  # noqa: ANN001
    space: str,
    time: str,
) -> None:
    image, written = _round_trip(tmp_path, space, time)
    world = image.transformations[-1].output
    kinds = {axis.type: axis.unit for axis in world.axes}
    assert unit_to_nifti(kinds["space"], "space") == space
    assert unit_to_nifti(kinds["time"], "time") == time
    assert written == (space, time)


def test_an_unknown_file_is_written_back_in_millimetres(tmp_path) -> None:  # noqa: ANN001
    image, written = _round_trip(tmp_path, "unknown", "unknown")
    world = image.transformations[-1].output
    assert all(
        type(axis.unit) is units.MilliMeter
        for axis in world.axes
        if axis.type == "space"
    )
    assert written == ("mm", "unknown")
