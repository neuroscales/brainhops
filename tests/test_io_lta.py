"""Tests for the coordinate systems of FreeSurfer LTA transforms."""

import pytest

from brainhops.datamodel.systems import FVoxelCoordinateSystem
from brainhops.datamodel.units import SampleUnit, Unit
from brainhops.io.transformations.freesurfer.lta import (
    LTAPhysicalSystem,
    LTAScaledSystem,
    LTAVoxelSystem,
)
from brainhops.io.transformations.freesurfer.lta._enums import LTAType
from brainhops.io.transformations.freesurfer.lta._struct import LTAStruct
from brainhops.io.transformations.freesurfer.lta._xforms import (
    LTATransformation,
)

_SYSTEMS = [
    (LTAVoxelSystem, SampleUnit()),
    (LTAScaledSystem, Unit("mm")),
    (LTAPhysicalSystem, Unit("mm")),
]


@pytest.mark.parametrize("cls, unit", _SYSTEMS)
def test_default_axes_carry_their_unit(cls: type, unit: object) -> None:
    system = cls()
    assert all(axis.unit is unit for axis in system.axes)


@pytest.mark.parametrize("cls, unit", _SYSTEMS)
def test_from_struct_gives_axes_their_unit(cls: type, unit: object) -> None:
    # Regression: the scaled and physical systems passed `units="mm"`,
    # which is no field of theirs, and raised a `TypeError`.
    struct = LTAStruct.SrcVolumeInfo(filename="a.nii")
    system = cls.from_struct(struct)
    assert type(system) is cls
    assert system.name == "a.nii"
    assert all(axis.unit is unit for axis in system.axes)
    orientations = [axis.orientation.value for axis in system.axes]
    assert orientations == [
        "left-to-right",
        "posterior-to-anterior",
        "inferior-to-superior",
    ]


@pytest.mark.parametrize("cls, unit", _SYSTEMS)
def test_from_struct_names_an_anonymous_volume(
    cls: type, unit: object
) -> None:
    # A volume with no file name is named after its role in the transform.
    assert cls.from_struct(LTAStruct.SrcVolumeInfo()).name == "src"
    assert cls.from_struct(LTAStruct.DstVolumeInfo()).name == "dst"
    # Regression: a bare volume has no role either, and `struct.NAME`
    # raised an `AttributeError`. The system keeps its class's name.
    assert cls.from_struct(LTAStruct.VolumeInfo()).name == cls().name


def test_the_voxel_system_is_a_voxel_system() -> None:
    assert isinstance(LTAVoxelSystem(), FVoxelCoordinateSystem)


@pytest.mark.parametrize(
    "type, cls",
    [
        (LTAType.LINEAR_VOX_TO_VOX, LTAVoxelSystem),
        (LTAType.LINEAR_PHYSVOX_TO_PHYSVOX, LTAPhysicalSystem),
    ],
)
def test_an_lta_transform_reads_its_systems_from_its_volumes(
    type: LTAType, cls: type
) -> None:
    struct = LTAStruct(
        type=type,
        src=LTAStruct.SrcVolumeInfo(filename="src.nii"),
        dst=LTAStruct.DstVolumeInfo(filename="dst.nii"),
    )
    transform = LTATransformation(struct=struct)
    assert isinstance(transform.input, cls)
    assert isinstance(transform.output, cls)
    assert (transform.input.name, transform.output.name) == (
        "src.nii",
        "dst.nii",
    )


def test_lta_systems_do_not_hold_the_canonical_axes() -> None:
    from brainhops.datamodel import axes as _axes

    canonical = (_axes.R, _axes.L, _axes.A, _axes.P, _axes.S, _axes.I)
    for cls, _ in _SYSTEMS:
        for system in (cls(), cls.from_struct(LTAStruct.SrcVolumeInfo())):
            for axis in system.axes:
                assert all(axis is not c for c in canonical)
