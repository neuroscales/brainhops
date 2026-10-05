"""Tests for the coordinate systems of FreeSurfer LTA transforms."""

import pytest

from brainhops.datamodel.systems import FVoxelCoordinateSystem
from brainhops.datamodel.units import IndexUnit, Unit
from brainhops.io.transformations.freesurfer.lta import (
    LtaPhysicalSystem,
    LtaScaledSystem,
    LtaVoxelSystem,
)
from brainhops.io.transformations.freesurfer.lta._enums import LtaType
from brainhops.io.transformations.freesurfer.lta._struct import LtaStruct
from brainhops.io.transformations.freesurfer.lta._xforms import (
    LtaTransformation,
)

_SYSTEMS = [
    (LtaVoxelSystem, IndexUnit()),
    (LtaScaledSystem, Unit("mm")),
    (LtaPhysicalSystem, Unit("mm")),
]


@pytest.mark.parametrize("cls, unit", _SYSTEMS)
def test_default_axes_carry_their_unit(cls: type, unit: object) -> None:
    system = cls()
    assert all(axis.unit is unit for axis in system.axes)


@pytest.mark.parametrize("cls, unit", _SYSTEMS)
def test_from_struct_gives_axes_their_unit(cls: type, unit: object) -> None:
    # Regression: the scaled and physical systems passed `units="mm"`,
    # which is no field of theirs, and raised a `TypeError`.
    struct = LtaStruct.SrcVolumeInfo(filename="a.nii")
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
    assert cls.from_struct(LtaStruct.SrcVolumeInfo()).name == "src"
    assert cls.from_struct(LtaStruct.DstVolumeInfo()).name == "dst"
    # Regression: a bare volume has no role either, and `struct.NAME`
    # raised an `AttributeError`. The system keeps its class's name.
    assert cls.from_struct(LtaStruct.VolumeInfo()).name == cls().name


def test_the_voxel_system_is_a_voxel_system() -> None:
    assert isinstance(LtaVoxelSystem(), FVoxelCoordinateSystem)


@pytest.mark.parametrize(
    "type, cls",
    [
        (LtaType.LINEAR_VOX_TO_VOX, LtaVoxelSystem),
        (LtaType.LINEAR_PHYSVOX_TO_PHYSVOX, LtaPhysicalSystem),
    ],
)
def test_an_lta_transform_reads_its_systems_from_its_volumes(
    type: LtaType, cls: type
) -> None:
    struct = LtaStruct(
        type=type,
        src=LtaStruct.SrcVolumeInfo(filename="src.nii"),
        dst=LtaStruct.DstVolumeInfo(filename="dst.nii"),
    )
    transform = LtaTransformation(struct=struct)
    assert isinstance(transform.input, cls)
    assert isinstance(transform.output, cls)
    assert (transform.input.name, transform.output.name) == (
        "src.nii",
        "dst.nii",
    )
