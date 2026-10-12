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
from brainhops.io.transformations.freesurfer.lta._raw import LtaRaw
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
def test_from_raw_gives_axes_their_unit(cls: type, unit: object) -> None:
    # Regression: the unit was passed under a wrong keyword (TypeError).
    raw = LtaRaw.SrcVolumeInfo(filename="a.nii")
    system = cls.from_raw(raw)
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
def test_from_raw_names_an_anonymous_volume(cls: type, unit: object) -> None:
    # A volume without a file name is named after its role.
    assert cls.from_raw(LtaRaw.SrcVolumeInfo()).name == "src"
    assert cls.from_raw(LtaRaw.DstVolumeInfo()).name == "dst"
    # A bare volume has no role, so the system keeps its default name.
    assert cls.from_raw(LtaRaw.VolumeInfo()).name == cls().name


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
    raw = LtaRaw(
        type=type,
        src=LtaRaw.SrcVolumeInfo(filename="src.nii"),
        dst=LtaRaw.DstVolumeInfo(filename="dst.nii"),
    )
    transform = LtaTransformation.from_raw(raw)
    assert isinstance(transform.input, cls)
    assert isinstance(transform.output, cls)
    assert (transform.input.name, transform.output.name) == (
        "src.nii",
        "dst.nii",
    )


@pytest.mark.parametrize(
    "name", ["_get_vox2vox", "_get_phys2phys", "_get_ras2ras"]
)
def test_an_unsupported_lta_type_names_the_type(name: str) -> None:
    """A non-linear LTA type is reported by name in every helper."""
    from brainhops.io.transformations.freesurfer.lta import _matrix_utils

    raw = LtaRaw(
        type=LtaType.MNI_TRANSFORM_TYPE,
        affine=LtaRaw.Affine(
            matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        src=LtaRaw.SrcVolumeInfo(),
        dst=LtaRaw.DstVolumeInfo(),
    )
    with pytest.raises(AssertionError, match="MNI_TRANSFORM_TYPE"):
        getattr(_matrix_utils, name)(raw)
