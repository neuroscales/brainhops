"""Tests of the method resolution where a format reader meets a dispatcher.

A concrete format such as NiftiImage inherits from its format mixin and
from the registry machinery. If the machinery won without delegating, the
format reader would be skipped silently and NIfTI files would lose their
lazy nibabel handle.
"""

import inspect

import pytest

from brainhops.io.base._base import (
    FileBasedObject,
    FormatDispatcher,
    format_registry,
)
from brainhops.io.base.parsers import (
    BinaryFileReader,
    FileReader,
    FileSniffer,
)

nb = pytest.importorskip("nibabel")

from brainhops.io.common.nifti import NiftiParser  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SpmCoordinatesField,
)

NIFTI_FORMATS = [
    NiftiImage,
    NiftiVoxelToRAS,
    NiftiRASCoordinatesField,
    SpmCoordinatesField,
]

# Methods that NiftiParser specializes; the generic ladder re-dispatches
# into them.
SPECIALIZED = [
    "from_file",
    "from_fileobj",
    "from_bytes",
    "sniff_fileobj",
    "sniff_bytes",
]


def _owner(cls: type, name: str) -> type:
    return next(c for c in cls.__mro__ if name in c.__dict__)


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", SPECIALIZED)
def test_the_format_specific_reader_wins(cls: type, method: str) -> None:
    assert _owner(cls, method) is NiftiParser


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
def test_binary_read_mode_survives_the_diamond(cls: type) -> None:
    """With the text read mode of FileSniffer, NIfTI sniffing would fail."""
    assert cls._READ_MODE == "rb"


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
def test_concrete_formats_are_not_dispatchers(cls: type) -> None:
    """A concrete format with a registry would try every parser, itself too."""
    assert not cls._is_dispatcher()


def test_dispatcher_overrides_are_pass_throughs_for_concrete_formats() -> None:
    """For non-dispatchers, FormatDispatcher overrides defer to super()."""
    overridden = [
        name
        for name, value in vars(FormatDispatcher).items()
        if (name.startswith(("from_", "sniff")) or name == "load")
        and isinstance(value, classmethod)
    ]
    assert overridden, "no reading methods found to check"
    for name in overridden:
        source = inspect.getsource(getattr(FormatDispatcher, name).__func__)
        assert "_is_dispatcher()" in source, name
        assert "super()." + name in source, name


def test_resolution_does_not_depend_on_base_order() -> None:
    @format_registry
    class Root(FileBasedObject):
        pass

    class Special(BinaryFileReader):
        @classmethod
        def from_file(cls, file, **kwargs):  # noqa: ANN001, ANN206
            return "special-reader"

    class SpecialFirst(Special, Root):
        pass

    class RootFirst(Root, Special):
        pass

    assert _owner(SpecialFirst, "from_file") is Special
    assert _owner(RootFirst, "from_file") is FormatDispatcher
    assert SpecialFirst.from_file("x") == "special-reader"
    assert RootFirst.from_file("x") == "special-reader"


def test_the_generic_ladder_redispatches_through_cls() -> None:
    """Ladder rungs call cls.<next>, so subclass overrides are honoured."""
    rungs = {
        (FileSniffer, "sniff_file"): "sniff_fileobj",
        (FileSniffer, "sniff_fileobj"): "sniff_content",
        (FileSniffer, "sniff_text"): "sniff_lines",
        (FileSniffer, "sniff_lines"): "sniff_line",
        (FileReader, "from_file"): "from_fileobj",
        (FileReader, "from_fileobj"): "from_content",
        (FileReader, "from_text"): "from_lines",
        (FileReader, "from_lines"): "from_line",
    }
    for (owner, name), nxt in rungs.items():
        source = inspect.getsource(getattr(owner, name).__func__)
        assert "cls." + nxt in source, (name, nxt)
        assert "super()." + nxt not in source, (name, nxt)


def test_loading_a_nifti_goes_through_the_nifti_reader(tmp_path) -> None:  # noqa: ANN001
    """NiftiParser.from_file keeps the nibabel handle and lazy voxels."""
    import numpy as np

    img = nb.Nifti1Image(np.zeros((3, 4, 5), "float32"), np.eye(4))
    target = tmp_path / "scan.nii"
    nb.save(img, str(target))

    loaded = NiftiImage.from_file(target)
    assert loaded.image is not None, "nibabel handle was not kept"
    assert loaded.shape == (3, 4, 5)


# ----------------------------------------------------------------------
#   THE DATA MODEL A FORMAT REFINES COMES FIRST
# ----------------------------------------------------------------------
# The machinery is itself a generic data model, so a format lists its
# specific model first to keep the generic declarations from shadowing it.


def test_a_nifti_affine_defaults_to_voxel_to_ras() -> None:
    from brainhops.datamodel.systems import RASmm, VoxelCoordinateSystem

    affine = NiftiVoxelToRAS(
        matrix=[[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    )
    assert isinstance(affine.input, VoxelCoordinateSystem)
    assert isinstance(affine.output, RASmm)


def test_a_nifti_coordinate_field_defaults_to_ras() -> None:
    import numpy as np

    from brainhops.datamodel.systems import RASmm

    field = NiftiRASCoordinatesField(field=np.zeros((2, 2, 2, 3)))
    assert isinstance(field.output, RASmm)


def test_an_itk_transform_takes_its_chain_first() -> None:
    from brainhops.datamodel.transformations import Scaling
    from brainhops.io.transformations.itk.tfm import TfmTransform

    chain = [Scaling([1.0, 2.0, 3.0])]
    assert list(TfmTransform(chain).transformations) == chain
