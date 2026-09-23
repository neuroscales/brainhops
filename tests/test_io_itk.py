# stdlib
from pathlib import Path

# dependencies
import numpy as np
import pytest

# internals
from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.transformations import itk

data_dir = Path(__file__).parent / "data"

FILES_H5 = list(data_dir.glob("*.h5"))
FILES_TFM = list(data_dir.glob("*.tfm"))

TFMTransform = io.transformations.itk.tfm.TFMTransform


@pytest.mark.parametrize("filename", FILES_H5)
@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("keep_open", [True, False])
def test_read_h5(filename: str, load: bool, keep_open: bool) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    transform = H5Transform.from_file(filename, load=load, keep_open=keep_open)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


@pytest.mark.parametrize("filename", FILES_TFM)
def test_read_tfm(filename: str) -> None:
    transform = TFMTransform.from_file(filename)
    # trigger conversion
    transforms = transform.transformations  # noqa: F841


# ----------------------------------------------------------------------
#   REGISTRY / DISPATCH
# ----------------------------------------------------------------------
#
# The ITK readers register themselves with the shared parser registry, so
# the generic `io.load` and the scoped `io.transformations.load` both
# resolve an ITK file to its reader without being told the format.


@pytest.mark.parametrize("filename", FILES_TFM)
def test_tfm_is_dispatched(filename: str) -> None:
    assert io.transformations.sniff(filename) is TFMTransform
    assert io.sniff(filename) is TFMTransform
    assert type(io.transformations.load(filename)) is TFMTransform
    assert type(io.load(filename)) is TFMTransform


@pytest.mark.parametrize("filename", FILES_H5)
def test_h5_is_dispatched(filename: str) -> None:
    pytest.importorskip("h5py")
    H5Transform = io.transformations.itk.h5.H5Transform
    assert io.transformations.sniff(filename) is H5Transform
    assert io.sniff(filename) is H5Transform
    assert type(io.transformations.load(filename)) is H5Transform
    assert type(io.load(filename)) is H5Transform


def test_tfm_header_only_is_read_as_empty(tmp_path) -> None:  # noqa: ANN001
    """
    A `.tfm` file that carries only the version header has no transform
    block. `peekable_lines` drops the header comment and then yields its
    end sentinel, which is not a string, so `sniff_line` must return a
    numeric confidence rather than raise or claim the file.
    """
    header_only = tmp_path / "header_only.tfm"
    header_only.write_text("# Insight Transform File V1.0\n")

    assert TFMTransform.sniff_line("") == 0.0
    assert list(TFMTransform.from_file(header_only).transformations) == []
    # The `.tfm` extension still routes it to the ITK reader, which reads
    # it as an empty transform.
    assert io.transformations.sniff(header_only) is TFMTransform
    assert type(io.transformations.load(header_only)) is TFMTransform


# ----------------------------------------------------------------------
#   BLOCKS AS TRANSFORMATIONS
# ----------------------------------------------------------------------
#
# An ITK file is a chain of transform blocks, and each block is itself a
# brainhops transformation: a structured `Sequence` whose children are
# named, lazily evaluated slots. Nothing is converted after parsing.


@pytest.mark.parametrize("filename", FILES_TFM + FILES_H5)
def test_blocks_are_transformations(filename: str) -> None:
    transform = io.transformations.load(filename)
    assert isinstance(transform, xforms.Sequence)
    for block in transform.transformations:
        assert isinstance(block, itk.ITKStruct)
        assert isinstance(block, xforms.Sequence)
        # A block is a non-empty chain, and every child is a
        # transformation in its own right.
        assert len(block) >= 1
        assert all(isinstance(t, xforms.Transformation) for t in block)


def test_affine_block_exposes_named_cached_slots() -> None:
    block = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")[0]
    assert isinstance(block, itk.ITKAffineBase)

    assert np.allclose(block.center, block.fixed_parameters)
    assert isinstance(block.linear, xforms.Linear)
    assert isinstance(block.translation, xforms.Translation)
    assert list(block) == [
        block.recenter,
        block.linear,
        block.uncenter,
        block.translation,
    ]

    # Each slot is computed once and cached.
    assert block.linear is block.linear
    assert block.transformations is block.transformations

    # Assigning a chain overrides the derived one; clearing it restores
    # the derived one.
    block.transformations = [xforms.Identity()]
    assert len(block) == 1
    block.transformations = None
    assert len(block) == 4


def test_versor_rigid_3d_applies_its_translation() -> None:
    """The rotation acts about the center, then the translation applies."""
    block = itk.ITKStruct(
        type=itk.ITKTransformClass.VersorRigid3DTransform,
        precision="double",
        ndim_input=3,
        ndim_output=3,
        parameters=(0.0, 0.0, 0.0, 1.0, 2.0, 3.0),
        fixed_parameters=(4.0, 5.0, 6.0),
    )
    matrix = block.compute().to(xforms.Affine, lossy=True).matrix
    assert np.allclose(np.asarray(matrix)[:, :3], np.eye(3))
    assert np.allclose(np.asarray(matrix)[:, 3], [1.0, 2.0, 3.0])


def test_displacement_blocks_are_lps_to_lps_chains() -> None:
    pytest.importorskip("h5py")
    for name, order, coeff in [
        ("itk_displacement3d.h5", 1, False),
        ("itk_bspline3d.h5", 3, True),
    ]:
        block = io.transformations.load(data_dir / name)[-1]
        assert isinstance(block, itk.ITKDisplacementBase)
        assert list(block) == [
            block.lps2voxel,
            block.displacement,
            block.voxel2lps,
        ]
        assert block.order == order
        assert block.coeff == coeff
        assert block.displacement.field is block.field
        assert block.field.shape[-1] == 3


def test_transform_group_is_gone() -> None:
    """Blocks live in `transformations`, so there is no second list."""
    transform = TFMTransform.from_file(data_dir / "itk_affine3d.tfm")
    assert not hasattr(transform, "transform_group")
