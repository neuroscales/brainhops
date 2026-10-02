"""
Regression tests: io affines accept the `inverse()` signature of
`Transformation`.

`Sequence.inverse` forwards `compute` (and any keyword) to every child,
so an io affine whose override took no argument broke the inverse of
any sequence that held it -- an SPM `y_` field among them.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.transformations.base.affines import (  # noqa: E402
    RASToVoxel,
    VoxelToRAS,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASToVoxel,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SPMCoordinatesField,
)

SHAPE = (4, 5, 6)
VOX2RAS = np.array(
    [
        [2.0, 0.0, 0.0, -4.0],
        [0.0, 1.5, 0.0, 3.0],
        [0.0, 0.0, 1.0, -2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _image():  # noqa: ANN202
    return nb.Nifti1Image(np.zeros(SHAPE, "float32"), VOX2RAS)


def _spm(tmp_path):  # noqa: ANN001, ANN202
    # An SPM `y_` map is a field of absolute RAS coordinates; the
    # identity map is enough to exercise the inverse.
    ijk = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), -1)
    ras = ijk @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]
    img = nb.Nifti1Image(ras[:, :, :, None, :].astype("float32"), VOX2RAS)
    img.header["intent_code"] = 1007  # NIFTI_INTENT_VECTOR
    img.header["intent_name"] = "Mapping"
    path = tmp_path / "y_sub01.nii.gz"
    nb.save(img, str(path))
    return io.transformations.load(path)


@pytest.mark.parametrize("compute", [False, True])
@pytest.mark.parametrize(
    "cls, back", [(NiftiRASToVoxel, VoxelToRAS), (NiftiVoxelToRAS, RASToVoxel)]
)
def test_a_nifti_affine_inverse_takes_compute(cls, back, compute) -> None:  # noqa: ANN001
    img = _image()
    forward = cls(image=img, header=img.header)
    inverse = forward.inverse(compute=compute)
    assert isinstance(inverse, back)
    np.testing.assert_allclose(
        inverse.homogeneous_matrix,
        np.linalg.inv(forward.homogeneous_matrix),
    )


@pytest.mark.parametrize("compute", [False, True])
@pytest.mark.parametrize("cls", [NiftiRASToVoxel, NiftiVoxelToRAS])
def test_a_nifti_affine_with_a_set_matrix_inverts(cls, compute) -> None:  # noqa: ANN001
    # A set matrix takes the `super().inverse()` path, which must
    # receive `compute` too.
    forward = cls(matrix=VOX2RAS[:-1])
    inverse = forward.inverse(compute=compute)
    np.testing.assert_allclose(
        inverse.homogeneous_matrix, np.linalg.inv(VOX2RAS), atol=1e-12
    )


def test_a_sequence_holding_a_nifti_affine_inverts() -> None:
    img = _image()
    seq = xforms.Sequence([NiftiRASToVoxel(image=img, header=img.header)])
    inverse = seq.inverse()
    assert len(inverse) == 1
    assert isinstance(inverse[0], VoxelToRAS)
    np.testing.assert_allclose(inverse[0].homogeneous_matrix, VOX2RAS)


def test_an_spm_field_inverts(tmp_path) -> None:  # noqa: ANN001
    field = _spm(tmp_path)
    assert type(field) is SPMCoordinatesField
    inverse = field.inverse()
    assert len(inverse) == len(field)
    assert inverse.input == field.output
    assert inverse.output == field.input
