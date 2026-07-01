import numpy as np

from brainhops.io.transformations.fsl.displacement._xforms import (
    FslDisplacementTransformation,
)

COEF_FILE = "tests/data/fsl_coef.nii.gz"
REF_FILE = "tests/data/fsl_reference.nii.gz"


def test_fsl_voxels():
    # Pick a few interior voxels
    sample_voxels = np.array([
        [20, 20, 20],
        [10, 13, 8],
        [2, 3, 5],
    ], dtype=float)

    tfm = FslDisplacementTransformation.from_(COEF_FILE, REF_FILE)

    displacement_field = tfm.transformations[1]
    # values generaged from fslpy at Jul 1st 2026
    sample_values = np.array(
        [[-0.25221999, -1.26729276,  0.77738353],
         [0.49786049, -0.8663396, -0.70103445],
         [0.82674197,  0.18248677, -1.49884439]])

    for i in range(3):
        val = displacement_field[tuple(sample_voxels[i])]
        for j in range(3):
            assert (abs(val[j] - sample_values[i, j]) < 0.00000001)
