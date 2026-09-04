import pytest
from brainhops.io.transformations.fsl.displacement._xforms import (
    FslDisplacementTransformation,
)


def test_fsl_coef() -> None:
    transformation = FslDisplacementTransformation.from_file(
        "tests/data/fsl_coef.nii.gz"
    )
    assert len(transformation.transformations) == 3
    assert transformation.transformations[1].is_spline_coefficients


def test_fsl_dense() -> None:
    transformation = FslDisplacementTransformation.from_file(
        "tests/data/fsl_field.nii.gz"
    )
    assert len(transformation.transformations) == 3
    assert not transformation.transformations[1].is_spline_coefficients
