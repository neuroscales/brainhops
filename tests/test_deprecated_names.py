"""The old spellings of the coordinate fields still import (#547)."""

import importlib

import pytest

OLD_NAMES = [
    ("brainhops.datamodel.transformations", "CoordinatesField"),
    ("brainhops.datamodel.transformations", "InverseCoordinatesField"),
    ("brainhops.io.transformations.base.fields", "RASCoordinatesField"),
    ("brainhops.io.transformations.base.fields", "LPSCoordinatesField"),
    ("brainhops.io.transformations.nifti", "NiftiRASCoordinatesField"),
    ("brainhops.io.transformations.itk.nifti", "ItkNiftiCoordinatesField"),
    ("brainhops.io.transformations.mrtrix", "MrtrixCoordinatesField"),
    ("brainhops.io.transformations.spm", "SpmCoordinatesField"),
    ("brainhops.io.transformations.x5", "X5CoordinatesField"),
]


@pytest.mark.parametrize("module, old", OLD_NAMES)
def test_old_name_warns_and_returns_the_new_class(module, old) -> None:  # noqa: ANN001
    """Each old name returns the renamed class with a warning."""
    mod = importlib.import_module(module)
    new = old.replace("CoordinatesField", "CoordinateField")
    with pytest.warns(DeprecationWarning, match=new):
        cls = getattr(mod, old)
    assert cls is getattr(mod, new)


def test_unknown_name_still_raises() -> None:
    """A name that was never defined raises an AttributeError."""
    mod = importlib.import_module("brainhops.io.transformations.mrtrix")
    with pytest.raises(AttributeError):
        mod.NoSuchField  # noqa: B018
