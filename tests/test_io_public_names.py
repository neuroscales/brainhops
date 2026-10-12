"""Public names of the file format packages.

A format package exports its public classes from its `__init__` and keeps
its code in private modules, whose names start with an underscore, so
that the public surface is the list of names in `__all__`. `PACKAGES`
records that list for each package that has been checked. Some packages
still have public modules from before this rule, and the table lists them
until they are made private.
"""

import importlib
import pkgutil

import pytest
import typing_extensions as tx


class Package(tx.NamedTuple):
    """The expected public surface of a package."""

    names: tx.FrozenSet[str]
    """The names that `__all__` lists."""

    public_modules: tx.FrozenSet[str] = frozenset()
    """Modules whose names do not start with an underscore yet."""

    needs: tx.Tuple[str, ...] = ()
    """Optional dependencies without which `__all__` is shorter."""


PACKAGES: tx.Dict[str, Package] = {
    "brainhops.io.metadata": Package(
        names=frozenset({"MetadataFormat"}),
    ),
    "brainhops.io.common.nifti": Package(
        names=frozenset(
            {
                "NiftiMetadata",
                "NiftiRaw",
                "NiftiUnitWarning",
            }
        ),
        needs=("nibabel",),
    ),
    "brainhops.io.images.nifti": Package(
        names=frozenset({"NiftiImage", "NiftiMetadata", "NiftiRaw"}),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.nifti": Package(
        names=frozenset(
            {
                "NiftiBasedTransformation",
                "NiftiMetadata",
                "NiftiRASCoordinatesField",
                "NiftiRASDisplacementField",
                "NiftiRASToVoxel",
                "NiftiRaw",
                "NiftiVoxelToRAS",
            }
        ),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.spm": Package(
        names=frozenset({"SpmCoordinatesField"}),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.fsl.fnirt": Package(
        names=frozenset({"FnirtWarpField"}),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.niftyreg": Package(
        names=frozenset(
            {
                "NiftyRegAffine",
                "NiftyRegControlPointGrid",
                "NiftyRegDeformationField",
                "NiftyRegDisplacementField",
                "NiftyRegField",
                "NiftyRegSequence",
                "NiftyRegVelocity",
                "NiftyRegVelocityField",
                "NiftyRegVelocityGrid",
            }
        ),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.freesurfer.lta": Package(
        names=frozenset(
            {
                "LtaCoordinateSystem",
                "LtaMatrixType",
                "LtaMetadata",
                "LtaPhysicalSystem",
                "LtaRaw",
                "LtaScaledSystem",
                "LtaTransformation",
                "LtaTransformationPhysToPhys",
                "LtaTransformationRASToRAS",
                "LtaTransformationVoxToVox",
                "LtaType",
                "LtaValidity",
                "LtaVoxelSystem",
            }
        ),
    ),
    "brainhops.io.transformations.freesurfer.m3z": Package(
        names=frozenset(
            {
                "GCAM_RAS",
                "GCAM_VOX",
                "M3zFormat",
                "M3zGeometry",
                "M3zMetadata",
                "M3zMorph",
                "M3zRaw",
                "M3zXform",
            }
        ),
    ),
    "brainhops.io.transformations.x5": Package(
        names=frozenset(
            {
                "X5BSplineField",
                "X5CoordinatesField",
                "X5DisplacementField",
                "X5Domain",
                "X5Header",
                "X5Metadata",
                "X5Node",
                "X5Raw",
                "X5Transform",
            }
        ),
        needs=("h5py",),
    ),
    "brainhops.io.transformations.itk": Package(
        names=frozenset(
            {
                "ItkAffineBase",
                "ItkBlockBase",
                "ItkDisplacementBase",
                "ItkPrecision",
                "ItkStruct",
                "ItkTransform",
                "ItkTransformClass",
                "h5",
                "mat",
                "nifti",
                "tfm",
            }
        ),
        needs=("h5py", "nibabel"),
    ),
    "brainhops.io.transformations.itk.h5": Package(
        names=frozenset(
            {"DelayedH5Array", "H5Header", "H5Transform", "H5TransformReader"}
        ),
        needs=("h5py",),
    ),
    "brainhops.io.transformations.itk.mat": Package(
        names=frozenset({"MatTransform", "MatTransformReaderWriter"}),
    ),
    "brainhops.io.transformations.itk.nifti": Package(
        names=frozenset(
            {
                "ItkNiftiCoordinatesField",
                "ItkNiftiDisplacementField",
                "ItkNiftiField",
            }
        ),
        needs=("nibabel",),
    ),
    "brainhops.io.transformations.itk.tfm": Package(
        names=frozenset({"TfmTransform", "TfmTransformReader"}),
    ),
}


def _import(name: str, package: Package) -> tx.Any:
    for dependency in package.needs:
        pytest.importorskip(dependency)
    return importlib.import_module(name)


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_all_lists_the_public_names(name: str) -> None:
    package = PACKAGES[name]
    module = _import(name, package)
    assert len(set(module.__all__)) == len(module.__all__)
    assert set(module.__all__) == package.names


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_no_private_name_is_exported(name: str) -> None:
    module = _import(name, PACKAGES[name])
    assert not [item for item in module.__all__ if item.startswith("_")]


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_the_public_names_can_be_imported(name: str) -> None:
    module = _import(name, PACKAGES[name])
    for item in module.__all__:
        assert getattr(module, item) is not None, item


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_the_modules_are_private(name: str) -> None:
    # A subpackage is public when the package exports it, and a module is
    # public only while the table still lists it.
    package = PACKAGES[name]
    module = _import(name, package)
    public = set()
    for info in pkgutil.iter_modules(module.__path__):
        if info.name.startswith("_"):
            continue
        if info.ispkg:
            assert info.name in module.__all__, info.name
        else:
            public.add(info.name)
    assert public == package.public_modules
