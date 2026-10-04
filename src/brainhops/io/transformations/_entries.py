"""
The transformation formats, declared ahead of import.

Dispatch finds a format through its entry here, and imports the format's
module only when it needs the format (see `brainhops.io.base._registry`).
An entry must say what its class says -- its hints, extensions, prefixes,
priority and dispatchers -- which `tests/test_io_registry_entries.py`
checks, and prints the line to write when they disagree.

It imports nothing but the registry, so that importing
`brainhops.io.transformations` imports neither the data model nor any
format.
"""

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import has_abczarr_driver
from brainhops.io.base._registry import (
    OBJECT,
    TRANSFORMATION,
    WRITABLE,
    WRITABLE_TRANSFORMATION,
    FormatEntry,
    declare,
)


def _xform(
    module: str,
    qualname: str,
    hints: str,
    extensions: tx.Tuple[str, ...],
    writable: bool = True,
    **kwargs,
) -> FormatEntry:
    """A transformation format, of a module of
    `brainhops.io.transformations`."""
    if writable:
        dispatchers = (
            WRITABLE_TRANSFORMATION,
            TRANSFORMATION,
            WRITABLE,
            OBJECT,
        )
    else:
        dispatchers = (TRANSFORMATION, OBJECT)
    return FormatEntry(
        f"brainhops.io.transformations.{module}",
        qualname,
        hints,
        extensions,
        dispatchers=dispatchers,
        **kwargs,
    )


_NII = (".nii", ".nii.gz")

# The NiftyReg fields and grids are NIfTI files, read with nibabel.
_NIFTYREG = {"requires": ("nibabel",), "extra": "nibabel"}


declare(
    [
        _xform(
            "elastix._xform",
            "ElastixParameterTransform",
            "elastix elastix.params params transformix transformix.params "
            "xform xform.elastix xform.elastix.params xform.transformix "
            "xform.transformix.params",
            (".txt",),
        ),
        _xform(
            "elastix._xform",
            "ElastixTomlTransform",
            "elastix elastix.toml toml transformix transformix.toml xform "
            "xform.elastix xform.elastix.toml xform.transformix "
            "xform.transformix.toml",
            (".toml",),
        ),
        _xform(
            "freesurfer.lta._xforms",
            "LtaTransformation",
            "affine affine.lta freesurfer freesurfer.lta lta xform "
            "xform.affine xform.affine.lta xform.freesurfer "
            "xform.freesurfer.lta",
            (".lta",),
        ),
        _xform(
            "freesurfer.m3z._xform",
            "M3zMorph",
            "freesurfer freesurfer.m3z m3z xform xform.freesurfer "
            "xform.freesurfer.m3z",
            (".m3z", ".m3d"),
        ),
        _xform(
            "itk.mat._xform",
            "MatTransform",
            "ants ants.mat itk itk.mat mat xform xform.ants xform.ants.mat "
            "xform.itk xform.itk.mat xform.mat",
            (".mat",),
        ),
        _xform(
            "itk.tfm._xform",
            "TfmTransform",
            "ants ants.tfm itk itk.tfm tfm xform xform.ants xform.ants.tfm "
            "xform.itk xform.itk.tfm",
            (".tfm",),
            writable=False,
        ),
        # The ITK formats that need an optional dependency: HDF5 files are
        # read with h5py, and NIfTI fields with nibabel.
        _xform(
            "itk.h5._xform",
            "H5Transform",
            "ants ants.h5 h5 itk itk.h5 xform xform.ants xform.ants.h5 "
            "xform.itk xform.itk.h5",
            (".h5", ".hdf5"),
            writable=False,
            requires=("h5py",),
            extra="itk",
        ),
        _xform(
            "itk.nifti._fields",
            "ItkNiftiDisplacementField",
            "ants ants.displacements displacements itk itk.displacements "
            "nifti nifti.ants nifti.ants.displacements nifti.itk "
            "nifti.itk.displacements xform xform.ants "
            "xform.ants.displacements xform.itk xform.itk.displacements",
            _NII,
            requires=("nibabel",),
            extra="nifti",
        ),
        _xform(
            "itk.nifti._fields",
            "ItkNiftiCoordinatesField",
            "ants ants.coordinates coordinates itk itk.coordinates nifti "
            "nifti.ants nifti.ants.coordinates nifti.itk "
            "nifti.itk.coordinates xform xform.ants xform.ants.coordinates "
            "xform.itk xform.itk.coordinates",
            _NII,
            priority=-1,
            requires=("nibabel",),
            extra="nifti",
        ),
        _xform(
            "matrix._xform",
            "TxtMatrixAffine",
            "affine affine.matrix affine.matrix.txt matrix matrix.txt txt "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.txt "
            "xform.matrix xform.matrix.txt",
            (".txt", ".dat", ".1D"),
            writable=False,
        ),
        _xform(
            "matrix._xform",
            "CsvMatrixAffine",
            "affine affine.matrix affine.matrix.csv matrix matrix.csv csv "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.csv "
            "xform.matrix xform.matrix.csv",
            (".csv",),
            writable=False,
        ),
        _xform(
            "matrix._xform",
            "TsvMatrixAffine",
            "affine affine.matrix affine.matrix.tsv matrix matrix.tsv tsv "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.tsv "
            "xform.matrix xform.matrix.tsv",
            (".tsv",),
            writable=False,
        ),
        _xform(
            "matrix._xform",
            "NpyMatrixAffine",
            "affine affine.matrix affine.matrix.npy matrix matrix.npy npy "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.npy "
            "xform.matrix xform.matrix.npy",
            (".npy",),
            writable=False,
        ),
        _xform(
            "matrix._xform",
            "NpzMatrixAffine",
            "affine affine.matrix affine.matrix.npz matrix matrix.npz npz "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.npz "
            "xform.matrix xform.matrix.npz",
            (".npz",),
            writable=False,
        ),
        _xform(
            "matrix._xform",
            "MatLegacyMatrixAffine",
            "affine affine.matrix affine.matrix.mat mat matrix matrix.mat "
            "xform xform.affine xform.affine.matrix xform.affine.matrix.mat "
            "xform.matrix xform.matrix.mat",
            (".mat",),
            writable=False,
        ),
        # A MATLAB v7.3 file is read with h5py, imported only then.
        _xform(
            "matrix._xform",
            "Mat73MatrixAffine",
            "73 73.mat73 affine affine.matrix affine.matrix.mat "
            "affine.matrix.mat.73 affine.matrix.mat.mat73 mat mat.73 "
            "mat.73.mat73 mat.mat73 mat.mat73.73 mat73 mat73.73 matrix "
            "matrix.mat matrix.mat.73 matrix.mat.mat73 xform xform.affine "
            "xform.affine.matrix xform.affine.matrix.mat "
            "xform.affine.matrix.mat.73 xform.affine.matrix.mat.mat73 "
            "xform.matrix xform.matrix.mat xform.matrix.mat.73 "
            "xform.matrix.mat.mat73",
            (".mat",),
            writable=False,
        ),
        # The NiftyReg affine is plain text.
        _xform(
            "niftyreg._affine",
            "NiftyRegAffine",
            "affine affine.aladin aladin niftyreg niftyreg.aladin txt "
            "txt.aladin xform xform.affine xform.affine.aladin xform.aladin "
            "xform.niftyreg xform.niftyreg.aladin",
            (".txt",),
        ),
        _xform(
            "niftyreg._fields",
            "NiftyRegControlPointGrid",
            "cpp f3d nifti nifti.cpp nifti.f3d niftyreg niftyreg.cpp "
            "niftyreg.f3d xform xform.cpp xform.f3d xform.niftyreg "
            "xform.niftyreg.cpp xform.niftyreg.f3d",
            _NII,
            **_NIFTYREG,
        ),
        _xform(
            "niftyreg._fields",
            "NiftyRegDeformationField",
            "def deformation nifti nifti.def nifti.deformation niftyreg "
            "niftyreg.def niftyreg.deformation xform xform.def "
            "xform.deformation xform.niftyreg xform.niftyreg.def "
            "xform.niftyreg.deformation",
            _NII,
            **_NIFTYREG,
        ),
        _xform(
            "niftyreg._fields",
            "NiftyRegDisplacementField",
            "disp displacement nifti nifti.disp nifti.displacement niftyreg "
            "niftyreg.disp niftyreg.displacement xform xform.disp "
            "xform.displacement xform.niftyreg xform.niftyreg.disp "
            "xform.niftyreg.displacement",
            _NII,
            **_NIFTYREG,
        ),
        _xform(
            "niftyreg._fields",
            "NiftyRegVelocityField",
            "nifti nifti.vel nifti.velocity niftyreg niftyreg.vel "
            "niftyreg.velocity vel velocity xform xform.niftyreg "
            "xform.niftyreg.vel xform.niftyreg.velocity xform.vel "
            "xform.velocity",
            _NII,
            **_NIFTYREG,
        ),
        _xform(
            "niftyreg._fields",
            "NiftyRegVelocityGrid",
            "cpp nifti nifti.cpp nifti.vel nifti.velocity niftyreg "
            "niftyreg.cpp niftyreg.vel niftyreg.velocity vel velocity xform "
            "xform.cpp xform.niftyreg xform.niftyreg.cpp xform.niftyreg.vel "
            "xform.niftyreg.velocity xform.vel xform.velocity",
            _NII,
            **_NIFTYREG,
        ),
        _xform(
            "nifti.affines",
            "NiftiVoxelToRAS",
            "affine affine.nifti nifti xform xform.affine xform.affine.nifti "
            "xform.nifti",
            _NII,
            requires=("nibabel",),
            extra="nifti",
        ),
        _xform(
            "nifti.fields",
            "NiftiRASDisplacementField",
            "displacements nifti nifti.displacements xform "
            "xform.displacements",
            _NII,
            requires=("nibabel",),
            extra="nifti",
        ),
        _xform(
            "nifti.fields",
            "NiftiRASCoordinatesField",
            "coordinates nifti nifti.coordinates xform xform.coordinates",
            _NII,
            requires=("nibabel",),
            extra="nifti",
        ),
        _xform(
            "spm.y",
            "SpmCoordinatesField",
            "nifti nifti.spm spm xform xform.spm",
            _NII,
            prefixes=("y_", "iy_"),
            requires=("nibabel",),
            extra="spm",
        ),
        # FLIRT matrices are plain text, but the `fsl` package imports
        # FNIRT, which needs nibabel, so FLIRT needs it too.
        _xform(
            "fsl.flirt._xform",
            "FlirtTransform",
            "affine affine.flirt flirt fsl fsl.flirt xform xform.affine "
            "xform.affine.flirt xform.flirt xform.fsl xform.fsl.flirt",
            (".mat",),
            writable=False,
            requires=("nibabel",),
            extra="fsl",
        ),
        _xform(
            "fsl.fnirt._base",
            "FnirtWarpField",
            "fnirt fsl fsl.fnirt nifti nifti.fnirt xform xform.fnirt "
            "xform.fsl xform.fsl.fnirt",
            _NII,
            requires=("nibabel",),
            extra="fsl",
        ),
        _xform(
            "x5._xform",
            "X5Transform",
            "bids x5 xform xform.bids xform.x5",
            (".x5",),
            requires=("h5py",),
            extra="x5",
        ),
        # The OME-Zarr field reader needs abczarr and at least one of its
        # backend drivers, as the Zarr image readers do.
        _xform(
            "zarr._xforms",
            "OmeZarrField",
            "ome ome-zarr xform xform.ome xform.ome-zarr zarr zarr.ome "
            "zarr.ome-zarr",
            (".zarr", ".ome.zarr"),
            requires=("abczarr",),
            extra="zarr",
            check=has_abczarr_driver,
        ),
    ]
)
