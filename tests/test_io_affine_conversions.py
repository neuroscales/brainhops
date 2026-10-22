"""
Conversions between every pair of affine file formats.

Each affine format is written once, from one affine, by its own
converter, or is read from a fixture. Then, for every ordered pair of
formats, the file of the first is read, converted to the second, written,
read again, and compared with the affine that the first file holds. The
comparison maps points: both affines are read between the systems of
the format they started from, with the endpoints bridged, so an ITK file
between LPS spaces and a Convert3D matrix between RAS spaces hold the
same map when their matrices differ by the flip of two axes.

The formats fall into two groups. Most of them map a world to a world,
and convert into each other. A NIfTI affine and a BigDataViewer
registration map voxels to a world, and convert into each other. A
voxel system is placed in a world only by the geometry of a volume, so
a conversion between the two groups is refused, with a reason. A
Neuroglancer transform keeps the systems of what it converts, and takes
the worlds only. A FLIRT matrix, a `register.dat` file and the LTA file of
a map between voxels need the images of their two volumes, which are
given as conversion options and as options of the reader.
"""

import os

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")
pytest.importorskip("h5py")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    LPSmm,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.errors import (  # noqa: E402
    ConversionError,
    LossyConversionError,
)
from brainhops.io.common.bdv._geometry import (  # noqa: E402
    global_system,
    voxel_system,
)
from brainhops.io.transformations.afni import AfniAffine  # noqa: E402
from brainhops.io.transformations.base._conversions import (  # noqa: E402
    affine_between,
)
from brainhops.io.transformations.bdv import (  # noqa: E402
    BdvViewRegistration,
)
from brainhops.io.transformations.c3d import C3dAffine  # noqa: E402
from brainhops.io.transformations.freesurfer.lta import (  # noqa: E402
    LtaTransformation,
    LtaType,
)
from brainhops.io.transformations.freesurfer.regdat import (  # noqa: E402
    RegisterDatTransformation,
    RegisterDatTransformationRASToRAS,
)
from brainhops.io.transformations.fsl.flirt import (  # noqa: E402
    FlirtTransform,
)
from brainhops.io.transformations.itk.h5 import H5Transform  # noqa: E402
from brainhops.io.transformations.itk.mat import MatTransform  # noqa: E402
from brainhops.io.transformations.itk.tfm import TfmTransform  # noqa: E402
from brainhops.io.transformations.minc import (  # noqa: E402
    XfmAffine,
    XfmTransform,
)
from brainhops.io.transformations.mrtrix import (  # noqa: E402
    MrtrixLinearTransform,
)
from brainhops.io.transformations.neuroglancer import (  # noqa: E402
    NeuroglancerTransform,
)
from brainhops.io.transformations.nifti import NiftiVoxelToRAS  # noqa: E402
from brainhops.io.transformations.niftyreg import (  # noqa: E402
    NiftyRegAffine,
)
from brainhops.io.transformations.trakem2 import (  # noqa: E402
    TrakEM2PatchAffine,
)

DATA = os.path.join(os.path.dirname(__file__), "data")

RAS = RASmm()
LPS = LPSmm()
VOXEL = VoxelCoordinateSystem()

MATRIX = np.array(
    [
        [0.9, -0.1, 0.05, 10.0],
        [0.1, 1.1, 0.0, -20.0],
        [0.02, 0.0, 1.2, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""A general affine, with a shear, between RAS spaces."""

VOXEL2RAS = np.array(
    [
        [-2.0, 0.0, 0.125, 30.0],
        [0.0, 2.0, 0.0, -40.0],
        [0.0, -0.125, 2.5, -20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""An affine from voxels to RAS, as a NIfTI header holds one.

Its numbers are those of single precision, in which a NIfTI-1 header
stores them, so that the NIfTI file holds it exactly.
"""

REFERENCE = np.array(
    [
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 2.0, 0.0, -30.0],
        [0.0, 0.0, 3.0, -10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""The voxel-to-RAS affine of the reference image, which is radiological."""

MOVING = np.array(
    [
        [-1.5, 0.0, 0.0, 30.0],
        [0.0, 1.5, 0.0, -20.0],
        [0.0, 0.0, 2.0, -15.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""The voxel-to-RAS affine of the moving image, which is neurological."""


class Format(tx.NamedTuple):
    """An affine format, with how it is written, read and compared."""

    cls: type
    name: str
    """The name of its file."""
    voxels: bool = False
    """Whether it maps voxels to a world, rather than a world to a world."""
    images: bool = False
    """Whether its conversion and its reader take the two images."""
    fixture: tx.Optional[str] = None
    """A file of `tests/data` that it is read from, rather than written."""
    reader: tx.Optional[type] = None
    """The class that reads its file, when it is not `cls`."""


FORMATS = {
    "afni": Format(AfniAffine, "t.aff12.1D"),
    "c3d": Format(C3dAffine, "t.mat"),
    "mrtrix": Format(MrtrixLinearTransform, "t.txt"),
    "niftyreg": Format(NiftyRegAffine, "t.txt"),
    "neuroglancer": Format(NeuroglancerTransform, "t.ngtransform.json"),
    "xfm-affine": Format(XfmAffine, "t.xfm"),
    "xfm-chain": Format(XfmTransform, "t.xfm"),
    "lta": Format(LtaTransformation, "t.lta"),
    # A register.dat file is read in scanner RAS, which its view gives.
    "regdat": Format(
        RegisterDatTransformation,
        "t.dat",
        images=True,
        reader=RegisterDatTransformationRASToRAS,
    ),
    "flirt": Format(FlirtTransform, "t.mat", images=True),
    "itk-mat": Format(
        MatTransform, "t.mat", fixture="itk_affine3d_0GenericAffine.mat"
    ),
    "itk-tfm": Format(TfmTransform, "t.tfm", fixture="itk_affine3d.tfm"),
    "itk-h5": Format(H5Transform, "t.h5", fixture="itk_composite_affine3d.h5"),
    "nifti": Format(NiftiVoxelToRAS, "t.nii.gz", voxels=True),
    "bdv": Format(BdvViewRegistration, "t.xml", voxels=True),
}
"""The writable affine formats, by a short name."""

PAIRS = [(a, b) for a in FORMATS for b in FORMATS]


def _expected(source: str, target: str) -> bool:
    """Whether the conversion from `source` to `target` is exact.

    Formats of the same group convert into each other. A Neuroglancer
    transform keeps the systems of what it converts, but maps the space
    of a data source rather than its voxels, so it takes the worlds only.
    """
    if "neuroglancer" in (source, target):
        return not (FORMATS[source].voxels or FORMATS[target].voxels)
    return FORMATS[source].voxels == FORMATS[target].voxels


@pytest.fixture(scope="module")
def images(tmp_path_factory) -> tx.Dict[str, tx.Any]:  # noqa: ANN001
    """The reference and the moving image, read from small NIfTI files."""
    folder = tmp_path_factory.mktemp("images")
    loaded = {}
    for name, affine, shape in (
        ("reference", REFERENCE, (10, 12, 8)),
        ("moving", MOVING, (9, 11, 7)),
    ):
        filename = str(folder / f"{name}.nii.gz")
        nb.save(nb.Nifti1Image(np.zeros(shape, np.float32), affine), filename)
        loaded[name] = io.load(filename)
    return loaded


def _options(fmt: Format, images: tx.Dict[str, tx.Any]) -> dict:
    return dict(images) if fmt.images else {}


def _read(fmt: Format, filename: str, images: tx.Dict[str, tx.Any]):  # noqa: ANN202
    reader = fmt.cls if fmt.reader is None else fmt.reader
    return reader.load(filename, **_options(fmt, images))


@pytest.fixture(scope="module")
def files(tmp_path_factory, images) -> tx.Dict[str, str]:  # noqa: ANN001
    """One file of each format, read from a fixture or written."""
    folder = tmp_path_factory.mktemp("sources")
    written = {}
    for key, fmt in FORMATS.items():
        if fmt.fixture is not None:
            written[key] = os.path.join(DATA, fmt.fixture)
            continue
        if fmt.voxels:
            source = xforms.Affine(VOXEL2RAS[:-1], input=VOXEL, output=RAS)
        elif fmt.cls is NeuroglancerTransform:
            # A Neuroglancer transform names the dimensions that it maps,
            # so its source has systems that it can name.
            source = xforms.Affine(MATRIX[:-1], input=LPS, output=LPS)
        else:
            source = xforms.Affine(MATRIX[:-1], input=RAS, output=RAS)
        filename = str(folder / f"{key}-{fmt.name}")
        io.save(source.to(fmt.cls, **_options(fmt, images)), filename)
        written[key] = filename
    return written


def _world_map(t: tx.Any, voxels: bool, raw: bool) -> np.ndarray:
    """The map of `t` read between the systems of its group.

    A Neuroglancer file names its dimensions without their orientation,
    so the systems that it reads match any orientation. The map of a
    pair that involves one is therefore compared in the systems of each
    affine, with `raw`, which is what such a file keeps.
    """
    if raw:
        return affine_between(t, t.input, t.output, type(t))
    if voxels:
        return affine_between(t, voxel_system(), global_system(), type(t))
    return affine_between(t, RAS, RAS, type(t))


@pytest.mark.parametrize(("source", "target"), PAIRS)
def test_an_affine_file_converts_to_every_affine_format(
    source: str,
    target: str,
    files: tx.Dict[str, str],
    images: tx.Dict[str, tx.Any],
    tmp_path,  # noqa: ANN001
) -> None:
    src, dst = FORMATS[source], FORMATS[target]
    loaded = _read(src, files[source], images)
    if not _expected(source, target):
        with pytest.raises(ConversionError):
            loaded.to(dst.cls, **_options(dst, images))
        return
    converted = loaded.to(dst.cls, **_options(dst, images))
    assert type(converted) is dst.cls
    filename = str(tmp_path / dst.name)
    io.save(converted, filename)
    reread = _read(dst, filename, images)
    raw = "neuroglancer" in (source, target)
    expected = _world_map(loaded, src.voxels, raw)
    found = _world_map(reread, src.voxels, raw)
    np.testing.assert_allclose(found, expected, rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------
#   REFUSALS AND CHOICES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", [FlirtTransform, RegisterDatTransformation])
def test_a_format_that_needs_the_images_names_them(cls: type) -> None:
    t = xforms.Affine(MATRIX[:-1], input=RAS, output=RAS)
    with pytest.raises(ConversionError, match="reference= and moving="):
        t.to(cls)


def test_an_lta_file_between_voxels_needs_the_images(images) -> None:  # noqa: ANN001
    t = xforms.Affine(MATRIX[:-1], input=VOXEL, output=VOXEL)
    with pytest.raises(ConversionError, match="reference= and moving="):
        t.to(LtaTransformation)
    lta = t.to(LtaTransformation, **images)
    raw = lta.consolidate().metadata.raw
    assert raw.type == LtaType.LINEAR_VOX_TO_VOX
    assert tuple(raw.src.volume) == (10, 12, 8)
    np.testing.assert_allclose(lta.homogeneous_matrix, MATRIX)


def test_an_lta_file_between_worlds_keeps_the_geometries(images) -> None:  # noqa: ANN001
    t = xforms.Affine(MATRIX[:-1], input=LPS, output=LPS)
    lta = t.to(LtaTransformation, **images)
    raw = lta.metadata.raw
    assert raw.type == LtaType.LINEAR_RAS_TO_RAS
    assert tuple(raw.src.volume) == (10, 12, 8)
    assert tuple(raw.dst.volume) == (9, 11, 7)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_allclose(lta.homogeneous_matrix, flip @ MATRIX @ flip)


def test_an_xfm_chain_with_a_field_is_refused() -> None:
    field = xforms.DisplacementField(field=np.zeros((2, 2, 2, 3)))
    with pytest.raises(ConversionError):
        xforms.Sequence([field]).to(XfmTransform)


@pytest.mark.parametrize(
    ("systems", "cls"),
    [
        ((RAS, RAS), C3dAffine),
        ((LPS, LPS), MatTransform),
        ((None, None), C3dAffine),
    ],
)
def test_a_mat_file_is_written_in_the_format_that_keeps_the_endpoints(
    systems: tuple,
    cls: type,
    tmp_path,  # noqa: ANN001
) -> None:
    t = xforms.Affine(MATRIX[:-1], input=systems[0], output=systems[1])
    filename = str(tmp_path / "t.mat")
    io.save(t, filename)
    assert type(cls.load(filename)) is cls


def test_a_mat_file_is_written_as_flirt_given_the_images(
    images: dict,
    tmp_path,  # noqa: ANN001
) -> None:
    t = xforms.Affine(MATRIX[:-1], input=RAS, output=RAS)
    filename = str(tmp_path / "t.mat")
    io.save(t, filename, conversion=images)
    reread = io.load(filename, **images)
    assert type(reread) is FlirtTransform
    np.testing.assert_allclose(reread.homogeneous_matrix, MATRIX, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        pytest.param(
            "t.mat",
            FlirtTransform,
            marks=pytest.mark.xfail(
                reason="io.save ranks the ITK MATLAB file, which keeps the "
                "LPS endpoints of the affine, above the FLIRT matrix that "
                "the images were given for.",
                strict=True,
            ),
        ),
        ("t.dat", RegisterDatTransformationRASToRAS),
        ("t.lta", LtaTransformation),
    ],
)
def test_io_save_gives_the_images_to_the_conversion(
    name: str,
    cls: type,
    images: dict,
    tmp_path,  # noqa: ANN001
) -> None:
    t = xforms.Affine(MATRIX[:-1], input=LPS, output=LPS)
    filename = str(tmp_path / name)
    io.save(t, filename, conversion=images)
    options = {} if cls is LtaTransformation else images
    reread = cls.load(filename, **options)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        reread.homogeneous_matrix, flip @ MATRIX @ flip, atol=1e-9
    )
    if cls is LtaTransformation:
        assert tuple(reread.metadata.raw.src.volume) == (10, 12, 8)


def test_an_xfm_file_is_written_as_an_affine(tmp_path) -> None:  # noqa: ANN001
    t = xforms.Affine(MATRIX[:-1], input=LPS, output=LPS)
    filename = str(tmp_path / "t.xfm")
    io.save(t, filename)
    assert type(io.load(filename)) is XfmAffine


def test_a_nifti_affine_that_single_precision_rounds_is_lossy() -> None:
    t = xforms.Affine(VOXEL2RAS[:-1] * 0.1, input=VOXEL, output=RAS)
    with pytest.raises(LossyConversionError):
        t.to(NiftiVoxelToRAS)
    affine = t.to(NiftiVoxelToRAS, lossy=True)
    np.testing.assert_allclose(affine.matrix, t.matrix)


def test_a_trakem2_affine_converts_from_a_2d_affine() -> None:
    t = xforms.Affine([[0.0, -1.0, 5.0], [1.0, 0.0, 2.0]])
    affine = t.to(TrakEM2PatchAffine)
    np.testing.assert_allclose(affine.matrix, t.matrix)
    with pytest.raises(ConversionError):
        xforms.Affine(MATRIX[:-1]).to(TrakEM2PatchAffine)
