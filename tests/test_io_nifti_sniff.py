"""Tests of the NIfTI sniffer on NIfTI and other files.

The sniffer checks the magic before nibabel, which would parse MGH, PNG or
LTA files as a header and warn (#258).
"""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.io.common.nifti import NiftiReaderWriter  # noqa: E402

DATA = Path(__file__).parent / "data"

LTA = """\
type      = 1 # LINEAR_RAS_TO_RAS
nxforms   = 1
mean      = 0.0000 0.0000 0.0000
sigma     = 1.0000
1 4 4
+1.000000  +0.000000  +0.000000  +2.000000
+0.000000  +1.000000  +0.000000  -3.000000
+0.000000  +0.000000  +1.000000  +4.000000
+0.000000  +0.000000  +0.000000  +1.000000
src volume info
valid = 1  # volume info valid
filename = orig.mgz
volume = 4 5 6
voxelsize = 1.0 1.0 1.0
xras   = -1.0 0.0 0.0
yras   = 0.0 0.0 -1.0
zras   = 0.0 1.0 0.0
cras   = 0.0 0.0 0.0
dst volume info
valid = 1  # volume info valid
filename = T1.mgz
volume = 4 5 6
voxelsize = 1.0 1.0 1.0
xras   = -1.0 0.0 0.0
yras   = 0.0 0.0 -1.0
zras   = 0.0 1.0 0.0
cras   = 0.0 0.0 0.0
"""


def _volume() -> np.ndarray:
    return np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)


def _mgh(tmp_path: Path, name: str) -> Path:
    file = tmp_path / name
    nb.save(nb.MGHImage(_volume(), np.eye(4)), str(file))
    return file


def _png(tmp_path: Path, name: str) -> Path:
    pil = pytest.importorskip("PIL.Image")
    file = tmp_path / name
    pil.fromarray(np.arange(30, dtype="uint8").reshape(5, 6)).save(file)
    return file


def _lta(tmp_path: Path, name: str) -> Path:
    file = tmp_path / name
    file.write_text(LTA)
    return file


def _tfm(tmp_path: Path, name: str) -> Path:
    return DATA / "itk_affine3d.tfm"


def _no_warnings(func, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    """Call `func` and fail on any UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    caught = [w for w in caught if issubclass(w.category, UserWarning)]
    assert [str(w.message) for w in caught] == []
    return result


# ----------------------------------------------------------------------
#   NOT A NIFTI: DECLINED QUIETLY
# ----------------------------------------------------------------------

NOT_NIFTI = [
    (_mgh, "x.mgh"),
    (_mgh, "x.mgz"),
    (_png, "x.png"),
    (_lta, "x.lta"),
    (_tfm, "x.tfm"),
]
NOT_NIFTI_IDS = ["mgh", "mgz", "png", "lta", "tfm"]


@pytest.mark.parametrize("make, name", NOT_NIFTI, ids=NOT_NIFTI_IDS)
def test_loading_a_non_nifti_emits_no_warning(
    tmp_path: Path,
    make,  # noqa: ANN001
    name: str,
) -> None:
    file = make(tmp_path, name)
    assert _no_warnings(io.load, file) is not None


@pytest.mark.parametrize("make, name", NOT_NIFTI, ids=NOT_NIFTI_IDS)
def test_the_nifti_sniffer_declines_a_non_nifti_quietly(
    tmp_path: Path,
    make,  # noqa: ANN001
    name: str,
) -> None:
    data = make(tmp_path, name).read_bytes()
    assert not _no_warnings(NiftiReaderWriter.sniff_bytes, data)


def test_a_header_sized_without_magic_is_declined() -> None:
    header = bytearray(nb.Nifti1Header().binaryblock)
    header[344:348] = b"xxx\0"
    assert not NiftiReaderWriter.sniff_bytes(bytes(header))
    assert not NiftiReaderWriter.sniff_bytes(gzip.compress(bytes(header)))


# ----------------------------------------------------------------------
#   A NIFTI: STILL SNIFFED AND LOADED
# ----------------------------------------------------------------------

NIFTI = [
    (nb.Nifti1Image, "x.nii"),
    (nb.Nifti1Image, "x.nii.gz"),
    (nb.Nifti2Image, "x.nii"),
    (nb.Nifti2Image, "x.nii.gz"),
]
NIFTI_IDS = ["nii1", "nii1.gz", "nii2", "nii2.gz"]


@pytest.mark.parametrize("cls, name", NIFTI, ids=NIFTI_IDS)
def test_a_nifti_is_still_sniffed_and_loaded(
    tmp_path: Path,
    cls: type,
    name: str,
) -> None:
    file = tmp_path / name
    nb.save(cls(_volume(), np.eye(4)), str(file))
    assert NiftiReaderWriter.sniff_bytes(file.read_bytes())
    image = _no_warnings(io.images.load, file)
    np.testing.assert_array_equal(np.asarray(image.data), _volume())


@pytest.mark.parametrize("cls", [nb.Nifti1Pair, nb.Nifti2Pair])
def test_the_header_of_a_pair_is_sniffed(tmp_path: Path, cls: type) -> None:
    """The .hdr file of a pair carries the ni1 or ni2 magic."""
    file = tmp_path / "x.hdr"
    nb.save(cls(_volume(), np.eye(4)), str(file))
    assert _no_warnings(NiftiReaderWriter.sniff_bytes, file.read_bytes())


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("order", ["<", ">"])
def test_the_magic_is_found_in_either_byte_order(
    version: int, order: str
) -> None:
    cls = {1: nb.Nifti1Header, 2: nb.Nifti2Header}[version]
    header = cls(endianness=order)
    assert NiftiReaderWriter.sniff_bytes(header.binaryblock)
