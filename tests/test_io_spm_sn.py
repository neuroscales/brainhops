"""Tests for SPM normalisation files (`*_sn.mat`).

No file written by SPM ships with the tests, so files are written by
`_save`, with `scipy.io.savemat`, in the layout that `spm_normalise`
saves, including the MATLAB `nifti` object in the `private` field of an
`spm_vol` structure. The map is checked against `_spm_sn2def`, a literal
transcription of `get_sn2def` in SPM12's `spm_deformations.m` (r7700)
and of `spm_dctmtx.m` (r6416), with MATLAB's 1-based voxels and
column-major reshapes, which is independent of the reader.
"""

import io as _io
import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import typing_extensions as tx
from bagof.magic import fields, replace
from scipy.io.matlab import MatlabObject

from brainhops import io
from brainhops.backends import backend
from brainhops.datamodel import transformations as xforms
from brainhops.datamodel.metadata import Metadata
from brainhops.errors import ConversionError
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserNotImplementedError,
    WriterError,
    _overrides_from_fileobj,
)
from brainhops.io.metadata import MetadataFormat
from brainhops.io.transformations import TransformationFormat
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.spm import (
    SpmCoordinatesField,
    SpmSnMetadata,
    SpmSnRaw,
    SpmSnTransformation,
    SpmVolume,
)
from brainhops.io.transformations.spm._sn import dct_basis

nb = pytest.importorskip("nibabel")

DIM = (11, 13, 9)
"""Shape of the template."""

# The template is an MNI template in SPM's convention: 2 mm voxels, x from
# right to left, and voxel (46, 64, 37), numbered from 1, at the origin.
VG_MAT = np.array(
    [
        [-2.0, 0.0, 0.0, 92.0],
        [0.0, 2.0, 0.0, -128.0],
        [0.0, 0.0, 2.0, -74.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# The source image is oblique, with 1 x 1.2 x 1.5 mm voxels.
VF_MAT = np.array(
    [
        [0.98, -0.1, 0.05, -90.0],
        [0.12, 1.15, -0.2, -110.0],
        [-0.03, 0.25, 1.45, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# From template voxels to source voxels, both numbered from 1.
AFFINE = np.array(
    [
        [1.9, 0.08, -0.05, 4.5],
        [-0.06, 1.7, 0.11, 12.25],
        [0.02, -0.09, 1.3, 7.75],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

COUNTS = (3, 4, 2)
"""Number of cosines along each axis."""


def _coefficients(seed: int = 0) -> np.ndarray:
    # Coefficients of a warp of a few voxels, as SPM's are.
    return np.random.default_rng(seed).normal(0.0, 3.0, (*COUNTS, 3))


# ----------------------------------------------------------------------
#   WRITING, AS SPM DOES
# ----------------------------------------------------------------------


def _nifti_object(dim: tx.Sequence[int]) -> MatlabObject:
    """Imitate the `nifti` object of the `private` field of SPM5 and later."""
    hdr = np.zeros((1, 1), dtype=[("sizeof_hdr", "O"), ("dim", "O")])
    hdr[0, 0]["sizeof_hdr"] = np.array([[348.0]])
    hdr[0, 0]["dim"] = np.array([[3.0, *dim, 1.0, 1.0, 1.0, 1.0]])
    obj = np.zeros((1, 1), dtype=[("hdr", "O"), ("extras", "O")])
    obj[0, 0]["hdr"] = hdr
    obj[0, 0]["extras"] = np.zeros((0, 0))
    return MatlabObject(obj, "nifti")


def _volume(fname: str, dim: tx.Sequence[int], mat: np.ndarray) -> dict:
    """Return an `spm_vol` structure of SPM5 and later (`spm_vol_nifti`)."""
    return {
        "fname": fname,
        "dim": np.array([dim], dtype=float),
        "dt": np.array([[16.0, 0.0]]),
        "pinfo": np.array([[1.0], [0.0], [352.0]]),
        "mat": mat,
        "n": np.array([[1.0, 1.0]]),
        "descrip": "SPM template",
        "private": _nifti_object(dim),
    }


def _spm2_volume(fname: str, dim: tx.Sequence[int], mat: np.ndarray) -> dict:
    """Return an `spm_vol` structure of SPM2 (`spm_vol_ana`).

    Its `dim` has a fourth element, the data type, and its fields come in
    another order.
    """
    return {
        "fname": fname,
        "dim": np.array([[*dim, 16.0]]),
        "mat": mat,
        "pinfo": np.array([[1.0], [0.0], [0.0]]),
        "descrip": "",
        "private": {"hdr": {"sizeof_hdr": 348.0}},
    }


def _flags() -> dict:
    return {
        "smosrc": 8.0,
        "smoref": 0.0,
        "regtype": "mni",
        "cutoff": 25.0,
        "nits": 16.0,
        "reg": 1.0,
        "graphics": 1.0,
        "version": "$Rev: 5745 $",
        "date": "13-Oct-2026",
        "weight": np.zeros((0, 0)),
        "nested": {"ignored": 1.0},
    }


def _variables(
    tr: tx.Optional[np.ndarray] = None, volume: tx.Callable = _volume
) -> dict:
    return {
        "Affine": AFFINE,
        "Tr": _coefficients() if tr is None else tr,
        "VF": volume("/data/sub-01_T1w.nii", (40, 48, 36), VF_MAT),
        "VG": volume("/spm/templates/T1.nii", DIM, VG_MAT),
        "flags": _flags(),
    }


def _save(path: Path, **variables: tx.Any) -> Path:
    """Save variables as `spm_normalise` saves them (`-V6`)."""
    scipy.io.savemat(str(path), variables or _variables(), format="5")
    return path


# ----------------------------------------------------------------------
#   THE MAP, AS SPM COMPUTES IT
# ----------------------------------------------------------------------


def _spm_dctmtx(N: int, K: int, n: np.ndarray) -> np.ndarray:
    """Transcribe `C = spm_dctmtx(N, K, n)`."""
    n = np.asarray(n, dtype=float).reshape(-1)
    C = np.zeros((n.size, K))
    C[:, 0] = np.ones(n.size) / np.sqrt(N)
    for k in range(2, K + 1):
        C[:, k - 1] = np.sqrt(2 / N) * np.cos(
            np.pi * (2 * n + 1) * (k - 1) / (2 * N)
        )
    return C


def _spm_sn2def(sn: dict) -> np.ndarray:
    """Transcribe `[Def, mat] = get_sn2def(job)` with NaN `vox` and `bb`.

    Voxels are numbered from 1 and arrays are reshaped in column-major
    order, as in MATLAB. The result is `Def`, of shape `(X, Y, Z, 3)`.
    """
    dim = sn["VG"]["dim"][0][:3].astype(int)
    x = np.arange(1, dim[0] + 1)
    y = np.arange(1, dim[1] + 1)
    z = np.arange(1, dim[2] + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Tr = sn["Tr"]
    st = Tr.shape
    affine_only = Tr.size == 0
    if not affine_only:
        basX = _spm_dctmtx(dim[0], st[0], x - 1)
        basY = _spm_dctmtx(dim[1], st[1], y - 1)
        basZ = _spm_dctmtx(dim[2], st[2], z - 1)
    Def = np.zeros((x.size, y.size, z.size, 3))
    Mult = sn["VF"]["mat"] @ sn["Affine"]
    for j in range(z.size):
        if affine_only:
            X1, Y1, Z1 = X, Y, z[j]
        else:

            def plane(c: int, j: int = j) -> np.ndarray:
                coeffs = Tr[:, :, :, c].reshape(
                    st[0] * st[1], st[2], order="F"
                )
                return (coeffs @ basZ[j, :]).reshape(st[0], st[1], order="F")

            X1 = X + basX @ plane(0) @ basY.T
            Y1 = Y + basX @ plane(1) @ basY.T
            Z1 = z[j] + basX @ plane(2) @ basY.T
        for c in range(3):
            Def[:, :, j, c] = (
                Mult[c, 0] * X1
                + Mult[c, 1] * Y1
                + Mult[c, 2] * Z1
                + Mult[c, 3]
            )
    return Def


def _template_nifti_affine() -> np.ndarray:
    # `decode_qform0` reads a NIfTI affine M as mat = M * [eye(4,3) -1].
    return VG_MAT @ np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1.0]]
    )


# ----------------------------------------------------------------------
#   THE CONVENTIONS
# ----------------------------------------------------------------------


def test_the_basis_is_spm_dctmtx() -> None:
    for size, count in ((9, 4), (13, 13), (5, 1)):
        expected = _spm_dctmtx(size, count, np.arange(size))
        assert np.allclose(dct_basis(size, count), expected, atol=1e-15)
        # The basis is orthonormal over the grid.
        basis = dct_basis(size, count)
        assert np.allclose(basis.T @ basis, np.eye(count))


def test_the_field_is_what_spm_computes(tmp_path: Path) -> None:
    path = _save(tmp_path / "sub-01_T1w_sn.mat")
    sn = SpmSnTransformation.load(path)
    expected = _spm_sn2def(_variables())
    # The deformation field that SPM writes holds Def on the template grid.
    y = sn.to(SpmCoordinatesField)
    assert np.allclose(np.asarray(y.rasfield.values), expected, atol=1e-9)
    assert np.allclose(
        np.linalg.inv(y.ras2voxel.homogeneous_matrix),
        _template_nifti_affine(),
    )
    # The field holds the template voxels moved by the warp, numbered from
    # 0, and the last affine carries them to the source.
    ras2grid, field, grid2source = sn.transformations
    to_source = (
        VF_MAT @ AFFINE @ np.linalg.inv(VG_MAT) @ (_template_nifti_affine())
    )
    assert np.allclose(grid2source.homogeneous_matrix, to_source)
    moved = np.asarray(field.field)
    assert np.allclose(
        np.asarray(grid2source.homogeneous_matrix[:3, :3] @ moved[..., None])[
            ..., 0
        ]
        + grid2source.homogeneous_matrix[:3, 3],
        expected,
        atol=1e-9,
    )


def test_points_are_mapped_through_the_chain(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    expected = _spm_sn2def(_variables())
    # A point at the centre of template voxel (3, 5, 2), numbered from 1,
    # maps to Def at that voxel.
    voxel = np.array([3.0, 5.0, 2.0, 1.0])
    point = (VG_MAT @ voxel)[:3]
    mapped = sn.compute()(
        xforms.CoordinatesField(field=point[None, None, None, :]),
        compute=True,
    )
    assert np.allclose(np.asarray(mapped.field)[0, 0, 0], expected[2, 4, 1])
    # Halfway between two voxels, the field is interpolated linearly.
    between = (VG_MAT @ (voxel + [0.5, 0.0, 0.0, 0.0]))[:3]
    mapped = sn.compute()(
        xforms.CoordinatesField(field=between[None, None, None, :]),
        compute=True,
    )
    middle = (expected[2, 4, 1] + expected[3, 4, 1]) / 2
    assert np.allclose(np.asarray(mapped.field)[0, 0, 0], middle)


def test_an_affine_normalisation_has_no_field(tmp_path: Path) -> None:
    variables = _variables(tr=np.zeros((0, 0)))
    sn = SpmSnTransformation.load(_save(tmp_path / "a_sn.mat", **variables))
    assert sn.metadata.raw.affine_only
    ras2grid, grid2source = sn.transformations
    assert isinstance(ras2grid, RASToVoxel)
    assert isinstance(grid2source, VoxelToRAS)
    # The two affines compose into VF.mat * Affine * inv(VG.mat).
    matrix = sn.to(xforms.Affine).homogeneous_matrix
    assert np.allclose(matrix, VF_MAT @ AFFINE @ np.linalg.inv(VG_MAT))
    expected = _spm_sn2def(variables)
    grid = np.stack(np.meshgrid(*map(np.arange, DIM), indexing="ij"), -1)
    world = grid @ _template_nifti_affine()[:3, :3].T
    world += _template_nifti_affine()[:3, 3]
    assert np.allclose(world @ matrix[:3, :3].T + matrix[:3, 3], expected)


def test_spm2_volumes_are_read(tmp_path: Path) -> None:
    # SPM2 reads Analyze headers: `dim` has a fourth element, the data type.
    variables = _variables(volume=_spm2_volume)
    sn = SpmSnTransformation.load(_save(tmp_path / "spm2_sn.mat", **variables))
    assert sn.metadata.raw.template.dim == DIM
    y = sn.to(SpmCoordinatesField)
    expected = _spm_sn2def(_variables())
    assert np.allclose(np.asarray(y.rasfield.values), expected, atol=1e-9)


# ----------------------------------------------------------------------
#   THE RECORD
# ----------------------------------------------------------------------


def test_the_record_holds_the_file(tmp_path: Path) -> None:
    record = SpmSnMetadata.load(_save(tmp_path / "s_sn.mat")).raw
    assert isinstance(record, SpmSnRaw)
    assert np.array_equal(record.affine, AFFINE)
    assert np.array_equal(record.tr, _coefficients())
    assert record.template == record.vg[0]
    assert isinstance(record.template, SpmVolume)
    assert record.template.dim == DIM
    assert record.template.fname == "/spm/templates/T1.nii"
    assert record.template.descrip == "SPM template"
    assert np.array_equal(record.template.mat, VG_MAT)
    assert np.allclose(record.template.vox2world, _template_nifti_affine())
    assert record.source.dim == (40, 48, 36)
    assert np.array_equal(record.source.mat, VF_MAT)
    # Text and numbers are kept, and other fields are dropped.
    assert record.flags["regtype"] == "mni"
    assert record.flags["cutoff"] == 25.0
    assert record.flags["date"] == "13-Oct-2026"
    assert "nested" not in record.flags
    for array in (record.affine, record.tr, record.template.mat):
        assert not array.flags.writeable


def test_several_templates_are_kept_in_order(tmp_path: Path) -> None:
    variables = _variables()
    other = _volume("/spm/templates/T2.nii", DIM, VG_MAT)
    # A 1x2 structure array, which MATLAB stores as one structure with
    # two elements, rather than as a cell array.
    structs = np.zeros((1, 2), dtype=[(name, "O") for name in other])
    for i, volume in enumerate((variables["VG"], other)):
        for name in other:
            structs[0, i][name] = volume[name]
    variables["VG"] = structs
    record = SpmSnRaw.load(_save(tmp_path / "multi_sn.mat", **variables))
    assert [v.fname for v in record.vg] == [
        "/spm/templates/T1.nii",
        "/spm/templates/T2.nii",
    ]


def test_the_metadata_pickles(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    clone = pickle.loads(pickle.dumps(sn.metadata))
    assert type(clone) is SpmSnMetadata
    assert np.array_equal(clone.raw.tr, sn.metadata.raw.tr)
    assert clone.raw.template.fname == sn.metadata.raw.template.fname
    assert set(clone.raw.flags) == set(sn.metadata.raw.flags)
    assert clone.raw.flags["regtype"] == "mni"


def test_a_copy_of_the_record_shares_its_arrays(tmp_path: Path) -> None:
    record = SpmSnRaw.load(_save(tmp_path / "s_sn.mat"))
    copy = record.copy()
    assert copy is not record
    assert copy.tr is record.tr
    assert copy.vg is record.vg


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_chain_is_decoded_lazily(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    assert "_cache_transformations" not in vars(sn)
    assert sn._transformations is None
    chain = sn.transformations
    assert sn.transformations is chain
    assert len(chain) == 3
    field = chain[1]
    assert type(field) is xforms.CoordinatesField
    assert field.degree == 1
    assert field.bound == "nearest"
    # With NumPy, the decoded positions are read-only.
    with backend("numpy"):
        sn = SpmSnTransformation.load(tmp_path / "s_sn.mat")
        data = sn.transformations[1].data
    assert isinstance(data, np.ndarray)
    assert not data.flags.writeable


def test_with_dask_the_field_stays_lazy(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    path = _save(tmp_path / "s_sn.mat")
    with backend("dask"):
        sn = SpmSnTransformation.load(path)
        data = sn.transformations[1].data
    assert isinstance(data, da.Array)
    ras2grid, _, grid2source = sn.transformations
    moved = np.asarray(data.compute())
    world = moved @ grid2source.homogeneous_matrix[:3, :3].T
    world += grid2source.homogeneous_matrix[:3, 3]
    assert np.allclose(world, _spm_sn2def(_variables()), atol=1e-9)


def test_a_normalisation_reads_a_stream_bytes_and_a_record(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "s_sn.mat")
    content = path.read_bytes()
    with open(path, "rb") as file:
        streamed = SpmSnTransformation.from_fileobj(file)
    built = (
        streamed,
        SpmSnTransformation.from_bytes(content),
        SpmSnTransformation.from_raw(streamed.metadata.raw),
        SpmSnTransformation.from_any(streamed.metadata.raw),
    )
    expected = np.asarray(streamed.transformations[1].data)
    for sn in built:
        assert np.array_equal(np.asarray(sn.transformations[1].data), expected)
    assert SpmSnMetadata.from_bytes(content).raw.template.dim == DIM


@pytest.mark.parametrize(
    "name", ["s_sn.mat", "s_seg_sn.mat", "s_seg_inv_sn.mat", "params.mat"]
)
def test_the_file_is_recognised_by_its_content(
    tmp_path: Path, name: str
) -> None:
    # Any MATLAB file that holds the four variables is one, whatever its
    # name, and the readers of plain matrices do not take it.
    path = _save(tmp_path / name)
    assert type(io.load(path)) is SpmSnTransformation
    assert type(TransformationFormat.load(path)) is SpmSnTransformation
    assert MetadataFormat.sniff(path) is SpmSnMetadata
    assert type(io.load(path, hint="sn")) is SpmSnTransformation


def test_other_mat_files_are_not_normalisations(tmp_path: Path) -> None:
    path = tmp_path / "affine.mat"
    scipy.io.savemat(str(path), {"Affine": AFFINE}, format="5")
    assert SpmSnTransformation.sniff(path) == 0
    with pytest.raises(ParserContentError):
        SpmSnTransformation.load(path)


@pytest.mark.parametrize(
    "variables, message",
    [
        ({"Tr": np.zeros((3, 4, 2))}, "Kx, Ky, Kz, 3"),
        ({"Affine": np.eye(3)}, "4x4"),
        ({"VG": {"fname": "T1.nii"}}, "spm_vol"),
    ],
)
def test_a_malformed_file_is_refused(
    tmp_path: Path, variables: dict, message: str
) -> None:
    path = _save(tmp_path / "bad_sn.mat", **{**_variables(), **variables})
    with pytest.raises(ParserContentError, match=message):
        SpmSnRaw.load(path)


def test_a_file_without_a_variable_is_refused(tmp_path: Path) -> None:
    variables = _variables()
    del variables["VF"]
    content = _save(tmp_path / "bad_sn.mat", **variables).read_bytes()
    with pytest.raises(ParserContentError, match="VF"):
        SpmSnRaw.from_bytes(content)


def test_v73_and_v4_files_are_refused(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "v73_sn.mat"
    with h5py.File(path, "w", userblock_size=512) as f:
        f["Affine"] = AFFINE
    with open(path, "r+b") as file:
        file.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(128))
    with pytest.raises(ParserNotImplementedError, match="v7.3"):
        SpmSnRaw.load(path)
    # SPM99 saved its *_sn3d.mat files in the v4 format.
    path = tmp_path / "old_sn.mat"
    scipy.io.savemat(str(path), {"mgc": 960209.0, "MF": VF_MAT}, format="4")
    with pytest.raises(ParserContentError, match="SPM99"):
        SpmSnRaw.load(path)


def test_unknown_reader_options_are_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "s_sn.mat")
    with pytest.raises(TypeError):
        SpmSnTransformation.load(path, steps=3)
    with pytest.raises(TypeError):
        SpmSnRaw.load(path, steps=3)


# ----------------------------------------------------------------------
#   COPYING, CONSOLIDATING AND WRITING
# ----------------------------------------------------------------------


def test_copies_keep_the_metadata_and_stay_lazy(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    for copy in (replace(sn), SpmSnTransformation.from_instance(sn)):
        assert copy.metadata is sn.metadata
        assert copy._transformations is None
        assert "_cache_transformations" not in vars(copy)
    assert "_cache_transformations" not in vars(sn)


def test_consolidating_keeps_everything(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    metadata = sn.metadata
    copy = sn.consolidate()
    assert copy is not sn
    assert copy.metadata is metadata
    assert "_cache_transformations" not in vars(copy)
    assert sn.consolidate(inplace=True) is sn
    # A chain that is assigned cannot be encoded, so it is kept.
    chain = (RASToVoxel(matrix=np.eye(4)[:3]),)
    sn.transformations = chain
    assert sn.consolidate().transformations == chain
    assert sn.consolidate().metadata is metadata
    sn.transformations = None
    assert len(sn.transformations) == 3


def test_other_metadata_gives_another_chain(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    first = sn.transformations
    affine = _variables(tr=np.zeros((0, 0)))
    sn.metadata = SpmSnMetadata.load(_save(tmp_path / "a_sn.mat", **affine))
    assert sn.transformations is not first
    assert len(sn.transformations) == 2


def test_a_normalisation_is_not_written(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    assert not hasattr(sn, "save")
    assert not hasattr(sn.metadata, "save")
    with pytest.raises(WriterError):
        io.save(sn, tmp_path / "copy_sn.mat")


def test_nothing_is_converted_into_a_normalisation(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    y = sn.to(SpmCoordinatesField)
    with pytest.raises(ConversionError, match="discrete cosines"):
        y.to(SpmSnTransformation)
    with pytest.raises(ConversionError):
        SpmSnTransformation.from_instance(xforms.Affine(np.eye(4)[:3]))
    assert sn.to(SpmSnTransformation).metadata is sn.metadata


def test_the_map_is_written_as_a_deformation_field(tmp_path: Path) -> None:
    sn = SpmSnTransformation.load(_save(tmp_path / "s_sn.mat"))
    target = tmp_path / "y_s.nii"
    io.save(sn.to(SpmCoordinatesField), target)
    image = nb.load(str(target))
    assert image.shape == (*DIM, 1, 3)
    assert np.allclose(image.affine, _template_nifti_affine())
    values = np.asarray(image.dataobj)[:, :, :, 0, :]
    # NIfTI stores the coordinates in single precision, as SPM does.
    assert np.allclose(values, _spm_sn2def(_variables()), atol=1e-4)
    assert type(io.load(target)) is SpmCoordinatesField


def test_bytes_hold_one_file() -> None:
    buffer = _io.BytesIO()
    scipy.io.savemat(buffer, _variables(), format="5")
    sn = SpmSnTransformation.from_bytes(buffer.getvalue())
    assert sn.metadata.raw.template.dim == DIM


# ----------------------------------------------------------------------
#   THE CHECKS OF THE CONFORMANCE TESTS THAT APPLY
# ----------------------------------------------------------------------
# The conformance tests list this format as not migrated, because their
# checks of assigned data expect consolidation to leave nothing pending,
# and a format that cannot write keeps an assigned chain (design memo,
# item 260). The checks below are those that apply, numbered as in
# `tests/test_io_conformance.py`.


def test_01_the_constructor_takes_the_model_and_metadata() -> None:
    names = {f.public_name for f in fields(SpmSnTransformation) if f.init}
    model = {f.public_name for f in fields(xforms.ImmutableSequence) if f.init}
    assert names <= model | {"metadata"}
    assert not names & {"image", "header", "_header", "struct", "raw"}
    metadata = {f.public_name for f in fields(SpmSnMetadata) if f.init}
    assert metadata <= {f.public_name for f in fields(Metadata)} | {"raw"}


def test_06_the_metadata_reads_the_whole_file(tmp_path: Path) -> None:
    # The check is vacuous: the file is read in one pass, and its record
    # holds the coefficients of the warp, which are its data.
    path = _save(tmp_path / "s_sn.mat")
    record = MetadataFormat.load(path).raw
    assert np.array_equal(record.tr, _coefficients())


def test_12_the_adapter_contract_holds(tmp_path: Path) -> None:
    for klass in (SpmSnTransformation, SpmSnMetadata):
        assert not klass._is_dispatcher()
        assert _overrides_from_fileobj(klass)
        assert klass._READ_MODE == "rb"
