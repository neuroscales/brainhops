"""Tests for FreeSurfer MGH and MGZ images.

The reader must agree with nibabel on the voxels and on both the scanner and
tkr vox-to-RAS matrices, and must recognise the format by content.
"""

import gc
import gzip
import io as _io
import os
import pickle
import struct
import warnings

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from bagof.magic import replace  # noqa: E402
from nibabel.freesurfer.mghformat import MGHImage as NibabelMgh  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.backends import backend  # noqa: E402
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
    Scaling,
)
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.freesurfer._geometry import (  # noqa: E402
    fs_geometry_from_vox2ras,
    fs_vox2ras,
    fs_vox2tkr,
    mat2orient,
)
from brainhops.io.images.freesurfer.mgh import (  # noqa: E402
    MghImage,
    MghMetadata,
)
from brainhops.io.images.nifti import NiftiImage  # noqa: E402

# The geometry is oblique and anisotropic, so that every header field
# matters.
_ROT = np.array(
    [
        [np.cos(0.3), -np.sin(0.3), 0.0],
        [np.sin(0.3), np.cos(0.3), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
AFFINE = np.eye(4)
AFFINE[:3, :3] = (
    _ROT
    @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    * [
        1.0,
        2.0,
        1.5,
    ]
)
AFFINE[:3, 3] = [10.0, -20.0, 5.0]

TAGS = struct.pack(">iq", 30, 9) + b"cmdline\x00\x00"


def _write(tmp_path, name, data, affine=AFFINE, tr=0.0, tags=b""):  # noqa: ANN001, ANN202
    img = NibabelMgh(data, affine)
    img.header["tr"] = tr
    img.header["te"] = 3.5
    img.header["flip_angle"] = 0.15
    target = tmp_path / name
    nb.save(img, str(target))
    if tags:
        raw = target.read_bytes()
        gz = raw[:2] == b"\x1f\x8b"
        if gz:
            raw = gzip.decompress(raw)
        raw += tags
        target.write_bytes(gzip.compress(raw) if gz else raw)
    return target


def _data(shape=(4, 5, 6), dtype="float32"):  # noqa: ANN001, ANN202
    return np.arange(np.prod(shape)).reshape(shape).astype(dtype)


# ----------------------------------------------------------------------
#   SHARED FREESURFER GEOMETRY
# ----------------------------------------------------------------------


def test_shared_geometry_matches_nibabel() -> None:
    img = NibabelMgh(np.zeros((7, 8, 9), "float32"), AFFINE)
    header = img.header
    shape = header["dims"][:3]
    mdc = header["Mdc"]
    vox2ras = fs_vox2ras(
        shape, header["delta"], mdc[0], mdc[1], mdc[2], header["Pxyz_c"]
    )
    assert np.allclose(vox2ras, header.get_vox2ras(), atol=1e-5)
    assert np.allclose(
        fs_vox2tkr(shape, header["delta"]),
        header.get_vox2ras_tkr(),
        atol=1e-5,
    )
    size, x, y, z, c = fs_geometry_from_vox2ras(AFFINE, shape)
    assert np.allclose(size, header["delta"], atol=1e-6)
    assert np.allclose(c, header["Pxyz_c"], atol=1e-5)
    assert np.allclose([x, y, z], header["Mdc"], atol=1e-6)


def test_tkr_of_a_conformed_volume_is_lia() -> None:
    assert mat2orient(fs_vox2tkr((256, 256, 256), (1, 1, 1))) == "LIA"


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vol.mgh", "vol.mgz"])
def test_read_matches_nibabel(tmp_path, name) -> None:  # noqa: ANN001
    """Voxels, scanner and tkr matrices all match nibabel."""
    data = _data()
    source = _write(tmp_path, name, data)
    image = io.images.load(source)

    assert isinstance(image, MghImage)
    assert np.array_equal(np.asarray(image.data), data)
    record = image.metadata.raw
    assert record.good_ras is True

    header = nb.load(str(source)).header
    names = [x.output.name for x in image.transformations]
    assert names == ["physical", "tkr", "scanner"]
    assert image.transformation.output.name == "scanner"

    scanner = image.transformations[-1].homogeneous_matrix
    tkr = image.transformations[1].homogeneous_matrix
    assert np.allclose(scanner, header.get_vox2ras(), atol=1e-5)
    assert np.allclose(tkr, header.get_vox2ras_tkr(), atol=1e-5)
    assert np.allclose(record.vox2ras, header.get_vox2ras(), atol=1e-5)
    assert np.allclose(record.vox2tkr, header.get_vox2ras_tkr(), atol=1e-5)
    assert not np.allclose(scanner, tkr)

    scaling = image.transformations[0]
    assert isinstance(scaling, Scaling)
    assert np.allclose(scaling.scale, header["delta"])


@pytest.mark.parametrize("name", ["vol.mgh", "vol.mgz"])
def test_load_leaves_no_file_open(tmp_path, name) -> None:  # noqa: ANN001
    """Loading closes the header file (#262) while the voxels stay lazy."""
    data = _data()
    source = _write(tmp_path, name, data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        image = io.load(source)
        assert isinstance(image, MghImage)
        assert nb.is_proxy(image.raw)
        assert np.array_equal(np.asarray(image.data), data)
        del image
        gc.collect()
    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaks, [str(w.message) for w in leaks]


def test_tkr_and_scanner_differ_by_a_ras_to_ras_rigid(tmp_path) -> None:  # noqa: ANN001
    """scanner = M @ tkr, where M = Norig @ inv(Torig) is a rigid transform."""
    source = _write(tmp_path, "vol.mgz", _data())
    record = io.images.load(source).metadata.raw
    tkr2scanner = record.vox2ras @ np.linalg.inv(record.vox2tkr)
    # The voxel size cancels, leaving a rotation.
    linear = tkr2scanner[:3, :3]
    assert np.allclose(linear @ linear.T, np.eye(3), atol=1e-5)
    # The volume centre maps to c_ras in scanner space and 0 in tkr space.
    centre = np.r_[np.asarray(record.shape[:3]) / 2, 1]
    assert np.allclose(record.vox2tkr @ centre, [0, 0, 0, 1], atol=1e-5)
    assert np.allclose(
        (record.vox2ras @ centre)[:3], record.header["Pxyz_c"], atol=1e-4
    )


def test_read_4d(tmp_path) -> None:  # noqa: ANN001
    """Frames become the fourth axis, with TR as the time step."""
    data = _data((3, 4, 5, 2))
    source = _write(tmp_path, "vol4d.mgz", data, tr=2500.0)
    image = io.images.load(source)

    assert image.data.shape == (3, 4, 5, 2)
    assert np.array_equal(np.asarray(image.data), data)
    assert [a.name for a in image.system.axes] == ["x", "y", "z", "t"]
    scaling = image.transformations[0]
    assert np.allclose(scaling.scale[-1], 2500.0)
    assert scaling.output.axes[-1].unit.name in ("ms", "millisecond")
    header = nb.load(str(source)).header
    assert np.allclose(
        image.transformation.homogeneous_matrix[[0, 1, 2, -1]][
            :, [0, 1, 2, -1]
        ],
        header.get_vox2ras(),
        atol=1e-5,
    )


def test_4d_without_tr_invents_no_time_unit(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol4d.mgh", _data((3, 4, 5, 2)))
    scaling = io.images.load(source).transformations[0]
    assert scaling.scale[-1] == 1.0
    assert scaling.output.axes[-1].unit is None


def test_data_is_f_ordered_like_nibabel(tmp_path) -> None:  # noqa: ANN001
    data = _data((2, 3, 4))
    source = _write(tmp_path, "vol.mgh", data)
    raw = source.read_bytes()[284 : 284 + data.size * 4]
    flat = np.frombuffer(raw, ">f4")
    assert np.array_equal(flat, data.ravel(order="F"))
    assert io.images.load(source).system.order == "F"


def test_mri_params_are_kept(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgz", _data(), tr=1234.0)
    params = io.images.load(source).metadata.raw.mri_params
    assert params["tr"] == pytest.approx(1234.0)
    assert params["te"] == pytest.approx(3.5)
    assert params["flip_angle"] == pytest.approx(0.15)


def test_bad_ras_flag_uses_freesurfer_defaults(tmp_path) -> None:  # noqa: ANN001
    """Without goodRASFlag, the voxels are 1 mm, LIA and centred at 0."""
    source = _write(tmp_path, "vol.mgh", _data())
    raw = bytearray(source.read_bytes())
    raw[28:30] = struct.pack(">h", 0)
    source.write_bytes(bytes(raw))

    record = io.images.load(source).metadata.raw
    assert record.good_ras is False
    assert int(record.header["goodRASFlag"]) == 0
    assert record.voxel_size == (1.0, 1.0, 1.0)
    assert np.allclose(record.vox2ras, record.vox2tkr)
    assert mat2orient(record.vox2ras) == "LIA"


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("compress", [False, True])
def test_sniff_by_content(tmp_path, compress) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgh", _data())
    raw = source.read_bytes()
    if compress:
        raw = gzip.compress(raw)
    disguised = tmp_path / "volume.bin"
    disguised.write_bytes(raw)

    assert MghImage.sniff(raw) >= Confidence.LIKELY
    assert MghImage.sniff(disguised) >= Confidence.LIKELY
    image = MghImage.load(disguised)
    assert np.array_equal(np.asarray(image.data), _data())


def test_a_gzipped_file_named_mgh_is_read(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgz", _data())
    misnamed = tmp_path / "vol.mgh"
    misnamed.write_bytes(source.read_bytes())
    image = io.images.load(misnamed)
    assert np.array_equal(np.asarray(image.data), _data())


def test_sniff_rejects_other_content(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "vol.nii"
    nb.save(nb.Nifti1Image(_data(), np.eye(4)), str(target))
    assert MghImage.sniff(target) == Confidence.NO
    assert MghImage.sniff(b"not an mgh file at all, really not") == 0
    bad_version = struct.pack(">7ih", 2, 4, 5, 6, 1, 3, 0, 1)
    assert MghImage.sniff(bad_version) == Confidence.NO


def test_from_bytes_and_fileobj(tmp_path) -> None:  # noqa: ANN001
    """Bytes and streams, compressed or not, read the same image."""
    source = _write(tmp_path, "vol.mgz", _data(), tags=TAGS)
    raw = source.read_bytes()
    a = MghImage.from_bytes(raw)
    b = MghImage.from_bytes(gzip.decompress(raw))
    with open(source, "rb") as f:
        # As for NIfTI, the voxels are read lazily from the open stream.
        c = MghImage.from_fileobj(f)
        assert np.array_equal(np.asarray(c.data), _data())
    for image in (a, b, c):
        assert np.array_equal(np.asarray(image.data), _data())
        assert image.metadata.raw.tags == TAGS
        assert np.allclose(image.metadata.raw.vox2ras, a.metadata.raw.vox2ras)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("out", ["out.mgh", "out.mgz"])
def test_round_trip(tmp_path, out) -> None:  # noqa: ANN001
    """Voxels, geometry, footer parameters and tags survive a round trip."""
    source = _write(tmp_path, "vol.mgz", _data(), tr=900.0, tags=TAGS)
    image = io.images.load(source)
    record = image.metadata.raw
    assert record.tags == TAGS
    target = tmp_path / out
    image.save(target)

    assert (target.read_bytes()[:2] == b"\x1f\x8b") == out.endswith(".mgz")
    again = io.images.load(target)
    copied = again.metadata.raw
    assert np.array_equal(np.asarray(again.data), _data())
    assert np.allclose(copied.vox2ras, record.vox2ras, atol=1e-5)
    assert np.allclose(copied.vox2tkr, record.vox2tkr, atol=1e-5)
    assert copied.mri_params == pytest.approx(record.mri_params)
    assert copied.tags == TAGS
    assert copied.header.get_data_dtype() == np.dtype(">f4")


def test_round_trip_4d(tmp_path) -> None:  # noqa: ANN001
    data = _data((3, 4, 5, 2), "int16")
    source = _write(tmp_path, "vol.mgz", data, tr=2000.0)
    target = tmp_path / "out.mgz"
    io.images.load(source).save(target)
    again = io.images.load(target)
    assert again.data.shape == (3, 4, 5, 2)
    assert np.array_equal(np.asarray(again.data), data)
    record = again.metadata.raw
    assert record.mri_params["tr"] == pytest.approx(2000.0)
    assert np.allclose(
        record.vox2ras, nb.load(str(source)).header.get_vox2ras(), atol=1e-5
    )


def test_bytes_round_trip(tmp_path) -> None:  # noqa: ANN001
    image = io.images.load(_write(tmp_path, "vol.mgh", _data(), tags=TAGS))
    for compress in (False, True):
        again = MghImage.from_bytes(image.to_bytes(compress=compress))
        assert np.array_equal(np.asarray(again.data), _data())
        assert again.metadata.raw.tags == TAGS


def test_write_from_data_and_affine(tmp_path) -> None:  # noqa: ANN001
    """The preferred affine of an in-memory image is written as scanner RAS."""
    image = MghImage(
        data=_data(dtype="float64"),
        transformations=[Affine(matrix=AFFINE[:3])],
    )
    target = tmp_path / "out.mgz"
    image.save(target)
    loaded = nb.load(str(target))
    assert np.allclose(loaded.affine, AFFINE, atol=1e-5)
    # float64 is not an MGH type, so float32 is stored.
    assert loaded.get_data_dtype() == np.dtype(">f4")


def test_a_scaling_and_no_geometry_are_written(tmp_path) -> None:  # noqa: ANN001
    """A scaling is written as its matrix; no transformation means 1 mm."""
    target = tmp_path / "a.mgh"
    MghImage(
        data=_data(), transformations=[Scaling(scale=[2.0, 3.0, 4.0])]
    ).save(target)
    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])

    target = tmp_path / "b.mgh"
    MghImage(data=_data()).save(target)
    assert np.allclose(nb.load(str(target)).affine, np.eye(4))


def test_a_preferred_tkr_is_not_written_as_scanner(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgz", _data())
    image = io.images.load(source)
    image.transformations = list(image.transformations)
    image.transformation = "tkr"
    assert image.transformation.output.name == "tkr"
    target = tmp_path / "out.mgz"
    image.save(target)
    assert np.allclose(
        nb.load(str(target)).affine,
        nb.load(str(source)).affine,
        atol=1e-5,
    )


def test_world_units_are_converted_to_mm(tmp_path) -> None:  # noqa: ANN001
    affine = Affine(matrix=np.diag([0.1, 0.2, 0.3, 1.0])[:3])
    world = affine.output
    if world is None or world.ndim is None:
        world = CoordinateSystem(name="world", axes=["x", "y", "z"])
    world = replace(
        world,
        axes=[replace(a, unit="centimeter") for a in world.axes],
    )
    image = MghImage(
        data=_data(), transformations=[replace(affine, output=world)]
    )
    target = tmp_path / "cm.mgh"
    image.save(target)
    assert np.allclose(np.diag(nb.load(str(target)).affine), [1, 2, 3, 1])


@pytest.mark.parametrize(
    "dtype, stored",
    [
        ("uint8", "u1"),
        ("int16", "i2"),
        ("int32", "i4"),
        ("float32", "f4"),
        ("bool", "u1"),
        ("int8", "i2"),
        ("uint16", "i4"),
        ("int64", "i4"),
        ("float64", "f4"),
    ],
)
def test_dtypes(tmp_path, dtype, stored) -> None:  # noqa: ANN001
    """Each dtype is stored as the nearest MGH type."""
    data = (_data() % 2).astype(dtype)
    target = tmp_path / "out.mgh"
    MghImage(data=data).save(target)
    loaded = nb.load(str(target))
    assert loaded.get_data_dtype() == np.dtype(">" + stored)
    assert np.array_equal(np.asarray(loaded.dataobj), data.astype(stored))


def test_unstorable_values_are_refused() -> None:
    """Integers beyond int32, complex values and 5-D data are refused."""
    with pytest.raises(WriterError):
        MghImage(data=np.full((2, 2, 2), 2**40, "int64")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=np.zeros((2, 2, 2), "complex64")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=np.zeros((2, 2, 2, 2, 2), "float32")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=None).to_bytes()


def test_explicit_dtype_and_params(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "out.mgz"
    MghImage(data=_data()).save(target, dtype="int16", tr=42.0)
    loaded = nb.load(str(target))
    assert loaded.get_data_dtype() == np.dtype(">i2")
    assert float(loaded.header["tr"]) == pytest.approx(42.0)


def test_like_copies_mri_params(tmp_path) -> None:  # noqa: ANN001
    """`like` supplies the footer parameters but never the geometry."""
    template = _write(tmp_path, "template.mgz", _data(), tr=777.0)
    image = MghImage(data=_data(), transformations=[Scaling(scale=[2, 2, 2])])
    target = tmp_path / "out.mgz"
    image.save(target, like=template)
    loaded = nb.load(str(target))
    assert float(loaded.header["tr"]) == pytest.approx(777.0)
    assert np.allclose(np.diag(loaded.affine), [2, 2, 2, 1])


def test_a_field_is_unrepresentable() -> None:
    field = DisplacementField(field=np.zeros((4, 5, 6, 3), dtype="float32"))
    image = MghImage(data=_data(), transformations=[field])
    with pytest.raises(UnrepresentableTransformationError):
        image.to_bytes()


def test_mgh_to_nifti_keeps_scanner_geometry(tmp_path) -> None:  # noqa: ANN001
    """The scanner geometry becomes the NIfTI sform."""
    source = _write(tmp_path, "vol.mgz", _data())
    image = io.images.load(source)
    target = tmp_path / "out.nii.gz"
    NiftiImage(data=image.data, transformations=image.transformations).save(
        target
    )
    assert np.allclose(
        nb.load(str(target)).affine,
        nb.load(str(source)).affine,
        atol=1e-5,
    )


def test_io_save_converts_between_mgh_and_nifti(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgz", _data())
    nifti = tmp_path / "out.nii.gz"
    io.save(io.load(source), nifti)
    assert np.allclose(
        nb.load(str(nifti)).affine, nb.load(str(source)).affine, atol=1e-5
    )

    back = tmp_path / "back.mgz"
    io.save(io.load(nifti), back)
    image = io.load(back)
    assert isinstance(image, MghImage)
    assert np.array_equal(np.asarray(image.data), _data())
    assert np.allclose(
        image.metadata.raw.vox2ras,
        nb.load(str(source)).header.get_vox2ras(),
        atol=1e-5,
    )


class _RemotePath(os.PathLike):
    """An in-memory remote path that nibabel cannot open by name."""

    def __init__(self, url: str, store: dict) -> None:
        self.url, self.store = url, store

    def __fspath__(self) -> str:
        raise NotImplementedError("a remote path has no local path")

    def __str__(self) -> str:
        return self.url

    def exists(self) -> bool:
        return self.url in self.store

    def open(self, mode: str = "rb", **kwargs) -> _io.BytesIO:  # noqa: A003
        if "w" not in mode:
            return _io.BytesIO(self.store[self.url])
        url, store = self.url, self.store

        class _Writer(_io.BytesIO):
            def close(self) -> None:
                store[url] = self.getvalue()
                super().close()

        return _Writer()


@pytest.mark.parametrize("name", ["vol.mgh", "vol.mgz"])
def test_a_remote_path_round_trips(tmp_path, name) -> None:  # noqa: ANN001
    """A remote path is read and written as a stream by its own backend."""
    source = _write(tmp_path, name, _data(), tags=TAGS)
    store = {f"s3://bucket/{name}": source.read_bytes()}
    image = MghImage.load(_RemotePath(f"s3://bucket/{name}", store))
    assert np.array_equal(np.asarray(image.data), _data())
    assert image.metadata.raw.tags == TAGS
    target = _RemotePath(f"s3://bucket/out/{name}", store)
    image.save(target)
    raw = store[target.url]
    assert (raw[:2] == b"\x1f\x8b") == name.endswith(".mgz")
    again = MghImage.from_bytes(raw)
    assert np.array_equal(np.asarray(again.data), _data())
    record = again.metadata.raw
    assert np.allclose(record.vox2ras, image.metadata.raw.vox2ras)
    assert record.tags == TAGS


# ----------------------------------------------------------------------
#   REVIEW FOLLOW-UPS
# ----------------------------------------------------------------------

_LTA = """\
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
filename = /subjects/bert/mri/orig.mgz
volume = 4 5 6
voxelsize = 1.0 1.0 1.0
xras   = -1.0 0.0 0.0
yras   = 0.0 0.0 -1.0
zras   = 0.0 1.0 0.0
cras   = 0.0 0.0 0.0
dst volume info
valid = 1  # volume info valid
filename = /subjects/bert/mri/orig.mgz
volume = 4 5 6
voxelsize = 1.0 1.0 1.0
xras   = -1.0 0.0 0.0
yras   = 0.0 0.0 -1.0
zras   = 0.0 1.0 0.0
cras   = 0.0 0.0 0.0
"""


def test_the_freesurfer_hint_selects_mgh_and_lta(tmp_path) -> None:  # noqa: ANN001
    """The shared 'freesurfer' hint selects both MGH and LTA."""
    from brainhops.io.base.specs import format_hints
    from brainhops.io.common.freesurfer import FreesurferFormat
    from brainhops.io.transformations.freesurfer.lta import (
        LtaTransformation,
    )

    assert issubclass(MghImage, FreesurferFormat)
    assert issubclass(LtaTransformation, FreesurferFormat)
    assert {"freesurfer", "mgh", "mgz", "freesurfer.mgh"} <= set(
        format_hints(MghImage)
    )

    volume = _write(tmp_path, "vol.mgz", _data())
    lta = tmp_path / "reg.lta"
    lta.write_text(_LTA)
    assert type(io.load(volume, hint="freesurfer")) is MghImage
    assert type(io.load(lta, hint="freesurfer")) is LtaTransformation
    assert type(io.load(volume, hint="freesurfer.mgh")) is MghImage


def test_conversion_does_not_carry_nibabel_objects(tmp_path) -> None:  # noqa: ANN001
    """Conversion copies the data model, not the record or the proxy."""
    mgh = MghImage.from_file(_write(tmp_path, "vol.mgz", _data()))
    nii = NiftiImage.from_instance(mgh)
    assert nii.metadata is None
    assert np.array_equal(np.asarray(nii.data), _data())
    back = MghImage.from_instance(nii)
    assert back.metadata is None
    assert not nb.is_proxy(back.raw)
    # Within one format, the record and the proxy are still shared.
    copy = MghImage.from_instance(mgh)
    assert copy.metadata is mgh.metadata
    assert copy.raw is mgh.raw


class _Unseekable(_io.RawIOBase):
    """A forward-only stream, like a pipe."""

    def __init__(self, data: bytes) -> None:
        self._buffer = _io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:  # noqa: ANN001
        chunk = self._buffer.read(len(b))
        b[: len(chunk)] = chunk
        return len(chunk)


def test_an_unseekable_stream_is_read(tmp_path) -> None:  # noqa: ANN001
    raw = _write(tmp_path, "vol.mgz", _data(), tags=TAGS).read_bytes()
    image = MghImage.from_fileobj(_Unseekable(raw))
    assert np.array_equal(np.asarray(image.data), _data())
    assert image.metadata.raw.tags == TAGS


# ----------------------------------------------------------------------
#   RECORD AND WRITE PRECEDENCE
# ----------------------------------------------------------------------


def _content(path) -> bytes:  # noqa: ANN001
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _no_ras_flag(path) -> None:  # noqa: ANN001
    """Set the `goodRASFlag` of an uncompressed file to 0 in place."""
    raw = bytearray(path.read_bytes())
    raw[28:30] = struct.pack(">h", 0)
    path.write_bytes(bytes(raw))


@pytest.mark.parametrize("name", ["vol.mgh", "vol.mgz"])
def test_an_untouched_oblique_file_keeps_its_bytes(tmp_path, name) -> None:  # noqa: ANN001
    """Oblique geometry, footer and tags are written back byte for byte."""
    data = _data((3, 4, 5, 2), "int16")
    source = _write(tmp_path, name, data, tr=2300.0, tags=TAGS)
    loaded = MghImage.load(source)
    assert loaded.to_raw() is loaded.metadata.raw
    targets = {
        "save": lambda target: loaded.save(target),
        "io.save": lambda target: io.save(MghImage.load(source), target),
        "replace": lambda target: replace(loaded).save(target),
        "from_instance": lambda target: MghImage.from_instance(loaded).save(
            target
        ),
    }
    for label, write in targets.items():
        target = tmp_path / (label + "-" + name)
        write(target)
        assert _content(target) == _content(source), label


def test_a_file_without_valid_geometry_keeps_its_flag(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgh", _data())
    _no_ras_flag(source)
    loaded = MghImage.load(source)
    record = loaded.metadata.raw
    # The record keeps the stored geometry, which FreeSurfer ignores.
    assert int(record.header["goodRASFlag"]) == 0
    stored = nb.load(str(_write(tmp_path, "ref.mgh", _data()))).header
    assert np.array_equal(record.header["Mdc"], stored["Mdc"])
    assert np.array_equal(record.copy().header["Mdc"], stored["Mdc"])
    assert mat2orient(loaded.transformation.homogeneous_matrix) == "LIA"
    target = tmp_path / "copy.mgh"
    loaded.save(target)
    assert target.read_bytes() == source.read_bytes()
    # An image whose geometry is encoded sets the flag.
    loaded.transformations = [Affine(matrix=AFFINE[:3])]
    loaded.save(target)
    again = MghImage.load(target).metadata.raw
    assert again.good_ras
    assert np.allclose(again.vox2ras, AFFINE, atol=1e-5)


def test_to_raw_follows_each_knob(tmp_path) -> None:  # noqa: ANN001
    """Each public knob is either written or leaves the record as read."""
    source = _write(tmp_path, "vol.mgh", _data(), tr=900.0, tags=TAGS)

    def loaded():  # noqa: ANN202
        return MghImage.load(source)

    image = loaded()
    record = image.metadata.raw
    assert image.to_raw() is record

    # New voxels of the same shape and type are written under the record.
    image.data = _data() + 1
    assert image.to_raw() is record
    assert np.array_equal(
        np.asarray(MghImage.from_bytes(image.to_bytes()).data), _data() + 1
    )

    # Another type changes the stored voxel type only.
    image = loaded()
    image.data = _data(dtype="int16")
    written = image.to_raw()
    assert written is not record
    assert written.header.get_data_dtype() == np.dtype(">i2")
    assert np.array_equal(written.header["Mdc"], record.header["Mdc"])
    assert written.tags == TAGS

    # Another number of frames keeps the stored geometry.
    image = loaded()
    image.data = _data((4, 5, 6, 3))
    written = image.to_raw()
    assert written.shape == (4, 5, 6, 3)
    assert np.array_equal(written.header["delta"], record.header["delta"])
    assert np.array_equal(written.header["Pxyz_c"], record.header["Pxyz_c"])

    # Another spatial shape keeps the corner, so the centre moves.
    image = loaded()
    image.data = _data((2, 5, 6))
    written = image.to_raw()
    assert written.shape == (2, 5, 6)
    assert np.allclose(written.vox2ras, record.vox2ras, atol=1e-4)
    assert not np.allclose(written.header["Pxyz_c"], record.header["Pxyz_c"])

    # Assigned transformations give the geometry.
    image = loaded()
    image.transformations = [Scaling(scale=[2.0, 3.0, 4.0])]
    written = image.to_raw()
    assert np.allclose(np.diag(written.vox2ras), [2, 3, 4, 1])
    assert written.tags == TAGS
    assert written.mri_params["tr"] == pytest.approx(900.0)

    # Other metadata gives its record and its geometry.
    image = loaded()
    edited = image.metadata.to_raw()
    edited.tags = b"other"
    image.metadata = MghMetadata.from_raw(edited)
    assert image.to_raw() is edited

    # `like` and the overrides are written, the overrides last.
    template = _write(tmp_path, "template.mgh", _data(), tr=777.0)
    image = loaded()
    assert image.to_raw(like=template).mri_params["tr"] == pytest.approx(777.0)
    written = image.to_raw(like=template, tr=55.0, dtype="uint8")
    assert written.mri_params["tr"] == pytest.approx(55.0)
    assert written.header.get_data_dtype() == np.dtype("u1")
    held = image.metadata.raw
    assert image.to_raw(te=3.5, flip_angle=0.15, tr=900.0) is held

    # The record of the metadata is never changed.
    assert record == MghImage.load(source).metadata.raw

    # The voxel system is derived from the record and cannot be assigned.
    with pytest.raises(AttributeError):
        image.system = None


def test_decoded_data_is_read_only(tmp_path) -> None:  # noqa: ANN001
    # An edit in place would not reach `raw`, which is what is written.
    image = MghImage.load(_write(tmp_path, "vol.mgh", _data()))
    with backend("numpy"):
        data = image.data
    assert not data.flags.writeable
    with pytest.raises(ValueError):
        data[0, 0, 0] = 1


def test_metadata_reads_and_writes_the_record(tmp_path) -> None:  # noqa: ANN001
    source = _write(tmp_path, "vol.mgz", _data(), tr=900.0, tags=TAGS)
    metadata = MghMetadata.load(source)
    assert metadata.raw.tags == TAGS
    assert metadata.raw.mri_params["tr"] == pytest.approx(900.0)
    assert metadata.raw == MghImage.load(source).metadata.raw
    assert pickle.loads(pickle.dumps(metadata)).raw == metadata.raw
    assert metadata.to_raw() == metadata.raw
    assert metadata.to_raw() is not metadata.raw
    # Writing the metadata alone writes the fixed header, without the
    # footer and the tags that a file stores after the voxels.
    target = tmp_path / "header.mgz"
    metadata.save(target)
    assert len(_content(target)) == 284
    assert _content(target) == _content(source)[:284]
    header = MghMetadata.load(target).raw
    assert header.shape == metadata.raw.shape
    assert header.tags == b""
    assert header.mri_params["tr"] == 0.0


def test_from_nibabel_keeps_the_header() -> None:
    nibabel_image = NibabelMgh(_data(), AFFINE)
    image = MghImage.from_nibabel(nibabel_image)
    assert image.raw is nibabel_image.dataobj
    assert image.metadata.raw.header == nibabel_image.header
    assert image.metadata.raw.header is not nibabel_image.header
    back = image.to_nibabel()
    assert np.allclose(back.affine, AFFINE, atol=1e-5)
    assert MghImage.from_nibabel(nibabel_image.header).raw is None
