"""
Tests for FreeSurfer MGH / MGZ images.

The reader must agree with `nibabel` on the voxels and on both the
scanner and the tkr voxel-to-RAS matrices, recognise the format from its
content rather than its name, follow FreeSurfer when `goodRASFlag` is not
set, and write files that read back unchanged -- footer parameters and
trailing tags included.
"""

import gzip
import io as _io
import os
import struct

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from bagof.magic import replace  # noqa: E402
from nibabel.freesurfer.mghformat import MGHImage as NibabelMgh  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
    Scaling,
)
from brainhops.datamodel.units import SpaceUnit  # noqa: E402
from brainhops.io.base.freesurfer import (  # noqa: E402
    fs_geometry_from_vox2ras,
    fs_vox2ras,
    fs_vox2tkr,
    mat2orient,
)
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.images.freesurfer import MghImage  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402

# An oblique, anisotropic geometry, so that every part of the header
# (voxel size, direction cosines, centre) matters.
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
    """Write an MGH fixture with nibabel, and optional trailing tags."""
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
    """The shared FreeSurfer geometry reproduces nibabel's matrices."""
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
    """The tkr matrix is FreeSurfer's coronal LIA orientation."""
    assert mat2orient(fs_vox2tkr((256, 256, 256), (1, 1, 1))) == "LIA"


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vol.mgh", "vol.mgz"])
def test_read_matches_nibabel(tmp_path, name) -> None:  # noqa: ANN001
    """Voxels, scanner and tkr matrices all agree with nibabel."""
    data = _data()
    source = _write(tmp_path, name, data)
    image = io.images.load(source)

    assert isinstance(image, MghImage)
    assert np.array_equal(np.asarray(image.data), data)
    assert image.good_ras is True

    header = nb.load(str(source)).header
    names = [x.output.name for x in image.transformations]
    assert names == ["physical", "tkr", "scanner"]
    assert image.transformation.output.name == "scanner"

    scanner = image.transformations[-1].homogeneous_matrix
    tkr = image.transformations[1].homogeneous_matrix
    assert np.allclose(scanner, header.get_vox2ras(), atol=1e-5)
    assert np.allclose(tkr, header.get_vox2ras_tkr(), atol=1e-5)
    assert np.allclose(image.vox2ras, header.get_vox2ras(), atol=1e-5)
    assert np.allclose(image.vox2ras_tkr, header.get_vox2ras_tkr(), atol=1e-5)
    assert not np.allclose(scanner, tkr)

    scaling = image.transformations[0]
    assert isinstance(scaling, Scaling)
    assert np.allclose(scaling.scale, header["delta"])


def test_tkr_and_scanner_differ_by_a_ras_to_ras_rigid(tmp_path) -> None:  # noqa: ANN001
    """scanner = M @ tkr with M independent of the voxel (FreeSurfer's
    `Norig @ inv(Torig)`)."""
    source = _write(tmp_path, "vol.mgz", _data())
    image = io.images.load(source)
    tkr2scanner = image.vox2ras @ np.linalg.inv(image.vox2ras_tkr)
    # The linear part is a rotation: the voxel size cancels out.
    linear = tkr2scanner[:3, :3]
    assert np.allclose(linear @ linear.T, np.eye(3), atol=1e-5)
    # The volume centre maps to c_ras in scanner RAS and to 0 in tkr RAS.
    centre = np.r_[np.asarray(image.shape[:3]) / 2, 1]
    assert np.allclose(image.vox2ras_tkr @ centre, [0, 0, 0, 1], atol=1e-5)
    assert np.allclose(
        (image.vox2ras @ centre)[:3], image.header["Pxyz_c"], atol=1e-4
    )


def test_read_4d(tmp_path) -> None:  # noqa: ANN001
    """A multi-frame volume reads as (x, y, z, t), with TR as time step."""
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
    """With no TR recorded, the frame step is 1 with no unit."""
    source = _write(tmp_path, "vol4d.mgh", _data((3, 4, 5, 2)))
    scaling = io.images.load(source).transformations[0]
    assert scaling.scale[-1] == 1.0
    assert scaling.output.axes[-1].unit is None


def test_data_is_f_ordered_like_nibabel(tmp_path) -> None:  # noqa: ANN001
    """Disk order is x fastest: the array matches nibabel's dataobj."""
    data = _data((2, 3, 4))
    source = _write(tmp_path, "vol.mgh", data)
    raw = source.read_bytes()[284 : 284 + data.size * 4]
    flat = np.frombuffer(raw, ">f4")
    assert np.array_equal(flat, data.ravel(order="F"))
    assert io.images.load(source).system.order == "F"


def test_mri_params_are_kept(tmp_path) -> None:  # noqa: ANN001
    """The footer's MRI parameters are read."""
    source = _write(tmp_path, "vol.mgz", _data(), tr=1234.0)
    params = io.images.load(source).mri_params
    assert params["tr"] == pytest.approx(1234.0)
    assert params["te"] == pytest.approx(3.5)
    assert params["flip_angle"] == pytest.approx(0.15)


def test_bad_ras_flag_uses_freesurfer_defaults(tmp_path) -> None:  # noqa: ANN001
    """goodRASFlag = 0: 1 mm, LIA, centre 0 -- whatever the header says."""
    source = _write(tmp_path, "vol.mgh", _data())
    raw = bytearray(source.read_bytes())
    raw[28:30] = struct.pack(">h", 0)
    source.write_bytes(bytes(raw))

    image = io.images.load(source)
    assert image.good_ras is False
    assert image.voxel_size == (1.0, 1.0, 1.0)
    assert np.allclose(image.vox2ras, image.vox2ras_tkr)
    assert mat2orient(image.vox2ras) == "LIA"


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("compress", [False, True])
def test_sniff_by_content(tmp_path, compress) -> None:  # noqa: ANN001
    """The format is recognised from the bytes, whatever the name."""
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
    """A gzipped file with a `.mgh` name is still decompressed."""
    source = _write(tmp_path, "vol.mgz", _data())
    misnamed = tmp_path / "vol.mgh"
    misnamed.write_bytes(source.read_bytes())
    image = io.images.load(misnamed)
    assert np.array_equal(np.asarray(image.data), _data())


def test_sniff_rejects_other_content(tmp_path) -> None:  # noqa: ANN001
    """A NIfTI file, or garbage, is not an MGH."""
    target = tmp_path / "vol.nii"
    nb.save(nb.Nifti1Image(_data(), np.eye(4)), str(target))
    assert MghImage.sniff(target) == Confidence.NO
    assert MghImage.sniff(b"not an mgh file at all, really not") == 0
    bad_version = struct.pack(">7ih", 2, 4, 5, 6, 1, 3, 0, 1)
    assert MghImage.sniff(bad_version) == Confidence.NO


def test_from_bytes_and_fileobj(tmp_path) -> None:  # noqa: ANN001
    """Bytes and open streams, gzipped or not, read the same image."""
    source = _write(tmp_path, "vol.mgz", _data(), tags=TAGS)
    raw = source.read_bytes()
    a = MghImage.from_bytes(raw)
    b = MghImage.from_bytes(gzip.decompress(raw))
    with open(source, "rb") as f:
        c = MghImage.from_fileobj(f)
    for image in (a, b, c):
        assert np.array_equal(np.asarray(image.data), _data())
        assert image.tags == TAGS
        assert np.allclose(image.vox2ras, a.vox2ras)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("out", ["out.mgh", "out.mgz"])
def test_round_trip(tmp_path, out) -> None:  # noqa: ANN001
    """Voxels, geometry, footer and tags survive a save and reload."""
    source = _write(tmp_path, "vol.mgz", _data(), tr=900.0, tags=TAGS)
    image = io.images.load(source)
    assert image.tags == TAGS
    target = tmp_path / out
    image.save(target)

    assert (target.read_bytes()[:2] == b"\x1f\x8b") == out.endswith(".mgz")
    again = io.images.load(target)
    assert np.array_equal(np.asarray(again.data), _data())
    assert np.allclose(again.vox2ras, image.vox2ras, atol=1e-5)
    assert np.allclose(again.vox2ras_tkr, image.vox2ras_tkr, atol=1e-5)
    assert again.mri_params == pytest.approx(image.mri_params)
    assert again.tags == TAGS
    assert again.header.get_data_dtype() == np.dtype(">f4")


def test_round_trip_4d(tmp_path) -> None:  # noqa: ANN001
    """A multi-frame volume round-trips."""
    data = _data((3, 4, 5, 2), "int16")
    source = _write(tmp_path, "vol.mgz", data, tr=2000.0)
    target = tmp_path / "out.mgz"
    io.images.load(source).save(target)
    again = io.images.load(target)
    assert again.data.shape == (3, 4, 5, 2)
    assert np.array_equal(np.asarray(again.data), data)
    assert again.mri_params["tr"] == pytest.approx(2000.0)
    assert np.allclose(
        again.vox2ras, nb.load(str(source)).header.get_vox2ras(), atol=1e-5
    )


def test_bytes_round_trip(tmp_path) -> None:  # noqa: ANN001
    """to_bytes / from_bytes round-trip, with and without compression."""
    image = io.images.load(_write(tmp_path, "vol.mgh", _data(), tags=TAGS))
    for compress in (False, True):
        again = MghImage.from_bytes(image.to_bytes(compress=compress))
        assert np.array_equal(np.asarray(again.data), _data())
        assert again.tags == TAGS


def test_write_from_data_and_affine(tmp_path) -> None:  # noqa: ANN001
    """An image built in memory writes its preferred affine as scanner
    RAS, and nibabel reads the same matrix back."""
    image = MghImage(
        data=_data(dtype="float64"),
        transformations=[Affine(matrix=AFFINE[:3])],
    )
    target = tmp_path / "out.mgz"
    image.save(target)
    loaded = nb.load(str(target))
    assert np.allclose(loaded.affine, AFFINE, atol=1e-5)
    # float64 is not an MGH type: it is stored as float32.
    assert loaded.get_data_dtype() == np.dtype(">f4")


def test_a_scaling_and_no_geometry_are_written(tmp_path) -> None:  # noqa: ANN001
    """A Scaling is written as its matrix; no transformation as 1 mm RAS."""
    target = tmp_path / "a.mgh"
    MghImage(
        data=_data(), transformations=[Scaling(scale=[2.0, 3.0, 4.0])]
    ).save(target)
    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])

    target = tmp_path / "b.mgh"
    MghImage(data=_data()).save(target)
    assert np.allclose(nb.load(str(target)).affine, np.eye(4))


def test_a_preferred_tkr_is_not_written_as_scanner(tmp_path) -> None:  # noqa: ANN001
    """Selecting tkr as preferred still writes the scanner geometry."""
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
    """A world space in centimetres is written in millimetres."""
    affine = Affine(matrix=np.diag([0.1, 0.2, 0.3, 1.0])[:3])
    world = affine.output
    if world is None or world.ndim is None:
        world = CoordinateSystem(name="world", axes=["x", "y", "z"])
    world = replace(
        world,
        axes=[replace(a, unit=SpaceUnit("centimeter")) for a in world.axes],
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
    """Every array type is stored as the nearest MGH type."""
    data = (_data() % 2).astype(dtype)
    target = tmp_path / "out.mgh"
    MghImage(data=data).save(target)
    loaded = nb.load(str(target))
    assert loaded.get_data_dtype() == np.dtype(">" + stored)
    assert np.array_equal(np.asarray(loaded.dataobj), data.astype(stored))


def test_unstorable_values_are_refused() -> None:
    """Integers beyond int32, complex numbers and 5D arrays are refused."""
    with pytest.raises(WriterError):
        MghImage(data=np.full((2, 2, 2), 2**40, "int64")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=np.zeros((2, 2, 2), "complex64")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=np.zeros((2, 2, 2, 2, 2), "float32")).to_bytes()
    with pytest.raises(WriterError):
        MghImage(data=None).to_bytes()


def test_explicit_dtype_and_params(tmp_path) -> None:  # noqa: ANN001
    """Keyword arguments set the stored type and the footer fields."""
    target = tmp_path / "out.mgz"
    MghImage(data=_data()).save(target, dtype="int16", tr=42.0)
    loaded = nb.load(str(target))
    assert loaded.get_data_dtype() == np.dtype(">i2")
    assert float(loaded.header["tr"]) == pytest.approx(42.0)


def test_like_copies_mri_params(tmp_path) -> None:  # noqa: ANN001
    """`like` supplies the footer parameters, never the geometry."""
    template = _write(tmp_path, "template.mgz", _data(), tr=777.0)
    image = MghImage(data=_data(), transformations=[Scaling(scale=[2, 2, 2])])
    target = tmp_path / "out.mgz"
    image.save(target, like=template)
    loaded = nb.load(str(target))
    assert float(loaded.header["tr"]) == pytest.approx(777.0)
    assert np.allclose(np.diag(loaded.affine), [2, 2, 2, 1])


def test_a_field_is_unrepresentable() -> None:
    """A displacement field has no affine, so MGH cannot store it."""
    field = DisplacementField(field=np.zeros((4, 5, 6, 3), dtype="float32"))
    image = MghImage(data=_data(), transformations=[field])
    with pytest.raises(UnrepresentableTransformationError):
        image.to_bytes()


def test_mgh_to_nifti_keeps_scanner_geometry(tmp_path) -> None:  # noqa: ANN001
    """Converting to NIfTI writes the scanner RAS matrix as the sform."""
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
    """`io.save` picks the format from the name, both ways."""
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
        image.vox2ras, nb.load(str(source)).header.get_vox2ras(), atol=1e-5
    )


class _RemotePath(os.PathLike):
    """A path to remote storage, held in memory: `nibabel` cannot open
    it by name, since `os.fspath` raises (see test_io_nifti_remote)."""

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
    """A remote path is read and written through its own backend, as a
    stream, and never handed to `nibabel` by name."""
    source = _write(tmp_path, name, _data(), tags=TAGS)
    store = {f"s3://bucket/{name}": source.read_bytes()}
    image = MghImage.load(_RemotePath(f"s3://bucket/{name}", store))
    assert np.array_equal(np.asarray(image.data), _data())
    assert image.tags == TAGS
    target = _RemotePath(f"s3://bucket/out/{name}", store)
    image.save(target)
    raw = store[target.url]
    assert (raw[:2] == b"\x1f\x8b") == name.endswith(".mgz")
    again = MghImage.from_bytes(raw)
    assert np.array_equal(np.asarray(again.data), _data())
    assert np.allclose(again.vox2ras, image.vox2ras)
    assert again.tags == TAGS
