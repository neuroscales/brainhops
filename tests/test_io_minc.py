"""Tests for MINC1 and MINC2 images.

nibabel cannot write MINC, so the fixtures are built from the specification:
MINC1 with scipy's netcdf_file and MINC2 with h5py.
"""

import gzip
import io as _io
import warnings

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")
h5py = pytest.importorskip("h5py")
netcdf_file = pytest.importorskip("scipy.io").netcdf_file

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.transformations import Affine, Scaling  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.common.minc import MincDimension  # noqa: E402
from brainhops.io.common.minc._utils import minc_version  # noqa: E402
from brainhops.io.images.minc import (  # noqa: E402
    Minc1Image,
    Minc2Image,
    MincImage,
)
from brainhops.io.images.nifti import NiftiImage  # noqa: E402

_C, _S = np.cos(0.3), np.sin(0.3)
STEPS = {"xspace": 1.5, "yspace": -2.0, "zspace": 3.0, "time": 2.5}
STARTS = {"xspace": -10.0, "yspace": 20.0, "zspace": -5.0, "time": 0.0}
COSINES = {
    "xspace": [_C, _S, 0.0],
    "yspace": [-_S, _C, 0.0],
    "zspace": [0.0, 0.0, 1.0],
}
LENGTHS = {"xspace": 6, "yspace": 5, "zspace": 4, "time": 3}


def _raw(dims):  # noqa: ANN001, ANN202
    shape = tuple(LENGTHS[d] for d in dims)
    return np.arange(np.prod(shape), dtype=np.int16).reshape(shape)


def _scales(dims):  # noqa: ANN001, ANN202
    """Per-slice image-min and image-max along the slowest dimension."""
    n = LENGTHS[dims[0]]
    return np.linspace(-1.0, 0.0, n), np.linspace(1.0, 2.0, n)


def _attrs(dim, geometry=True):  # noqa: ANN001, ANN202
    if not geometry:
        return {}
    attrs = {"step": STEPS[dim], "start": STARTS[dim]}
    if dim in COSINES:
        attrs["direction_cosines"] = np.asarray(COSINES[dim], dtype=float)
        attrs["units"] = "mm"
    else:
        attrs["units"] = "s"
    return attrs


def write_minc1(
    fileobj: str,
    dims: tuple = ("zspace", "yspace", "xspace"),
    dtype: str = "h",
    geometry: bool = True,
    scaled: bool = True,
) -> None:
    raw = _raw(dims)
    f = netcdf_file(fileobj, "w", version=1)
    for d in dims:
        f.createDimension(d, LENGTHS[d])
    for d in dims:
        var = f.createVariable(d, "i", ())
        var.spacing = b"regular__"
        for key, value in _attrs(d, geometry).items():
            setattr(
                var, key, value.encode() if isinstance(value, str) else value
            )
    image = f.createVariable("image", dtype, dims)
    if dtype == "h":
        image[:] = raw
        image.signtype = b"signed__"
        image.valid_range = np.array([0.0, float(raw.max())])
    else:
        image[:] = raw.astype(np.float32)
    if scaled:
        imin, imax = _scales(dims)
        mx = f.createVariable("image-max", "d", (dims[0],))
        mx[:] = imax
        mn = f.createVariable("image-min", "d", (dims[0],))
        mn[:] = imin
    f.close()


def write_minc2(
    filename: str,
    dims: tuple = ("zspace", "yspace", "xspace"),
    geometry: bool = True,
    scaled: bool = True,
    dtype: type = np.int16,
) -> None:
    raw = _raw(dims).astype(dtype)
    with h5py.File(filename, "w") as f:
        root = f.create_group("minc-2.0")
        root.attrs["minc_version"] = np.bytes_(b"2.0")
        group = root.create_group("dimensions")
        for d in dims:
            ds = group.create_dataset(d, data=np.int32(0))
            ds.attrs["spacing"] = np.bytes_(b"regular__")
            ds.attrs["length"] = np.int32(LENGTHS[d])
            for key, value in _attrs(d, geometry).items():
                if isinstance(value, str):
                    value = np.bytes_(value.encode())
                ds.attrs[key] = value
        image = root.create_group("image/0")
        ds = image.create_dataset("image", data=raw)
        ds.attrs["dimorder"] = np.bytes_(",".join(dims).encode())
        if np.dtype(dtype).kind in "iu":
            ds.attrs["valid_range"] = np.array([0.0, float(raw.max())])
        if scaled:
            imin, imax = _scales(dims)
            for name, values in (("image-min", imin), ("image-max", imax)):
                v = image.create_dataset(name, data=values)
                v.attrs["dimorder"] = np.bytes_(dims[0].encode())


def expected_voxels(dims):  # noqa: ANN001, ANN201
    """Real voxel values of the fixture, in file order."""
    raw = _raw(dims).astype(np.float64)
    imin, imax = _scales(dims)
    shape = (-1,) + (1,) * (len(dims) - 1)
    slope = (imax - imin) / raw.max()
    return raw * slope.reshape(shape) + imin.reshape(shape)


def expected_vox2world(dims):  # noqa: ANN001, ANN201
    """The vox2world matrix of the F-ordered array."""
    spatial = [d for d in reversed(dims) if d in COSINES]
    cos = np.array([COSINES[d] for d in spatial]).T
    matrix = np.eye(4)
    matrix[:3, :3] = cos * [STEPS[d] for d in spatial]
    matrix[:3, 3] = cos @ [STARTS[d] for d in spatial]
    return matrix


@pytest.fixture
def minc1(tmp_path):  # noqa: ANN001, ANN201
    filename = tmp_path / "t1.mnc"
    write_minc1(str(filename))
    return filename


@pytest.fixture
def minc2(tmp_path):  # noqa: ANN001, ANN201
    filename = tmp_path / "t2.mnc"
    write_minc2(str(filename))
    return filename


@pytest.fixture(params=[1, 2], ids=["minc1", "minc2"])
def anyminc(request, minc1, minc2):  # noqa: ANN001, ANN201
    return request.param, (minc1 if request.param == 1 else minc2)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_class_by_version(anyminc) -> None:  # noqa: ANN001
    version, filename = anyminc
    image = io.images.load(filename)
    assert type(image) is (Minc1Image if version == 1 else Minc2Image)
    assert image.version == version


def test_voxels_match_nibabel(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    image = io.images.load(filename)
    reference = nb.load(str(filename)).get_fdata()
    data = np.asarray(image.data)
    assert data.shape == (6, 5, 4)
    np.testing.assert_allclose(data, reference.T)
    np.testing.assert_allclose(
        data, expected_voxels(("zspace", "yspace", "xspace")).T
    )


def test_data_is_native_float(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    data = np.asarray(io.images.load(filename).data)
    assert data.dtype.isnative


def test_vox2world_matches_nibabel(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    image = io.images.load(filename)
    reference = nb.load(str(filename)).affine
    # nibabel's affine indexes the C-ordered array; ours is F-ordered.
    flip = np.eye(4)[[2, 1, 0, 3]]
    np.testing.assert_allclose(image.vox2world, reference @ flip)
    np.testing.assert_allclose(
        image.vox2world, expected_vox2world(("zspace", "yspace", "xspace"))
    )


def test_transformations(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    image = io.images.load(filename)
    phys, world = image.transformations
    assert isinstance(phys, Scaling)
    np.testing.assert_allclose(phys.scale, [1.5, 2.0, 3.0])
    assert phys.output.name == "physical"
    assert [str(a.unit) for a in phys.output.axes] == ["millimeter"] * 3
    assert isinstance(world, Affine)
    assert world.output.name == "world"
    assert [a.name for a in world.output.axes] == ["x", "y", "z"]
    assert image.transformation is not None
    np.testing.assert_allclose(
        image.transformation.matrix, image.vox2world[:3]
    )
    assert image.system.order == "F"
    assert [a.name for a in image.system.axes] == ["x", "y", "z"]


def test_dimensions_are_kept(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    image = io.images.load(filename)
    names = [d.name for d in image.dimensions]
    assert names == ["zspace", "yspace", "xspace"]
    y = image.dimensions[1]
    assert isinstance(y, MincDimension)
    assert y.step == -2.0 and y.start == 20.0 and y.units == "mm"
    np.testing.assert_allclose(y.direction_cosines, COSINES["yspace"])


@pytest.mark.parametrize("version", [1, 2])
def test_permuted_dimensions(tmp_path, version) -> None:  # noqa: ANN001
    """Axes of a sagittal file are named after the MINC dimensions."""
    dims = ("xspace", "zspace", "yspace")
    filename = str(tmp_path / "sag.mnc")
    (write_minc1 if version == 1 else write_minc2)(filename, dims=dims)
    image = io.images.load(filename)
    assert [a.name for a in image.system.axes] == ["y", "z", "x"]
    assert image.data.shape == (5, 4, 6)
    np.testing.assert_allclose(np.asarray(image.data), expected_voxels(dims).T)
    reference = nb.load(filename).affine
    flip = np.eye(4)[[2, 1, 0, 3]]
    np.testing.assert_allclose(image.vox2world, reference @ flip)
    np.testing.assert_allclose(image.vox2world, expected_vox2world(dims))


@pytest.mark.parametrize("version", [1, 2])
def test_time_dimension(tmp_path, version) -> None:  # noqa: ANN001
    """Time is an axis of both spaces, in seconds, and does not move points."""
    dims = ("time", "zspace", "yspace", "xspace")
    filename = str(tmp_path / "4d.mnc")
    (write_minc1 if version == 1 else write_minc2)(filename, dims=dims)
    image = io.images.load(filename)
    assert image.data.shape == (6, 5, 4, 3)
    assert [a.name for a in image.system.axes] == ["x", "y", "z", "t"]
    np.testing.assert_allclose(np.asarray(image.data), expected_voxels(dims).T)
    phys, world = image.transformations
    np.testing.assert_allclose(phys.scale, [1.5, 2.0, 3.0, 2.5])
    assert str(phys.output.axes[3].unit) == "second"
    expected = expected_vox2world(dims)
    np.testing.assert_allclose(world.matrix[:, :3], expected[:3, :3])
    np.testing.assert_allclose(world.matrix[:, 3], 0)
    np.testing.assert_allclose(world.matrix[:, 4], expected[:3, 3])


@pytest.mark.parametrize("version", [1, 2])
def test_default_geometry(tmp_path, version) -> None:  # noqa: ANN001
    """Without geometry, MINC defaults to unit steps in mm from the origin."""
    filename = str(tmp_path / "plain.mnc")
    (write_minc1 if version == 1 else write_minc2)(filename, geometry=False)
    image = io.images.load(filename)
    np.testing.assert_allclose(image.vox2world, np.eye(4))
    np.testing.assert_allclose(image.transformations[0].scale, [1, 1, 1])
    units = [str(a.unit) for a in image.transformations[0].output.axes]
    assert units == ["millimeter"] * 3


def test_minc1_float(tmp_path) -> None:  # noqa: ANN001
    filename = str(tmp_path / "float.mnc")
    write_minc1(filename, dtype="f")
    data = np.asarray(io.images.load(filename).data)
    assert data.dtype == np.float32 and data.dtype.isnative
    np.testing.assert_array_equal(data, _raw(("zspace", "yspace", "xspace")).T)


def test_minc2_float(tmp_path) -> None:  # noqa: ANN001
    filename = str(tmp_path / "float.mnc")
    write_minc2(filename, dtype=np.float32)
    data = np.asarray(io.images.load(filename).data)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, _raw(("zspace", "yspace", "xspace")).T)


@pytest.mark.parametrize("version", [1, 2])
def test_unscaled_is_unsupported(tmp_path, version) -> None:  # noqa: ANN001
    """nibabel needs image-min and image-max; their absence is reported."""
    filename = str(tmp_path / "unscaled.mnc")
    (write_minc1 if version == 1 else write_minc2)(filename, scaled=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ParserContentError, match="image-m"):
            MincImage.load(filename)


def test_minc1_gzipped(tmp_path, minc1) -> None:  # noqa: ANN001
    filename = tmp_path / "t1.mnc.gz"
    filename.write_bytes(gzip.compress(minc1.read_bytes()))
    image = io.images.load(filename)
    assert type(image) is Minc1Image
    np.testing.assert_allclose(
        np.asarray(image.data), np.asarray(io.images.load(minc1).data)
    )


@pytest.mark.parametrize("how", ["bytes", "stream"])
def test_from_memory(anyminc, how) -> None:  # noqa: ANN001
    version, filename = anyminc
    content = filename.read_bytes()
    source = content if how == "bytes" else _io.BytesIO(content)
    image = MincImage.load(source)
    assert type(image) is (Minc1Image if version == 1 else Minc2Image)
    np.testing.assert_allclose(
        np.asarray(image.data),
        expected_voxels(("zspace", "yspace", "xspace")).T,
    )


def test_name_does_not_matter(tmp_path, anyminc) -> None:  # noqa: ANN001
    version, filename = anyminc
    other = tmp_path / "volume.bin"
    other.write_bytes(filename.read_bytes())
    image = io.images.load(other)
    assert type(image) is (Minc1Image if version == 1 else Minc2Image)


def test_header_only_until_data(anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = io.images.load(filename)
        assert getattr(image, "_data", None) is None
        assert image.shape == (6, 5, 4)
        image.data  # noqa: B018
        assert getattr(image, "_data", None) is not None


def test_convert_to_nifti(tmp_path, anyminc) -> None:  # noqa: ANN001
    _, filename = anyminc
    image = io.images.load(filename)
    out = tmp_path / "out.nii.gz"
    io.save(image, out)
    nifti = io.images.load(out)
    assert isinstance(nifti, NiftiImage)
    np.testing.assert_allclose(
        np.asarray(nifti.data), np.asarray(image.data), rtol=1e-6
    )
    np.testing.assert_allclose(
        nifti.transformation.matrix, image.vox2world[:3], atol=1e-5
    )


def test_not_writable(tmp_path, minc1) -> None:  # noqa: ANN001
    image = io.images.load(minc1)
    assert not hasattr(image, "save")
    with pytest.raises(WriterError):
        io.save(image, tmp_path / "copy.mnc")


# ----------------------------------------------------------------------
#   SNIFFING AND DISPATCH
# ----------------------------------------------------------------------


def test_minc_version() -> None:
    assert minc_version(b"CDF\x01rest") == 1
    assert minc_version(b"CDF\x02rest") == 1
    assert minc_version(b"\x89HDF\r\n\x1a\nrest") == 2
    assert minc_version(b"\x00" * 8) is None


def test_sniff_by_version(minc1, minc2) -> None:  # noqa: ANN001
    assert Minc1Image.sniff(minc1) == Confidence.CERTAIN
    assert Minc2Image.sniff(minc1) == Confidence.NO
    assert Minc2Image.sniff(minc2) == Confidence.CERTAIN
    assert Minc1Image.sniff(minc2) == Confidence.NO
    assert MincImage.sniff(minc1) == MincImage.sniff(minc2) > 0
    with open(minc2, "rb") as f:
        assert Minc2Image.sniff(f) == Confidence.CERTAIN
        assert f.tell() == 0
    assert Minc2Image.sniff(minc2.read_bytes()) == Confidence.CERTAIN


def test_sniff_rejects_other_hdf5(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "other.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=np.zeros(3))
    assert Minc2Image.sniff(filename) == Confidence.NO
    with pytest.raises(SnifferContentError):
        Minc2Image.sniff(filename, error=True)


def test_sniff_plain_netcdf_is_weak(tmp_path) -> None:  # noqa: ANN001
    filename = str(tmp_path / "plain.nc")
    f = netcdf_file(filename, "w", version=1)
    f.createDimension("n", 3)
    f.createVariable("values", "d", ("n",))[:] = [1, 2, 3]
    f.close()
    assert Minc1Image.sniff(filename) == Confidence.WEAK


def test_sniff_rejects_nifti(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "x.nii"
    nb.save(nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), filename)
    for klass in (Minc1Image, Minc2Image, MincImage):
        assert klass.sniff(filename) == Confidence.NO


@pytest.mark.parametrize(
    "hint, klass",
    [
        ("minc", None),
        ("minc.1", Minc1Image),
        ("minc1", Minc1Image),
        ("minc.2", Minc2Image),
        ("minc2", Minc2Image),
    ],
)
def test_hints(anyminc, hint, klass) -> None:  # noqa: ANN001
    version, filename = anyminc
    expected = Minc1Image if version == 1 else Minc2Image
    if klass is None or klass is expected:
        assert type(io.images.load(filename, hint=hint)) is expected
    else:
        with pytest.raises(ParserContentError):
            io.images.load(filename, hint=hint)


def test_variant_refuses_other_version(minc1, minc2) -> None:  # noqa: ANN001
    with pytest.raises(ParserContentError):
        Minc2Image.load(minc1)
    with pytest.raises(ParserContentError):
        Minc1Image.load(minc2)


def test_generic_load(anyminc) -> None:  # noqa: ANN001
    version, filename = anyminc
    image = io.load(filename)
    assert type(image) is (Minc1Image if version == 1 else Minc2Image)


def test_unsupported_vector_dimension(tmp_path) -> None:  # noqa: ANN001
    """A dimension without a variable, such as vector_dimension, fails."""
    filename = str(tmp_path / "rgb.mnc")
    f = netcdf_file(filename, "w", version=1)
    for d, n in (("zspace", 2), ("yspace", 2), ("xspace", 2)):
        f.createDimension(d, n)
        f.createVariable(d, "i", ())
    f.createDimension("vector_dimension", 3)
    image = f.createVariable(
        "image", "f", ("zspace", "yspace", "xspace", "vector_dimension")
    )
    image[:] = np.zeros((2, 2, 2, 3), dtype=np.float32)
    f.close()
    with pytest.raises(ParserContentError, match="vector_dimension"):
        Minc1Image.load(filename)
