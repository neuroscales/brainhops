"""Tests of the lazy reading and writing of chunked files under Dask.

Under the backend setting `"dask"`, a format whose storage can be read
in part gives a Dask array whose blocks are the units that the storage
reads at once, such as the chunks of a Zarr array, and computing it
reads only those blocks. A writer of a Zarr store writes a lazy array
block by block, and the new store keeps the chunks of the source.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

da = pytest.importorskip("dask.array")
dask = pytest.importorskip("dask")

import brainhops.io as io  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402

# ----------------------------------------------------------------------
#   OME-ZARR FIELDS
# ----------------------------------------------------------------------

FIELD_SHAPE = (32, 24, 16, 3)
"""The shape of a field stored as OME-Zarr, vector axis last."""

FIELD_CHUNKS = (8, 8, 8, 3)
"""The chunks of the stored field, which divide its spatial axes."""


def _field_values() -> np.ndarray:
    """Return displacements whose values differ at every voxel."""
    count = int(np.prod(FIELD_SHAPE))
    return (np.arange(count, dtype="float32") / count).reshape(FIELD_SHAPE)


def _write_zarr_field(path: Path) -> str:
    """Write a displacement field of one level, chunked, as OME-Zarr."""
    abczarr = pytest.importorskip("abczarr")
    from abczarr.ome import v0_6 as v6

    location = str(path)
    group = abczarr.open_group(location, mode="w")
    group.create_array("0", data=_field_values(), chunks=FIELD_CHUNKS)
    axes = [
        ("z", "space"),
        ("y", "space"),
        ("x", "space"),
        ("c", "displacement"),
    ]
    group.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {
                            "name": "field",
                            "axes": [
                                {"name": name, "type": kind}
                                for name, kind in axes
                            ],
                        }
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [2.0, 3.0, 4.0, 1.0],
                                    "input": {"path": "0"},
                                    "output": {"name": "field"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    return location


@pytest.fixture
def zarr_field(tmp_path: Path) -> str:
    """The location of a chunked OME-Zarr displacement field."""
    return _write_zarr_field(tmp_path / "field.zarr")


@pytest.fixture
def read_boxes(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the shape of every box that a lazy Zarr array reads."""
    from brainhops.io.common.zarr._data import _ZarrProxy

    boxes = []
    read = _ZarrProxy._read

    def recording(self, index):  # noqa: ANN001, ANN202
        values = read(self, index)
        boxes.append(values.shape)
        return values

    monkeypatch.setattr(_ZarrProxy, "_read", recording)
    return boxes


def test_zarr_field_levels_are_lazy_with_the_chunks_of_the_store(
    zarr_field: str, read_boxes: list
) -> None:
    from brainhops.io.transformations.zarr import OmeZarrField

    with backend("dask"):
        field = OmeZarrField.from_store(zarr_field)
        level = field.scales[0]
        displacements = level.transformations[1].data
    assert read_boxes == []
    assert field.raw_levels[0].chunks == FIELD_CHUNKS
    assert isinstance(displacements, da.Array)
    assert displacements.chunksize == FIELD_CHUNKS


def test_zarr_field_values_do_not_depend_on_the_backend(
    zarr_field: str,
) -> None:
    from brainhops.io.transformations.zarr import OmeZarrField

    with backend("numpy"):
        eager = OmeZarrField.from_store(zarr_field).scales[0]
        expected = np.asarray(eager.transformations[1].data)
    with backend("dask"):
        lazy = OmeZarrField.from_store(zarr_field).scales[0]
        values = lazy.transformations[1].data
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(values.compute(), expected, rtol=1e-6)


def test_zarr_field_is_written_chunk_by_chunk(
    zarr_field: str, read_boxes: list, tmp_path: Path
) -> None:
    import abczarr

    from brainhops.io.transformations.zarr import OmeZarrField

    copy = str(tmp_path / "copy.zarr")
    with backend("dask"):
        OmeZarrField.from_store(zarr_field).to_store(copy)
    # No box larger than a chunk is read, so the field is never read
    # whole into memory.
    assert read_boxes
    assert all(
        box[i] <= FIELD_CHUNKS[i] for box in read_boxes for i in range(4)
    )
    written = abczarr.open_group(copy, mode="r")["0"]
    assert tuple(written.chunks) == FIELD_CHUNKS
    np.testing.assert_array_equal(np.asarray(written[...]), _field_values())


def test_zarr_field_keeps_the_chunks_of_a_dask_level(tmp_path: Path) -> None:
    import abczarr

    from brainhops.io.transformations.zarr import OmeZarrField

    location = _write_zarr_field(tmp_path / "field.zarr")
    field = OmeZarrField.from_store(location)
    values = da.from_array(_field_values(), chunks=(16, 12, 8, 3))
    field.raw_levels = [values]
    copy = str(tmp_path / "copy.zarr")
    field.to_store(copy)
    written = abczarr.open_group(copy, mode="r")["0"]
    assert tuple(written.chunks) == (16, 12, 8, 3)
    np.testing.assert_array_equal(np.asarray(written[...]), _field_values())


def test_zarr_field_is_written_over_the_store_that_it_reads(
    zarr_field: str,
) -> None:
    import abczarr

    from brainhops.io.transformations.zarr import OmeZarrField

    with backend("dask"):
        OmeZarrField.from_store(zarr_field).to_store(zarr_field)
    written = abczarr.open_group(zarr_field, mode="r")["0"]
    np.testing.assert_array_equal(np.asarray(written[...]), _field_values())


# ----------------------------------------------------------------------
#   OME-ZARR IMAGES
# ----------------------------------------------------------------------


def test_zarr_image_pyramid_keeps_the_chunks_of_its_store(
    tmp_path: Path,
) -> None:
    abczarr = pytest.importorskip("abczarr")
    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.zarr import OmeZarrImage

    values = np.random.default_rng(0).random((8, 10, 12)).astype("float32")
    pyramid = OmeZarrImage(images=[SingleScaleImage(data=values)])
    source = str(tmp_path / "source.zarr")
    pyramid.save(source, chunks=(4, 5, 6))
    copy = str(tmp_path / "copy.zarr")
    with backend("dask"):
        io.images.load(source).save(copy)
    written = abczarr.open_group(copy, mode="r")["0"]
    original = abczarr.open_group(source, mode="r")["0"]
    assert tuple(written.chunks) == tuple(original.chunks)
    np.testing.assert_array_equal(
        np.asarray(written[...]), np.asarray(original[...])
    )


def test_zarr_image_pyramid_is_written_over_the_store_that_it_reads(
    tmp_path: Path,
) -> None:
    pytest.importorskip("abczarr")
    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.zarr import OmeZarrImage

    values = np.random.default_rng(0).random((8, 10, 12)).astype("float32")
    location = str(tmp_path / "image.zarr")
    OmeZarrImage(images=[SingleScaleImage(data=values)]).save(location)
    io.images.load(location).save(location)
    again = io.images.load(location)
    np.testing.assert_array_equal(np.asarray(again.images[0].data), values)


# ----------------------------------------------------------------------
#   NIFTI AND MGH
# ----------------------------------------------------------------------

SPATIAL = (6, 5, 8)
"""The spatial shape of the NIfTI and MGH files."""


def _slab_chunks(shape: tuple, itemsize: int, budget: int) -> tuple:
    """Return whole slices along the third axis, as many as fit."""
    slice_bytes = int(np.prod(shape[:2] + shape[3:])) * itemsize
    slices = max(1, budget // slice_bytes)
    return shape[:2] + (min(slices, shape[2]),) + shape[3:]


@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
def test_nifti_image_is_read_in_whole_slices(
    tmp_path: Path, extension: str
) -> None:
    nb = pytest.importorskip("nibabel")
    values = np.random.default_rng(0).random(SPATIAL + (2,))
    values = values.astype("float32")
    path = tmp_path / f"image{extension}"
    nb.save(nb.Nifti1Image(values, np.eye(4)), str(path))
    budget = 3 * 6 * 5 * 2 * 4
    with dask.config.set({"array.chunk-size": budget}), backend("dask"):
        image = io.images.load(path)
        data = image.data
    assert isinstance(data, da.Array)
    assert data.chunksize == _slab_chunks(values.shape, 4, budget)
    np.testing.assert_array_equal(data.compute(), values)


def test_mgh_image_is_read_in_whole_slices(tmp_path: Path) -> None:
    nb = pytest.importorskip("nibabel")
    values = np.arange(np.prod(SPATIAL), dtype="float32").reshape(SPATIAL)
    path = tmp_path / "image.mgz"
    nb.save(nb.MGHImage(values, np.eye(4)), str(path))
    budget = 2 * 6 * 5 * 4
    with dask.config.set({"array.chunk-size": budget}), backend("dask"):
        data = io.images.load(path).data
    assert isinstance(data, da.Array)
    assert data.chunksize == (6, 5, 2)
    np.testing.assert_array_equal(data.compute(), values)


def test_nifti_field_is_read_in_whole_slices(tmp_path: Path) -> None:
    nb = pytest.importorskip("nibabel")
    from brainhops.io.transformations.nifti import NiftiRASDisplacementField

    vectors = np.random.default_rng(0).random(SPATIAL + (1, 3))
    vectors = vectors.astype("float32")
    image = nb.Nifti1Image(vectors, np.diag([2.0, 3.0, 4.0, 1.0]))
    image.header.set_intent(1006)
    path = tmp_path / "field.nii.gz"
    nb.save(image, str(path))
    budget = 4 * 6 * 5 * 3 * 4
    with dask.config.set({"array.chunk-size": budget}), backend("dask"):
        field = NiftiRASDisplacementField.from_filename(path)
        stored = field.data
        displacements = field.transformations[1].data
    assert isinstance(stored, da.Array)
    assert stored.chunksize == (6, 5, 4, 3)
    # The rotation into voxel units is applied block by block.
    assert isinstance(displacements, da.Array)
    assert displacements.chunksize == (6, 5, 4, 3)
    expected = vectors[:, :, :, 0, :] / np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(displacements.compute(), expected, rtol=1e-6)


# ----------------------------------------------------------------------
#   N5 AND NEUROGLANCER PRECOMPUTED
# ----------------------------------------------------------------------


def test_n5_dataset_is_read_in_its_blocks(tmp_path: Path) -> None:
    ts = pytest.importorskip("tensorstore")
    values = np.arange(10 * 9 * 8, dtype="uint16").reshape(10, 9, 8)
    spec = {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(tmp_path / "d") + os.sep},
        "metadata": {
            "dimensions": list(values.shape),
            "blockSize": [4, 3, 8],
            "dataType": "uint16",
            "compression": {"type": "raw"},
        },
        "create": True,
    }
    ts.open(spec).result().write(values).result()
    with backend("dask"):
        data = io.images.load(tmp_path / "d").data
    assert isinstance(data, da.Array)
    assert data.chunksize == (4, 3, 8)
    np.testing.assert_array_equal(data.compute(), values)


def test_precomputed_scale_is_read_in_its_chunks(tmp_path: Path) -> None:
    pytest.importorskip("tensorstore")
    values = np.arange(6 * 5 * 4 * 2, dtype="uint16").reshape(6, 5, 4, 2)
    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "image",
        "data_type": "uint16",
        "num_channels": 2,
        "scales": [
            {
                "key": "s0",
                "size": [6, 5, 4],
                "resolution": [1, 1, 1],
                "chunk_sizes": [[3, 5, 2]],
                "encoding": "raw",
            }
        ],
    }
    directory = tmp_path / "volume"
    os.makedirs(directory / "s0")
    (directory / "info").write_text(json.dumps(info))
    for x in (0, 3):
        for z in (0, 2):
            block = values[x : x + 3, :, z : z + 2]
            name = f"{x}-{x + 3}_0-5_{z}-{z + 2}"
            (directory / "s0" / name).write_bytes(
                block.astype("<u2").tobytes(order="F")
            )
    with backend("dask"):
        data = io.images.load(directory).data
    assert isinstance(data, da.Array)
    assert data.chunksize == (3, 5, 2, 2)
    np.testing.assert_array_equal(data.compute(), values)


# ----------------------------------------------------------------------
#   HDF5: BIGDATAVIEWER, X5 AND ITK
# ----------------------------------------------------------------------

_BDV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5">
      <hdf5 type="relative">dataset.h5</hdf5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>0</name>
        <size>8 6 4</size>
        <voxelSize>
          <unit>um</unit>
          <size>1.0 1.0 1.0</size>
        </voxelSize>
      </ViewSetup>
    </ViewSetups>
    <Timepoints type="range">
      <first>0</first>
      <last>0</last>
    </Timepoints>
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <affine>1 0 0 0 0 1 0 0 0 0 1 0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""


def test_bdv_view_is_read_in_its_hdf5_chunks(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    values = np.arange(8 * 6 * 4, dtype="uint16").reshape(8, 6, 4)
    (tmp_path / "dataset.xml").write_text(_BDV_XML)
    with h5py.File(tmp_path / "dataset.h5", "w") as f:
        f.create_dataset("s00/resolutions", data=[[1.0, 1.0, 1.0]])
        f.create_dataset("s00/subdivisions", data=[[4, 3, 2]])
        f["s00"].attrs["dataType"] = np.bytes_(b"uint16\x00")
        # BigDataViewer stores the voxels in the order z, y, x.
        f.create_dataset("t00000/s00/0/cells", data=values.T, chunks=(2, 3, 4))
    with backend("dask"):
        data = io.images.load(tmp_path / "dataset.xml").data
    assert isinstance(data, da.Array)
    assert data.chunksize == (4, 3, 2)
    np.testing.assert_array_equal(data.compute(), values)


def test_x5_field_is_read_in_its_hdf5_chunks(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    vectors = np.random.default_rng(0).random(SPATIAL + (3,))
    vectors = vectors.astype("float32")
    vox2ras = np.diag([2.0, 3.0, 4.0, 1.0])
    path = tmp_path / "warp.x5"
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = "X5"
        f.attrs["Version"] = np.uint16(1)
        node = f.create_group("TransformGroup/0")
        node.attrs["Type"] = "nonlinear"
        node.attrs["ArrayLength"] = 1
        node.attrs["SubType"] = "densefield"
        node.attrs["Representation"] = "displacements"
        node.create_dataset("Transform", data=vectors, chunks=(3, 5, 4, 3))
        node.create_dataset(
            "DimensionKinds",
            data=np.asarray(["space", "space", "space", "vector"], "S"),
        )
        domain = node.create_group("Domain")
        domain.create_dataset("Grid", data=np.uint8(1))
        domain.create_dataset("Size", data=np.asarray(SPATIAL))
        domain.create_dataset("Mapping", data=vox2ras)
        domain.attrs["Coordinates"] = "cartesian"
    with backend("dask"):
        chain = io.transformations.load(path)[0]
        displacements = chain.displacement.data
    assert isinstance(displacements, da.Array)
    assert displacements.chunksize == (3, 5, 4, 3)
    expected = vectors / np.array([2.0, 3.0, 4.0], dtype="float32")
    np.testing.assert_allclose(displacements.compute(), expected, rtol=1e-6)


def test_itk_h5_field_is_read_in_its_hdf5_chunks(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source = Path(__file__).parent / "data" / "itk_displacement3d.h5"
    with backend("numpy"):
        expected = np.asarray(io.transformations.load(source)[-1].field)
    # The same file, with its parameters stored in chunks of 8 slices of
    # the slowest axis of the buffer, which is the third axis of the field.
    path = tmp_path / "chunked.h5"
    with h5py.File(source, "r") as old, h5py.File(path, "w") as new:
        for key in old:
            old.copy(old[key], new, name=key)
        name = "TransformGroup/0/TransformParameters"
        parameters = np.asarray(new[name][()])
        del new[name]
        chunk = 8 * expected.shape[0] * expected.shape[1] * 3
        new.create_dataset(name, data=parameters, chunks=(chunk,))
    with backend("dask"):
        field = io.transformations.load(path)[-1].field
    assert isinstance(field, da.Array)
    assert field.chunksize[2] == 8
    np.testing.assert_allclose(field.compute(), expected)
