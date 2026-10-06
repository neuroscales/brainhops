"""
`FileBasedMetadata.load`: reading the metadata of a file without its
data, through the `FileBasedMetadata` registry (NIfTI, MGH, plain Zarr
and OME-Zarr, x5, ITK `.h5`, BIDS sidecars), and the `to_raw` /
`to_file` side.
"""

import gzip
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.metadata import Metadata  # noqa: E402
from brainhops.io.base._metadata_parser import MetadataParser  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    ParserContentError,
    ParserNotImplementedError,
)
from brainhops.io.images.freesurfer.mgh import (  # noqa: E402
    MghImage,
    MghMetadata,
)
from brainhops.io.images.nifti import NiftiMetadata  # noqa: E402
from brainhops.io.metadata import FileBasedMetadata  # noqa: E402
from brainhops.io.transformations.fsl.flirt import FlirtMetadata  # noqa: E402


def _nifti(path: Path) -> Path:
    values = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    nii = nb.Nifti1Image(values, np.eye(4))
    nii.header["descrip"] = b"a header"
    nii.header.set_slope_inter(0.5, 10.0)
    nb.save(nii, str(path))
    return path


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scan.nii", "scan.nii.gz"])
def test_nifti_metadata_is_read_from_the_header(tmp_path, name) -> None:  # noqa: ANN001
    meta = FileBasedMetadata.load(_nifti(tmp_path / name))
    assert type(meta) is NiftiMetadata
    assert meta.description == "a header"
    assert meta.data_type == np.int16
    # The scaling is read from the header, as stored.
    assert (meta.scale_slope, meta.scale_intercept) == (0.5, 10.0)
    assert isinstance(meta.raw, nb.Nifti1Header)
    assert not meta._changed_fields()


def test_every_entry_point_reads_the_same(tmp_path) -> None:  # noqa: ANN001
    path = _nifti(tmp_path / "scan.nii.gz")
    by_path = FileBasedMetadata.load(str(path))
    with open(path, "rb") as f:
        by_stream = FileBasedMetadata.load(f)
        assert f.tell() == 0
    by_bytes = FileBasedMetadata.load(path.read_bytes())
    by_class = NiftiMetadata.load(path)
    assert by_path == by_stream == by_bytes == by_class
    # A name says nothing: the content is sniffed.
    renamed = tmp_path / "scan.bin"
    renamed.write_bytes(gzip.decompress(path.read_bytes()))
    assert FileBasedMetadata.load(renamed) == by_path


def test_a_hint_selects_the_format(tmp_path) -> None:  # noqa: ANN001
    path = _nifti(tmp_path / "scan.nii")
    assert type(FileBasedMetadata.load(path, hint="nifti")) is NiftiMetadata
    with pytest.raises(ParserContentError):
        FileBasedMetadata.load(path, hint="mgh")


def test_metadata_files_stay_out_of_the_generic_load(tmp_path) -> None:  # noqa: ANN001
    path = _nifti(tmp_path / "scan.nii")
    # `io.load` still reads an image; the metadata formats have their
    # own registry.
    assert not isinstance(io.load(path), Metadata)
    assert NiftiMetadata in FileBasedMetadata._REGISTRY
    assert NiftiMetadata not in io.FileBasedObject._REGISTRY
    assert not issubclass(FileBasedMetadata, io.FileBasedObject)
    # The parsers own no registry: `FileBasedMetadata` dispatches.
    assert "_REGISTRY" not in vars(MetadataParser)
    assert type(FileBasedMetadata.load(path)) is NiftiMetadata


def test_an_unknown_file_is_refused(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "notes.txt"
    path.write_text("not metadata")
    with pytest.raises(ParserContentError):
        FileBasedMetadata.load(path)


def test_a_format_without_metadata_in_its_files_refuses(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    np.savetxt(path, np.eye(4))
    with pytest.raises(ParserNotImplementedError):
        FlirtMetadata.load(path)


@pytest.mark.parametrize("name", ["ItkMetadata", "OpaqueMetadata"])
def test_an_opaque_format_refuses(tmp_path, name) -> None:  # noqa: ANN001
    from brainhops.io.metadata import OpaqueMetadata
    from brainhops.io.transformations.itk._metadata import ItkMetadata

    cls = {"ItkMetadata": ItkMetadata, "OpaqueMetadata": OpaqueMetadata}
    path = _nifti(tmp_path / "a.nii")
    # Even a file that another format reads: the class decides, not the
    # file.
    with pytest.raises(ParserNotImplementedError, match=name):
        cls[name].load(path)


def test_load_is_resolved_by_the_bases() -> None:
    from brainhops.io.base._base import FormatDispatcher
    from brainhops.io.metadata import OpaqueMetadata

    def owner(cls: type) -> type:
        return next(c for c in cls.__mro__ if "load" in c.__dict__)

    # `FileBasedMetadata` dispatches; on the class of a format whose
    # files hold metadata, its `load` reads the file with the parser.
    assert owner(FileBasedMetadata) is FileBasedMetadata
    assert FileBasedMetadata._is_dispatcher()
    assert issubclass(FileBasedMetadata, FormatDispatcher)
    assert owner(NiftiMetadata) is FileBasedMetadata
    assert owner(MghMetadata) is FileBasedMetadata
    assert not NiftiMetadata._is_dispatcher()
    # A format whose files hold none refuses, explicitly.
    assert owner(FlirtMetadata) is FlirtMetadata
    assert owner(OpaqueMetadata) is OpaqueMetadata
    # The data model does no input or output.
    assert not hasattr(Metadata, "load")


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


def test_mgh_metadata_reads_the_footer_and_the_tags(tmp_path) -> None:  # noqa: ANN001
    image = MghImage(
        np.zeros((3, 4, 5), "float32"),
        metadata=Metadata(repetition_time=2.3, history=("recon-all -s bert",)),
    )
    path = tmp_path / "scan.mgz"
    image.save(str(path), on_loss="ignore")
    meta = FileBasedMetadata.load(path)
    assert type(meta) is MghMetadata
    assert meta.repetition_time == 2.3
    # The tags, after the voxels, are read for `history`.
    assert meta.history == ("recon-all -s bert",)
    with open(path, "rb") as f:
        assert MghMetadata.load(f).history == ("recon-all -s bert",)


def test_zarr_metadata_reads_and_writes_the_attributes(tmp_path) -> None:  # noqa: ANN001
    zarr = pytest.importorskip("brainhops.io.images.zarr")
    path = str(tmp_path / "plain.zarr")
    zarr.ZarrImage(
        np.zeros((3, 4, 5), "int16"),
        metadata=Metadata(description="plain", extra={"Lab": "x"}),
    ).save(path)
    meta = FileBasedMetadata.load(path)
    assert type(meta) is zarr.ZarrMetadata
    assert (meta.description, meta.extra) == ("plain", {"Lab": "x"})
    assert meta.data_type == np.int16  # from the array, not its data
    # The record is an object of its own on disk, so it is written back.
    meta.description = "edited"
    meta.extra = {}
    meta.to_file(path)
    again = FileBasedMetadata.load(path)
    assert (again.description, again.extra) == ("edited", {})
    assert np.asarray(io.load(path).data).shape == (3, 4, 5)


def test_ome_zarr_metadata_reads_the_pyramid(tmp_path) -> None:  # noqa: ANN001
    zarr = pytest.importorskip("brainhops.io.images.zarr")
    from brainhops.datamodel.axes import SpaceAxis

    path = str(tmp_path / "brain.ome.zarr")
    zarr.OmeZarrImage(
        images=[SingleScaleImage(np.zeros((4, 4, 4), "float32"))],
        axes=[SpaceAxis("x"), SpaceAxis("y"), SpaceAxis("z")],
        metadata=Metadata(name="brain"),
    ).save(path)
    meta = FileBasedMetadata.load(path)
    assert type(meta) is zarr.OmeZarrMetadata
    assert meta.name == "brain"
    assert meta.data_type == np.float32
    # Its record is written with the pyramid: no `to_file`.
    assert not hasattr(meta, "to_file")


def test_x5_metadata_reads_the_node(tmp_path) -> None:  # noqa: ANN001
    h5py = pytest.importorskip("h5py")
    from brainhops.io.transformations.x5 import X5Metadata

    path = tmp_path / "affine.x5"
    with h5py.File(path, "w") as f:
        f.attrs["Format"], f.attrs["Version"] = "X5", np.uint16(1)
        node = f.create_group("TransformGroup/0")
        node.attrs["Type"], node.attrs["SubType"] = "linear", "affine"
        node.attrs["Representation"] = "matrix"
        node.attrs["ArrayLength"] = 1
        node.attrs["Metadata"] = json.dumps(
            {"Description": "to MNI", "Lab": "x"}
        )
        node.create_dataset("Transform", data=np.eye(4)[None])
    meta = FileBasedMetadata.load(path)
    assert type(meta) is X5Metadata
    assert (meta.description, meta.extra) == ("to MNI", {"Lab": "x"})
    assert meta == io.load(path).metadata


def test_itk_h5_metadata_reads_the_root_header(tmp_path) -> None:  # noqa: ANN001
    h5py = pytest.importorskip("h5py")
    from brainhops.io.transformations.itk import ItkH5Metadata

    path = tmp_path / "affine.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("ITKVersion", data="5.4.0")
    meta = FileBasedMetadata.load(path)
    assert type(meta) is ItkH5Metadata
    assert [(g.name, g.version) for g in meta.generated_by] == [
        ("ITK", "5.4.0")
    ]


def test_a_bids_sidecar_reads_as_generic_metadata(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "sub-01_bold.json"
    path.write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}))
    meta = FileBasedMetadata.load(path)
    assert type(meta) is Metadata
    assert meta.repetition_time == 2.0
    assert meta.extra == {"TaskName": "rest"}
    assert FileBasedMetadata.load(BytesIO(path.read_bytes())) == meta


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_to_raw_encodes_into_a_copy(tmp_path) -> None:  # noqa: ANN001
    meta = FileBasedMetadata.load(_nifti(tmp_path / "scan.nii"))
    meta.description = "edited"
    record = meta.to_raw()
    assert record is not meta.raw
    assert record["descrip"].item() == b"edited"
    assert meta.raw["descrip"].item() == b"a header"


@pytest.mark.parametrize("cls", [NiftiMetadata, MghMetadata])
def test_formats_written_with_their_data_have_no_to_file(cls) -> None:  # noqa: ANN001
    # Their record is written with the data; only a format whose record
    # is an object of its own on disk (plain Zarr) defines `to_file`.
    assert issubclass(cls, FileBasedMetadata)
    assert not hasattr(cls, "to_file")
