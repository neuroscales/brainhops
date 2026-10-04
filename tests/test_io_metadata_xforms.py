"""
The metadata of transformation formats: X5, ITK (`.tfm`, `.mat`, `.h5`)
and FSL FLIRT.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import fields


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, on_loss=report, **kwargs), report


h5py = pytest.importorskip("h5py")
nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import systems  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel.metadata import (  # noqa: E402
    UNSUPPORTED,
    ConversionReport,
    GeneratedBy,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    OpaqueMetadata,
    metadata_loss_policy,
)
from brainhops.io.base._base import FileBasedObject  # noqa: E402
from brainhops.io.transformations.fsl.flirt import (  # noqa: E402
    FlirtMatrixParser,
    FlirtMetadata,
    FlirtTransform,
)
from brainhops.io.transformations.itk import (  # noqa: E402
    ItkH5Metadata,
    ItkMetadata,
    ItkTransform,
)
from brainhops.io.transformations.itk.h5 import H5Transform  # noqa: E402
from brainhops.io.transformations.itk.mat import MatTransform  # noqa: E402
from brainhops.io.transformations.itk.tfm import TfmTransform  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiMetadata,
    NiftiRASDisplacementField,
)
from brainhops.io.transformations.x5 import (  # noqa: E402
    X5Metadata,
    X5Transform,
    X5TransformParser,
)

SHAPE = (3, 4, 5)

JSON = {
    "Description": "sub-01 T1w to MNI",
    "GeneratedBy": [{"Name": "fMRIPrep", "Version": "24.1.0"}],
    "History": ["antsRegistration ..."],
    "Moving": "sub-01_T1w.nii.gz",
    "Fixed": "tpl-MNI152NLin2009cAsym_T1w.nii.gz",
    "InputSpace": "T1w",
    "OutputSpace": "MNI152NLin2009cAsym",
    "WrittenBy": "NiTransforms 25.1.0",
}


def _write_x5(path: Path, nodes: list, chains: tuple = ()) -> Path:
    """An X5 file in nitransforms' layout; `nodes` are `(kind, json)`."""
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = "X5"
        f.attrs["Version"] = np.uint16(1)
        group = f.create_group("TransformGroup")
        for i, (kind, metadata) in enumerate(nodes):
            node = group.create_group(str(i))
            node.attrs["ArrayLength"] = 1
            if kind == "linear":
                node.attrs["Type"] = "linear"
                node.attrs["SubType"] = "affine"
                node.attrs["Representation"] = "matrix"
                node.create_dataset("Transform", data=np.eye(4))
            else:
                node.attrs["Type"] = "nonlinear"
                node.attrs["SubType"] = "densefield"
                node.attrs["Representation"] = "displacements"
                node.create_dataset(
                    "Transform", data=np.zeros(SHAPE + (3,), "float32")
                )
                kinds = ("space",) * 3 + ("vector",)
                node.create_dataset(
                    "DimensionKinds", data=np.asarray(kinds, dtype="S")
                )
                domain = node.create_group("Domain")
                domain.create_dataset("Grid", data=np.uint8(1))
                domain.create_dataset("Size", data=np.asarray(SHAPE))
                domain.create_dataset("Mapping", data=np.eye(4))
                domain.attrs["Coordinates"] = "cartesian"
            if metadata is not None:
                node.attrs["Metadata"] = (
                    metadata
                    if isinstance(metadata, str)
                    else json.dumps(metadata)
                )
        if chains:
            group = f.create_group("TransformChain")
            for i, chain in enumerate(chains):
                group.create_dataset(str(i), data="/".join(map(str, chain)))
    return path


def _metadata_attr(path: Path, index: int = 0) -> str:
    with h5py.File(path, "r") as f:
        value = f[f"TransformGroup/{index}"].attrs.get("Metadata")
    return value.decode() if isinstance(value, bytes) else value


@pytest.fixture
def warp_x5(tmp_path: Path) -> Path:
    return _write_x5(tmp_path / "warp.x5", [("field", JSON)])


# ----------------------------------------------------------------------
#   X5
# ----------------------------------------------------------------------


def test_x5_decodes_the_node_json(warp_x5: Path) -> None:
    xform = io.load(warp_x5)
    meta = xform.metadata
    assert type(meta) is X5Metadata
    assert meta.description == "sub-01 T1w to MNI"
    assert meta.generated_by == (
        GeneratedBy(name="fMRIPrep", version="24.1.0"),
    )
    assert meta.history == ("antsRegistration ...",)
    assert meta.moving == "sub-01_T1w.nii.gz"
    assert meta.fixed == "tpl-MNI152NLin2009cAsym_T1w.nii.gz"
    assert meta.input_space == "T1w"
    assert meta.output_space == "MNI152NLin2009cAsym"
    assert meta.extra == {"WrittenBy": "NiTransforms 25.1.0"}
    assert meta.node is xform.nodes[0]
    assert meta.header is xform.header
    assert meta._changed_fields() == {}
    assert not X5Metadata.unsupported_fields


def test_x5_untouched_round_trip_keeps_the_json_string(
    warp_x5: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.x5"
    io.load(warp_x5).save(out, on_loss="raise")
    assert _metadata_attr(out) == _metadata_attr(warp_x5)


def test_x5_edits_are_written_into_the_node(
    warp_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(warp_x5)
    xform.metadata.description = "edited"
    xform.metadata.history = None
    xform.metadata.extra["Note"] = "kept"
    out = tmp_path / "out.x5"
    xform.save(out, on_loss="raise")
    written = json.loads(_metadata_attr(out))
    assert written["Description"] == "edited"
    assert "History" not in written
    assert written["Note"] == "kept"
    assert written["WrittenBy"] == "NiTransforms 25.1.0"
    again = io.load(out)
    assert again.metadata.description == "edited"
    assert again.metadata.history is None
    assert again.metadata.extra == {
        "WrittenBy": "NiTransforms 25.1.0",
        "Note": "kept",
    }


def test_x5_node_without_metadata(tmp_path: Path) -> None:
    path = _write_x5(tmp_path / "bare.x5", [("linear", None)])
    xform = io.load(path)
    assert xform.metadata.description is None
    out = tmp_path / "out.x5"
    xform.save(out)
    assert _metadata_attr(out) is None
    xform.metadata.description = "now"
    xform.save(out)
    assert json.loads(_metadata_attr(out)) == {"Description": "now"}


def test_x5_chain_has_no_metadata_of_its_own(tmp_path: Path) -> None:
    path = _write_x5(
        tmp_path / "chain.x5",
        [("linear", {"Description": "a"}), ("field", {"Description": "b"})],
        chains=[(0, 1)],
    )
    xform = io.load(path)
    assert xform.metadata.raw[1] is None
    assert xform.metadata.description is None
    out = tmp_path / "out.x5"
    xform.save(out, on_loss="raise")
    assert json.loads(_metadata_attr(out, 0)) == {"Description": "a"}
    assert json.loads(_metadata_attr(out, 1)) == {"Description": "b"}
    # A chain has no node to hold its own metadata.
    xform.metadata.description = "the chain"
    with pytest.raises(MetadataLossError, match="description"):
        xform.save(out, on_loss="raise")
    # One node of the chain carries its own.
    one = X5Transform.from_file(path, position=1)
    assert one.metadata.description == "b"


def test_x5_built_in_memory_writes_its_metadata(tmp_path: Path) -> None:
    affine = xforms.Affine(
        np.eye(4)[:3], input=systems.RASmm(), output=systems.RASmm()
    )
    xform = X5Transform(
        transformations=[affine],
        metadata=Metadata(description="in memory", moving="a.nii"),
    )
    assert type(xform.metadata) is X5Metadata
    out = tmp_path / "mem.x5"
    xform.save(out, on_loss="raise")
    assert json.loads(_metadata_attr(out)) == {
        "Description": "in memory",
        "Moving": "a.nii",
    }


def test_x5_reassigned_chain_keeps_the_metadata(
    warp_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(warp_x5)
    xform.transformations = [
        xforms.Affine(
            np.eye(4)[:3], input=systems.RASmm(), output=systems.RASmm()
        )
    ]
    out = tmp_path / "out.x5"
    xform.save(out, on_loss="raise")
    assert json.loads(_metadata_attr(out)) == JSON


def test_x5_to_generic_to_bids_and_back(warp_x5: Path) -> None:
    meta = io.load(warp_x5).metadata
    generic, report = _to(meta, Metadata)
    assert not report.lossy
    assert not hasattr(generic, "raw")  # the raw record stays
    sidecar = generic.to_bids()
    assert sidecar == JSON
    back, report = _to(Metadata.from_bids(sidecar), X5Metadata)
    assert not report.lossy
    assert back == meta


def test_x5_to_itk_loses_everything(warp_x5: Path) -> None:
    meta = io.load(warp_x5).metadata
    itk, report = _to(meta, ItkMetadata)
    assert set(report.lost) == {
        "description",
        "generated_by",
        "history",
        "moving",
        "fixed",
        "input_space",
        "output_space",
        "extra",
    }
    assert itk.description is UNSUPPORTED
    # An implicit conversion follows the policy in effect.
    with metadata_loss_policy("raise"), pytest.raises(MetadataLossError):
        TfmTransform(metadata=meta)
    with pytest.warns(MetadataLossWarning):
        tfm = TfmTransform(metadata=meta)
    assert type(tfm.metadata) is ItkMetadata


def test_x5_to_nifti_field(warp_x5: Path, tmp_path: Path) -> None:
    xform = io.load(warp_x5)
    with pytest.warns(MetadataLossWarning, match="input_space"):
        field = NiftiRASDisplacementField.from_other(
            xform.transformations[0], metadata=xform.metadata
        )
    assert type(field.metadata) is NiftiMetadata
    assert field.metadata.description == "sub-01 T1w to MNI"


@pytest.mark.parametrize("which", ["file", "block"])
def test_x5_to_nifti_field_carries_the_metadata(
    warp_x5: Path,
    which: str,  # noqa: ANN001
) -> None:
    xform = io.load(warp_x5)
    source = xform if which == "file" else xform.transformations[0]
    with pytest.warns(MetadataLossWarning) as caught:
        field = NiftiRASDisplacementField.from_other(source)
    assert len(caught) == 1
    assert set(caught[0].message.report.lost) == {
        "extra",
        "generated_by",
        "input_space",
        "output_space",
        "history",
        "moving",
        "fixed",
    }
    assert type(field.metadata) is NiftiMetadata
    assert field.metadata.description == "sub-01 T1w to MNI"


def test_a_single_node_block_holds_a_copy_of_the_metadata(
    warp_x5: Path, tmp_path: Path
) -> None:
    xform = io.load(warp_x5)
    block = xform.transformations[0]
    assert block.metadata == _to(xform.metadata, Metadata)[0]
    block.metadata.description = "edited block"
    assert xform.metadata.description == "sub-01 T1w to MNI"
    # A chain of several nodes has no metadata of its own: nor do they.
    path = _write_x5(
        tmp_path / "chain.x5",
        [("linear", {"Description": "a"}), ("linear", {"Description": "b"})],
        chains=((0, 1),),
    )
    chain = io.load(path)
    assert [b.metadata for b in chain.transformations] == [None, None]


def test_metadata_given_later_is_converted_by_every_parser() -> None:
    parser = X5TransformParser()
    parser.metadata = Metadata(description="later")
    assert type(parser.metadata) is X5Metadata
    flirt = FlirtMatrixParser(flirt_matrix=np.eye(4))
    flirt.metadata = Metadata(moving="m.nii")
    assert type(flirt.metadata) is FlirtMetadata
    with pytest.warns(MetadataLossWarning):
        flirt.metadata = Metadata(description="lost")
    assert flirt.metadata.description is UNSUPPORTED


# ----------------------------------------------------------------------
#   ITK
# ----------------------------------------------------------------------


def _write_itk_h5(path: Path, version: str = "5.4.0") -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("ITKVersion", data=version)
        f.create_dataset("HDFVersion", data="1.14.0")
        node = f.create_group("TransformGroup/0")
        node.create_dataset("TransformType", data="AffineTransform_double_3_3")
        node.create_dataset(
            "TransformParameters",
            data=np.r_[np.eye(3).ravel(), 1.0, 2.0, 3.0],
        )
        node.create_dataset("TransformFixedParameters", data=np.zeros(3))
    return path


def test_itk_h5_generated_by(tmp_path: Path) -> None:
    xform = io.load(_write_itk_h5(tmp_path / "affine.h5"))
    assert type(xform) is H5Transform
    meta = xform.metadata
    assert type(meta) is ItkH5Metadata
    assert meta.raw is xform.header
    assert meta.generated_by == (GeneratedBy(name="ITK", version="5.4.0"),)
    assert meta.description is UNSUPPORTED
    # Composition does not merge: the blocks have none of their own.
    assert xform.transformations[0].metadata is None
    generic, report = _to(meta, Metadata)
    assert generic.to_bids() == {
        "GeneratedBy": [{"Name": "ITK", "Version": "5.4.0"}]
    }
    assert not report.lossy


def test_itk_h5_encodes_only_the_itk_entry(tmp_path: Path) -> None:
    xform = io.load(_write_itk_h5(tmp_path / "affine.h5"))
    xform.metadata.generated_by = (
        GeneratedBy(name="ITK", version="5.3.0"),
        GeneratedBy(name="ANTs", version="2.5"),
    )
    report = xform.metadata.check_writable()
    assert set(report.lost) == {"generated_by"}
    header = xform.metadata.update_raw(on_loss="ignore")
    assert header.ITKVersion == "5.3.0"
    assert xform.header.ITKVersion == "5.4.0"  # the record is not edited


def test_itk_tfm_and_mat_are_opaque(tmp_path: Path) -> None:
    for cls in (TfmTransform, MatTransform):
        meta = cls().metadata
        assert type(meta) is ItkMetadata
        assert isinstance(meta, OpaqueMetadata)
        assert meta.description is UNSUPPORTED


def test_itk_mat_write_reports_what_was_set(tmp_path: Path) -> None:
    affine = xforms.Affine(
        np.eye(4)[:3], input=systems.LPSmm(), output=systems.LPSmm()
    )
    xform = MatTransform(transformations=[affine])
    xform.save(tmp_path / "a.mat", on_loss="raise")
    xform.metadata.description = "set after construction"
    with pytest.raises(MetadataLossError, match="description"):
        xform.save(tmp_path / "b.mat", on_loss="raise")
    xform.save(tmp_path / "b.mat", on_loss="ignore")


# ----------------------------------------------------------------------
#   FLIRT
# ----------------------------------------------------------------------


def test_flirt_moving_and_fixed_are_kept_in_memory_only(
    tmp_path: Path,
) -> None:
    image = nb.Nifti1Image(np.zeros(SHAPE, "float32"), np.eye(4))
    nb.save(image, str(tmp_path / "moving.nii.gz"))
    nb.save(image, str(tmp_path / "reference.nii.gz"))
    np.savetxt(tmp_path / "flirt.mat", np.eye(4))
    xform = FlirtTransform.from_file(
        tmp_path / "flirt.mat",
        moving=nb.load(str(tmp_path / "moving.nii.gz")),
        reference=nb.load(str(tmp_path / "reference.nii.gz")),
    )
    meta = xform.metadata
    assert type(meta) is FlirtMetadata
    # Not opaque: it keeps two fields, with no raw record.
    assert not isinstance(meta, OpaqueMetadata)
    assert meta.raw is None
    assert FlirtMetadata.supported_fields == {"moving", "fixed"}
    assert meta.moving == str(tmp_path / "moving.nii.gz")
    assert meta.fixed == str(tmp_path / "reference.nii.gz")
    assert meta.description is UNSUPPORTED
    # A `.mat` file stores nothing: both are lost on write.
    assert set(meta.check_writable().lost) == {"moving", "fixed"}
    # In-memory images have no path.
    image = nb.Nifti1Image(np.zeros(SHAPE, "float32"), np.eye(4))
    bare = FlirtTransform(flirt_matrix=np.eye(4), moving=image)
    assert bare.metadata.moving is None
    # A copy carries them, as any metadata.
    assert meta.copy().moving == meta.moving
    assert meta.to(Metadata).fixed == meta.fixed


# ----------------------------------------------------------------------
#   NARROWING
# ----------------------------------------------------------------------


def _formats():  # noqa: ANN202
    expected = (
        (X5TransformParser, X5Metadata),
        (H5Transform, ItkH5Metadata),
        (ItkTransform, ItkMetadata),
        (FlirtMatrixParser, FlirtMetadata),
        (FlirtTransform, FlirtMetadata),
    )
    out = []
    for cls in FileBasedObject._REGISTRY:
        for base, meta in expected:
            if issubclass(cls, base):
                out.append((cls, meta))
                break
    return sorted(out, key=lambda c: c[0].__name__)


def test_every_format_is_covered() -> None:
    names = {cls.__name__ for cls, _ in _formats()}
    assert {
        "X5Transform",
        "H5Transform",
        "TfmTransform",
        "MatTransform",
        "FlirtTransform",
    } <= names


@pytest.mark.parametrize(
    "cls,meta", _formats(), ids=lambda c: getattr(c, "__name__", "")
)
def test_every_xform_format_narrows_the_field(cls, meta) -> None:  # noqa: ANN001
    field = next(f for f in fields(cls) if f.name == "metadata")
    assert field.type is meta
    assert not field.positional
