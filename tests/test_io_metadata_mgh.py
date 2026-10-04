"""
Tests for the metadata of MGH/MGZ files (`MghMetadata`).

The footer of MRI parameters is decoded into the vocabulary in BIDS
units (ms -> s, rad -> deg), the command-line tags into `history`; a
read-then-save keeps the footer and the tags, a field set by the user is
written over them, and a conversion to or from another format reports
what the target cannot hold.
"""

import copy
import gzip
import struct
import warnings

import numpy as np
import pytest


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, report=report, **kwargs), report


nb = pytest.importorskip("nibabel")

from bagof.magic import fields  # noqa: E402
from nibabel.freesurfer.mghformat import MGHImage as NibabelMgh  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.metadata import (  # noqa: E402
    UNSUPPORTED,
    ConversionReport,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from brainhops.io.base._mgh_metadata import (  # noqa: E402
    MghRaw,
    decode_history,
    encode_history,
    parse_tags,
)
from brainhops.io.base.mgh import MghParser  # noqa: E402
from brainhops.io.images.freesurfer import MghImage, MghMetadata  # noqa: E402
from brainhops.io.images.nifti import NiftiImage, NiftiMetadata  # noqa: E402


def _cmdline(command: bytes) -> bytes:
    payload = command + b"\0"
    return struct.pack(">iq", 3, len(payload)) + payload


# A tag FreeSurfer writes with a 64-bit length, then two command lines,
# then a legacy tag with no length (which must stay last).
OTHER_TAG = struct.pack(">iq", 43, 4) + struct.pack(">f", 3.0)
TAGS = (
    OTHER_TAG
    + _cmdline(b"mri_convert in.nii orig.mgz")
    + _cmdline(b"mri_normalize orig.mgz T1.mgz")
    + struct.pack(">i", 2)
)


def _write(path, tags=TAGS, shape=(4, 5, 6), **params):  # noqa: ANN001, ANN003, ANN202
    image = NibabelMgh(np.zeros(shape, "float32"), np.eye(4))
    footer = dict(tr=2300.0, te=2.98, ti=900.0, flip_angle=np.deg2rad(9.0))
    footer.update(params)
    for name, value in footer.items():
        image.header[name] = value
    nb.save(image, str(path))
    raw = gzip.decompress(path.read_bytes())
    path.write_bytes(gzip.compress(raw + tags))
    return path


@pytest.fixture
def scan(tmp_path):  # noqa: ANN001, ANN201
    return _write(tmp_path / "T1.mgz")


def _footer(path):  # noqa: ANN001, ANN202
    return io.load(path).mri_params


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_footer_and_tags_are_decoded(scan) -> None:  # noqa: ANN001
    image = io.load(scan)
    meta = image.metadata
    assert type(meta) is MghMetadata
    assert meta.repetition_time == 2.3
    assert meta.echo_time == 0.00298
    assert meta.inversion_time == 0.9
    assert meta.flip_angle == 9.0
    assert meta.history == (
        "mri_convert in.nii orig.mgz",
        "mri_normalize orig.mgz T1.mgz",
    )
    assert meta.description is UNSUPPORTED
    assert meta.extra is UNSUPPORTED
    assert meta.changed_fields() == {}


def test_the_record_is_the_header_and_the_tags(scan) -> None:  # noqa: ANN001
    image = io.load(scan)
    assert image.metadata.raw.header is image.header
    assert image.metadata.header is image.header
    assert image.metadata.tags == image.tags == TAGS


def test_zero_reads_as_unknown(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "x.mgz", tags=b"", tr=0.0, ti=0.0)
    meta = io.load(path).metadata
    assert meta.repetition_time is None
    assert meta.inversion_time is None
    assert meta.history is None
    assert meta.echo_time == 0.00298


def test_unparseable_tags_have_no_history(tmp_path) -> None:  # noqa: ANN001
    tags = struct.pack(">iq", 3, 1000) + b"short"
    image = io.load(_write(tmp_path / "x.mgz", tags=tags))
    assert image.metadata.history is None
    assert image.tags == tags


def test_an_image_built_in_memory_has_empty_metadata() -> None:
    image = MghImage(data=np.zeros((2, 3, 4), "float32"))
    assert type(image.metadata) is MghMetadata
    assert image.metadata.raw is None


def test_explicit_metadata_wins_over_the_header(scan) -> None:  # noqa: ANN001
    nib = nb.load(str(scan))
    image = MghImage(image=nib, metadata=MghMetadata(echo_time=0.005))
    assert image.metadata.raw.header is image.header
    assert image.metadata.echo_time == 0.005
    assert image.metadata.repetition_time == 2.3
    assert image.metadata.changed_fields() == {"echo_time": 0.005}


def test_new_tags_are_read_again(scan) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.tags = _cmdline(b"recon-all")
    assert image.metadata.history == ("recon-all",)


# ----------------------------------------------------------------------
#   TAGS
# ----------------------------------------------------------------------


def test_tags_parse_into_chunks() -> None:
    parsed = parse_tags(TAGS)
    assert [tag for tag, _ in parsed] == [43, 3, 3, 2]
    assert b"".join(chunk for _, chunk in parsed) == TAGS
    assert parse_tags(b"") == []
    assert parse_tags(b"\0\0") is None


def test_history_replaces_only_the_command_lines() -> None:
    tags = encode_history(TAGS, ("a", "b", "c"))
    assert decode_history(tags) == ("a", "b", "c")
    parsed = parse_tags(tags)
    assert [tag for tag, _ in parsed] == [43, 3, 3, 3, 2]
    assert parsed[0][1] == OTHER_TAG
    # No command line yet: they go before the legacy tag.
    tags = encode_history(OTHER_TAG + struct.pack(">i", 2), ("a",))
    assert [tag for tag, _ in parse_tags(tags)] == [43, 3, 2]
    assert encode_history(TAGS, None) == OTHER_TAG + struct.pack(">i", 2)


# ----------------------------------------------------------------------
#   ROUND TRIP
# ----------------------------------------------------------------------


def test_a_read_then_save_keeps_the_footer_and_tags(scan, tmp_path) -> None:  # noqa: ANN001
    out = tmp_path / "out.mgz"
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetadataLossWarning)
        io.load(scan).save(out)
    back = io.load(out)
    assert back.tags == TAGS
    assert back.mri_params == pytest.approx(io.load(scan).mri_params)
    assert back.metadata == io.load(scan).metadata


def test_a_common_field_set_is_written(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.echo_time = 0.005
    image.metadata.flip_angle = 30.0
    image.metadata.history += ("recon-all -s bert",)
    image.save(tmp_path / "out.mgz")
    back = io.load(tmp_path / "out.mgz")
    assert back.mri_params["te"] == pytest.approx(5.0)
    assert back.mri_params["flip_angle"] == pytest.approx(np.pi / 6)
    assert back.metadata.flip_angle == 30.0
    assert back.metadata.history[-1] == "recon-all -s bert"
    assert parse_tags(back.tags)[0][1] == OTHER_TAG


def test_a_record_edit_survives(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.raw.header["fov"] = 256.0
    image.metadata.raw.header["tr"] = 1000.0
    image.save(tmp_path / "out.mgz")
    params = _footer(tmp_path / "out.mgz")
    assert params["fov"] == 256.0
    assert params["tr"] == 1000.0


def test_none_clears_the_slot(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.inversion_time = None
    image.metadata.history = None
    image.save(tmp_path / "out.mgz")
    back = io.load(tmp_path / "out.mgz")
    assert back.mri_params["ti"] == 0.0
    assert back.metadata.history is None
    assert back.tags == OTHER_TAG + struct.pack(">i", 2)


def test_history_cannot_be_written_over_unparseable_tags(tmp_path) -> None:  # noqa: ANN001
    tags = struct.pack(">iq", 3, 1000) + b"short"
    image = io.load(_write(tmp_path / "x.mgz", tags=tags))
    image.metadata.history = ("recon-all",)
    with pytest.raises(MetadataLossError) as caught:
        image.save(tmp_path / "out.mgz", on_loss="raise")
    assert caught.value.report.lost == {"history": ("recon-all",)}
    image.save(tmp_path / "out.mgz", on_loss="ignore")
    assert io.load(tmp_path / "out.mgz").tags == tags


# ----------------------------------------------------------------------
#   UNIT CONVERSION AND THE WRITER KEYWORDS
# ----------------------------------------------------------------------


def test_units_are_converted_both_ways(tmp_path) -> None:  # noqa: ANN001
    image = MghImage(
        data=np.zeros((2, 3, 4), "float32"),
        metadata=MghMetadata(
            repetition_time=2.0,
            echo_time=0.03,
            inversion_time=1.1,
            flip_angle=90.0,
        ),
    )
    image.save(tmp_path / "out.mgh")
    params = _footer(tmp_path / "out.mgh")
    assert params["tr"] == pytest.approx(2000.0)
    assert params["te"] == pytest.approx(30.0)
    assert params["ti"] == pytest.approx(1100.0)
    assert params["flip_angle"] == pytest.approx(np.pi / 2)
    meta = io.load(tmp_path / "out.mgh").metadata
    assert (meta.repetition_time, meta.echo_time) == (2.0, 0.03)
    assert (meta.inversion_time, meta.flip_angle) == (1.1, 90.0)


def test_footer_keywords_go_through_the_metadata(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    image.metadata.echo_time = 0.005
    # The keyword wins, even over a field set by the user, and even when
    # it equals the value that was read.
    image.save(tmp_path / "out.mgz", te=7.0, tr=2300.0, fov=200.0)
    params = _footer(tmp_path / "out.mgz")
    assert params["te"] == pytest.approx(7.0)
    assert params["tr"] == pytest.approx(2300.0)
    assert params["fov"] == pytest.approx(200.0)
    # ... and a zero clears the slot.
    image.save(tmp_path / "zero.mgz", ti=0.0)
    assert _footer(tmp_path / "zero.mgz")["ti"] == 0.0
    # The image's own metadata is untouched.
    assert image.metadata.echo_time == 0.005


def test_like_sits_under_a_changed_field(scan, tmp_path) -> None:  # noqa: ANN001
    template = _write(tmp_path / "template.mgz", tr=777.0, te=1.0)
    image = io.load(scan)
    image.metadata.echo_time = 0.004
    image.save(tmp_path / "out.mgz", like=template)
    params = _footer(tmp_path / "out.mgz")
    assert params["tr"] == pytest.approx(777.0)
    assert params["te"] == pytest.approx(4.0)


# ----------------------------------------------------------------------
#   CROSS-FORMAT
# ----------------------------------------------------------------------


def test_mgh_to_nifti_loses_the_acquisition_parameters(scan) -> None:  # noqa: ANN001
    meta = io.load(scan).metadata
    nifti, report = _to(meta, NiftiMetadata, on_loss="ignore")
    assert set(report.lost) == {
        "history",
        "echo_time",
        "inversion_time",
        "flip_angle",
    }
    # NIfTI derives the TR from the time axis: it is copied, and checked
    # when written.
    assert nifti.repetition_time == 2.3


def test_saving_mgh_as_nifti_reports_the_loss(scan, tmp_path) -> None:  # noqa: ANN001
    image = io.load(scan)
    with pytest.warns(MetadataLossWarning) as caught:
        io.save(image, tmp_path / "out.nii.gz")
    # One save, one warning: the conversion's report and the write's.
    assert len(caught) == 1
    # It points at the caller of `io.save`.
    assert caught[0].filename == __file__
    report = caught[0].message.report
    assert (report.source, report.target) == ("mgh", "nifti")
    assert set(report.lost) == {
        "echo_time",
        "flip_angle",
        "history",
        "inversion_time",
        "repetition_time",  # a 3-D NIfTI image has no time axis
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.save(image, tmp_path / "quiet.nii.gz", on_loss="ignore")
    with pytest.raises(MetadataLossError):
        io.save(image, tmp_path / "loud.nii.gz", on_loss="raise")


def test_nifti_to_mgh(tmp_path) -> None:  # noqa: ANN001
    nii = nb.Nifti1Image(np.zeros((4, 5, 6, 5), "float32"), np.eye(4))
    nii.header["descrip"] = b"bold"
    nii.header.set_xyzt_units("mm", "sec")
    nii.header["pixdim"][4] = 2.0
    nb.save(nii, str(tmp_path / "bold.nii.gz"))
    meta = io.load(tmp_path / "bold.nii.gz").metadata
    mgh, report = _to(meta, MghMetadata, on_loss="ignore")
    assert report.lost == {"description": "bold", "space": "aligned"}
    assert mgh.repetition_time == 2.0
    with metadata_loss_policy("ignore"):
        io.save(io.load(tmp_path / "bold.nii.gz"), tmp_path / "bold.mgz")
    assert _footer(tmp_path / "bold.mgz")["tr"] == pytest.approx(2000.0)


def test_mgh_through_generic_to_bids(scan) -> None:  # noqa: ANN001
    generic, report = _to(io.load(scan).metadata, Metadata)
    assert not report.lossy
    sidecar = generic.to_bids()
    assert sidecar["RepetitionTime"] == 2.3
    assert sidecar["EchoTime"] == 0.00298
    assert sidecar["InversionTime"] == 0.9
    assert sidecar["FlipAngle"] == 9.0
    back, report = _to(Metadata.from_bids(sidecar), MghMetadata)
    assert not report.lossy
    assert back == io.load(scan).metadata


def test_record_copies_are_independent() -> None:
    record = MghRaw(tags=b"x")
    record.header["tr"] = 5.0
    other = copy.deepcopy(record)
    other.header["tr"] = 6.0
    assert float(record.header["tr"]) == 5.0
    assert other != record
    other.header["tr"] = 5.0
    assert other == record


# ----------------------------------------------------------------------
#   THE FIELD
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [MghParser, MghImage], ids=lambda c: c.__name__
)
def test_every_mgh_format_narrows_the_field(cls) -> None:  # noqa: ANN001
    field = next(f for f in fields(cls) if f.name == "metadata")
    assert field.type is MghMetadata
    assert not field.positional
    assert not field.repr


def test_a_nifti_image_becomes_an_mgh_image(tmp_path) -> None:  # noqa: ANN001
    nifti = NiftiImage(data=np.zeros((2, 3, 4), "float32"))
    nifti.metadata.description = "lost"
    with metadata_loss_policy("ignore"):
        mgh = MghImage.from_instance(nifti)
    assert type(mgh.metadata) is MghMetadata


# ----------------------------------------------------------------------
#   LAZY TAGS
# ----------------------------------------------------------------------


def test_the_tags_are_read_lazily(scan, monkeypatch) -> None:  # noqa: ANN001
    from brainhops.io.base import mgh

    reads = []
    read = mgh._read_tags_file

    def counting(*args):  # noqa: ANN002, ANN202
        reads.append(1)
        return read(*args)

    monkeypatch.setattr(mgh, "_read_tags_file", counting)
    image = io.load(scan)
    meta = image.metadata
    # The footer is in the header: no need for the tags.
    assert meta.repetition_time == 2.3
    assert not meta.raw.tags_loaded
    assert reads == []
    assert meta.history == (
        "mri_convert in.nii orig.mgz",
        "mri_normalize orig.mgz T1.mgz",
    )
    assert meta.changed_fields() == {}
    # Read once, for the metadata and the image alike.
    assert image.tags == meta.raw.tags
    assert reads == [1]


def test_lazy_tags_survive_an_untouched_save(scan, tmp_path) -> None:  # noqa: ANN001
    io.load(scan).save(tmp_path / "out.mgz")
    assert io.load(tmp_path / "out.mgz").tags == io.load(scan).tags


def test_a_new_header_is_read_again_through_replace(scan) -> None:  # noqa: ANN001
    from bagof.magic import replace

    image = io.load(scan)
    image.metadata.echo_time = 0.004
    header = image.header.copy()
    header["tr"] = 1000.0
    other = replace(image, header=header)
    assert other.metadata.raw.header is other.header
    assert other.metadata.repetition_time == 1.0
    assert other.metadata.echo_time == 0.004
    assert image.metadata.repetition_time == 2.3


def test_the_data_type_round_trips(tmp_path) -> None:  # noqa: ANN001
    data = np.arange(60).reshape((3, 4, 5)).astype("int16")
    nb.save(NibabelMgh(data, np.eye(4)), str(tmp_path / "a.mgz"))
    image = io.load(tmp_path / "a.mgz")
    assert image.metadata.data_type == np.int16
    image.data = np.asarray(image.data, "int64")
    image.save(tmp_path / "b.mgz")
    stored = nb.load(str(tmp_path / "b.mgz")).get_data_dtype()
    assert stored.newbyteorder("=") == np.int16  # MGH is big-endian
    # A type MGH cannot store is approximated by the nearest one.
    image.metadata.data_type = "uint16"
    report = image.metadata.check_writable(image=image)
    assert "data_type" in report.approximated
