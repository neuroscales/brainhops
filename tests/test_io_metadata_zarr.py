"""
Tests for the metadata of Zarr images: `ZarrMetadata` (a plain array,
the vocabulary as a sidecar under the attribute `"brainhops"`) and
`OmeZarrMetadata` (an OME-Zarr pyramid: the multiscale name, `omero` and
the other group attributes).
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("abczarr")

import abczarr  # noqa: E402
from bagof.magic import fields  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.axes import ChannelAxis, SpaceAxis  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.metadata import (  # noqa: E402
    UNSUPPORTED,
    Channel,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    convert,
    metadata_loss_policy,
)
from brainhops.io.images.nifti import NiftiImage, NiftiMetadata  # noqa: E402
from brainhops.io.images.zarr import (  # noqa: E402
    OmeZarrImage,
    OmeZarrMetadata,
    ZarrImage,
    ZarrMetadata,
)
from brainhops.io.images.zarr._multiscale import OmeZarrLevel  # noqa: E402

AXES = [SpaceAxis("x"), SpaceAxis("y"), SpaceAxis("z"), ChannelAxis("c")]


def _pyramid(metadata=None, levels=2):  # noqa: ANN001, ANN202
    data = np.arange(4 * 4 * 6 * 2, dtype="uint16").reshape((4, 4, 6, 2))
    images = [
        SingleScaleImage(data=data[:: 2**i, :: 2**i, :: 2**i])
        for i in range(levels)
    ]
    kwargs = {} if metadata is None else {"metadata": metadata}
    return OmeZarrImage(images=images, axes=AXES, **kwargs)


def _channels():  # noqa: ANN202
    return (
        Channel(name="DAPI", color="0000FFFF", display_range=(0.0, 900.0)),
        Channel(name="GFP", color="00FF00FF", display_range=(10.0, 500.0)),
    )


@pytest.fixture
def stained(tmp_path):  # noqa: ANN001, ANN201
    path = str(tmp_path / "stained.ome.zarr")
    meta = OmeZarrMetadata(
        name="brain", channels=_channels(), extra={"lab": "neuro"}
    )
    _pyramid(meta).save(path)
    return path


def _attrs(path):  # noqa: ANN001, ANN202
    node = abczarr.open(path, mode="r")
    return {key: node.attrs[key] for key in node.attrs}


# ----------------------------------------------------------------------
#   PLAIN ZARR
# ----------------------------------------------------------------------


def test_plain_zarr_stores_the_vocabulary_as_a_sidecar(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "a.zarr")
    meta = Metadata(
        description="a volume",
        repetition_time=2.0,
        channels=(Channel(name="a"),),
        extra={"owner": "me"},
    )
    image = ZarrImage(data=np.zeros((3, 4), "f4"), metadata=meta)
    assert type(image.metadata) is ZarrMetadata
    image.save(path)
    attrs = _attrs(path)
    assert attrs["owner"] == "me"
    assert attrs["brainhops"] == {
        "Description": "a volume",
        "RepetitionTime": 2.0,
        "Channels": [{"Name": "a"}],
    }
    back = io.load(path)
    assert type(back) is ZarrImage
    assert back.metadata == image.metadata
    assert back.metadata.raw == attrs
    assert back.metadata.changed_fields() == {}


def test_plain_zarr_round_trip_and_edits(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "a.zarr")
    ZarrImage(
        data=np.zeros((3, 4), "f4"),
        metadata=ZarrMetadata(description="x", extra={"a": 1, "b": 2}),
    ).save(path)
    image = io.load(path)
    image.metadata.description = None
    image.metadata.echo_time = 0.03
    image.metadata.extra = {"a": 1, "c": 3}
    out = str(tmp_path / "b.zarr")
    image.save(out)
    assert _attrs(out) == {"a": 1, "c": 3, "brainhops": {"EchoTime": 0.03}}


def test_plain_zarr_cannot_hold_diffusion() -> None:
    assert ZarrMetadata.unsupported_fields == {
        "diffusion_bvalues",
        "diffusion_bvectors",
    }
    _, report = convert(
        Metadata(diffusion_bvalues=(0.0, 1000.0)),
        ZarrMetadata,
        on_loss="ignore",
    )
    assert report.lost == {"diffusion_bvalues": (0.0, 1000.0)}


def test_a_reserved_extra_key_is_reported(tmp_path) -> None:  # noqa: ANN001
    image = ZarrImage(data=np.zeros((2, 2), "f4"))
    image.metadata.extra = {"brainhops": "mine"}
    with pytest.raises(MetadataLossError) as caught:
        image.save(str(tmp_path / "a.zarr"), on_loss="raise")
    assert caught.value.report.lost == {"extra['brainhops']": "mine"}


# ----------------------------------------------------------------------
#   OME-ZARR: READING
# ----------------------------------------------------------------------


def test_the_pyramid_metadata_is_decoded(stained) -> None:  # noqa: ANN001
    image = io.load(stained)
    assert type(image) is OmeZarrImage
    meta = image.metadata
    assert type(meta) is OmeZarrMetadata
    assert meta.name == "brain"
    assert meta.channels == _channels()
    assert meta.display_range is None  # the channels disagree
    assert meta.extra == {"lab": "neuro"}
    assert meta.description is UNSUPPORTED
    assert meta.data_unit is UNSUPPORTED
    assert meta.changed_fields() == {}
    assert meta.multiscale.name == "brain"
    assert meta.omero["channels"][0]["label"] == "DAPI"


def test_omero_is_written(stained) -> None:  # noqa: ANN001
    attrs = _attrs(stained)
    omero = attrs["ome"]["omero"]
    assert [c["label"] for c in omero["channels"]] == ["DAPI", "GFP"]
    assert [c["color"] for c in omero["channels"]] == ["0000FF", "00FF00"]
    assert omero["channels"][1]["window"]["start"] == 10.0
    assert omero["channels"][1]["window"]["end"] == 500.0
    assert attrs["ome"]["multiscales"][0]["name"] == "brain"
    assert attrs["lab"] == "neuro"


def test_each_level_holds_a_derived_copy(stained) -> None:  # noqa: ANN001
    image = io.load(stained)
    levels = image.images
    assert all(type(level) is OmeZarrLevel for level in levels)
    for level in levels:
        assert type(level.metadata) is OmeZarrMetadata
        assert level.metadata is not image.metadata
        assert level.metadata == image.metadata
    levels[1].metadata.name = "level"
    assert image.metadata.name == "brain"


def test_a_shared_window_is_the_display_range(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "p.zarr")
    _pyramid(OmeZarrMetadata(display_range=(0.0, 100.0))).save(path)
    meta = io.load(path).metadata
    assert meta.display_range == (0.0, 100.0)
    # One channel per entry of the channel axis, white.
    assert [c.color for c in meta.channels] == ["FFFFFFFF", "FFFFFFFF"]


def test_a_window_nothing_gives_is_the_dtype_range(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "p.zarr")
    _pyramid(OmeZarrMetadata(channels=(Channel(), Channel()))).save(path)
    window = _attrs(path)["ome"]["omero"]["channels"][0]["window"]
    assert (window["start"], window["end"]) == (0.0, 65535.0)


# ----------------------------------------------------------------------
#   OME-ZARR: ROUND TRIP
# ----------------------------------------------------------------------


def test_a_read_then_save_keeps_everything(stained, tmp_path) -> None:  # noqa: ANN001
    out = str(tmp_path / "out.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetadataLossWarning)
        io.load(stained).save(out)
    before, after = _attrs(stained), _attrs(out)
    assert after["ome"]["omero"] == before["ome"]["omero"]
    assert after["ome"]["multiscales"][0]["name"] == "brain"
    assert after["lab"] == "neuro"
    assert io.load(out).metadata == io.load(stained).metadata


def test_record_content_outside_the_vocabulary_survives(
    stained,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    # Free-form omero keys (nested dicts too), a channel's other keys, and
    # the multiscale's type and downsampling metadata.
    node = abczarr.open(stained, mode="r+")
    block = dict(node.attrs["ome"])
    block["omero"]["rdefs"] = {"model": "color", "defaultZ": 2}
    block["omero"]["channels"][0]["active"] = False
    block["multiscales"][0]["type"] = "gaussian"
    block["multiscales"][0]["metadata"] = {"method": "skimage", "kwargs": {}}
    node.attrs["ome"] = block
    image = io.load(stained)
    image.metadata.name = "renamed"
    image.metadata.channels = (
        Channel(name="nuclei", color="0000FFFF", display_range=(0.0, 900.0)),
        image.metadata.channels[1],
    )
    out = str(tmp_path / "out.zarr")
    image.save(out)
    after = _attrs(out)["ome"]
    assert after["omero"]["rdefs"] == {"model": "color", "defaultZ": 2}
    assert after["omero"]["channels"][0]["active"] is False
    assert after["omero"]["channels"][0]["label"] == "nuclei"
    assert after["multiscales"][0]["name"] == "renamed"
    assert after["multiscales"][0]["type"] == "gaussian"
    assert after["multiscales"][0]["metadata"]["method"] == "skimage"


def test_clearing_fields(stained, tmp_path) -> None:  # noqa: ANN001
    image = io.load(stained)
    image.metadata.name = None
    image.metadata.channels = None
    image.metadata.extra = {}
    out = str(tmp_path / "out.zarr")
    image.save(out)
    attrs = _attrs(out)
    assert "name" not in attrs["ome"]["multiscales"][0]
    assert "omero" not in attrs["ome"]
    assert "lab" not in attrs


def test_alpha_and_units_are_approximated(tmp_path) -> None:  # noqa: ANN001
    meta = OmeZarrMetadata(
        channels=(
            Channel(name="a", color="FF000080", unit="photons"),
            Channel(name="b"),
        )
    )
    report = meta.check_writable(image=_pyramid())
    assert set(report.approximated) == {"channels"}
    with pytest.warns(MetadataLossWarning):
        _pyramid(meta).save(str(tmp_path / "p.zarr"))


# ----------------------------------------------------------------------
#   CROSS-FORMAT
# ----------------------------------------------------------------------


def test_ome_zarr_channels_to_generic(stained) -> None:  # noqa: ANN001
    generic, report = convert(io.load(stained).metadata, Metadata)
    assert not report.lossy
    assert [c.name for c in generic.channels] == ["DAPI", "GFP"]
    assert generic.name == "brain"
    assert generic.raw is None


def test_ome_zarr_to_nifti_reports_the_loss(stained) -> None:  # noqa: ANN001
    _, report = convert(
        io.load(stained).metadata, NiftiMetadata, on_loss="ignore"
    )
    assert set(report.lost) == {"name", "channels", "extra"}


def test_nifti_to_ome_zarr(tmp_path) -> None:  # noqa: ANN001
    nifti = NiftiImage(data=np.zeros((4, 4, 6), "float32"))
    nifti.metadata.description = "a scan"
    nifti.metadata.display_range = (0.0, 50.0)
    meta, report = convert(nifti.metadata, OmeZarrMetadata, on_loss="ignore")
    assert report.lost == {"description": "a scan"}
    assert meta.display_range == (0.0, 50.0)
    path = str(tmp_path / "p.zarr")
    pyramid = OmeZarrImage(
        images=[SingleScaleImage(data=np.zeros((4, 4, 6), "float32"))],
        axes=AXES[:3],
        metadata=meta,
    )
    pyramid.save(path)
    back = io.load(path).metadata
    assert back.display_range == (0.0, 50.0)
    # And back to NIfTI: the display range is kept, the channel is lost.
    nifti, report = convert(back, NiftiMetadata, on_loss="ignore")
    assert nifti.display_range == (0.0, 50.0)
    assert set(report.lost) == {"channels"}


def test_a_generic_pyramid_converts_on_save(tmp_path) -> None:  # noqa: ANN001
    pyramid = _pyramid()
    with metadata_loss_policy("ignore"):
        pyramid.metadata = Metadata(name="from memory", description="lost")
    assert type(pyramid.metadata) is OmeZarrMetadata
    path = str(tmp_path / "p.zarr")
    pyramid.save(path)
    assert io.load(path).metadata.name == "from memory"


# ----------------------------------------------------------------------
#   THE FIELD
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, expected",
    [
        (ZarrImage, ZarrMetadata),
        (OmeZarrImage, OmeZarrMetadata),
        (OmeZarrLevel, OmeZarrMetadata),
    ],
    ids=lambda c: getattr(c, "__name__", ""),
)
def test_every_zarr_format_narrows_the_field(cls, expected) -> None:  # noqa: ANN001
    field = next(f for f in fields(cls) if f.name == "metadata")
    assert field.type is expected
    assert not field.positional
    assert not field.repr
