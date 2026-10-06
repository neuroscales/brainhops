"""
The cross-format matrix test (`docs/design/format-metadata.md`, section
12 (c)): a fully populated, *encodable* `Metadata` is converted into each
format and back, and the conversion loses exactly the fields the format
declares unsupported, no more, no less, so a wrong `supports=` list
fails here. Where a fresh record can hold the values, they are written
into it and read back too.
"""

import datetime

import pytest


def _to(source, target, **kwargs):  # noqa: ANN001, ANN003, ANN202
    """`source.to(target, ...)`, and the report it filled."""
    report = ConversionReport()
    return source.to(target, on_loss=report, **kwargs), report


nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402, F401
from brainhops.datamodel.metadata import (  # noqa: E402
    UNSUPPORTED,
    Channel,
    ConversionReport,
    GeneratedBy,
    Metadata,
)
from brainhops.datamodel.metadata._vocabulary import VOCABULARY  # noqa: E402
from brainhops.io.images.freesurfer.mgh import MghMetadata  # noqa: E402
from brainhops.io.images.nifti import NiftiMetadata  # noqa: E402
from brainhops.io.metadata import (  # noqa: E402  # noqa: E402
    FileBasedMetadata,
    OpaqueMetadata,
)
from brainhops.io.transformations.fsl.flirt import FlirtMetadata  # noqa: E402
from brainhops.io.transformations.itk import (  # noqa: E402
    ItkH5Metadata,
    ItkMetadata,
)
from brainhops.io.transformations.x5 import X5Metadata  # noqa: E402

zarr = pytest.importorskip("brainhops.io.images.zarr")
OmeZarrImage = zarr.OmeZarrImage
OmeZarrMetadata = zarr.OmeZarrMetadata
ZarrMetadata = zarr.ZarrMetadata

# Three volumes (or channels), four slices along `k`.
FULL = dict(
    name="sub-01",
    description="a short description",
    history=("mri_convert a b", "recon-all -s bert"),
    generated_by=(GeneratedBy(name="ITK", version="5.4.0"),),
    creation_time=datetime.datetime(2024, 1, 2, 3, 4, 5),
    sources=("sub-01_T1w.nii",),
    space="MNI152NLin2009cAsym",
    intent="label",
    repetition_time=2.0,
    echo_time=0.03,
    inversion_time=0.9,
    flip_angle=9.0,
    magnetic_field_strength=3.0,
    manufacturer="Siemens",
    manufacturers_model_name="Prisma",
    institution_name="Somewhere",
    acquisition_time=datetime.datetime(2024, 1, 2, 3, 0, 0),
    phase_encoding_direction="j",
    total_readout_time=0.05,
    effective_echo_spacing=0.0005,
    slice_encoding_direction="k",
    slice_timing=(0.0, 0.5, 1.0, 1.5),
    multiband_acceleration_factor=2,
    bvalues=(0.0, 1000.0, 1000.0),
    bvectors=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    display_range=(0.0, 100.0),
    channels=tuple(
        Channel(name=name, color="FFFFFFFF", display_range=(0.0, 100.0))
        for name in ("a", "b", "c")
    ),
    data_unit="a.u.",
    data_type="float32",
    scale_slope=0.5,
    scale_intercept=10.0,
    objective_magnification=10.0,
    objective_numerical_aperture=0.3,
    illumination_type="epifluorescence",
    contrast_method="fluorescence",
    moving="sub-01_T1w.nii.gz",
    fixed="tpl-MNI_T1w.nii.gz",
    input_space="T1w",
    output_space="MNI152NLin2009cAsym",
    extra={"TaskName": "rest"},
)
# The values as the hub holds them (`"a.u."` is a `Unit`), to compare
# with what a conversion gives back.
HUB = Metadata(**FULL)


def _nifti_record() -> nb.Nifti1Header:
    header = nb.Nifti1Header()
    header.set_data_shape((5, 5, 4, 3))
    header.set_xyzt_units("mm", "sec")
    return header


# Format -> a fresh record its writer would hold, or `None` when the
# format has no record to hold the values (or is checked otherwise).
FORMATS = {
    NiftiMetadata: _nifti_record,
    MghMetadata: MghMetadata._raw_class,
    X5Metadata: X5Metadata._raw_class,
    ZarrMetadata: ZarrMetadata._raw_class,
    ItkH5Metadata: ItkH5Metadata._raw_class,
    OmeZarrMetadata: None,  # written by a real save below
    ItkMetadata: None,
    FlirtMetadata: None,
}


def _subclasses(cls: type) -> set:
    out = set()
    for sub in cls.__subclasses__():
        out |= {sub} | _subclasses(sub)
    return out


def test_every_format_is_in_the_matrix() -> None:
    formats = {
        cls
        for cls in _subclasses(Metadata)
        if cls not in (FileBasedMetadata, OpaqueMetadata)
        and cls.__module__.startswith("brainhops.io")
        # A shared base, not a format (`_ZarrMetadataParser`).
        and not cls.__name__.startswith("_")
        # A parametrized base (`FileBasedMetadata[nb.Nifti1Header]`).
        and "[" not in cls.__name__
    }
    assert formats == set(FORMATS)


def test_the_fixture_is_fully_populated() -> None:
    full = Metadata(**FULL)
    assert all(getattr(full, name) is not None for name in VOCABULARY)


@pytest.mark.parametrize("cls", list(FORMATS), ids=lambda c: c.__name__)
def test_a_conversion_loses_exactly_the_unsupported_fields(cls) -> None:  # noqa: ANN001
    converted, report = _to(Metadata(**FULL), cls)
    assert set(report.lost) == set(cls.unsupported_fields)
    assert not report.approximated
    for name in cls.unsupported_fields:
        assert getattr(converted, name) is UNSUPPORTED
    # And back: nothing more is lost on the way to the hub.
    back, report = _to(converted, Metadata)
    assert not report.lossy
    for name in VOCABULARY + ("extra",):
        expected = getattr(HUB, name)
        if name in cls.unsupported_fields:
            expected = {} if name == "extra" else None
        assert getattr(back, name) == expected, name


# The fields each format derives from the geometry of the data model.
_GEOMETRY = {
    # The image writer stores the scaling with the data, not the record.
    NiftiMetadata: {
        "repetition_time",
        "intent",
        "space",
        "scale_slope",
        "scale_intercept",
    },
    ZarrMetadata: {"data_type"},
}


@pytest.mark.parametrize(
    "cls",
    [cls for cls, record in FORMATS.items() if record is not None],
    ids=lambda c: c.__name__,
)
def test_a_fresh_record_holds_what_the_format_supports(cls) -> None:  # noqa: ANN001
    converted, _ = _to(Metadata(**FULL), cls)
    report = ConversionReport()
    record = converted.update_raw(FORMATS[cls](), on_loss=report)
    back = cls.from_raw(record)
    if cls is ItkH5Metadata:
        # Only the ITK version is recorded: the fixture names ITK alone.
        assert back.generated_by == FULL["generated_by"]
        return
    # Fields that are views of geometry (the writer sets them), which a bare
    # record does not hold.
    expected = set(VOCABULARY) & cls.supported_fields
    expected -= _GEOMETRY.get(cls, set())
    assert not report.lost
    for name in sorted(expected):
        assert getattr(back, name) == getattr(HUB, name), name
    if cls.supports("extra"):
        assert back.extra == FULL["extra"]


def _format_classes() -> list:
    """Every metadata class of a format in the package."""
    found, stack = [], [Metadata]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            stack.append(sub)
            if (
                sub.__module__.startswith("brainhops.")
                and issubclass(sub, FileBasedMetadata)
                and not sub.__name__.startswith("_")
                # Not the class `bagof` builds for a subscript such as
                # `FileBasedMetadata[MghRaw]`: its subclass is the format.
                and "__magic_generic_origin__" not in vars(sub)
            ):
                found.append(sub)
    return found


def test_every_format_declares_a_record_type_of_its_own() -> None:
    # A conversion gives a record back to a format only when the record is
    # of the type the format declares (`_accepts_raw`): no two formats
    # may declare the same type, or one that is a subclass of another's.
    declared = {}
    for cls in _format_classes():
        if cls is FileBasedMetadata:
            continue
        raw_type = cls._raw_class
        assert raw_type is not None, f"{cls.__name__} declares no raw type"
        if raw_type is not type(None):
            declared[cls] = raw_type
    assert {cls.__name__ for cls in declared} >= {
        "NiftiMetadata",
        "MghMetadata",
        "ZarrMetadata",
        "OmeZarrMetadata",
        "X5Metadata",
        "ItkH5Metadata",
    }
    for cls, raw_type in declared.items():
        for other, other_type in declared.items():
            if other is not cls:
                assert not issubclass(raw_type, other_type), (cls, other)
