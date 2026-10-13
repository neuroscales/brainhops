"""
The cross-format matrix of the vocabulary.

A fully populated `Metadata` is converted into each metadata format that
stores a vocabulary, and consolidated into a fresh record of the format.
The conversion reports a lower bound of what is lost, from the
`SUPPORTED_FIELDS` of the format, and the consolidation reports exactly
what the record could not hold. Both are the complement of
`SUPPORTED_FIELDS`, no more and no less, so a constant that disagrees
with the codec of its format fails here. The record then holds every
value that the format supports, except the slots that belong to the
geometry, and the way back to `Metadata` loses nothing.
"""

import datetime

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")
h5py = pytest.importorskip("h5py")

import brainhops.io  # noqa: E402, F401
from brainhops.datamodel.metadata import (  # noqa: E402
    Channel,
    ConversionReport,
    GeneratedBy,
    Metadata,
)
from brainhops.datamodel.metadata._vocabulary import (  # noqa: E402
    FIELDS,
    VOCABULARY,
)
from brainhops.io.common.mgh import MghMetadata, MghRaw  # noqa: E402
from brainhops.io.common.nifti import NiftiMetadata, NiftiRaw  # noqa: E402
from brainhops.io.metadata import BidsSidecar, MetadataFormat  # noqa: E402
from brainhops.io.transformations.x5 import X5Metadata  # noqa: E402
from brainhops.io.transformations.x5._raw import X5Node, X5Raw  # noqa: E402

# Three volumes, four slices along `k`.
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
    dtype="float32",
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
# The values as generic metadata holds them (`"a.u."` is a `Unit`).
HUB = Metadata(**FULL)


def _nifti_record() -> NiftiRaw:
    header = nb.Nifti1Header()
    header.set_data_shape((5, 5, 4, 3))
    header.set_xyzt_units("mm", "sec")
    return NiftiRaw(header=header)


def _x5_record() -> X5Raw:
    return X5Raw(nodes=[X5Node(transform=np.eye(4))])


# Each metadata format that stores a vocabulary, and a fresh record of
# it that can hold the vocabulary.
FORMATS = {
    NiftiMetadata: _nifti_record,
    MghMetadata: MghRaw,
    X5Metadata: _x5_record,
    BidsSidecar: dict,
}

# The fields of each format whose slot belongs to the geometry, which a
# consolidation reports as approximated.
GEOMETRY = {NiftiMetadata: {"space"}}


def test_every_format_with_a_vocabulary_is_in_the_matrix() -> None:
    formats = {
        cls
        for cls in MetadataFormat._REGISTRY
        if cls.__module__.startswith("brainhops.") and cls.SUPPORTED_FIELDS
    }
    assert formats == set(FORMATS)


def test_the_fixture_is_fully_populated() -> None:
    assert all(getattr(HUB, name) is not None for name in VOCABULARY)


@pytest.mark.parametrize("cls", list(FORMATS), ids=lambda c: c.__name__)
def test_the_constant_names_fields_of_the_vocabulary(cls) -> None:  # noqa: ANN001
    assert set(cls.SUPPORTED_FIELDS) <= set(FIELDS)
    assert len(set(cls.SUPPORTED_FIELDS)) == len(cls.SUPPORTED_FIELDS)


@pytest.mark.parametrize("cls", list(FORMATS), ids=lambda c: c.__name__)
def test_the_losses_are_the_complement_of_the_constant(cls) -> None:  # noqa: ANN001
    unsupported = set(FIELDS) - set(cls.SUPPORTED_FIELDS)
    lower = ConversionReport()
    converted = HUB.to(cls, on_loss=lower)
    assert set(lower.lost) == unsupported
    # The conversion carries every value; the consolidation drops them.
    for name in VOCABULARY:
        assert getattr(converted, name) == getattr(HUB, name), name
    exact = ConversionReport()
    fields = {name: FULL[name] for name in FIELDS}
    fresh = cls.from_raw(FORMATS[cls]())
    consolidated = fresh.consolidate(on_loss=exact, **fields)
    assert set(exact.lost) == unsupported
    assert set(exact.approximated) == GEOMETRY.get(cls, set())
    for name in sorted(set(cls.SUPPORTED_FIELDS) - GEOMETRY.get(cls, set())):
        assert getattr(consolidated, name) == getattr(HUB, name), name
    for name in unsupported:
        expected = {} if name == "extra" else None
        assert getattr(consolidated, name) == expected, name
    # And back: nothing is lost on the way to generic metadata.
    back = ConversionReport()
    generic = consolidated.to(Metadata, on_loss=back)
    assert not back.lossy
    for name in FIELDS:
        assert getattr(generic, name) == getattr(consolidated, name), name
