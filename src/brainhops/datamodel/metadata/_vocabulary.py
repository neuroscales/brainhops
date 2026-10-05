"""The common vocabulary: its scopes, its annotations and its groups."""

__all__ = [
    "ACQUISITION",
    "BIDS_KEYS",
    "FIELDS",
    "FILE",
    "GRID",
    "GROUPS",
    "SCOPES",
    "VOCABULARY",
    "VOLUME",
    "Bids",
    "DiffusionVocabulary",
    "DisplayVocabulary",
    "MRIVocabulary",
    "MicroscopyVocabulary",
    "ProvenanceVocabulary",
    "Scope",
    "StorageVocabulary",
    "TransformVocabulary",
]

# stdlib
import datetime
import itertools

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo, Field, Magic, fields

# internals
from ..enums import (
    ContrastMethod,
    IlluminationType,
    IntentEnum,
    Manufacturer,
    SpaceEnum,
)
from ..units import Unit
from ._sentinel import Maybe
from ._terms import (
    Channel,
    EncodingDirection,
    GeneratedBy,
    MaybeEnumConverter,
    direction,
    dtype,
    unit,
)

FILE = "file"
"""Scope of a field about the file itself (kept by `derive`)."""


ACQUISITION = "acquisition"
"""Scope of a field invariant under resampling (kept by `derive`)."""


GRID = "grid"
"""Scope of a field tied to the voxel grid (cleared when it changes)."""


VOLUME = "volume"
"""Scope of a field with one entry per volume or channel."""


class Bids(Field):
    """
    The BIDS sidecar key of a vocabulary field.

    Used as an annotation, `tx.Annotated[Maybe[float],
    Bids("RepetitionTime")]`; the key lands in the field's `metadata`
    under `"bids"`, where the sidecar codec reads it.
    """

    def __init__(self, key: str) -> None:
        super().__init__(metadata={"bids": key})


class Scope(Field):
    """
    The propagation scope of a vocabulary field: one of `"file"`,
    `"acquisition"`, `"grid"` or `"volume"` (see `derive`).

    Used as an annotation; the scope lands in the field's `metadata`
    under `"scope"`.
    """

    def __init__(self, scope: str) -> None:
        scopes = (FILE, ACQUISITION, GRID, VOLUME)
        if scope not in scopes:
            raise ValueError(f"A scope is one of {scopes}, not {scope!r}.")
        super().__init__(metadata={"scope": scope})


# ----------------------------------------------------------------------
#   VOCABULARY GROUPS
# ----------------------------------------------------------------------


class Vocabulary(Magic, kw_only=True, convert=True):
    """
    The base class of the vocabulary groups.

    A vocabulary group is a `Magic` mixin that declares a few fields of
    the common vocabulary, and nothing else. [`Metadata`][] inherits every
    group, and a format names the groups it can store in its `supports=`
    declaration. A group is not meant to be instantiated. The groups
    convert their fields (`convert=True`), because the fields of a mixin
    keep the options of the class that declares them.
    """


class ProvenanceVocabulary(Vocabulary):
    """
    Vocabulary group: what the data is and where it comes from (`file`
    scope). Not meant to be instantiated; see [`Metadata`][].
    """

    name: tx.Annotated[
        Maybe[str], tx.Doc("A short name for the data."), Scope(FILE)
    ] = None

    description: tx.Annotated[
        Maybe[str],
        tx.Doc("A free-text description."),
        Bids("Description"),
        Scope(FILE),
    ] = None

    history: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("The commands that produced the data, oldest first."),
        Scope(FILE),
    ] = None

    generated_by: tx.Annotated[
        Maybe[tx.Tuple[GeneratedBy, ...]],
        tx.Doc("The programs that produced the data."),
        Bids("GeneratedBy"),
        Scope(FILE),
    ] = None

    creation_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the file was created."),
        Scope(FILE),
    ] = None

    sources: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("Files the data was derived from (BIDS provenance)."),
        Bids("Sources"),
        Scope(FILE),
    ] = None

    space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc(
            "The label of the world space (`'MNI152NLin6Asym'`, "
            "`'scanner'`, ...; a known one is a `SpaceEnum`); the space "
            "itself is geometry."
        ),
        Bids("SpatialReference"),
        Scope(FILE),
        ConvertTo(MaybeEnumConverter(SpaceEnum)),
    ] = None

    intent: tx.Annotated[
        Maybe[tx.Union[IntentEnum, str]],
        tx.Doc(
            "What the values are, as a NIfTI intent name (a known one is "
            "an `IntentEnum`)."
        ),
        Scope(FILE),
        ConvertTo(MaybeEnumConverter(IntentEnum)),
    ] = None


class MRIVocabulary(Vocabulary):
    """
    Vocabulary group: MRI acquisition parameters (`acquisition` scope,
    except the encoding directions and the slice timing, `grid`). Not
    meant to be instantiated; see [`Metadata`][].
    """

    repetition_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Repetition time, in seconds."),
        Bids("RepetitionTime"),
        Scope(ACQUISITION),
    ] = None

    echo_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Echo time, in seconds."),
        Bids("EchoTime"),
        Scope(ACQUISITION),
    ] = None

    inversion_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Inversion time, in seconds."),
        Bids("InversionTime"),
        Scope(ACQUISITION),
    ] = None

    flip_angle: tx.Annotated[
        Maybe[float],
        tx.Doc("Flip angle, in degrees."),
        Bids("FlipAngle"),
        Scope(ACQUISITION),
    ] = None

    magnetic_field_strength: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal field strength, in tesla."),
        Bids("MagneticFieldStrength"),
        Scope(ACQUISITION),
    ] = None

    manufacturer: tx.Annotated[
        Maybe[tx.Union[Manufacturer, str]],
        tx.Doc(
            "Manufacturer of the equipment (a known one is a `Manufacturer`)."
        ),
        Bids("Manufacturer"),
        Scope(ACQUISITION),
        ConvertTo(MaybeEnumConverter(Manufacturer)),
    ] = None

    manufacturers_model_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Model name of the equipment."),
        Bids("ManufacturersModelName"),
        Scope(ACQUISITION),
    ] = None

    institution_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Institution responsible for the equipment."),
        Bids("InstitutionName"),
        Scope(ACQUISITION),
    ] = None

    acquisition_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the acquisition started."),
        Bids("AcquisitionTime"),
        Scope(ACQUISITION),
    ] = None

    phase_encoding_direction: tx.Annotated[
        Maybe[EncodingDirection],
        tx.Doc(
            "Phase-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'j-'` is accepted)."
        ),
        Bids("PhaseEncodingDirection"),
        Scope(GRID),
        ConvertTo(direction),
    ] = None

    total_readout_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Total readout time, in seconds."),
        Bids("TotalReadoutTime"),
        Scope(ACQUISITION),
    ] = None

    effective_echo_spacing: tx.Annotated[
        Maybe[float],
        tx.Doc("Effective echo spacing, in seconds."),
        Bids("EffectiveEchoSpacing"),
        Scope(ACQUISITION),
    ] = None

    slice_encoding_direction: tx.Annotated[
        Maybe[EncodingDirection],
        tx.Doc(
            "Slice-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'k'` is accepted)."
        ),
        Bids("SliceEncodingDirection"),
        Scope(GRID),
        ConvertTo(direction),
    ] = None

    slice_timing: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc(
            "Acquisition time of each slice, in seconds, along the slice "
            "encoding direction."
        ),
        Bids("SliceTiming"),
        Scope(GRID),
    ] = None

    multiband_acceleration_factor: tx.Annotated[
        Maybe[int],
        tx.Doc("Multiband acceleration factor."),
        Bids("MultibandAccelerationFactor"),
        Scope(ACQUISITION),
    ] = None


class DiffusionVocabulary(Vocabulary):
    """
    Vocabulary group: the diffusion gradient table (`volume` scope). BIDS
    stores it in `.bval`/`.bvec` files, not in the sidecar. Not meant to
    be instantiated; see [`Metadata`][].
    """

    bvalues: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc("One b-value per volume, in s/mm^2."),
        Scope(VOLUME),
    ] = None

    bvectors: tx.Annotated[
        Maybe[tx.Tuple[tx.Tuple[float, float, float], ...]],
        tx.Doc(
            "One unit gradient direction per volume, in world (RAS) "
            "coordinates."
        ),
        Scope(VOLUME),
    ] = None


class DisplayVocabulary(Vocabulary):
    """
    Vocabulary group: how the values are shown and what they are
    (`volume` scope). Not meant to be instantiated; see [`Metadata`][].
    """

    display_range: tx.Annotated[
        Maybe[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
        Scope(VOLUME),
    ] = None

    channels: tx.Annotated[
        Maybe[tx.Tuple[Channel, ...]],
        tx.Doc("One description per channel."),
        Scope(VOLUME),
    ] = None

    data_unit: tx.Annotated[
        Maybe[tx.Union[Unit, str]],
        tx.Doc(
            "Unit of the data values: a `Unit` whenever the units module "
            "parses the name (`'a.u.'`, `'mm/s'`, `'HU'`), and the name "
            "itself, as a string, when it does not, so that a file with "
            "an odd unit still reads. Formats write it as its symbol "
            "(`Unit.symbol`), which parses back to the same unit."
        ),
        Scope(VOLUME),
        ConvertTo(unit),
    ] = None


class StorageVocabulary(Vocabulary):
    """
    Vocabulary group: how the values of the data are stored in the file
    (`file` scope). Not meant to be instantiated; see [`Metadata`][].

    The loaded array always holds the values themselves. A file that
    stores integers with an intensity scaling (NIfTI `scl_slope` and
    `scl_inter`) loads as floating-point values, which are
    `stored * scale_slope + scale_intercept`. A writer stores the data as
    `data_type`, with the scaling, when the values of the array are of
    that kind and fit the scaling, so that a scaled integer file is
    written back as it was read. Otherwise the writer stores the array as
    it is, and reports a changed field it could not use.
    """

    data_type: tx.Annotated[
        Maybe[np.dtype],
        tx.Doc(
            "The element type of the data as stored, in native byte "
            "order. It may differ from the type of the loaded array: a "
            "scaled integer file loads as floating-point values. A "
            "`dtype=` writer option wins over it."
        ),
        Scope(FILE),
        ConvertTo(dtype),
    ] = None

    scale_slope: tx.Annotated[
        Maybe[float],
        tx.Doc(
            "The slope of the intensity scaling: a value is `stored * "
            "scale_slope + scale_intercept`. `None` means no scaling."
        ),
        Scope(FILE),
    ] = None

    scale_intercept: tx.Annotated[
        Maybe[float],
        tx.Doc(
            "The intercept of the intensity scaling (see `scale_slope`). "
            "`None` means no intercept."
        ),
        Scope(FILE),
    ] = None


class MicroscopyVocabulary(Vocabulary):
    """
    Vocabulary group: microscopy acquisition (`acquisition` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    objective_magnification: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal magnification of the objective."),
        Scope(ACQUISITION),
    ] = None

    objective_numerical_aperture: tx.Annotated[
        Maybe[float],
        tx.Doc("Numerical aperture of the objective."),
        Scope(ACQUISITION),
    ] = None

    illumination_type: tx.Annotated[
        Maybe[tx.Union[IlluminationType, str]],
        tx.Doc("Illumination type (a known one is an `IlluminationType`)."),
        Scope(ACQUISITION),
        ConvertTo(MaybeEnumConverter(IlluminationType)),
    ] = None

    contrast_method: tx.Annotated[
        Maybe[tx.Union[ContrastMethod, str]],
        tx.Doc("Contrast method (a known one is a `ContrastMethod`)."),
        Scope(ACQUISITION),
        ConvertTo(MaybeEnumConverter(ContrastMethod)),
    ] = None


class TransformVocabulary(Vocabulary):
    """
    Vocabulary group: what a transformation relates (`file` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    moving: tx.Annotated[
        Maybe[str],
        tx.Doc("The moving image of a registration (file reference)."),
        Scope(FILE),
    ] = None

    fixed: tx.Annotated[
        Maybe[str],
        tx.Doc("The fixed image of a registration (file reference)."),
        Scope(FILE),
    ] = None

    input_space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc("Label of the space a transformation maps from."),
        Scope(FILE),
        ConvertTo(MaybeEnumConverter(SpaceEnum)),
    ] = None

    output_space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc("Label of the space a transformation maps to."),
        Scope(FILE),
        ConvertTo(MaybeEnumConverter(SpaceEnum)),
    ] = None


GROUPS: tx.Dict[type, tx.Tuple[str, ...]] = {
    # `fields` lists exactly the fields of the group: the base class
    # declares none.
    group: tuple(field.name for field in fields(group))
    for group in (
        ProvenanceVocabulary,
        MRIVocabulary,
        DiffusionVocabulary,
        DisplayVocabulary,
        StorageVocabulary,
        MicroscopyVocabulary,
        TransformVocabulary,
    )
}
"""Each vocabulary group class, and the names of its fields."""


VOCABULARY: tx.Tuple[str, ...] = tuple(
    itertools.chain.from_iterable(GROUPS.values())
)
"""
The names of the common vocabulary fields, group by group. `extra` is not
one (it is the free-form store next to them). `bagof` lists the fields of
a class with several bases in reverse MRO order, so this is the order to
iterate in, never `fields(Metadata)`.
"""

FIELDS: tx.Tuple[str, ...] = ("extra",) + VOCABULARY
"""The fields `supports=` speaks of, in the order reports list them."""


BIDS_KEYS: tx.Dict[str, str] = {
    field.name: field.metadata["bids"]
    for group in GROUPS
    for field in fields(group)
    if "bids" in (field.metadata or {})
}
"""Vocabulary field -> its BIDS sidecar key (`Bids(...)`), for the fields
BIDS has a key for."""


SCOPES: tx.Dict[str, str] = {
    field.name: (field.metadata or {}).get("scope", FILE)
    for group in GROUPS
    for field in fields(group)
}
"""Vocabulary field -> its propagation scope (`Scope(...)`)."""
