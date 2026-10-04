"""The common vocabulary: its scopes, its annotations and its groups."""

__all__ = [
    "ACQUISITION",
    "FILE",
    "GRID",
    "GROUPS",
    "VOCABULARY",
    "VOLUME",
    "Bids",
    "DiffusionMetadata",
    "DisplayMetadata",
    "MRIMetadata",
    "MicroscopyMetadata",
    "ProvenanceMetadata",
    "Scope",
    "TransformMetadata",
]

# stdlib
import datetime
import itertools

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo, Field, Magic

# internals
from brainhops._core.compat import own_annotations

from ..enums import (
    ContrastMethod,
    IlluminationType,
    Intent,
    Manufacturer,
    Space,
)
from ..units import Unit
from ._sentinel import Maybe
from ._terms import (
    Channel,
    EncodingDirection,
    GeneratedBy,
    _direction,
    _dtype,
    _Term,
    _unit,
)

# ----------------------------------------------------------------------
#   FIELD ANNOTATIONS
# ----------------------------------------------------------------------

FILE = "file"
"""Scope of a field about the file itself (kept by `derive`)."""

ACQUISITION = "acquisition"
"""Scope of a field invariant under resampling (kept by `derive`)."""

GRID = "grid"
"""Scope of a field tied to the voxel grid (cleared when it changes)."""

VOLUME = "volume"
"""Scope of a field with one entry per volume or channel."""

_SCOPES = (FILE, ACQUISITION, GRID, VOLUME)


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
        if scope not in _SCOPES:
            raise ValueError(f"A scope is one of {_SCOPES}, not {scope!r}.")
        super().__init__(metadata={"scope": scope})


# ----------------------------------------------------------------------
#   VOCABULARY GROUPS
# ----------------------------------------------------------------------


class _VocabularyGroup(Magic, kw_only=True, convert=True):
    """
    Base of the vocabulary groups: `Magic` mixins that declare fields and
    nothing else. They convert their fields (`convert=True`: a mixin's
    fields keep the options of the class that declares them), and are
    not meant to be instantiated: [`Metadata`][] inherits them all.
    """


class ProvenanceMetadata(_VocabularyGroup):
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
        Maybe[tx.Union[Space, str]],
        tx.Doc(
            "The label of the world space (`'MNI152NLin6Asym'`, "
            "`'scanner'`, ...; a known one is a `Space`); the space itself "
            "is geometry."
        ),
        Bids("SpatialReference"),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None

    intent: tx.Annotated[
        Maybe[tx.Union[Intent, str]],
        tx.Doc(
            "What the values are, as a NIfTI intent name (a known one is "
            "an `Intent`)."
        ),
        Scope(FILE),
        ConvertTo(_Term(Intent)),
    ] = None


class MRIMetadata(_VocabularyGroup):
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
        ConvertTo(_Term(Manufacturer)),
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
            "axes (BIDS `'j-'` is accepted and compares equal)."
        ),
        Bids("PhaseEncodingDirection"),
        Scope(GRID),
        ConvertTo(_direction),
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
            "axes (BIDS `'k'` is accepted and compares equal)."
        ),
        Bids("SliceEncodingDirection"),
        Scope(GRID),
        ConvertTo(_direction),
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


class DiffusionMetadata(_VocabularyGroup):
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


class DisplayMetadata(_VocabularyGroup):
    """
    Vocabulary group: how the values are shown and what they are
    (`volume` scope, except `data_type`, `grid`). Not meant to be
    instantiated; see [`Metadata`][].
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
        ConvertTo(_unit),
    ] = None

    data_type: tx.Annotated[
        Maybe[np.dtype],
        tx.Doc(
            "The element type of the data as stored (in native byte "
            "order), which may differ from the type of the loaded array "
            "(a scaled integer file loads as floats). Writers store the "
            "data as this type when the array's values are of its kind, "
            "and a `dtype=` writer option wins over it."
        ),
        Scope(GRID),
        ConvertTo(_dtype),
    ] = None


class MicroscopyMetadata(_VocabularyGroup):
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
        ConvertTo(_Term(IlluminationType)),
    ] = None

    contrast_method: tx.Annotated[
        Maybe[tx.Union[ContrastMethod, str]],
        tx.Doc("Contrast method (a known one is a `ContrastMethod`)."),
        Scope(ACQUISITION),
        ConvertTo(_Term(ContrastMethod)),
    ] = None


class TransformMetadata(_VocabularyGroup):
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
        Maybe[tx.Union[Space, str]],
        tx.Doc("Label of the space a transformation maps from."),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None

    output_space: tx.Annotated[
        Maybe[tx.Union[Space, str]],
        tx.Doc("Label of the space a transformation maps to."),
        Scope(FILE),
        ConvertTo(_Term(Space)),
    ] = None


_GROUP_CLASSES = (
    ProvenanceMetadata,
    MRIMetadata,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    TransformMetadata,
)

GROUPS: tx.Dict[type, tx.Tuple[str, ...]] = {
    group: tuple(own_annotations(group)) for group in _GROUP_CLASSES
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
