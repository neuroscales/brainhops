"""The common vocabulary: its scopes, its annotations and its groups."""

__all__ = [
    "ALONG",
    "BIDS_KEYS",
    "FIELDS",
    "GROUPS",
    "SCOPES",
    "VOCABULARY",
    "Vocabulary",
    "Along",
    "Bids",
    "DiffusionVocabulary",
    "DisplayVocabulary",
    "MRIVocabulary",
    "MicroscopyVocabulary",
    "ProvenanceVocabulary",
    "Scope",
    "Scoped",
    "StorageVocabulary",
    "TransformVocabulary",
]

# stdlib
import datetime
import itertools

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import ConvertTo, Field, HideIf, Magic, fields

# internals
from brainhops._core.enum import StrEnum

from ..enums import (
    AxisType,
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
    _is_absent,
    direction,
    dtype,
    unit,
)


class Scope(StrEnum):
    """
    How a vocabulary field propagates when an image is derived from
    another one (see `Metadata.derive`).

    +-----------------+-------------------+---------------------------------+
    | Member          | Value             | Meaning                         |
    +=================+===================+=================================+
    | `FILE`          | `"file"`          | The field is about the file or  |
    |                 |                   | about the data as a whole. It   |
    |                 |                   | is kept.                        |
    +-----------------+-------------------+---------------------------------+
    | `ACQUISITION`   | `"acquisition"`   | The field describes the         |
    |                 |                   | acquisition, and does not       |
    |                 |                   | change under a resampling. It   |
    |                 |                   | is kept.                        |
    +-----------------+-------------------+---------------------------------+
    | `SPATIAL`       | `"spatial"`       | The field is tied to the        |
    |                 |                   | spatial sampling. It is cleared |
    |                 |                   | when the spatial axes change,   |
    |                 |                   | or mapped along with them.      |
    +-----------------+-------------------+---------------------------------+
    | `AXIS`          | `"axis"`          | The field has one entry per     |
    |                 |                   | index along one non-spatial     |
    |                 |                   | axis (see [`Along`][]). It is   |
    |                 |                   | indexed when that axis changes. |
    +-----------------+-------------------+---------------------------------+
    """

    FILE = "file"
    ACQUISITION = "acquisition"
    SPATIAL = "spatial"
    AXIS = "axis"


class Bids(Field):
    """
    The annotation that gives a vocabulary field its BIDS sidecar key.

    The key lands in the `metadata` of the field, under `"bids"`, where
    the sidecar codec reads it.

    Examples
    --------
    ```python
    repetition_time: tx.Annotated[
        Maybe[float], Bids("RepetitionTime")
    ] = None
    ```
    """

    def __init__(self, key: str) -> None:
        """
        Parameters
        ----------
        key : str
            The BIDS key, such as `"RepetitionTime"`.
        """
        super().__init__(metadata={"bids": key})


class Scoped(Field):
    """
    The annotation that gives a vocabulary field its [`Scope`][].

    The scope lands in the `metadata` of the field, under `"scope"`. A
    field without a scope is in the `FILE` scope. A field in the `AXIS`
    scope is declared with [`Along`][] instead, which also names its axis.

    Examples
    --------
    ```python
    echo_time: tx.Annotated[Maybe[float], Scoped(Scope.ACQUISITION)] = None
    ```
    """

    def __init__(self, scope: tx.Union[Scope, str]) -> None:
        """
        Parameters
        ----------
        scope : Scope or str
            The scope of the field.

        Raises
        ------
        ValueError
            If `scope` is not a scope, or is `AXIS` (use [`Along`][]).
        """
        scope = Scope(scope)
        if scope is Scope.AXIS:
            raise ValueError(
                "A field in the AXIS scope names its axis: use Along(...)."
            )
        super().__init__(metadata={"scope": scope})


class Along(Field):
    """
    The annotation of a vocabulary field that has one entry per index
    along one non-spatial axis of an image.

    The field is in the `AXIS` scope, and its entries run along the axes
    of the given type. NIfTI and OME-Zarr both map their time and channel
    dimensions to brainhops axis types, so the b-values of a diffusion
    image run along the time axis (NIfTI dimension 4), and the
    descriptions of the channels run along the channel axis.

    The axis type lands in the `metadata` of the field, under `"along"`,
    next to the scope.

    Examples
    --------
    ```python
    channels: tx.Annotated[
        Maybe[tx.Tuple[Channel, ...]], Along(AxisType.channel)
    ] = None
    ```
    """

    def __init__(self, axis: tx.Union[AxisType, str]) -> None:
        """
        Parameters
        ----------
        axis : AxisType or str
            The type of the axis the entries run along. It cannot be
            `space`: a field tied to the spatial axes is in the `SPATIAL`
            scope.

        Raises
        ------
        ValueError
            If `axis` is not an axis type, or is `space`.
        """
        axis = AxisType(axis)
        if axis is AxisType.space:
            raise ValueError(
                "A field tied to the spatial axes is Scoped(Scope.SPATIAL)."
            )
        super().__init__(metadata={"scope": Scope.AXIS, "along": axis})


# ----------------------------------------------------------------------
#   VOCABULARY GROUPS
# ----------------------------------------------------------------------


class Vocabulary(
    Magic,
    kw_only=True,
    convert=True,
    # A field is shown in `repr` only when it holds a value: not `None`
    # (unknown), not `UNSUPPORTED` (no slot).
    repr=HideIf(_is_absent),
):
    """
    The base class of the vocabulary groups.

    A vocabulary group is a `Magic` mixin that declares a few fields of
    the common vocabulary, and nothing else. [`Metadata`][] inherits every
    group, and a format names the groups it can store in its `supports=`
    declaration. A group is not meant to be instantiated. The groups
    convert their fields (`convert=True`), and hide from `repr` a field
    that holds `None` or `UNSUPPORTED`, because the fields of a mixin
    keep the options of the class that declares them.
    """


class ProvenanceVocabulary(Vocabulary):
    """
    Vocabulary group: what the data is and where it comes from (`file`
    scope). Not meant to be instantiated; see [`Metadata`][].
    """

    name: tx.Annotated[
        Maybe[str], tx.Doc("A short name for the data."), Scoped(Scope.FILE)
    ] = None

    description: tx.Annotated[
        Maybe[str],
        tx.Doc("A free-text description."),
        Bids("Description"),
        Scoped(Scope.FILE),
    ] = None

    history: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("The commands that produced the data, oldest first."),
        Scoped(Scope.FILE),
    ] = None

    generated_by: tx.Annotated[
        Maybe[tx.Tuple[GeneratedBy, ...]],
        tx.Doc("The programs that produced the data."),
        Bids("GeneratedBy"),
        Scoped(Scope.FILE),
    ] = None

    creation_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the file was created."),
        Scoped(Scope.FILE),
    ] = None

    sources: tx.Annotated[
        Maybe[tx.Tuple[str, ...]],
        tx.Doc("Files the data was derived from (BIDS provenance)."),
        Bids("Sources"),
        Scoped(Scope.FILE),
    ] = None

    space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc(
            "The label of the world space (`'MNI152NLin6Asym'`, "
            "`'scanner'`, ...; a known one is a `SpaceEnum`); the space "
            "itself is geometry."
        ),
        Bids("SpatialReference"),
        Scoped(Scope.FILE),
        ConvertTo(MaybeEnumConverter(SpaceEnum)),
    ] = None

    intent: tx.Annotated[
        Maybe[tx.Union[IntentEnum, str]],
        tx.Doc(
            "What the values are, as a NIfTI intent name (a known one is "
            "an `IntentEnum`)."
        ),
        Scoped(Scope.FILE),
        ConvertTo(MaybeEnumConverter(IntentEnum)),
    ] = None


class MRIVocabulary(Vocabulary):
    """
    Vocabulary group: MRI acquisition parameters. They are in the
    `ACQUISITION` scope, except the encoding directions and the slice
    timing, which are tied to the spatial sampling (`SPATIAL` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    repetition_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Repetition time, in seconds."),
        Bids("RepetitionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None

    echo_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Echo time, in seconds."),
        Bids("EchoTime"),
        Scoped(Scope.ACQUISITION),
    ] = None

    inversion_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Inversion time, in seconds."),
        Bids("InversionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None

    flip_angle: tx.Annotated[
        Maybe[float],
        tx.Doc("Flip angle, in degrees."),
        Bids("FlipAngle"),
        Scoped(Scope.ACQUISITION),
    ] = None

    magnetic_field_strength: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal field strength, in tesla."),
        Bids("MagneticFieldStrength"),
        Scoped(Scope.ACQUISITION),
    ] = None

    manufacturer: tx.Annotated[
        Maybe[tx.Union[Manufacturer, str]],
        tx.Doc(
            "Manufacturer of the equipment (a known one is a `Manufacturer`)."
        ),
        Bids("Manufacturer"),
        Scoped(Scope.ACQUISITION),
        ConvertTo(MaybeEnumConverter(Manufacturer)),
    ] = None

    manufacturers_model_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Model name of the equipment."),
        Bids("ManufacturersModelName"),
        Scoped(Scope.ACQUISITION),
    ] = None

    institution_name: tx.Annotated[
        Maybe[str],
        tx.Doc("Institution responsible for the equipment."),
        Bids("InstitutionName"),
        Scoped(Scope.ACQUISITION),
    ] = None

    acquisition_time: tx.Annotated[
        Maybe[datetime.datetime],
        tx.Doc("When the acquisition started."),
        Bids("AcquisitionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None

    phase_encoding_direction: tx.Annotated[
        Maybe[EncodingDirection],
        tx.Doc(
            "Phase-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'j-'` is accepted)."
        ),
        Bids("PhaseEncodingDirection"),
        Scoped(Scope.SPATIAL),
        ConvertTo(direction),
    ] = None

    total_readout_time: tx.Annotated[
        Maybe[float],
        tx.Doc("Total readout time, in seconds."),
        Bids("TotalReadoutTime"),
        Scoped(Scope.ACQUISITION),
    ] = None

    effective_echo_spacing: tx.Annotated[
        Maybe[float],
        tx.Doc("Effective echo spacing, in seconds."),
        Bids("EffectiveEchoSpacing"),
        Scoped(Scope.ACQUISITION),
    ] = None

    slice_encoding_direction: tx.Annotated[
        Maybe[EncodingDirection],
        tx.Doc(
            "Slice-encoding direction: a unit vector, by default in voxel "
            "axes (BIDS `'k'` is accepted)."
        ),
        Bids("SliceEncodingDirection"),
        Scoped(Scope.SPATIAL),
        ConvertTo(direction),
    ] = None

    slice_timing: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc(
            "Acquisition time of each slice, in seconds, along the slice "
            "encoding direction."
        ),
        Bids("SliceTiming"),
        Scoped(Scope.SPATIAL),
    ] = None

    multiband_acceleration_factor: tx.Annotated[
        Maybe[int],
        tx.Doc("Multiband acceleration factor."),
        Bids("MultibandAccelerationFactor"),
        Scoped(Scope.ACQUISITION),
    ] = None


class DiffusionVocabulary(Vocabulary):
    """
    Vocabulary group: the diffusion gradient table, one entry per index
    along the time axis (`AXIS` scope). BIDS stores it in `.bval` and
    `.bvec` files, not in the sidecar. Not meant to
    be instantiated; see [`Metadata`][].
    """

    bvalues: tx.Annotated[
        Maybe[tx.Tuple[float, ...]],
        tx.Doc(
            "One b-value per index along the time axis (the volumes of a "
            "diffusion series), in s/mm^2."
        ),
        Along(AxisType.time),
    ] = None

    bvectors: tx.Annotated[
        Maybe[tx.Tuple[tx.Tuple[float, float, float], ...]],
        tx.Doc(
            "One unit gradient direction per index along the time axis, "
            "in world (RAS) coordinates."
        ),
        Along(AxisType.time),
    ] = None


class DisplayVocabulary(Vocabulary):
    """
    Vocabulary group: how the values are shown and what they are. The
    channels run along the channel axis (`AXIS` scope); the display range
    and the unit apply to every value (`FILE` scope). Not meant to be
    instantiated; see [`Metadata`][].
    """

    display_range: tx.Annotated[
        Maybe[tx.Tuple[float, float]],
        tx.Doc("Display window `(min, max)`."),
        Scoped(Scope.FILE),
    ] = None

    channels: tx.Annotated[
        Maybe[tx.Tuple[Channel, ...]],
        tx.Doc("One description per channel."),
        Along(AxisType.channel),
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
        Scoped(Scope.FILE),
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
        Scoped(Scope.FILE),
        ConvertTo(dtype),
    ] = None

    scale_slope: tx.Annotated[
        Maybe[float],
        tx.Doc(
            "The slope of the intensity scaling: a value is `stored * "
            "scale_slope + scale_intercept`. `None` means no scaling."
        ),
        Scoped(Scope.FILE),
    ] = None

    scale_intercept: tx.Annotated[
        Maybe[float],
        tx.Doc(
            "The intercept of the intensity scaling (see `scale_slope`). "
            "`None` means no intercept."
        ),
        Scoped(Scope.FILE),
    ] = None


class MicroscopyVocabulary(Vocabulary):
    """
    Vocabulary group: microscopy acquisition (`acquisition` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    objective_magnification: tx.Annotated[
        Maybe[float],
        tx.Doc("Nominal magnification of the objective."),
        Scoped(Scope.ACQUISITION),
    ] = None

    objective_numerical_aperture: tx.Annotated[
        Maybe[float],
        tx.Doc("Numerical aperture of the objective."),
        Scoped(Scope.ACQUISITION),
    ] = None

    illumination_type: tx.Annotated[
        Maybe[tx.Union[IlluminationType, str]],
        tx.Doc("Illumination type (a known one is an `IlluminationType`)."),
        Scoped(Scope.ACQUISITION),
        ConvertTo(MaybeEnumConverter(IlluminationType)),
    ] = None

    contrast_method: tx.Annotated[
        Maybe[tx.Union[ContrastMethod, str]],
        tx.Doc("Contrast method (a known one is a `ContrastMethod`)."),
        Scoped(Scope.ACQUISITION),
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
        Scoped(Scope.FILE),
    ] = None

    fixed: tx.Annotated[
        Maybe[str],
        tx.Doc("The fixed image of a registration (file reference)."),
        Scoped(Scope.FILE),
    ] = None

    input_space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc("Label of the space a transformation maps from."),
        Scoped(Scope.FILE),
        ConvertTo(MaybeEnumConverter(SpaceEnum)),
    ] = None

    output_space: tx.Annotated[
        Maybe[tx.Union[SpaceEnum, str]],
        tx.Doc("Label of the space a transformation maps to."),
        Scoped(Scope.FILE),
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


SCOPES: tx.Dict[str, Scope] = {
    field.name: (field.metadata or {}).get("scope", Scope.FILE)
    for group in GROUPS
    for field in fields(group)
}
"""Vocabulary field -> its propagation scope (`Scoped` or `Along`)."""


ALONG: tx.Dict[str, AxisType] = {
    field.name: field.metadata["along"]
    for group in GROUPS
    for field in fields(group)
    if "along" in (field.metadata or {})
}
"""Vocabulary field in the `AXIS` scope -> the type of the axis its
entries run along (`Along`)."""
