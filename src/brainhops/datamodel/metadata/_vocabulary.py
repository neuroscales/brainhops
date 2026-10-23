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
from bagof.magic import ConvertTo, Field, fields

# internals
from brainhops._core.enum import StrEnum

from ..base import DataModelBase
from ..enums import (
    AxisType,
    ContrastMethod,
    IlluminationType,
    IntentEnum,
    Manufacturer,
    SpaceEnum,
)
from ..units import Unit
from ._terms import (
    Channel,
    EncodingDirection,
    GeneratedBy,
    direction,
    to_dtype,
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
    |                 |                   | indexed when that axis changes, |
    |                 |                   | and cleared when the values     |
    |                 |                   | along that axis are             |
    |                 |                   | re-expressed in another frame.  |
    +-----------------+-------------------+---------------------------------+
    | `VALUES`        | `"values"`        | The field describes the values  |
    |                 |                   | of the data, such as their unit |
    |                 |                   | or their display range. It is   |
    |                 |                   | kept when the values are moved  |
    |                 |                   | or resampled, and cleared when  |
    |                 |                   | they are re-expressed in        |
    |                 |                   | another frame.                  |
    +-----------------+-------------------+---------------------------------+
    """

    FILE = "file"
    ACQUISITION = "acquisition"
    SPATIAL = "spatial"
    AXIS = "axis"
    VALUES = "values"


class Bids(Field):
    """
    The annotation that gives a vocabulary field its BIDS sidecar key.

    The key lands in the `metadata` of the field, under `"bids"`, where
    the sidecar codec reads it.

    Examples
    --------
    ```python
    repetition_time: tx.Annotated[
        tx.Optional[float], Bids("RepetitionTime")
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
    echo_time: tx.Annotated[
        tx.Optional[float], Scoped(Scope.ACQUISITION)
    ] = None
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
        tx.Optional[tx.Tuple[Channel, ...]], Along(AxisType.channel)
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


class Vocabulary(DataModelBase):
    """
    The base class of the vocabulary groups.

    A vocabulary group declares a few fields of the common vocabulary,
    and nothing else. [`Metadata`][] inherits every group, and so does
    the metadata of every file format, so that each group is one page of
    the documentation with the fields of one topic. A group is not meant
    to be instantiated. The groups convert their fields
    (`convert=True`), and hide from `repr` a field that holds `None`,
    because the fields of a base class keep the options of the class
    that declares them.
    """


class ProvenanceVocabulary(Vocabulary):
    """
    Vocabulary group: what the data is and where it comes from (`file`
    scope). Not meant to be instantiated; see [`Metadata`][].
    """

    name: tx.Annotated[tx.Optional[str], Scoped(Scope.FILE)] = None
    """A short name for the data."""

    description: tx.Annotated[
        tx.Optional[str],
        Bids("Description"),
        Scoped(Scope.FILE),
    ] = None
    """A free-text description."""

    history: tx.Annotated[
        tx.Optional[tx.Tuple[str, ...]],
        Scoped(Scope.FILE),
    ] = None
    """The commands that produced the data, oldest first."""

    generated_by: tx.Annotated[
        tx.Optional[tx.Tuple[GeneratedBy, ...]],
        Bids("GeneratedBy"),
        Scoped(Scope.FILE),
    ] = None
    """The programs that produced the data."""

    creation_time: tx.Annotated[
        tx.Optional[datetime.datetime],
        Scoped(Scope.FILE),
    ] = None
    """When the file was created."""

    sources: tx.Annotated[
        tx.Optional[tx.Tuple[str, ...]],
        Bids("Sources"),
        Scoped(Scope.FILE),
    ] = None
    """Files the data was derived from (BIDS provenance)."""

    space: tx.Annotated[
        tx.Optional[tx.Union[SpaceEnum, str]],
        Bids("SpatialReference"),
        Scoped(Scope.FILE),
    ] = None
    """
    The label of the world space (`'MNI152NLin6Asym'`, `'scanner'`, ...;
    a known one is a `SpaceEnum`); the space itself is geometry.
    """

    intent: tx.Annotated[
        tx.Optional[tx.Union[IntentEnum, str]],
        Scoped(Scope.FILE),
    ] = None
    """
    What the values are, as a NIfTI intent name
    (a known one is an `IntentEnum`).
    """


class MRIVocabulary(Vocabulary):
    """
    Vocabulary group: MRI acquisition parameters. They are in the
    `ACQUISITION` scope, except the encoding directions and the slice
    timing, which are tied to the spatial sampling (`SPATIAL` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    repetition_time: tx.Annotated[
        tx.Optional[float],
        Bids("RepetitionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Repetition time, in seconds."""

    echo_time: tx.Annotated[
        tx.Optional[float],
        Bids("EchoTime"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Echo time, in seconds."""

    inversion_time: tx.Annotated[
        tx.Optional[float],
        Bids("InversionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Inversion time, in seconds."""

    flip_angle: tx.Annotated[
        tx.Optional[float],
        Bids("FlipAngle"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Flip angle, in degrees."""

    magnetic_field_strength: tx.Annotated[
        tx.Optional[float],
        Bids("MagneticFieldStrength"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Nominal field strength, in tesla."""

    manufacturer: tx.Annotated[
        tx.Optional[tx.Union[Manufacturer, str]],
        Bids("Manufacturer"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Manufacturer of the equipment (a known one is a `Manufacturer`)."""

    manufacturers_model_name: tx.Annotated[
        tx.Optional[str],
        Bids("ManufacturersModelName"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Model name of the equipment."""

    institution_name: tx.Annotated[
        tx.Optional[str],
        Bids("InstitutionName"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Institution responsible for the equipment."""

    acquisition_time: tx.Annotated[
        tx.Optional[datetime.datetime],
        Bids("AcquisitionTime"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """When the acquisition started."""

    phase_encoding_direction: tx.Annotated[
        tx.Optional[EncodingDirection],
        Bids("PhaseEncodingDirection"),
        Scoped(Scope.SPATIAL),
        ConvertTo(direction),
    ] = None
    """
    Phase-encoding direction: a unit vector.
    By default in voxel axes (BIDS `'j-'` is accepted).
    """

    total_readout_time: tx.Annotated[
        tx.Optional[float],
        Bids("TotalReadoutTime"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Total readout time, in seconds."""

    effective_echo_spacing: tx.Annotated[
        tx.Optional[float],
        Bids("EffectiveEchoSpacing"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Effective echo spacing, in seconds."""

    slice_encoding_direction: tx.Annotated[
        tx.Optional[EncodingDirection],
        Bids("SliceEncodingDirection"),
        Scoped(Scope.SPATIAL),
        ConvertTo(direction),
    ] = None
    """
    Slice-encoding direction: a unit vector, by default in voxel axes
    (BIDS `'k'` is accepted)."
    """

    slice_timing: tx.Annotated[
        tx.Optional[tx.Tuple[float, ...]],
        Bids("SliceTiming"),
        Scoped(Scope.SPATIAL),
    ] = None
    """
    Acquisition time of each slice, in seconds, along the slice
    encoding direction.
    """

    multiband_acceleration_factor: tx.Annotated[
        tx.Optional[int],
        Bids("MultibandAccelerationFactor"),
        Scoped(Scope.ACQUISITION),
    ] = None
    """Multiband acceleration factor."""


class DiffusionVocabulary(Vocabulary):
    """
    Vocabulary group: the diffusion gradient table, one entry per index
    along the time axis (`AXIS` scope). BIDS stores it in `.bval` and
    `.bvec` files, not in the sidecar. Not meant to
    be instantiated; see [`Metadata`][].
    """

    bvalues: tx.Annotated[
        tx.Optional[tx.Tuple[float, ...]],
        Along(AxisType.time),
    ] = None
    """
    One b-value per index along the time axis (the volumes of a diffusion
    series), in s/mm^2."
    """

    bvectors: tx.Annotated[
        tx.Optional[tx.Tuple[tx.Tuple[float, float, float], ...]],
        Along(AxisType.time),
    ] = None
    """
    One unit gradient direction per index along the time axis, in world
    (RAS) coordinates."
    """


class DisplayVocabulary(Vocabulary):
    """
    Vocabulary group: how the values are shown and what they are. The
    channels run along the channel axis (`AXIS` scope); the display range
    and the unit apply to every value (`VALUES` scope). Not meant to be
    instantiated; see [`Metadata`][].
    """

    display_range: tx.Annotated[
        tx.Optional[tx.Tuple[float, float]],
        Scoped(Scope.VALUES),
    ] = None
    """Display window `(min, max)`."""

    channels: tx.Annotated[
        tx.Optional[tx.Tuple[Channel, ...]],
        Along(AxisType.channel),
    ] = None
    """One description per channel."""

    data_unit: tx.Annotated[
        tx.Optional[tx.Union[Unit, str]],
        Scoped(Scope.VALUES),
        ConvertTo(unit),
    ] = None
    """
    Unit of the data values: a `Unit` whenever the units module parses
    the name (`'a.u.'`, `'mm/s'`, `'HU'`), and the name itself, as a
    string, when it does not, so that a file with an odd unit still reads.
    Formats write it as its symbol (`Unit.symbol`), which parses back to
    the same unit.
    """


class StorageVocabulary(Vocabulary):
    """
    Vocabulary group: how the values of the data are stored in the file
    (`values` scope). Not meant to be instantiated; see [`Metadata`][].

    The loaded array always holds the values themselves. A file that
    stores integers with an intensity scaling (NIfTI `scl_slope` and
    `scl_inter`) loads as floating-point values, which are
    `stored * scale_slope + scale_intercept`. When an image is written,
    its values are stored as `dtype`, so that a label map read as
    `uint8` is written as `uint8` again, even after a computation that
    gave `int64` values. An override given to `save` or to `consolidate`
    takes precedence over `dtype`, and the type of the data is used only
    when neither is given.
    """

    dtype: tx.Annotated[
        tx.Optional[np.dtype],
        Scoped(Scope.VALUES),
        ConvertTo(to_dtype),
    ] = None
    """
    The element type of the data as stored, in native byte order.

    It may differ from the type of the loaded array: a scaled integer
    file loads as floating-point values. An override given to `save` or
    to `consolidate` takes precedence over it.
    """

    scale_slope: tx.Annotated[tx.Optional[float], Scoped(Scope.VALUES)] = None
    """
    The slope of the intensity scaling:
    a value is `stored * scale_slope + scale_intercept`.
    `None` means no scaling.
    """

    scale_intercept: tx.Annotated[
        tx.Optional[float],
        Scoped(Scope.VALUES),
    ] = None
    """
    The intercept of the intensity scaling (see `scale_slope`).
    `None` means no intercept.
    """


class MicroscopyVocabulary(Vocabulary):
    """
    Vocabulary group: microscopy acquisition (`acquisition` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    objective_magnification: tx.Annotated[
        tx.Optional[float],
        Scoped(Scope.ACQUISITION),
    ] = None
    """Nominal magnification of the objective."""

    objective_numerical_aperture: tx.Annotated[
        tx.Optional[float],
        Scoped(Scope.ACQUISITION),
    ] = None
    """Numerical aperture of the objective."""

    illumination_type: tx.Annotated[
        tx.Optional[tx.Union[IlluminationType, str]],
        Scoped(Scope.ACQUISITION),
    ] = None
    """Illumination type (a known one is an `IlluminationType`)."""

    contrast_method: tx.Annotated[
        tx.Optional[tx.Union[ContrastMethod, str]],
        Scoped(Scope.ACQUISITION),
    ] = None
    """Contrast method (a known one is a `ContrastMethod`)."""


class TransformVocabulary(Vocabulary):
    """
    Vocabulary group: what a transformation relates (`file` scope). Not
    meant to be instantiated; see [`Metadata`][].
    """

    moving: tx.Annotated[
        tx.Optional[str],
        Scoped(Scope.FILE),
    ] = None
    """The moving image of a registration (file reference)."""

    fixed: tx.Annotated[
        tx.Optional[str],
        Scoped(Scope.FILE),
    ] = None
    """The fixed image of a registration (file reference)."""

    input_space: tx.Annotated[
        tx.Optional[tx.Union[SpaceEnum, str]],
        Scoped(Scope.FILE),
    ] = None
    """Label of the space a transformation maps from."""

    output_space: tx.Annotated[
        tx.Optional[tx.Union[SpaceEnum, str]],
        Scoped(Scope.FILE),
    ] = None
    """Label of the space a transformation maps to."""


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
"""The fields of the metadata: `extra`, then the vocabulary. They are the
names that `consolidate` takes as overrides, in the order reports list
them."""


BIDS_KEYS: tx.Dict[str, str] = {
    field.name: field.metadata["bids"]
    for group in GROUPS
    for field in fields(group)
    if "bids" in (field.metadata or {})
}
"""The BIDS sidecar key of each vocabulary field that BIDS has a key
for, as `Bids(...)` declares it."""


SCOPES: tx.Dict[str, Scope] = {
    field.name: (field.metadata or {}).get("scope", Scope.FILE)
    for group in GROUPS
    for field in fields(group)
}
"""The propagation scope of each vocabulary field (`Scoped` or `Along`)."""


ALONG: tx.Dict[str, AxisType] = {
    field.name: field.metadata["along"]
    for group in GROUPS
    for field in fields(group)
    if "along" in (field.metadata or {})
}
"""The type of the axis that the entries of each field in the `AXIS`
scope run along, as `Along` declares it."""
