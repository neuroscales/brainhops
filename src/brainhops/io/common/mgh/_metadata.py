"""The metadata of an MGH file."""

# stdlib
import functools
import math

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.numeric import shortest_decimal
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileWriter
from brainhops.io.metadata import MetadataFormat, MetadataReader
from brainhops.io.metadata._base import consolidated

# this format
from ._raw import MghRaw
from ._tags import decode_history, encode_history


@register_format
class MghMetadata(MetadataReader, BinaryFileWriter, MetadataFormat):
    """Metadata of an MGH or MGZ file, which holds everything but the voxels.

    The metadata of an MGH file is its header, the footer of acquisition
    parameters and the trailing tags, held as an [`MghRaw`][] record.
    [`MghImage`][brainhops.io.images.freesurfer.mgh.MghImage] uses that
    record as the base of the file that it writes. Reading the metadata
    never decodes the voxels, and a file cut after its header can still be
    read. The metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads an
    [`MghRaw`][] and wraps it. Writing the metadata writes the fixed header
    of the consolidated record alone, without the footer and the tags,
    which a file stores after the voxels.

    The vocabulary is decoded when the metadata is built. The fields that
    MGH stores are listed in `SUPPORTED_FIELDS`, in these slots.

    | Field             | Slot                   | Unit in the file    |
    |-------------------|------------------------|---------------------|
    | `repetition_time` | footer `tr`            | milliseconds        |
    | `echo_time`       | footer `te`            | milliseconds        |
    | `inversion_time`  | footer `ti`            | milliseconds        |
    | `flip_angle`      | footer `flip_angle`    | radians             |
    | `history`         | the `TAG_CMDLINE` tags | one command per tag |
    | `dtype`           | header `type`          | one of four types   |

    A zero in the footer means that the parameter was not recorded, and
    reads as `None`. `history` is decoded only when the whole tag stream
    parses, and writing it replaces the command lines and keeps every
    other tag. MGH stores the types `uint8`, `int16`, `int32` and
    `float32`, and another type is reported as approximated.
    MGH has no free-form store, so `extra` is not supported.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mgh", ".mgz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mgh", "mgz")
    RAW: tx.ClassVar[type] = MghRaw
    SUPPORTED_FIELDS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "repetition_time",
        "echo_time",
        "inversion_time",
        "flip_angle",
        "history",
        "dtype",
    )
    """The fields of the vocabulary that an MGH file stores."""

    raw: NoRepr[tx.Optional[MghRaw]] = None
    """The record of the file, or `None` for metadata without a record."""

    @classmethod
    def from_raw(
        cls, raw: tx.Optional[MghRaw], **kwargs: tx.Any
    ) -> "MghMetadata":
        """Build the metadata of a record, decoding its vocabulary.

        Parameters
        ----------
        raw : MghRaw, optional
            The record, held as it is given.
        **kwargs : Any
            Other fields, which take precedence over the decoded values.

        Returns
        -------
        MghMetadata
            The metadata.
        """
        return cls(raw=raw, **{**_decode(raw), **kwargs})

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> "MghMetadata":
        """Encode the fields that differ from the record into a copy of it.

        The contract is that of `consolidated`:
        the changes are written into a copy of the record, or into a new
        record, and the result is decoded from it afresh.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this metadata rather than a copy.
        **fields : Any
            Fields to override before the encoding.

        Returns
        -------
        MghMetadata
            The consolidated copy, or this metadata with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If a field is reported under the `"raise"` policy.
        """
        return consolidated(
            self,
            _encode,
            MghRaw,
            inplace=inplace,
            fields=fields,
        )

    @takes_error_policy
    def to_bytes(
        self,
        **fields: tx.Any,
    ) -> bytes:
        """Return the fixed header of the consolidated record.

        The metadata is consolidated first, with the overrides given, and
        the fixed header of the record is returned as an MGH file stores
        it. The footer and the tags, which a file stores after the voxels,
        are not written. Metadata without a record is written as the
        header of a new record.

        Parameters
        ----------
        **fields : Any
            Fields to override, as in [`consolidate`][].

        Returns
        -------
        bytes
            The fixed header.
        """
        raw = self.consolidate(**fields).raw
        return (MghRaw() if raw is None else raw).to_bytes()

    @takes_error_policy
    def to_filename(
        self,
        filename: path.FilenameLike,
        **fields: tx.Any,
    ) -> None:
        """Write the fixed header to a path, compressed for `.mgz` and `.gz`.

        The metadata is consolidated first, as in [`to_bytes`][], and the
        header is written with [`MghRaw.to_filename`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **fields : Any
            Fields to override, as in [`consolidate`][].
        """
        raw = self.consolidate(**fields).raw
        (MghRaw() if raw is None else raw).to_filename(filename)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------

# Vocabulary field, its footer slot, and the factor from the unit of the
# footer to the unit of the vocabulary; `None` for radians to degrees.
_FOOTER = {
    "repetition_time": ("tr", 1e-3),
    "echo_time": ("te", 1e-3),
    "inversion_time": ("ti", 1e-3),
    "flip_angle": ("flip_angle", None),
}

# The voxel types that MGH stores.
_MGH_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.int16, np.int32, np.float32)
)


def _decode(raw: tx.Optional[MghRaw]) -> tx.Dict[str, tx.Any]:
    """Return the fields of the vocabulary that an MGH record holds."""
    if raw is None:
        return {}
    out: tx.Dict[str, tx.Any] = {}
    for name, (slot, factor) in _FOOTER.items():
        stored = float(raw.header[slot])
        if not stored:
            continue
        # The shortest decimal that is stored as the same number: a flip
        # angle of `9.0`, not `9.000000250447817`.
        if factor is None:
            out[name] = shortest_decimal(math.degrees(stored), math.radians)
        else:
            out[name] = shortest_decimal(
                stored * factor, functools.partial(_divide, factor)
            )
    out["dtype"] = raw.header.get_data_dtype()
    out["history"] = decode_history(raw.tags)
    return out


def _encode(
    raw: MghRaw, changed: tx.Dict[str, tx.Any], report: ConversionReport
) -> MghRaw:
    """Write the changed fields into a record, which is already a copy.

    Each field that the record takes is taken out of `changed`. A
    history over tags that do not parse is left in `changed`, so that it
    is reported as lost.
    """
    for name, (slot, factor) in _FOOTER.items():
        if name not in changed:
            continue
        value = changed.pop(name)
        if value is None:
            stored = 0.0
        elif factor is None:
            stored = math.radians(value)
        else:
            stored = value / factor
        raw.header[slot] = stored
    if "dtype" in changed:
        dtype = changed.pop("dtype")
        if dtype in _MGH_DTYPES:
            raw.header.set_data_dtype(dtype)
        else:
            report.approximated["dtype"] = (
                f"MGH cannot store {getattr(dtype, 'name', dtype)} (it "
                f"stores uint8, int16, int32 and float32)"
            )
    if "history" in changed:
        tags = encode_history(raw.tags, changed["history"])
        if tags is not None:
            raw = replace(raw, tags=tags)
            changed.pop("history")
    return raw


def _divide(factor: float, value: float) -> float:
    """Return a value of the vocabulary in the unit of the footer."""
    return value / factor
