"""
The metadata of NIfTI files: [`NiftiMetadata`][], shared by `NiftiImage`
and every NIfTI-based transformation (plain fields and affines, FSL
FNIRT, ITK NIfTI fields, NiftyReg, SPM).

Its raw record (`raw`) is the `nibabel` header. What the vocabulary
covers:

| Field | Header slots |
|---|---|
| `description` | `descrip` (80 bytes) |
| `display_range` | `cal_min`, `cal_max` |
| `sources` | `aux_file` (one name, 24 bytes) |
| `slice_encoding_direction` | `dim_info` (a voxel axis, no polarity) |
| `phase_encoding_direction` | `dim_info` (a voxel axis, no polarity) |
| `slice_timing` | `slice_code`, `slice_start`, `slice_end`, `slice_duration` |
| `repetition_time` (derived) | `pixdim[4]` (the time step) |
| `intent` (from the axes) | `intent_code` |
| `space` (from the geometry) | `sform_code` / `qform_code` |
| `data_type` | `datatype` (the writer's, see below) |
| `scale_slope`, `scale_intercept` | `scl_slope`, `scl_inter` (the writer's) |

`repetition_time`, `intent` and `space` are views of geometry that the
writer takes from the data model; a value that disagrees with it is
reported, not written. `repetition_time` is the time step of the image
(the scale of its time axis, `time_step`), which the writer stores as
`pixdim[4]` (`_geometry` gives it); only an image whose data model has no
time step gets the field's value there. `intent` and `space` are checked
against the codes the writer set from the axes and the world space.
NIfTI has no free-form store, so `extra` is unsupported (open question 7
of the design memo).

An encoding direction is stored as the voxel axis it is along: its
polarity is dropped (approximated), and one along no voxel axis (an
oblique direction, or one in a world space) is lost.

The writer keeps, from the raw record of the file that was read, what is
safe to keep: `descrip`, `aux_file`, `cal_*`, `dim_info`, the `slice_*`
fields (when the slice axis kept its length), a non-structural intent
(images only) and the header extensions. Geometry and `xyzt_units`
always come from the data model. The image writer stores the data as
`data_type` when the values of the array are of its kind, and with the
intensity scaling `scale_slope` and `scale_intercept` when the values
fit it, so that a scaled integer file is written back as it was read
(see
[`preferred_storage`][brainhops.datamodel.metadata._dtype.preferred_storage]).
A `dtype=` writer option wins over both.
"""

__all__ = ["NiftiMetadata"]

# stdlib
from io import BytesIO

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr

# internals
from brainhops._core.numeric import float32_repr
from brainhops.datamodel.enums import AxisType
from brainhops.datamodel.images import Image
from brainhops.datamodel.metadata import (
    ConversionReport,
    EncodingDirection,
    FileBasedMetadata,
)
from brainhops.datamodel.metadata._terms import AXES
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base._metadata_parser import MetadataParser
from brainhops.io.base.parsers import Confidence, SnifferContentError

# NIfTI xform codes and their names; see `brainhops.io.base.nifti`.
_XCODES = {
    1: "scanner",
    2: "aligned",
    3: "talairach",
    4: "mni",
    5: "template",
}


# Intent codes that retype the axes of the data (see `nifti.py`); a
# writer never takes them from a record, only from the data model.
_STRUCTURAL_INTENTS = frozenset(
    {1004, 1006, 1008, 1009, 2001, 2002, 2003, 2004, 2005}
    | {2006, 2007, 2008, 2009}
)


# Seconds per NIfTI time unit.
_TIME_UNITS = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}


_DESCRIP_BYTES = 80


_AUX_FILE_BYTES = 24


@register_format
class NiftiMetadata(
    FileBasedMetadata,
    MetadataParser,
    on={"format": "nifti"},
    supports=(
        "description",
        "data_type",
        "scale_slope",
        "scale_intercept",
        "intent",
        "space",
        "display_range",
        "slice_timing",
        "slice_encoding_direction",
        "phase_encoding_direction",
        "sources",
        "repetition_time",
    ),
):
    """
    The metadata of a NIfTI file; its raw record is the `nibabel` header.

    `header` is the raw record under its familiar name.
    `NiftiMetadata.load(path)` reads the header of a NIfTI-1 or NIfTI-2
    file, gzipped or not, without its voxels.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS = ("nifti",)
    _READ_MODE = "rb"

    raw: tx.Annotated[
        tx.Optional[nb.Nifti1Header],
        tx.Doc(
            """
            The `nibabel` header of the file that was read. Edit it only
            for what the vocabulary does not cover; geometry, units and
            scaling are rewritten from the data model on save.
            """
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def header(self) -> tx.Optional[nb.Nifti1Header]:
        """The `nibabel` header (the raw record, `raw`)."""
        return self.raw

    # --- reading the header of a file ---------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the class is that an open file holds a NIfTI
        header.

        Parameters
        ----------
        file : file object
            A binary stream, possibly gzipped.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        from brainhops.io.base.nifti import is_nifti_stream

        if is_nifti_stream(file):
            return Confidence.CERTAIN
        if error:
            raise (SnifferContentError if error is True else error)(
                "Content is not a NIfTI file"
            )
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the class is that bytes hold a NIfTI header.

        Parameters
        ----------
        content : bytes
            The content of a file, possibly gzipped.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error)

    @classmethod
    def _read_raw(cls, file: tx.Any, **kwargs: tx.Any) -> nb.Nifti1Header:
        from brainhops.io.base.nifti import _load_nifti_header

        return _load_nifti_header(file)

    # --- hooks --------------------------------------------------------

    @classmethod
    def _default_raw(cls) -> nb.Nifti1Header:
        return nb.Nifti1Header()

    @classmethod
    def _decode(
        cls, raw: tx.Optional[nb.Nifti1Header], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        if raw is None:
            return {}
        h = raw
        out: tx.Dict[str, tx.Any] = {}
        out["description"] = _bytes_field(h, "descrip")
        aux = _bytes_field(h, "aux_file")
        out["sources"] = (aux,) if aux else None

        code = int(h["intent_code"])
        out["intent"] = _intent_label(code) if code else None

        scode, qcode = int(h["sform_code"]), int(h["qform_code"])
        out["space"] = _XCODES.get(scode) or _XCODES.get(qcode)

        cal = (float32_repr(h["cal_min"]), float32_repr(h["cal_max"]))
        out["display_range"] = cal if any(cal) else None

        try:
            out["data_type"] = h.get_data_dtype()
        except Exception:
            pass
        out["scale_slope"], out["scale_intercept"] = _scaling(h, image)

        freq, phase, slice_ = h.get_dim_info()
        if phase is not None and phase < len(AXES):
            out["phase_encoding_direction"] = AXES[phase]
        if slice_ is not None and slice_ < len(AXES):
            out["slice_encoding_direction"] = AXES[slice_]
            out["slice_timing"] = _decode_slice_timing(h)

        shape = _shape(h)
        scale = _time_scale(h)
        if len(shape) >= 4 and scale is not None:
            step = float32_repr(h["pixdim"][4])
            if step > 0:
                out["repetition_time"] = step * scale
        return out

    def _encode(
        self,
        raw: nb.Nifti1Header,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> nb.Nifti1Header:
        h = raw
        if "description" in changed:
            _encode_bytes(
                h,
                "descrip",
                changed["description"],
                _DESCRIP_BYTES,
                "description",
                report,
            )
        if "sources" in changed:
            sources = changed["sources"]
            first = sources[0] if sources else None
            if sources and len(sources) > 1:
                report.approximated["sources"] = (
                    f"only the first of {len(sources)} kept (aux_file)"
                )
            _encode_bytes(
                h, "aux_file", first, _AUX_FILE_BYTES, "sources", report
            )
        if "display_range" in changed:
            lo, hi = changed["display_range"] or (0.0, 0.0)
            h["cal_min"], h["cal_max"] = lo, hi
        for name, position in (
            ("phase_encoding_direction", 1),
            ("slice_encoding_direction", 2),
        ):
            if name in changed:
                _encode_dim_info(h, name, changed[name], position, report)
        if "slice_timing" in changed:
            _encode_slice_timing(h, changed["slice_timing"], report)
        if "repetition_time" in changed:
            # Only when the data model has no time step (see `_geometry`).
            _encode_repetition_time(h, changed["repetition_time"], report)
        if "intent" in changed:
            _encode_intent(h, changed["intent"], image, report)
        if "space" in changed:
            _check_space(h, changed["space"], report)
        # `scale_slope` and `scale_intercept` are not encoded here: the
        # image writer stores them with the data, when the values fit them
        # (see `preferred_storage`), and `nibabel` rewrites `scl_*` from
        # the data on save anyway.
        if "data_type" in changed and changed["data_type"] is not None:
            # The image writer settles it against the data afterwards
            # (see `preferred_dtype`).
            try:
                h.set_data_dtype(changed["data_type"])
            except Exception:
                report.lost["data_type"] = changed["data_type"]
        return h

    def _geometry(self, image: tx.Any) -> tx.Dict[str, tx.Any]:
        if not isinstance(image, Image):
            return {}
        return {"repetition_time": time_step(image.transformations)}

    def _check_raw(self, image: tx.Any) -> nb.Nifti1Header:
        # The writer's header has the shape of the data, whatever the
        # record says.
        h = self._raw_or_default()
        shape = _data_shape(image) if isinstance(image, Image) else None
        if shape and shape != _shape(h):
            try:
                h.set_data_shape(shape)
            except Exception:
                pass
        return h

    def _derive_raw(
        self,
        raw: tx.Optional[nb.Nifti1Header],
        *,
        changed: tx.Mapping[AxisType, tx.Any],
    ) -> tx.Optional[nb.Nifti1Header]:
        raw = super()._derive_raw(raw, changed=changed)
        if raw is not None and AxisType.space in changed:
            # The slice timing and the encoding axes follow the spatial
            # axes.
            _clear_slices(raw)
            raw.set_dim_info(None, None, None)
        return raw


# ----------------------------------------------------------------------
#   DECODING
# ----------------------------------------------------------------------


def _bytes_field(header: nb.Nifti1Header, name: str) -> tx.Optional[str]:
    value = np.asarray(header[name]).item()
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
    value = str(value).strip()
    return value or None


def _scaling(
    header: nb.Nifti1Header, image: tx.Any
) -> tx.Tuple[tx.Optional[float], tx.Optional[float]]:
    """
    The intensity scaling of a file, `(slope, intercept)`, or `None` for
    each part that is absent.

    A header read on its own holds `scl_slope` and `scl_inter`. The
    header of an image loaded by `nibabel` does not (`nibabel` moves the
    scaling into the array proxy and resets the header's to NaN), so the
    proxy of the image is read then. A slope of 0 or NaN means no
    scaling, as the NIfTI standard says, and so does the identity.
    """
    slope, intercept = float(header["scl_slope"]), float(header["scl_inter"])
    if not slope or not np.isfinite(slope):
        proxy = getattr(getattr(image, "image", None), "dataobj", None)
        slope = float(getattr(proxy, "slope", 0.0) or 0.0)
        intercept = float(getattr(proxy, "inter", 0.0) or 0.0)
    if not slope or not np.isfinite(slope):
        return None, None
    if not np.isfinite(intercept):
        intercept = 0.0
    if slope == 1.0 and intercept == 0.0:
        return None, None
    return float32_repr(slope), float32_repr(intercept) or None


def _time_scale(header: nb.Nifti1Header) -> tx.Optional[float]:
    """Seconds per unit of the header's time axis, `None` if unknown."""
    try:
        return _TIME_UNITS.get(header.get_xyzt_units()[1])
    except Exception:
        return None


def _intent_label(code: int) -> tx.Optional[str]:
    try:
        return str(nb.nifti1.intent_codes.label[code])
    except KeyError:
        return None


def _shape(header: nb.Nifti1Header) -> tx.Tuple[int, ...]:
    return tuple(int(d) for d in header.get_data_shape())


def _decode_slice_timing(
    h: nb.Nifti1Header,
) -> tx.Optional[tx.Tuple[float, ...]]:
    scale = _time_scale(h) or 1.0
    try:
        if not int(h["slice_code"]) or not float(h["slice_duration"]):
            return None
        times = h.get_slice_times()
    except Exception:
        return None
    if any(t is None for t in times):
        # Padding slices have no time: BIDS has no way to say so, and
        # the record keeps them.
        return None
    # Times are multiples of the (single-precision) slice duration:
    # recompute them in double precision from the duration as written.
    stored = float(h["slice_duration"])
    duration = float32_repr(stored) * scale
    return tuple(round(round(float(t) / stored) * duration, 9) for t in times)


# ----------------------------------------------------------------------
#   ENCODING
# ----------------------------------------------------------------------


def _encode_bytes(
    h: nb.Nifti1Header,
    slot: str,
    value: tx.Optional[str],
    size: int,
    name: str,
    report: ConversionReport,
) -> None:
    data = (value or "").encode("utf-8")
    if len(data) > size:
        report.approximated[name] = f"truncated to {size} bytes ({slot})"
        data = data[:size]
    h[slot] = data


def _clear_slices(h: nb.Nifti1Header) -> None:
    h["slice_code"] = 0
    h["slice_start"] = 0
    h["slice_end"] = 0
    h["slice_duration"] = 0


def _encode_dim_info(
    h: nb.Nifti1Header,
    name: str,
    value: tx.Optional[EncodingDirection],
    position: int,
    report: ConversionReport,
) -> None:
    dims = list(h.get_dim_info())
    if value is None:
        dims[position] = None
    else:
        bids = value.to_bids()
        if bids is None:
            # Along no voxel axis: `dim_info` stores an axis.
            report.lost[name] = value
            return
        if bids.endswith("-"):
            report.approximated[name] = "polarity dropped (dim_info)"
        dims[position] = AXES.index(bids[0])
    h.set_dim_info(*dims)


def _encode_slice_timing(
    h: nb.Nifti1Header,
    value: tx.Optional[tx.Sequence[float]],
    report: ConversionReport,
) -> None:
    if value is None:
        _clear_slices(h)
        return
    scale = _time_scale(h) or 1.0
    times = [float(t) / scale for t in value]
    slice_dim = h.get_dim_info()[2]
    try:
        if slice_dim is None:
            raise ValueError("no slice axis (dim_info)")
        h.set_slice_times(times)
    except Exception:
        # No slice axis, not one time per slice, or not one of the NIfTI
        # slice orders (sequential, alternating, their reverses). The
        # value replaced the record's, so the record's is cleared too.
        _clear_slices(h)
        report.lost["slice_timing"] = tuple(value)


def _encode_repetition_time(
    h: nb.Nifti1Header, value: tx.Optional[float], report: ConversionReport
) -> None:
    """Write a repetition time the data model says nothing about (an
    image with no physical time axis) as the time step, `pixdim[4]`."""
    if value is None:
        return
    if len(_shape(h)) < 4:
        report.lost["repetition_time"] = value
        return
    set_time_step(h, value)


def _intent_code(name: str) -> tx.Optional[int]:
    try:
        return int(nb.nifti1.intent_codes.code[name])
    except KeyError:
        return None


def _encode_intent(
    h: nb.Nifti1Header,
    value: tx.Optional[str],
    image: tx.Any,
    report: ConversionReport,
) -> None:
    if value is None:
        return
    current = int(h["intent_code"])
    if current:
        if _intent_label(current) != value:
            report.approximated["intent"] = (
                f"derived from the axes ({_intent_label(current)})"
            )
        return
    code = _intent_code(value)
    if code is None:
        report.lost["intent"] = value
    elif code in _STRUCTURAL_INTENTS:
        report.approximated["intent"] = (
            "derived from the axes (none): this intent retypes the axes"
        )
    else:
        h.set_intent(code)


def _check_space(
    h: nb.Nifti1Header, value: tx.Optional[str], report: ConversionReport
) -> None:
    if value is None:
        return
    scode, qcode = int(h["sform_code"]), int(h["qform_code"])
    written = _XCODES.get(scode) or _XCODES.get(qcode)
    if written != value:
        report.approximated["space"] = (
            f"derived from the world space name ({written})"
        )


def _data_shape(image: tx.Any) -> tx.Optional[tx.Tuple[int, ...]]:
    """The shape of the data of an image, without reading the data."""
    data = image.__dict__.get("_data")
    if data is None:
        nib = getattr(image, "image", None)
        if nib is not None and hasattr(nib, "shape"):
            return tuple(int(d) for d in nib.shape)
        try:
            data = image.data
        except Exception:
            return None
    shape = getattr(data, "shape", None)
    return None if shape is None else tuple(int(d) for d in shape)


# ----------------------------------------------------------------------
#   SHARED WITH THE IMAGE CLASS
# ----------------------------------------------------------------------


def time_step(
    transformations: tx.Optional[tx.Iterable[tx.Any]],
) -> tx.Optional[float]:
    """
    The time step of an image, in seconds, as its data model gives it.

    The time step is the scale of the time axis of the first scaling
    whose output has a time axis with a physical time unit. The NIfTI and
    MGH readers build such a scaling, from the voxel space to the
    physical space.

    Parameters
    ----------
    transformations : iterable of Transformation or None
        The transformations of the image.

    Returns
    -------
    float or None
        The time step in seconds, or `None` when no scaling gives one.
    """
    for xform in transformations or ():
        scale = getattr(xform, "scale", None)
        output = getattr(xform, "output", None)
        if scale is None or output is None:
            continue
        axes = [
            axis
            for axis in (getattr(output, "axes", None) or ())
            if axis is not Ellipsis
        ]
        for index, axis in enumerate(axes):
            if getattr(axis, "type", None) != "time":
                continue
            unit = getattr(axis, "unit", None)
            if not (is_physicalunit(unit) and is_timeunit(unit)):
                break
            scale = np.ravel(np.asarray(scale, dtype=float))
            if index < scale.size and scale[index] > 0:
                return float(scale[index]) * float(unit.scale)
            break
    return None


def set_time_step(h: nb.Nifti1Header, seconds: float) -> None:
    """
    Store a time step as `pixdim[4]`, in the time unit of the header.

    A header without a time unit is given seconds.

    Parameters
    ----------
    h : nibabel.Nifti1Header
        The header to edit, in place.
    seconds : float
        The time step, in seconds.
    """
    space, time = h.get_xyzt_units()
    if time not in _TIME_UNITS:
        h.set_xyzt_units(space, "sec")
        time = "sec"
    h["pixdim"][4] = float(seconds) / _TIME_UNITS[time]


def copy_record(
    target: nb.Nifti1Header,
    record: tx.Optional[nb.Nifti1Header],
    *,
    intent: bool = True,
) -> None:
    """
    Copy what is safe to keep from the record of the file that was read
    onto a header that the writer has just built.

    `descrip`, `aux_file`, `cal_min` and `cal_max` are always kept, and
    so are the header extensions, unless the writer added its own.
    `dim_info` is kept when its axes still exist, and the `slice_*` slots
    are kept when the slice axis kept its length. An intent that does not
    retype the axes is kept when the writer set none. The geometry,
    `xyzt_units`, the data type and the intensity scaling are never
    touched.

    Parameters
    ----------
    target : nibabel.Nifti1Header
        The header that the writer built, edited in place.
    record : nibabel.Nifti1Header or None
        The header of the file that was read. Nothing is copied when it
        is `None`.
    intent : bool, optional
        Whether to keep the intent of the record.
    """
    if record is None:
        return
    for slot in ("descrip", "aux_file", "cal_min", "cal_max"):
        try:
            target[slot] = record[slot]
        except (KeyError, ValueError):
            pass

    shape, old_shape = _shape(target), _shape(record)
    dims = record.get_dim_info()
    if all(d is None or d < len(shape) for d in dims):
        target.set_dim_info(*dims)
        slice_dim = dims[2]
        if (
            slice_dim is not None
            and slice_dim < len(old_shape)
            and shape[slice_dim] == old_shape[slice_dim]
        ):
            for slot in (
                "slice_code",
                "slice_start",
                "slice_end",
                "slice_duration",
            ):
                target[slot] = record[slot]

    code = int(record["intent_code"])
    if (
        intent
        and code
        and code not in _STRUCTURAL_INTENTS
        and not int(target["intent_code"])
    ):
        for slot in (
            "intent_code",
            "intent_name",
            "intent_p1",
            "intent_p2",
            "intent_p3",
        ):
            target[slot] = record[slot]

    # The extensions are kept unless the writer added its own (NiftyReg
    # writes structural ones, which the record holds too).
    extensions = getattr(record, "extensions", None)
    if extensions and not getattr(target, "extensions", True):
        for extension in extensions:
            target.extensions.append(extension)
