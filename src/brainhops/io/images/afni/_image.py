# stdlib
import math
import time
import uuid
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.afni import (
    AFNI_VIEWS,
    AfniHeader,
    AfniParser,
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_view,
    afni_voxel_to_dicom,
    afni_world,
    brick_code,
    brick_dtype,
)
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.images.base import WritableFileBasedImage

_SAMPLE = "sample"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_PHYSICAL = "physical"
_CARDINAL = "-cardinal"

# `TAXIS_NUMS[2]` for the time units a writer may meet.
_TAXIS_CODES = {"millisecond": 77001, "second": 77002, "hertz": 77003}

# The attributes the writer computes, which are never copied from the
# header an image was read from.
_GENERATED = (
    "TYPESTRING",
    "IDCODE_STRING",
    "IDCODE_DATE",
    "SCENE_DATA",
    "ORIENT_SPECIFIC",
    "ORIGIN",
    "DELTA",
    "IJK_TO_DICOM",
    "IJK_TO_DICOM_REAL",
    "DATASET_RANK",
    "DATASET_DIMENSIONS",
    "BRICK_TYPES",
    "BRICK_STATS",
    "BRICK_FLOAT_FACS",
    "BYTEORDER_STRING",
)

# The attributes that describe each sub-brick (or the time axis), which
# are copied from the source header only when the number of sub-bricks is
# unchanged.
_PER_BRICK = (
    "BRICK_LABS",
    "BRICK_KEYWORDS",
    "BRICK_STATAUX",
    "BRICK_STATSYM",
    "STAT_AUX",
    "TAXIS_NUMS",
    "TAXIS_FLOATS",
    "TAXIS_OFFSETS",
)

# The attributes that describe the grid, which are copied from the source
# header only when the grid (shape and geometry) is unchanged.
_PER_GRID = (
    "MARKS_XYZ",
    "MARKS_LAB",
    "MARKS_HELP",
    "MARKS_FLAGS",
    "TAGSET_NUM",
    "TAGSET_FLOATS",
    "TAGSET_LABELS",
)

# `TYPESTRING` values, in the order of their `SCENE_DATA[2]` code.
_TYPESTRINGS = (
    "3DIM_HEAD_ANAT",
    "3DIM_HEAD_FUNC",
    "3DIM_GEN_ANAT",
    "3DIM_GEN_FUNC",
)
_ANAT_SPGR, _ANAT_EPI, _ANAT_BUCK = 0, 2, 11


def _afni_axes(header: AfniHeader) -> tx.List[Axis]:
    """
    The axes of the voxel space of an AFNI image.

    The first three are spatial (`x`, `y`, `z`). A dataset with several
    sub-bricks has a fourth axis: `t`, of type time, for a time series
    (one with a `TAXIS_NUMS` attribute); otherwise `brick`, which AFNI
    gives no meaning of its own (the statistics of a "bucket", ...).
    """
    axes = [Axis(name, "space", unit=_SAMPLE) for name in _SPATIAL]
    if header.nvals > 1:
        if header.taxis is not None:
            axes.append(Axis("t", "time", unit=_SAMPLE))
        else:
            axes.append(Axis("brick", unit=_SAMPLE))
    return axes


@register_format
class AfniImage(AfniParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by an AFNI dataset (`.HEAD` + `.BRIK`).

    The data are indexed `[x, y, z]`, or `[x, y, z, sub-brick]` for a
    dataset with several sub-bricks, in F order. The data of an
    uncompressed local BRIK stay memory-mapped until they are indexed.
    The scaling factors of the sub-bricks (`BRICK_FLOAT_FACS`) are applied
    on access; `dataobj` holds the stored values.

    The voxel-to-world transformations are, in order:

    1. `voxel` -> `physical`: a `Scaling` by the voxel sizes (`|DELTA|`,
       in mm), and the repetition time of a time series;
    2. `voxel` -> `<view>-cardinal` (`orig-cardinal`, `tlrc-cardinal`,
       ...): the `Affine` from `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA`,
       the grid AFNI programs compute on;
    3. `voxel` -> `<view>` (`orig`, `acpc` or `tlrc`): the `Affine` of
       `IJK_TO_DICOM_REAL`, the true (possibly oblique) geometry, which
       AFNI exports to NIfTI. Without that attribute, it is the cardinal
       one.

    The world spaces are LPS millimetres (AFNI's "DICOM order"). The
    last transformation is the preferred one. The attributes the data
    model has no slot for are kept in `header` and written back.

    !!! note "Why the bases are in this order"
        As for `NiftiImage`: `SingleScaleImage` comes last so that its
        `data` field follows the defaulted fields of the parser, and the
        lazy properties of this class take precedence over the plain
        fields.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".HEAD",
        ".BRIK",
        ".BRIK.gz",
        ".BRIK.bz2",
        ".head",
        ".brik",
        ".brik.gz",
        ".brik.bz2",
    )

    # --- reading ------------------------------------------------------

    @classmethod
    def _from_header(
        cls, header: AfniHeader, bricks: np.ndarray, **kwargs
    ) -> tx.Self:
        """A dataset with a single sub-brick is a 3D image."""
        if header.nvals == 1:
            bricks = bricks[..., 0]
        return cls(header=header, dataobj=bricks, **kwargs)

    @classmethod
    def _score_header(cls, header: AfniHeader) -> float:
        """
        Score an AFNI header as a plain image.

        Any AFNI dataset can be read as an image. One with three
        sub-bricks may be a warp (`3dQwarp`'s `_WARP` datasets), so it is
        only a weak match.
        """
        if header.nvals == 3:
            return Confidence.WEAK
        return Confidence.LIKELY

    # --- data model ---------------------------------------------------

    @property
    def data(self) -> tx.Optional[tx.Any]:
        """The image data, scaled, unless set explicitly."""
        if getattr(self, "_data", None) is not None:
            return self._data
        data = self._scaled_data()
        if data is not None and data is not getattr(self, "dataobj", None):
            self._data = data
        return data

    @data.setter
    def data(self, value: tx.Optional[tx.Any]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system, derived from the header, unless
        set explicitly. `None` when there is no header."""
        if getattr(self, "_system", None) is not None:
            return self._system
        if self.header is None:
            return None
        axes = _afni_axes(self.header)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The voxel-to-world transformations recorded by the header,
        decoded on access unless set explicitly.

        An image built from data alone has no header, so it records no
        transformation and the list is empty.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _afni_to_transformations(self.header)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    # --- writing ------------------------------------------------------

    def _afni_data(self) -> tx.Any:
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        ndim = len(np.shape(data))
        if ndim == 0 or ndim > 4:
            raise WriterError(
                f"AFNI stores three spatial axes and sub-bricks, so a "
                f"{ndim}D array cannot be written."
            )
        if ndim < 3:
            data = np.reshape(data, tuple(np.shape(data)) + (1,) * (3 - ndim))
        return data

    def _afni_header(
        self,
        filename: tx.Any = None,
        view: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        attributes: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        **kwargs,
    ) -> AfniHeader:
        """
        Build the header that encodes this image.

        The geometry comes from the preferred transformation, converted
        to voxel-to-DICOM: it is written as `IJK_TO_DICOM_REAL`, and its
        closest cardinal grid as `ORIENT_SPECIFIC`, `ORIGIN`, `DELTA` and
        `IJK_TO_DICOM`. When it is the geometry of the header the image
        was read from, that header's cardinal grid is kept as it was.

        Parameters
        ----------
        view : {"orig", "acpc", "tlrc"}, optional
            The view. Defaults to the one in the file name
            (`out+tlrc.HEAD`), else the one the preferred transformation's
            world space names (`"tlrc"`, `"talairach"`, `"mni"`, ...),
            else the view of the header the image was read from, else
            `"orig"`.
        datatype : str | dtype, optional
            The stored type, as AFNI names it (`"byte"`, `"short"`,
            `"int"`, `"float"`, `"double"`, `"complex"`) or as a numpy
            type. Defaults to the data's own type, or the closest one AFNI
            has. Values are rounded, but never rescaled, to fit an integer
            type.
        attributes : mapping, optional
            Extra attributes, merged into the computed ones and those
            copied from the source header; a value of `None` removes an
            attribute. A string is a string attribute; a sequence of
            numbers is a float attribute if any is a float (or if AFNI
            stores it as floats), else an integer one.
        """
        if kwargs:
            raise TypeError(
                f"Unknown AFNI writer option(s): {', '.join(kwargs)}"
            )
        data = self._afni_data()
        shape = tuple(int(d) for d in np.shape(data))
        nvals = shape[3] if len(shape) == 4 else 1
        source = self.header

        # --- geometry -------------------------------------------------
        matrix = afni_voxel_to_dicom(self.transformation)
        if not np.all(np.isfinite(matrix)) or not np.linalg.det(matrix):
            raise WriterError(
                "The voxel-to-world matrix is degenerate or not finite, so "
                "it cannot be written as AFNI geometry."
            )
        same_grid = (
            source is not None
            and source.shape == shape[:3]
            and np.allclose(source.voxel_to_dicom, matrix, atol=1e-5)
        )
        if same_grid:
            orient, origin, delta = source.orient, source.origin, source.delta
        else:
            orient, origin, delta = afni_geometry_from_matrix(matrix)
        cardinal = afni_cardinal_matrix(orient, origin, delta)

        # --- view and type --------------------------------------------
        output = getattr(self.transformation, "output", None)
        name = getattr(output, "name", None)
        chosen = (
            view
            or afni_view(filename)
            or afni_view(name)
            or (source.view if source is not None else None)
            or "orig"
        )
        if chosen not in AFNI_VIEWS:
            raise WriterError(
                f"Unknown AFNI view {chosen!r}: use one of {AFNI_VIEWS}."
            )
        if datatype is None:
            datatype = getattr(data, "dtype", np.dtype(np.float32))
        stored = brick_dtype(datatype)
        taxis = _time_axis(self, source, nvals)

        # --- attributes -----------------------------------------------
        attrs = OrderedDict()
        typestring, scene = _typestring_and_scene(source, nvals, taxis)
        attrs["TYPESTRING"] = typestring
        attrs["IDCODE_STRING"] = "AFN_" + uuid.uuid4().hex[:22]
        attrs["IDCODE_DATE"] = time.ctime()
        attrs["SCENE_DATA"] = (AFNI_VIEWS.index(chosen),) + scene
        attrs["ORIENT_SPECIFIC"] = tuple(int(o) for o in orient)
        attrs["ORIGIN"] = tuple(float(o) for o in origin)
        attrs["DELTA"] = tuple(float(d) for d in delta)
        attrs["IJK_TO_DICOM"] = tuple(float(v) for v in cardinal[:3].ravel())
        attrs["IJK_TO_DICOM_REAL"] = tuple(
            float(v) for v in matrix[:3].ravel()
        )
        attrs["DATASET_RANK"] = (3, nvals, 0, 0, 0, 0, 0, 0)
        attrs["DATASET_DIMENSIONS"] = tuple(shape[:3]) + (0, 0)
        attrs["BRICK_TYPES"] = (brick_code(stored),) * nvals
        attrs["BRICK_STATS"] = _brick_stats(data, nvals)
        attrs["BRICK_FLOAT_FACS"] = (0.0,) * nvals
        attrs["BYTEORDER_STRING"] = "LSB_FIRST"

        if source is not None:
            for key, value in source.attributes.items():
                if key in attrs or key in _GENERATED:
                    continue
                if key in _PER_BRICK and source.nvals != nvals:
                    continue
                if key in _PER_GRID and not same_grid:
                    continue
                if key == "TEMPLATE_SPACE" and source.view != chosen:
                    continue
                attrs[key] = value
        if taxis is not None and "TAXIS_NUMS" not in attrs:
            tr, unit = taxis
            attrs["TAXIS_NUMS"] = (nvals, 0, _TAXIS_CODES.get(unit, 77002))
            attrs["TAXIS_FLOATS"] = (0.0, float(tr), 0.0, 0.0, 0.0)

        header = AfniHeader(attributes=attrs)
        if attributes:
            header = header.replace(**dict(attributes))
        header = AfniHeader(
            attributes=OrderedDict(
                (key, _as_value(value))
                for key, value in header.attributes.items()
            )
        )
        return header.validate()


def _as_value(value: tx.Any) -> tx.Any:
    """An attribute value as the header holds it."""
    if isinstance(value, str):
        return value
    values = np.ravel(np.asarray(value)).tolist()
    return tuple(values)


def _brick_stats(data: tx.Any, nvals: int) -> tx.Tuple[float, ...]:
    """The minimum and maximum of each sub-brick (`BRICK_STATS`), over
    its finite values."""
    stats = []
    for i in range(nvals):
        brick = data[..., i] if np.ndim(data) == 4 else data
        brick = np.real(np.asarray(brick))
        if brick.dtype.kind == "f":
            brick = brick[np.isfinite(brick)]
        if brick.size:
            stats += [float(np.min(brick)), float(np.max(brick))]
        else:
            stats += [0.0, 0.0]
    return tuple(stats)


def _typestring_and_scene(
    source: tx.Optional[AfniHeader],
    nvals: int,
    taxis: tx.Optional[tx.Tuple[float, str]],
) -> tx.Tuple[str, tx.Tuple[int, ...]]:
    """
    The `TYPESTRING`, and `SCENE_DATA` after its view code.

    The source header's are kept when it has them. Otherwise the dataset
    is an anatomical one (`3DIM_HEAD_ANAT`), of type SPGR for a single
    volume, EPI for a time series and bucket for other sub-bricks.
    """
    if source is not None:
        typestring = source.get("TYPESTRING")
        scene = source.get("SCENE_DATA")
        if (
            isinstance(typestring, str)
            and typestring in _TYPESTRINGS
            and isinstance(scene, tuple)
            and len(scene) >= 3
        ):
            return typestring, tuple(int(s) for s in scene[1:])
    if nvals == 1:
        func = _ANAT_SPGR
    elif taxis is not None:
        func = _ANAT_EPI
    else:
        func = _ANAT_BUCK
    return _TYPESTRINGS[0], (func, 0, -999, -999, -999, -999, -999)


def _time_axis(
    image: AfniImage, source: tx.Optional[AfniHeader], nvals: int
) -> tx.Optional[tx.Tuple[float, str]]:
    """
    The `(TR, unit)` to write for a time series, or `None`.

    The source header's time axis is kept when the number of sub-bricks
    is unchanged. Otherwise, an image whose fourth axis is of type time
    is a time series: its repetition time is the scale of that axis in a
    `Scaling` among the transformations (such as the `physical` one),
    in that axis's unit; it is 1 second when no transformation gives it.
    """
    if source is not None and source.nvals == nvals and source.taxis:
        return source.taxis
    if nvals < 2:
        return None
    system = getattr(image, "system", None)
    for xform in image.transformations or []:
        if system is not None:
            break
        system = getattr(xform, "input", None)
    axes = list(getattr(system, "axes", None) or [])
    if len(axes) < 4 or getattr(axes[3], "type", None) != "time":
        return None
    for xform in reversed(list(image.transformations or [])):
        if not isinstance(xform, Scaling):
            continue
        scale = np.ravel(np.asarray(getattr(xform, "scale", []), float))
        if scale.size != 4 or not math.isfinite(scale[3]):
            continue
        out_axes = list(getattr(xform.output, "axes", None) or []) + [None]
        unit = getattr(out_axes[3], "unit", None)
        unit = None if unit is None else str(unit)
        if unit in _TAXIS_CODES:
            return float(scale[3]), unit
    return 1.0, "second"


def _afni_to_transformations(header: AfniHeader) -> tx.List[Transformation]:
    """
    Convert an AFNI header to a list of transformations.

    1. voxel -> "physical": a `Scaling` by `|DELTA|` (and the TR of a
       time series);
    2. voxel -> "<view>-cardinal": the cardinal grid, in LPS mm;
    3. voxel -> "<view>": `IJK_TO_DICOM_REAL` (or the cardinal grid), in
       LPS mm.
    """
    voxel_axes = _afni_axes(header)
    voxel_space = CoordinateSystem(name="voxel", axes=voxel_axes, order="F")

    taxis = header.taxis
    scale = [abs(d) for d in header.delta]
    phys_axes = [replace(axis, unit=_MM) for axis in voxel_axes[:3]]
    if len(voxel_axes) > 3:
        axis = voxel_axes[3]
        if axis.type == "time" and taxis and taxis[0] > 0 and taxis[1]:
            phys_axes.append(replace(axis, unit=taxis[1]))
            scale.append(taxis[0])
        else:
            phys_axes.append(replace(axis, unit=None))
            scale.append(1.0)
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=scale)

    def _affine(matrix: np.ndarray, name: str) -> Affine:
        return Affine(
            input=voxel_space,
            output=afni_world(name),
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    view = header.view
    return [
        vox2phys,
        _affine(header.cardinal_matrix, view + _CARDINAL),
        _affine(header.voxel_to_dicom, view),
    ]
