import math
import time
import uuid
from collections import OrderedDict

import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.common.afni import AfniHeader, AfniReaderWriter
from brainhops.io.common.afni._constants import AFNI_VIEWS
from brainhops.io.common.afni._data import brick_code, brick_dtype
from brainhops.io.common.afni._geometry import (
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_view,
    afni_voxel_to_dicom,
    afni_world,
)
from brainhops.io.images.base import ImageFormat

_INDEX = "index"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_PHYSICAL = "physical"
_CARDINAL = "-cardinal"

# Codes of `TAXIS_NUMS[2]` for the time units the writer may meet.
_TAXIS_CODES = {"millisecond": 77001, "second": 77002, "hertz": 77003}

# Attributes the writer computes itself, never copied from the source.
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

# Attributes that describe sub-bricks or the time axis, copied from the
# source header only if the number of sub-bricks is unchanged.
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

# Attributes that describe the grid, copied from the source header only
# if the shape and geometry are unchanged.
_PER_GRID = (
    "MARKS_XYZ",
    "MARKS_LAB",
    "MARKS_HELP",
    "MARKS_FLAGS",
    "TAGSET_NUM",
    "TAGSET_FLOATS",
    "TAGSET_LABELS",
)

# TYPESTRING values, indexed by the code in `SCENE_DATA[2]`.
_TYPESTRINGS = (
    "3DIM_HEAD_ANAT",
    "3DIM_HEAD_FUNC",
    "3DIM_GEN_ANAT",
    "3DIM_GEN_FUNC",
)
_ANAT_SPGR, _ANAT_EPI, _ANAT_BUCK = 0, 2, 11


def _afni_axes(header: AfniHeader) -> tx.List[Axis]:
    """Return the voxel axes of an AFNI dataset.

    Several sub-bricks add a fourth axis: `t` for a time series, and
    `brick`, with no meaning defined by AFNI, otherwise.
    """
    axes = [Axis(name, "space", unit=_INDEX) for name in _SPATIAL]
    if header.nvals > 1:
        if header.taxis is not None:
            axes.append(Axis("t", "time", unit=_INDEX))
        else:
            axes.append(Axis("brick", unit=_INDEX))
    return axes


@register_format
class AfniImage(AfniReaderWriter, ImageFormat, SingleScaleImage):
    """An image stored as an AFNI dataset (`.HEAD` and `.BRIK` files).

    The data are indexed `[x, y, z]`, or `[x, y, z, sub-brick]`, in
    Fortran order and scaled by `BRICK_FLOAT_FACS`. The transformations
    lead to `physical`, `<view>-cardinal` and `<view>`, the last of which
    is preferred. Header attributes without a place in the data model are
    kept in `header` and written back.

    !!! note "Why the bases are in this order"
        As for `NiftiImage`, [`SingleScaleImage`][] comes last so that its
        `data` field follows the defaulted fields of the parser, and so that
        the lazy properties of this class override plain fields.
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
        """Build an image, dropping the last axis of a single sub-brick."""
        if header.nvals == 1:
            bricks = bricks[..., 0]
        return cls(header=header, dataobj=bricks, **kwargs)

    @classmethod
    def _score_header(cls, header: AfniHeader) -> float:
        """Score a header as a plain image.

        A dataset with exactly three sub-bricks may be a warp (such as the
        `_WARP` output of 3dQwarp), so it is only a weak match.
        """
        if header.nvals == 3:
            return Confidence.WEAK
        return Confidence.LIKELY

    # --- data model ---------------------------------------------------

    @smartproperty
    def data(self) -> tx.Optional[tx.Any]:
        """Scaled image data, unless set explicitly."""
        data = self._scaled_data()
        if data is not None and data is not getattr(self, "dataobj", None):
            self._data = data
        return data

    @smartproperty
    def system(self) -> tx.Optional[CoordinateSystem]:
        """Voxel coordinate system, derived from the header unless set."""
        if self.header is None:
            return None
        axes = _afni_axes(self.header)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @smartproperty(unset="empty")
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the header.

        The transformations are decoded lazily unless they are set
        explicitly. An image built from data alone has none.
        """
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _afni_to_transformations(self.header)

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
        """Build a header that encodes this image.

        The preferred transformation is written as `IJK_TO_DICOM_REAL`, and
        its closest cardinal grid as `ORIENT_SPECIFIC`, `ORIGIN` and
        `DELTA`. The grid of the source header is kept if the geometry is
        unchanged.

        Parameters
        ----------
        filename : path, optional
            File being written, whose name may carry a view.
        view : {"orig", "acpc", "tlrc"}, optional
            Defaults to the view in the file name, then to the view named
            by the world space, then to the source view, then to `"orig"`.
        datatype : str or dtype, optional
            AFNI type name or numpy type, by default that of the data.
            Values are never rescaled.
        attributes : mapping, optional
            Extra attributes. A value of `None` removes an attribute.

        Returns
        -------
        AfniHeader
            The validated header.

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the data, the geometry or the view cannot be written.
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
    """Normalize an attribute value to a string or a flat tuple."""
    if isinstance(value, str):
        return value
    values = np.ravel(np.asarray(value)).tolist()
    return tuple(values)


def _brick_stats(data: tx.Any, nvals: int) -> tx.Tuple[float, ...]:
    """Return `BRICK_STATS`: the finite minimum and maximum of each brick."""
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
    """Return the TYPESTRING and the `SCENE_DATA` values after the view.

    Valid source values are kept. Otherwise the dataset is anatomical:
    SPGR for one volume, EPI for a time series and bucket otherwise.
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
    """Return the TR and its unit for a time series, or `None`.

    The source time axis is kept if the number of sub-bricks is unchanged.
    Otherwise the TR of a time axis is read from a [`Scaling`][], and
    defaults to one second.
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
    """Convert an AFNI header to transformations, in this order.

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
