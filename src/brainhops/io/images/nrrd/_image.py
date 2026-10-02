# stdlib
import math
import re

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.datamodel.units import Unit
from brainhops.io.base._base import register_format
from brainhops.io.base._geometry import RAS_FROM_ORIENTATION, reduce_to_affine
from brainhops.io.base.nrrd import (
    _ENCODINGS,
    _SPACE_NAMES,
    SPACES,
    NrrdHeader,
    NrrdParser,
    _format_float,
    _format_strings,
    _format_vectors,
    dtype_to_nrrd,
)
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.images.base import WritableFileBasedImage

_SAMPLE = "sample"

_CHANNEL_KINDS = frozenset(
    (
        "vector",
        "list",
        "point",
        "covariant-vector",
        "normal",
        "2-vector",
        "3-vector",
        "4-vector",
        "3-gradient",
        "3-normal",
        "rgb-color",
        "hsv-color",
        "xyz-color",
        "rgba-color",
        "complex",
        "quaternion",
        "2d-symmetric-matrix",
        "2d-masked-symmetric-matrix",
        "2d-matrix",
        "2d-masked-matrix",
        "3d-symmetric-matrix",
        "3d-masked-symmetric-matrix",
        "3d-matrix",
        "3d-masked-matrix",
    )
)
"""The kinds of an axis whose samples are the components of one value
(a vector, a colour, a tensor, a list of measurements), read as a channel
axis."""

_ROLES = ("space", "time", "channel", "other")

_ANATOMICAL = {
    "right-anterior-superior": ("R", "A", "S"),
    "left-anterior-superior": ("L", "A", "S"),
    "left-posterior-superior": ("L", "P", "S"),
}

_LETTERS = {
    "R": "left-to-right",
    "L": "right-to-left",
    "A": "posterior-to-anterior",
    "P": "anterior-to-posterior",
    "S": "inferior-to-superior",
    "I": "superior-to-inferior",
}

_FLIPS = {
    "right-anterior-superior": np.array([1.0, 1.0, 1.0]),
    "left-anterior-superior": np.array([-1.0, 1.0, 1.0]),
    "left-posterior-superior": np.array([-1.0, -1.0, 1.0]),
}
"""The sign of each axis of an anatomical space, relative to RAS."""

_PER_AXIS = (
    "spacings",
    "thicknesses",
    "axis mins",
    "axis maxs",
    "centers",
    "labels",
    "units",
)

_KEPT = ("content", "sample units", "old min", "old max")
"""Fields of the source header that are written back as they are."""


# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------


def _space_name(space: str) -> str:
    """The name of the world space: the abbreviation of `space`."""
    return SPACES[space][0]


def _roles(header: NrrdHeader) -> tx.List[str]:
    """
    The role of each axis of a NRRD file, fastest first: `"space"`,
    `"time"`, `"channel"` or `"other"`.

    With `space directions`, an axis is spatial when it has a direction.
    Without them, an axis is spatial when its kind is `domain` or
    `space`, or, when the header has no `kinds` at all, when it is one of
    the first three. Otherwise, a `time` axis is temporal, an axis whose
    kind is a vector, a list, a colour, a tensor, ... is a channel axis,
    and any other (`scalar`, `stub`, `none`, ...) is an untyped one.
    """
    dirs = header.space_directions
    kinds = header.kinds
    has_kinds = "kinds" in header.fields
    roles = []
    for i, kind in enumerate(kinds):
        if dirs is not None and dirs[i] is not None:
            roles.append("space")
        elif kind == "time":
            roles.append("time")
        elif kind in _CHANNEL_KINDS:
            roles.append("channel")
        elif dirs is None and (
            kind in ("domain", "space") or (not has_kinds and i < 3)
        ):
            roles.append("space")
        else:
            roles.append("other")
    return roles


def _layout(header: NrrdHeader) -> tx.Tuple[tx.List[int], tx.List[Axis]]:
    """
    The order of the axes of the image, and the axes themselves.

    The image lists the spatial axes first, then the temporal, channel
    and other ones, each group in the order of the file. Element `j` of
    the permutation is the file axis that becomes image axis `j`.
    """
    roles = _roles(header)
    perm = [i for role in _ROLES for i, r in enumerate(roles) if r == role]
    return perm, _axes([roles[i] for i in perm], perm)


def _axes(roles: tx.Sequence[str], perm: tx.Sequence[int]) -> tx.List[Axis]:
    """The axes of the index space, named `x, y, z`, `t`, `c` or `dim<i>`
    (`i` the axis' position in the file)."""
    axes = []
    nspace = 0
    seen = set()
    for role, i in zip(roles, perm):
        if role == "space":
            name = "xyz"[nspace] if nspace < 3 else f"dim{i}"
            nspace += 1
            axes.append(Axis(name, "space", unit=_SAMPLE))
        elif role in ("time", "channel"):
            name = role[0] if role not in seen else f"dim{i}"
            seen.add(role)
            axes.append(Axis(name, role, unit=_SAMPLE))
        else:
            axes.append(Axis(f"dim{i}", unit=_SAMPLE))
    return axes


def _role(axis: tx.Any) -> str:
    """The role of an axis of the datamodel."""
    type_ = getattr(axis, "type", None)
    if type_ in ("space", "time"):
        return type_
    if type_ in ("channel", "displacement", "coordinate"):
        return "channel"
    return "other"


def _with_unit(axis: Axis, unit: tx.Optional[str]) -> Axis:
    """`axis` measured in `unit`, or with no unit when `unit` is unknown
    to the datamodel or does not fit the axis."""
    if unit:
        try:
            if type(Unit(unit)) is not Unit:
                return replace(axis, unit=unit)
        except Exception:  # noqa: BLE001
            pass
    return replace(axis, unit=None)


def _unit_symbol(axis: tx.Any) -> str:
    """The NRRD spelling of an axis' unit (`""` when it has none)."""
    unit = getattr(axis, "unit", None)
    if unit is None or getattr(unit, "name", None) in (None, _SAMPLE):
        return ""
    try:
        return unit.symbol
    except Exception:  # noqa: BLE001
        return ""


# ----------------------------------------------------------------------
#   GEOMETRY (reading)
# ----------------------------------------------------------------------


def _world(header: NrrdHeader, sdim: int) -> CoordinateSystem:
    """
    The world space of the `space` (or `space dimension`) of a header.

    An anatomical space (`right-anterior-superior`, ...) has three axes
    `x, y, z` that carry its orientation; `scanner-xyz` and
    `3D-*-handed` have three that carry none; a `*-time` space has a
    fourth, time, axis `t`. A world given by its `space dimension` only
    has `sdim` axes with no orientation, and is named `"world"`.

    The units are the `space units`, or millimetres for the spatial axes
    of every `space` but the `3D-*-handed` ones, as ITK and 3D Slicer
    assume. A world given by its dimension only has no unit by default.
    """
    space = header.space
    units = list(header.space_units) + [None] * sdim
    letters = _ANATOMICAL.get((space or "").replace("-time", ""))
    default = None
    if space is not None and not space.startswith("3d"):
        default = "mm"
    axes = []
    for i in range(sdim):
        if space is not None and space.endswith("-time") and i == 3:
            axis = Axis("t", "time")
            axes.append(_with_unit(axis, units[i]))
            continue
        name = "xyz"[i] if i < 3 else f"dim{i}"
        orientation = None
        if letters is not None and i < 3:
            orientation = Orientation(
                type="anatomical", value=_LETTERS[letters[i]]
            )
        axis = Axis(name, "space", orientation=orientation)
        axes.append(_with_unit(axis, units[i] or default))
    name = _space_name(space) if space is not None else "world"
    return CoordinateSystem(name=name, axes=axes)


def _nrrd_to_transformations(header: NrrdHeader) -> tx.List[Transformation]:
    """
    Convert a NRRD header to a list of transformations.

    With a world space (`space` or `space dimension`, and `space
    directions`):

    1. index -> `"physical"`: a `Scaling` by the length of each space
       direction (the `spacings` of the other axes, else 1);
    2. index -> world: the `Affine` whose columns are the space directions
       and whose translation is the `space origin`, the position of the
       centre of the first sample.

    Without one, a single index -> `"physical"` map built from the
    `spacings`, `axis mins` and `axis maxs`, and the `centers`, of each
    axis: it is a `Scaling`, or an `Affine` when the first sample is not
    at zero.
    """
    perm, axes = _layout(header)
    voxel = CoordinateSystem(name="voxel", axes=axes, order="F")
    roles = [_role(a) for a in axes]
    nspace = roles.count("space")
    dirs = header.space_directions
    sdim = header.space_dimension
    spacings = header.spacings
    units = header.units

    if dirs is not None and sdim and nspace:
        world = _world(header, sdim)
        world_unit = getattr(world.axes[0], "unit", None)
        scale, phys_axes = [], []
        for axis, role, i in zip(axes, roles, perm):
            if role == "space":
                scale.append(float(np.linalg.norm(dirs[i])) or 1.0)
                phys_axes.append(replace(axis, unit=world_unit))
            else:
                s = spacings[i]
                scale.append(s if math.isfinite(s) and s != 0 else 1.0)
                phys_axes.append(_with_unit(axis, units[i]))
        physical = CoordinateSystem(name="physical", axes=phys_axes)
        matrix = np.zeros((sdim, nspace + 1))
        for j in range(nspace):
            matrix[:, j] = dirs[perm[j]]
        origin = header.space_origin
        if origin is not None and origin.size == sdim:
            matrix[:, -1] = np.nan_to_num(origin)
        return [
            Scaling(input=voxel, output=physical, scale=scale),
            Affine(input=voxel, output=world, matrix=matrix),
        ]

    sizes = header.sizes
    centers = header.centers
    mins, maxs = header.axis_mins, header.axis_maxs
    scale, offset, phys_axes = [], [], []
    for axis, i in zip(axes, perm):
        cell = (centers[i] or "cell") == "cell"
        n = sizes[i]
        s = spacings[i]
        lo, hi = mins[i], maxs[i]
        if not math.isfinite(s) and math.isfinite(lo) and math.isfinite(hi):
            steps = n if cell else n - 1
            s = (hi - lo) / steps if steps > 0 else math.nan
        if not math.isfinite(s) or s == 0:
            s = 1.0
        half = s / 2 if cell else 0.0
        if math.isfinite(lo):
            first = lo + half
        elif math.isfinite(hi):
            first = hi - half - s * (n - 1)
        else:
            first = 0.0
        scale.append(s)
        offset.append(first)
        phys_axes.append(_with_unit(axis, units[i]))
    physical = CoordinateSystem(name="physical", axes=phys_axes)
    if not any(offset):
        return [Scaling(input=voxel, output=physical, scale=scale)]
    n = len(axes)
    matrix = np.zeros((n, n + 1))
    matrix[:, :-1] = np.diag(scale)
    matrix[:, -1] = offset
    return [Affine(input=voxel, output=physical, matrix=matrix)]


# ----------------------------------------------------------------------
#   IMAGE
# ----------------------------------------------------------------------


class NrrdImage(NrrdParser, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a NRRD file, attached (`.nrrd`) or
    detached (`.nhdr` and its data files).

    It is the shared base of [`AttachedNrrdImage`][] and
    [`DetachedNrrdImage`][], which answer to its hint `"nrrd"`; it is not
    registered itself, so that it does not compete with them. Either one
    reads both kinds of file, and writes the kind the file name asks for.

    The data are indexed `[x, y, z, t, c, ...]`, F order: the spatial axes
    first, then the time, channel and other axes, each group in the order
    of the file (whose first axis is the fastest). This is a view of the
    stored values (`dataobj`, in the file's axis order), so the values of
    a `raw` local file stay memory-mapped.

    The transformations are an index -> `"physical"` `Scaling`, then,
    when the header has a world space, the index -> world `Affine` built
    from `space directions` and `space origin` (preferred). The world is
    named after the `space` (`"RAS"`, `"LPS"`, `"LAS"`, `"scanner-xyz"`,
    ...; `"world"` for a bare `space dimension`). Header fields and
    key/value pairs that the data model has no slot for are kept in
    `header` and written back.

    !!! note "Why the bases are in this order"
        As for `NiftiImage`: `SingleScaleImage` comes last so that its
        `data` field follows the defaulted fields of the parser, and the
        lazy properties of this class take precedence over the plain
        fields.
    """

    # --- data model ---------------------------------------------------

    @property
    def data(self) -> tx.Optional[tx.Any]:
        """The image data, `[x, y, z, ...]`, unless set explicitly."""
        if getattr(self, "_data", None) is not None:
            return self._data
        raw = getattr(self, "dataobj", None)
        if raw is None or self.header is None:
            return raw
        perm, _ = _layout(self.header)
        return raw.transpose(perm)

    @data.setter
    def data(self, value: tx.Optional[tx.Any]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The index coordinate system, derived from the header, unless
        set explicitly. `None` when there is no header."""
        if getattr(self, "_system", None) is not None:
            return self._system
        if self.header is None:
            return None
        _, axes = _layout(self.header)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The index-to-world transformations recorded by the header,
        decoded on access unless set explicitly."""
        if getattr(self, "_transformations", None):
            return self._transformations
        if self.header is None:
            return list(getattr(self, "_transformations", None) or [])
        return _nrrd_to_transformations(self.header)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value

    # --- writing ------------------------------------------------------

    def _storage(
        self,
    ) -> tx.Tuple[tx.List[int], tx.List[str], tx.List[str]]:
        """
        How the image axes are stored: the file axis of each image axis,
        the role of each image axis, and the kind of each file axis.

        An image read from NRRD is written in the order, and with the
        kinds, of its file, as long as its shape has not changed. Any
        other is written in its own order, with the kinds `domain`
        (space), `time`, `vector` (channel) and `none`.
        """
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        shape = tuple(int(n) for n in np.shape(data))
        ndim = len(shape)
        system = self.system
        if system is None and self.transformations:
            system = getattr(self.transformation, "input", None)
        try:
            axes = list(getattr(system, "axes", None) or [])
        except TypeError:
            axes = []
        if len(axes) == ndim and not any(a is ... for a in axes):
            roles = [_role(a) for a in axes]
        else:
            roles = ["space"] * min(3, ndim) + ["other"] * (ndim - 3)
        source = self.header
        if source is not None and source.dimension == ndim:
            perm, _ = _layout(source)
            if tuple(source.sizes[i] for i in perm) == shape:
                kinds = source.fields.get("kinds", "").split()
                if len(kinds) != ndim:
                    kinds = _kinds(roles, perm)
                return perm, roles, kinds
        perm = list(range(ndim))
        return perm, roles, _kinds(roles, perm)

    def _nrrd_data(self, header: NrrdHeader) -> tx.Any:
        perm, _, _ = self._storage()
        return np.asarray(self.data).transpose(np.argsort(perm))

    def _nrrd_header(
        self,
        encoding: tx.Optional[str] = None,
        endian: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        space: tx.Optional[str] = None,
        keyvalue: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
        **kwargs,
    ) -> NrrdHeader:
        """
        Build the header that encodes this image.

        Parameters
        ----------
        encoding : {"raw", "gzip", "bzip2", "ascii", "hex"}, optional
            Defaults to the encoding of the source header, else `gzip`.
        endian : {"little", "big"}, optional
            Defaults to the byte order of the source header, else
            `little`.
        datatype : str | dtype, optional
            The NRRD type (`"short"`, `"float"`, ...) or a numpy one.
            Defaults to the data's own type.
        space : str, optional
            The anatomical space to write the geometry in
            (`"right-anterior-superior"`, `"LPS"`, ...). Defaults to the
            source header's, else the one the preferred transformation
            maps to (when it is RAS, LAS or LPS), else RAS. Only used when
            the world space is anatomical.
        keyvalue : mapping, optional
            Extra `key:=value` pairs, merged into those of the source
            header; a value of `None` removes a key.
        """
        if kwargs:
            raise TypeError(
                f"Unknown NRRD writer option(s): {', '.join(kwargs)}"
            )
        perm, roles, kinds = self._storage()
        data = self.data
        shape = tuple(int(n) for n in np.shape(data))
        ndim = len(shape)
        if ndim == 0:
            raise WriterError("NRRD cannot store a zero-dimensional array.")
        source = self.header
        fields: tx.Dict[str, str] = {}

        # --- storage --------------------------------------------------
        sizes = [0] * ndim
        for j, i in enumerate(perm):
            sizes[i] = shape[j]
        if datatype is None:
            datatype = getattr(data, "dtype", np.float32)
        fields["type"] = dtype_to_nrrd(datatype)
        fields["dimension"] = str(ndim)
        fields["sizes"] = " ".join(str(n) for n in sizes)
        fields["kinds"] = " ".join(kinds)
        if encoding is None:
            encoding = source.encoding if source is not None else "gzip"
        if str(encoding).lower() not in _ENCODINGS:
            raise WriterError(f"Unknown NRRD encoding: {encoding!r}")
        fields["encoding"] = _ENCODINGS[str(encoding).lower()]
        if endian is None:
            endian = (source.endian if source is not None else None) or (
                "little"
            )
        if endian not in ("little", "big"):
            raise WriterError(f"Unknown NRRD endian: {endian!r}")
        fields["endian"] = endian

        # --- per-axis fields of the source ----------------------------
        matched = (
            source is not None
            and len(kinds) == ndim
            and (
                source.dimension == ndim
                and tuple(source.sizes) == tuple(sizes)
            )
        )
        per_axis: tx.Dict[str, tx.List[str]] = {}
        if matched:
            for name in _PER_AXIS:
                value = source.fields.get(name)
                if value is None:
                    continue
                if name in ("labels", "units"):
                    values = ['"' + v + '"' for v in _split_quoted(value)]
                else:
                    values = value.split()
                if len(values) == ndim:
                    per_axis[name] = values
            for name in _KEPT:
                if name in source.fields:
                    fields[name] = source.fields[name]

        # --- geometry -------------------------------------------------
        spatial = [perm[j] for j, r in enumerate(roles) if r == "space"]
        geometry = _geometry(self, roles, space, source)
        if geometry is not None:
            kind, values = geometry
            if kind == "space":
                fields.update(values["fields"])
                directions = [None] * ndim
                for i, v in zip(spatial, values["directions"]):
                    directions[i] = v
                fields["space directions"] = _format_vectors(directions)
                # Spatial axes are described by their direction alone.
                for name in ("spacings", "axis mins", "axis maxs", "units"):
                    if name in per_axis:
                        empty = '""' if name == "units" else "nan"
                        for i in spatial:
                            per_axis[name][i] = empty
            else:
                spacings = per_axis.setdefault("spacings", ["nan"] * ndim)
                mins = per_axis.get("axis mins", ["nan"] * ndim)
                centers = per_axis.get("centers", ["???"] * ndim)
                units = per_axis.get("units", ['""'] * ndim)
                for i, s, t, u in zip(spatial, *values):
                    spacings[i] = _format_float(s)
                    cell = centers[i].lower() != "node"
                    mins[i] = _format_float(t - s / 2 if cell else t)
                    units[i] = _format_strings([u])
                if any(m != "nan" for m in mins):
                    per_axis["axis mins"] = mins
                    per_axis.pop("axis maxs", None)
                if any(u != '""' for u in units):
                    per_axis["units"] = units
        for name, values in per_axis.items():
            fields[name] = " ".join(values)
        if "space directions" not in fields:
            fields.pop("measurement frame", None)

        # --- key/value pairs ------------------------------------------
        merged = dict(source.keyvalue) if source is not None else {}
        for key, value in (keyvalue or {}).items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = str(value)

        version = max(4, source.version if source is not None else 4)
        return NrrdHeader(version=version, fields=fields, keyvalue=merged)


def _split_quoted(value: str) -> tx.List[str]:
    """The quoted strings of a field, still escaped."""
    return re.findall(r'"((?:[^"\\]|\\.)*)"', value)


def _kinds(roles: tx.Sequence[str], perm: tx.Sequence[int]) -> tx.List[str]:
    """The kind of each file axis, from the role of each image axis."""
    names = {
        "space": "domain",
        "time": "time",
        "channel": "vector",
        "other": "none",
    }
    kinds = ["none"] * len(perm)
    for j, i in enumerate(perm):
        kinds[i] = names[roles[j]]
    return kinds


def _orientation_matrix(axes: tx.Sequence[tx.Any]) -> tx.Optional[np.ndarray]:
    """The `(3, 3)` matrix that maps coordinates along `axes` to RAS, or
    `None` when they are not three anatomically oriented axes."""
    if len(axes) != 3:
        return None
    matrix = np.zeros((3, 3))
    for column, axis in enumerate(axes):
        value = getattr(getattr(axis, "orientation", None), "value", None)
        if value not in RAS_FROM_ORIENTATION:
            return None
        row, sign = RAS_FROM_ORIENTATION[value]
        matrix[row, column] = sign
    if abs(np.linalg.det(matrix)) != 1:
        return None
    return matrix


def _target_space(
    requested: tx.Optional[str],
    source: tx.Optional[NrrdHeader],
    to_ras: np.ndarray,
) -> str:
    """The anatomical `space` to write the geometry in."""
    if requested is not None:
        key = str(requested).lower()
        key = _SPACE_NAMES.get(key, key)
        if key not in _FLIPS:
            raise WriterError(
                f"NRRD writes an anatomical geometry in "
                f"{', '.join(_FLIPS)} (or RAS, LAS, LPS), not {requested!r}."
            )
        return key
    if source is not None:
        try:
            space = source.space
        except Exception:  # noqa: BLE001
            space = None
        if space in _FLIPS:
            return space
    for name, flips in _FLIPS.items():
        if np.array_equal(to_ras, np.diag(flips)):
            return name
    return "right-anterior-superior"


def _geometry(
    image: NrrdImage,
    roles: tx.Sequence[str],
    space: tx.Optional[str],
    source: tx.Optional[NrrdHeader],
) -> tx.Optional[tx.Tuple[str, tx.Any]]:
    """
    The geometry fields of the preferred transformation, in the order of
    the image's spatial axes.

    Returns `None` when there is nothing to write (no transformation, no
    spatial axis, or an identity scaling with no unit), `("space", ...)`
    for `space`, `space directions` and `space origin`, or
    `("spacings", (spacings, positions, units))` for a scaling (plus
    translation) onto unoriented axes, when the source header had no
    space either.
    """
    nspace = list(roles).count("space")
    if not image.transformations or not nspace:
        return None
    affine = reduce_to_affine(image.transformation, "NRRD", "world")
    matrix = affine.homogeneous_matrix
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    n_out, n_in = matrix.shape[0] - 1, matrix.shape[1] - 1
    output = getattr(affine, "output", None)
    out_axes = list(getattr(output, "axes", None) or [])
    out_axes = out_axes[:n_out]
    if len(out_axes) != n_out or any(a is ... for a in out_axes):
        out_axes = [None] * n_out
    cols = [j for j in range(n_in) if j < len(roles) and roles[j] == "space"]
    if n_in < len(roles):
        cols = list(range(min(n_in, nspace)))
    rows = list(range(n_out))
    if n_out == n_in and n_in == len(roles):
        rows = cols
    cols = cols[:nspace]
    if len(cols) < nspace:
        raise WriterError(
            f"The preferred transformation maps {len(cols)} of the "
            f"{nspace} spatial axes of the image, so NRRD cannot store "
            f"it."
        )
    linear = matrix[np.ix_(rows, cols)]
    shift = matrix[rows, -1]
    world_axes = [out_axes[r] for r in rows]

    to_ras = _orientation_matrix(world_axes)
    if to_ras is not None:
        target = _target_space(space, source, to_ras)
        convert = np.diag(_FLIPS[target]) @ to_ras
        fields = {"space": target}
        fields["space origin"] = _format_vectors([convert @ shift])
        units = [_unit_symbol(a) for a in world_axes]
        if all(units):
            fields["space units"] = _format_strings(units)
        frame = _measurement_frame(source, target)
        if frame is not None:
            fields["measurement frame"] = frame
        directions = list((convert @ linear).T)
        return "space", {"fields": fields, "directions": directions}

    units = [_unit_symbol(a) for a in world_axes]
    no_space = source is None or (
        "space" not in source.fields and "space dimension" not in source.fields
    )
    square = linear.shape[0] == linear.shape[1]
    if (
        no_space
        and square
        and not np.any(linear - np.diag(np.diag(linear)))
        and np.all(np.diag(linear) != 0)
    ):
        scale = np.diag(linear)
        if np.all(scale == 1) and not np.any(shift) and not any(units):
            return None
        return "spacings", (list(scale), list(shift), units)
    fields = {
        "space dimension": str(len(rows)),
        "space origin": _format_vectors([shift]),
    }
    if all(units):
        fields["space units"] = _format_strings(units)
    return "space", {"fields": fields, "directions": list(linear.T)}


def _measurement_frame(
    source: tx.Optional[NrrdHeader], target: str
) -> tx.Optional[str]:
    """
    The measurement frame of the source header, in the `target` space.

    Its vectors are expressed in the world space, so they change sign
    with it when the source was written in another anatomical space.
    """
    if source is None or "measurement frame" not in source.fields:
        return None
    try:
        space = source.space
        frame = source.measurement_frame
    except Exception:  # noqa: BLE001
        return source.fields["measurement frame"]
    if space == target or frame is None or frame.shape != (3, 3):
        return source.fields["measurement frame"]
    if space not in _FLIPS:
        return None
    flips = _FLIPS[target] * _FLIPS[space]
    return _format_vectors(list((flips[:, None] * frame).T))


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


@register_format
class AttachedNrrdImage(NrrdImage):
    """
    An image that is encoded by a NRRD file whose header and data are in
    the same file (`.nrrd`). See [`NrrdImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nrrd",)
    HINTS = ("attached",)

    @classmethod
    def _score_header(cls, header: NrrdHeader) -> float:
        return Confidence.NO if header.data_files else Confidence.LIKELY


@register_format
class DetachedNrrdImage(NrrdImage):
    """
    An image that is encoded by a detached NRRD header (`.nhdr`) and the
    data file(s) it names. See [`NrrdImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nhdr",)
    HINTS = ("nhdr", "detached")

    @classmethod
    def _score_header(cls, header: NrrdHeader) -> float:
        return Confidence.LIKELY if header.data_files else Confidence.NO
