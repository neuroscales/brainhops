"""The NRRD image formats."""

# stdlib
import math
import os
import re
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.datamodel.units import Unit, is_indexunit
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    WriterError,
)
from brainhops.io.common._geometry import (
    RAS_FROM_ORIENTATION,
    reduce_to_affine,
)
from brainhops.io.common.nrrd import NrrdMetadata, NrrdRaw
from brainhops.io.common.nrrd._codecs import (
    _format_float,
    _format_strings,
    _format_vectors,
    dtype_to_nrrd,
)
from brainhops.io.common.nrrd._constants import (
    _DATA_EXTENSIONS,
    _ENCODINGS,
    _SPACE_NAMES,
    SPACES,
)
from brainhops.io.common.nrrd._data import (
    _NrrdProxy,
    encode_data,
    read_nrrd,
    write_file,
)
from brainhops.io.common.nrrd._layout import axis_roles, image_order
from brainhops.io.common.nrrd._raw import _declined, _sniffed_raw
from brainhops.io.common.nrrd._views import _image_to_disk, _image_to_model
from brainhops.io.images.base import ImageFormat

_INDEX = "index"

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
"""Sign of each axis of an anatomical space relative to RAS."""

_STORAGE = (
    "type",
    "block size",
    "dimension",
    "sizes",
    "endian",
    "encoding",
    "line skip",
    "byte skip",
    "data file",
)
"""Fields that describe how the samples are stored."""

_GEOMETRY = (
    "space",
    "space dimension",
    "space directions",
    "space origin",
    "space units",
    "measurement frame",
)
"""Fields that describe the world space and the geometry."""

_PER_AXIS = (
    "spacings",
    "thicknesses",
    "axis mins",
    "axis maxs",
    "centers",
    "labels",
    "units",
)
"""Fields that hold one value per axis, other than `kinds`."""

_VALUES = ("min", "max")
"""Fields that describe the samples, which new samples make stale."""


# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------


def _space_name(space: str) -> str:
    """Return the name of the world space, the abbreviation of `space`."""
    return SPACES[space][0]


def _layout(header: NrrdRaw) -> tx.Tuple[tx.List[int], tx.List[Axis]]:
    """Return the axis order of the image and its axes.

    Element `j` of the order is the file axis that becomes image axis `j`
    (see [`image_order`][brainhops.io.common.nrrd._layout.image_order]).
    """
    roles = axis_roles(header)
    perm = image_order(header)
    return perm, _axes([roles[i] for i in perm], perm)


def _axes(roles: tx.Sequence[str], perm: tx.Sequence[int]) -> tx.List[Axis]:
    """Name the index axes `x`, `y`, `z`, `t`, `c` or `dim<file index>`."""
    axes = []
    nspace = 0
    seen = set()
    for role, i in zip(roles, perm):
        if role == "space":
            name = "xyz"[nspace] if nspace < 3 else f"dim{i}"
            nspace += 1
            axes.append(Axis(name, "space", unit=_INDEX))
        elif role in ("time", "channel"):
            name = role[0] if role not in seen else f"dim{i}"
            seen.add(role)
            axes.append(Axis(name, role, unit=_INDEX))
        else:
            axes.append(Axis(f"dim{i}", unit=_INDEX))
    return axes


def _role(axis: tx.Any) -> str:
    """Return the role of a data model axis."""
    type_ = getattr(axis, "type", None)
    if type_ in ("space", "time"):
        return type_
    if type_ in ("channel", "displacement", "coordinate"):
        return "channel"
    return "other"


def _with_unit(axis: Axis, unit: tx.Optional[str]) -> Axis:
    """Return `axis` in `unit`, or without a unit if the unit is unknown."""
    if unit:
        try:
            if type(Unit(unit)) is not Unit:
                return replace(axis, unit=unit)
        except Exception:  # noqa: BLE001
            pass
    return replace(axis, unit=None)


def _unit_symbol(axis: tx.Any) -> str:
    """Return the NRRD spelling of the unit of an axis, or `""`."""
    unit = getattr(axis, "unit", None)
    if unit is None or is_indexunit(unit):
        return ""
    try:
        return unit.symbol
    except Exception:  # noqa: BLE001
        return ""


# ----------------------------------------------------------------------
#   GEOMETRY (reading)
# ----------------------------------------------------------------------


def _world(header: NrrdRaw, sdim: int) -> CoordinateSystem:
    """Return the world space of the header.

    Only anatomical spaces have oriented axes, and a `-time` space adds an
    axis `t`. A world given only by `space dimension` is named `"world"`.
    Without `space units`, spatial axes are in millimeters, as ITK and
    Slicer assume, except in the 3D handed spaces and the bare `"world"`.
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


def _nrrd_to_transformations(header: NrrdRaw) -> tx.List[Transformation]:
    """Convert a NRRD header to transformations.

    With a world space (`space` or `space dimension`, and
    `space directions`), the transformations are:

    1. index -> `"physical"`: a `Scaling` by the length of each space
       direction (the `spacings` of the other axes, else 1);
    2. index -> world: the `Affine` whose columns are the space directions
       and whose translation is the `space origin`, the position of the
       centre of the first sample.

    Without a world space, a single `Scaling` or `Affine` to `"physical"`
    is built from `spacings`, `axis mins`, `axis maxs` and `centers`. A
    cell-centred axis, the default, spans `size` samples edge to edge.
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


class NrrdImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """An image stored in a NRRD file, attached or detached.

    This class is the unregistered base of [`AttachedNrrdImage`][] and
    [`DetachedNrrdImage`][], which answer to its hint `"nrrd"`. A NRRD
    image holds the header of its file as [`NrrdMetadata`][], whose
    record is an [`NrrdRaw`][], and the samples in `raw`. When the image
    is read from a file, `raw` is a lazy array that reads the samples
    only when they are needed, and `data` is decoded from it on first
    access. The data are indexed `[x, y, z, t, c, ...]` in Fortran order,
    whatever the order of the axes in the file.

    The transformations are decoded from the record, unless other
    transformations are assigned. They are a [`Scaling`][] to
    `"physical"` and, with a world space, the preferred [`Affine`][] to
    the world, named after `space`.

    When the image is written, the record of its metadata is the base of
    the new header, as [`to_raw`][] describes, so the fields and the
    key/value pairs that the data model has no place for, such as the
    `measurement frame` or `DWMRI_b-value`, are kept. An image that is
    read and written again without changes gives the same bytes, and a
    detached image keeps both of its files when it is written under the
    same name in another directory. A detached header names its data
    file, so a copy under another name differs in that line only.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nrrd",)

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The samples as the file stores them, or `None` without data.

    After a read, `raw` is a lazy array, which presents the samples in
    the axis order of the image. Setting `data` stores the new array
    here.
    """

    _metadata: KwOnly[NoRepr[tx.Optional[NrrdMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        # The constructor assigns the fields in order, so the default of
        # `raw`, assigned after `data`, erases the array stored by `data=`.
        # The array is therefore stored again. When both are given, `data`
        # takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(cache=True, invalidates=("data",))
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, decoded from `raw` on first access and cached.

        Data decoded from a file is read-only, because changing it in
        place would not change `raw`, which is what the image writes. To
        change the samples, assign `data`: setting it stores the new array
        in `raw` and drops the cached value.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.raw = None if value is None else _image_to_disk(value)

    @smartproperty(cache=True, unset=(None, "empty"))
    def transformations(self) -> tx.List[Transformation]:
        """Index-to-world transformations, decoded from the record.

        The decoded list is cached, so `img.transformation is
        img.transformation`. Assigning a list replaces the decoded one, and
        assigning other metadata drops it. An image without metadata has
        no transformations.
        """
        if self.metadata is None or self.metadata.raw is None:
            return []
        return _nrrd_to_transformations(self.metadata.raw)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read from `raw` without reading the samples.

        The data is `raw` as the lazy array presents it, so the two have
        the same shape, and the lazy array knows its shape from the
        record. An image without data raises as [`SingleScaleImage`][]
        does.
        """
        if self.raw is None:
            return self.data.shape
        return tuple(int(d) for d in self.raw.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The index coordinate system described by the record.

        The spatial axes come first, named `x`, `y` and `z`, then time
        (`t`), channels (`c`) and other axes (`dim<i>`, after their index
        in the file), in Fortran order. An image without metadata has no
        system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        _, axes = _layout(self.metadata.raw)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(cls, header: NrrdRaw) -> float:
        """Return how well a valid header matches this class.

        The subclasses score the kind of file, attached or detached, and
        this base class accepts both as a weak match.
        """
        return Confidence.MAYBE

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds this kind of NRRD file.

        The header is parsed and scored with [`_score_header`][], and the
        position of the stream is restored.
        """
        raw, failure = _sniffed_raw(file)
        if raw is None:
            return _declined(error, failure)
        score = cls._score_header(raw)
        if score:
            return score
        return _declined(error, None)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold this kind of NRRD file."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "NrrdImage":
        """Read an image from the path of a `.nrrd` or `.nhdr` file.

        The record is read from the header, and the lazy array reads the
        samples when they are needed. Other keyword arguments are ignored.

        Parameters
        ----------
        filename : path
            The file of the image, or of its header.
        mmap : bool, default=True
            Whether the lazy array memory-maps raw samples that lie in a
            single local file.
        **kwargs : Any
            Ignored.

        Raises
        ------
        ParserExistsError
            If the file or a data file that its header names does not
            exist.
        ParserContentError
            If the file is not a NRRD image, or a local file of raw
            samples is too short.
        """
        raw, proxy = read_nrrd(filename, mmap=mmap)
        return cls(raw=proxy, metadata=NrrdMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NrrdImage":
        """Read an image from an open NRRD file, without memory mapping.

        The record is read at once, and the lazy array reads the samples
        from the stream when they are needed, so the stream must stay open
        while the data may be read. The data files that a detached header
        names are found from the name of the stream. The position of the
        stream is restored, and the keyword arguments are ignored.

        Raises
        ------
        ParserContentError
            If the stream is not a NRRD image, or its header names data
            files and the stream has no name.
        """
        raw, proxy = read_nrrd(file, mmap=False)
        return cls(raw=proxy, metadata=NrrdMetadata.from_raw(raw))

    # --- writing ------------------------------------------------------

    def to_raw(
        self,
        encoding: tx.Optional[str] = None,
        endian: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        space: tx.Optional[str] = None,
        keyvalue: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
    ) -> NrrdRaw:
        """Return the [`NrrdRaw`][] record that writing the image stores.

        The record is encoded in a fixed order. The base is the record of
        the metadata, or a new header for an image without metadata.

        1. The fields that describe how the samples are stored (`type`,
           `dimension`, `sizes`, `endian`, `encoding` and the skips)
           follow `raw`. A lazy array that the options do not change, and
           whose samples lie in one file, is copied as the file stores
           it, so these fields are those of its own record. Any other
           array is stored with the options, or else with the encoding
           and the byte order of the base, gzip and little-endian by
           default, and with the type of the data. An image whose shape
           is that of the base keeps the order of the axes in the file
           and their `kinds`, and the axes of any other image are stored
           in the order of the data.
        2. The geometry comes from the preferred transformation. The
           geometry fields of the base (`space`, `space directions`,
           `space origin`, `space units`, `measurement frame` and the
           fields of one value per axis) are kept as they are stored when
           the shape and the preferred transformation are those of the
           base, exactly, and `space` does not ask for another space.
           Otherwise, the transformation becomes the `space directions`
           and the `space origin`, in `space`, or else in the space of
           the base, or in the anatomical space that the transformation
           maps to, and in RAS by default. A scaling onto unoriented axes
           of a base without a space becomes `spacings` and `axis mins`.
        3. The other fields of the base, such as `content`, are kept when
           the shape is that of the base, except `min` and `max` when the
           samples are new. The key/value pairs of the base are kept, and
           `keyvalue` is merged into them last.

        The data files of the base are kept. When this encoding gives the
        header of the base, the record of the metadata is returned as it
        is, without a copy, so that an image that was read and not changed
        writes the header that it read.

        Parameters
        ----------
        encoding : {"raw", "gzip", "bzip2", "ascii", "hex"}, optional
            The encoding of the samples.
        endian : {"little", "big"}, optional
            The byte order of the samples.
        datatype : str or dtype, optional
            NRRD or NumPy type of the stored samples, to which the data
            are converted.
        space : str, optional
            The anatomical space of the geometry, such as `"LPS"`.
        keyvalue : mapping, optional
            Key/value pairs merged into those of the base. A value of
            `None` removes a key.

        Returns
        -------
        NrrdRaw
            The record of the file.

        Raises
        ------
        WriterError
            If there is no data, the data are zero-dimensional, the type,
            the encoding, the byte order or the space is unknown, or the
            geometry cannot be written.
        UnrepresentableTransformationError
            If the preferred transformation has no affine form.
        """
        options = dict(
            encoding=encoding,
            endian=endian,
            datatype=datatype,
            space=space,
            keyvalue=keyvalue,
        )
        return _encoded(self, options, None)[0]

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a NRRD file with attached samples.

        The header is the record of [`to_raw`][], which receives the
        keyword arguments. A lazy array that reaches the writer unchanged
        is written as the file stores it.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        record, copied, order = _encoded(self, _options(kwargs), ())
        return record.to_bytes() + _payload(self, record, copied, order)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a NRRD file with attached samples to a stream."""
        file.write(self.to_bytes(**kwargs))

    def to_filename(
        self,
        filename: path.FilenameLike,
        data_file: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write to a path, attached or detached as the name says.

        The samples are written to a separate data file when the name ends
        with `.nhdr` or `data_file` is given. The data file is named after
        the header, with the extension of its encoding, such as
        `.raw.gz`, unless `data_file` names it. The other keyword
        arguments are those of [`to_raw`][]. The content is encoded before
        any file is written, and each local file is replaced at once, so
        an image can be written over the files that it was read from.
        When the lazy array of the image reads a file that is written,
        `raw` is then pointed at the new file, which holds the same
        samples.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        options = _options(kwargs)
        name = filename.name
        nhdr = name.lower().endswith(".nhdr")
        if not nhdr and data_file is None:
            stale = _reads_from(self.raw, [filename])
            write_file(filename, self.to_bytes(**options))
            if stale:
                self.raw = read_nrrd(filename, mmap=self.raw.mmap)[1]
            return
        if data_file is None:
            held = None if self.metadata is None else self.metadata.raw
            encoding = _target_encoding(held, options["encoding"])
            data_file = name[: -len(".nhdr")] + _DATA_EXTENSIONS[encoding]
        record, copied, order = _encoded(self, options, (data_file,))
        header = record.to_bytes()
        data = _payload(self, record, copied, order)
        target = _sibling(filename, data_file)
        stale = _reads_from(self.raw, [filename, target])
        write_file(target, data)
        write_file(filename, header)
        if stale:
            self.raw = read_nrrd(filename, mmap=self.raw.mmap)[1]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


_OPTIONS = ("encoding", "endian", "datatype", "space", "keyvalue")
"""The options of the NRRD writer, besides the `data_file` of a path."""


def _options(kwargs: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the writer options, refusing unknown ones.

    Raises
    ------
    TypeError
        If an unknown option is given.
    """
    unknown = [name for name in kwargs if name not in _OPTIONS]
    if unknown:
        raise TypeError(f"Unknown NRRD writer option(s): {', '.join(unknown)}")
    return {name: kwargs.get(name) for name in _OPTIONS}


def _target_encoding(
    held: tx.Optional[NrrdRaw], encoding: tx.Optional[str]
) -> str:
    """Return the encoding of the samples to write.

    The option wins, then the encoding of the base, and gzip by default.

    Raises
    ------
    WriterError
        If the encoding is unknown.
    """
    if encoding is None:
        return held.encoding if held is not None else "gzip"
    key = str(encoding).lower()
    if key not in _ENCODINGS:
        raise WriterError(f"Unknown NRRD encoding: {encoding!r}")
    return _ENCODINGS[key]


def _encoded(
    image: NrrdImage,
    options: tx.Mapping[str, tx.Any],
    data_files: tx.Optional[tx.Tuple[str, ...]],
) -> tx.Tuple[NrrdRaw, bool, tx.List[int]]:
    """Return the record that writing an image stores, and how to store it.

    The record is encoded as [`NrrdImage.to_raw`][] describes. The result
    also tells whether `raw` is a lazy array whose samples are copied as
    the file stores them, and gives the order of the image axes in the
    file. The `data_files` are those of the header: an empty tuple for
    attached samples, one name for a detached header, or `None` to keep
    those of the base.

    Raises
    ------
    WriterError
        If there is no data, or an option or the geometry cannot be
        written.
    """
    raw = image.raw
    if raw is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    shape = tuple(int(n) for n in raw.shape)
    ndim = len(shape)
    if ndim == 0:
        raise WriterError("NRRD cannot store a zero-dimensional array.")
    held = None if image.metadata is None else image.metadata.raw
    order, roles = _storage_order(image, held, shape)
    sizes = [0] * ndim
    for j, i in enumerate(order):
        sizes[i] = shape[j]
    # The shape is that of the base when the axes are stored in its order
    # and with its sizes.
    matched = (
        held is not None
        and list(image_order(held)) == list(order)
        and tuple(held.sizes) == tuple(sizes)
    )
    encoding = _target_encoding(held, options["encoding"])
    endian = options["endian"]
    if endian is not None and endian not in ("little", "big"):
        raise WriterError(f"Unknown NRRD endian: {endian!r}")
    copied = _copies_samples(raw, order, encoding, options)
    fields: tx.Dict[str, str] = {}

    # --- storage --------------------------------------------------
    if copied:
        for name in _STORAGE:
            if name in raw.record.fields and name != "data file":
                fields[name] = raw.record.fields[name]
        if endian is not None:
            fields["endian"] = endian
    else:
        datatype = options["datatype"]
        if datatype is None:
            datatype = getattr(raw, "dtype", np.float32)
        nrrd_type = dtype_to_nrrd(datatype)
        if endian is None:
            endian = (held.endian if held is not None else None) or "little"
        if matched and _same_storage(held, nrrd_type, encoding, endian):
            # The base spells the same storage, so its text is kept.
            for name in ("type", "dimension", "sizes", "endian", "encoding"):
                if name in held.fields:
                    fields[name] = held.fields[name]
        else:
            fields["type"] = nrrd_type
            fields["dimension"] = str(ndim)
            fields["sizes"] = " ".join(str(n) for n in sizes)
            fields["encoding"] = encoding
            fields["endian"] = endian

    # --- kinds and other fields of the base -----------------------
    if matched:
        for name, value in held.fields.items():
            if name in _STORAGE or name in _GEOMETRY or name in _PER_AXIS:
                continue
            if name in _VALUES and not copied:
                continue
            fields[name] = value
    else:
        fields["kinds"] = " ".join(_kinds(roles, order))

    # --- geometry -------------------------------------------------
    if matched and _keeps_geometry(image, held, options["space"]):
        for name in _GEOMETRY + _PER_AXIS:
            if name in held.fields:
                fields[name] = held.fields[name]
    else:
        fields.update(
            _geometry_fields(
                image, held, matched, roles, order, options["space"]
            )
        )
    if "space directions" not in fields:
        fields.pop("measurement frame", None)

    merged = _merged_keyvalue(held, options["keyvalue"])
    if data_files is None:
        data_files = held.data_files if held is not None else ()
    version = held.version if held is not None else 4
    record = NrrdRaw(
        version=version,
        fields=fields,
        keyvalue=merged,
        data_files=tuple(data_files),
    )
    if held is not None and record == held:
        return held, copied, order
    if version < 4:
        record = replace(record, version=4)
    return record, copied, order


def _storage_order(
    image: NrrdImage, held: tx.Optional[NrrdRaw], shape: tx.Tuple[int, ...]
) -> tx.Tuple[tx.List[int], tx.List[str]]:
    """Return the file axis of each image axis, and the role of each.

    An image whose shape is that of the base keeps the order of the axes
    in the file of the base. Any other image stores its axes in the order
    of the data. The roles come from the axes of the voxel system, or of
    the input of the preferred transformation, and otherwise the first
    three axes are spatial.
    """
    ndim = len(shape)
    system = image.system
    if system is None and image.transformations:
        system = getattr(image.transformation, "input", None)
    try:
        axes = list(getattr(system, "axes", None) or [])
    except TypeError:
        axes = []
    if len(axes) == ndim and not any(a is ... for a in axes):
        roles = [_role(a) for a in axes]
    else:
        roles = ["space"] * min(3, ndim) + ["other"] * (ndim - 3)
    if held is not None and held.dimension == ndim:
        order = image_order(held)
        if tuple(held.sizes[i] for i in order) == shape:
            return order, roles
    return list(range(ndim)), roles


def _copies_samples(
    raw: ArrayProtocol,
    order: tx.Sequence[int],
    encoding: str,
    options: tx.Mapping[str, tx.Any],
) -> bool:
    """Tell whether the samples of `raw` are copied as the file stores them.

    They are when `raw` is a lazy array whose samples lie in one file,
    stored in the order that the writer chose, and the options name the
    encoding, the type and the byte order that its record stores, or are
    not given.
    """
    if not isinstance(raw, _NrrdProxy) or not raw.copyable:
        return False
    record = raw.record
    if list(image_order(record)) != list(order):
        return False
    if encoding != record.encoding:
        return False
    datatype = options["datatype"]
    if datatype is not None and dtype_to_nrrd(datatype) != dtype_to_nrrd(
        record.dtype
    ):
        return False
    endian = options["endian"]
    if (
        endian is not None
        and record.dtype.itemsize > 1
        and endian != record.endian
    ):
        return False
    return True


def _same_storage(
    held: NrrdRaw, nrrd_type: str, encoding: str, endian: str
) -> bool:
    """Tell whether the base stores samples of a type as the writer would.

    The byte order matters only for types of more than one byte.
    """
    if dtype_to_nrrd(held.dtype) != nrrd_type or held.encoding != encoding:
        return False
    return held.dtype.itemsize == 1 or held.endian == endian


def _keeps_geometry(
    image: NrrdImage, held: NrrdRaw, space: tx.Optional[str]
) -> bool:
    """Tell whether the geometry fields of the base are written as stored.

    The image must have the shape of the base. The fields are kept when
    `space` names the space of the base or is not given, and the
    preferred transformation is the one that the base decodes to,
    exactly. The test cannot be whether `transformations` was assigned,
    because `from_instance` and `replace` assign the decoded list.
    """
    if space is not None:
        key = str(space).lower()
        if _SPACE_NAMES.get(key, key) != held.space:
            return False
    decoded = _nrrd_to_transformations(held)
    current = list(image.transformations)
    if not decoded or not current:
        return len(decoded) == len(current)
    return _same_transformation(current[-1], decoded[-1])


def _same_transformation(first: tx.Any, second: tx.Any) -> bool:
    """Tell whether two transformations are the same map between the same
    spaces, exactly.

    The parameters are compared with NaN equal to NaN, so that a geometry
    that holds NaN is still recognized.
    """
    if type(first) is not type(second):
        return False
    if first.input != second.input or first.output != second.output:
        return False
    a = np.asarray(first.data, dtype=float)
    b = np.asarray(second.data, dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _geometry_fields(
    image: NrrdImage,
    held: tx.Optional[NrrdRaw],
    matched: bool,
    roles: tx.Sequence[str],
    order: tx.Sequence[int],
    space: tx.Optional[str],
) -> tx.Dict[str, str]:
    """Encode the preferred transformation into the geometry fields.

    The fields of one value per axis are taken from the base when the
    shape is that of the base, and the spatial axes are then described by
    their geometry alone. The base also gives the space and the
    measurement frame of the result, whatever its shape.

    Raises
    ------
    WriterError
        If the space is unknown, or the transformation maps too few axes.
    """
    ndim = len(order)
    fields: tx.Dict[str, str] = {}
    per_axis: tx.Dict[str, tx.List[str]] = {}
    if matched:
        for name in _PER_AXIS:
            value = held.fields.get(name)
            if value is None:
                continue
            if name in ("labels", "units"):
                values = ['"' + v + '"' for v in _split_quoted(value)]
            else:
                values = value.split()
            if len(values) == ndim:
                per_axis[name] = values
    spatial = [order[j] for j, r in enumerate(roles) if r == "space"]
    geometry = _geometry(image, roles, space, held)
    if geometry is not None:
        kind, values = geometry
        if kind == "space":
            fields.update(values["fields"])
            directions = [None] * ndim
            for i, v in zip(spatial, values["directions"]):
                directions[i] = v
            fields["space directions"] = _format_vectors(directions)
            # Spatial axes are described by their directions alone.
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
    return fields


def _merged_keyvalue(
    held: tx.Optional[NrrdRaw],
    keyvalue: tx.Optional[tx.Mapping[str, tx.Optional[str]]],
) -> tx.Dict[str, str]:
    """Merge key/value pairs into those of the base.

    A value of `None` removes a key.

    Raises
    ------
    WriterError
        If a key is empty, starts with `#`, holds `:=`, or a key or a
        value cannot be written as UTF-8. A header would then read back
        other pairs.
    """
    merged = dict(held.keyvalue) if held is not None else {}
    for key, value in (keyvalue or {}).items():
        key = str(key)
        if not key or key.startswith("#") or ":=" in key:
            raise WriterError(
                f"The NRRD key {key!r} cannot be written: a key must not be "
                f"empty, start with '#' or hold ':='."
            )
        if value is None:
            merged.pop(key, None)
            continue
        value = str(value)
        try:
            (key + value).encode("utf-8")
        except UnicodeEncodeError as error:
            raise WriterError(
                f"The NRRD key/value pair {key!r} cannot be written as "
                f"UTF-8: {error}"
            ) from error
        merged[key] = value
    return merged


def _payload(
    image: NrrdImage,
    record: NrrdRaw,
    copied: bool,
    order: tx.Sequence[int],
) -> bytes:
    """Return the bytes of the samples that a header describes.

    A lazy array that is copied gives the bytes that the file stores, from
    the start of its samples. Any other array is put in the order of the
    file and encoded.
    """
    if copied:
        return image.raw.read_stored()
    values = image.data if isinstance(image.raw, _NrrdProxy) else image.raw
    return encode_data(record, np.asarray(values).transpose(np.argsort(order)))


def _reads_from(raw: tx.Any, written: tx.Sequence[tx.Any]) -> bool:
    """Tell whether a lazy array reads one of the files just written.

    Paths are compared once resolved, and a stream is never one of them.
    """
    if not isinstance(raw, _NrrdProxy):
        return False
    read = [raw.source, *raw.files]
    resolved = {_resolved(file) for file in read} - {None}
    return any(_resolved(file) in resolved for file in written)


def _resolved(file: tx.Any) -> tx.Optional[str]:
    """Return the resolved name of a local path, or `None`."""
    if not isinstance(file, (str, os.PathLike)):
        return None
    try:
        if path.Path(file).protocol.lower() not in ("", "file", "local"):
            return None
        return os.path.realpath(os.fspath(file))
    except Exception:  # noqa: BLE001
        return None


def _split_quoted(value: str) -> tx.List[str]:
    """Return the quoted strings of a field, still escaped."""
    return re.findall(r'"((?:[^"\\]|\\.)*)"', value)


def _kinds(roles: tx.Sequence[str], perm: tx.Sequence[int]) -> tx.List[str]:
    """Return the kind of each file axis from the role of each image axis."""
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
    """Return the matrix from three oriented `axes` to RAS, or `None`."""
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
    source: tx.Optional[NrrdRaw],
    to_ras: np.ndarray,
) -> str:
    """Return the anatomical `space` in which to write the geometry.

    The requested space wins, then the source space, then the space
    matching `to_ras`, and finally RAS.

    Raises
    ------
    WriterError
        If the requested space is not RAS, LAS or LPS.
    """
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
    source: tx.Optional[NrrdRaw],
) -> tx.Optional[tx.Tuple[str, tx.Any]]:
    """Return the geometry fields of the preferred transformation.

    The result is `None` when there is nothing to write, `("space", ...)`
    for space fields, or `("spacings", (spacings, positions, units))` for
    a scaling onto unoriented axes when the source had no space either.
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
    source: tx.Optional[NrrdRaw], target: str
) -> tx.Optional[str]:
    """Return the source measurement frame expressed in the `target` space.

    The vectors are in world coordinates, so their signs follow the
    change of space.
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
    """An image whose header and samples share one `.nrrd` file.

    See [`NrrdImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nrrd",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("attached",)

    @classmethod
    def _score_header(cls, header: NrrdRaw) -> float:
        """Score a header whose samples follow it in the same file."""
        return Confidence.NO if header.data_files else Confidence.LIKELY


@register_format
class DetachedNrrdImage(NrrdImage):
    """An image whose `.nhdr` header names separate data files.

    See [`NrrdImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nhdr",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nhdr", "detached")

    @classmethod
    def _score_header(cls, header: NrrdRaw) -> float:
        """Score a header that names separate data files."""
        return Confidence.LIKELY if header.data_files else Confidence.NO
