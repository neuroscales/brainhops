"""The NRRD image formats."""

# stdlib
import math
import re
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis, is_component_type
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import ArrayCoordinateSystem, CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.datamodel.units import Unit, is_indexunit
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import same_file as _same_file
from brainhops.io.base._utils_files import sibling as _sibling
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    WriterError,
)
from brainhops.io.common._geometry import (
    RAS_FROM_ORIENTATION,
    held_transformations,
    reduce_to_affine,
    same_transformations,
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
    _decode,
    _NrrdProxy,
    encode_data,
    read_nrrd,
)
from brainhops.io.common.nrrd._layout import axis_roles, image_order
from brainhops.io.common.nrrd._raw import (
    _declined,
    _no_options,
    _sniffed_raw,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._base import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._views import _image_to_disk, _image_to_model

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
    """Return the role of a data model axis.

    Component axes, which hold the components of points, vectors or
    tensors, have the role of a channel.
    """
    type_ = getattr(axis, "type", None)
    if type_ in ("space", "time"):
        return type_
    if type_ == "channel" or is_component_type(type_):
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


def _record_transformations(
    metadata: tx.Optional[NrrdMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _nrrd_to_transformations(metadata.raw)


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
    voxel = ArrayCoordinateSystem(name="voxel", axes=axes, order="F")
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
    BinaryFileReader, BinaryFileWriter, ImageFormat, SingleScaleImage
):
    """An image stored in a NRRD file, attached or detached.

    This class is the unregistered base of [`AttachedNrrdImage`][] and
    [`DetachedNrrdImage`][], which answer to its hint `"nrrd"`. A NRRD
    image holds the header of its file as [`NrrdMetadata`][], whose
    record is an [`NrrdRaw`][], and the samples in `rawdata`. When the
    image is read from a file, `rawdata` is a lazy array that reads the
    samples only when they are needed, and `data` is decoded from it on
    first access. The data are indexed `[x, y, z, t, c, ...]` in Fortran
    order, whatever the order of the axes in the file.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. They are a [`Scaling`][]
    to `"physical"` and, with a world space, the preferred [`Affine`][]
    to the world, named after `space`.

    `rawdata` and the record describe the file as it was read, or as the image
    was last consolidated. Data that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the image is
    written, which writes such a copy and leaves the image as it was. The
    transformations, read or assigned, are compared at [`consolidate`][]
    with those that the record decodes to, and the geometry of the record
    is kept when they are the same.

    When the image is consolidated, the record of its metadata is the base
    of the new header, as [`consolidate`][] describes, so the fields and
    the key/value pairs that the data model has no place for, such as the
    `measurement frame` or `DWMRI_b-value`, are kept. An image that is
    read and written again without changes gives the same bytes, and a
    detached image keeps both of its files when it is written under the
    same name in another directory. A detached header names its data
    file, so a copy under another name differs in that line only.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nrrd",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The samples as the file stores them, or `None` without data.

    After a read, `rawdata` is a lazy array, which presents the samples
    in the axis order of the image. `rawdata` describes the file as it
    was read, or as the image was last consolidated, so assigning `data`
    leaves it as it is.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending: it is encoded into `rawdata` only by
    [`consolidate`][] or by a write.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The transformations assigned to the image, or `None`.

    The image takes over the `transformations` field of
    [`SingleScaleImage`][] to store it under this private name. The
    constructor still takes `transformations` in the same place, and the
    property of the same name reads it. Only an assigned list, even an
    empty one, is stored here. `None`, the default, means that no list
    was assigned, so the property decodes the list of the record and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[NrrdMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning
    other metadata drops the transformations decoded from the previous
    record, so that they are decoded from the new one, and keeps the
    transformations that were assigned.
    """

    @smartproperty
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the samples, assign `data`, and assigning
        `None` gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept.
        Assigning other metadata drops the decoded list, so that it is
        decoded from the new record, and keeps an assigned list. A change
        made in place to a decoded list is therefore lost when other
        metadata is assigned. An image without metadata has no
        transformations.

        Assigning a list replaces the stored one. In both cases, the list
        is compared at [`consolidate`][], and by a write, with the list
        that the record decodes to, exactly. An empty list means that the
        image has no geometry, and the image is then written as an image
        built from data alone. Assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the samples.

        The shape is that of the assigned data, or else of `rawdata`: the data
        is `rawdata` as the lazy array presents it, so the two have the
        same shape, and the lazy array knows its shape from the record.
        An image without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

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
        return ArrayCoordinateSystem(name="voxel", axes=axes, order="F")

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

        Parameters
        ----------
        file : IO
            The stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream does not hold this
            kind of NRRD file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
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
        """Return the confidence that bytes hold this kind of NRRD file.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes do not hold this
            kind of NRRD file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "NrrdImage":
        """Read an image from the path of a `.nrrd` or `.nhdr` file.

        The record is read from the header, and the lazy array reads the
        samples when they are needed.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a NRRD
        image holds no field of the vocabulary yet, so a field that is set
        is reported as lost, once, by `on_loss`.

        Parameters
        ----------
        filename : path
            The file of the image, or of its header.
        mmap : bool, default=True
            Whether the lazy array memory-maps raw samples that lie in a
            single local file.
        **kwargs : Any
            Refused, since NRRD has no other option.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the file or a data file that its header names does not
            exist.
        ParserContentError
            If the file is not a NRRD image, or a local file of raw
            samples is too short.
        """
        _no_options(kwargs)
        raw, proxy = read_nrrd(filename, mmap=mmap)
        return cls(rawdata=proxy, metadata=NrrdMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, mmap: bool = False, **kwargs
    ) -> "NrrdImage":
        """Read an image from an open NRRD file, without memory mapping.

        The record is read at once, and the lazy array reads the samples
        from the stream when they are needed, so the stream must stay open
        while the data may be read. The data files that a detached header
        names are found from the name of the stream. The position of the
        stream is restored.

        Parameters
        ----------
        file : file object
            The stream of the file, open in binary mode.
        mmap : bool, default=False
            Accepted for the readers of paths, and without effect: the
            samples of a stream are never memory-mapped.
        **kwargs : Any
            Refused, since NRRD has no other option.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream is not a NRRD image, or its header names data
            files and the stream has no name.
        """
        _no_options(kwargs)
        raw, proxy = read_nrrd(file, mmap=False)
        return cls(rawdata=proxy, metadata=NrrdMetadata.from_raw(raw))

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        encoding: tx.Optional[str] = None,
        endian: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        space: tx.Optional[str] = None,
        keyvalue: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> "NrrdImage":
        """Encode the pending data and geometry into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the image, as a
        file written from it would. The record is encoded in a fixed order.
        The base is the record of the metadata, or a new header for an
        image without metadata.

        1. The fields that describe how the samples are stored (`type`,
           `dimension`, `sizes`, `endian`, `encoding` and the skips)
           follow `rawdata`. A lazy array that the options do not change, and
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
           An image without transformations is written without geometry,
           as an image built from data alone.
        3. The other fields of the base, such as `content`, are kept when
           the shape is that of the base, except `min` and `max` when the
           samples are new. The key/value pairs of the base are kept, and
           `keyvalue` is merged into them last.

        The data files of the base are kept, and a writer sets those of the
        files that it writes. When this encoding gives the header of the
        base, the metadata is kept as it is, so that an image that was
        read and not changed writes the header that it read. Otherwise,
        the new record is held by new metadata, and the record held before
        is never changed. The samples to write become `rawdata`, which keeps a
        lazy array, and nothing is pending afterwards, so the
        transformations are decoded from the new record, with the
        precision of the file. The transformations that the image holds,
        read or assigned, are compared with those that the record decodes
        to, exactly, and an image with a record, no pending data, and the
        transformations of its record or none read, consolidated without
        options, is left as it is: the copy shares `rawdata` and the metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        encoding : {"raw", "gzip", "bzip2", "ascii", "hex"}, optional
            The encoding of the samples.
        endian : {"little", "big"}, optional
            The byte order of the samples.
        datatype : str or dtype, optional
            NRRD or NumPy type of the stored samples, to which the data
            are converted.
        space : str, optional
            The anatomical space of the geometry, such as `"LPS"`. This
            option is the space of the NRRD header, and not the field
            `space` of the metadata, which cannot be given by keyword here.
        keyvalue : mapping, optional
            Key/value pairs merged into those of the base. A value of
            `None` removes a key.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        NrrdImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
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
        report = ConversionReport(source="NrrdImage", target="NrrdImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, NrrdMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        # The record is kept when the transformations were never read, or
        # are the same as those that it decodes to.
        held = held_transformations(self)
        kept = held is None or same_transformations(
            held, _record_transformations(self.metadata)
        )
        if (
            self._data is None
            and kept
            and all(value is None for value in options.values())
            and not requested
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            if not inplace:
                return replace(self, transformations=None)
            self.transformations = None
            return self
        record, raw = _encoded(self, options, None)
        if (
            self.metadata is not None
            and record is self.metadata.raw
            and not requested
        ):
            metadata = self.metadata
        elif metadata is None:
            metadata = NrrdMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                rawdata=raw,
                metadata=metadata,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        return self

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a NRRD file with attached samples.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], and the header and `rawdata` of the copy are
        written. A lazy array that reaches the writer unchanged is written
        as the file stores it. The image itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        image = self.consolidate(**_options(kwargs))
        record = _nrrd_written(image, ())
        return record.to_bytes() + _payload(image, record)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a NRRD file with attached samples to a stream.

        Parameters
        ----------
        file : IO
            The stream.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
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
        `.raw.gz`, unless `data_file` names it. The image is consolidated
        into a copy with the other keyword arguments, as by
        [`consolidate`][], which is written, and the image itself is not
        changed. The content is encoded before any file is written, and
        each local file is replaced at once, so an image can be written
        over the files that it was read from. When the lazy array of the
        image reads a file that is written, the image is consolidated in
        place instead, so that it holds what the files hold (#426), and
        its lazy array reads the new files.

        Parameters
        ----------
        filename : path-like
            The path of the `.nrrd` or `.nhdr` file.
        data_file : str, optional
            The name of the data file, beside the header.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        options = _options(kwargs)
        name = filename.name
        attached = data_file is None and not name.lower().endswith(".nhdr")
        if data_file is None and not attached:
            held = None if self.metadata is None else self.metadata.raw
            encoding = _target_encoding(held, options["encoding"])
            data_file = name[: -len(".nhdr")] + _DATA_EXTENSIONS[encoding]
        if attached:
            written = [filename]
        else:
            written = [filename, _sibling(filename, data_file)]
        reading = _reads_from(self.rawdata, written)
        if reading:
            mmap = self.rawdata.root.mmap
            image = self.consolidate(inplace=True, **options)
        else:
            image = self.consolidate(**options)
        if attached:
            record = _nrrd_written(image, ())
            write_file(filename, record.to_bytes() + _payload(image, record))
        else:
            record = _nrrd_written(image, (data_file,))
            header, data = record.to_bytes(), _payload(image, record)
            write_file(written[1], data)
            write_file(filename, header)
        if reading:
            # The image holds what the files hold, and its lazy array reads
            # the new files, since the lazy array of the old files describes
            # how they stored the samples.
            self.rawdata = read_nrrd(filename, mmap=mmap)[1]
            self.data = None


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


_OPTIONS = ("encoding", "endian", "datatype", "space", "keyvalue")
"""The options of the NRRD writer, besides the `data_file` of a path."""


def _options(kwargs: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the writer options, refusing unknown ones.

    The options of the writer, `on_loss` and the fields of the metadata,
    which go to the consolidation of the metadata, are known. `space` is
    the option of the writer.

    Raises
    ------
    TypeError
        If an unknown option is given.
    """
    unknown = [
        name
        for name in kwargs
        if name not in _OPTIONS and name not in FIELDS and name != "on_loss"
    ]
    if unknown:
        raise TypeError(f"Unknown NRRD writer option(s): {', '.join(unknown)}")
    options = {name: kwargs.get(name) for name in _OPTIONS}
    options.update(
        {name: v for name, v in kwargs.items() if name not in _OPTIONS}
    )
    return options


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
) -> tx.Tuple[NrrdRaw, ArrayProtocol]:
    """Return the record and the samples that consolidating an image stores.

    The record is encoded as [`NrrdImage.consolidate`][] describes, and the
    held record is returned when the encoding changes nothing. The samples
    are the assigned data, or else `rawdata`, in the order of the image,
    converted to the stored type as the writer converts them, with
    `_stored_samples`. The `data_files` are those of the header: an empty
    tuple for attached samples, one name for a detached header, or `None`
    to keep those of the base.

    Raises
    ------
    WriterError
        If there is no data, or an option or the geometry cannot be
        written.
    """
    # The helpers read the transformations of a copy that holds the list
    # of the image, or the list of the record when the image holds none,
    # so that a write leaves the image as it was.
    held = held_transformations(image)
    if held is None:
        held = _record_transformations(image.metadata)
    image = replace(image, transformations=held)
    if image._data is not None:
        raw = _image_to_disk(image._data)
    else:
        raw = image.rawdata
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
        fields["type"] = nrrd_type
        fields["dimension"] = str(ndim)
        fields["sizes"] = " ".join(str(n) for n in sizes)
        fields["encoding"] = encoding
        fields["endian"] = endian
        if matched:
            # The base keeps its spelling of what it stores alike.
            for name in _same_storage(held, nrrd_type, encoding, endian):
                if name in held.fields:
                    fields[name] = held.fields[name]
                else:
                    fields.pop(name, None)

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
        return held, _stored_samples(held, raw)
    if version < 4:
        record = replace(record, version=4)
    return record, _stored_samples(record, raw)


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
) -> tx.List[str]:
    """Return the storage fields of the base that the writer would store
    alike.

    The base has the shape of the image, so its `dimension` and `sizes`
    are alike. Its `type` is alike when it names the same type, its
    `encoding` when it names the same encoding, and its `endian` when it
    names the same byte order or the type has a single byte, for which
    the byte order does not matter.
    """
    names = ["dimension", "sizes"]
    if dtype_to_nrrd(held.dtype) == nrrd_type:
        names.append("type")
        if held.dtype.itemsize == 1 or held.endian == endian:
            names.append("endian")
    if held.encoding == encoding:
        names.append("encoding")
    return names


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
    their geometry alone. An image without transformations, or whose
    preferred transformation is the unit scaling that a file without
    geometry decodes to, has no geometry to write, so the spacings,
    positions and units of its spatial axes are dropped as well. The base
    also gives the space and the measurement frame of the result,
    whatever its shape.

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
    if geometry is None:
        # The image is written as an image built from data alone, which
        # has none of these fields, rather than with the geometry of the
        # base.
        for name in ("spacings", "axis mins", "axis maxs", "units"):
            if name not in per_axis:
                continue
            empty = '""' if name == "units" else "nan"
            for i in spatial:
                per_axis[name][i] = empty
            if all(value == empty for value in per_axis[name]):
                del per_axis[name]
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


def _payload(image: NrrdImage, record: NrrdRaw) -> bytes:
    """Return the bytes of the samples that the header of an image describes.

    A lazy array that the header stores as its own record does, in one
    file, gives the bytes that the file stores, from the start of its
    samples. Any other array is put in the order of the file and encoded.
    """
    order = image_order(record)
    storage = {"datatype": record.dtype, "endian": record.endian}
    if _copies_samples(image.rawdata, order, record.encoding, storage):
        return image.rawdata.read_stored()
    # A lazy array is read here rather than through `data`, so that no
    # cached view of the old file outlives a write over that file.
    values = np.asarray(image.rawdata)
    return encode_data(record, values.transpose(np.argsort(order)))


def _stored_samples(record: NrrdRaw, samples: ArrayProtocol) -> ArrayProtocol:
    """Return samples as the file of a record stores them.

    The writer, `encode_data`, rounds floating-point values that are
    stored as integers and converts them to the stored type, and a text
    encoding writes them as decimal numbers. Samples of another type are
    encoded into memory in the same way and decoded as the reader decodes
    a file, so that they are the values of the file, in its type and byte
    order. Samples of the stored type, in either byte order, are returned
    as they are, so that a lazy array is not read.
    """
    if record.dtype.newbyteorder("=") == np.dtype(samples.dtype).newbyteorder(
        "="
    ):
        return samples
    order = image_order(record)
    values = np.asarray(samples).transpose(np.argsort(order))
    content = BytesIO(encode_data(record, values))
    return _decode(record, [(content, 0)], mmap=False).transpose(order)


def _nrrd_written(image: NrrdImage, data_files: tx.Tuple[str, ...]) -> NrrdRaw:
    """Return the header that a writer writes for a consolidated image.

    The header is the record of the image, which names the data files that
    the writer writes: none for attached samples, or one name for a
    detached header. A copy of the record names other files when they
    differ.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image.rawdata is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    record = image.metadata.raw
    if tuple(record.data_files) != tuple(data_files):
        record = replace(record, data_files=tuple(data_files))
    return record


def _reads_from(raw: tx.Any, written: tx.Sequence[tx.Any]) -> bool:
    """Tell whether a lazy array reads one of the files just written."""
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(raw, _NrrdProxy):
        return False
    return any(
        _same_file(read, file)
        for read in (raw.source, *raw.files)
        for file in written
    )


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

    Raises
    ------
    WriterError
        If `space` is given and the image has no world with an anatomical
        orientation, or the transformation maps too few spatial axes.
    """
    nspace = list(roles).count("space")
    if not image.transformations or not nspace:
        _refuse_space(space)
        return None
    affine = reduce_to_affine(image.transformation, "NRRD", "world")
    matrix = affine.homogeneous_matrix
    if matrix is None:
        _refuse_space(space)
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

    _refuse_space(space)
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


def _refuse_space(space: tx.Optional[str]) -> None:
    """Refuse a requested space for a world without anatomical axes.

    Raises
    ------
    WriterError
        If a space is requested.
    """
    if space is not None:
        raise WriterError(
            f"The image has no world whose axes have an anatomical "
            f"orientation, so its geometry cannot be written in the space "
            f"{space!r}."
        )


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
