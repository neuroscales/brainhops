"""The record of an image stack: the list of its files, their headers and
the geometry of the stack."""

# stdlib
import math
import os
import pathlib
import re
import warnings

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_PILLOW, HAS_TIFFFILE
from brainhops.io.base._utils_files import is_local
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    SnifferTypeError,
)
from brainhops.io.images.base import _utils_raster as raster

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

_ZARR_MARKERS = (".zarray", ".zgroup", ".zattrs", "zarr.json")
"""Files whose presence makes a folder a Zarr store, which is never read as
a stack."""

_CHANNELS = "SC"
"""The storage codes of a channel axis, as tifffile spells them: samples
per pixel (`S`) or channels (`C`)."""

_LISTED = 5
"""The number of names or numbers that an error message lists at most."""

PLANES: tx.Dict[str, str] = {"z": "YX", "y": "ZX", "x": "ZY"}
"""The axes of the volume along the rows and the columns of each file, as
storage codes, for each stacking axis.

The columns of a file lie along the first axis of the volume that is not
the stacking axis, and its rows along the second, so that a file of a
stack along z is an `(x, y)` plane, a file of a stack along y is an
`(x, z)` plane and a file of a stack along x is a `(y, z)` plane.
"""


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class StackMember(Magic, frozen=True):
    """One file of an image stack, and what its header says.

    A member is a two-dimensional image, possibly with several channels,
    stored in a TIFF file or in a raster format that Pillow reads, such as
    PNG. The member records the name of the file, the number that the name
    carries, and the header that the reader of that format parses: a
    `SeriesMetadata` from `brainhops.io.images.tiff._utils` for a TIFF
    file, and a `PillowHeader` from `brainhops.io.images.pillow._utils`
    for any other file.
    """

    name: str
    """The path of the file relative to the folder, with `/` separators."""

    number: int
    """The number that the name of the file carries.

    The number gives the position of the file in the stack. A stack whose
    names carry several numbers, such as a time point and a slice, would
    give each member one number per stacking axis, which is not supported
    yet.
    """

    format: str
    """The reader of the file: `"tiff"` (tifffile) or `"pillow"`."""

    axes: str
    """The axes of the decoded file, slowest first, spelled as in tifffile,
    such as `"YX"`, `"YXS"` or `"SYX"`."""

    shape: tx.Tuple[int, ...]
    """The shape of the decoded file, in the order of `axes`."""

    dtype: str
    """The NumPy type of the decoded pixels, in native byte order."""

    header: tx.Any = None
    """The header that the reader of the format parses, or `None`.

    It is a `SeriesMetadata` for a TIFF file, which holds its tags, its
    OME-XML and its ImageJ metadata, and a `PillowHeader` for any other
    file, which holds its mode and the `info` dictionary of Pillow. It is
    `None` in a record that was encoded for files that are not written
    yet, since the header of a file is known only once the file is
    written.
    """

    @property
    def plane_axes(self) -> str:
        """The axes in which the stack presents the file: `"YX"`, or
        `"YXS"` for a file with channels."""
        return "YXS" if len(self.axes) == 3 else "YX"

    @property
    def plane_order(self) -> tx.Tuple[int, ...]:
        """The permutation from `axes` to `plane_axes`.

        Element `i` is the position, in the decoded file, of axis `i` of
        the presented plane. A channel axis, stored as `S` or `C` and
        wherever it lies, comes last.
        """
        order = [self.axes.index("Y"), self.axes.index("X")]
        order += [i for i, code in enumerate(self.axes) if code in _CHANNELS]
        return tuple(order)

    @property
    def plane_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the presented plane, `(rows, columns[, channels])`."""
        return tuple(self.shape[i] for i in self.plane_order)


class StackRaw(Magic, frozen=True):
    """The record of an image stack, which lists its files.

    An image stack is a folder of numbered two-dimensional images, such as
    `slice_000.tif`, `slice_001.tif` and so on, which together form one
    volume. The folder has no header of its own, so the record lists its
    members, sorted by the number that their names carry, with the header
    of each file, and it holds the geometry of the stack: the axis along
    which the files are stacked, the size of their pixels and the spacing
    of the slices. The geometry is decoded from the files when they
    record it, and the options of the reader can set it. Every member has
    the same format, shape, type and number of channels. The record never
    reads pixels.

    The numbers of the members may skip some values. The stack then has a
    slice for every number from the first to the last, and the slices
    whose number no member carries are missing: they have no file, and
    the lazy array of the image fills them (see [`missing`][]).

    The record is held by
    [`StackMetadata`][brainhops.io.images.stack.StackMetadata].
    It is frozen, and two records are equal when their fields are.
    """

    members: tx.Tuple[StackMember, ...] = ()
    """The files of the stack, sorted by their numbers."""

    pattern: tx.Optional[str] = None
    """The glob pattern that selected the files in the folder, or `None`
    when every image file of the folder was taken."""

    axis: str = "z"
    """The axis of the volume along which the files are stacked: `"x"`,
    `"y"` or `"z"`.

    The files of a stack along z are `(x, y)` planes, those of a stack
    along y are `(x, z)` planes and those of a stack along x are `(y, z)`
    planes. In each case, the columns of a file lie along the first of the
    two other axes, and its rows along the second.
    """

    pixel_size: tx.Optional[tx.Tuple[float, float]] = None
    """The size of a pixel of the files along their columns and their rows,
    in `unit`, or `None` when it is unknown."""

    spacing: tx.Optional[float] = None
    """The distance from one slice to the next, in `unit`, or `None` when
    it is unknown.

    A negative spacing means that the slices run backwards along the
    stacking axis.
    """

    unit: tx.Optional[str] = None
    """The unit of length of `pixel_size` and `spacing`, or `None` when
    they have no unit."""

    @property
    def names(self) -> tx.Tuple[str, ...]:
        """The names of the files, relative to the folder, in order."""
        return tuple(member.name for member in self.members)

    @property
    def format(self) -> tx.Optional[str]:
        """The reader of the files, `"tiff"` or `"pillow"`, or `None`
        for a record without members."""
        return self.members[0].format if self.members else None

    @property
    def numbers(self) -> tx.Tuple[int, ...]:
        """The numbers of the slices, from the first member to the last,
        whether a member carries them or not."""
        if not self.members:
            return ()
        first, last = self.members[0].number, self.members[-1].number
        return tuple(range(first, last + 1))

    @property
    def missing(self) -> tx.Tuple[int, ...]:
        """The numbers of the slices that no member carries, in order."""
        held = {member.number for member in self.members}
        return tuple(n for n in self.numbers if n not in held)

    @property
    def positions(self) -> tx.Tuple[int, ...]:
        """The position of each member along the stacking axis, in order."""
        if not self.members:
            return ()
        first = self.members[0].number
        return tuple(member.number - first for member in self.members)

    @property
    def axes(self) -> str:
        """The axes of the stored volume, slowest first, as storage codes.

        The first axis is the stacking axis, and the two others are the
        rows and the columns of the files (see [`PLANES`][]), followed by
        `S` for files with channels. A stack along z is therefore stored
        as `"ZYX"` or `"ZYXS"`, as tifffile spells a volume.
        """
        codes = self.axis.upper() + PLANES[self.axis]
        if self.members and len(self.members[0].plane_shape) == 3:
            codes += "S"
        return codes

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stored volume, in the order of `axes`."""
        if not self.members:
            return (0, 0, 0)
        return (len(self.numbers),) + self.members[0].plane_shape

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the pixels."""
        if not self.members:
            return np.dtype("float64")
        return np.dtype(self.members[0].dtype)

    def copy(self) -> "StackRaw":
        """Return a copy of the record.

        The record is frozen and its fields are tuples, so the copy shares
        its members.

        Returns
        -------
        StackRaw
            The copy.
        """
        return replace(self)

    @classmethod
    def from_folder(
        cls,
        folder: path.FilenameLike,
        pattern: tx.Optional[str] = None,
        axis: tx.Optional[tx.Union[str, int]] = None,
        pixel_size: tx.Optional[tx.Union[float, tx.Sequence[float]]] = None,
        spacing: tx.Optional[float] = None,
        unit: tx.Optional[tx.Any] = None,
    ) -> "StackRaw":
        """Read the record of the image stack that a folder holds.

        The files are listed, their names are checked and sorted by the
        number that they carry, and the header of each file is read. The
        pixels are not read. The size of the pixels and the spacing of the
        slices are those on which every file agrees, unless the options
        set them.

        Parameters
        ----------
        folder : path-like
            The path of a local folder.
        pattern : str, optional
            A glob pattern, relative to the folder, that selects the files
            of the stack, such as `"slice_*.tif"`. By default, every file of
            the folder whose extension is that of a TIFF file or of a
            format that Pillow reads is taken. Hidden files, whose names
            start with a dot, are never taken.
        axis : {"x", "y", "z", 0, 1, 2}, optional
            The axis of the volume along which the files are stacked, by
            name or by its position in `(x, y, z)`. It is z by default.
        pixel_size : float or Sequence[float], optional
            The size of a pixel of the files, one size for the columns and
            the rows or a pair `(columns, rows)`, which replaces the size
            that the files record.
        spacing : float, optional
            The distance from one slice to the next, which replaces the
            spacing that the files record. A negative spacing reverses the
            stacking axis.
        unit : str or Unit, optional
            The unit of length of `pixel_size` and `spacing`. The sizes
            that the files record are converted to it.

        Returns
        -------
        StackRaw
            The record.

        Raises
        ------
        ValueError
            If an option is not valid.
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the path is not a local folder, if it holds no image file,
            if the names of the files do not follow one numbering, if a
            number is repeated, or if the files differ in format, shape,
            type or number of channels.
        """
        axis = checked_axis("z" if axis is None else axis)
        local = local_folder(folder)
        numbered = _numbered(_listed(local, pattern))
        return read_record(
            local,
            numbered,
            pattern=pattern,
            axis=axis,
            pixel_size=pixel_size,
            spacing=spacing,
            unit=unit,
        )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def read_record(
    folder: str,
    numbered: tx.Sequence[tx.Tuple[int, str]],
    pattern: tx.Optional[str] = None,
    axis: str = "z",
    pixel_size: tx.Any = None,
    spacing: tx.Any = None,
    unit: tx.Any = None,
) -> StackRaw:
    """Read the record of a stack whose files and numbers are known.

    The reader of a folder lists and numbers the files first, and the
    writer of a stack knows the files that it wrote, so both read the
    headers of the files here and decode the geometry of the stack.

    Raises
    ------
    ValueError
        If an option of the geometry is not valid.
    ParserContentError
        If a file cannot be read, or if the files cannot be stacked.
    """
    members = tuple(
        _read_member(folder, name, number) for number, name in numbered
    )
    _check_alike(members)
    recorded = _recorded_geometry(members)
    sizes, step, length = _geometry_options(
        recorded, pixel_size, spacing, unit
    )
    return StackRaw(
        members=members,
        pattern=pattern,
        axis=axis,
        pixel_size=sizes,
        spacing=step,
        unit=length,
    )


def checked_axis(axis: tx.Any) -> str:
    """Return the name of a stacking axis given by name or position.

    Raises
    ------
    ValueError
        If the axis is not x, y or z, or 0, 1 or 2.
    """
    if isinstance(axis, str) and axis.lower() in PLANES:
        return axis.lower()
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
        if 0 <= int(axis) < 3:
            return "xyz"[int(axis)]
    raise ValueError(
        f"The stacking axis must be 'x', 'y' or 'z', or its position 0, 1 "
        f"or 2 in (x, y, z), not {axis!r}."
    )


def sniff_folder(
    folder: tx.Any, pattern: tx.Optional[str] = None
) -> tx.Optional[str]:
    """Tell why a path cannot be an image stack, or `None` if it can.

    Only the names of the files are checked, so no file is opened. A path
    can hold a stack if it is a local folder that is not a Zarr store,
    and if the files that `pattern` selects are image files whose names
    follow one numbering, with no repeated number.

    Parameters
    ----------
    folder : path-like
        The path to check.
    pattern : str, optional
        The glob pattern that selects the files of the stack.

    Returns
    -------
    str or None
        The reason why the path is not a stack, or `None`.
    """
    try:
        local = local_folder(folder)
    except (ParserExistsError, ParserContentError) as e:
        return str(e)
    for marker in _ZARR_MARKERS:
        if os.path.exists(os.path.join(local, marker)):
            return f"The folder {local!r} is a Zarr store."
    try:
        _numbered(_listed(local, pattern))
    except ParserContentError as e:
        return str(e)
    return None


def member_format(name: str) -> tx.Optional[str]:
    """Return the reader of a file from its extension, or `None`.

    A TIFF extension gives `"tiff"` when tifffile is installed, and an
    extension that the Pillow reader claims gives `"pillow"` when Pillow
    is installed. Other files, and the files of a library that is not
    installed, give `None`. The comparison ignores case.
    """
    return image_extension(name)[1]


def image_extension(name: str) -> tx.Tuple[str, tx.Optional[str]]:
    """Return the image extension that a name ends with, and its reader.

    The longest extension wins, so that `"a.ome.tif"` ends with
    `".ome.tif"`. The extension keeps the case of the name. A name without
    an image extension, or with the extension of a library that is not
    installed, gives `("", None)`.
    """
    lower = name.lower()
    found = ("", None)
    for fmt, extensions in _extensions():
        for extension in extensions:
            if lower.endswith(extension) and len(extension) > len(found[0]):
                found = (name[len(name) - len(extension) :], fmt)
    return found


def local_folder(folder: tx.Any) -> str:
    """Return the local path of an existing folder.

    Raises
    ------
    ParserExistsError
        If the path does not exist.
    ParserContentError
        If the path is not local, or is not a folder.
    """
    local = local_name(folder)
    if not os.path.exists(local):
        raise ParserExistsError(f"No such folder: {local}")
    if not os.path.isdir(local):
        raise ParserContentError(
            f"An image stack is a folder, and {local!r} is a file."
        )
    return local


def local_name(target: tx.Any) -> str:
    """Return the local path that a path names, which may not exist.

    Raises
    ------
    ParserContentError
        If `target` is not a path, or not a local one.
    """
    if isinstance(target, str):
        target = path.Path(target)
    if not isinstance(target, (str, os.PathLike)):
        raise ParserContentError(
            f"An image stack is read from the path of a folder, not from "
            f"an object of type {type(target).__name__}."
        )
    if not is_local(target):
        raise ParserContentError(
            f"Image stacks are stored in local folders only, and "
            f"{str(target)!r} is not local."
        )
    return os.fspath(target)


def _extensions() -> tx.List[tx.Tuple[str, tx.Tuple[str, ...]]]:
    """Return the extensions of the readers that are installed."""
    out = []
    if HAS_TIFFFILE:
        # The TIFF package registers its formats when it is imported, so
        # it is imported only when tifffile is installed, as
        # `brainhops.io.images` does.
        from brainhops.io.images.tiff._utils import EXTENSIONS

        out.append(("tiff", EXTENSIONS))
    if HAS_PILLOW:
        from brainhops.io.images.pillow._utils import EXTENSIONS

        out.append(("pillow", EXTENSIONS))
    return out


def _listed(folder: str, pattern: tx.Optional[str]) -> tx.List[str]:
    """List the image files of a folder, relative to it, with `/`.

    Without a pattern, the files of the folder itself are listed, except
    files whose extension is not an image extension. With a pattern, the
    files that the glob pattern selects are listed, and each must have an
    image extension. Hidden files, whose names start with a dot, are
    never listed, which also leaves out the temporary files of a writer.
    The names are sorted, so that errors name the same files each time.

    Raises
    ------
    ParserContentError
        If no file is listed, if a selected file is not an image file, or
        if the pattern reaches outside the folder.
    """
    names: tx.List[str] = []
    if pattern is None:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.startswith(".") or not entry.is_file():
                    continue
                if member_format(entry.name) is not None:
                    names.append(entry.name)
        where = f"in the folder {folder!r}"
    else:
        if os.path.isabs(pattern) or ".." in pathlib.PurePath(pattern).parts:
            raise ParserContentError(
                f"The pattern {pattern!r} must select files inside the "
                f"folder, relative to it."
            )
        for found in pathlib.Path(folder).glob(pattern):
            name = found.relative_to(folder).as_posix()
            hidden = any(part.startswith(".") for part in name.split("/"))
            if hidden or not found.is_file():
                continue
            if member_format(name) is None:
                raise ParserContentError(
                    f"The pattern {pattern!r} selects {name!r}, which is "
                    f"not a TIFF file nor an image that Pillow reads, judged "
                    f"by its extension (TIFF files need tifffile and other "
                    f"images need Pillow)."
                )
            names.append(name)
        where = f"selected by the pattern {pattern!r} in {folder!r}"
    if not names:
        raise ParserContentError(
            f"There is no image file {where}. Files are recognized by "
            f"their extensions: TIFF files need tifffile and other images "
            f"need Pillow."
        )
    return sorted(names)


def _numbered(names: tx.Sequence[str]) -> tx.List[tx.Tuple[int, str]]:
    """Return the number that each name carries, sorted by number.

    The names must differ in one run of digits only, which is the number
    of the file. Numbers are compared as integers, so that `"2"` comes
    before `"10"`, and zero padding does not matter. A number may be
    missing, which leaves a slice without a file, but a number may not be
    repeated. A single name needs no number, and its number is its last
    run of digits, or 0.

    Raises
    ------
    ParserContentError
        If the names differ in something other than one number, or if a
        number is repeated.
    """
    split = [re.split(r"(\d+)", name) for name in names]
    first = split[0]
    for name, parts in zip(names, split):
        same = len(parts) == len(first) and all(
            a == b for a, b in zip(parts[0::2], first[0::2])
        )
        if not same:
            raise ParserContentError(
                f"The names {names[0]!r} and {name!r} do not follow one "
                f"numbering pattern. The files of a stack differ only "
                f"in one number, as in 'slice_001.tif' and "
                f"'slice_002.tif'. Select them with pattern=."
            )
    varying = [
        i
        for i in range(1, len(first), 2)
        if len({int(parts[i]) for parts in split}) > 1
    ]
    if len(varying) > 1:
        raise ParserContentError(
            f"The names of the files differ in {len(varying)} numbers, as "
            f"{names[0]!r} and {names[-1]!r} do, so the axis along which "
            f"they are stacked is ambiguous. Select the files of one "
            f"stack with pattern=, such as 't01_z*.tif'."
        )
    if varying:
        index = varying[0]
    elif len(first) > 1:
        index = len(first) - 2
    else:
        index = None
    numbered = sorted(
        (0 if index is None else int(parts[index]), name)
        for parts, name in zip(split, names)
    )
    numbers = [number for number, _ in numbered]
    repeated = sorted({n for n in numbers if numbers.count(n) > 1})
    if repeated:
        clashing = [name for number, name in numbered if number in repeated]
        raise ParserContentError(
            f"Several files carry the same number: "
            f"{', '.join(repr(n) for n in clashing[:_LISTED])}."
        )
    return numbered


def _read_member(folder: str, name: str, number: int) -> StackMember:
    """Read the header of one file of a stack.

    Raises
    ------
    ParserContentError
        If the library that reads the file is not installed, if the file
        cannot be read, or if it is not a single two-dimensional image.
    """
    filename = os.path.join(folder, *name.split("/"))
    fmt = member_format(name)
    if fmt == "tiff":
        return _read_tiff_member(filename, name, number)
    if fmt == "pillow":
        return _read_pillow_member(filename, name, number)
    raise ParserContentError(
        f"The file {name!r} is not a TIFF file nor an image that Pillow "
        f"reads, judged by its extension (TIFF files need tifffile and "
        f"other images need Pillow)."
    )


def _read_tiff_member(filename: str, name: str, number: int) -> StackMember:
    """Read the header of a TIFF file of a stack with tifffile."""
    # The TIFF package registers its formats when it is imported, which
    # happens only when tifffile is installed, and a TIFF member exists
    # only then.
    from brainhops.io.images.tiff import _utils as tiff

    with tiff.TiffSource(filename=filename).open() as tif:
        if len(tif.series) != 1:
            raise ParserContentError(
                f"The TIFF file {name!r} holds {len(tif.series)} series, "
                f"but a file of a stack holds one image."
            )
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(n) for n in series.shape)
        dtype = np.dtype(series.dtype).newbyteorder("=")
        header = tiff.series_metadata(tif, 0)
    _check_axes(name, axes)
    return StackMember(
        name=name,
        number=number,
        format="tiff",
        axes=axes,
        shape=shape,
        dtype=str(dtype),
        header=header,
    )


def _read_pillow_member(filename: str, name: str, number: int) -> StackMember:
    """Read the header of a raster file of a stack with Pillow."""
    from brainhops.io.images.pillow._utils import read_pillow_header

    header = read_pillow_header(filename)
    if header.n_frames != 1:
        raise ParserContentError(
            f"The file {name!r} holds {header.n_frames} frames, but a file "
            f"of a stack holds one image. Read it on its own with "
            f"`frame=`."
        )
    return StackMember(
        name=name,
        number=number,
        format="pillow",
        axes=header.axes,
        shape=header.shape,
        dtype=header.dtype,
        header=header,
    )


def _check_axes(name: str, axes: str) -> None:
    """Check that a file holds one plane, with at most one channel axis.

    Raises
    ------
    ParserContentError
        If the axes are not `Y` and `X` with at most one channel axis.
    """
    rest = axes.replace("Y", "", 1).replace("X", "", 1)
    if len(axes) - len(rest) == 2 and (rest == "" or rest in _CHANNELS):
        return
    raise ParserContentError(
        f"The file {name!r} has the axes {axes!r}, but a file of a stack "
        f"holds one plane (Y and X), with at most one channel axis (S or "
        f"C). Read a volume or a hyperstack on its own."
    )


def _check_alike(members: tx.Sequence[StackMember]) -> None:
    """Check that the files of a stack can be stacked.

    Raises
    ------
    ParserContentError
        If two files differ in format, shape, type or number of channels.
    """
    first = members[0]
    for member in members[1:]:
        for what, mine, theirs in (
            ("format", member.format, first.format),
            ("shape and channels", member.plane_shape, first.plane_shape),
            ("type", member.dtype, first.dtype),
        ):
            if mine != theirs:
                raise ParserContentError(
                    f"The files of a stack must have the same {what}, "
                    f"but {first.name!r} has {theirs!r} and {member.name!r} "
                    f"has {mine!r}."
                )


_Length = tx.Tuple[float, str]


class _Recorded(tx.NamedTuple):
    """The geometry on which every file of a stack agrees."""

    pixel_size: tx.Optional[tx.Tuple[float, float]]
    pixel_unit: tx.Optional[str]
    spacing: tx.Optional[_Length]


def _recorded_geometry(members: tx.Sequence[StackMember]) -> _Recorded:
    """Return the size of the pixels and the spacing that every file records.

    A size is compared with its sign and its unit, so that files whose
    slices run in opposite directions disagree. A size that some files do
    not record, or on which they disagree, is unknown, with a warning when
    some files record it.
    """
    per_file = [_file_geometry(member) for member in members]
    pixels = [pixel for pixel, _ in per_file]
    steps = [step for _, step in per_file]
    found = []
    for what, values in (("pixel size", pixels), ("spacing", steps)):
        if all(value is None for value in values):
            found.append(None)
        elif any(value != values[0] for value in values):
            warnings.warn(
                f"The files of the stack do not all record the same "
                f"{what}, so it is unknown. Pass {what.replace(' ', '_')}= "
                f"to set it.",
                stacklevel=4,
            )
            found.append(None)
        else:
            found.append(values[0])
    pixel, step = found
    if pixel is None:
        return _Recorded(None, None, step)
    return _Recorded(pixel[:2], pixel[2], step)


def _file_geometry(
    member: StackMember,
) -> tx.Tuple[tx.Optional[tx.Tuple[float, float, str]], tx.Optional[_Length]]:
    """Return the size of the pixels and the spacing that one file records.

    The size of the pixels along the columns and the rows is the size
    along x and y that the TIFF reader takes from the file
    (`series_metadata` in `io/images/tiff/_utils.py`), converted to the
    unit along x. The spacing is the `PhysicalSizeZ` of the OME-XML,
    which the TIFF writer records for a single slice of a volume, or else
    the ImageJ `spacing`, read as the TIFF reader reads it for ImageJ
    hyperstacks, with its sign. A Pillow file records no size that the
    Pillow reader uses by default.
    """
    if member.format != "tiff" or member.header is None:
        return None, None
    # A TIFF member exists only when tifffile is installed, so the TIFF
    # package, which registers its formats when imported, can be imported.
    from brainhops.io.images.tiff import _utils as tiff

    header = member.header
    pixel = None
    if "x" in header.scales and "y" in header.scales:
        (x, unit), (y, yunit) = header.scales["x"], header.scales["y"]
        pixel = (x, raster.convert_length(y, yunit, unit), unit)
    step = None
    if header.ome_xml:
        _, pixels = tiff.ome_image(header.ome_xml, header.ome_index)
        if pixels is not None:
            step = tiff.ome_scales(pixels, ("z",))[0].get("z")
    if step is None:
        scales, signs = tiff.imagej_scales(header.imagej, None, ("z",))
        if "z" in scales:
            size, unit = scales["z"]
            step = (size * signs.get("z", 1.0), unit)
    return pixel, step


def _geometry_options(
    recorded: _Recorded, pixel_size: tx.Any, spacing: tx.Any, unit: tx.Any
) -> tx.Tuple[
    tx.Optional[tx.Tuple[float, float]], tx.Optional[float], tx.Optional[str]
]:
    """Combine the geometry that the files record with the options.

    The unit of the stack is `unit`, or else the unit of the pixel size
    that the files record, or else the unit of their spacing. A size that
    an option gives is in that unit, and a size that the files record is
    converted to it.

    Raises
    ------
    ValueError
        If a size is not a finite number, if a pixel size is not positive
        or the spacing is zero, or if `unit` is not a unit of length.
    """
    if unit is not None:
        length = str(unit)
        try:
            raster.convert_length(1.0, length, length)
        except ValueError:
            raise ValueError(
                f"unit must be a unit of length, not {unit!r}."
            ) from None
    elif recorded.pixel_size is not None:
        length = recorded.pixel_unit
    elif recorded.spacing is not None:
        length = recorded.spacing[1]
    else:
        length = None

    if pixel_size is not None:
        values = [float(value) for value in np.ravel(pixel_size)]
        if len(values) == 1:
            values = values * 2
        if len(values) != 2 or not all(
            math.isfinite(v) and v > 0 for v in values
        ):
            raise ValueError(
                f"pixel_size must be one positive size, or a pair of them "
                f"for the columns and the rows of the files, not "
                f"{pixel_size!r}."
            )
        sizes = (values[0], values[1])
    elif recorded.pixel_size is not None:
        sizes = tuple(
            _converted(v, recorded.pixel_unit, length)
            for v in recorded.pixel_size
        )
    else:
        sizes = None

    if spacing is not None:
        try:
            step = float(spacing)
        except (TypeError, ValueError):
            step = float("nan")
        if not math.isfinite(step) or step == 0:
            raise ValueError(
                f"spacing must be a finite number other than zero, not "
                f"{spacing!r}."
            )
    elif recorded.spacing is not None:
        step = _converted(*recorded.spacing, length)
    else:
        step = None
    return sizes, step, length


def _converted(value: float, unit: str, target: tx.Optional[str]) -> float:
    """Convert a length to another unit, or keep it without a target."""
    if target is None or target == unit:
        return float(value)
    return raster.convert_length(float(value), unit, target)


NOT_A_STREAM = (
    "An image stack is a folder of files, so it is read from the path "
    "of the folder and not from a stream or from bytes."
)


def sniff_score(reason: tx.Optional[str], error: tx.Any) -> float:
    """Turn the reason why a path is not a stack into a confidence.

    Raises
    ------
    SnifferContentError
        If there is a reason and `error` is `True`, or the class given as
        `error`.
    """
    if reason is None:
        return Confidence.LIKELY
    if error:
        if error is True:
            error = SnifferContentError
        raise error(reason)
    return Confidence.NO


def refuse_stream(error: tx.Any) -> float:
    """Refuse a stream or bytes, which never hold a stack.

    Raises
    ------
    SnifferTypeError
        If `error` is `True`, or the class given as `error`.
    """
    if error:
        if error is True:
            error = SnifferTypeError
        raise error(NOT_A_STREAM)
    return Confidence.NO


def refuse_options(cls: type, options: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader does not take.

    Raises
    ------
    TypeError
        If `options` is not empty.
    """
    if options:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(options))}."
        )
