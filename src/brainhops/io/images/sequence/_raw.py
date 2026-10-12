"""The record of an image sequence: the list of its files and their
headers."""

# stdlib
import os
import pathlib
import re

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

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

_ZARR_MARKERS = (".zarray", ".zgroup", ".zattrs", "zarr.json")
"""Files whose presence makes a folder a Zarr store, which is never read as
a sequence."""

_CHANNELS = "SC"
"""The storage codes of a channel axis, as tifffile spells them: samples
per pixel (`S`) or channels (`C`)."""

_LISTED = 5
"""The number of names or numbers that an error message lists at most."""


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


class SequenceMember(Magic, frozen=True):
    """One file of an image sequence, and what its header says.

    A member is a two-dimensional image, possibly with several channels,
    stored in a TIFF file or in a raster format that Pillow reads, such as
    PNG. The member records the name of the file, the number that the name
    carries, and the header that the reader of that format parses. TIFF
    and Pillow images have no record class yet, so the header is the
    structure that their readers already parse: a `TiffMetadata` from
    `brainhops.io.images.tiff._utils` for a TIFF file, and a
    `PillowHeader` from `brainhops.io.images.pillow._utils` for any other
    file.
    """

    name: str
    """The path of the file relative to the folder, with `/` separators."""

    number: int
    """The number that the name of the file carries."""

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
    """The header that the reader of the format parses.

    It is a `TiffMetadata` for a TIFF file, which holds its tags, its
    OME-XML and its ImageJ metadata, and a `PillowHeader` for any other
    file, which holds its mode and the `info` dictionary of Pillow.
    """

    @property
    def plane_axes(self) -> str:
        """The axes in which the sequence presents the file: `"YX"`, or
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


class SequenceRaw(Magic, frozen=True):
    """The record of an image sequence, which lists its files.

    An image sequence is a folder of numbered two-dimensional images, such
    as `slice_000.tif`, `slice_001.tif` and so on, which together form one
    volume. The folder has no header of its own, so the record is the
    list of its members, sorted by the number that their names carry, with
    the header of each file. Every member has the same format, shape,
    type and number of channels, and their numbers follow each other
    without a gap. The record never reads pixels.

    The record is held by
    [`SequenceMetadata`][brainhops.io.images.sequence.SequenceMetadata].
    It is frozen, and two records are equal when their fields are.
    """

    members: tx.Tuple[SequenceMember, ...] = ()
    """The files of the sequence, sorted by their numbers."""

    pattern: tx.Optional[str] = None
    """The glob pattern that selected the files in the folder, or `None`
    when every image file of the folder was taken."""

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
    def axes(self) -> str:
        """The axes of the stored volume, slowest first: `"ZYX"`, or
        `"ZYXS"` for files with channels.

        The files are stacked along `Z`, the slowest axis, so the volume is
        stored as the files are listed, one after the other.
        """
        if not self.members:
            return "ZYX"
        return "Z" + self.members[0].plane_axes

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stored volume, in the order of `axes`."""
        if not self.members:
            return (0, 0, 0)
        return (len(self.members),) + self.members[0].plane_shape

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the pixels."""
        if not self.members:
            return np.dtype("float64")
        return np.dtype(self.members[0].dtype)

    def copy(self) -> "SequenceRaw":
        """Return a copy of the record.

        The record is frozen and its fields are tuples, so the copy shares
        its members.

        Returns
        -------
        SequenceRaw
            The copy.
        """
        return replace(self)

    @classmethod
    def from_folder(
        cls,
        folder: path.FilenameLike,
        pattern: tx.Optional[str] = None,
    ) -> "SequenceRaw":
        """Read the record of the image sequence that a folder holds.

        The files are listed, their names are checked and sorted by the
        number that they carry, and the header of each file is read. The
        pixels are not read.

        Parameters
        ----------
        folder : path-like
            The path of a local folder.
        pattern : str, optional
            A glob pattern, relative to the folder, that selects the files
            of the sequence, such as `"slice_*.tif"`. By default, every
            file of the folder whose extension is that of a TIFF file or
            of a format that Pillow reads is taken. Hidden files, whose
            names start with a dot, are never taken.

        Returns
        -------
        SequenceRaw
            The record.

        Raises
        ------
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the path is not a local folder, if it holds no image file,
            if the names of the files do not follow one numbering, if a
            number is missing or repeated, or if the files differ in
            format, shape, type or number of channels.
        """
        local = _local_folder(folder)
        numbered = _numbered(_listed(local, pattern))
        members = tuple(
            _read_member(local, name, number) for number, name in numbered
        )
        _check_alike(members)
        return cls(members=members, pattern=pattern)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def sniff_folder(
    folder: tx.Any, pattern: tx.Optional[str] = None
) -> tx.Optional[str]:
    """Tell why a path cannot be an image sequence, or `None` if it can.

    Only the names of the files are checked, so no file is opened. A path
    can hold a sequence if it is a local folder that is not a Zarr store,
    and if the files that `pattern` selects are image files whose names
    follow one numbering without a gap.

    Parameters
    ----------
    folder : path-like
        The path to check.
    pattern : str, optional
        The glob pattern that selects the files of the sequence.

    Returns
    -------
    str or None
        The reason why the path is not a sequence, or `None`.
    """
    try:
        local = _local_folder(folder)
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
    lower = name.lower()
    if HAS_TIFFFILE:
        # The TIFF package registers its formats when it is imported, so
        # it is imported only when tifffile is installed, as
        # `brainhops.io.images` does.
        from brainhops.io.images.tiff._utils import EXTENSIONS

        if lower.endswith(EXTENSIONS):
            return "tiff"
    if HAS_PILLOW:
        from brainhops.io.images.pillow._utils import EXTENSIONS

        if lower.endswith(EXTENSIONS):
            return "pillow"
    return None


def _local_folder(folder: tx.Any) -> str:
    """Return the local path of a folder.

    Raises
    ------
    ParserExistsError
        If the path does not exist.
    ParserContentError
        If the path is not local, or is not a folder.
    """
    if isinstance(folder, str):
        folder = path.Path(folder)
    if not isinstance(folder, (str, os.PathLike)):
        raise ParserContentError(
            f"An image sequence is read from the path of a folder, not from "
            f"an object of type {type(folder).__name__}."
        )
    if not is_local(folder):
        raise ParserContentError(
            f"Image sequences are read from local folders only, and "
            f"{str(folder)!r} is not local."
        )
    local = os.fspath(folder)
    if not os.path.exists(local):
        raise ParserExistsError(f"No such folder: {local}")
    if not os.path.isdir(local):
        raise ParserContentError(
            f"An image sequence is a folder, and {local!r} is a file."
        )
    return local


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
    before `"10"`, and zero padding does not matter. The numbers must
    follow each other without a gap or a repetition. A single name needs
    no number, and its number is its last run of digits, or 0.

    Raises
    ------
    ParserContentError
        If the names differ in something other than one number, or if a
        number is repeated or missing.
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
                f"numbering pattern. The files of a sequence differ only "
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
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    if missing:
        shown = ", ".join(str(n) for n in missing[:_LISTED])
        more = " and others" if len(missing) > _LISTED else ""
        raise ParserContentError(
            f"The sequence numbered from {numbers[0]} to {numbers[-1]} "
            f"misses {len(missing)} file(s), numbered {shown}{more}. A "
            f"sequence cannot have gaps, since the files would no longer "
            f"be evenly spaced."
        )
    return numbered


def _read_member(folder: str, name: str, number: int) -> SequenceMember:
    """Read the header of one file of a sequence.

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
    return _read_pillow_member(filename, name, number)


def _read_tiff_member(filename: str, name: str, number: int) -> SequenceMember:
    """Read the header of a TIFF file of a sequence with tifffile."""
    if not HAS_TIFFFILE:
        raise ParserContentError(
            f"Reading {name!r} needs tifffile, which is not installed. "
            f"Install it with `pip install brainhops[tiff]`."
        )
    # The TIFF package registers its formats when it is imported, which
    # happens only when tifffile is installed.
    from brainhops.io.images.tiff import _utils as tiff

    with tiff.TiffSource(filename=filename).open() as tif:
        if len(tif.series) != 1:
            raise ParserContentError(
                f"The TIFF file {name!r} holds {len(tif.series)} series, "
                f"but a file of a sequence holds one image."
            )
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(n) for n in series.shape)
        dtype = np.dtype(series.dtype).newbyteorder("=")
        header = tiff.series_metadata(tif, 0)
    _check_axes(name, axes)
    return SequenceMember(
        name=name,
        number=number,
        format="tiff",
        axes=axes,
        shape=shape,
        dtype=str(dtype),
        header=header,
    )


def _read_pillow_member(
    filename: str, name: str, number: int
) -> SequenceMember:
    """Read the header of a raster file of a sequence with Pillow."""
    if not HAS_PILLOW:
        raise ParserContentError(
            f"Reading {name!r} needs Pillow, which is not installed. "
            f"Install it with `pip install brainhops[pillow]`."
        )
    from brainhops.io.images.pillow._utils import read_pillow_header

    header = read_pillow_header(filename)
    if header.n_frames != 1:
        raise ParserContentError(
            f"The file {name!r} holds {header.n_frames} frames, but a file "
            f"of a sequence holds one image. Read it on its own with "
            f"`frame=`."
        )
    return SequenceMember(
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
        f"The file {name!r} has the axes {axes!r}, but a file of a sequence "
        f"holds one plane (Y and X), with at most one channel axis (S or "
        f"C). Read a stack or a hyperstack on its own."
    )


def _check_alike(members: tx.Sequence[SequenceMember]) -> None:
    """Check that the files of a sequence can be stacked.

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
                    f"The files of a sequence must have the same {what}, "
                    f"but {first.name!r} has {theirs!r} and {member.name!r} "
                    f"has {mine!r}."
                )


NOT_A_STREAM = (
    "An image sequence is a folder of files, so it is read from the path "
    "of the folder and not from a stream or from bytes."
)


def sniff_score(reason: tx.Optional[str], error: tx.Any) -> float:
    """Turn the reason why a path is not a sequence into a confidence.

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
    """Refuse a stream or bytes, which never hold a sequence.

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
