"""How an image stack is written: the names of its files, their content,
and the folder that holds them."""

# stdlib
import glob
import os
import re
import shutil
import uuid

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base.parsers import ParserContentError, WriterError
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._data import _StackProxy
from ._geometry import stored_geometry
from ._raw import (
    StackMember,
    StackRaw,
    image_extension,
    local_name,
    member_format,
)

DEFAULT_NAME = ("slice_", ".tif")
"""The prefix and the suffix of the names of the files of a new stack,
which are TIFF files named `slice_0.tif`, `slice_1.tif` and so on."""


class Written(tx.NamedTuple):
    """The files that a writer wrote, which the image may read again."""

    folder: str
    """The folder that holds the files."""

    numbered: tx.List[tx.Tuple[int, str]]
    """The number and the name, relative to `folder`, of each file."""

    pattern: tx.Optional[str]
    """A glob pattern that selects the files in `folder`."""

    copied: bool
    """Whether the files are copies of the files of the lazy array."""


class Target(tx.NamedTuple):
    """Where a stack is written, as the path given to the writer says."""

    folder: str
    """The local folder that holds the files."""

    template: tx.Optional[tx.Tuple[str, str]]
    """The prefix and the suffix of the names of the files, or `None` when
    the files keep the names of the record and `folder` is replaced."""

    def names(self, record: StackRaw) -> tx.List[str]:
        """Return the name of the file of each member of a record.

        Parameters
        ----------
        record : StackRaw
            The record of the stack to write.

        Returns
        -------
        list[str]
            The names, relative to `folder`, in the order of the members.
        """
        if self.template is None:
            return list(record.names)
        numbers = [member.number for member in record.members]
        return numbered_names(*self.template, numbers)

    def paths(self, record: StackRaw) -> tx.List[str]:
        """Return the path of the file of each member of a record.

        Parameters
        ----------
        record : StackRaw
            The record of the stack to write.

        Returns
        -------
        list[str]
            The paths, in the order of the members.
        """
        return [
            os.path.join(self.folder, *name.split("/"))
            for name in self.names(record)
        ]


def stack_target(filename: tx.Any) -> Target:
    """Tell where the files of a stack go, from the path given to a writer.

    A path whose name ends with the extension of a TIFF file or of a
    format that Pillow writes, such as `out.tif`, gives the names of the
    files: the number of each slice is appended to the name, after an
    underscore, so that the files are `out_0.tif`, `out_1.tif` and so on,
    in the folder of the path. Any other path is the folder of the stack,
    and the files keep the names that the record gives them.

    Raises
    ------
    WriterError
        If the path is not a local path.
    """
    try:
        local = local_name(filename)
    except ParserContentError as e:
        raise WriterError(str(e)) from None
    local = os.path.abspath(local)
    directory, name = os.path.split(local)
    extension, fmt = image_extension(name)
    if fmt is None or extension == name:
        return Target(folder=local, template=None)
    return Target(
        folder=directory, template=(name[: -len(extension)] + "_", extension)
    )


def numbered_names(
    prefix: str, suffix: str, numbers: tx.Sequence[int]
) -> tx.List[str]:
    """Return the names of numbered files.

    The numbers are padded with zeros to the number of digits of the
    largest one, so that the names sort in the order of the numbers.

    Parameters
    ----------
    prefix : str
        The part of the names before the number.
    suffix : str
        The part of the names after the number.
    numbers : Sequence[int]
        The numbers.

    Returns
    -------
    list[str]
        The names.
    """
    width = len(str(max(numbers))) if numbers else 1
    return [f"{prefix}{n:0{width}d}{suffix}" for n in numbers]


def name_parts(member: StackMember) -> tx.Tuple[str, str]:
    """Return what comes before and after the number in the name of a file.

    The number is the last run of digits whose value is the number of the
    member, and a name without one gives the name and an empty suffix.
    """
    parts = re.split(r"(\d+)", member.name)
    for i in range(len(parts) - 2, 0, -2):
        if int(parts[i]) == member.number:
            return "".join(parts[:i]), "".join(parts[i + 1 :])
    return member.name, ""


def reads_from(raw: tx.Any, target: Target, record: StackRaw) -> bool:
    """Tell whether a lazy array reads a file that a write replaces.

    A write to a folder replaces every file in the folder, and a write to
    numbered names replaces the files of those names.

    Parameters
    ----------
    raw : Any
        The `rawdata` of the image to write.
    target : Target
        Where the stack is written.
    record : StackRaw
        The record of the stack to write.

    Returns
    -------
    bool
        Whether `raw` is a lazy array that reads one of those files.
    """
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(raw, _StackProxy):
        return False
    read = {os.path.realpath(name) for name in raw.filenames()}
    if target.template is None:
        folder = os.path.realpath(target.folder)
        return any(
            name == folder or name.startswith(folder + os.sep) for name in read
        )
    return bool(read & {os.path.realpath(p) for p in target.paths(record)})


def write_stack(
    target: Target,
    record: StackRaw,
    raw: tx.Any,
    read: tx.Collection[str] = (),
) -> Written:
    """Write the files of a stack.

    The lazy array of a stack that was read and not changed is copied file
    by file, so that each file keeps its bytes, when the file that it is
    copied to has a name of the same format. Any other slice is encoded by
    the writer of the single-file format of its name, tifffile for a TIFF
    file and Pillow otherwise, with the geometry that the files can record
    (see `stored_geometry`). A slice whose number no member carries is not
    written.

    Every file is written before any file is replaced. A folder is written
    beside the target, then moved in its place, and a file with a numbered
    name is written beside it, under a hidden name, then renamed. A lazy
    array that reads the old files is therefore never left reading a file
    that is half written.

    Parameters
    ----------
    target : Target
        Where the stack is written.
    record : StackRaw
        The record of the stack, consolidated for the write.
    raw : Any
        The stored volume of the stack, `(slices, rows, columns[,
        channels])`.
    read : Collection[str], default=()
        The paths of the files that the lazy array of the image read
        before it was consolidated for the write. A folder that holds some
        of them is replaced only if it holds nothing else.

    Returns
    -------
    Written
        The files that were written.

    Raises
    ------
    WriterError
        If the record has no member, if a file cannot be encoded, or if
        the folder to replace holds files that are not images.
    """
    if not record.members:
        raise WriterError(
            "This stack has no slice, so there is nothing to write."
        )
    names = target.names(record)
    copied = isinstance(raw, _StackProxy) and raw.record is record
    sources = raw.filenames() if copied else [None] * len(names)

    def fill(folder: str) -> None:
        for member, position, name, source in zip(
            record.members, record.positions, names, sources
        ):
            filename = os.path.join(folder, *name.split("/"))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if source is not None and _same_format(member.name, name):
                shutil.copyfile(source, filename)
                continue
            values = np.asarray(raw[position])
            content = _encoded(record, member, values, name)
            with open(filename, "xb") as file:
                file.write(content)

    numbered = list(zip((m.number for m in record.members), names))
    if target.template is None:
        _replace_folder(target.folder, fill, read)
        return Written(target.folder, numbered, record.pattern, copied)
    _replace_files(target.folder, names, fill)
    prefix, suffix = target.template
    pattern = glob.escape(prefix) + "*" + glob.escape(suffix)
    return Written(target.folder, numbered, pattern, copied)


def _same_format(source: str, target: str) -> bool:
    """Tell whether a file can be copied to a name without encoding it.

    Two TIFF names, or two names with the same extension, ignoring case,
    name files of the same format.
    """
    first, fmt = image_extension(source)
    second, other = image_extension(target)
    if fmt != other:
        return False
    return fmt == "tiff" or first.lower() == second.lower()


def _encoded(
    record: StackRaw, member: StackMember, plane: ArrayProtocol, name: str
) -> bytes:
    """Encode one slice, `(rows, columns[, channels])`, as a file.

    The slice becomes an image of the single-file format of the name,
    whose x axis lies along the columns and y axis along the rows, with
    the geometry that the file can record.

    Raises
    ------
    WriterError
        If the writer of the format cannot encode the slice.
    """
    plane = np.asarray(plane)
    channels = plane.ndim == 3
    fmt = member_format(name)
    pixel_size, spacing, unit = stored_geometry(
        fmt, channels, (record.pixel_size, record.spacing, record.unit)
    )
    data = plane.transpose((1, 0, 2) if channels else (1, 0))
    scales: tx.Dict[str, raster.AxisScale] = {}
    if pixel_size is not None:
        scales = {"x": (pixel_size[0], unit), "y": (pixel_size[1], unit)}
    if spacing is not None:
        # A file that records the spacing is a volume of one slice, which
        # tifffile reads as a plane.
        data = data[:, :, None]
        scales["z"] = (spacing, unit)
        codes = "ZYX"
    else:
        codes = "YXS" if channels else "YX"
    axes = raster.storage_axes(codes)
    axes = [axes[p] for p in raster.canonical_permutation(axes)]
    xforms = raster.raster_transformations(axes, scales)
    if fmt == "tiff":
        from brainhops.io.images.tiff import TiffImage

        return TiffImage(data=data, transformations=xforms).to_bytes()
    if fmt == "pillow":
        from brainhops.io.images.pillow import PillowImage
        from brainhops.io.images.pillow._utils import format_for_name

        image = PillowImage(data=data, transformations=xforms)
        return image.to_bytes(format=format_for_name(name), dpi=False)
    raise WriterError(
        f"Cannot write {name!r}: it is not the name of a TIFF file nor of "
        f"an image that Pillow writes (TIFF files need tifffile and other "
        f"images need Pillow)."
    )


def _replace_folder(
    folder: str, fill: tx.Callable[[str], None], read: tx.Collection[str]
) -> None:
    """Write a folder beside its target, then move it in its place.

    The function `fill` receives a new, empty folder beside the target,
    with a hidden name, and fills it. Then the target, if it exists, is
    moved aside, the new folder takes its name, and the old folder is
    removed. A lazy array that opens its files by name then reads the new
    files. An existing folder is replaced only if it holds nothing but
    image files and folders of them, or, when the image reads some of the
    files in `read` from it, nothing but those, so that a write never
    removes another file, such as the files of another channel that a
    pattern left out. This follows `replace_directory` of the pyramid core
    (#451), which is not on the main branch yet, and which this function
    is to be replaced with.

    Raises
    ------
    WriterError
        If the target is a file, or a folder that holds other files.
    """
    target = os.path.realpath(folder)
    if os.path.lexists(target) and not os.path.isdir(target):
        raise WriterError(
            f"Cannot write an image stack to {folder!r}, which is a file."
        )
    if os.path.isdir(target):
        _check_replaceable(target, read)
    parent, name = os.path.split(target)
    os.makedirs(parent, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = os.path.join(parent, f".{token}.{name}")
    old = os.path.join(parent, f".{token}.old.{name}")
    os.mkdir(temporary)
    try:
        fill(temporary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if not os.path.lexists(target):
        os.replace(temporary, target)
        return
    os.replace(target, old)
    try:
        os.replace(temporary, target)
    except BaseException:
        os.replace(old, target)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    shutil.rmtree(old)


def _check_replaceable(folder: str, read: tx.Collection[str]) -> None:
    """Check that a folder holds nothing but the files of a stack.

    The folder may hold image files and folders of them. When it holds
    some of the files in `read`, it may hold nothing else.

    Raises
    ------
    WriterError
        If the folder holds another file, hidden files included.
    """
    read = {os.path.realpath(name) for name in read}
    owned = any(name.startswith(folder + os.sep) for name in read)
    for root, folders, files in os.walk(folder):
        # The names are sorted, so that the error names the same file each
        # time.
        folders.sort()
        for name in sorted(files):
            filename = os.path.realpath(os.path.join(root, name))
            if owned:
                if filename in read:
                    continue
                why = "which the image does not read"
            elif name.startswith(".") or member_format(name) is None:
                why = "which is not an image file"
            else:
                continue
            found = os.path.relpath(filename, folder)
            raise WriterError(
                f"Cannot replace the folder {folder!r} with an image "
                f"stack, because it holds {found!r}, {why}. Remove that "
                f"file or write to another folder."
            )


def _replace_files(
    folder: str, names: tx.Sequence[str], fill: tx.Callable[[str], None]
) -> None:
    """Write numbered files beside their targets, then rename them.

    The function `fill` receives a new, empty folder beside the files,
    with a hidden name, and fills it with files of the given names, which
    are then moved to the folder of the targets one by one. The files of
    the folder that the names do not give are left as they are.

    Raises
    ------
    OSError
        If a file cannot be written or moved.
    """
    os.makedirs(folder, exist_ok=True)
    temporary = os.path.join(folder, f".{uuid.uuid4().hex}.stack")
    os.mkdir(temporary)
    try:
        fill(temporary)
        for name in names:
            os.replace(
                os.path.join(temporary, name), os.path.join(folder, name)
            )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
