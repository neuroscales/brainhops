"""
Format dispatch for writing: choosing which writable format a file gets.

Reading guesses the format from the content and the file name. Writing
has no content to look at, so it guesses from the file name and from the
object being written. The name says which formats are candidates, with
the same specificity rules reading uses, and the object says which of
them can hold it.
"""

__all__ = ["save"]

# dependencies
import typing_extensions as tx
from bagof.magic import fields

# internals
from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._base import WritableFileBasedObject
from brainhops.io.base._dispatch import _match_name, _tiers, file_name
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    FileSniffer,
    WriterError,
)


def save(obj: tx.Any, file: path.FileLike, **kwargs) -> None:
    """
    Write an object to a file, in the format the file name calls for.

    The format is chosen from the registered writable formats, in three
    steps.

    1. **The file name.** The formats that declare the longest of the
       extensions the name ends with are the candidates, so `a.ome.zarr`
       asks for OME-Zarr and `a.zarr` for any Zarr format. A format that
       requires a prefix is a candidate only when the name has one of
       its prefixes.
    2. **The object's own format.** If `obj` already is of one of the
       candidates, it is written as it is.
    3. **A format that can hold the object.** Otherwise, a candidate can
       hold `obj` if it is a file-backed version of the very data model
       `obj` is an instance of, and takes every field that data model
       has (`NiftiImage` and `ZarrImage` are file-backed
       `SingleScaleImage`s). `obj` is converted to it, which neither
       loses nor changes anything. When several can, the most specific
       is used, by the rules reading uses: the longest prefix, then the
       narrowest declaration, then `PRIORITY`. Two that are equally
       specific are an ambiguity, and nothing is written.

    !!! note "No conversion beyond the file format"
        `obj` is never converted to another data model on the way. A
        `Scaling` is not turned into an `Affine`, and a general `Affine`
        is not turned into the voxel-to-RAS affine a NIfTI file holds,
        since it would come back meaning something it did not say. Build
        the format you want when that is what the file should hold:
        `NiftiVoxelToRAS.from_other(affine).save(file)`.

    Parameters
    ----------
    obj : Any
        The object to write.
    file : FileLike
        The file to write to: a path, or a file object opened for
        writing. An unnamed file object, such as a buffer, has no name to
        choose a format from, so `obj` must already be of a writable
        format, and is written in that format.
    **kwargs
        Format-specific options, passed on to the chosen format's `save`.

    Raises
    ------
    AmbiguousFormatError
        If several equally specific formats can hold `obj`.
    WriterError
        If no registered format claims the file name, or none of those
        that do can hold `obj`, or the chosen format cannot write it.
    """
    name = file_name(file)
    if name is None:
        if isinstance(obj, WritableFileBasedObject):
            obj.save(file, **kwargs)
            return
        raise WriterError(
            f"Cannot choose a format to write {type(obj).__name__} in: the "
            f"file has no name to choose it from. Give the path of the "
            f"file, or build the format you want and save that."
        )

    registry = WritableFileBasedObject._REGISTRY
    matches = ((fmt, _match_name(name, fmt)) for fmt in registry)
    claimed = [(fmt, match) for fmt, match in matches if match is not None]
    if not claimed:
        raise WriterError(
            f"Cannot write {name!r}: no writable format is registered for "
            f"its extension."
        )
    # The longest extension decides which formats the name asks for: a
    # name ending in `.ome.zarr` asks for OME-Zarr, and is not written
    # as a plain `.zarr` store when no OME-Zarr format can hold `obj`.
    longest = max(extension for _, (extension, _) in claimed)
    claimed = [(fmt, match) for fmt, match in claimed if match[0] == longest]

    # An object already of a format the name asks for is not rebuilt:
    # that would run its constructor again, and a file-based object keeps
    # state that is not a field, such as the data it has already read.
    if any(isinstance(obj, fmt) for fmt, _ in claimed):
        obj.save(file, **kwargs)
        return

    reasons: tx.List[str] = []
    holders = [
        (fmt, match) for fmt, match in claimed if _holds(fmt, obj, reasons)
    ]
    for tier in _tiers(holders):
        writers = []
        for fmt in tier:
            try:
                writers.append((fmt, fmt.from_instance(obj)))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
        if len(writers) > 1:
            names = ", ".join(sorted(fmt.__name__ for fmt, _ in writers))
            raise AmbiguousFormatError(
                f"Cannot choose a format for {name!r}: {names} can all "
                f"hold {type(obj).__name__}, and nothing tells them "
                f"apart. Give one of them an explicit PRIORITY, or build "
                f"the format you want and save that."
            )
        if writers:
            writers[0][1].save(file, **kwargs)
            return

    formats = ", ".join(sorted(fmt.__name__ for fmt, _ in claimed))
    detail = "".join(f"\n  - {reason}" for reason in reasons)
    raise WriterError(
        f"Cannot write {type(obj).__name__} to {name!r}: none of the "
        f"formats registered for it ({formats}) can hold it as it is. "
        f"Build the format you want with its `from_other` and save "
        f"that.{detail}"
    )


def _model(cls: type) -> tx.Optional[type]:
    """
    The data model that `cls` is an instance of, file formats aside.

    It is the first class in the MRO that is a data model and is not a
    file format: `SingleScaleImage` for `NiftiImage`, and `VoxelToRAS`
    for `NiftiVoxelToRAS`. A class that is not a data model has none.
    """
    for base in cls.__mro__:
        if issubclass(base, DataModelBase) and not issubclass(
            base, FileSniffer
        ):
            return base
    return None


def _init_fields(cls: type) -> tx.Set[str]:
    """The names of the fields that the constructor of `cls` takes."""
    return {field.name for field in fields(cls) if field.init}


def _holds(fmt: type, obj: tx.Any, reasons: tx.List[str]) -> bool:
    """
    Whether `obj` can be converted to `fmt` without losing or changing
    any of it.

    The format must be a file-backed version of the very data model
    `obj` is an instance of, and must take every field of that data
    model.

    - *The very data model*, not a more specific one: `NiftiVoxelToRAS`
      is an `Affine`, but one that means voxel-to-RAS, so a general
      `Affine` written through it would come back meaning something it
      did not say.
    - *Every field*: a format that stores its content differently, such
      as an OME-Zarr field that keeps the arrays it read rather than a
      list of scales, would build itself from `obj` with that content
      missing.
    """
    model = _model(type(obj))
    if model is None or _model(fmt) is not model:
        return False
    missing = _init_fields(model) - _init_fields(fmt)
    if missing:
        reasons.append(
            f"{fmt.__name__} does not take the field(s) "
            f"{', '.join(sorted(missing))} of a {model.__name__}"
        )
        return False
    return True
