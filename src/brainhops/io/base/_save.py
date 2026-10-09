"""Choice of the format in which an object is written.

Reading chooses a format from the content and the name of a file. Writing has
no content to inspect, so the format is chosen from the file name, which
selects the candidates, and from the object, which decides the candidate that
can hold it.
"""

__all__ = ["save"]

import inspect

import typing_extensions as tx
from bagof.magic import fields

from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import FileBasedObject, Format
from brainhops.io.base._dispatch import _match_name, _tiers, _to_filename
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    FileReader,
    FileWriter,
    WriterError,
)


def save(obj: tx.Any, file: path.FileLike, **kwargs) -> None:
    """Write an object to a file in the format that the file name calls for.

    The format is chosen from the registered writable formats, in four
    steps.

    1. **The file name.** The formats that declare the longest of the
       extensions the name ends with are the candidates, so `a.ome.zarr`
       asks for OME-Zarr and `a.zarr` for any Zarr format. A format that
       requires a prefix is a candidate only when the name has one of
       its prefixes.
    2. **The object's own format.** If `obj` already is of one of the
       candidates, it is written as it is.
    3. **A format that can hold the object as it is.** Otherwise, a
       candidate can hold `obj` if it is a file-backed version of the
       very data model `obj` is an instance of, and takes every field
       that data model has (`NiftiImage` and `ZarrImage` are file-backed
       `SingleScaleImage`s). `obj` is converted to it, which neither
       loses nor changes anything.
    4. **A format a transformation converts to.** Otherwise, a
       transformation is converted to the candidates, with the same
       converters as `obj.to(Format)`, and written in the one it
       converts to. A converter returns the very map `obj` is, with
       its endpoints bridged to the format's (an affine to LPS is
       flipped into RAS), or refuses: a general `Affine` is written as
       the voxel-to-RAS affine of a NIfTI file only when that is what it
       maps, or when its coordinate systems are not known.

    In steps 3 and 4, when several candidates can hold `obj`, the most
    specific is used, by the rules reading uses: the longest prefix,
    then the narrowest declaration, then `PRIORITY`. Two that are
    equally specific are an ambiguity, and nothing is written.

    !!! note "Nothing is approximated"
        A transformation is written only in a format that holds it
        exactly. One that no candidate holds -- a field interpolated
        with cubic splines, to be written as NIfTI's linearly
        interpolated values, say -- is refused with each candidate's
        reason, rather than resampled.

    !!! note "Two passes, one conversion"
        Steps 3 and 4 are two passes only until every writable
        transformation format has converters. Until then, step 3 is what
        writes a transformation to a format without them, by copying it
        into a file-backed version of its data model. For the formats
        that have converters (`NiftiVoxelToRAS`, `NiftiRASToVoxel`,
        `NiftiRASDisplacementField`, `NiftiRASCoordinatesField`,
        `SpmCoordinatesField`), `from_instance` is `obj.to(Format)`, so
        both passes run the same conversion.

    Parameters
    ----------
    obj : object
        The object to write.
    file : FileLike
        A path, or a file object opened for writing. An object without a name,
        such as a buffer, gives no name to choose from, so `obj` must already
        be a writable format.
    **kwargs : Any
        Options passed to the `save` method of the chosen format.

    Raises
    ------
    AmbiguousFormatError
        If several equally specific formats can hold `obj`.
    WriterError
        If no format claims the file name, if none of those that do can hold
        `obj`, or if the chosen format cannot write it.
    """
    name = _to_filename(file)
    if name is None:
        if isinstance(obj, FileBasedObject) and isinstance(obj, FileWriter):
            obj.save(file, **kwargs)
            return
        raise WriterError(
            f"Cannot choose a format to write {type(obj).__name__} in: the "
            f"file has no name to choose it from. Give the path of the "
            f"file, or build the format you want and save that."
        )

    registry = (
        fmt for fmt in FileBasedObject._REGISTRY if issubclass(fmt, FileWriter)
    )
    matches = ((fmt, _match_name(name, fmt)) for fmt in registry)
    claimed = [(fmt, match) for fmt, match in matches if match is not None]
    if not claimed:
        raise WriterError(
            f"Cannot write {name!r}: no writable format is registered for "
            f"its extension."
        )
    # The longest extension decides: .ome.zarr is not written as plain .zarr
    # when no OME-Zarr format can hold the object.
    longest = max(extension for _, (extension, _) in claimed)
    claimed = [(fmt, match) for fmt, match in claimed if match[0] == longest]

    # Rebuilding the object would rerun its constructor and lose state that is
    # not a field, such as data already read.
    if any(isinstance(obj, fmt) for fmt, _ in claimed):
        obj.save(file, **kwargs)
        return

    reasons: tx.List[str] = []
    holders = [
        (fmt, match) for fmt, match in claimed if _holds(fmt, obj, reasons)
    ]
    writer = _first_writer(name, obj, holders, _copy, reasons)
    # The second pass converts a transformation (see "Two passes, one
    # conversion" above): the first stays for the formats without
    # converters, and for those with them it runs the same conversion.
    if writer is None and isinstance(obj, Transformation):
        others = [
            candidate for candidate in claimed if candidate not in holders
        ]
        writer = _first_writer(name, obj, others, _convert, reasons)
    if writer is not None:
        writer.save(file, **kwargs)
        return

    formats = ", ".join(sorted(fmt.__name__ for fmt, _ in claimed))
    detail = "".join(f"\n  - {reason}" for reason in reasons)
    raise WriterError(
        f"Cannot write {type(obj).__name__} to {name!r}: none of the "
        f"formats registered for it ({formats}) can hold it as it is. "
        f"Build the format you want with its `from_any` and save "
        f"that.{detail}"
    )


def _copy(obj: tx.Any, fmt: type) -> tx.Any:
    """`obj` copied into `fmt`, a file-backed version of its data model."""
    return fmt.from_instance(obj)


def _convert(obj: Transformation, fmt: type) -> tx.Any:
    """`obj` converted to `fmt`, as `obj.to(fmt)` converts it."""
    return obj.to(fmt)


def _first_writer(
    name: str,
    obj: tx.Any,
    candidates: tx.List[tx.Tuple[type, tx.Any]],
    build: tx.Callable[[tx.Any, type], tx.Any],
    reasons: tx.List[str],
) -> tx.Any:
    """
    `obj` built into the most specific candidate format that takes it.

    The candidates are tried tier by tier, most specific first, and
    `build(obj, fmt)` makes the object to write. The first tier where it
    succeeds for one format gives the writer; a tier where it succeeds
    for several is an ambiguity. Why it fails for the others is added to
    `reasons`. `None` when it fails for every candidate.
    """
    for tier in _tiers(candidates):
        writers = []
        for fmt in tier:
            try:
                writers.append((fmt, build(obj, fmt)))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
        if len(writers) > 1:
            raise AmbiguousFormatError(
                _ambiguity_message(name, obj, [fmt for fmt, _ in writers])
            )
        if writers:
            return writers[0][1]
    return None


def _model(cls: type) -> tx.Optional[type]:
    """Return the data model that `cls` instantiates, or `None`.

    This is the first class in the MRO that is a data model but not I/O,
    such as `SingleScaleImage` for `NiftiImage`.
    """
    for base in cls.__mro__:
        if issubclass(base, DataModelBase) and not issubclass(
            base, (Format, FileReader, FileWriter)
        ):
            return base
    return None


def _init_fields(cls: type) -> tx.Set[str]:
    return {field.name for field in fields(cls) if field.init}


def _holds(fmt: type, obj: tx.Any, reasons: tx.List[str]) -> bool:
    """Whether `obj` converts to `fmt` without loss.

    The format must be a file-backed version of the very data model that `obj`
    instantiates, and take every field of that model. `NiftiVoxelToRAS` is an
    `Affine` that means voxel-to-RAS, so a general `Affine` written through it
    would come back with another meaning. A format that stores its content
    differently, such as OME-Zarr keeping the arrays it read rather than a list
    of scales, would be built with content missing.
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


# _describe and _ambiguity_message mirror the messages of parse and sniff
# so that the three could share a helper. save has no hint=, so a format is
# chosen by converting the object.


def _describe(cls: type) -> str:
    """Describe a format by the first paragraph of its own docstring."""
    doc = cls.__dict__.get("__doc__")
    if not isinstance(doc, str) or not doc.strip():
        return ""
    paragraph = inspect.cleandoc(doc).split("\n\n", 1)[0]
    lines = paragraph.splitlines()
    if len(lines) > 1 and set(lines[1].strip()) == {"-"}:
        return ""
    return " ".join(paragraph.split()).rstrip(".")


def _ambiguity_message(
    name: str, obj: tx.Any, candidates: tx.Iterable[type]
) -> str:
    """Explain which formats an object could be written in.

    Each candidate is listed with the call that writes the object in it.
    """
    candidates = sorted(candidates, key=lambda cls: cls.__name__)
    lines = [
        f"Cannot tell which format to write {name!r} in: this "
        f"{type(obj).__name__} can be written equally well in any of "
        f"these {len(candidates)} formats, which would give different "
        f"results."
    ]
    for cls in candidates:
        line = f"  - {cls.__name__}"
        about = _describe(cls)
        if about:
            line += f" ({about})"
        line += f": `{cls.__name__}.from_any(obj).save(path)`"
        lines.append(line)
    lines.append(
        f"Choose one by converting the object to that format and saving "
        f"it, as in `{candidates[0].__name__}.from_any(obj).save(path)`."
    )
    return "\n".join(lines)
