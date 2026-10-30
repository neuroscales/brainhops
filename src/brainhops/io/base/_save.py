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
from brainhops._core.errors import (
    UNSET,
    apply_error_policy,
    error_policy,
    policy_action,
)
from brainhops.datamodel._internal_api import (
    build_pyramid,
    pop_pyramid_options,
)
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata import (
    ConversionReport,
    metadata_loss_policy,
)
from brainhops.datamodel.metadata._internal_api import (
    apply_loss_policy,
    call_under_policy,
)
from brainhops.datamodel.transformations import Transformation
from brainhops.errors import (
    ConversionError,
    ErrorAction,
    ErrorPolicy,
    LossyConversionError,
    WriterError,
)
from brainhops.io.base._base import FileFormat, takes_over
from brainhops.io.base._dispatch import (
    group_tiers,
    match_name,
    normalize_hints,
)
from brainhops.io.base.parsers import (
    AmbiguousFormatError,
    FileReader,
    FileWriter,
)
from brainhops.io.base.specs import format_hints


def save(
    obj: tx.Any,
    file: path.FileLike,
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
    *,
    on_loss: ErrorAction = UNSET,
    on_error: ErrorPolicy = UNSET,
    conversion: tx.Optional[tx.Mapping[str, tx.Any]] = None,
    **kwargs,
) -> tx.Any:
    """Write an object to a file in the format that the file name calls for.

    The format is chosen from the registered writable formats, in four
    steps.

    1. **The file name.** The formats that declare the longest of the
       extensions the name ends with are the candidates, so `a.ome.zarr`
       asks for OME-Zarr and `a.zarr` for any Zarr format. A format that
       requires a prefix is a candidate only when the name has one of
       its prefixes. With a `hint`, only the formats that answer to it
       are candidates, as for reading, and when none of them claims the
       name, they are all candidates, so that `a.json` can be written
       with `hint="neuroglancer.affine"`.
    2. **The object's own format.** If `obj` already is of one of the
       candidates, it is written as it is.
    3. **A format that can hold the object as it is.** Otherwise, an
       object that is not a transformation is written in a candidate
       that is a file-backed version of the very data model `obj` is an
       instance of, and takes every field that data model has
       (`NiftiImage` and `ZarrImage` are file-backed
       `SingleScaleImage`s). `obj` is converted to it, which neither
       loses nor changes anything. A `SingleScaleImage` that no
       candidate holds as it is, because no candidate stores a single
       image, becomes the finest level of a pyramid in a candidate that
       is a file-backed `MultiScaleImage`, as `obj.to(OmeZarrImage)`
       converts it. A format that builds its scales, as OME-Zarr does,
       builds the coarser levels as `with_scales()` builds them, unless
       `nscales=1` asks for the single level. A name that ends with
       `.ome.zarr` therefore writes an image as an OME-Zarr pyramid,
       while a name that ends with `.zarr` still asks for a plain Zarr
       array, which refuses an image that has a geometry.
    4. **A format a transformation converts to.** A transformation is
       instead converted to every candidate, with the same
       converters as `obj.to(Format, **conversion)`. A converter returns
       the map `obj` is, with its endpoints bridged to the format's (an
       affine to LPS is flipped into RAS), an approximation of it, or
       refuses: a general `Affine` is written as the voxel-to-RAS
       affine of a NIfTI file only when that is what it maps, or when
       its coordinate systems are not known. An approximation, such as
       a field interpolated with cubic splines that a NIfTI file stores
       as linearly interpolated values, is kept only when `on_loss`
       accepts it.

    In step 3, the candidates are tried from the most specific, by the
    rules reading uses: the longest prefix, then the narrowest
    declaration, then `PRIORITY`. Two that are equally specific and can
    both hold `obj` are an ambiguity, and nothing is written.

    In step 4, the conversions are ranked. The extension and the prefix
    of the name come first, so that a `y_` prefix asks for an SPM
    deformation. An exact conversion then comes before a lossy one, and a
    format that takes every option of `conversion` before one that does
    not, so that the images given for an FSL file select it. Then comes
    the conversion that keeps more of what `obj` holds: a field
    of the same kind (displacements, coordinates or a velocity), and the
    known endpoints of `obj`, so that a field between LPS spaces is
    written as an ITK field rather than as a NIfTI field between RAS
    spaces. The narrowest declaration and `PRIORITY` come last. Two
    conversions that rank first together are an ambiguity, and nothing
    is written.

    A candidate that refuses `obj` gives way to the next, in step 3
    and then in step 4, until one writes it. A candidate refuses when
    `obj` cannot be converted to it, or when its writer cannot write
    what was built from `obj` and raises a `ConversionError` or a
    `WriterError`, such as `UnrepresentableTransformationError`: a
    `register.dat` file refuses an affine between scanner RAS spaces
    without the geometries of both volumes. An `OSError`, as when the
    file cannot be written, is raised at once, as is any other error of
    a writer, and no other candidate is tried.

    !!! note "Approximations are opted into"
        A transformation is written in a format that holds it only
        approximately when `on_loss` accepts the loss, and a format
        that holds it exactly is still preferred. Without it, one that
        no candidate holds exactly is refused with each candidate's
        reason.

    !!! note "One pass for a transformation"
        Some formats have no exact converter yet (#312), and hold a
        transformation only by relabelling it, that is, by being
        rebuilt from its parameters, when the coordinate systems allow
        it. Step 4 relabels a transformation only into a format that is
        a file-backed version of its data model, so that the formats
        that would only relabel it do not make the choice ambiguous.
        Every candidate, including such a format, is given the options
        of the conversion, so that an LTA file or a `register.dat` file
        receives the geometries of its volumes.

    Parameters
    ----------
    obj : object
        The object to write.
    file : FileLike
        A path, or a file object opened for writing. An object without a name,
        such as a buffer, gives no name to choose from, so `obj` must already
        be a writable format.
    hint : str or iterable of str, optional
        Hints that name the formats to choose from, such as
        `"neuroglancer.affine"`, as for `load`. A file without a name is
        written by `obj` itself, and the hint is not used.
    on_loss : ErrorAction, optional
        What to do with a loss, as for a conversion (see
        [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). A transformation
        that a format holds only approximately, such as a field
        interpolated with cubic splines written as the linearly
        interpolated values of a NIfTI file, is a loss: `"ignore"` keeps
        such a format among the candidates, ranked after those that hold
        the transformation exactly, `"warn"` keeps it and warns when it
        is chosen, and a report keeps it and records the loss of the one
        that is chosen. Any other action drops such a format, and a
        value to return, or an exception to raise, takes the place of
        the error when only such formats remain. The action also
        applies to the metadata that the file cannot hold. By default,
        such a format is dropped, and a loss of metadata follows
        `on_error`, or else the policy in effect.
    on_error : ErrorPolicy, optional
        An error policy (see
        [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), whose action for
        a [`MetadataLossError`][brainhops.datamodel.metadata.MetadataLossError]
        decides what happens to the metadata that the file cannot hold,
        such as `"ignore"`, `"warn"`, `"raise"`, or a
        [`ConversionReport`][brainhops.datamodel.metadata.ConversionReport]
        to fill. An object that is converted, then written, is acted on
        once, after the file is written, with one report of what the
        conversion and the write lose. An object that is already of the
        chosen format is written by its own `save` under the policy. The
        metadata of a format that does not hold metadata is not written,
        and its fields are reported as lost. `on_loss` takes precedence
        over `on_error` for a loss, as in a conversion. The
        `WriterError` that `save` raises when no format can hold `obj`
        is not a conversion error, so it is outside the scope of a
        single action, and it is raised even under `on_error=None`. The
        `OSError` of a write that fails is raised as well. By default,
        an error is raised, and a loss of metadata follows the policy in
        effect.
    conversion : mapping, optional
        Options of the conversion of a transformation to the chosen
        format, as `obj.to(Format, **conversion)` takes them. A format
        that needs the geometry of an image, such as an FSL warp, takes
        it there, as in `conversion={"reference": image}`.
    **kwargs : Any
        Options passed to the `save` method of the chosen format. An
        option named after a field of the metadata, such as `dtype`,
        overrides that field (see the `consolidate` method of the
        format). The options that build the scales of a pyramid,
        `nscales=`, `factor=`, `mode=`, `align=`, `min_size=` and
        `strict=`, are taken out first and given to the conversion, for
        an image as for a transformation, so that
        `io.save(image, "image.ome.zarr", nscales=1)` writes a single
        level. They are those of
        [`Multiscale.with_scales`][brainhops.datamodel.multiscale.Multiscale.with_scales].
        A single scale saved to OME-Zarr is the exception: the writer
        takes these options, writes the finest level, and builds each
        coarser level from the levels that it has written, read back
        from the store, so that the finest level is read once and the
        memory stays bounded by a few chunks (#638). A pyramid built in
        memory is written as it is, so that
        `image.to(OmeZarrImage, nscales=4).save(path)` computes its
        levels from one Dask graph, while
        `io.save(image, path, nscales=4)` reads each level back. The
        values are the same, except that a level of an integer type
        that a mode averages is computed from the level before it once
        rounded, which may change a value by one.

    Returns
    -------
    Any
        `None`, or the value that the error policy returns in place of a
        refused loss.

    Raises
    ------
    AmbiguousFormatError
        If several equally specific formats can hold `obj`.
    WriterError
        If no format claims the file name, or if every one that does
        refuses `obj`, with the reason of each. If `obj` already is of one
        of these formats, the error of its own writer is raised. If a hint
        is given, no writable format answers to it, or none of those that do
        can hold `obj`, or the chosen format cannot write it.
    OSError
        If the file cannot be written, or the data of `obj` cannot be
        read.
    MetadataLossError
        If metadata is lost under the `"raise"` policy.
    """
    # The options that build the scales of a pyramid are options of the
    # conversion, for an image as for a transformation.
    pyramid = pop_pyramid_options(kwargs)
    file = path.as_path_or_stream(file)
    name = path.base_name(file)
    if name is None:
        if isinstance(obj, FileFormat) and isinstance(obj, FileWriter):
            obj = _own_scales(obj, pyramid, kwargs)
            return _save_as_it_is(obj, file, kwargs, on_error, on_loss)
        raise WriterError(
            f"Cannot choose a format to write {type(obj).__name__} in: the "
            f"file has no name to choose it from. Give the path of the "
            f"file, or build the format you want and save that."
        )

    hints = normalize_hints((), hint)
    registry = (
        fmt
        for fmt in FileFormat._REGISTRY
        if issubclass(fmt, FileWriter)
        and (not hints or format_hints(fmt) & hints)
    )
    matches = [(fmt, match_name(name, fmt)) for fmt in registry]
    claimed = [(fmt, match) for fmt, match in matches if match is not None]
    if not claimed and hints:
        # A hint names the formats, which are written whatever the name.
        claimed = [(fmt, (0, 0)) for fmt, _ in matches]
    if not claimed and hints:
        raise WriterError(
            f"Cannot write {name!r}: no writable format answers to the "
            f"hint(s) {', '.join(sorted(repr(h) for h in hints))}."
        )
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
        obj = _own_scales(obj, pyramid, kwargs)
        return _save_as_it_is(obj, file, kwargs, on_error, on_loss)

    # The policy acts once on the report of the conversion and the
    # write, after the file is written.
    policy = error_policy(on_error, on_loss=on_loss)
    loss = policy_action(policy, LossyConversionError)
    reasons: tx.List[str] = []
    dropped: tx.List[LossyConversionError] = []
    holders = [
        (fmt, match) for fmt, match in claimed if _holds(fmt, obj, reasons)
    ]
    if isinstance(obj, Transformation):
        # A transformation is converted to every candidate in one pass,
        # which gives each one the options of the conversion, and a format
        # of its own data model is converted to as well (see "One pass for
        # a transformation" above).
        own = {fmt for fmt, _ in holders}
        report = _write_best(
            file,
            name,
            obj,
            claimed,
            own,
            dict(conversion or {}),
            pyramid,
            loss,
            reasons,
            dropped,
            kwargs,
        )
    elif not holders and _is_one_level(obj, claimed):
        pyramids = [
            (fmt, match)
            for fmt, match in claimed
            if data_model_of(fmt) is MultiScaleImage
        ]
        report = _write_first(
            file,
            name,
            obj,
            pyramids,
            lambda obj, fmt: _one_level(obj, fmt, pyramid),
            reasons,
            kwargs,
            lambda obj, fmt: _split_pyramid(obj, fmt, pyramid)[1],
        )
    else:
        report = _write_first(
            file,
            name,
            obj,
            holders,
            lambda obj, fmt: _with_scales(
                _copy(obj, fmt), _split_pyramid(obj, fmt, pyramid)[0]
            ),
            reasons,
            kwargs,
            lambda obj, fmt: _split_pyramid(obj, fmt, pyramid)[1],
        )
    if report is not None:
        return _act_on_loss(report, policy)
    if dropped and not _drops_quietly(loss):
        # Only formats that lose were left, and the policy returns a
        # value, or raises an exception, in place of the loss.
        return apply_error_policy(
            dropped[0], {LossyConversionError: loss}, stacklevel=2
        )

    formats = ", ".join(sorted(fmt.__name__ for fmt, _ in claimed))
    detail = "".join(f"\n  - {reason}" for reason in reasons)
    raise WriterError(
        f"Cannot write {type(obj).__name__} to {name!r}: none of the "
        f"formats registered for it ({formats}) can hold it as it is. "
        f"Build the format you want with its `from_any` and save "
        f"that.{detail}"
    )


def _save_as_it_is(
    obj: tx.Any,
    file: path.FileLike,
    options: tx.Mapping[str, tx.Any],
    on_error: ErrorPolicy,
    on_loss: ErrorAction,
) -> tx.Any:
    """Write an object of a writable format with its own `save`.

    The policy of `save` applies around the write, as it applies to a
    call of `obj.save` that is given it.
    """
    return call_under_policy(
        lambda: obj.save(file, **options),
        on_error,
        on_loss=on_loss,
        stacklevel=3,
    )


def _copy(obj: tx.Any, fmt: type) -> tx.Any:
    """`obj` copied into `fmt`, a file-backed version of its data model."""
    return fmt.from_instance(obj)


def _is_one_level(
    obj: tx.Any, claimed: tx.List[tx.Tuple[type, tx.Any]]
) -> bool:
    """Whether `obj` is written as the finest level of a pyramid.

    This is the case for a single-scale image when no candidate format
    stores a single-scale image, so that a format that does, such as a
    plain Zarr array, keeps the choice that the file name gives it.
    """
    return isinstance(obj, SingleScaleImage) and not any(
        data_model_of(fmt) is SingleScaleImage for fmt, _ in claimed
    )


def _one_level(
    obj: SingleScaleImage, fmt: type, pyramid: tx.Mapping[str, tx.Any]
) -> tx.Any:
    """Return `obj` as the finest level of a pyramid of format `fmt`.

    A format whose `BUILDS_SCALES` is true, such as OME-Zarr, builds the
    coarser levels as `with_scales()` builds them, unless `nscales=1`
    asks for the single level, and the options in `pyramid` are those of
    `with_scales`.
    """
    built = fmt(images=[obj])
    if fmt.WRITES_SCALES:
        # The writer builds the scales, as `_split_pyramid` says.
        return built
    return build_pyramid(built, pyramid, default=fmt.BUILDS_SCALES)


def _split_pyramid(
    obj: tx.Any, fmt: type, pyramid: tx.Mapping[str, tx.Any]
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """Split the options of a pyramid between the conversion and the writer.

    The writer of a format whose `WRITES_SCALES` is true, such as
    OME-Zarr, builds the coarser scales of a single scale while it
    writes them, from the scales that it has written, read back from
    the store (#638). The conversion then keeps the single scale, and
    the writer takes the options, with `nscales=None` by default when
    the format builds its scales by default. An object of several
    scales is converted as it is, and the writer refuses options that
    would build more. Any other format takes the options in its
    conversion.

    Returns
    -------
    tuple of (dict, dict)
        The options of the conversion and those of the writer.
    """
    if not getattr(fmt, "WRITES_SCALES", False):
        return dict(pyramid), {}
    if getattr(obj, "nscales", 1) > 1:
        return {}, dict(pyramid)
    writer = dict(pyramid)
    if fmt.BUILDS_SCALES:
        writer.setdefault("nscales", None)
    return {"nscales": 1}, writer


def _own_scales(
    obj: tx.Any,
    pyramid: tx.Mapping[str, tx.Any],
    options: tx.Dict[str, tx.Any],
) -> tx.Any:
    """Return `obj`, saved in its own format, with its scales.

    A format whose `WRITES_SCALES` is true takes the options of a
    pyramid in `options`, and builds the scales while it writes them.
    Any other format builds them with `with_scales`.
    """
    if getattr(type(obj), "WRITES_SCALES", False):
        options.update(pyramid)
        return obj
    return _with_scales(obj, pyramid)


def _with_scales(obj: tx.Any, pyramid: tx.Mapping[str, tx.Any]) -> tx.Any:
    """Return `obj` with the scales that `pyramid` asks for.

    Raises
    ------
    TypeError
        If the options are given for an object that has no scales.
    """
    if not pyramid:
        return obj
    if not isinstance(obj, MultiScaleImage) and not hasattr(
        obj, "with_scales"
    ):
        names = ", ".join(f"{key}=" for key in sorted(pyramid))
        raise TypeError(
            f"{names} build the scales of a multiscale object, which a "
            f"{type(obj).__name__} is not."
        )
    return build_pyramid(obj, pyramid)


def _convert(
    obj: Transformation,
    fmt: type,
    options: tx.Mapping[str, tx.Any],
    own: bool = False,
) -> tx.Tuple[tx.Any, tx.Optional[LossyConversionError], int]:
    """`obj` converted to `fmt`, as `obj.to(fmt, **options)` converts it.

    The result is returned with the error that says what the conversion
    loses, or `None` when it is exact, and with the number of options
    that the format did not take. A lossy result is the one that
    `obj.to(fmt, on_loss="ignore")` returns. The options of the conversion
    are given to every format, so a format that does not take some of them
    is converted without them. A format without an exact converter for
    `obj` may relabel it (#312). Such a format is chosen only when `own`
    says that it is a file-backed version of the data model of `obj`, so
    that a format that converts `obj` is not made ambiguous by the formats
    that would only relabel it.

    The conversion goes through `obj.to`, so that a transformation that
    takes options of its own, such as an FSL warp given the images it
    was estimated for, receives them too.
    """
    # The transformation formats import this package, so their helpers are
    # imported when they are used.
    from brainhops.io.transformations.base._internal_api import (
        UnusedOptionError,
        relabels,
    )

    if not own and relabels(obj, fmt):
        raise ConversionError(
            f"{fmt.__name__} has no exact converter for a "
            f"{type(obj).__name__} (#312), and a transformation is saved "
            f"by relabelling it only in a format of its own data model."
        )
    unused = 0
    try:
        try:
            return obj.to(fmt, **options), None, unused
        except UnusedOptionError as error:
            # The options are given to every format, and one that does not
            # take some of them is converted without them.
            kept = {
                key: value
                for key, value in options.items()
                if key not in error.names
            }
            unused = len(options) - len(kept)
            return obj.to(fmt, **kept), None, unused
    except LossyConversionError as error:
        if error.result is None:
            raise
        return error.result, error, unused


def _write_best(
    file: path.FileLike,
    name: str,
    obj: Transformation,
    candidates: tx.List[tx.Tuple[type, tx.Any]],
    own: tx.Set[type],
    options: tx.Mapping[str, tx.Any],
    pyramid: tx.Mapping[str, tx.Any],
    loss: tx.Any,
    reasons: tx.List[str],
    dropped: tx.List[LossyConversionError],
    save_options: tx.Dict[str, tx.Any],
) -> tx.Optional[ConversionReport]:
    """
    Write a transformation in the candidate format that holds it best.

    `obj` is converted to every candidate. The conversions are then
    ranked: first by the extension and the prefix of the name that the
    format matches, then by whether the conversion is exact, then by the
    number of conversion options that the format did not take, so that a
    FLIRT matrix given its images comes before another matrix, then by how
    closely the format keeps what `obj` holds (see [`_fidelity`][]), and
    last by the declared extensions and the `PRIORITY` of the format. A
    lossy conversion is kept only when `loss`, the action of the policy
    for a loss, keeps it (see `_keeps_lossy`), and the loss of one that
    is dropped is added to `dropped`. Several
    conversions that rank first together are an ambiguity. A writer that
    refuses what it was given gives way to the next conversion.

    Returns
    -------
    ConversionReport or None
        The report of what the write of `obj` lost, or `None` when
        `obj` was not written.

    Raises
    ------
    AmbiguousFormatError
        If several conversions rank first together.
    OSError
        If converting or writing raises it.
    """
    ranked = []
    for fmt, match in _drop_base(candidates):
        report = ConversionReport(
            source=type(obj).__name__, target=fmt.__name__
        )
        conversion, written = _split_pyramid(obj, fmt, pyramid)
        try:
            with metadata_loss_policy(report):
                writer, loses, unused = _convert(
                    obj, fmt, {**options, **conversion}, fmt in own
                )
        except OSError:
            raise
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
            continue
        if loses is not None and not _keeps_lossy(loss):
            reasons.append(f"{fmt.__name__}: LossyConversionError: {loses}")
            dropped.append(loses)
            continue
        extension, prefix = match or (0, 0)
        key = (
            -extension,
            -prefix,
            int(loses is not None),
            unused,
            -_fidelity(obj, writer),
            0 if fmt.EXTENSIONS else 1,
            len(fmt.EXTENSIONS),
            -fmt.PRIORITY,
        )
        ranked.append((key, fmt, writer, report, loses, written))
    ranked.sort(key=lambda item: item[0])
    while ranked:
        best = [item for item in ranked if item[0] == ranked[0][0]]
        if len(best) > 1:
            raise AmbiguousFormatError(
                _ambiguity_message(name, obj, [item[1] for item in best])
            )
        _, fmt, writer, report, loses, written = best[0]
        ranked = ranked[1:]
        try:
            _write_once(obj, writer, file, report, {**save_options, **written})
        except (ConversionError, WriterError) as e:
            reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
            continue
        if loses is not None:
            # The chosen format holds `obj` only approximately, which a
            # warning or a report says.
            apply_error_policy(
                loses, {LossyConversionError: loss}, stacklevel=4
            )
        return report
    return None


def _fidelity(obj: Transformation, converted: Transformation) -> int:
    """
    How closely a converted transformation keeps what `obj` holds.

    Two formats may both hold the same map, such as a field of
    displacements in a NIfTI file between RAS spaces and in an ITK file
    between LPS spaces. The one that changes less is preferred: one point
    is given when the field it stores is of the same kind as the field of
    `obj` (displacements, coordinates or a velocity), and one point for
    each endpoint of `obj` that is known and that the format keeps.
    """
    score = 0
    if _field_kind(obj) == _field_kind(converted):
        score += 1
    for side in ("input", "output"):
        mine = getattr(obj, side, None)
        theirs = getattr(converted, side, None)
        if mine is not None and theirs is not None and mine == theirs:
            score += 1
    return score


def _field_kind(t: Transformation) -> tx.Optional[str]:
    """The kind of the field that `t` holds, or `None` for no field."""
    from brainhops.datamodel import transformations as xforms

    pending = [t]
    while pending:
        current = pending.pop()
        if isinstance(current, xforms.DisplacementField):
            return "velocity" if current.log else "displacements"
        if isinstance(current, xforms.CoordinateField):
            return "coordinates"
        if isinstance(current, xforms.Sequence):
            pending.extend(current.transformations or ())
    return None


def _drop_base(
    candidates: tx.List[tx.Tuple[type, tx.Any]],
) -> tx.List[tx.Tuple[type, tx.Any]]:
    """Drop the candidates that are base classes of another candidate."""
    classes = [cls for cls, _ in candidates]
    return [
        (cls, match)
        for cls, match in candidates
        if not any(
            other is not cls and issubclass(other, cls) for other in classes
        )
    ]


def _write_first(
    file: path.FileLike,
    name: str,
    obj: tx.Any,
    candidates: tx.List[tx.Tuple[type, tx.Any]],
    build: tx.Callable[[tx.Any, type], tx.Any],
    reasons: tx.List[str],
    options: tx.Dict[str, tx.Any],
    writer_options: tx.Optional[
        tx.Callable[[tx.Any, type], tx.Mapping[str, tx.Any]]
    ] = None,
) -> tx.Optional[ConversionReport]:
    """
    Write `obj` in the most specific candidate format that takes it.

    The candidates are tried tier by tier, most specific first, and
    `build(obj, fmt)` makes the object to write. A tier where it
    succeeds for several formats is an ambiguity. In a tier where it
    succeeds for one, that object is written with `options`, and a
    `ConversionError` or a `WriterError` from its writer moves on to the
    next tier. Why each format fails is added to `reasons`. Each build
    fills a report of its own with what the conversion of the metadata
    loses, and the write of the object adds to the report of its build
    (see `_write_once`).

    Building does not touch the file, and any error but an `OSError`
    means that the format cannot hold `obj`. Writing does, so only the
    errors by which a writer refuses an object move on: an `OSError`, or
    an error the writer does not raise on purpose, is raised at once.

    Returns
    -------
    ConversionReport or None
        The report of what the write of `obj` lost, or `None` when
        `obj` was not written.

    Raises
    ------
    AmbiguousFormatError
        If several formats of the first tier that takes `obj` take it.
    OSError
        If building or writing a candidate raises it.
    """
    for tier in group_tiers(candidates):
        writers = []
        for fmt in tier:
            report = ConversionReport(
                source=type(obj).__name__, target=fmt.__name__
            )
            try:
                with metadata_loss_policy(report):
                    writers.append((fmt, build(obj, fmt), report))
            except OSError:
                raise
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
        if len(writers) > 1:
            raise AmbiguousFormatError(
                _ambiguity_message(name, obj, [fmt for fmt, _, _ in writers])
            )
        if not writers:
            continue
        fmt, writer, report = writers[0]
        extra = {} if writer_options is None else writer_options(obj, fmt)
        try:
            _write_once(obj, writer, file, report, {**options, **extra})
        except (ConversionError, WriterError) as e:
            reasons.append(f"{fmt.__name__}: {type(e).__name__}: {e}")
            continue
        return report
    return None


def _write_once(
    obj: tx.Any,
    writer: tx.Any,
    file: path.FileLike,
    report: ConversionReport,
    options: tx.Mapping[str, tx.Any],
) -> None:
    """Write a converted object, and report what the write loses.

    The report of the conversion is the policy in effect while the
    object is written, so that the consolidation of the metadata adds
    to it. The policy of the save acts on the one
    report once the file is written (see `_act_on_loss`), outside of
    the loops that try the candidates, so that a refused loss is not
    taken for a format that refuses `obj`.
    A format that does not take the metadata of its data model over does
    not write it, so its fields that are set are reported as lost, and so
    are those of `obj` when the conversion gave the writer no metadata,
    as some converters between transformations do.
    """
    # The metadata formats import this package, so their helpers are
    # imported when they are used.
    from brainhops.io.metadata._base import MetadataFormat
    from brainhops.io.metadata._internal_api import (
        lost_fields,
        unsupported_values,
    )

    options = dict(options)
    source = getattr(obj, "metadata", None)
    if source is not None and getattr(writer, "metadata", None) is None:
        lost_fields(unsupported_values(MetadataFormat, source), report)
    if takes_over(type(writer), "metadata"):
        with metadata_loss_policy(report):
            writer.save(file, **options)
    else:
        metadata = getattr(writer, "metadata", None)
        if metadata is not None:
            lost_fields(unsupported_values(MetadataFormat, metadata), report)
        writer.save(file, **options)


def _keeps_lossy(loss: tx.Any) -> bool:
    """Whether the action for a loss keeps a format that loses.

    `"ignore"`, `"warn"` and a report keep it, since they return the
    result. The default, `"raise"`, a value and an exception drop it.
    """
    if isinstance(loss, str):
        return loss in ("ignore", "warn")
    return isinstance(loss, ConversionReport)


def _drops_quietly(loss: tx.Any) -> bool:
    """Whether the dropped formats leave `save` to raise its own error.

    Without a policy, or under `"raise"`, `save` raises the
    `WriterError` that gives the reason of each candidate.
    """
    return loss is UNSET or (isinstance(loss, str) and loss == "raise")


def _act_on_loss(
    report: ConversionReport, policy: tx.Mapping[type, ErrorAction]
) -> tx.Any:
    """Act on the report of a save that wrote its file.

    The report is acted on under the policy of the save, by
    `apply_loss_policy`, which warns or fills a report and returns, and
    raises a `MetadataLossError` for any other action. That error is
    then handed to the policy, which may raise it, raise another
    exception, or return a value, which `save` returns.
    """

    def act() -> None:
        apply_loss_policy(report, stacklevel=2)

    return call_under_policy(act, policy, stacklevel=3)


def data_model_of(cls: type) -> tx.Optional[type]:
    """Return the data model that `cls` instantiates, or `None`.

    This is the first class in the MRO that is a data model but not I/O,
    such as `SingleScaleImage` for `NiftiImage`.
    """
    for base in cls.__mro__:
        if issubclass(base, DataModelBase) and not issubclass(
            base, (FileFormat, FileReader, FileWriter)
        ):
            return base
    return None


def _init_fields(cls: type) -> tx.Set[str]:
    # Public names are compared, because a format may take a field of its
    # data model over under a private name, as `NiftiImage` stores
    # `transformations` in `_transformations` and `OmeZarrImage` stores
    # `images` in `_images`.
    return {field.public_name for field in fields(cls) if field.init}


def _holds(fmt: type, obj: tx.Any, reasons: tx.List[str]) -> bool:
    """Whether `obj` converts to `fmt` without loss.

    The format must be a file-backed version of the very data model that `obj`
    instantiates, and take every field of that model. `NiftiVoxelToRAS` is an
    `Affine` that means voxel-to-RAS, so a general `Affine` written through it
    would come back with another meaning. A format that stores its content
    differently, such as OME-Zarr keeping the arrays it read rather than a list
    of scales, would be built with content missing.
    """
    model = data_model_of(type(obj))
    if model is None or data_model_of(fmt) is not model:
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
# so that the three could share a helper. The formats that save chooses
# from are told apart by converting the object, so the message names the
# conversion.


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
