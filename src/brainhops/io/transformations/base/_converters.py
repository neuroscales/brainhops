"""
Private helpers that relabel a transformation as a file format.

Most file formats have no exact converter yet (#312). Such a format holds
another transformation only by being rebuilt from it with the same
parameters, which is called relabelling. This module holds the converter
that relabels, [`relabelling`][], and the helper it calls,
[`relabel`][]. The module registers nothing: each format that relabels
registers the converter explicitly, once for each family it holds, in
its own `_converters` module, for example

```python
converter(_xforms.Sequence, FnirtWarpField)(relabelling)
```

Such a registration names its format and a family, so it is as specific
as an exact converter would be, and more specific than the converters
of the data model, which also match a request for a format but know
nothing of its coordinate systems. Since every format registers the same
function, [`relabels`][] can tell a relabelling from an exact
conversion.

A relabelled transformation is the same map only if the format means
the same coordinate systems. Each endpoint of the transformation that
is known must therefore be compatible with the system that the format
has on that side, both the system that the format takes when it is
given none and the system that the relabelled format reports. The
check is the one that the classes bound to coordinate systems run when
they are built, which raises an
[`IncompatibleSystemError`][brainhops.errors.IncompatibleSystemError].
An endpoint that is not known is taken to be the format's. A format
that has no system of its own on a side accepts any endpoint there, and
keeps it.
"""

__all__ = ["relabel", "relabelling", "relabels"]

import typing_extensions as tx

from brainhops._core._internal_api import replace
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._internal_api import bridge, convert, find_converter
from brainhops.errors import (
    AdaptationError,
    ConversionError,
    IncompatibleSystemError,
)

from ._base import TransformationFormat
from ._conversions import unrepresentable
from ._endpoints import check_endpoint, compatible


def relabelling(
    t: _xforms.Transformation, cls: type, **kwargs
) -> TransformationFormat:
    """
    Convert a transformation to a format without an exact converter.

    This is the converter that a format registers, for each family that
    it holds, until an exact converter is written for it (#312). It
    relabels `t` as the format (see [`relabel`][]).

    Parameters
    ----------
    t : Transformation
        The transformation to convert.
    cls : type
        The format to convert it to.
    **kwargs : Any
        Options of the format, passed to its constructor.

    Returns
    -------
    TransformationFormat
        The format, holding the parameters of `t`.

    Raises
    ------
    ConversionError
        If the format cannot hold `t` by relabelling it.
    """
    return relabel(t, cls, **kwargs)


def relabels(t: _xforms.Transformation, cls: type) -> bool:
    """
    Whether `t.to(cls)` relabels `t` rather than converting it exactly.

    A format without an exact converter for the family of `t` holds it
    only by relabelling, through the converter [`relabelling`][] that it
    registers. `io.save` writes a transformation
    in a format that it is not already of only by an exact conversion,
    or by a copy into a format of its own data model, so it asks this
    question before it converts.

    Parameters
    ----------
    t : Transformation
        The transformation to convert.
    cls : type
        The format to convert it to.

    Returns
    -------
    bool
        Whether the conversion runs [`relabelling`][].
    """
    return find_converter(t, cls) is relabelling


def relabel(
    t: _xforms.Transformation, cls: type, **kwargs
) -> TransformationFormat:
    """
    Rebuild a transformation as a format, keeping its parameters.

    This is what a format without an exact converter for `t` does with
    it (#312), through [`relabelling`][], and what `cls.from_any(t)`,
    `cls.from_instance(t)` and `t.to(cls)` then return. The converters
    of a format also call it for another variant of the format. The
    format is rebuilt from the data model that it stores: a format built
    on `Affine` takes the affine that `t` exactly is, which the
    converters of the data model compute, and a format built on a
    `Sequence` takes the chain of `t`. The relabelled
    format is then built as the bases of
    [`TransformationFormat`][brainhops.io.transformations.base.TransformationFormat]
    copy an instance, so that the fields that belong to the format of
    `t`, such as the header of a NIfTI file, are not carried over.

    The result is the map that `t` is only when the format means the
    same coordinate systems. Each known endpoint of `t` must therefore
    be compatible with the system that the format takes when it is given
    none, and with the system that the relabelled format reports, as the
    classes bound to coordinate systems check when they are built. An
    endpoint of `t` that is not known passes, and so does any endpoint
    on a side where the format has no system of its own. A format that
    stores an affine first carries `t` into its own systems when they
    differ: each such endpoint is bridged as composition bridges two
    neighbours, by reordering, flipping and rescaling axes, so an affine
    between LPS systems is held exactly by a format between RAS systems.
    A voxel system is not bridged to a world system, because that needs
    the geometry of the volume, which a format converts in its own exact
    converter.

    Parameters
    ----------
    t : Transformation
        The transformation to relabel.
    cls : type
        The format to relabel it as.
    **kwargs : Any
        Options of the format, passed to its constructor.

    Returns
    -------
    TransformationFormat
        The format, holding the parameters of `t`.

    Raises
    ------
    ConversionError
        If `cls` is not a transformation, if `t` is not of the family
        that the format stores and does not convert to it exactly, if the
        format cannot be built from the parameters of `t` or cannot read
        its systems from them, or if an endpoint of `t` is not compatible
        with the system of the format on that side.
    """
    model = _data_model(cls)
    if model is None:
        raise unrepresentable(
            t,
            cls,
            "it is not a transformation but a family of formats. Convert "
            "to one of the formats of the family.",
        )
    if issubclass(model, _xforms.Sequence):
        if not isinstance(t, _xforms.Sequence):
            raise unrepresentable(
                t,
                cls,
                f"the format holds a chain of transformations, and a "
                f"{type(t).__name__} is not one.",
            )
        # A chain that a format decodes from its file is not held in a
        # field, so the copy would not find it there.
        kwargs.setdefault("transformations", tuple(t))
        # Another kind of chain, such as a mutable one, holds the same
        # fields, and is rebuilt as the data model so that the copy is
        # between related classes.
        source = t if isinstance(t, model) else replace(t, model)
    else:
        t = _adapted(t, cls)
        source = (
            t if isinstance(t, model) else _within_data_model(t, cls, model)
        )
    _check_endpoints(t, cls, cls())
    result = _rebuild(t, cls, source, kwargs)
    _check_endpoints(t, cls, result)
    return result


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _data_model(cls: type) -> tx.Optional[type]:
    # The class of the data model that the format stores, which is the
    # first transformation in its method resolution order that is not a
    # format, such as `Affine` for an LTA file or `ImmutableSequence` for
    # a NiftyReg field. A family of formats, which is not a
    # transformation, has none.
    for base in cls.__mro__:
        if issubclass(base, _xforms.Transformation) and not issubclass(
            base, TransformationFormat
        ):
            return base
    return None


def _adapted(t: _xforms.Transformation, cls: type) -> _xforms.Transformation:
    # `t` carried into the coordinate systems of the format, when an
    # endpoint of `t` is known but differs from the system of the format
    # on that side. Each such endpoint is bridged, as composition bridges
    # two neighbours: by reordering, flipping and rescaling axes, which is
    # exact, and the bridged chain is composed into an affine. A system
    # that cannot be bridged this way, such as a voxel system against a
    # world system, leaves `t` as it is, and the check of the endpoints
    # then refuses it.
    held = cls()
    chain = [t]
    if held.input is not None and not compatible(t.input, held.input):
        chain.insert(0, (held.input, t.input))
    if held.output is not None and not compatible(t.output, held.output):
        chain.append((t.output, held.output))
    if len(chain) == 1:
        return t
    try:
        chain = [
            bridge(*link, allow_type_grouped_positional=True)
            if isinstance(link, tuple)
            else link
            for link in chain
        ]
        return convert(_xforms.Sequence(chain), _xforms.Affine)
    except (AdaptationError, ConversionError):
        return t


def _within_data_model(
    t: _xforms.Transformation, cls: type, model: type
) -> _xforms.Transformation:
    # `t` converted exactly to the data model of the format, such as a
    # scaling to the affine that a matrix file holds.
    try:
        return convert(t, model)
    except ConversionError as error:
        raise unrepresentable(
            t,
            cls,
            f"the format holds a {model.__name__}, and {error}",
        ) from error


def _rebuild(
    t: _xforms.Transformation,
    cls: type,
    source: _xforms.Transformation,
    kwargs: tx.Dict[str, tx.Any],
) -> TransformationFormat:
    # The format built from `source` as the bases of `TransformationFormat`
    # copy an instance, which skips the conversion that the format itself
    # would run. A format that cannot be built from these parameters, such
    # as a pyramid of fields from a chain, does not hold `t`.
    try:
        return super(TransformationFormat, cls).from_instance(source, **kwargs)
    except IncompatibleSystemError as error:
        raise unrepresentable(t, cls, str(error)) from error
    except ConversionError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise unrepresentable(
            t,
            cls,
            f"the format, which has no exact converter yet (#312), cannot "
            f"be built from its parameters. {error}",
        ) from error


def _check_endpoints(
    t: _xforms.Transformation, cls: type, held: _xforms.Transformation
) -> None:
    # Refuse an endpoint of `t` that is not compatible with the system
    # that `held`, an instance of the format, has on the same side. An
    # endpoint that is not known on either side passes. A format that
    # decodes its systems from content it did not write, such as an ITK
    # field from another chain, may fail to read them, and then does not
    # hold `t` either.
    for side in ("input", "output"):
        try:
            system = getattr(held, side)
        except (AttributeError, TypeError, ValueError) as error:
            raise unrepresentable(
                t,
                cls,
                f"the format, which has no exact converter yet (#312), "
                f"cannot read its {side} from the parameters of a "
                f"{type(t).__name__}. {error}",
            ) from error
        if getattr(t, side) is None or system is None:
            continue
        try:
            check_endpoint(t, side, system)
        except IncompatibleSystemError as error:
            raise unrepresentable(
                t,
                cls,
                f"its {side} is not compatible with the {system.name} "
                f"system of the format. A transformation is relabelled as "
                f"a format without an exact converter (#312) only when "
                f"its systems are not known, or are compatible with those "
                f"of the format.",
            ) from error
