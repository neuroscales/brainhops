"""Bridge coordinate systems that meet at a composition boundary.

Two consecutive transformations compose only when the output system of
the first describes the same frame as the input system of the second.
When the two systems disagree, this module builds the exact, invertible
transformation that carries coordinates from the first frame to the
second. The bridge reorders, rescales, and flips axes. It never resamples
and never loses information.

The bridge is built from the axis descriptions alone, by matching the
axes of the two systems and reading off the reordering, the unit ratios,
and the orientation signs the match implies. Matching compares axes in
priority order: first their type and orientation, then their name, then
their unit, and finally, only when explicitly permitted, their position.

There are two entry points.
[`bridge`][brainhops.datamodel._xform_adaptors.bridge] returns the bridge
between two systems.
[`adapt`][brainhops.datamodel._xform_adaptors.adapt] ingests two
consecutive transformations and returns a sequence that contains both of
them with the bridge inserted between them.
"""

# externals
import warnings

import numpy as np
import typing_extensions as tx

# internals
from .axes import Axis
from .systems import CoordinateSystem
from .transformations import (
    AdaptationError,
    Identity,
    Permutation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
    _register_bridge,
    is_identity,
)

# An extent table maps an axis, by its position in the target system or by
# its name, to the number of samples along that axis. It is needed only to
# reverse an array-index axis, where a flip shifts the origin by one less
# than the extent.
Extents = tx.Union[tx.Sequence[tx.Optional[int]], tx.Mapping[tx.Any, int]]


def _orientation_line(axis: Axis) -> tx.Optional[tx.FrozenSet[str]]:
    # The undirected line an oriented axis lies along, as the unordered
    # pair of its two poles. A left-to-right axis and a right-to-left axis
    # share the line ``{"left", "right"}``. An axis with no orientation, or
    # one whose orientation names no value, has no line.
    orientation = getattr(axis, "orientation", None)
    if orientation is None:
        return None
    value = getattr(orientation, "value", None)
    if not isinstance(value, str) or not value:
        return None
    poles = value.split("-to-")
    if len(poles) != 2:
        return frozenset({value})
    return frozenset(poles)


def _orientation_sign(source: Axis, target: Axis) -> int:
    # The sign that aligns two collinear axes. It is ``+1`` when the two
    # axes point the same way and ``-1`` when they point opposite ways.
    source_orientation = getattr(source, "orientation", None)
    target_orientation = getattr(target, "orientation", None)
    if source_orientation is None or target_orientation is None:
        return 1
    source_value = getattr(source_orientation, "value", None)
    target_value = getattr(target_orientation, "value", None)
    if source_value is None or target_value is None:
        return 1
    return 1 if source_value == target_value else -1


def _unit_ratio(source: Axis, target: Axis) -> float:
    # The factor that converts a value measured in the source axis's unit
    # to the target axis's unit. Two axes that carry no unit, or the same
    # unit, have a ratio of one. An axis with a unit matched to an axis
    # without one has no defined ratio, which is reported as a failure.
    source_unit = getattr(source, "unit", None)
    target_unit = getattr(target, "unit", None)
    if source_unit is None and target_unit is None:
        return 1.0
    if source_unit is None or target_unit is None:
        raise AdaptationError(
            "One axis carries a unit and the axis it matches does not, so "
            "no conversion factor exists between them. Give both axes a "
            "unit, or neither."
        )
    source_type = getattr(source_unit, "type", None)
    target_type = getattr(target_unit, "type", None)
    if source_type != target_type:
        raise AdaptationError(
            "Two matched axes are measured in units of different kinds, "
            "so no conversion factor exists between them."
        )
    return float(source_unit.scale) / float(target_unit.scale)


def _is_array_axis(axis: Axis) -> bool:
    # Whether an axis indexes an array rather than measures a world
    # coordinate. An array-index axis carries no unit, so its samples are
    # plain indices. Reversing such an axis shifts the origin, while
    # reversing a world axis does not.
    return getattr(axis, "unit", None) is None


def _extent(extents: tx.Optional[Extents], position: int, axis: Axis) -> int:
    # The number of samples along an axis, read from the extent table by
    # position or by name. A reversed array-index axis needs it, and its
    # absence is a failure rather than a guess.
    value = None
    if isinstance(extents, tx.Mapping):
        name = getattr(axis, "name", None)
        if position in extents:
            value = extents[position]
        elif name is not None and name in extents:
            value = extents[name]
    elif extents is not None and position < len(extents):
        value = extents[position]
    if value is None:
        raise AdaptationError(
            "A reversed array-index axis shifts the origin by one less "
            "than its extent, so the number of samples along it is "
            "needed. Pass the extent of the axis to adapt these systems."
        )
    return int(value)


def _match_axes(
    source_axes: tx.List[Axis],
    target_axes: tx.List[Axis],
    allow_positional: bool,
) -> tx.List[int]:
    # Establish, for each target axis, the source axis that corresponds to
    # it. The result is a list `match` where `match[j]` is the index of the
    # source axis that feeds target axis `j`. An axis for which no unique
    # correspondence is found is left unmatched, and the caller reports the
    # failure.
    n_source, n_target = len(source_axes), len(target_axes)
    match: tx.List[tx.Optional[int]] = [None] * n_target
    used = [False] * n_source

    def take(j: int, i: int) -> None:
        match[j] = i
        used[i] = True

    # First tier: axes of the same type that lie along the same oriented
    # line. This is the strongest signal, and it pairs a right-to-left axis
    # with a left-to-right one regardless of their names.
    for j, target in enumerate(target_axes):
        line = _orientation_line(target)
        if line is None:
            continue
        candidates = [
            i
            for i, source in enumerate(source_axes)
            if not used[i]
            and source.type == target.type
            and _orientation_line(source) == line
        ]
        if len(candidates) == 1:
            take(j, candidates[0])
        elif len(candidates) > 1:
            named = [
                i for i in candidates if source_axes[i].name == target.name
            ]
            if len(named) == 1:
                take(j, named[0])

    # Second tier: axes that share a name.
    for j, target in enumerate(target_axes):
        if match[j] is not None or target.name is None:
            continue
        candidates = [
            i
            for i, source in enumerate(source_axes)
            if not used[i] and source.name == target.name
        ]
        if len(candidates) == 1:
            take(j, candidates[0])

    # Third tier: axes of the same type, when only one such axis remains on
    # each side, so the pairing is unambiguous.
    for j, target in enumerate(target_axes):
        if match[j] is not None or target.type is None:
            continue
        candidates = [
            i
            for i, source in enumerate(source_axes)
            if not used[i] and source.type == target.type
        ]
        if len(candidates) == 1:
            take(j, candidates[0])

    # Last tier: position, used only when the two systems have the same
    # number of axes and the caller permitted it. A positional pairing can
    # mask a genuine mismatch, so it warns.
    if allow_positional and n_source == n_target:
        paired = False
        for j in range(n_target):
            if match[j] is None and not used[j]:
                take(j, j)
                paired = True
        if paired:
            warnings.warn(
                "Some axes were paired by position, because no stronger "
                "correspondence was found between them. Check that the two "
                "coordinate systems describe the same axes in the same "
                "order.",
                stacklevel=2,
            )

    return match


def _unmatched_report(
    source: CoordinateSystem,
    target: CoordinateSystem,
    match: tx.List[tx.Optional[int]],
) -> tx.NoReturn:
    matched_source = {i for i in match if i is not None}
    unmatched_target = [
        target.axes[j] for j, i in enumerate(match) if i is None
    ]
    unmatched_source = [
        axis
        for i, axis in enumerate(source.axes)
        if i not in matched_source
    ]

    def names(axes: tx.List[Axis]) -> str:
        return ", ".join(
            repr(getattr(axis, "name", None) or getattr(axis, "type", None))
            for axis in axes
        )

    parts = []
    if unmatched_target:
        parts.append(
            f"target axes with no source ({names(unmatched_target)})"
        )
    if unmatched_source:
        parts.append(
            f"source axes with no target ({names(unmatched_source)})"
        )
    raise AdaptationError(
        "Cannot bridge {} to {}: {}. Adaptation reorders, rescales, and "
        "flips matched axes, and does not change the number of axes.".format(
            source.name or "the source system",
            target.name or "the target system",
            " and ".join(parts),
        )
    )


def bridge(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
    *,
    extents: tx.Optional[Extents] = None,
    allow_positional: bool = False,
) -> Transformation:
    """Return the transformation that carries `source` coordinates to `target`.

    The bridge maps coordinates expressed in `source` to the coordinates
    of the same points expressed in `target`. It is built from the axis
    descriptions of the two systems, and reorders, rescales, and flips
    axes so that a value written in one frame is read correctly in the
    other. The bridge never resamples and never loses information, and its
    inverse is the bridge from `target` back to `source`.

    The axes of the two systems are matched by type and orientation, then
    by name, then by unit, and finally, only when `allow_positional` is
    true, by position. From the match the bridge derives a
    [`Permutation`][brainhops.datamodel.transformations.Permutation] that
    reorders the axes, a
    [`Scaling`][brainhops.datamodel.transformations.Scaling] that applies
    the per-axis unit ratio and an orientation sign of `-1` for each
    reversed axis, and, when a reversed axis indexes an array, a
    [`Translation`][brainhops.datamodel.transformations.Translation] that
    shifts the origin by one less than the axis extent.

    When either system is unspecified, or carries no axes, the two are
    assumed compatible and the identity is returned. When an axis that
    must be matched has no correspondence, an
    [`AdaptationError`][brainhops.datamodel.transformations.AdaptationError]
    names the two systems and the unmatched axes.

    Parameters
    ----------
    source : CoordinateSystem, optional
        The system the coordinates are expressed in.
    target : CoordinateSystem, optional
        The system the coordinates are carried to.
    extents : sequence or mapping, optional
        The number of samples along each target axis, needed only to
        reverse an array-index axis. A sequence is read by target axis
        position, and a mapping by position or by axis name.
    allow_positional : bool, default False
        Whether to pair axes by position when no stronger correspondence
        is found. A positional pairing warns.

    Returns
    -------
    Transformation
        The bridge, as a single primitive, a
        [`Sequence`][brainhops.datamodel.transformations.Sequence] of
        primitives, or an
        [`Identity`][brainhops.datamodel.transformations.Identity] when no
        adaptation is needed.
    """
    if source is None or target is None:
        return Identity(input=source, output=target)
    if source.axes is None or target.axes is None:
        return Identity(input=source, output=target)
    if source == target:
        return Identity(input=source, output=target)

    source_axes = list(source.axes)
    target_axes = list(target.axes)
    match = _match_axes(source_axes, target_axes, allow_positional)
    if any(i is None for i in match):
        _unmatched_report(source, target, match)

    permutation = [int(i) for i in match]
    scales = []
    translations = []
    for j, i in enumerate(permutation):
        source_axis, target_axis = source_axes[i], target_axes[j]
        sign = _orientation_sign(source_axis, target_axis)
        ratio = _unit_ratio(source_axis, target_axis)
        scales.append(sign * ratio)
        offset = 0.0
        if sign == -1 and (
            _is_array_axis(source_axis) or _is_array_axis(target_axis)
        ):
            offset = float(_extent(extents, j, target_axis) - 1)
        translations.append(offset)

    # Build the intermediate systems so each primitive names its own
    # endpoints and the whole bridge runs from `source` to `target`.
    permuted_axes = [source_axes[i] for i in permutation]
    permuted_system = CoordinateSystem(axes=permuted_axes)

    elements: tx.List[Transformation] = []
    current = source
    needs_permutation = permutation != list(range(len(permutation)))
    if needs_permutation:
        elements.append(
            Permutation(
                permutation=np.asarray(permutation),
                input=source,
                output=permuted_system,
            )
        )
        current = permuted_system
    if any(scale != 1 for scale in scales):
        elements.append(
            Scaling(
                scale=np.asarray(scales, dtype=float),
                input=current,
                output=target,
            )
        )
        current = target
    if any(offset != 0 for offset in translations):
        elements.append(
            Translation(
                translation=np.asarray(translations, dtype=float),
                input=current,
                output=target,
            )
        )
        current = target

    if not elements:
        return Identity(input=source, output=target)
    if len(elements) == 1:
        only = elements[0]
        return only.to(input=source, output=target)
    return Sequence(transformations=elements, input=source, output=target)


def adapt(
    first: Transformation,
    second: Transformation,
    *,
    extents: tx.Optional[Extents] = None,
    allow_positional: bool = False,
) -> Transformation:
    """Bridge two consecutive transformations and return them together.

    The transformation `first` is applied before `second`. The output
    system of `first` and the input system of `second` meet at the
    boundary between them. When the two systems disagree, the bridge that
    reconciles them is inserted between the two transformations. The
    result is a [`Sequence`][brainhops.datamodel.transformations.Sequence]
    that contains `first`, then the bridge, then `second`, so applying it
    is applying `first`, adapting the coordinates, and applying `second`.

    When the two systems already agree, no bridge is inserted and the
    result is a sequence of `first` and `second` alone.

    Parameters
    ----------
    first : Transformation
        The transformation applied first. Its output system is the source
        of the bridge.
    second : Transformation
        The transformation applied second. Its input system is the target
        of the bridge.
    extents : sequence or mapping, optional
        The extents needed to reverse an array-index axis, passed through
        to [`bridge`][brainhops.datamodel._xform_adaptors.bridge].
    allow_positional : bool, default False
        Whether to pair axes by position, passed through to
        [`bridge`][brainhops.datamodel._xform_adaptors.bridge].

    Returns
    -------
    Transformation
        A sequence of `first`, the bridge, and `second`.
    """
    between = bridge(
        first.output,
        second.input,
        extents=extents,
        allow_positional=allow_positional,
    )
    elements: tx.List[Transformation] = [first]
    if not is_identity(between):
        if isinstance(between, Sequence):
            elements.extend(between.transformations or [])
        else:
            elements.append(between)
    elements.append(second)
    return Sequence(
        transformations=elements, input=first.input, output=second.output
    )


# The sequence machinery in `transformations.py` inserts a bridge at every
# boundary where two adjacent transforms disagree. It calls back into this
# routine, which is registered here so the two modules need not import each
# other at module scope.
_register_bridge(bridge)
