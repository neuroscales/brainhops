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
their unit, and finally, only when a positional pairing is permitted,
their position. Two axes that name different types are never matched,
whatever else they share. A positional pairing keeps to each type group,
so a spatial axis pairs with a spatial axis and never with a time axis.
Once two axes are matched, the ratio between their units is read off and
applied as part of the scaling.

There are two entry points.
[`bridge`][brainhops.datamodel._xform_adaptors.bridge] returns the bridge
from one system to another. The bridge is a single primitive, a sequence
of primitives, or the identity, and not always a sequence.
[`adapt`][brainhops.datamodel._xform_adaptors.adapt] ingests two
consecutive transformations and returns a sequence that contains both of
them, with the reconciling bridge placed where the two systems meet.
"""

# externals
import warnings

import numpy as np
import typing_extensions as tx

# internals
from .axes import Axis
from .systems import ArrayCoordinateSystem, CoordinateSystem
from .transformations import (
    AdaptationError,
    Identity,
    Permutation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
    _register_bridge,
    _register_subspace_wrap,
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


def _collinear(source: Axis, target: Axis) -> bool:
    # Whether two axes may be matched by their orientation. Two axes that
    # are both oriented may only be matched when they lie along the same
    # line, because a flip aligns two axes on one line and a rotation
    # aligns two axes on different lines. When at most one of the axes is
    # oriented, the orientation places no constraint on the match.
    source_line = _orientation_line(source)
    target_line = _orientation_line(target)
    if source_line is None or target_line is None:
        return True
    return source_line == target_line


def _orientation_sign(source: Axis, target: Axis) -> int:
    # The sign that aligns two collinear axes. It is ``+1`` when the two
    # axes point the same way and ``-1`` when they point opposite ways.
    # Two axes oriented along different lines are related by a rotation,
    # not a flip, so no single sign aligns them and the case is refused.
    source_line = _orientation_line(source)
    target_line = _orientation_line(target)
    if source_line is None or target_line is None:
        return 1
    if source_line != target_line:
        raise AdaptationError(
            "Two matched axes are oriented along different anatomical "
            "lines, so aligning them is a rotation rather than a flip. The "
            "adaptor builds only exact flips, reorderings, and rescalings, "
            "and cannot bridge these two systems."
        )
    source_value = getattr(getattr(source, "orientation", None), "value", None)
    target_value = getattr(getattr(target, "orientation", None), "value", None)
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
    # The ratio between two SI-prefixed units is a power of ten, so it is
    # computed from the difference of the two base-ten scale exponents. A
    # millimetre to a micrometre is then exactly 1000, and a ratio and its
    # reciprocal multiply back to exactly one, which a division of the two
    # scales does not guarantee.
    source_log10 = getattr(source_unit, "log10_scale", None)
    target_log10 = getattr(target_unit, "log10_scale", None)
    if source_log10 is not None and target_log10 is not None:
        return 10.0 ** (source_log10 - target_log10)
    return float(source_unit.scale) / float(target_unit.scale)


def _is_array_side(system: tx.Optional[CoordinateSystem], axis: Axis) -> bool:
    # Whether an axis indexes an array rather than measures a world
    # coordinate. Reversing an array-index axis shifts the origin by one
    # less than its extent, while reversing a world axis is a pure sign
    # flip. Three signals mark an array-index axis. Its system is an array
    # coordinate system, such as a voxel grid, even one whose axes are
    # named and oriented and carry a length unit. Or the axis is discrete.
    # Or the axis carries no unit, so its samples are plain indices.
    if isinstance(system, ArrayCoordinateSystem):
        return True
    if getattr(axis, "discrete", None):
        return True
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
            "than its extent, so the number of samples along it is needed "
            "to bridge these systems. Compose these transformations next "
            "to a grid that carries the axis sizes, or bridge the two "
            "systems explicitly and give the sizes, with "
            "adapt(first, second, extents=...) or "
            "bridge(source, target, extents=...)."
        )
    return int(value)


def _type_conflict(source: Axis, target: Axis) -> bool:
    # Whether two axes carry definite, different types. A shared name, a
    # shared unit, or a shared position never matches two axes across such
    # a conflict. An axis whose type is unset places no constraint, so a
    # conflict is present only when both axes name a type and the two types
    # differ.
    source_type = getattr(source, "type", None)
    target_type = getattr(target, "type", None)
    if source_type is None or target_type is None:
        return False
    return source_type != target_type


def _same_unit_kind(source: Axis, target: Axis) -> bool:
    # Whether two axes are measured in units of the same kind, such as two
    # lengths or two durations. Two axes matched this way have a defined
    # conversion factor between their units. An axis with no unit is not
    # matched by unit at all.
    source_unit = getattr(source, "unit", None)
    target_unit = getattr(target, "unit", None)
    if source_unit is None or target_unit is None:
        return False
    return getattr(source_unit, "type", None) == getattr(
        target_unit, "type", None
    )


def _group_by_type(
    axes: tx.List[Axis], indices: tx.List[int]
) -> "tx.Dict[tx.Optional[str], tx.List[int]]":
    # The given axis indices grouped by the `type` of the axis, keeping the
    # indices of each group in ascending order. An axis with no type forms
    # a group of its own, keyed by ``None``.
    groups: tx.Dict[tx.Optional[str], tx.List[int]] = {}
    for i in indices:
        key = getattr(axes[i], "type", None)
        groups.setdefault(key, []).append(i)
    return groups


def _pair_type_groups(
    source_groups: "tx.Dict[tx.Optional[str], tx.List[int]]",
    target_groups: "tx.Dict[tx.Optional[str], tx.List[int]]",
) -> "tx.List[tx.Tuple[int, int]]":
    # Pair the still-unmatched axes by order within each type group, as a
    # list of ``(target_index, source_index)`` pairs. A group of a definite
    # type pairs with the group of the same type on the other side, and a
    # typeless group is a wildcard that pairs with the one remaining typed
    # group of equal count on the other side. A group is paired only when
    # the same number of its axes remain on each side.
    #
    # The mappings passed in are consumed as pairs are made, so a group that
    # is left in either mapping afterwards had no counterpart and is
    # reported by the caller. A typeless group that could map to two or more
    # remaining typed groups, or whose count matches none of them, is
    # genuinely ambiguous and is left unpaired rather than guessed.
    pairs: tx.List[tx.Tuple[int, int]] = []
    for key in list(target_groups):
        sources = source_groups.get(key)
        if sources is not None and len(sources) == len(target_groups[key]):
            targets = target_groups.pop(key)
            del source_groups[key]
            for j, i in zip(targets, sources):
                pairs.append((j, i))
    # A leftover typeless group is a wildcard. It pairs, by order, with the
    # single remaining typed group of equal count on the other side. The two
    # directions cover a typeless source meeting a typed target and the
    # reverse.
    for wildcard, other, wildcard_is_source in (
        (source_groups, target_groups, True),
        (target_groups, source_groups, False),
    ):
        none_indices = wildcard.get(None)
        if not none_indices:
            continue
        candidates = [
            key
            for key, indices in other.items()
            if key is not None and len(indices) == len(none_indices)
        ]
        if len(candidates) != 1:
            continue
        other_indices = other.pop(candidates[0])
        del wildcard[None]
        for a, b in zip(none_indices, other_indices):
            if wildcard_is_source:
                pairs.append((b, a))
            else:
                pairs.append((a, b))
    return pairs


def _match_axes(
    source_axes: tx.List[Axis],
    target_axes: tx.List[Axis],
    allow_positional: bool,
    allow_type_grouped_positional: bool = False,
) -> "tx.Tuple[tx.List[tx.Optional[int]], tx.Optional[str]]":
    # Establish, for each target axis, the source axis that corresponds to
    # it. The result is a pair of a list `match`, where `match[j]` is the
    # index of the source axis that feeds target axis `j`, and a warning
    # message, or `None`. An axis for which no unique correspondence is
    # found is left unmatched, and the caller reports the failure. The
    # warning message names a positional pairing that was made, so the
    # caller can emit it once the pairing is committed to, rather than
    # before code that might still raise.
    #
    # Two axes that both name a type, and name different types, are never
    # matched. A shared name, a shared unit, or a shared position is a
    # coincidence that a definite type conflict overrides, so a spatial
    # axis and a time axis that happen to share a name stay unmatched.
    n_source, n_target = len(source_axes), len(target_axes)
    match: tx.List[tx.Optional[int]] = [None] * n_target
    used = [False] * n_source
    warning: tx.Optional[str] = None

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

    # Second tier: axes that share a name. A shared name does not match two
    # axes of different types, and two axes oriented along different lines
    # are never paired, because a shared name cannot align a rotation.
    for j, target in enumerate(target_axes):
        if match[j] is not None or target.name is None:
            continue
        candidates = [
            i
            for i, source in enumerate(source_axes)
            if not used[i]
            and source.name == target.name
            and not _type_conflict(source, target)
            and _collinear(source, target)
        ]
        if len(candidates) == 1:
            take(j, candidates[0])

    # Third tier: axes measured in the same kind of unit, when only one
    # such axis remains on each side, so the pairing is unambiguous. A
    # shared unit does not match two axes of different types, and two axes
    # oriented along different lines are again not paired.
    for j, target in enumerate(target_axes):
        if match[j] is not None:
            continue
        candidates = [
            i
            for i, source in enumerate(source_axes)
            if not used[i]
            and _same_unit_kind(source, target)
            and not _type_conflict(source, target)
            and _collinear(source, target)
        ]
        if len(candidates) == 1:
            take(j, candidates[0])

    # Last tier: position. A positional pairing can mask a genuine
    # mismatch, so it warns and is used only when the caller permits it. The
    # warning is not emitted here, but returned, so the caller can raise on
    # the pairing before deciding to warn about it.
    if allow_positional and n_source == n_target:
        # The caller permitted a positional pairing outright. Any axis
        # still unmatched is paired with the source axis in its position,
        # unless the two axes at that position name conflicting types.
        paired = False
        for j in range(n_target):
            if match[j] is None and not used[j]:
                if _type_conflict(source_axes[j], target_axes[j]):
                    continue
                take(j, j)
                paired = True
        if paired:
            warning = (
                "Some axes were paired by position, because no stronger "
                "correspondence was found. Check that the two coordinate "
                "systems describe the same axes in the same order."
            )
    elif allow_type_grouped_positional:
        # An implicit bridge pairs the still-unmatched axes by order within
        # each type group: spatial axes with spatial axes, time with time,
        # and so on. An axis of no type is a wildcard, so a typeless group
        # pairs with the one remaining typed group of equal count. A group
        # is paired only when the same number of its axes remain on each
        # side. Two axes that both name a type are never paired across it,
        # so a spatial axis with no spatial counterpart is left unmatched
        # and reported. A group whose counts disagree, or a typeless group
        # that two or more typed groups could receive, is likewise left
        # unmatched, rather than pairing some of its axes and inventing the
        # rest.
        unmatched_source = [i for i in range(n_source) if not used[i]]
        unmatched_target = [j for j in range(n_target) if match[j] is None]
        source_groups = _group_by_type(source_axes, unmatched_source)
        target_groups = _group_by_type(target_axes, unmatched_target)
        pairs = _pair_type_groups(source_groups, target_groups)
        for j, i in pairs:
            take(j, i)
        if pairs:
            warning = (
                "Some axes were paired by position within their type group, "
                "because no name, unit, or orientation distinguished them. "
                "Check that the two coordinate systems describe the same "
                "axes in the same order."
            )

    return match, warning


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
        axis for i, axis in enumerate(source.axes) if i not in matched_source
    ]

    def names(axes: tx.List[Axis]) -> str:
        return ", ".join(
            repr(getattr(axis, "name", None) or getattr(axis, "type", None))
            for axis in axes
        )

    parts = []
    if unmatched_target:
        parts.append(f"target axes with no source ({names(unmatched_target)})")
    if unmatched_source:
        parts.append(f"source axes with no target ({names(unmatched_source)})")
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
    allow_type_grouped_positional: bool = False,
) -> Transformation:
    """Return the transformation that carries `source` coordinates to `target`.

    The bridge maps coordinates expressed in `source` to the coordinates
    of the same points expressed in `target`. It is built from the axis
    descriptions of the two systems, and reorders, rescales, and flips
    axes so that a value written in one frame is read correctly in the
    other. The bridge never resamples and never loses information, and its
    inverse is the bridge from `target` back to `source`.

    The axes of the two systems are matched by type and orientation, then
    by name, then by unit, and finally, only when a positional pairing is
    permitted, by position. Two axes that name different types are never
    matched, whatever else they share. From the
    match the bridge derives a
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
    allow_type_grouped_positional : bool, default False
        Whether to pair the still-unmatched axes by order within each type
        group, so spatial axes pair with spatial axes and time with time.
        An axis of no type is a wildcard, so a group of typeless axes pairs
        with the one remaining typed group of equal count. A group is paired
        only when the same number of its axes remain on each side. A pairing
        made this way warns. Two axes that both name a type are never paired
        across it. A type group whose counts disagree, or a typeless group
        that two or more typed groups could receive, is left unmatched and
        reported.

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
    if len(source.axes) != len(target.axes):
        raise AdaptationError(
            "Cannot bridge {} to {}: the two systems have different numbers "
            "of axes. A bridge reorders, rescales, and flips matched axes "
            "and never adds or drops one. When one transform acts on a "
            "subset of the other's axes, such as a spatial transform meeting "
            "a spatial-and-time image, composition lifts the smaller "
            "transform onto the axes it acts on. Those axes could not be "
            "matched to a subset of the fuller system here, so the lift was "
            "not possible. Give the shared axes matching names, units, or "
            "orientations so they can be identified.".format(
                source.name or "the source system",
                target.name or "the target system",
            )
        )

    source_axes = list(source.axes)
    target_axes = list(target.axes)
    match, positional_warning = _match_axes(
        source_axes,
        target_axes,
        allow_positional,
        allow_type_grouped_positional,
    )
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
            _is_array_side(source, source_axis)
            or _is_array_side(target, target_axis)
        ):
            offset = float(_extent(extents, j, target_axis) - 1)
        translations.append(offset)

    # The positional pairing, if one was made, is now committed to. Every
    # matched pair has a defined sign, ratio, and offset, and nothing below
    # raises. The warning is emitted here, rather than at match time, so a
    # pairing that a later step rejects does not warn. The stack level names
    # the caller of `bridge`, not `bridge` itself.
    if positional_warning is not None:
        warnings.warn(positional_warning, stacklevel=2)

    # Build the intermediate systems so each primitive names its own
    # endpoints and the whole bridge runs from `source` to `target`. Only
    # the last primitive lands on `target`. A permutation lands on the
    # source axes reordered into the target order, and a scaling that a
    # shift follows lands on the target axes before the origin shift.
    permuted_axes = [source_axes[i] for i in permutation]
    permuted_system = CoordinateSystem(axes=permuted_axes)
    has_scale = any(scale != 1 for scale in scales)
    has_offset = any(offset != 0 for offset in translations)

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
    if has_scale:
        scaled_system = (
            CoordinateSystem(axes=target_axes) if has_offset else target
        )
        elements.append(
            Scaling(
                scale=np.asarray(scales, dtype=float),
                input=current,
                output=scaled_system,
            )
        )
        current = scaled_system
    if has_offset:
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
    boundary where the two are composed. The result always contains both
    `first` and `second`. When their systems disagree, the bridge that
    reconciles them is placed at that boundary, so the result is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence] of `first`,
    the bridge, and `second`, and applying it is applying `first`, adapting
    the coordinates, and applying `second`.

    When the two systems already agree, the result is a sequence of
    `first` and `second` with nothing added.

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
        A sequence that contains both `first` and `second`, with the
        reconciling bridge placed where their systems meet.
    """
    source = first.output
    target = second.input
    if (
        source is not None
        and target is not None
        and source.axes is not None
        and target.axes is not None
        and len(target.axes) < len(source.axes)
    ):
        wrapped = _subspace_wrap(
            source, target, second, second.output, extents=extents
        )
        if wrapped is not None:
            return Sequence(
                transformations=[first, wrapped],
                input=first.input,
                output=wrapped.output,
            )
    reconciler = bridge(
        first.output,
        second.input,
        extents=extents,
        allow_positional=allow_positional,
    )
    elements: tx.List[Transformation] = [first]
    if not is_identity(reconciler):
        if isinstance(reconciler, Sequence):
            elements.extend(reconciler.transformations or [])
        else:
            elements.append(reconciler)
    elements.append(second)
    return Sequence(
        transformations=elements, input=first.input, output=second.output
    )


def _subset_positions(
    full_axes: tx.List[Axis], sub_axes: tx.List[Axis]
) -> tx.Optional[tx.List[int]]:
    # The position, in `full_axes`, of each axis in `sub_axes`, when
    # `sub_axes` is a clean subset of `full_axes`. Each subset axis is
    # matched to a fuller axis by the same kernel that builds a bridge, so
    # a spatial axis pairs with a spatial axis and an oriented axis with
    # its collinear counterpart. The result is `None` when a subset axis
    # has no match, or when an unmatched fuller axis is of the same kind as
    # an axis the subset acts on. The second case is a genuine
    # dimensionality mismatch, such as a spatial axis with no spatial
    # counterpart, and is left for the caller to refuse rather than
    # absorbed as a pass-through.
    #
    # The grouped-positional fallback is enabled, so a subset of unnamed and
    # unoriented axes of one type still lifts, the same way an implicit
    # bridge pairs such axes. The warning belongs to the bridge built over
    # the subset below, so the message returned here is discarded.
    match, _ = _match_axes(
        full_axes,
        sub_axes,
        allow_positional=False,
        allow_type_grouped_positional=True,
    )
    if any(i is None for i in match):
        return None
    positions = [int(i) for i in match]
    acted_types = {getattr(axis, "type", None) for axis in sub_axes}
    used = set(positions)
    for i, axis in enumerate(full_axes):
        if i in used:
            continue
        if getattr(axis, "type", None) in acted_types:
            return None
    return positions


def _subspace_wrap(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
    transform: Transformation,
    transform_output: tx.Optional[CoordinateSystem],
    *,
    extents: tx.Optional[Extents] = None,
) -> tx.Optional[Transformation]:
    """Lift a transform onto the axes it acts on inside a fuller space.

    The transformation `transform` acts on the axes of `target`, which are
    a subset of the axes of the fuller system `source`. This routine wraps
    `transform` in a
    [`SubspaceTransformation`][brainhops.datamodel.transformations.SubspaceTransformation]
    that acts on those axes within `source` and leaves the extra axes
    unchanged. The dimensionality is preserved, so a spatial transform
    meeting a spatial-and-time image acts on the spatial axes and leaves
    time untouched.

    The matched axes may still need a reordering, a rescaling, or a flip,
    such as the sign flip between RAS and LPS. That intra-subset bridge is
    built by [`bridge`][brainhops.datamodel._xform_adaptors.bridge] over
    the subset and placed before `transform` inside the wrapper.

    The return value is `None` when `source` and `target` are not a
    same-dimensionality subset, so the caller can fall back to refusing a
    genuine dimensionality mismatch.

    Parameters
    ----------
    source : CoordinateSystem, optional
        The fuller system that the wrapped transform reads and writes.
    target : CoordinateSystem, optional
        The input system of `transform`, a subset of `source`.
    transform : Transformation
        The transform to lift into the axis space of `source`.
    transform_output : CoordinateSystem, optional
        The output system of `transform`.
    extents : sequence or mapping, optional
        The extents needed to reverse an array-index axis within the
        subset, passed through to
        [`bridge`][brainhops.datamodel._xform_adaptors.bridge].

    Returns
    -------
    Transformation or None
        The wrapped transform, or `None` when no same-dimensionality
        subset match exists.
    """
    if source is None or target is None:
        return None
    if source.axes is None or target.axes is None:
        return None
    if transform_output is None or transform_output.axes is None:
        return None
    full_axes = list(source.axes)
    sub_in_axes = list(target.axes)
    sub_out_axes = list(transform_output.axes)
    if len(sub_in_axes) >= len(full_axes):
        return None
    if len(sub_out_axes) != len(sub_in_axes):
        return None
    positions = _subset_positions(full_axes, sub_in_axes)
    if positions is None:
        return None

    # The wrapped transform reads the acted-on axes from the fuller space
    # in their fuller order. The transform itself expects them in the order
    # and frame of `target`, so a bridge over the subset carries them the
    # rest of the way, and sits before the transform inside the wrapper.
    sub_source = CoordinateSystem(axes=[full_axes[i] for i in positions])
    sub_bridge = bridge(
        sub_source,
        target,
        extents=extents,
        allow_type_grouped_positional=True,
    )
    if is_identity(sub_bridge):
        inner: Transformation = transform
    else:
        inner = Sequence(
            transformations=[sub_bridge, transform],
            input=sub_source,
            output=transform_output,
        )

    # The output system is the fuller system with the acted-on axes
    # replaced by the transform's output axes, and every extra axis kept in
    # place. The acted-on axes stay in the positions they came from.
    out_full_axes = list(full_axes)
    for k, position in enumerate(positions):
        out_full_axes[position] = sub_out_axes[k]
    axis_vector = np.asarray(positions, dtype=int)
    return SubspaceTransformation(
        transformation=inner,
        input_axes=axis_vector,
        output_axes=axis_vector,
        input=source,
        output=CoordinateSystem(axes=out_full_axes),
    )


# The sequence machinery in `transformations.py` inserts a bridge at every
# boundary where two adjacent transforms disagree. It calls back into this
# routine, which is registered here so the two modules need not import each
# other at module scope. The subspace-wrapping routine is registered the
# same way, for the boundaries where the two systems differ in the number
# of axes.
_register_bridge(bridge)
_register_subspace_wrap(_subspace_wrap)
