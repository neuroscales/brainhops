"""Bridge coordinate systems that meet at a composition boundary.

Two consecutive transformations compose only when the output system of
the first describes the same frame as the input system of the second.

When the two systems disagree, this module builds the exact, invertible
transformation that carries coordinates from the first frame to the
second. The bridge reorders, rescales, and flips axes. It never resamples
and never loses information.

The bridge is built from the axis descriptions alone, by matching the
axes of the two systems and reading off the reordering, the unit ratios,
and the orientation signs implied by the match. Axes are matched, in
priority order, by:

1. their type and orientation,
2. their name,
3. their unit, and finally
4. their position (only when a positional pairing is permitted).

Furthermore:

* Two axes that have different types are never matched, even if they
  share the same name or unit.

* When pairing axes by position, matching is restricted to axes of the
  same type (so a spatial axis pairs with a spatial axis and never with
  a time axis) and order *within each type group* is preserved.

* Once two axes are matched, the ratio between their units is read off
  and applied as part of the scaling.

There are two entry points:

1. [`bridge`][] returns the bridge from one system to another.
   The bridge is a single primitive, a sequence of primitives, or the
   identity, and not always a sequence. It assumes that the two systems
   have the same dimensionality.

2. [`adapt`][] ingests two consecutive transformations and returns a
   sequence that contains both of them, with the reconciling bridge
   placed where the two systems meet.
   When the two systems have different numbers of axes, and one transform
   acts on a subset of the other's axes, `adapt` lifts the smaller
   transform into the fuller space instead, so it acts on those axes and
   leaves the extra axes unchanged.
"""

# externals
import warnings
from functools import partial

import numpy as np
import typing_extensions as tx

# api
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import ArrayCoordinateSystem, CoordinateSystem
from brainhops.datamodel.units import Unit

# internals
from .base import Transformation
from .concrete import (
    CartesianField,
    Identity,
    Permutation,
    Scaling,
    Translation,
    is_identity,
)
from .errors import AdaptationError
from .meta import SubspaceTransformation
from .registries import register_adapt
from .sequence import Sequence, _interpolates

Extents = tx.Union[tx.Sequence[tx.Optional[int]], tx.Mapping[tx.Any, int]]
"""
An extent table maps an axis, by its position in the target system or by
its name, to the number of samples along that axis. It is needed only to
reverse an array-index axis, where a flip shifts the origin by one less
than the extent.
"""

# --- Orientation ------------------------------------------------------

_OrientationLike = tx.Union[Axis, Orientation, str, None]


def _orientation(orientation: _OrientationLike) -> tx.Optional[str]:
    """
    Unwrap an [`Axis`][] or [`Orientation`][] into its orientation value.
    """
    if isinstance(orientation, Axis):
        orientation = orientation.orientation
    if isinstance(orientation, Orientation):
        orientation = orientation.value
    return orientation


def _orientation_line(
    orientation: _OrientationLike,
) -> tx.Optional[tx.FrozenSet[str]]:
    """
    The undirected orientation of an oriented axis, represented as the
    unordered pair of its two poles.

    !!! example
        ```pycon
        >>> _orientation_line("left-to-right") == frozenset({"left", "right"})
        True
        >>> _orientation_line("right-to-left") == frozenset({"left", "right"})
        True
        >>> _orientation_line("what-is-this") == frozenset({"what-is-this"})
        True
        >>> _orientation_line(Axis()) == None
        True
        ```
    """
    orientation = _orientation(orientation)
    if not isinstance(orientation, str) or not orientation:
        return None
    poles = orientation.split("-to-")
    if len(poles) != 2:
        return frozenset({orientation})
    return frozenset(poles)


def _orientation_conflict(
    source: _OrientationLike, target: _OrientationLike
) -> bool:
    """
    Whether two axes conflict because of their orientation.

    This is the case when they have set orientations that are not
    collinear. If one of the orientations is unset (`None`), there is
    no conflict.
    """
    source_line = _orientation_line(source)
    target_line = _orientation_line(target)
    if source_line is None or target_line is None:
        return False
    return source_line != target_line


def _orientation_sign(
    source: _OrientationLike, target: _OrientationLike
) -> int:
    """
    The sign that aligns two collinear axes.

    It is `+1` when they point the same way and `-1` when they point
    opposite ways.
    """
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
    source_value = _orientation(source)
    target_value = _orientation(target)
    return 1 if source_value == target_value else -1


# --- Units ------------------------------------------------------------

_UnitLike = tx.Union[Axis, Unit, None]


def _unit(unit: _UnitLike) -> tx.Optional[Unit]:
    """
    Unwrap an [`Axis`][] into its unit.

    An axis with no unit, or a unit that is not a [`Unit`][] instance, is
    reported as `None`.
    """
    if isinstance(unit, Axis):
        unit = unit.unit
    if isinstance(unit, Unit):
        return unit
    return None


def _unit_ratio(source: _UnitLike, target: _UnitLike) -> float:
    """
    The factor that converts a value measured in the source axis's unit
    to the target axis's unit.

    Two axes that carry no unit, or the same unit, have a ratio of one.

    An axis with a unit matched to an axis without one has no defined
    ratio, which is reported as a failure.
    """
    source_unit = _unit(source)
    target_unit = _unit(target)
    if source_unit is None and target_unit is None:
        return 1.0
    if source_unit is None or target_unit is None:
        raise AdaptationError(
            "One axis carries a unit and the axis it matches does not, so "
            "no conversion factor exists between them. Give both axes a "
            "unit, or neither."
        )
    if source_unit.type != target_unit.type:
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
        return 10 ** (source_log10 - target_log10)
    return float(source_unit.scale) / float(target_unit.scale)


def _same_unit_kind(source: _UnitLike, target: _UnitLike) -> bool:
    """
    Whether two axes are measured in units of the same kind, such as two
    lengths or two durations.

    Two axes matched this way have a defined conversion factor between
    their units.

    Axes with no unit are not matched by unit at all, so this function
    returns `False` if this is the case for any of the two axes.
    """
    source_unit = _unit(source)
    target_unit = _unit(target)
    if source_unit is None or target_unit is None:
        return False
    return source_unit.type == target_unit.type


# --- Shape ------------------------------------------------------------


def _is_array_side(system: tx.Optional[CoordinateSystem], axis: Axis) -> bool:
    """
    Whether an axis indexes an array rather than measures a world
    coordinate.

    Reversing an array-index axis shifts the origin by one less than its
    extent, while reversing a world axis is a pure sign flip.

    Three signals mark an array-index axis. Its system is an array
    coordinate system, such as a voxel grid, even one whose axes are
    named and oriented and carry a length unit. Or the axis is discrete.
    Or the axis carries no unit, so its samples are plain indices.
    """
    if isinstance(system, ArrayCoordinateSystem):
        return True
    if axis.discrete:
        return True
    return axis.unit is None


def _extent(extents: tx.Optional[Extents], position: int, axis: Axis) -> int:
    """
    The number of samples along an axis, read from the extent table by
    position or by name. A reversed array-index axis needs it, and its
    absence is a failure rather than a guess.
    """
    value = None
    if isinstance(extents, tx.Mapping):
        name = axis.name
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


# --- Type -------------------------------------------------------------

_TypeLike = tx.Union[Axis, str, None]
_TypeGroup = tx.Dict[tx.Optional[str], tx.List[int]]


def _type(axis: _TypeLike) -> tx.Optional[str]:
    if isinstance(axis, Axis):
        return axis.type
    if isinstance(axis, str):
        return axis
    return None


def _type_conflict(source: _TypeLike, target: _TypeLike) -> bool:
    """
    Whether two axes carry definite, different types.

    A shared name, unit, or position never matches two axes across
    such a conflict.

    An axis whose type is unset places no constraint, so a conflict is
    present only when both axes name a type and the two types differ.
    """
    source_type = _type(source)
    target_type = _type(target)
    if source_type is None or target_type is None:
        return False
    return source_type != target_type


def _group_by_type(
    axes: tx.List[_TypeLike], indices: tx.List[int]
) -> _TypeGroup:
    """
    The given axis indices grouped by the `type` of the axis,
    keeping the indices of each group in ascending order.

    An axis with no type forms a group of its own, keyed by `None`.
    """
    groups: _TypeGroup = {}
    for i in indices:
        key = _type(axes[i])
        groups.setdefault(key, []).append(i)
    return groups


def _pair_type_groups(
    source_groups: _TypeGroup,
    target_groups: _TypeGroup,
) -> tx.List[tx.Tuple[int, int]]:
    """
    Pair the still-unmatched axes by order within each type group, as a
    list of `(target_index, source_index)` pairs.

    A group of a definite type pairs with the group of the same type on
    the other side, and a typeless group is a wildcard that pairs with
    the one remaining typed group of equal count on the other side.

    A group is paired only when the same number of its axes remain on
    each side.
    """

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


# --- Bridge -----------------------------------------------------------


def _match_axes(
    source_axes: tx.List[Axis],
    target_axes: tx.List[Axis],
    allow_positional: bool,
    allow_type_grouped_positional: bool = False,
) -> tx.Tuple[tx.List[tx.Optional[int]], tx.Optional[str]]:
    """
    Match each target axis to its corresponding source axis.

    This function returns:

    1. a list `match`, where `match[j]` is the index of the source axis
       that feeds target axis `j`;
    2. a warning message, or `None`.

    An axis for which no unique correspondence is found is left unmatched,
    and the caller reports the failure.

    The warning message names a positional pairing that was made, so the
    caller can emit it once the pairing is committed to, rather than
    before code that might still raise.
    """

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
                # Only take a candidate if it is unique (no ambiguity).
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
            and not _orientation_conflict(source, target)
        ]
        if len(candidates) == 1:
            # Only take candidate if it is unique (no ambiguity).
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
            and not _orientation_conflict(source, target)
        ]
        if len(candidates) == 1:
            # Only take candidate if it is unique (no ambiguity).
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
        return ", ".join(repr(axis.name or axis.type) for axis in axes)

    parts = []
    if unmatched_target:
        parts.append(f"target axes with no source ({names(unmatched_target)})")
    if unmatched_source:
        parts.append(f"source axes with no target ({names(unmatched_source)})")
    source_name = source.name or "the source system"
    target_name = target.name or "the target system"
    reason = " and ".join(parts)
    raise AdaptationError(
        f"Cannot bridge {source_name} to {target_name}: {reason}. "
        f"Adaptation reorders, rescales, and flips matched axes, and "
        f"does not change the number of axes."
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
    matched, whatever else they share. From the match the bridge derives a
    [`Permutation`][] that reorders the axes, a [`Scaling`][] that applies
    the per-axis unit ratio and an orientation sign of `-1` for each
    reversed axis, and, when a reversed axis indexes an array, a
    [`Translation`][] that shifts the origin by one less than the axis
    extent.

    When either system is unspecified, or carries no axes, the two are
    assumed compatible and the identity is returned. When an axis that
    must be matched has no correspondence, an [`AdaptationError`][]
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
    # --- special cases ------------------------------------------------
    if source is None or target is None:
        return Identity(input=source, output=target)
    if source.axes is None or target.axes is None:
        return Identity(input=source, output=target)
    if source == target:
        return Identity(input=source, output=target)

    # --- dimensionality check -----------------------------------------
    if len(source.axes) != len(target.axes):
        source_name = source.name or "the source system"
        target_name = target.name or "the target system"
        raise AdaptationError(
            f"Cannot bridge {source_name} to {target_name}: "
            f"the two systems have different numbers of axes. A bridge "
            f"reorders, rescales, and flips matched axes and never adds "
            f"or drops one. When one transform acts on a subset of the "
            f"other's axes, such as a spatial transform meeting a "
            f"spatial-and-time image, composition lifts the smaller "
            f"transform onto the axes it acts on. Those axes could not "
            f"be matched to a subset of the fuller system here, so the "
            f"lift was not possible. Give the shared axes matching "
            f"names, units, or orientations so they can be identified."
        )

    # --- matching -----------------------------------------------------
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

    # --- compute parameters -------------------------------------------
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

    # --- positional warning -------------------------------------------
    # The positional pairing, if one was made, is now committed to. Every
    # matched pair has a defined sign, ratio, and offset, and nothing below
    # raises. The warning is emitted here, rather than at match time, so a
    # pairing that a later step rejects does not warn. The stack level names
    # the caller of `bridge`, not `bridge` itself.
    if positional_warning is not None:
        warnings.warn(positional_warning, stacklevel=2)

    # --- build transformations and systems ----------------------------
    # Build the intermediate systems so each primitive names its own
    # endpoints and the whole bridge runs from `source` to `target`. Only
    # the last primitive lands on `target`. A permutation lands on the
    # source axes reordered into the target order, and a scaling that a
    # shift follows lands on the target axes before the origin shift.
    permuted_axes = [source_axes[i] for i in permutation]
    permuted_system = CoordinateSystem(axes=permuted_axes)
    needs_scale = any(scale != 1 for scale in scales)
    needs_offset = any(offset != 0 for offset in translations)
    needs_perm = permutation != list(range(len(permutation)))

    elements: tx.List[Transformation] = []
    current = source
    # --- permutation --------------------------------------------------
    if needs_perm:
        elements.append(
            Permutation(
                permutation=np.asarray(permutation),
                input=source,
                output=permuted_system,
            )
        )
        current = permuted_system
    # --- scale --------------------------------------------------------
    if needs_scale:
        scaled_system = (
            CoordinateSystem(axes=target_axes) if needs_offset else target
        )
        elements.append(
            Scaling(
                scale=np.asarray(scales, dtype=float),
                input=current,
                output=scaled_system,
            )
        )
        current = scaled_system
    # --- offset -------------------------------------------------------
    if needs_offset:
        elements.append(
            Translation(
                translation=np.asarray(translations, dtype=float),
                input=current,
                output=target,
            )
        )
        current = target

    # --- returns ------------------------------------------------------
    if not elements:
        # length-zero sequence -> return identity
        return Identity(input=source, output=target)
    if len(elements) == 1:
        # length-one sequence -> return its single element
        only = elements[0]
        return only.to(input=source, output=target)
    # length-n sequence -> return a sequence of its elements
    return Sequence(transformations=elements, input=source, output=target)


# --- Adapt ------------------------------------------------------------


def _systems_disagree(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
) -> bool:
    """
    Whether a bridge is needed between two adjacent systems.

    A system that is unspecified, or that carries no axes, is treated
    as compatible with its neighbour, so only two fully described and
    unequal systems disagree.
    """
    if source is None or target is None:
        return False
    if source.axes is None or target.axes is None:
        return False
    return source != target


def adapt(
    first: Transformation,
    second: Transformation,
    *,
    extents: tx.Optional[Extents] = None,
    allow_positional: bool = False,
    allow_type_grouped_positional: bool = False,
) -> Sequence:
    """Reconcile two consecutive transformations and return them together.

    The transformation `first` is applied before `second`. The output
    system of `first` and the input system of `second` meet at the
    boundary where the two are composed.

    The result always contains both `first` and `second`, as a
    [`Sequence`][]. The sequence usually runs from the input of `first`
    to the output of `second`. A lift is the exception:

    * A forward lift wraps `second` so that it acts in the fuller space,
      and the sequence then ends in that fuller output system rather
      than the output of `second`.

    * A backward lift wraps `first`, and the sequence then begins in
      that fuller input system rather than the input of `first`.

    In general:

    * When the two systems already agree, the sequence holds `first` and
      `second` with nothing between them

    * When the two systems have the same number of axes but merely reorder,
      rescale, or flip them, the bridge that reconciles them is placed
      between the two, so applying the result is applying `first`,
      adapting the coordinates, and applying `second`.

    * When the two systems have different numbers of axes, one transform
      acts on a subset of the other's axes. The transform on the fewer
      axes is lifted into the fuller space by a [`SubspaceTransformation`][]
      that acts on those axes and leaves the extra axes unchanged.

      - A 3D spatial transform meeting a 4D spatial-and-time boundary
        is lifted this way, whether the fuller space is on the input
        side or the output side of the boundary.

      - A genuine dimensionality mismatch, where the extra axes are not
        a clean pass-through, is refused with an [`AdaptationError`][].

    Parameters
    ----------
    first : Transformation
        The transformation applied first. Its output system is the source
        of the boundary.
    second : Transformation
        The transformation applied second. Its input system is the target
        of the boundary.
    extents : sequence or mapping, optional
        The extents needed to reverse an array-index axis, passed through
        to [`bridge`][]. When not given, they are read from a grid on
        either side of the boundary.
    allow_positional : bool, default False
        Whether to pair axes by position, passed through to [`bridge`][].
    allow_type_grouped_positional : bool, default False
        Whether to pair the still-unmatched axes by order within each
        type group, passed through to [`bridge`][].

    Returns
    -------
    Sequence
        A sequence that contains both `first` and `second`, with any
        reconciling bridge or subspace lift placed where their systems
        meet. `first` and `second` are never rebuilt, so each is the same
        object it was passed as, unless it was lifted into a fuller space.
    """
    source = first.output
    target = second.input

    # --- systems match: short circuit ---------------------------------
    if not _systems_disagree(source, target):
        return Sequence(
            transformations=[first, second],
            input=first.input,
            output=second.output,
        )

    # --- compute extents for grid coordinate systems ------------------
    if extents is None:
        extents = _grid_extents(first, at_output=True)
        extents.update(_grid_extents(second, at_output=False))
    extents = extents or None

    # The sequence normally runs from the input of `first` to the output of
    # `second`. A lift replaces one of them with a wrapper that acts in the
    # fuller space, so the endpoint on the lifted side comes from the wrapper
    # rather than the original transform.
    seq_input = first.input
    seq_output = second.output

    # --- dimensionality mismatch: build the lift ----------------------
    if len(source.axes) != len(target.axes):
        # One transform acts on a subset of the other's axes. Lift the
        # smaller one into the fuller space, trying the fuller input side of
        # `second` first and then the fuller output side of `first`.
        lifted = lift(second, full=source, side="input", extents=extents)
        if lifted is not None:
            pieces: tx.List[Transformation] = [first, lifted]
            # The lifted `second` acts in the fuller space, so the sequence
            # leaves its coordinates in that fuller output system.
            seq_output = lifted.output
        else:
            lifted = lift(first, full=target, side="output", extents=extents)
            if lifted is not None:
                pieces = [lifted, second]
                # The lifted `first` reads the fuller space, so the sequence
                # takes its coordinates from that fuller input system.
                seq_input = lifted.input
            else:
                # Not a same-dimensionality subset on either side, so this
                # is a genuine dimensionality mismatch. `bridge` cannot add
                # or drop an axis, so it raises the descriptive count-mismatch
                # error rather than the adaptor inventing one. The raise that
                # follows carries a message in case `bridge` ever returns.
                bridge(source, target, extents=extents)
                raise AdaptationError(
                    "Cannot compose two transforms whose boundary systems "
                    "have different numbers of axes, when neither acts on a "
                    "clean subset of the other's axes."
                )

    # --- same dimensionality: build the bridge ------------------------
    else:
        reconciler = bridge(
            source,
            target,
            extents=extents,
            allow_positional=allow_positional,
            allow_type_grouped_positional=allow_type_grouped_positional,
        )
        if is_identity(reconciler):
            pieces = [first, second]
        elif isinstance(reconciler, Sequence):
            pieces = [first, *(reconciler.transformations or []), second]
        else:
            pieces = [first, reconciler, second]

    # --- return -------------------------------------------------------
    return Sequence(transformations=pieces, input=seq_input, output=seq_output)


def _subset_positions(
    full_axes: tx.List[Axis], sub_axes: tx.List[Axis]
) -> tx.Optional[tx.List[int]]:
    """
    The position, in `full_axes`, of each axis in `sub_axes`, when
    `sub_axes` is a clean subset of `full_axes`.

    Each subset axis is matched to a fuller axis by the same kernel
    that builds a bridge, so a spatial axis pairs with a spatial axis
    and an oriented axis with its collinear counterpart.

    The result is `None` when a subset axis has no match, or when an
    unmatched fuller axis is of the same kind as an axis the subset
    acts on. The second case is a genuine dimensionality mismatch,
    such as a spatial axis with no spatial counterpart, and is left
    for the caller to refuse rather than absorbed as a pass-through.
    """

    # The grouped-positional fallback is enabled, so a subset of unnamed
    # and unoriented axes of one type still lifts, the same way an
    # implicit bridge pairs such axes. The warning belongs to the bridge
    # built overthe subset below, so the message returned here is discarded.
    match, _ = _match_axes(
        full_axes,
        sub_axes,
        allow_positional=False,
        allow_type_grouped_positional=True,
    )
    if any(i is None for i in match):
        return None
    positions = [int(i) for i in match]
    acted_types = {axis.type for axis in sub_axes}
    used = set(positions)
    for i, axis in enumerate(full_axes):
        if i in used:
            continue
        if axis.type in acted_types:
            return None
    return positions


def _grid_extents(t: Transformation, at_output: bool) -> tx.Dict[tx.Any, int]:
    """
    The number of samples along each named axis of a grid that sits at
    the boundary of a transform, or an empty mapping when the boundary is
    not a grid.

    A reversed array-index axis needs its extent, and a `CartesianField`
    next to the boundary carries it as its shape. The mapping is keyed
    by axis name, so it aligns whichever side of the boundary the grid
    describes.

    `at_output` reads the grid on the output side of the transform,
    and its clearing reads the input side.
    """
    if isinstance(t, CartesianField) and t.shape is not None:
        system = t.output if at_output else t.input
        axes = list(system.axes) if system is not None else []
        extents: tx.Dict[tx.Any, int] = {}
        for axis, size in zip(axes, t.shape):
            name = axis.name
            if name is not None:
                extents[name] = int(size)
        return extents
    if isinstance(t, Sequence) and t.transformations:
        edge = t.transformations[-1] if at_output else t.transformations[0]
        return _grid_extents(edge, at_output)
    return {}


def lift(
    transform: Transformation,
    *,
    full: tx.Optional[CoordinateSystem],
    side: str,
    extents: tx.Optional[Extents] = None,
) -> tx.Optional[SubspaceTransformation]:
    """Lift a transform onto the axes it acts on inside a fuller space.

    The transformation `transform` acts on a subset of the axes of the
    fuller system `full`. This routine wraps `transform` in a
    [`SubspaceTransformation`][] that acts on those axes within `full`
    and leaves the extra axes unchanged. The dimensionality is preserved,
    so a spatial transform meeting a spatial-and-time boundary acts on
    the spatial axes and leaves time untouched.

    The fuller space may lie on either side of the boundary. When `side` is
    `"input"`, `full` is the system a preceding transform hands in, and the
    subset is the input axes of `transform`. When `side` is `"output"`,
    `full` is the system a following transform expects, and the subset is
    the output axes of `transform`.

    The matched axes may still need a reordering, a rescaling, or a flip,
    such as the sign flip between RAS and LPS. That intra-subset bridge is
    built by [`bridge`][] over the subset and placed on the fuller side
    of `transform` inside the wrapper.

    The return value is `None` when the subset is not a clean
    same-dimensionality subset of `full`, so the caller can fall back to
    refusing a genuine dimensionality mismatch. A [`CartesianField`][]
    is never lifted, because it defines a sampling domain rather than a
    transform to place inside a subspace.

    Parameters
    ----------
    transform : Transformation
        The transform to lift into the axis space of `full`.
    full : CoordinateSystem, optional
        The fuller system that the wrapped transform reads and writes.
    side : {"input", "output"}
        Which side of `transform` meets the fuller space.
    extents : sequence or mapping, optional
        The extents needed to reverse an array-index axis within the
        subset, passed through to
        [`bridge`][].

    Returns
    -------
    SubspaceTransformation or None
        The wrapped transform, or `None` when no same-dimensionality
        subset match exists.
    """
    make_bridge = partial(
        bridge, extents=extents, allow_type_grouped_positional=True
    )

    # --- special cases ------------------------------------------------
    if isinstance(transform, CartesianField):
        return None
    if full is None or full.axes is None:
        return None
    sub_input = transform.input
    sub_output = transform.output
    if sub_input is None or sub_input.axes is None:
        return None
    if sub_output is None or sub_output.axes is None:
        return None
    full_axes = list(full.axes)
    in_axes = list(sub_input.axes)
    out_axes = list(sub_output.axes)
    if len(in_axes) != len(out_axes):
        return None
    if len(out_axes if side == "output" else in_axes) >= len(full_axes):
        return None

    # ------------------------------------------------------------------
    if side == "input":
        # The fuller space is on the input side. The wrapper reads the
        # acted-on axes from `full` and a bridge carries them into the
        # frame `transform` expects, sitting before `transform` inside.
        sub_axes = in_axes
    elif side == "output":
        # The fuller space is on the output side. The wrapper writes the
        # acted-on axes into `full`, and a bridge carries the transform's
        # output into the frame `full` uses, sitting after `transform`.
        sub_axes = out_axes
    else:
        raise ValueError("side must be 'input' or 'output'")
    positions = _subset_positions(full_axes, sub_axes)
    if positions is None:
        return None

    # --- discrete check -----------------------------------------------
    # An interpolating transform reads a value off its field at the acted-on
    # coordinates, so it cannot act along a discrete axis, where no value
    # lies between the samples.
    if _interpolates(transform):
        for position in positions:
            axis = full_axes[position]
            if axis.discrete:
                raise AdaptationError(
                    f"Cannot lift an interpolating transform onto the "
                    f"discrete axis {(axis.name or axis.type)!r}. A field "
                    f"is sampled between grid points, which a discrete "
                    f"axis does not allow."
                )

    sub_full = CoordinateSystem(axes=[full_axes[i] for i in positions])

    # --- lift input ---------------------------------------------------
    if side == "input":
        sub_bridge = make_bridge(sub_full, sub_input)
        if is_identity(sub_bridge):
            inner: Transformation = transform
        else:
            inner = Sequence(
                transformations=[sub_bridge, transform],
                input=sub_full,
                output=sub_output,
            )
        # The output system is the fuller system with the acted-on axes
        # replaced by the transform's output axes, kept in place.
        out_full_axes = list(full_axes)
        for k, position in enumerate(positions):
            out_full_axes[position] = out_axes[k]
        wrapper_input = full
        wrapper_output = CoordinateSystem(axes=out_full_axes)

    # --- lift output --------------------------------------------------
    else:
        sub_bridge = make_bridge(sub_output, sub_full)
        if is_identity(sub_bridge):
            inner = transform
        else:
            inner = Sequence(
                transformations=[transform, sub_bridge],
                input=sub_input,
                output=sub_full,
            )
        # The input system is the fuller system with the acted-on axes
        # replaced by the transform's input axes, kept in place.
        in_full_axes = list(full_axes)
        for k, position in enumerate(positions):
            in_full_axes[position] = in_axes[k]
        wrapper_input = CoordinateSystem(axes=in_full_axes)
        wrapper_output = full

    # --- return -------------------------------------------------------
    return SubspaceTransformation(
        transformation=inner,
        input_axes=positions,
        output_axes=positions,
        input=wrapper_input,
        output=wrapper_output,
    )


# The sequence machinery in `transformations.py` reconciles every boundary
# where two adjacent transforms disagree. It calls back into `adapt`, which
# is registered here so the two modules need not import each other at module
# scope.
register_adapt(adapt)
