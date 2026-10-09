"""Bridges between coordinate systems at a composition boundary.

Two consecutive transformations compose only when the output system of
the first describes the same frame as the input system of the second.
When the two systems disagree, this module builds an exact, invertible
transformation that reorders, rescales and flips axes to carry
coordinates from one frame to the other, without resampling. The bridge
is derived from the axis descriptions alone. Axes are matched by type
and orientation, then by name, then by kind of unit, and finally by
position when positional pairing is permitted. Axes of different types
never match.

[`bridge`][] maps one coordinate system onto another with the same
number of axes. [`adapt`][] takes two consecutive transformations and
returns a sequence that contains both, with a bridge at the junction.
When one transformation acts on a subset of the other's axes,
[`adapt`][] instead embeds it in the fuller space, where it leaves the
extra axes unchanged.
"""

__all__ = ["bridge", "adapt"]

import warnings
from functools import partial

import numpy as np
import typing_extensions as tx

from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import (
    ArrayCoordinateSystem,
    CoordinateSystem,
)
from brainhops.datamodel.units import Unit, is_indexunit, is_physicalunit
from brainhops.errors import AdaptationError

from ..base import Transformation
from ..concrete import (
    CartesianField,
    Identity,
    Permutation,
    Scaling,
    Translation,
)
from ..meta import SubspaceTransformation
from ..nocycles import register_adapt
from ..sequence import Sequence, _interpolates
from .utils import systems_disagree

Extents = tx.Union[tx.Sequence[tx.Optional[int]], tx.Mapping[tx.Any, int]]
"""Sample counts, by target axis position or by axis name.

Extents are needed only to reverse an array-index axis, because the
flip shifts the origin by the extent minus one.
"""


def bridge(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
    *,
    extents: tx.Optional[Extents] = None,
    allow_positional: bool = False,
    allow_type_grouped_positional: bool = False,
) -> Transformation:
    """Return the transformation that carries `source` coordinates to `target`.

    The bridge maps every point expressed in `source` to the same point
    expressed in `target`; its inverse is `bridge(target, source)`. Axes are
    matched by type and orientation, then by name, then by kind of unit,
    and finally by position when permitted. The match yields a
    [`Permutation`][], a [`Scaling`][] by the unit ratio and the
    orientation sign, and, when a reversed axis indexes an array, a
    [`Translation`][] by the extent minus one.

    A missing system, or an open system whose axes hold `...`, describes
    only some of its axes. The bridge is then an [`Identity`][] when the two
    systems are
    [`compatible_with`][brainhops.datamodel.systems.CoordinateSystem.compatible_with]
    each other, and an [`AdaptationError`][] is raised otherwise.

    Parameters
    ----------
    source : CoordinateSystem or None
        Coordinate system on the input side of the bridge.
    target : CoordinateSystem or None
        Coordinate system on the output side of the bridge.
    extents : Extents, optional
        Number of samples along each target axis, needed only to reverse
        an array-index axis.
    allow_positional : bool, default=False
        Pair the remaining axes by position, when both systems have the
        same number of axes. This option takes precedence over
        `allow_type_grouped_positional`.
    allow_type_grouped_positional : bool, default=False
        Pair the remaining axes by their order within each type group. The
        typeless group acts as a wildcard for the single remaining typed
        group of the same size, and groups whose sizes differ on the two
        sides stay unmatched.

    Returns
    -------
    Transformation
        A single primitive, a [`Sequence`][] of primitives, or an
        [`Identity`][].

    Raises
    ------
    AdaptationError
        If the systems have different numbers of axes, if an axis remains
        unmatched, or if a matched pair cannot be aligned by a flip and a
        rescaling.

    Warns
    -----
    UserWarning
        If any pair of axes was formed by position.
    """
    # Only two closed systems can be bridged. A missing system is open as well,
    # since its axes read `[...]`.
    source_axes = get_axes(source)
    target_axes = get_axes(target)
    if source_axes.is_open or target_axes.is_open:
        if source_axes.compatible_with(target_axes):
            return Identity(input=source, output=target)
        _cannot_bridge_report(source, target)
    if source == target:
        return Identity(input=source, output=target)

    if source.ndim != target.ndim:
        source_name = source.name or "the source system"
        target_name = target.name or "the target system"
        raise AdaptationError(
            f"Cannot bridge {source_name} to {target_name}: "
            f"the two systems have different numbers of axes. A bridge "
            f"reorders, rescales, and flips matched axes and never adds "
            f"or drops one. When one transform acts on a subset of the "
            f"other's axes, such as a spatial transform meeting a "
            f"spatial-and-time image, composition embeds the smaller "
            f"transform in the axes it acts on. Those axes could not "
            f"be matched to a subset of the fuller system here, so the "
            f"embedding was not possible. Give the shared axes matching "
            f"names, units, or orientations so they can be identified."
        )

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

    # Warn only once every pair has a sign, a ratio and an offset, so that a
    # bridge that fails above does not also warn. The stack level points at the
    # caller of `bridge`.
    if positional_warning is not None:
        warnings.warn(positional_warning, stacklevel=2)

    # Each primitive names its own endpoints, and only the last one lands on
    # `target`. The permutation lands on the source axes in target order, and a
    # scaling followed by a shift lands on the target axes before the shift.
    permuted_axes = [source_axes[i] for i in permutation]
    permuted_system = CoordinateSystem(axes=permuted_axes)
    needs_scale = any(scale != 1 for scale in scales)
    needs_offset = any(offset != 0 for offset in translations)
    needs_perm = permutation != list(range(len(permutation)))

    elements: tx.List[Transformation] = []
    current = source
    if needs_perm:
        elements.append(
            Permutation(
                permutation=np.asarray(permutation),
                input=source,
                output=permuted_system,
            )
        )
        current = permuted_system
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
    if needs_offset:
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


def _match_axes(
    source_axes: tx.List[Axis],
    target_axes: tx.List[Axis],
    allow_positional: bool,
    allow_type_grouped_positional: bool = False,
) -> tx.Tuple[tx.List[tx.Optional[int]], tx.Optional[str]]:
    """Match each target axis to the source axis that feeds it.

    Returns `(match, warning)`, where `match[j]` is the index of the source
    axis matched to target axis `j`, or None, and `warning` describes a
    positional pairing, or is None. The warning is returned so that the
    caller can issue it once the pairing can no longer fail.
    """

    # Axes with different types never match: a shared name, unit or position is
    # a coincidence when the types definitely conflict.
    n_source, n_target = len(source_axes), len(target_axes)
    match: tx.List[tx.Optional[int]] = [None] * n_target
    used = [False] * n_source
    warning: tx.Optional[str] = None

    def take(j: int, i: int) -> None:
        match[j] = i
        used[i] = True

    # Tier 1: same type on the same orientation line, the strongest signal. It
    # pairs right-to-left with left-to-right whatever the names.
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

    # Tier 2: same name, but never across different types or orientation lines,
    # because a name cannot turn a rotation into a flip.
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
            take(j, candidates[0])

    # Tier 3: same kind of unit, when exactly one unused source axis qualifies.
    # Pairs across types or orientation lines are excluded.
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
            take(j, candidates[0])

    # Last tier: position. Positional pairing may hide a genuine mismatch, so
    # it is opt-in and produces a warning.
    if allow_positional and n_source == n_target:
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
        # Implicit bridge: pair the remaining axes in order within each type
        # group, leaving ambiguous groups unmatched rather than partially
        # paired.
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
        axis for axis, i in zip(get_axes(target), match) if i is None
    ]
    unmatched_source = [
        axis
        for i, axis in enumerate(get_axes(source))
        if i not in matched_source
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


def _cannot_bridge_report(
    source: tx.Optional[CoordinateSystem],
    target: tx.Optional[CoordinateSystem],
) -> tx.NoReturn:
    source_name = getattr(source, "name", None) or "the source system"
    target_name = getattr(target, "name", None) or "the target system"
    which = "the source" if get_axes(source).is_open else "the target"
    raise AdaptationError(
        f"Cannot bridge {source_name} to {target_name}: {which} system is "
        f"open (its axes hold `...`), and the axes it states do not match "
        f"the other system's. A bridge would have to reorder, rescale, or "
        f"flip axes that the open system does not describe. Declare the "
        f"full systems, or close the open one with CoordinateSystem.expand."
    )


@register_adapt
def adapt(
    first: Transformation,
    second: Transformation,
    *,
    extents: tx.Optional[Extents] = None,
    allow_positional: bool = False,
    allow_type_grouped_positional: bool = False,
) -> Sequence:
    """Reconcile two consecutive transformations at their boundary.

    `first` is applied before `second`, so the boundary lies between
    `first.output` and `second.input`. When the two systems agree, the
    transformations follow each other directly. When they have the same
    number of axes, a [`bridge`][] is placed between them. When they have
    different numbers of axes, the transformation with fewer axes is
    wrapped in a [`SubspaceTransformation`][] that acts on the matching
    axes of the fuller space, for example a spatial transformation meeting
    a boundary with space and time. The sequence then begins or ends in
    the fuller system of the wrapper. If neither transformation acts on a
    clean subset of the other's axes, an [`AdaptationError`][] is raised.

    Parameters
    ----------
    first : Transformation
        Transformation applied first.
    second : Transformation
        Transformation applied after `first`.
    extents : Extents, optional
        Sample counts passed to [`bridge`][]. When omitted, they are read
        from a grid on either side of the boundary.
    allow_positional : bool, default=False
        Passed to [`bridge`][] when the boundary systems have the same
        number of axes.
    allow_type_grouped_positional : bool, default=False
        Passed to [`bridge`][] when the boundary systems have the same
        number of axes.

    Returns
    -------
    Sequence
        A sequence that contains `first` and `second`, which are reused
        unchanged unless one of them is embedded.
    """
    source = first.output
    target = second.input

    if not systems_disagree(source, target):
        return Sequence(
            transformations=[first, second],
            input=first.input,
            output=second.output,
        )

    if extents is None:
        extents = _grid_extents(first, at_output=True)
        extents.update(_grid_extents(second, at_output=False))
    extents = extents or None

    # An embedding replaces one transformation with a wrapper in the fuller
    # space, so the endpoint on that side comes from the wrapper.
    seq_input = first.input
    seq_output = second.output

    # Only two closed systems can differ in their number of axes. An open
    # system that disagrees is refused by `bridge` below.
    n_source, n_target = (
        get_axes(source).ndim,
        get_axes(target).ndim,
    )
    if n_source is not None and n_target is not None and n_source != n_target:
        # Try the fuller input side of `second` first, then the fuller output
        # side of `first`.
        embedded = embed(second, full=source, side="input", extents=extents)
        if embedded is not None:
            pieces: tx.List[Transformation] = [first, embedded]
            seq_output = embedded.output
        else:
            embedded = embed(
                first, full=target, side="output", extents=extents
            )
            if embedded is not None:
                pieces = [embedded, second]
                seq_input = embedded.input
            else:
                # Neither side is a clean subset, so the mismatch is genuine.
                # `bridge` raises its descriptive error about the axis counts;
                # the error below is only a fallback.
                bridge(source, target, extents=extents)
                raise AdaptationError(
                    "Cannot compose two transforms whose boundary systems "
                    "have different numbers of axes, when neither acts on a "
                    "clean subset of the other's axes."
                )

    else:
        reconciler = bridge(
            source,
            target,
            extents=extents,
            allow_positional=allow_positional,
            allow_type_grouped_positional=allow_type_grouped_positional,
        )
        if reconciler.is_identity():
            pieces = [first, second]
        elif isinstance(reconciler, Sequence):
            pieces = [first, *(reconciler.transformations or []), second]
        else:
            pieces = [first, reconciler, second]

    return Sequence(transformations=pieces, input=seq_input, output=seq_output)


def embed(
    transform: Transformation,
    *,
    full: tx.Optional[CoordinateSystem],
    side: str,
    extents: tx.Optional[Extents] = None,
) -> tx.Optional[SubspaceTransformation]:
    """Embed a transformation in the axes it acts on within a fuller space.

    The transformation is wrapped in a [`SubspaceTransformation`][] that
    acts on the matching axes of `full` and leaves the other axes
    unchanged. With `side="input"`, `full` is the system handed in by a
    preceding transformation and is matched against the input axes of
    `transform`; with `side="output"`, `full` is the system expected by a
    following transformation and is matched against the output axes. A
    [`bridge`][] over the subset, for example a flip between RAS and LPS,
    is placed inside the wrapper on the fuller side of `transform`.

    Parameters
    ----------
    transform : Transformation
        Transformation to embed.
    full : CoordinateSystem or None
        Fuller system that the wrapper reads from or writes to.
    side : {"input", "output"}
        Side of `transform` on which the fuller space lies.
    extents : Extents, optional
        Sample counts passed to [`bridge`][].

    Returns
    -------
    SubspaceTransformation or None
        The wrapper, or None when `transform` is a [`CartesianField`][]
        (which defines a sampling domain), when a system is open, when the
        numbers of input and output axes differ, or when the axes do not
        form a clean subset of `full`.

    Raises
    ------
    ValueError
        If `side` is neither `"input"` nor `"output"`.
    AdaptationError
        If an interpolating transformation would act on a discrete axis.
    """
    make_bridge = partial(
        bridge, extents=extents, allow_type_grouped_positional=True
    )

    if isinstance(transform, CartesianField):
        return None
    # The embedding is read from the axes that the fuller system has and the
    # subset lacks, so every system must be closed.
    sub_input = transform.input
    sub_output = transform.output
    full_axes = get_axes(full)
    in_axes = get_axes(sub_input)
    out_axes = get_axes(sub_output)
    if any(axes.is_open for axes in (full_axes, in_axes, out_axes)):
        return None
    if len(in_axes) != len(out_axes):
        return None
    if len(out_axes if side == "output" else in_axes) >= len(full_axes):
        return None

    # ------------------------------------------------------------------
    if side == "input":
        # The wrapper reads the acted-on axes from `full`, and a bridge before
        # `transform` carries them into the frame that `transform` expects.
        sub_axes = in_axes
    elif side == "output":
        # The wrapper writes the acted-on axes into `full`, and a bridge after
        # `transform` carries its output into the frame of `full`.
        sub_axes = out_axes
    else:
        raise ValueError("side must be 'input' or 'output'")
    positions = _subset_positions(full_axes, sub_axes)
    if positions is None:
        return None

    # An interpolating transformation reads values between samples, which a
    # discrete axis does not have.
    if _interpolates(transform):
        for position in positions:
            axis = full_axes[position]
            if axis.discrete:
                raise AdaptationError(
                    f"Cannot embed an interpolating transform in the "
                    f"discrete axis {(axis.name or axis.type)!r}. A field "
                    f"is sampled between grid points, which a discrete "
                    f"axis does not allow."
                )

    sub_full = CoordinateSystem(axes=[full_axes[i] for i in positions])

    if side == "input":
        sub_bridge = make_bridge(sub_full, sub_input)
        if sub_bridge.is_identity():
            inner: Transformation = transform
        else:
            inner = Sequence(
                transformations=[sub_bridge, transform],
                input=sub_full,
                output=sub_output,
            )
        out_full_axes = list(full_axes)
        for k, position in enumerate(positions):
            out_full_axes[position] = out_axes[k]
        wrapper_input = full
        wrapper_output = CoordinateSystem(axes=out_full_axes)

    else:
        sub_bridge = make_bridge(sub_output, sub_full)
        if sub_bridge.is_identity():
            inner = transform
        else:
            inner = Sequence(
                transformations=[transform, sub_bridge],
                input=sub_input,
                output=sub_full,
            )
        in_full_axes = list(full_axes)
        for k, position in enumerate(positions):
            in_full_axes[position] = in_axes[k]
        wrapper_input = CoordinateSystem(axes=in_full_axes)
        wrapper_output = full

    return SubspaceTransformation(
        transformation=inner,
        input_axes=positions,
        output_axes=positions,
        input=wrapper_input,
        output=wrapper_output,
    )


def _subset_positions(
    full_axes: tx.List[Axis], sub_axes: tx.List[Axis]
) -> tx.Optional[tx.List[int]]:
    """Return the position in `full_axes` of each axis of `sub_axes`.

    The axes are matched as in [`bridge`][], so spatial axes match spatial
    axes and oriented axes match collinear ones. The result is None when an
    axis of the subset has no match, or when an unmatched axis of the
    fuller system has the same type as an axis of the subset. The second
    case is a genuine mismatch, such as a spatial axis without a spatial
    counterpart, and is left for the caller to refuse instead of being
    absorbed as a pass-through axis.
    """

    # Grouped positional pairing lets a subset of unnamed, unoriented axes of
    # one type embed, as an implicit bridge would. The warning belongs to the
    # bridge built over the subset, so it is discarded here.
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


_OrientationLike = tx.Union[Axis, Orientation, str, None]


def _orientation(orientation: _OrientationLike) -> tx.Optional[str]:
    """Return the orientation value of an axis or orientation."""
    if isinstance(orientation, Axis):
        orientation = orientation.orientation
    if isinstance(orientation, Orientation):
        orientation = orientation.value
    return orientation


def _orientation_line(
    orientation: _OrientationLike,
) -> tx.Optional[tx.FrozenSet[str]]:
    """Return the orientation line of an axis, ignoring its direction.

    The line is the unordered pair of the two poles of the orientation. An
    orientation that cannot be split into two poles gives a singleton, and
    an unset orientation gives None.

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
    """Return whether two axes conflict by orientation.

    Two axes conflict when both orientations are set and lie on different
    lines. An unset orientation never conflicts.
    """
    source_line = _orientation_line(source)
    target_line = _orientation_line(target)
    if source_line is None or target_line is None:
        return False
    return source_line != target_line


def _orientation_sign(
    source: _OrientationLike, target: _OrientationLike
) -> int:
    """Return the sign that aligns two collinear axes.

    The sign is +1 when the axes point in the same direction, or when
    either orientation is unset, and -1 when they point in opposite
    directions.

    Raises
    ------
    AdaptationError
        If the axes lie on different orientation lines, since aligning
        them would require a rotation.
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


_UnitLike = tx.Union[Axis, Unit, None]


def _unit(unit: _UnitLike) -> tx.Optional[Unit]:
    """Return the physical unit of an axis or unit.

    The result is None when there is no unit, when the value is not a
    [`Unit`][], or when the axis is measured in samples, since a sample has
    neither a scale nor a kind. Sampled axes are detected separately by
    [`_is_sampled`][].
    """
    if isinstance(unit, Axis):
        unit = unit.unit
    if is_physicalunit(unit):
        return unit
    return None


def _is_sampled(unit: _UnitLike) -> bool:
    """Return whether an axis or unit counts samples."""
    if isinstance(unit, Axis):
        unit = unit.unit
    return is_indexunit(unit)


def _unit_ratio(source: _UnitLike, target: _UnitLike) -> float:
    """Return the factor that converts the source unit into the target unit.

    An unspecified unit is compatible with any unit and gives a ratio of
    one, since an unknown unit is never grounds for refusal. The same unit,
    or two units that both count samples, also give a ratio of one.

    Raises
    ------
    AdaptationError
        If a sampled axis is matched to a physical unit, since the size of
        a sample is stated by a scaling rather than by the axes, or if the
        two physical units are of different kinds, such as a length and a
        duration.
    """
    source_sampled, target_sampled = _is_sampled(source), _is_sampled(target)
    source_unit, target_unit = _unit(source), _unit(target)
    if source_sampled and target_sampled:
        return 1.0
    if source_sampled or target_sampled:
        physical = target_unit if source_sampled else source_unit
        if physical is None:
            return 1.0
        raise AdaptationError(
            f"One axis is sampled (its unit is an index unit, such as "
            f"'index' or 'voxel') and the axis it matches is in "
            f"{physical.name}, so no conversion factor exists "
            f"between them: the size of a sample in {physical.name} is what "
            f"a scaling transformation states, not the axes. Map the "
            f"samples to {physical.name} with a transformation, or give "
            f"both axes the same kind of unit."
        )
    if source_unit is None or target_unit is None:
        return 1.0
    if source_unit.type != target_unit.type:
        raise AdaptationError(
            f"Two matched axes are measured in units of different kinds "
            f"({source_unit.name} and {target_unit.name}), so no conversion "
            f"factor exists between them."
        )
    # For SI-prefixed units, the ratio is a power of ten computed from the
    # exponents, so that mm to um is exactly 1000 and the product of a ratio
    # and its reciprocal is exactly one; dividing the scales does not guarantee
    # that. Other units, such as the inch, fall back to division.
    source_log10 = getattr(source_unit, "log10_scale", None)
    target_log10 = getattr(target_unit, "log10_scale", None)
    if isinstance(source_log10, int) and isinstance(target_log10, int):
        return 10 ** (source_log10 - target_log10)
    return float(source_unit.scale) / float(target_unit.scale)


def _same_unit_kind(source: _UnitLike, target: _UnitLike) -> bool:
    """Return whether two axes have physical units of the same kind.

    Units of the same kind, such as two lengths, have a defined conversion
    factor. An axis without a unit never matches by unit.
    """
    source_unit = _unit(source)
    target_unit = _unit(target)
    if source_unit is None or target_unit is None:
        return False
    return source_unit.type == target_unit.type


def _is_array_side(system: tx.Optional[CoordinateSystem], axis: Axis) -> bool:
    """Return whether an axis indexes an array rather than a world coordinate.

    Reversing an array-index axis shifts the origin by the extent minus
    one, whereas reversing a world axis only flips its sign. An axis
    indexes an array when its system is an array coordinate system, when
    it is discrete, or when its unit is an
    [`IndexUnit`][brainhops.datamodel.units.IndexUnit]. An unspecified unit
    is not read as an array index.
    """
    if isinstance(system, ArrayCoordinateSystem):
        return True
    if axis.discrete:
        return True
    return is_indexunit(axis.unit)


def _extent(extents: tx.Optional[Extents], position: int, axis: Axis) -> int:
    """Return the number of samples along an axis, by position or by name.

    Raises
    ------
    AdaptationError
        If the extent is not available, since a reversed array-index axis
        cannot be bridged without it.
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


def _grid_extents(t: Transformation, at_output: bool) -> tx.Dict[tx.Any, int]:
    """Return the sample counts of a grid at the boundary of a transformation.

    A [`CartesianField`][] next to the boundary carries the extents as its
    shape. The counts are keyed by axis name and read on the output side
    when `at_output` is true, or on the input side otherwise. A
    [`Sequence`][] is searched through its element on that side, and the
    mapping is empty when no grid is found.
    """
    if isinstance(t, CartesianField) and t.shape is not None:
        axes = get_axes(t.output if at_output else t.input)
        if axes.is_open:
            axes = axes.expand(len(t.shape))
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


_TypeLike = tx.Union[Axis, str, None]
_TypeGroup = tx.Dict[tx.Optional[str], tx.List[int]]


def _type(axis: _TypeLike) -> tx.Optional[str]:
    if isinstance(axis, Axis):
        return axis.type
    if isinstance(axis, str):
        return axis
    return None


def _type_conflict(source: _TypeLike, target: _TypeLike) -> bool:
    """Return whether two axes carry definite and different types.

    An unset type places no constraint, so a conflict requires both types
    to be set.
    """
    source_type = _type(source)
    target_type = _type(target)
    if source_type is None or target_type is None:
        return False
    return source_type != target_type


def _group_by_type(
    axes: tx.List[_TypeLike], indices: tx.List[int]
) -> _TypeGroup:
    """Group axis indices by axis type, keeping the given order.

    Typeless axes form the group keyed by None.
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
    """Pair unmatched axes by their order within type groups.

    A group with a definite type pairs with the group of the same type on
    the other side. The typeless group acts as a wildcard and pairs with the
    single remaining typed group of the same size on the other side. A
    group is paired only when the same number of axes remains on each side.
    The pairs are returned as `(target_index, source_index)`.
    """

    # Both mappings are consumed as pairs are made; whatever remains has no
    # counterpart and is reported by the caller.
    pairs: tx.List[tx.Tuple[int, int]] = []
    for key in list(target_groups):
        sources = source_groups.get(key)
        if sources is not None and len(sources) == len(target_groups[key]):
            targets = target_groups.pop(key)
            del source_groups[key]
            for j, i in zip(targets, sources):
                pairs.append((j, i))

    # A leftover typeless group is a wildcard, on either side, for the single
    # typed group of the same size on the other side. When several typed groups
    # or none qualify, the typeless group stays unpaired.
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
