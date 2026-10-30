"""Pyramids of scales, and the method that builds their coarser scales.

A multiscale object, such as a
[`MultiScaleImage`][brainhops.datamodel.images.MultiScaleImage] or a
[`MultiscaleField`][brainhops.datamodel.transformations.MultiscaleField],
holds the same image or field at several resolutions, finest first, so
that a viewer or an algorithm that only needs a coarse view reads less
data. [`Multiscale.with_scales`][] gives it the scales that it should
have: as many as the rules allow, a number of them, the factors of each
of them, or a complete list of them. The arithmetic, its modes, its
alignments and its edges are explained in
[`brainhops._core.pyramids`][].

A conversion builds the scales too, with the same values under the
keyword `nscales=` and the keywords `factor=`, `mode=`, `align=`,
`min_size=` and `strict=` of `with_scales`:
`image.to(MultiScaleImage, nscales=4)` calls `with_scales(4)` on the
converted pyramid. A class whose
[`BUILDS_SCALES`][Multiscale.BUILDS_SCALES] is true, such as the
formats of OME-Zarr, builds them even without `nscales=` when it is
converted from a single scale, as `with_scales()` builds them, so that
`image.to(OmeZarrImage)` and `io.save(image, "image.ome.zarr")` write a
full pyramid, and `nscales=1` keeps the single scale.
"""

# stdlib
from collections.abc import Iterable, Mapping
from numbers import Integral

# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly

# core
from brainhops._core.pyramids import (
    DEFAULT_ALIGN,
    DEFAULT_MODE,
    MIN_SIZE,
)
from brainhops._core.typing import ArrayProtocol

# datamodel
from ._multiscale import build_scales, is_factor, plan_scales
from .axes import AxisQuery
from .base import DataModelBase

__all__ = ["SINGLE_SCALE", "AxisReference", "FactorLike", "Multiscale"]

SINGLE_SCALE = tx.TypeVar("SINGLE_SCALE", covariant=True)

AxisReference: tx.TypeAlias = tx.Union[int, AxisQuery]
"""One axis of a pyramid: its position, its name or another axis query."""

FactorLike: tx.TypeAlias = tx.Union[
    int, tx.Sequence[int], tx.Mapping[tx.Optional[AxisReference], int]
]
"""The factor of each axis of one level, in one of three forms.

An int is the factor of every axis that is downsampled by default, which
are the spatial axes, and the other axes get 1. A sequence gives one
factor per axis of the arrays, or one factor per axis that is
downsampled by default, in their order. A mapping gives the factor of
the axes that it names, by position, which counts from the end when it
is negative, or by name, and the other axes get 1, unless the mapping
has the key `None`. The factor of the key `None` goes to every axis that
an int would downsample and that the mapping does not name, so that
`{"z": 1, None: 2}` halves the spatial axes but `z`. It never reaches an
axis that an int leaves alone, such as a time axis or the axis of the
vectors of a field, which a mapping must name to downsample. A factor of
1 leaves its axis as it is.
"""


class Multiscale(DataModelBase, tx.Generic[SINGLE_SCALE]):
    """Mixin for a pyramid of scales, finest first.

    Index 0 is the scale with the highest resolution.

    This base class provides the list of scales, the operations that
    select one of these scales, and the method that gives the pyramid
    its scales, [`with_scales`][].

    It does not define what a scale is: a subclass supplies the scales
    and, through the `_level_resolution` hook, the physical grid size of
    each one, from which `nearest_level` picks the scale whose
    resolution is closest to a target grid. A subclass that builds its
    scales also supplies the hooks `_keeps_scale`, `_check_scales` and
    `_adopt_scales`, and registers its versions of the functions
    `pyramid_layout` and `build_scales` of `brainhops.datamodel._multiscale`.
    """

    BUILDS_SCALES: tx.ClassVar[bool] = False
    """Whether converting a single scale builds the coarser scales.

    When it is true, a conversion to this class from a single-scale
    object builds as many scales as `with_scales()` builds, unless
    `nscales=` says otherwise. The formats of OME-Zarr set it.
    """

    WRITES_SCALES: tx.ClassVar[bool] = False
    """Whether the writer builds the coarser scales of a single scale.

    When it is true, `io.save` gives the options of a pyramid to the
    writer of this class rather than to the conversion, when it saves a
    single scale. The writer then writes the finest scale, and builds
    each coarser scale from the scales that it has written, read back
    from the store. The formats of OME-Zarr set it.
    """

    # --- attributes ---------------------------------------------------

    scales: NotKwOnly[tx.List[SINGLE_SCALE]] = ()
    """Scales, from finest to coarsest."""

    @property
    def nscales(self) -> int:
        """Number of scales."""
        return len(self.scales)

    @property
    def _finest(self) -> SINGLE_SCALE:
        # Return the finest scale, or `None` if there are no scales.
        scales = self.scales
        return scales[0] if scales else None

    # --- methods ------------------------------------------------------

    def to_singlescale(self, index: int = 0) -> SINGLE_SCALE:
        """Return the scale at `index`, where 0 is the finest.

        The scale is returned as it is stored: for a field, it is the plain
        transformation of that scale, not a multiscale field.
        """
        return self.scales[int(index)]

    @tx.overload
    def with_scales(
        self,
        scales: None = None,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(
        self,
        scales: int,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(
        self,
        scales: tx.Sequence[FactorLike],
        *,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
    ) -> tx.Self: ...

    @tx.overload
    def with_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> tx.Self: ...

    def with_scales(
        self,
        scales: tx.Union[
            None, int, tx.Sequence[FactorLike], tx.Sequence[SINGLE_SCALE]
        ] = None,
        *,
        factor: tx.Optional[FactorLike] = None,
        mode: str = DEFAULT_MODE,
        align: str = DEFAULT_ALIGN,
        min_size: int = MIN_SIZE,
        strict: bool = False,
    ) -> tx.Self:
        """Return a copy with the given scales, finest first.

        A pyramid lets a viewer that shows a large image whole, or an
        algorithm that works from coarse to fine, read a coarse scale
        instead of the finest one. This method gives a multiscale object
        the scales that it should have. The finest scale is always kept.

        `scales` is read by its type:

        - `None`, the default, builds as many scales as the rules allow.
        - An int is the number of scales, the finest one included. When
          the rules stop before, fewer scales are built, unless `strict`
          is true.
        - A sequence of factors gives, for each coarser scale, its
          factor from the finest scale along each axis, in a form that
          `FactorLike` describes. The factors are positive integers, and
          the factor of an axis never decreases from a scale to the
          next.
        - A sequence of scales, such as single-scale images for an
          image, is the complete list of scales, finest first, which the
          copy takes as they are. They must be ordered from fine to
          coarse and share one world space.

        The rules build the scales step by step. At each step, an axis
        is downsampled by its factor of `factor`, which is 2 for the
        spatial axes and 1, which leaves an axis as it is, for the other
        axes by default. An axis is downsampled while it has more than
        `min_size` voxels, even when its next scale has fewer, so that
        an axis of 40 voxels becomes an axis of 20 voxels and an axis of
        32 voxels stays as it is. The rules stop when no axis can be
        downsampled. An axis whose voxels are more than `sqrt(f)` times
        larger than the smallest voxels of the axes that can still be
        downsampled waits, so that anisotropic voxels become roughly
        isotropic first.

        A scale is computed from the scale before it whenever its
        factors are multiples of the factors of that scale, and from the
        finest scale otherwise, or always in the mode `"nearest"`, which
        picks the voxels of the finest scale. A coarser scale that the
        object already holds is kept when its factors, its placement and
        its shape are those that are asked for, and the scales after it
        are built from it. The voxels of a scale lie where `align` says,
        and its values are computed as `mode` says, as
        [`brainhops._core.pyramids`][] explains.

        !!! example
            ```python
            pyramid.with_scales()
            pyramid.with_scales(4, factor={"z": 1, None: 2})
            pyramid.with_scales([2, 4, {"x": 8, "y": 8, "z": 4}])
            pyramid.with_scales(3, mode="mode")  # labels
            ```

        Parameters
        ----------
        scales : int or sequence, optional
            The number of scales, the factors of the coarser scales, or
            the list of scales. By default, as many scales as the rules
            allow.
        factor : int or sequence of int or mapping, optional
            The factor of each axis at each step of the rules, in a form
            that `FactorLike` describes. By default, 2 for the spatial
            axes. It is refused with a sequence of `scales`.
        mode : str, default="mean"
            How a coarse voxel is computed from the voxels that it
            covers, one of [`MODES`][brainhops._core.pyramids.MODES].
        align : {"exact", "edges", "centers"}, default="exact"
            How the voxels of a coarse scale are laid over those of the
            finest scale. `"exact"` crops each axis to a multiple of `f`
            and puts the voxel `i` at the voxel `f * i + (f - 1) / 2`,
            which is the rule of OME-NGFF 0.6. `"edges"` keeps the outer
            edges of the field of view, with `ceil(n / f)` voxels, and
            `"centers"` keeps the centres of the first and the last
            voxels, and both pull the voxels by interpolation after
            smoothing them. The modes `"mode"`, `"max"` and `"min"`
            take `"exact"` only.
        min_size : int, default=32
            The number of voxels at or below which the rules stop
            downsampling an axis.
        strict : bool, default=False
            Whether to refuse to build fewer scales than an int `scales`
            asks for.

        Returns
        -------
        Self
            The copy with its scales.

        Raises
        ------
        TypeError
            If `scales` is none of these types, mixes factors and
            scales, or comes with `factor`, or if this class cannot
            build its scales.
        ValueError
            If `scales` is empty, if `strict` is true and the rules stop
            early, if the factors are not positive or decrease, if a
            list of scales is not ordered from fine to coarse or does
            not share one world space, if a factor downsamples the axis
            of the vectors of a field, or if the mode or the alignment
            does not apply.
        KeyError
            If no axis has a name that a factor gives.
        IndexError
            If a position that a factor gives is out of range.
        """
        if (
            scales is not None
            and not isinstance(scales, (Integral, str, bytes, Mapping))
            and isinstance(scales, Iterable)
        ):
            scales = list(scales)
            if scales and not any(is_factor(item) for item in scales):
                if factor is not None:
                    raise TypeError(
                        "`factor` gives the step of the rules, which a "
                        "sequence of scales replaces."
                    )
                self._check_scales(scales)
                return self._adopt_scales(scales)
        plan = plan_scales(
            self,
            {
                "nscales": scales,
                "factor": factor,
                "mode": mode,
                "align": align,
                "min_size": min_size,
                "strict": strict,
            },
        )
        kept = 0
        for index, level in enumerate(plan.factors, start=1):
            if index >= self.nscales:
                break
            if not self._keeps_scale(index, level, align):
                break
            kept = index
        return build_scales(self, plan.factors, mode, align, kept)

    # --- hooks --------------------------------------------------------

    def _level_resolution(self, index: int) -> tx.Optional[ArrayProtocol]:
        # Return the physical grid size of a level, as a vector with one entry
        # per axis in the input units, or `None` if the size is unknown. This
        # hook returns `None` here, and subclasses override it.
        return None

    def _keeps_scale(
        self, index: int, factors: tx.Tuple[int, ...], align: str
    ) -> bool:
        # Return whether the scale at `index` lies where `align` puts a
        # scale of `factors` from the finest scale, and has its shape.
        return False

    def _check_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> None:
        # Refuse a list of scales that is not ordered from fine to
        # coarse, or whose scales do not share one world space.
        return None

    def _adopt_scales(self, scales: tx.Sequence[SINGLE_SCALE]) -> tx.Self:
        # Return a copy that holds `scales` as they are.
        raise TypeError(
            f"A {type(self).__name__} cannot be given a list of scales."
        )
