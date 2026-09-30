# dependencies
import typing_extensions as tx

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation


class AdaptationError(TypeError):
    """Raised when one coordinate system cannot be adapted to another.

    Adaptation reorders, rescales, and flips the axes that two coordinate
    systems share, so it succeeds only when every axis of one system
    corresponds to an axis of the other. This error is raised when an axis
    that must be matched has no correspondence, or when two matched axes
    carry incompatible units. Its message names the two systems and the
    axes that could not be reconciled.
    """


class CompositionError(TypeError):
    """Raised when two transformations cannot be composed."""


class ConversionError(TypeError):
    """Raised when a transformation cannot be converted to another type."""


class RestrictionError(TypeError):
    """Raised when a transformation cannot be restricted to a block of axes.

    Either no restriction rule applies to its type, or the block cannot be
    cut out of it soundly (a chain inside which the block is coupled to the
    other axes, or a transform with no affine reading). The factor pass
    leaves a chain unfactored when this is raised, rather than drop a
    piece it cannot restrict.
    """


class LossyConversionError(ConversionError):
    """Raised when a conversion would discard information.

    This error is raised by [`Transformation.to`][] when a conversion is
    only possible at the cost of losing information, and the conversion
    was not explicitly allowed to be lossy. The `result` attribute holds
    the transformation that the conversion would have produced.
    """

    def __init__(
        self,
        *args,
        result: tx.Optional["Transformation"] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.result = result
