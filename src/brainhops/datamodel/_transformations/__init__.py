__all__ = [
    # base
    "Transformation",
    # multiscale
    "Multiscale",
    "MultiscaleField",
    # concrete
    "CoordinatesField",
    "CartesianField",
    "DisplacementField",
    "Affine",
    "Linear",
    "Rotation",
    "Permutation",
    "Scaling",
    "Translation",
    "Identity",
    # inverse
    "Inverse",
    "InverseTranslation",
    "InverseScaling",
    "InverseRotation",
    "InversePermutation",
    "InverseLinear",
    "InverseAffine",
    "InverseDisplacementField",
    "InverseCoordinatesField",
    # meta
    "Bijection",
    "SubspaceTransformation",
    "Projection",
    # sequence
    "Sequence",
    "MutableSequence",
    "ImmutableSequence",
    # modes / simplify
    "ModeLike",
    "SimplifyLike",
    "SimplifyTable",
    "SimplifyPolicy",
    "is_kind",
    # checks
    "is_identity",
    "is_translation",
    "is_scaling",
    "is_permutation",
    "is_rotation",
    "is_linear",
    # errors
    "ConversionError",
    "LossyConversionError",
    "CompositionError",
    "AdaptationError",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.lazy import lazy_exports

# The public symbols are resolved on first access. `__getattr__` is
# defined before the registration imports below, so that a module they
# import can already look a symbol up on this (half-imported) package.
__getattr__, __dir__ = lazy_exports(
    __name__,
    globals(),
    {
        "Transformation": ".base",
        "Multiscale": ".multiscale",
        "MultiscaleField": ".multiscale",
        "CoordinatesField": ".concrete",
        "CartesianField": ".concrete",
        "DisplacementField": ".concrete",
        "Affine": ".concrete",
        "Linear": ".concrete",
        "Rotation": ".concrete",
        "Permutation": ".concrete",
        "Scaling": ".concrete",
        "Translation": ".concrete",
        "Identity": ".concrete",
        "Inverse": ".inverse",
        "InverseTranslation": ".inverse",
        "InverseScaling": ".inverse",
        "InverseRotation": ".inverse",
        "InversePermutation": ".inverse",
        "InverseLinear": ".inverse",
        "InverseAffine": ".inverse",
        "InverseDisplacementField": ".inverse",
        "InverseCoordinatesField": ".inverse",
        "Bijection": ".meta",
        "SubspaceTransformation": ".meta",
        "Projection": ".meta",
        "Sequence": ".sequence",
        "MutableSequence": ".sequence",
        "ImmutableSequence": ".sequence",
        "ModeLike": ".modes",
        "SimplifyLike": ".simplify",
        "SimplifyTable": ".simplify",
        "SimplifyPolicy": ".simplify",
        "is_kind": ".check",
        "is_identity": ".concrete",
        "is_translation": ".concrete",
        "is_scaling": ".concrete",
        "is_permutation": ".concrete",
        "is_rotation": ".concrete",
        "is_linear": ".concrete",
        "ConversionError": ".errors",
        "LossyConversionError": ".errors",
        "CompositionError": ".errors",
        "AdaptationError": ".errors",
    },
)

if tx.TYPE_CHECKING:
    from .base import Transformation
    from .check import is_kind
    from .concrete import (
        Affine,
        CartesianField,
        CoordinatesField,
        DisplacementField,
        Identity,
        Linear,
        Permutation,
        Rotation,
        Scaling,
        Translation,
        is_identity,
        is_linear,
        is_permutation,
        is_rotation,
        is_scaling,
        is_translation,
    )
    from .errors import (
        AdaptationError,
        CompositionError,
        ConversionError,
        LossyConversionError,
    )
    from .inverse import (
        Inverse,
        InverseAffine,
        InverseCoordinatesField,
        InverseDisplacementField,
        InverseLinear,
        InversePermutation,
        InverseRotation,
        InverseScaling,
        InverseTranslation,
    )
    from .meta import Bijection, Projection, SubspaceTransformation
    from .modes import ModeLike
    from .multiscale import Multiscale, MultiscaleField
    from .sequence import ImmutableSequence, MutableSequence, Sequence
    from .simplify import SimplifyLike, SimplifyPolicy, SimplifyTable

# Registration into registries: these modules register what they define
# as they are imported, so they cannot be lazy.
from . import adaptors as _adaptors  # noqa: E402, F401
from . import checkers as _checkers  # noqa: E402, F401
from . import composers as _composers  # noqa: E402, F401
from . import converters as _converters  # noqa: E402, F401
from . import restrictors as _restrictors  # noqa: E402, F401
from . import simplifiers as _simplifiers  # noqa: E402, F401
