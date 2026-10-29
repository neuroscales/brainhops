__all__ = [
    "Transformation",
    "MultiscaleField",
    "CoordinateField",
    "CartesianField",
    "DisplacementField",
    "StationaryVelocityField",
    "Affine",
    "AffineExponential",
    "Linear",
    "LinearExponential",
    "Rotation",
    "RotationExponential",
    "Permutation",
    "Scaling",
    "ScalingExponential",
    "Translation",
    "Identity",
    "Inverse",
    "InverseTranslation",
    "InverseScaling",
    "InverseRotation",
    "InversePermutation",
    "InverseLinear",
    "InverseAffine",
    "InverseDisplacementField",
    "InverseCoordinateField",
    "InverseAffineExponential",
    "InverseLinearExponential",
    "InverseRotationExponential",
    "InverseScalingExponential",
    "InverseStationaryVelocityField",
    "Sqrt",
    "Bijection",
    "SubspaceTransformation",
    "Projection",
    "Sequence",
    "MutableSequence",
    "DerivedSequence",
    "ImmutableSequence",
    "ModeLike",
    "SimplifyLike",
    "SimplifyTable",
    "SimplifyPolicy",
    "is_kind",
    "is_identity",
    "is_translation",
    "is_scaling",
    "is_permutation",
    "is_rotation",
    "is_linear",
]

# -- deprecated names ----------------------------------------------------
# The old spellings of the coordinate fields (#547) still import, with a
# DeprecationWarning, for one release.
from brainhops._core._internal_api import (  # noqa: E402
    renamed_attributes as _renamed_attributes,
)

from .base import Transformation

# Imported for their side effect: each of these modules registers its
# implementations with the corresponding dispatcher.
from .compute import adaptors as _adaptors  # noqa: F401, F403
from .compute import checkers as _checkers  # noqa: F401, F403
from .compute import composers as _composers  # noqa: F401, F403
from .compute import converters as _converters  # noqa: F401, F403
from .compute import pruners as _pruners  # noqa: F401, F403
from .compute import simplifiers as _simplifiers  # noqa: F401, F403
from .compute.check import is_kind
from .compute.simplify import SimplifyLike, SimplifyPolicy, SimplifyTable
from .concrete import (
    Affine,
    CartesianField,
    CoordinateField,
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
from .derived import Sqrt
from .inverse import (
    Inverse,
    InverseAffine,
    InverseAffineExponential,
    InverseCoordinateField,
    InverseDisplacementField,
    InverseLinear,
    InverseLinearExponential,
    InversePermutation,
    InverseRotation,
    InverseRotationExponential,
    InverseScaling,
    InverseScalingExponential,
    InverseStationaryVelocityField,
    InverseTranslation,
)
from .meta import Bijection, Projection, SubspaceTransformation
from .modes import ModeLike
from .multiscale import MultiscaleField
from .sequence import (
    DerivedSequence,
    ImmutableSequence,
    MutableSequence,
    Sequence,
)
from .tangents import (
    AffineExponential,
    LinearExponential,
    RotationExponential,
    ScalingExponential,
    StationaryVelocityField,
)

__getattr__ = _renamed_attributes(
    __name__,
    {
        "CoordinatesField": "CoordinateField",
        "InverseCoordinatesField": "InverseCoordinateField",
    },
)
