"""Enumerations shared across the data model, such as interpolation orders
and boundary conditions."""

__all__ = [
    "BoundaryCondition",
    "InterpolationOrder",
    "OrientationType",
    "AnatomicalOrientationValue",
]

from brainhops._core.enum import IntEnum, StrEnum


# ruff: disable[E501]
# fmt: off
class BoundaryCondition(StrEnum):
    """Boundary conditions for interpolation and resampling.

    The following boundary conditions are supported:

    | Value         | Aliases                        | Description                     |
    |---------------|--------------------------------|---------------------------------|
    | `nearest`     | `edge`, `border`               | <code>(a a a a &vert; a b c d &vert; d d d d)</code> |
    | `reflect`     | `symmetric`, `dct2`            | <code>(d c b a &vert; a b c d &vert; d c b a)</code> |
    | `mirror`      | `dct1`                         | <code>  (d c b &vert; a b c d &vert; c b a)  </code> |
    | `grid-wrap`   | `circular`, `circulant`, `dft` | <code>(a b c d &vert; a b c d &vert; a b c d)</code> |
    | `wrap`        |                                | <code>(d b c d &vert; a b c d &vert; b c a b)</code> |
    | `constant`    | `zero`, `zeros`                | <code>(0 0 0 0 &vert; a b c d &vert; 0 0 0 0)</code> |
    """

    nearest = edge = border = "nearest"                     # (a a a a | a b c d | d d d d)
    reflect = symmetric = dct2 = "reflect"                  # (d c b a | a b c d | d c b a)
    mirror = dct1 = "mirror"                                #   (d c b | a b c d | c b a)
    gridwrap = circular = circulant = dft = "grid-wrap"     # (a b c d | a b c d | a b c d)
    wrap = "wrap"                                           # (d b c d | a b c d | b c a b)
    constant = zero = zeros = "constant"                    # (0 0 0 0 | a b c d | 0 0 0 0)
# fmt: on
# ruff: enable[E501]


class InterpolationOrder(IntEnum):
    """Interpolation order for interpolation and resampling.

    The following interpolation orders are supported:

    | Name          | Aliases     | Value | Description                     |
    |---------------|-------------|-------|---------------------------------|
    | `zeroth`      | `nearest`   | 0     | Nearest neighbor interpolation. |
    | `first`       | `linear`    | 1     | Linear interpolation.           |
    | `second`      | `quadratic` | 2     | Quadratic interpolation.        |
    | `third`       | `cubic`     | 3     | Cubic interpolation.            |
    | `fourth`      |             | 4     | Fourth-order interpolation.     |
    | `fifth`       |             | 5     | Fifth-order interpolation.      |
    | `barycentric` |             | -1    | Barycentric interpolation.      |
    | `fourier`     |             | -2    | Fourier interpolation.          |

    """

    zeroth = nearest = 0
    first = linear = 1
    second = quadratic = 2
    third = cubic = 3
    fourth = 4
    fifth = 5
    barycentric = -1
    fourier = -2


class OrientationType(StrEnum):
    """
    Orientation types for coordinate systems and transformations.

    Currently, only the `"anatomical"` orientation is supported.

    | Name          | Value          | Description             |
    |---------------|----------------|-------------------------|
    | `anatomical`  | `"anatomical"` | Anatomical orientation. |
    """

    anatomical = "anatomical"


# ruff: disable[E501]
# fmt: off
class AnatomicalOrientationValue(StrEnum):
    """
    Anatomical orientation values for coordinate systems and transformations.

    The following values are supported:

    | Name                    | Value                     | Description             |
    |-------------------------|---------------------------|-------------------------|
    | `left_to_right`         | `"left-to-right"`         |
    | `right_to_left`         | `"right-to-left"`         |
    | `proximal_to_distal`    | `"proximal-to-distal"`    |
    | `distal_to_proximal`    | `"distal-to-proximal"`    |
    | `anterior_to_posterior` | `"anterior-to-posterior"` | front-to-back
    | `posterior_to_anterior` | `"posterior-to-anterior"` | back-to-front
    | `inferior_to_superior`  | `"inferior-to-superior"`  | feet-to-head
    | `superior_to_inferior`  | `"superior-to-inferior"`  | head-to-feet
    | `dorsal_to_palmar`      | `"dorsal-to-palmar"`      | back of hand to palm
    | `palmar_to_dorsal`      | `"palmar-to-dorsal"`      | palm to back of hand
    | `dorsal_to_plantar`     | `"dorsal-to-plantar"`     | top of foot to sole
    | `plantar_to_dorsal`     | `"plantar-to-dorsal"`     | sole to top of foot
    | `rostral_to_caudal`     | `"rostral-to-caudal"`     | nose/beak-to-tail, especially for nervous system
    | `caudal_to_rostral`     | ` "caudal-to-rostral"`    | tail-to-nose/beak, especially for nervous system
    | `cranial_to_caudal`     | `"cranial-to-caudal"`     | head-to-tail
    | `caudal_to_cranial`     | `"caudal-to-cranial"`     | tail-to-head
    | `dorsal_to_ventral`     | `"dorsal-to-ventral"`     | back/top-to-belly/bottom
    | `ventral_to_dorsal`     | `"ventral-to-dorsal"`     | belly/bottom-to-back/top
    | `superficial_to_deep`   | `"superficial-to-deep"`   | outer surface to inner depth, e.g. skin, gut, cortex
    | `deep_to_superficial`   | `"deep-to-superficial"`   | inner depth to outer surface
    | `apical_to_basal`       | `"apical-to-basal"`       | apical to basal surface, e.g. epithelial layers, polarized cells
    | `basal_to_apical`       | `"basal-to-apical"`       | basal to apical surface
    | `apex_to_base`          | `"apex-to-base"`          | tip to broad base, e.g. heart, lungs
    | `base_to_apex`          | `"base-to-apex"`          | broad base to tip
    """
    # Common to both bipeds and quadrupeds
    left_to_right = "left-to-right"
    right_to_left = "right-to-left"
    proximal_to_distal = "proximal-to-distal"
    distal_to_proximal = "distal-to-proximal"

    # Primarily for bipeds (humans)
    anterior_to_posterior = "anterior-to-posterior"
    posterior_to_anterior = "posterior-to-anterior"
    inferior_to_superior = "inferior-to-superior"
    superior_to_inferior = "superior-to-inferior"
    dorsal_to_palmar = "dorsal-to-palmar"
    palmar_to_dorsal = "palmar-to-dorsal"
    dorsal_to_plantar = "dorsal-to-plantar"
    plantar_to_dorsal = "plantar-to-dorsal"

    # Primarily for quadrupeds:
    rostral_to_caudal = "rostral-to-caudal"
    caudal_to_rostral = "caudal-to-rostral"
    cranial_to_caudal = "cranial-to-caudal"
    caudal_to_cranial = "caudal-to-cranial"
    dorsal_to_ventral = "dorsal-to-ventral"
    ventral_to_dorsal = "ventral-to-dorsal"

    # For layered and polarized tissues (subject-local):
    superficial_to_deep = "superficial-to-deep"
    deep_to_superficial = "deep-to-superficial"
    apical_to_basal = "apical-to-basal"
    basal_to_apical = "basal-to-apical"
    apex_to_base = "apex-to-base"
    base_to_apex = "base-to-apex"
# fmt: on
# ruff: enable[E501]
