__all__ = ["BoundaryCondition", "InterpolationOrder"]

from enum import IntEnum, StrEnum


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
    """Orientation types for coordinate systems and transformations."""

    anatomical = "anatomical"
