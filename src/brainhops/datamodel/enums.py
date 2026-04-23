from enum import StrEnum, IntEnum


class BoundaryCondition(StrEnum):
    nearest = edge = border = "nearest"                     # (a a a a | a b c d | d d d d)
    reflect = symmetric = dct2 = "reflect"                  # (d c b a | a b c d | d c b a)
    mirror = dct1 = "mirror"                                #   (d c b | a b c d | c b a)
    gridwrap = circular = circulant = dft = "grid-wrap"     # (a b c d | a b c d | a b c d)
    wrap = "wrap"                                           # (d b c d | a b c d | b c a b)


class InterpolationOrder(IntEnum):
    nearest = 0
    linear = 1
    quadratic = 2
    cubic = 3
    fourth = 4
    fifth = 5
    barycentric = -1
    fourier = -2