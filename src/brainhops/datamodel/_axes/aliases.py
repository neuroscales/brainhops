__all__ = ["A", "AP", "I", "IS", "L", "LR", "P", "PA", "R", "RL", "S", "SI"]

import typing_extensions as tx

from .anat import (
    AnteriorToPosteriorAxis,
    InferiorToSuperiorAxis,
    LeftToRightAxis,
    PosteriorToAnteriorAxis,
    RightToLeftAxis,
    SuperiorToInferiorAxis,
)

# --- Aliases ----------------------------------------------------------
# The aliases are classes rather than instances, because axes are mutable and
# a shared instance would leak changes between systems. An axis is built with
# `R(unit='mm')` and tested with `isinstance(axis, R)`.

R: tx.TypeAlias = LeftToRightAxis
LR: tx.TypeAlias = LeftToRightAxis
"""Aliases of [`LeftToRightAxis`][].

Coordinates along this axis increase toward the right.
"""

L: tx.TypeAlias = RightToLeftAxis
RL: tx.TypeAlias = RightToLeftAxis
"""Aliases of [`RightToLeftAxis`][].

Coordinates along this axis increase toward the left.
"""

A: tx.TypeAlias = PosteriorToAnteriorAxis
PA: tx.TypeAlias = PosteriorToAnteriorAxis
"""Aliases of [`PosteriorToAnteriorAxis`][].

Coordinates along this axis increase toward the front.
"""

P: tx.TypeAlias = AnteriorToPosteriorAxis
AP: tx.TypeAlias = AnteriorToPosteriorAxis
"""Aliases of [`AnteriorToPosteriorAxis`][].

Coordinates along this axis increase toward the back.
"""

S: tx.TypeAlias = InferiorToSuperiorAxis
IS: tx.TypeAlias = InferiorToSuperiorAxis
"""Aliases of [`InferiorToSuperiorAxis`][].

Coordinates along this axis increase toward the top.
"""

I: tx.TypeAlias = SuperiorToInferiorAxis
SI: tx.TypeAlias = SuperiorToInferiorAxis
"""Aliases of [`SuperiorToInferiorAxis`][].

Coordinates along this axis increase toward the bottom.
"""
