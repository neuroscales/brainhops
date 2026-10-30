"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.datamodel` from this module, and not from the modules that
define them. This module only re-exports these names. No module of the
package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "PyramidLayout",
    "ScalePlan",
    "adopt_scales",
    "bridge",
    "build_pyramid",
    "build_scales",
    "convert",
    "convert_withlog",
    "convert_withsplines",
    "converter",
    "field_scale",
    "find_converter",
    "get_axes",
    "keeps_scale",
    "index2transform",
    "level_matches",
    "next_scale",
    "plan_scales",
    "pop_pyramid_options",
    "pyramid_layout",
    "smart_replace",
    "vector_axis",
]

from ._indexing import index2transform
from ._multiscale import (
    PyramidLayout,
    ScalePlan,
    adopt_scales,
    build_pyramid,
    build_scales,
    keeps_scale,
    level_matches,
    next_scale,
    plan_scales,
    pop_pyramid_options,
    pyramid_layout,
)
from ._sugar import get_axes, vector_axis
from ._transformations.compute.adaptors import bridge
from ._transformations.compute.convert import (
    convert,
    converter,
    find_converter,
)
from ._transformations.compute.converters import (
    convert_withlog,
    convert_withsplines,
    smart_replace,
)
from ._transformations.multiscale import field_scale
