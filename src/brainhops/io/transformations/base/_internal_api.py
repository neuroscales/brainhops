"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.base` from this module, and not from
the modules that define them. This module only re-exports these names.
No module of the package imports it, so it cannot create a circular
import inside the package. The rules are explained in
`spec/design/internal-api.md`.
"""

__all__ = [
    "affine_between",
    "AFFINE_SOURCES",
    "apply_affine",
    "cardinal_parts",
    "chain_leaves",
    "check_endpoint",
    "compatible",
    "control_grid",
    "coordinate_parts",
    "displacement_parts",
    "displacements_of",
    "field_slots",
    "FIELD_SOURCES",
    "FieldParts",
    "format_options",
    "held",
    "is_field",
    "matrix_to_disk",
    "matrix_to_model",
    "orthonormal_placement",
    "read_only",
    "record_of",
    "register_converter",
    "relabel",
    "relabelling",
    "relabels",
    "sample_on_grid",
    "source_grid",
    "split_affine",
    "split_field_chain",
    "undoes",
    "unrepresentable",
    "UnusedOptionError",
    "world_pieces",
]

from ._conversions import (
    AFFINE_SOURCES,
    FIELD_SOURCES,
    FieldParts,
    UnusedOptionError,
    affine_between,
    apply_affine,
    cardinal_parts,
    chain_leaves,
    coordinate_parts,
    displacement_parts,
    displacements_of,
    field_slots,
    format_options,
    held,
    is_field,
    register_converter,
    sample_on_grid,
    split_affine,
    split_field_chain,
    undoes,
    unrepresentable,
    world_pieces,
)
from ._converters import relabel, relabelling, relabels
from ._endpoints import check_endpoint, compatible
from ._records import matrix_to_disk, matrix_to_model, read_only, record_of
from ._splines import control_grid, orthonormal_placement, source_grid
