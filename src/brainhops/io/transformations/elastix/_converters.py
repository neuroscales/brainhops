"""
Private converters into the elastix formats.

A parameter file holds a chain of ITK blocks, and no exact conversion
into it is written yet (#312). A chain is therefore relabelled as an
[`ElastixParameterTransform`][brainhops.io.transformations.elastix.ElastixParameterTransform]
or an
[`ElastixTomlTransform`][brainhops.io.transformations.elastix.ElastixTomlTransform]
by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format, and whether the chain can be written is decided when it
is written.

The two formats differ only in the syntax of the file, so a
transformation of one of them that holds its record, and whose chain was
not assigned, is converted to the other by holding the same metadata.
Its consolidation then writes the same parameter map in the other
syntax. A transformation whose chain was assigned is relabelled, as any
other chain is. A transformation that is converted to its own class is
changed by the rules of any chain.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import (
    relabel,
    relabelling,
)

from ._xform import (
    ElastixParameterTransform,
    ElastixTomlTransform,
    ElastixTransform,
)

# A parameter file holds a chain of ITK blocks, and no exact conversion
# into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, ElastixParameterTransform)(relabelling)
converter(_xforms.Sequence, ElastixTomlTransform)(relabelling)


@converter
def _(
    t: ElastixParameterTransform,
    cls: tx.Type[ElastixParameterTransform],
    **kwargs,
) -> ElastixParameterTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: ElastixTomlTransform,
    cls: tx.Type[ElastixTomlTransform],
    **kwargs,
) -> ElastixTomlTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)


@converter
def _(
    t: ElastixTomlTransform,
    cls: tx.Type[ElastixParameterTransform],
    **kwargs,
) -> ElastixParameterTransform:
    # The text syntax holds the parameter map of a TOML file.
    return _other_syntax(t, cls, **kwargs)


@converter
def _(
    t: ElastixParameterTransform,
    cls: tx.Type[ElastixTomlTransform],
    **kwargs,
) -> ElastixTomlTransform:
    # The TOML syntax holds the parameter map of a text file.
    return _other_syntax(t, cls, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _other_syntax(
    t: ElastixTransform, cls: tx.Type[ElastixTransform], **kwargs
) -> ElastixTransform:
    # A transformation that holds its record, and whose chain was not
    # assigned, keeps its metadata, and its consolidation writes the
    # parameter map in the syntax of `cls`. Any other one is relabelled.
    if t.metadata is None or t._transformations is not None:
        return relabel(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)
