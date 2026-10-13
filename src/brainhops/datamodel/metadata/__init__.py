"""
Descriptive information about an image or a transformation.

Every file format keeps descriptive metadata, such as a description, a
repetition time, a slice timing or a provenance, under its own names
and types. This package gives that metadata one representation:
[`Metadata`][brainhops.datamodel.metadata.Metadata] holds the common
vocabulary, one field per concept, named after its BIDS key in snake
case and stored in BIDS units, plus `extra`, a free-form store of
string keys. A field holds `None` when its value is unknown.

The metadata of a file format, such as `NiftiMetadata`, is a subclass of
`Metadata` that also holds the header of the file. It lives in
`brainhops.io`, next to the format, and derives from
[`MetadataFormat`][brainhops.io.metadata.MetadataFormat]. It decodes
the vocabulary from the header when it is read, and encodes the fields
that changed into a copy of the header when it is consolidated. What a
format cannot store is reported in a
[`ConversionReport`][brainhops.datamodel.metadata.ConversionReport], and
the loss policy (`"ignore"`, `"warn"` or `"raise"`, see
[`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy])
decides what happens to the report.

The vocabulary is declared by seven groups, exported here with their
base, [`Vocabulary`][brainhops.datamodel.metadata.Vocabulary], so that
the documentation lists the fields topic by topic. A
[`Scope`][brainhops.datamodel.metadata.Scope] says how each field
follows an image operation, such as an index or a resampling (see
[`Metadata.derive`][brainhops.datamodel.metadata.Metadata.derive]).
"""

__all__ = [
    "Metadata",
    "Vocabulary",
    "ProvenanceVocabulary",
    "MRIVocabulary",
    "DiffusionVocabulary",
    "DisplayVocabulary",
    "StorageVocabulary",
    "MicroscopyVocabulary",
    "TransformVocabulary",
    "Scope",
    "GeneratedBy",
    "Channel",
    "EncodingDirection",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
]

from ._base import Metadata
from ._report import (
    ConversionReport,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from ._terms import Channel, EncodingDirection, GeneratedBy
from ._vocabulary import (
    DiffusionVocabulary,
    DisplayVocabulary,
    MicroscopyVocabulary,
    MRIVocabulary,
    ProvenanceVocabulary,
    Scope,
    StorageVocabulary,
    TransformVocabulary,
    Vocabulary,
)

# The public names keep the `__module__` of the private module that
# defines them: rewriting it to this package's name would break
# `inspect.getsource`, IPython's `??` and doctest discovery, which look
# the source up through `__module__`. Pickles name the private module,
# and load as well.
