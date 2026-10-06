"""
Non-spatial metadata, shared across file formats.

Every file format keeps descriptive metadata, such as a description, a
repetition time, a slice timing or a provenance, under its own names
and types. This package gives that metadata one representation, in a
class hierarchy that mirrors the hierarchy of the images (`Image`,
`FileBasedImage`, `NiftiImage`):

- [`Metadata`][brainhops.datamodel.metadata.Metadata] holds the common
  vocabulary, one field per concept, named after its BIDS key in snake
  case and stored in BIDS units, plus `extra`, a free-form store of
  string keys. In-memory images and transformations carry it, and
  formats convert through it.
- [`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata], which
  lives in `brainhops.io.metadata` as `FileBasedImage` lives in
  `brainhops.io.images`, is the base of the metadata of a file format,
  which reads its fields from the raw record of the format (a `nibabel`
  header, the attributes of a Zarr array, ...) and writes them back.
- Each format has its own `<Fmt>Metadata` class, next to its parser
  under `brainhops.io`. `FileBasedMetadata.load(path)` reads the
  metadata of a file without its data, and `brainhops.io.metadata.bids`
  reads and writes BIDS sidecars. The data model does no input or
  output.

A vocabulary field holds a value, `None` when the value is unknown, or
[`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED] when the format
has no slot for the field. Converting into a format, with
`metadata.to(NiftiMetadata)`, records what the format cannot hold in a
[`ConversionReport`][brainhops.datamodel.metadata.ConversionReport], and
the loss policy (`"ignore"`, `"warn"` or `"raise"`, see
[`metadata_loss_policy`][brainhops.datamodel.metadata.metadata_loss_policy])
decides what happens to the report. A
[`Scope`][brainhops.datamodel.metadata.Scope] says how each field
propagates to a derived image.

The names exported here are those a user of the library needs, and the
vocabulary groups (`ProvenanceVocabulary`, `MRIVocabulary`, ..., and
their base `Vocabulary`), which a format names in its `supports=`
declaration. What else a format author needs (the field annotations,
the `metadata` field of images, the loss helpers) is imported from the
private modules of this package, which the format author's guide lists
(`docs/dev/metadata-formats.md`). The user guide is
`docs/start/metadata.md`.
"""

__all__ = [
    "Metadata",
    "UNSUPPORTED",
    "Scope",
    "GeneratedBy",
    "Channel",
    "EncodingDirection",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
    "Vocabulary",
    "ProvenanceVocabulary",
    "MRIVocabulary",
    "DiffusionVocabulary",
    "DisplayVocabulary",
    "StorageVocabulary",
    "MicroscopyVocabulary",
    "TransformVocabulary",
]

from ._base import Metadata
from ._report import (
    ConversionReport,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from ._sentinel import UNSUPPORTED
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
