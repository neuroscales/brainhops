"""
Non-spatial metadata, shared across file formats.

Every file format keeps descriptive metadata (a description, a repetition
time, slice timing, provenance, ...) under its own names and types. This
package gives them one representation, in a class hierarchy that mirrors
the images' (`Image` -> `FileBasedImage` -> `NiftiImage`):

- [`Metadata`][brainhops.datamodel.metadata.Metadata]: the **common
  vocabulary**, one field per concept, named after its BIDS key in snake
  case and stored in BIDS units, declared in six groups
  ([`GROUPS`][brainhops.datamodel.metadata.GROUPS]), plus `extra`, a
  free-form `str -> Any` store. It is what in-memory images and
  transformations carry, and the hub through which formats convert.
- [`FileBasedMetadata`][brainhops.datamodel.metadata.FileBasedMetadata]
  adds the format's own **raw record**, `raw` (a `nibabel` header, ...),
  and the read-time snapshot of what was decoded from it.
- One `<Fmt>Metadata` per format, next to its parser under
  `brainhops.io`.

**Three values per field.** A vocabulary field holds a value, `None`
("unknown") or [`UNSUPPORTED`][brainhops.datamodel.metadata.UNSUPPORTED]
("this format has no slot for it"). Converting into a format
(`metadata.to(NiftiMetadata)`) reports what it cannot hold in a
[`ConversionReport`][brainhops.datamodel.metadata.ConversionReport], and
the loss policy (`"ignore"`, `"warn"`, `"raise"`) decides what happens to
the report.

Read next: the user guide (`docs/start/metadata.md`), and, to add a
format, the format author's guide (`docs/dev/metadata-formats.md`).
"""

__all__ = [
    "ALL",
    "UNSUPPORTED",
    "Unsupported",
    "Maybe",
    "Bids",
    "Scope",
    "FILE",
    "ACQUISITION",
    "GRID",
    "VOLUME",
    "VOCABULARY",
    "GROUPS",
    "BIDS_KEYS",
    "SCOPES",
    "GeneratedBy",
    "Channel",
    "EncodingDirection",
    "ProvenanceMetadata",
    "MRIMetadata",
    "DiffusionMetadata",
    "DisplayMetadata",
    "MicroscopyMetadata",
    "TransformMetadata",
    "Metadata",
    "FileBasedMetadata",
    "OpaqueMetadata",
    "MetadataField",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
    "apply_loss_policy",
    "collect_loss_reports",
    "LossPolicy",
    "preferred_dtype",
    "Lazy",
    "LazyField",
]

from brainhops._core.fields import Lazy, LazyField

from ._base import Metadata
from ._dtype import preferred_dtype
from ._field import MetadataField
from ._filebased import FileBasedMetadata, OpaqueMetadata
from ._report import (
    ConversionReport,
    LossPolicy,
    MetadataLossError,
    MetadataLossWarning,
    apply_loss_policy,
    collect_loss_reports,
    metadata_loss_policy,
)
from ._sentinel import ALL, UNSUPPORTED, Maybe, Unsupported
from ._terms import Channel, EncodingDirection, GeneratedBy
from ._vocabulary import (
    ACQUISITION,
    BIDS_KEYS,
    FILE,
    GRID,
    GROUPS,
    SCOPES,
    VOCABULARY,
    VOLUME,
    Bids,
    DiffusionMetadata,
    DisplayMetadata,
    MicroscopyMetadata,
    MRIMetadata,
    ProvenanceMetadata,
    Scope,
    TransformMetadata,
)

# The public names keep the `__module__` of the private module that
# defines them: rewriting it to this package's name would break
# `inspect.getsource`, IPython's `??` and doctest discovery, which look
# the source up through `__module__`. Pickles name the private module,
# and load as well.
