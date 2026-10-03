"""
Generic representation of transforms and their interactions.

Mostly based on OME-NGFF, but with more flexibility.
Should be able to accommodate a large variety of existing transform formats:

- OME-NGFF
- FreeSurfer LTA (affine)
- Freesurfer XFM (nonlinear)
- ANTs
- SPM
- nitorch
- ...

(FSL defines its transformations with respect to the fixed and moving
images, but their metadata is not stored in the transform, which means
that the fixed and moving images must be accessible when applying the
transform on some third image. This is very inconvenient and not a use
case I am fond of supporting).

"""

__all__ = [
    "axes",
    "base",
    "enums",
    "kinds",
    "images",
    "metadata",
    "orientation",
    "systems",
    "transformations",
    "units",
    # metadata
    "UNSUPPORTED",
    "Unsupported",
    "Maybe",
    "Bids",
    "Scope",
    "GeneratedBy",
    "Channel",
    "FormatMetadata",
    "Metadata",
    "OpaqueMetadata",
    "ConversionReport",
    "MetadataLossWarning",
    "MetadataLossError",
    "metadata_loss_policy",
    "convert_metadata",
]

# trigger registration
from . import (
    axes,
    base,
    enums,
    images,
    kinds,
    metadata,
    orientation,
    systems,
    transformations,
    units,
)
from .metadata import (
    UNSUPPORTED,
    Bids,
    Channel,
    ConversionReport,
    FormatMetadata,
    GeneratedBy,
    Maybe,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    OpaqueMetadata,
    Scope,
    Unsupported,
    metadata_loss_policy,
)
from .metadata import convert as convert_metadata
