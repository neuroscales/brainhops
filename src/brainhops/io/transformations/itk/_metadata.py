"""
The metadata of ITK transformation files.

- `.tfm` and `.mat` store a bare chain of parameters, and no metadata:
  [`ItkMetadata`][] is an `OpaqueMetadata`, and every field is
  unsupported.
- `.h5` records the version of ITK that wrote it (`/ITKVersion`):
  [`ItkH5Metadata`][brainhops.io.transformations.itk.h5.ItkH5Metadata],
  next to the parser of `.h5` files (which needs `h5py`), reads it as
  `generated_by`, with the small root header ([`H5Header`][], defined
  here) as its raw record.

ITK's `precision` (`float`/`double`) is the element type of the stored
parameters; it stays in the raw record (`ItkStruct.precision`) for now,
rather than `data_type`.

The blocks of a chain (or of a `CompositeTransform`) are data model
transformations with no metadata of their own: composition does not
merge, so the metadata of the file is on the transformation read from
it, not on its blocks.
"""

__all__ = ["H5Header", "ItkMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic

# internals
from brainhops.io.metadata import OpaqueMetadata


class H5Header(
    Magic,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """
    The root header of an ITK `.h5` file: the versions of the libraries
    and of the system that wrote it, as the root datasets of the file
    record them. It is the raw record of
    [`ItkH5Metadata`][brainhops.io.transformations.itk.h5.ItkH5Metadata].
    """

    HDFVersion: tx.Optional[str] = None
    """
    A string describing the version of the HDF5 library used.
    Ex: "HDF5 library version: 1.10.4"
    """

    ITKVersion: tx.Optional[str] = None
    """
    A string describing the version of the ITK library used.
    Ex: "5.1.0"
    """

    OSName: tx.Optional[str] = None
    """
    A string describing the operating system name.
    Ex: "Linux"
    """

    OSVersion: tx.Optional[str] = None
    """
    A string describing the operating system version.
    Ex: "6.1.0-1007-oem"
    """


class ItkMetadata(OpaqueMetadata, on={"format": "itk"}):
    """The metadata of an ITK `.tfm` or `.mat` file: none."""

    # Declared again: the field of a subclass of a pinned format is
    # narrowed to the parent's value (`'opaque'`), which `'itk'` is not.
    format: tx.Annotated[tx.Literal["itk"], tx.Doc("Always `'itk'`.")] = "itk"
