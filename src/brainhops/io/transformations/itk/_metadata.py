"""
The metadata of ITK transformation files.

- `.tfm` and `.mat` store a bare chain of parameters, and no metadata:
  [`ItkMetadata`][] is an `OpaqueMetadata`, and every field is
  unsupported.
- `.h5` records the version of ITK that wrote it (`/ITKVersion`):
  [`ItkH5Metadata`][] reads it as `generated_by`, with the small root
  header ([`H5Header`][brainhops.io.transformations.itk.h5.H5Header]) as
  its raw record.

ITK's `precision` (`float`/`double`) is the element type of the stored
parameters; it stays in the raw record (`ItkStruct.precision`) for now,
rather than `data_type`.

The blocks of a chain (or of a `CompositeTransform`) are data model
transformations with no metadata of their own: composition does not
merge, so the metadata of the file is on the transformation read from
it, not on its blocks.
"""

__all__ = ["H5Header", "ItkH5Metadata", "ItkMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic, replace

from brainhops.datamodel.metadata import (
    ConversionReport,
    GeneratedBy,
)

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base._metadata_parser import Hdf5MetadataParser
from brainhops.io.base.parsers import Confidence
from brainhops.io.metadata import (
    FileBasedMetadata,
    OpaqueMetadata,
)

_ITK = "ITK"


class H5Header(
    Magic,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """
    The root header of an ITK `.h5` file: the versions of the libraries
    and of the system that wrote it, as the root datasets of the file
    record them. It is the raw record of [`ItkH5Metadata`][].
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


@register_format
class ItkH5Metadata(
    Hdf5MetadataParser,
    FileBasedMetadata[H5Header],
    on={"format": "itk-h5"},
    supports=("generated_by",),
):
    """
    The metadata of an ITK `.h5` file: the version of ITK that wrote
    it, as `generated_by`. Its raw record (`raw`) is the root header of
    the file (an `H5Header`: `/ITKVersion`, ...).
    `ItkH5Metadata.load(path)` reads the root header alone (which needs
    `h5py`).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5",)
    HINTS = ("itk", "h5")

    @classmethod
    def sniff_h5(cls, h5file: tx.Any) -> float:
        """
        Score how confident the class is that an open HDF5 file is an ITK
        transform file: one that records the ITK version at its root.

        Parameters
        ----------
        h5file : h5py.File
            The open file.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        if "ITKVersion" in h5file.keys():
            return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def from_h5(cls, h5file: tx.Any, **kwargs: tx.Any) -> tx.Self:
        """
        Read the root header of an open ITK `.h5` file, without its
        transforms.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        **kwargs
            Ignored.

        Returns
        -------
        ItkH5Metadata
            The metadata of the file, with its root header as `raw`.
        """
        from .h5._parser import read_h5_header

        return cls.from_raw(read_h5_header(h5file))

    @property
    def header(self) -> tx.Optional[H5Header]:
        """The root header (the raw record, `raw`)."""
        return self.raw

    @classmethod
    def _decode_raw(
        cls, raw: tx.Optional[H5Header], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        version = getattr(raw, "ITKVersion", None)
        if not version:
            return {}
        return {"generated_by": (GeneratedBy(name=_ITK, version=version),)}

    def _encode_raw(
        self,
        raw: H5Header,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> H5Header:
        if "generated_by" not in changed:
            return raw
        entries = tuple(changed["generated_by"] or ())
        itk = [g for g in entries if g.name == _ITK]
        others = tuple(g for g in entries if g.name != _ITK)
        if others:
            report.lost["generated_by"] = others
        version = itk[0].version if itk else None
        return replace(raw, ITKVersion=version)
