"""
The metadata of ITK `.h5` files: [`ItkH5Metadata`][], next to the parser
of the files, since reading them needs `h5py`, which is optional.
"""

__all__ = ["ItkH5Metadata", "read_h5_header"]

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops.datamodel.metadata import ConversionReport, GeneratedBy
from brainhops.io.base._base import register_format
from brainhops.io.base.hdf5 import Hdf5MetadataParser, read_string
from brainhops.io.base.parsers import Confidence, SnifferContentError
from brainhops.io.metadata import FileBasedMetadata

# locals
from .._metadata import H5Header


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
    `ItkH5Metadata.load(path)` reads the root header alone.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5",)
    HINTS = ("itk", "h5")

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """
        Score how confident the class is that an open HDF5 file is an ITK
        transform file: one that records the ITK version at its root.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type, optional
            Raise an error (this one, or `SnifferContentError` for `True`)
            instead of returning 0.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        if "ITKVersion" in h5file.keys():
            return Confidence.CERTAIN
        if error:
            raise (SnifferContentError if error is True else error)(
                "HDF5 file is not an ITK transform file"
            )
        return Confidence.NO

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs: tx.Any) -> tx.Self:
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


def read_h5_header(h5file: h5py.File) -> H5Header:
    """
    Read the root header of an open ITK `.h5` file.

    Parameters
    ----------
    h5file : h5py.File
        The open file.

    Returns
    -------
    H5Header
        The versions recorded at the root of the file.
    """
    header = H5Header()
    for name in ("HDFVersion", "ITKVersion", "OSName", "OSVersion"):
        if f"/{name}" in h5file:
            setattr(header, name, read_string(h5file[f"/{name}"]))
    return header


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


_ITK = "ITK"
