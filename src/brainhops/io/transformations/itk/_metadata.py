"""
The metadata of ITK transformation files.

- `.tfm` and `.mat` store a bare chain of parameters, and no metadata:
  [`ItkMetadata`][] is an
  [`OpaqueMetadata`][brainhops.datamodel.metadata.OpaqueMetadata], and
  every field is unsupported.
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

__all__ = ["ItkH5Metadata", "ItkMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoEq, NoRepr, replace

# internals
from brainhops.datamodel.metadata import (
    ConversionReport,
    FileBasedMetadata,
    GeneratedBy,
    OpaqueMetadata,
)

# The raw record of an `.h5` file is an `H5Header`, from the h5 reader, which
# needs the optional h5py: it is imported only where a record is built.
H5Header = tx.Any

_ITK = "ITK"


class ItkMetadata(OpaqueMetadata, on={"format": "itk"}):
    """The metadata of an ITK `.tfm` or `.mat` file: none."""

    format: tx.Annotated[tx.Literal["itk"], tx.Doc("Always `'itk'`.")] = "itk"


class ItkH5Metadata(
    FileBasedMetadata, on={"format": "itk-h5"}, supports=("generated_by",)
):
    """
    The metadata of an ITK `.h5` file: the version of ITK that wrote
    it, as `generated_by`. Its raw record is the root header.
    """

    format: tx.Annotated[
        tx.Literal["itk-h5"], tx.Doc("Always `'itk-h5'`.")
    ] = "itk-h5"

    raw: tx.Annotated[
        tx.Optional[H5Header],
        tx.Doc(
            "The root header of the file (an `H5Header`: `/ITKVersion`, ...)."
        ),
        NoRepr(),
        NoEq(),
    ] = None

    @property
    def header(self) -> tx.Optional[H5Header]:
        """The root header (the raw record, `raw`)."""
        return self.raw

    @classmethod
    def _default_raw(cls) -> H5Header:
        from .h5._parser import H5Header

        return H5Header()

    @classmethod
    def _decode(
        cls, raw: tx.Optional[H5Header], *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        version = getattr(raw, "ITKVersion", None)
        if not version:
            return {}
        return {"generated_by": (GeneratedBy(name=_ITK, version=version),)}

    def _encode(
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
