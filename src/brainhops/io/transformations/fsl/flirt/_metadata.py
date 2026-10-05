"""
The metadata of FLIRT `.mat` files: [`FlirtMetadata`][].

A FLIRT matrix is a bare `(4, 4)` matrix: it stores no metadata, and
there is no raw record. The reader needs the moving and reference
images, though, and when they were read from files their paths are
`moving` and `fixed`. So the metadata is not opaque: it holds these two
fields, in memory only. A `.mat` file has no place for them, so a write
reports them as lost.
"""

__all__ = ["FlirtMetadata"]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.metadata import ConversionReport, FileBasedMetadata


class FlirtMetadata(
    FileBasedMetadata[None],
    on={"format": "flirt"},
    supports=("moving", "fixed"),
):
    """
    The metadata of a FLIRT `.mat` file: only `moving` and `fixed`, the
    paths of the images the reader was given, kept in memory (a read and
    a copy keep them; a write reports them as lost). There is no raw
    record: `raw` is always `None`.
    """

    @classmethod
    def _decode(
        cls, raw: tx.Any, *, image: tx.Any = None
    ) -> tx.Dict[str, tx.Any]:
        return {
            "moving": _filename(getattr(image, "moving", None)),
            "fixed": _filename(getattr(image, "reference", None)),
        }

    def _encode(
        self,
        raw: tx.Any,
        changed: tx.Dict[str, tx.Any],
        *,
        image: tx.Any = None,
        report: ConversionReport,
    ) -> tx.Any:
        # Nothing is stored: the images are given again at read time.
        for name in ("moving", "fixed"):
            value = getattr(self, name)
            if value is not None:
                report.lost[name] = value
        return raw


def _filename(image: tx.Any) -> tx.Optional[str]:
    """The path a (`nibabel` or brainhops NIfTI) image was read from."""
    for obj in (image, getattr(image, "image", None)):
        getter = getattr(obj, "get_filename", None)
        if callable(getter):
            name = getter()
            if name:
                return str(name)
    return None
