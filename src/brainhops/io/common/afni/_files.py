"""The two files of an AFNI dataset, and how they are found."""

# stdlib
import re

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path

# this format
from ._constants import _BRIK_SUFFIXES

# ----------------------------------------------------------------------
#   FILES
# ----------------------------------------------------------------------


def afni_dataset_files(
    filename: path.FilenameLike,
) -> tx.Tuple[tx.Any, tx.Any, str]:
    """
    Find the `.HEAD` and `.BRIK` files of the dataset that a path names.

    The path may name the `.HEAD` file, the `.BRIK` file with any
    compression, or the bare dataset (`anat+orig`). The BRIK is the first
    existing file among the compression suffixes AFNI tries, or else the
    uncompressed name.

    Returns
    -------
    head : Path
        The `.HEAD` file.
    brik : Path
        The `.BRIK` file, which may not exist.
    stem : str
        The dataset name, without directory or extension.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    name = filename.name
    stem, head_ext, brik_ext = name, ".HEAD", ".BRIK"
    match = re.search(r"\.(HEAD|head|BRIK|brik)(\.[A-Za-z0-9]+)?$", name)
    if match and match.group(2) in (None,) + _BRIK_SUFFIXES[1:]:
        stem = name[: match.start()]
        if match.group(1).islower():
            head_ext, brik_ext = ".head", ".brik"
    parent = filename.parent
    head = parent / (stem + head_ext)
    brik = None
    for suffix in _BRIK_SUFFIXES:
        candidate = parent / (stem + brik_ext + suffix)
        if path.exists(candidate):
            brik = candidate
            break
    if brik is None:
        brik = parent / (stem + brik_ext)
    return head, brik, stem


def _brik_suffix(brik: tx.Any) -> str:
    """Return the compression suffix of a BRIK name, or `""` without one."""
    name = str(brik.name if hasattr(brik, "name") else brik)
    for suffix in _BRIK_SUFFIXES[1:]:
        if name.endswith(suffix):
            return suffix
    return ""
