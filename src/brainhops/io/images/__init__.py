__all__ = ["nifti", "zarr"]


import os

import typing_extensions as _tx

from brainhops.datamodel.images import Image

from . import nifti, zarr


class ImageEntry:
    prefix: _tx.Optional[tuple[str]]
    extension: tuple[str]
    class_value: type
    hints: tuple[str]

    def __init__(self,
                 prefix: _tx.Optional[_tx.Union[tuple[str], str]],
                 extension: _tx.Union[tuple[str], str],
                 class_value: type,
                 hints: _tx.Union[tuple[str], str]) -> None:
        self.prefix = prefix if isinstance(
            prefix, (tuple, type(None))) else (prefix,)
        self.extension = extension if isinstance(
            extension, tuple) else (extension,)
        self.class_value = class_value
        self.hints = hints if isinstance(
            hints, tuple) else (hints,)


image_entries = [
    ImageEntry(None, (".nii", ".nii.gz"), nifti.NiftiImage,
               ("nii", "nifti", "nii.gz", "nii_gz", "niigz")),
    ImageEntry(None, (".ome.zarr", ".zarr"), zarr.OmeZarrImage,
               ("zarr", "omezarr", "ome.zarr", "ome_zarr")),
]


def load(file_name: str, hint: _tx.Optional[str] = None) -> Image:
    base = os.path.basename(file_name)
    if hint is not None:
        hint = hint.lower().replace(" ", "_").removeprefix(".")
        for i in image_entries:
            if hint in i.hints:
                return i.class_value.from_file(file_name)
    for i in image_entries:
        if base.endswith(i.extension) and (i.prefix is None or
                                           base.startswith(i.prefix)):
            return i.class_value.from_file(file_name)
    for i in image_entries:
        if i.class_value.sniff_file(file_name):
            return i.class_value.from_file(file_name)
    raise NotImplementedError(f"can't parse the file: {file_name}")
