__all__ = ["itk", "freesurfer", "fsl", "spm", "tirl"]


import os

import typing_extensions as _tx

from brainhops.datamodel.transformations import Transformation

from . import freesurfer, fsl, itk, spm, tirl


class TransformationEntry:
    prefix: _tx.Optional[tuple[str]]
    extension: tuple[str]
    class_value: type
    hints: tuple[str]

    def __init__(
        self,
        prefix: _tx.Optional[_tx.Union[tuple[str], str]],
        extension: _tx.Union[tuple[str], str],
        class_value: type,
        hints: _tx.Union[tuple[str], str],
    ) -> None:
        self.prefix = (
            prefix if isinstance(prefix, (tuple, type(None))) else (prefix,)
        )
        self.extension = (
            extension if isinstance(extension, tuple) else (extension,)
        )
        self.class_value = class_value
        self.hints = hints if isinstance(hints, tuple) else (hints,)


transformation_entries = [
    TransformationEntry(
        ("y_", "iy_"),
        (".nii", ".nii.gz"),
        spm.y.SPMCoordinatesField,
        ("spm", "spmy"),
    ),
    TransformationEntry(None, ".lta", freesurfer.lta.LTATransformation, "lta"),
    TransformationEntry(
        None, (".h5", ".x5"), freesurfer.lta.LTATransformation, "lta"
    ),
    TransformationEntry(None, (".tfm"), itk.tfm.TFMTransform, "tfm"),
    TransformationEntry(None, (".tirl"), tirl.TIRLTransform, "tirl"),
]


def load(file_name: str, hint: _tx.Optional[str] = None) -> Transformation:
    base = os.path.basename(file_name)
    if hint is not None:
        hint = hint.lower().replace(" ", "_").removeprefix(".")
        for i in transformation_entries:
            if hint in i.hints:
                return i.class_value.from_file(file_name)
    for i in transformation_entries:
        if base.endswith(i.extension) and (
            i.prefix is None or base.startswith(i.prefix)
        ):
            return i.class_value.from_file(file_name)
    for i in transformation_entries:
        if i.class_value.sniff_file(file_name):
            return i.class_value.from_file(file_name)
    raise NotImplementedError(f"can't parse the file: {file_name}")
