"""Tests for structured I/O source specifications."""

import pytest
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core.path import Path
from brainhops.datamodel.images import Image
from brainhops.io.base import (
    ImageSpec,
    Parser,
    SourceSpec,
    TransformationSpec,
    format_hints,
    parser_for,
)
from brainhops.io.images.base import FileBasedImage


def test_source_spec_is_a_magic_model_with_a_path() -> None:
    spec = SourceSpec.from_arg("image.nii")

    assert isinstance(spec, Magic)
    assert isinstance(spec.path, Path)
    assert str(spec.path) == "image.nii"


def test_image_spec_parses_image_hints_and_options() -> None:
    spec = ImageSpec.from_arg("image.dat|nifti|mmap:false")

    assert isinstance(spec, ImageSpec)
    assert str(spec.path) == "image.dat"
    assert spec.hints == ("nifti",)
    assert spec.options == {"mmap": "false"}


def test_nested_source_keeps_its_own_hints_and_options() -> None:
    spec = TransformationSpec.from_arg(
        "affine.mat|flirt|reference:[ref.nii.gz|nifti|mmap:false]|inv"
    )

    assert str(spec.path) == "affine.mat"
    assert spec.hints == ("flirt",)
    assert [operation.name for operation in spec.operations] == ["inv"]
    assert list(spec.options) == ["reference"]
    reference = spec.options["reference"]
    assert isinstance(reference, SourceSpec)
    assert str(reference.path) == "ref.nii.gz"
    assert reference.hints == ("nifti",)
    assert reference.options == {"mmap": "false"}


def test_options_do_not_require_a_format_hint() -> None:
    spec = SourceSpec.from_arg("image.dat|mmap:false")

    assert spec.hints == ()
    assert spec.options == {"mmap": "false"}


def test_colons_in_uri_values_are_not_syntax() -> None:
    spec = SourceSpec.from_arg(
        "s3://bucket/warp.nii.gz|reference:[s3://bucket/ref.nii.gz|nifti]"
    )

    assert str(spec.path) == "s3://bucket/warp.nii.gz"
    reference = spec.options["reference"]
    assert isinstance(reference, SourceSpec)
    assert str(reference.path) == "s3://bucket/ref.nii.gz"


def test_literal_pipe_is_percent_encoded() -> None:
    spec = SourceSpec.from_arg("s3://bucket/a%7Cb%20c.nii|nifti")
    assert str(spec.path) == "s3://bucket/a|b%20c.nii"


def test_nested_sources_may_be_nested_recursively() -> None:
    spec = SourceSpec.from_arg("outer|child:[middle|child:[inner|nifti]]")

    middle = spec.options["child"]
    assert isinstance(middle, SourceSpec)
    inner = middle.options["child"]
    assert isinstance(inner, SourceSpec)
    assert str(inner.path) == "inner"
    assert inner.hints == ("nifti",)


@pytest.mark.parametrize(
    "text",
    [
        "outer|child:[inner",
        "outer|child:[inner].nii",
    ],
)
def test_unclosed_nested_source_brackets_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="Unclosed bracket"):
        SourceSpec.from_arg(text)


@pytest.mark.parametrize(
    "path",
    [
        "image[echo].nii",
        "s3://bucket/[session]/image.nii",
        "image].nii",
    ],
)
def test_brackets_in_source_paths_are_literal(path: str) -> None:
    spec = SourceSpec.from_arg(f"{path}|nifti")

    assert str(spec.path) == path
    assert spec.hints == ("nifti",)


def test_brackets_in_nested_source_paths_are_literal() -> None:
    spec = SourceSpec.from_arg(
        "outer|child:[s3://bucket/[session]/image[echo].nii|nifti]"
    )

    child = spec.options["child"]
    assert isinstance(child, SourceSpec)
    assert str(child.path) == "s3://bucket/[session]/image[echo].nii"
    assert child.hints == ("nifti",)


def test_balanced_path_brackets_at_a_nested_boundary_are_literal() -> None:
    spec = SourceSpec.from_arg("outer|child:[image[echo]|nifti]")

    child = spec.options["child"]
    assert isinstance(child, SourceSpec)
    assert str(child.path) == "image[echo]"
    assert child.hints == ("nifti",)


def test_brackets_in_plain_option_values_are_literal() -> None:
    spec = SourceSpec.from_arg("image.nii|label:run[1]")

    assert spec.options == {"label": "run[1]"}


def test_duplicate_options_are_rejected_before_construction() -> None:
    with pytest.raises(ValueError, match="Duplicate source option 'moving'"):
        SourceSpec.from_arg("warp|moving:a.nii|moving:b.nii")


def test_transformation_operations_are_structured_and_applied() -> None:
    class Value:
        def __init__(self, inverted: bool = False) -> None:
            self.inverted = inverted

        def inverse(self) -> "Value":
            return Value(not self.inverted)

    spec = TransformationSpec.from_arg("warp|inv|inv")

    assert all(operation.name == "inv" for operation in spec.operations)
    assert spec.apply_operations(Value()).inverted is False


def test_parameterless_operation_rejects_a_value() -> None:
    with pytest.raises(ValueError, match="takes no value"):
        TransformationSpec.from_arg("warp|inv:other.nii")


def test_registered_parser_is_inherited_by_subclasses() -> None:
    class DerivedImage(Image):
        pass

    assert parser_for(DerivedImage) is FileBasedImage


def test_explicit_parser_wins_for_a_union() -> None:
    annotation = tx.Annotated[tx.Union[Image, str], Parser(Image)]
    assert parser_for(annotation) is FileBasedImage


def test_unique_registered_union_parser_is_found_automatically() -> None:
    annotation = tx.Union[object, Image]
    assert parser_for(annotation) is FileBasedImage


def test_hints_are_qualified_along_individual_inheritance_branches() -> None:
    class Transform:
        HINTS = ("xform",)

    class FSL(Transform):
        HINTS = ("fsl",)

    class Affine(Transform):
        HINTS = ("affine",)

    class FSLAffine(FSL, Affine):
        pass

    class FLIRT(FSLAffine):
        HINTS = ("flirt",)

    hints = format_hints(FLIRT)
    assert {"flirt", "fsl.flirt", "affine.flirt"} <= hints
    assert {"xform.fsl.flirt", "xform.affine.flirt"} <= hints
    assert "xform.fsl.affine.flirt" not in hints


def test_qualification_distinguishes_ambiguous_leaf_hints() -> None:
    class Transform:
        HINTS = ("xform",)

    class ITK(Transform):
        HINTS = ("itk",)

    class Other(Transform):
        HINTS = ("other",)

    class ITKTFM(ITK):
        HINTS = ("tfm",)

    class OtherTFM(Other):
        HINTS = ("tfm",)

    assert "tfm" in format_hints(ITKTFM) & format_hints(OtherTFM)
    assert "itk.tfm" in format_hints(ITKTFM)
    assert "itk.tfm" not in format_hints(OtherTFM)


def test_builtin_formats_expose_semantic_hint_namespaces() -> None:
    from brainhops.io.transformations.fsl.flirt import FLIRTTransform
    from brainhops.io.transformations.itk.tfm import TFMTransform
    from brainhops.io.transformations.nifti.affines import NiftiVoxelToRAS

    assert {"flirt", "fsl.flirt", "affine.flirt"} <= format_hints(
        FLIRTTransform
    )
    assert {"tfm", "itk.tfm", "xform.itk.tfm"} <= format_hints(TFMTransform)
    assert {"nifti", "affine.nifti"} <= format_hints(NiftiVoxelToRAS)
