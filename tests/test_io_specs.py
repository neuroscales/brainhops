"""Tests for structured I/O source specifications."""

import typing_extensions as tx

from brainhops.datamodel.images import Image
from brainhops.io.base import Parser, SourceSpec, parser_for
from brainhops.io.images.base import FileBasedImage


def test_nested_source_keeps_its_own_hints_and_options() -> None:
    spec = SourceSpec.parse(
        "affine.mat|flirt|reference:[ref.nii.gz|nifti|mmap:false]|inv",
        operations={"inv"},
    )
    assert spec.value == "affine.mat"
    assert spec.hints == ("flirt",)
    assert spec.operations == ("inv",)
    assert spec.options == (
        (
            "reference",
            SourceSpec(
                value="ref.nii.gz",
                hints=("nifti",),
                options=(("mmap", "false"),),
            ),
        ),
    )


def test_colons_in_uri_values_are_not_syntax() -> None:
    spec = SourceSpec.parse(
        "s3://bucket/warp.nii.gz|reference:[s3://bucket/ref.nii.gz|nifti]"
    )
    assert spec.value == "s3://bucket/warp.nii.gz"
    assert spec.options[0][1].value == "s3://bucket/ref.nii.gz"


def test_literal_pipe_is_percent_encoded() -> None:
    assert SourceSpec.parse("s3://bucket/a%7Cb.nii|nifti").value == (
        "s3://bucket/a|b.nii"
    )


def test_registered_parser_is_inherited_by_subclasses() -> None:
    class DerivedImage(Image):
        pass

    assert parser_for(DerivedImage) is FileBasedImage


def test_explicit_parser_wins_for_a_union() -> None:
    annotation = tx.Annotated[tx.Union[Image, str], Parser(Image)]
    assert parser_for(annotation) is FileBasedImage
