"""Tests of structured I/O source specifications."""

import pytest
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core.path import Path
from brainhops.datamodel.images import Image
from brainhops.io.base import (
    ImageSpec,
    OperationSpec,
    Parser,
    SourceSpec,
    TransformationSpec,
    format_hints,
    parser_for,
)
from brainhops.io.images.base import ImageFormat


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


def test_operation_registered_on_a_subclass_stays_in_that_subclass() -> None:
    # Each subclass keeps its own registry of operations. All classes
    # used to share a single registry.
    class WarpSpec(TransformationSpec, frozen=True):
        pass

    class SiblingSpec(TransformationSpec, frozen=True):
        pass

    class LeafSpec(WarpSpec, frozen=True):
        pass

    @WarpSpec.register_operation("halve")
    class HalveOperation(OperationSpec, frozen=True):
        def apply(self, value: tx.Any) -> tx.Any:
            return value / 2

    for spec_class in (WarpSpec, LeafSpec):
        spec = spec_class.from_arg("warp|halve|inv")
        assert [type(op) for op in spec.operations][0] is HalveOperation
        assert [op.name for op in spec.operations] == ["halve", "inv"]
    for spec_class in (TransformationSpec, SiblingSpec):
        spec = spec_class.from_arg("warp|halve")
        assert spec.operations == ()
        assert spec.hints == ("halve",)
    assert "halve" not in vars(TransformationSpec)["_OPERATIONS"]


def test_registered_parser_is_inherited_by_subclasses() -> None:
    class DerivedImage(Image):
        pass

    assert parser_for(DerivedImage) is ImageFormat


def test_explicit_parser_wins_for_a_union() -> None:
    annotation = tx.Annotated[tx.Union[Image, str], Parser(Image)]
    assert parser_for(annotation) is ImageFormat


def test_unique_registered_union_parser_is_found_automatically() -> None:
    annotation = tx.Union[object, Image]
    assert parser_for(annotation) is ImageFormat


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
    # The FSL and NIfTI formats are only registered with nibabel.
    pytest.importorskip("nibabel")

    from brainhops.io.transformations.fsl.flirt import FlirtTransform
    from brainhops.io.transformations.itk.tfm import TfmTransform
    from brainhops.io.transformations.nifti import NiftiVoxelToRAS

    assert {"flirt", "fsl.flirt", "affine.flirt"} <= format_hints(
        FlirtTransform
    )
    assert {"tfm", "itk.tfm", "xform.itk.tfm"} <= format_hints(TfmTransform)
    assert {"nifti", "affine.nifti"} <= format_hints(NiftiVoxelToRAS)


# ----------------------------------------------------------------------
#   VELOCITIES: the `svf` alias and boolean options
# ----------------------------------------------------------------------


def test_svf_is_an_alias_of_displacements_with_log() -> None:
    spec = TransformationSpec.from_arg("warp.nii.gz|svf")
    assert spec.hints == ("displacements",)
    assert spec.options == {"log": "true"}
    assert spec == TransformationSpec.from_arg(
        "warp.nii.gz|displacements|log:true"
    )
    spec = TransformationSpec.from_arg("warp.nii.gz|svf|steps:6")
    assert spec.hints == ("displacements",)
    assert spec.options == {"steps": "6", "log": "true"}


def test_svf_refuses_a_log_option() -> None:
    with pytest.raises(ValueError, match="svf already means log:true"):
        TransformationSpec.from_arg("warp.nii.gz|svf|log:false")


@pytest.mark.parametrize(
    "text, value",
    [
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        ("0", False),
    ],
)
def test_a_boolean_option_is_parsed(text: str, value: bool) -> None:
    from brainhops.io.base.specs import parse_bool

    assert parser_for(bool) is parse_bool
    assert parser_for(tx.Annotated[bool, "doc"]) is parse_bool
    assert parse_bool(SourceSpec(path=text)) is value


def test_a_boolean_option_refuses_other_text() -> None:
    from brainhops.io.base.specs import parse_bool

    with pytest.raises(ValueError, match="boolean"):
        parse_bool(SourceSpec(path="maybe"))
    with pytest.raises(ValueError, match="boolean"):
        parse_bool(SourceSpec.from_arg("true|nifti"))
