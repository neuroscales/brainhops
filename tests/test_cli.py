"""Tests for the ``brainhops`` command-line interface."""

import numpy as np
import pytest

from brainhops.cli import main
from brainhops.cli._errors import CliError, WritingUnavailable
from brainhops.cli._io import _writable_image_formats, load_transform
from brainhops.cli._reslice import (
    _load_push_transform,
    _split_transform_spec,
    reslice_image,
)
from brainhops.datamodel.images import Image
from brainhops.io.base import SourceSpec

nb = pytest.importorskip("nibabel")


def _write_nifti(path, shape=(4, 5, 6), scale=2.0) -> str:  # noqa: ANN001
    """Write a small NIfTI image with a scaling affine and return its path."""
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    affine = np.diag([scale, scale, scale, 1.0])
    nb.save(nb.Nifti1Image(data, affine), str(path))
    return str(path)


def test_help_lists_the_three_subcommands(capsys) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("reslice", "compose", "convert"):
        assert command in out


def test_no_command_prints_help_and_returns_one(capsys) -> None:  # noqa: ANN001
    code = main([])
    assert code == 1
    assert "reslice" in capsys.readouterr().out


def test_reslice_image_resamples_onto_reference_grid(
    tmp_path,  # noqa: ANN001
) -> None:
    source = _write_nifti(tmp_path / "input.nii.gz", shape=(4, 5, 6))
    reference = _write_nifti(tmp_path / "ref.nii.gz", shape=(3, 3, 3))

    resliced = reslice_image(source, reference, [])

    assert isinstance(resliced, Image)
    # The output lives on the reference grid.
    assert resliced.shape == (3, 3, 3)


def test_reslice_command_reports_when_writing_is_unavailable(
    tmp_path,  # noqa: ANN001
) -> None:
    source = _write_nifti(tmp_path / "input.nii.gz")
    reference = _write_nifti(tmp_path / "ref.nii.gz")
    output = tmp_path / "out.nii.gz"

    if _writable_image_formats():
        pytest.skip("An image writer is registered; nothing to assert here.")

    code = main(
        [
            "reslice",
            source,
            "--reference",
            reference,
            "--output",
            str(output),
        ]
    )

    # The reslice succeeds but the output cannot be written yet.
    assert code == WritingUnavailable.exit_code
    assert not output.exists()


class _FakeTransform:
    """A stand-in transform that records whether it was inverted."""

    def __init__(self, inverted: bool = False) -> None:
        self.inverted = inverted

    def inverse(self) -> "_FakeTransform":
        return _FakeTransform(inverted=not self.inverted)


def test_split_transform_spec_extracts_format_hint_and_operators() -> None:
    assert _split_transform_spec("affine.mat|flirt|inv") == SourceSpec(
        value="affine.mat", hints=("flirt",), operations=("inv",)
    )
    assert _split_transform_spec(
        "affine.mat|inv|hint:flirt,fnirt|inv"
    ) == SourceSpec(
        value="affine.mat",
        hints=("flirt", "fnirt"),
        operations=("inv", "inv"),
    )


def test_split_transform_spec_decodes_a_literal_pipe_in_source() -> None:
    assert _split_transform_spec(
        "s3://bucket/a%7Cb/file.mat|flirt"
    ) == SourceSpec(value="s3://bucket/a|b/file.mat", hints=("flirt",))


def test_split_transform_spec_refuses_duplicate_options() -> None:
    with pytest.raises(CliError, match="Duplicate source option"):
        _split_transform_spec("affine.mat|moving:a.nii|moving:b.nii")


def test_plain_transform_value_is_applied_forward(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_load(path):  # noqa: ANN001, ANN202
        seen["path"] = path
        return _FakeTransform()

    monkeypatch.setattr("brainhops.cli._reslice.load_transform", fake_load)

    transform = _load_push_transform("warp.nii.gz")

    assert seen["path"] == SourceSpec(value="warp.nii.gz")
    assert transform.inverted is False


def test_inv_operator_inverts_the_loaded_transform(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_load(path):  # noqa: ANN001, ANN202
        seen["path"] = path
        return _FakeTransform()

    monkeypatch.setattr("brainhops.cli._reslice.load_transform", fake_load)

    transform = _load_push_transform("warp.nii.gz|inv")

    # The operator is stripped before the path reaches the loader.
    assert seen["path"] == SourceSpec(value="warp.nii.gz", operations=("inv",))
    # The loaded transform is inverted before it is composed.
    assert transform.inverted is True


def test_format_hint_is_passed_to_the_transform_loader(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_load(path):  # noqa: ANN001, ANN202
        seen["path"] = path
        return _FakeTransform()

    monkeypatch.setattr("brainhops.cli._reslice.load_transform", fake_load)
    transform = _load_push_transform("affine.mat|flirt|inv")

    assert seen == {
        "path": SourceSpec(
            value="affine.mat", hints=("flirt",), operations=("inv",)
        )
    }
    assert transform.inverted is True


def test_format_hint_selects_reader_without_a_matching_extension(
    tmp_path,  # noqa: ANN001
) -> None:
    path = tmp_path / "affine.unknown"
    path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )

    transform = load_transform(str(path), hint="flirt")

    assert type(transform).__name__ == "FLIRTTransform"
    np.testing.assert_array_equal(transform.flirt_matrix, np.eye(4))


def test_io_dispatch_accepts_singular_and_union_hints(
    tmp_path,  # noqa: ANN001
) -> None:
    from brainhops import io

    path = tmp_path / "affine.unknown"
    path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )

    singular = io.transformations.load(str(path), hint="flirt")
    union = io.transformations.load(str(path), hint=("flirt", "fnirt"))

    assert type(singular).__name__ == "FLIRTTransform"
    assert type(union).__name__ == "FLIRTTransform"


def test_flirt_options_load_nested_images_without_a_top_level_hint(
    tmp_path,  # noqa: ANN001
) -> None:
    matrix = tmp_path / "affine.mat"
    matrix.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )
    reference = _write_nifti(tmp_path / "reference.nii.gz")
    moving = _write_nifti(tmp_path / "moving.nii.gz")

    transform = _load_push_transform(
        f"{matrix}|reference:[{reference}|nifti]|moving:[{moving}]"
    )

    assert type(transform).__name__ == "FLIRTTransform"
    assert type(transform.reference).__name__ == "NiftiImage"
    assert type(transform.moving).__name__ == "NiftiImage"
    assert transform.matrix.shape == (3, 4)


def test_unknown_format_hint_reports_available_hints() -> None:
    with pytest.raises(CliError, match="Available hints"):
        load_transform("affine.mat", hint="not-a-format")


def test_unimplemented_operator_points_at_the_issue(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "brainhops.cli._reslice.load_transform",
        lambda path: _FakeTransform(),  # noqa: ARG005
    )

    with pytest.raises(CliError) as excinfo:
        _load_push_transform("warp.nii.gz|sqrt")

    message = str(excinfo.value)
    assert "sqrt" in message
    assert "#47" in message


def test_compose_reports_not_implemented(capsys) -> None:  # noqa: ANN001
    code = main(["compose", "a.tfm", "b.tfm", "-o", "out.tfm"])
    assert code == 1
    assert "not implemented" in capsys.readouterr().err.lower()


def test_convert_reports_not_implemented(capsys) -> None:  # noqa: ANN001
    code = main(["convert", "a.tfm", "-o", "out.nii.gz"])
    assert code == 1
    assert "not implemented" in capsys.readouterr().err.lower()
