"""Tests for the ``brainhops`` command-line interface."""

import numpy as np
import pytest

from brainhops.cli import main
from brainhops.cli._errors import WritingUnavailable
from brainhops.cli._io import _writable_image_formats
from brainhops.cli._reslice import _load_push_transform, reslice_image
from brainhops.datamodel.images import Image

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


def test_plain_transform_value_is_applied_forward(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_load(path):  # noqa: ANN001, ANN202
        seen["path"] = path
        return _FakeTransform()

    monkeypatch.setattr("brainhops.cli._reslice.load_transform", fake_load)

    transform = _load_push_transform("warp.nii.gz")

    assert seen["path"] == "warp.nii.gz"
    assert transform.inverted is False


def test_inv_prefix_strips_and_inverts_the_transform(monkeypatch) -> None:  # noqa: ANN001
    seen = {}

    def fake_load(path):  # noqa: ANN001, ANN202
        seen["path"] = path
        return _FakeTransform()

    monkeypatch.setattr("brainhops.cli._reslice.load_transform", fake_load)

    transform = _load_push_transform("inv:warp.nii.gz")

    # The prefix is stripped before the path reaches the loader.
    assert seen["path"] == "warp.nii.gz"
    # The loaded transform is inverted before it is composed.
    assert transform.inverted is True


def test_compose_reports_not_implemented(capsys) -> None:  # noqa: ANN001
    code = main(["compose", "a.tfm", "b.tfm", "-o", "out.tfm"])
    assert code == 1
    assert "not implemented" in capsys.readouterr().err.lower()


def test_convert_reports_not_implemented(capsys) -> None:  # noqa: ANN001
    code = main(["convert", "a.tfm", "-o", "out.nii.gz"])
    assert code == 1
    assert "not implemented" in capsys.readouterr().err.lower()
