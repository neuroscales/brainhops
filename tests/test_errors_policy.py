"""Tests of the error policy of conversions (#590)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

from brainhops import errors, io
from brainhops.datamodel.metadata import (
    ConversionReport,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from brainhops.datamodel.transformations import Linear, Rotation
from brainhops.errors import (
    UNSET,
    ConversionError,
    ConversionWarning,
    IncompatibleSystemError,
    LossyConversionError,
    apply_error_policy,
    error_policy,
    policy_action,
)
from brainhops.io.common.mgh import MghMetadata
from brainhops.io.transformations.c3d import C3dAffine

# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


class _Refused(Exception):
    """An exception that a policy raises in place of the error."""


def _loss(result: object = "the result") -> LossyConversionError:
    return LossyConversionError("lost", result=result)


def _lossy_metadata() -> Metadata:
    # MGH has no slot for a description.
    return Metadata(description="T1w", repetition_time=2.0)


def _linear() -> Linear:
    # A scaling, which a rotation holds only with a loss.
    return Linear(matrix=np.diag([2.0, 3.0]))


@pytest.fixture
def silent() -> tx.Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ----------------------------------------------------------------------
#   THE ACTIONS
# ----------------------------------------------------------------------


def test_raise_raises_the_error_itself() -> None:
    error = _loss()
    with pytest.raises(LossyConversionError) as info:
        apply_error_policy(error, "raise")
    assert info.value is error


def test_warn_warns_and_returns_the_result() -> None:
    with pytest.warns(ConversionWarning, match="lost"):
        assert apply_error_policy(_loss(), "warn") == "the result"
    # An error that carries no result returns None.
    with pytest.warns(ConversionWarning, match="ValueError: bad"):
        assert (
            apply_error_policy(ValueError("bad"), {ValueError: "warn"}) is None
        )


def test_ignore_returns_the_result_silently(silent: None) -> None:
    assert apply_error_policy(_loss(), "ignore") == "the result"
    assert (
        apply_error_policy(ValueError("bad"), {ValueError: "ignore"}) is None
    )


def test_a_report_records_the_error_and_returns_the_result(
    silent: None,
) -> None:
    report = ConversionReport()
    error = _loss()
    assert apply_error_policy(error, report) == "the result"
    assert report.errors == [error]
    assert report.lossy
    assert "LossyConversionError('lost')" in str(report)
    # A loss of metadata adds its fields instead.
    lost = ConversionReport(source="a", target="b", lost={"echo_time": 1})
    assert apply_error_policy(MetadataLossError(lost), report) is None
    assert report.lost == {"echo_time": 1}
    assert (report.source, report.target) == ("a", "b")
    # Another error is recorded, but it is not a loss.
    other = ConversionReport()
    apply_error_policy(ValueError("bad"), {ValueError: other})
    assert not other.lossy and len(other.errors) == 1


def test_an_exception_class_is_raised_from_the_error() -> None:
    error = _loss()
    with pytest.raises(_Refused) as info:
        apply_error_policy(error, _Refused)
    assert info.value.__cause__ is error


def test_an_exception_instance_is_raised_from_the_error() -> None:
    error = _loss()
    refused = _Refused("no")
    with pytest.raises(_Refused) as info:
        apply_error_policy(error, refused)
    assert info.value is refused
    assert info.value.__cause__ is error


def test_any_other_value_is_returned(silent: None) -> None:
    assert apply_error_policy(_loss(), None) is None
    assert apply_error_policy(_loss(), 0) == 0
    marker = object()
    assert apply_error_policy(ConversionError(), marker) is marker


def test_a_string_is_always_an_action() -> None:
    with pytest.raises(ValueError):
        error_policy("loud")
    with pytest.raises(ValueError):
        error_policy({ConversionError: "None"})


def test_a_policy_is_checked() -> None:
    with pytest.raises(TypeError):
        error_policy({"ConversionError": "raise"})
    with pytest.raises(TypeError):
        error_policy("raise", lossy={LossyConversionError: "warn"})


def test_without_a_policy_every_error_is_raised() -> None:
    assert error_policy() == {}
    error = ValueError("bad")
    with pytest.raises(ValueError) as info:
        apply_error_policy(error)
    assert info.value is error


# ----------------------------------------------------------------------
#   THE MAPPING
# ----------------------------------------------------------------------


def test_one_action_applies_to_the_scope() -> None:
    assert error_policy(None) == {ConversionError: None}
    assert error_policy("warn") == {ConversionError: "warn"}
    scope = (ConversionError, ValueError)
    assert error_policy(None, scope=scope) == {
        ConversionError: None,
        ValueError: None,
    }
    # A mapping is taken as written, outside the scope too.
    assert error_policy({Exception: None}) == {Exception: None}
    # An error outside the scope is raised unchanged.
    error = ValueError("bad")
    with pytest.raises(ValueError) as info:
        apply_error_policy(error, None)
    assert info.value is error
    assert apply_error_policy(error, {Exception: None}) is None


def test_a_boolean_is_not_a_policy() -> None:
    for value in (True, False):
        with pytest.raises(TypeError):
            error_policy(value)
        with pytest.raises(TypeError):
            _linear().to(Rotation, on_error=value)
    # A boolean is still a value that a mapping may return.
    assert _linear().to(Rotation, on_error={ConversionError: False}) is (False)


def test_the_action_is_chosen_by_the_method_resolution_order(
    silent: None,
) -> None:
    policy = {LossyConversionError: "ignore", ConversionError: 1}
    # The class of the error itself comes first.
    assert apply_error_policy(_loss(), policy) == "the result"
    # A subclass with no entry of its own takes the entry of a base.
    assert apply_error_policy(IncompatibleSystemError(), policy) == 1
    assert policy_action(policy, MetadataLossError) == "ignore"
    # An error whose classes have no entry is raised.
    assert policy_action(policy, ValueError) is UNSET
    with pytest.raises(ValueError):
        apply_error_policy(ValueError(), policy)


def test_a_single_value_does_not_catch_what_is_not_an_exception() -> None:
    # `Exception` is the class that a single value is read for, so an
    # interruption is still raised.
    with pytest.raises(KeyboardInterrupt):
        apply_error_policy(KeyboardInterrupt(), None)


# ----------------------------------------------------------------------
#   lossy=
# ----------------------------------------------------------------------


def test_lossy_sets_the_entry_of_a_loss() -> None:
    assert error_policy(lossy=True) == {LossyConversionError: "ignore"}
    assert error_policy(lossy="warn") == {LossyConversionError: "warn"}
    assert error_policy(lossy=None) == {}
    assert error_policy(lossy=False) == {}
    assert error_policy("raise", lossy=True) == {
        LossyConversionError: "ignore",
        ConversionError: "raise",
    }


def test_an_entry_of_on_error_wins_over_lossy() -> None:
    policy = error_policy({LossyConversionError: "raise"}, lossy=True)
    assert policy == {LossyConversionError: "raise"}


def test_lossy_accepts_a_loss_and_on_error_raises_the_rest(
    silent: None,
) -> None:
    lossy = _linear().to(Rotation, lossy=True, on_error="raise")
    assert type(lossy) is Rotation
    with pytest.raises(LossyConversionError):
        _linear().to(
            Rotation, lossy=True, on_error={LossyConversionError: "raise"}
        )


def test_to_takes_every_form_of_policy() -> None:
    lin = _linear()
    with pytest.raises(LossyConversionError):
        lin.to(Rotation)
    with pytest.warns(ConversionWarning):
        assert type(lin.to(Rotation, lossy="warn")) is Rotation
    report = ConversionReport()
    assert type(lin.to(Rotation, lossy=report)) is Rotation
    assert len(report.errors) == 1 and report.lossy
    assert lin.to(Rotation, on_error=None) is None
    # A loss is a conversion error, which the mapping finds by the order
    # of resolution.
    assert lin.to(Rotation, on_error={ConversionError: 0}) == 0
    with pytest.raises(_Refused):
        lin.to(Rotation, on_error=_Refused)
    # An error that is not a conversion error is acted on as well.
    with pytest.raises(TypeError):
        lin.to(Rotation, lossy=True, nonsense=1)
    assert lin.to(Rotation, nonsense=1, on_error={TypeError: 0}) == 0


# ----------------------------------------------------------------------
#   DEPRECATIONS
# ----------------------------------------------------------------------


def test_error_is_a_deprecated_alias_of_on_error() -> None:
    lin = _linear()
    with pytest.warns(DeprecationWarning, match="on_error="):
        assert lin.to(Rotation, error=False) is False
    with pytest.warns(DeprecationWarning):
        with pytest.raises(_Refused):
            lin.to(Rotation, error=_Refused)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(LossyConversionError):
            lin.to(Rotation, error=True)
    # As before, `error=` does not apply to an error that is not a
    # conversion error.
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError):
            lin.to(Rotation, nonsense=1, error=False)
    with pytest.raises(TypeError, match="both"):
        lin.to(Rotation, error=False, on_error=None)


def test_on_loss_is_a_deprecated_alias_of_on_error() -> None:
    meta = _lossy_metadata()
    with pytest.warns(DeprecationWarning, match="on_error="):
        with pytest.raises(MetadataLossError):
            meta.to(MghMetadata, on_loss="raise")
    report = ConversionReport()
    with pytest.warns(DeprecationWarning):
        meta.to(MghMetadata, on_loss=report)
    assert set(report.lost) == {"description"}
    with pytest.raises(TypeError, match="both"):
        meta.to(MghMetadata, on_loss="raise", on_error="raise")


def test_on_loss_is_deprecated_in_save(tmp_path: Path) -> None:
    from brainhops.datamodel.images import SingleScaleImage

    image = SingleScaleImage(
        np.zeros((2, 3, 4), np.float32), metadata=_lossy_metadata()
    )
    report = ConversionReport()
    with pytest.warns(DeprecationWarning):
        io.save(image, tmp_path / "image.mgz", on_loss=report)
    assert "description" in report.lost


def test_the_old_name_of_the_policy_type_still_imports() -> None:
    with pytest.warns(DeprecationWarning):
        assert errors.OnError is errors.ErrorPolicy


# ----------------------------------------------------------------------
#   LOSSES OF METADATA
# ----------------------------------------------------------------------


def test_a_loss_of_metadata_is_a_lossy_conversion_error() -> None:
    assert issubclass(MetadataLossError, LossyConversionError)
    assert issubclass(MetadataLossError, TypeError)
    assert issubclass(MetadataLossWarning, ConversionWarning)


def test_lossy_accepts_a_loss_of_metadata(silent: None) -> None:
    meta = _lossy_metadata()
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            meta.to(MghMetadata)
        assert type(meta.to(MghMetadata, lossy=True)) is MghMetadata
        report = ConversionReport()
        meta.to(MghMetadata, lossy=report)
        assert set(report.lost) == {"description"}
        assert meta.to(MghMetadata, on_error={MetadataLossError: 7}) == 7


def test_on_error_applies_to_a_loss_of_metadata() -> None:
    meta = _lossy_metadata()
    with pytest.raises(MetadataLossError):
        meta.to(MghMetadata, on_error="raise")
    with pytest.raises(_Refused):
        meta.to(MghMetadata, on_error={MetadataLossError: _Refused})
    with pytest.warns(MetadataLossWarning):
        meta.to(MghMetadata, on_error="warn")


def test_lossy_accepts_a_loss_of_metadata_in_a_transformation(
    silent: None,
) -> None:
    # Convert3D stores no field, so the description that the edit sets is
    # lost by the conversion of the metadata into that of the format.
    xform = C3dAffine(np.eye(4)[:-1])
    metadata = Metadata(description="r")
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            xform.to(metadata=metadata)
        edited = xform.to(metadata=metadata, lossy=True)
        assert type(edited) is C3dAffine
        report = ConversionReport()
        xform.to(metadata=metadata, lossy=report)
        assert set(report.lost) == {"description"}


def test_a_field_converter_lets_a_loss_of_metadata_through() -> None:
    # The metadata of a Convert3D matrix is converted by the converter of
    # the field, which does not wrap the loss into an error of its own.
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            C3dAffine(np.eye(4)[:-1], metadata=Metadata(description="r"))


def test_a_consolidation_returns_what_the_policy_returns() -> None:
    with metadata_loss_policy("ignore"):
        xform = C3dAffine(np.eye(4)[:-1], metadata=Metadata(description="r"))
    assert xform.consolidate(on_error={MetadataLossError: None}) is None
    # An entry for another error leaves a loss to the policy in effect.
    with pytest.warns(MetadataLossWarning):
        copy = xform.consolidate(on_error={ValueError: None})
    assert copy.metadata.description is None


def test_a_save_acts_on_a_loss_once_the_file_is_written(
    tmp_path: Path,
) -> None:
    from brainhops.datamodel.images import SingleScaleImage

    image = SingleScaleImage(
        np.zeros((2, 3, 4), np.float32), metadata=_lossy_metadata()
    )
    # The refused loss is raised, and not taken for a format that refuses
    # the image.
    with pytest.raises(MetadataLossError):
        io.save(image, tmp_path / "raise.mgz", on_error="raise")
    assert (tmp_path / "raise.mgz").exists()
    marker = object()
    returned = io.save(
        image, tmp_path / "value.mgz", on_error={MetadataLossError: marker}
    )
    assert returned is marker


# ----------------------------------------------------------------------
#   THE SCOPE OF A CALL
# ----------------------------------------------------------------------


def test_an_error_outside_the_scope_of_to_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from brainhops.datamodel._transformations import base

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a bug")

    monkeypatch.setattr(base, "convert", broken)
    lin = _linear()
    for policy in (None, "ignore", "warn", 0):
        with pytest.raises(RuntimeError, match="a bug"):
            lin.to(Rotation, on_error=policy)
    # A mapping may reach outside the scope.
    assert lin.to(Rotation, on_error={Exception: None}) is None
    assert lin.to(Rotation, on_error={RuntimeError: 1}) == 1


def test_an_error_outside_the_scope_of_a_writer_is_raised(
    tmp_path: Path,
) -> None:
    from brainhops.io.common.nifti import NiftiMetadata

    missing = tmp_path / "missing" / "header.nii"
    for policy in (None, "ignore"):
        with pytest.raises(OSError):
            NiftiMetadata().to_filename(missing, on_error=policy)
    assert NiftiMetadata().to_filename(missing, on_error={OSError: 0}) == 0


def test_an_error_outside_the_scope_of_a_consolidation_is_raised() -> None:
    with metadata_loss_policy("ignore"):
        xform = C3dAffine(np.eye(4)[:-1], metadata=Metadata(description="r"))
    # An unknown field is a `TypeError`, which is not a conversion error.
    with pytest.raises(TypeError):
        xform.consolidate(on_error=None, nonsense=1)
    assert xform.consolidate(on_error={TypeError: None}, nonsense=1) is None


def test_to_affine_returns_none_for_a_field() -> None:
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem
    from brainhops.datamodel.transformations import DisplacementField

    def system(name: str) -> CoordinateSystem:
        return CoordinateSystem(name=name, axes=[R(), A(), S()])

    field = DisplacementField(
        np.zeros((4, 5, 6, 3)), input=system("b"), output=system("c")
    )
    assert field.to_affine(on_error=None) is None
    with pytest.raises(ConversionError):
        field.to_affine()


def test_the_policy_of_a_lazy_wrapper_covers_the_edit_of_its_forward() -> None:
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem
    from brainhops.datamodel.transformations import (
        Affine,
        DisplacementField,
        Inverse,
        Sequence,
    )

    def system(name: str) -> CoordinateSystem:
        return CoordinateSystem(name=name, axes=[R(), A(), S()])

    scaling = np.diag([2.0, 1.0, 1.0, 1.0])[:3]
    field = DisplacementField(
        np.zeros((4, 5, 6, 3)), input=system("b"), output=system("c")
    )
    chain = Sequence(
        transformations=[
            Affine(scaling, input=system("a"), output=system("b")),
            field,
        ]
    )
    # `log=` edits the forward, a chain that has no tangent.
    wrapper = Inverse(forward=chain)
    with pytest.raises(ConversionError):
        wrapper.to(log=True)
    assert wrapper.to(log=True, on_error=None) is None
    with pytest.warns(DeprecationWarning):
        assert wrapper.to(log=True, error=None) is None


# ----------------------------------------------------------------------
#   WHERE A WARNING POINTS
# ----------------------------------------------------------------------


def test_a_warning_points_at_the_call_of_the_user() -> None:
    from brainhops.datamodel.transformations import Sequence

    with pytest.warns(MetadataLossWarning) as caught:
        _lossy_metadata().to(MghMetadata, on_error="warn")
    assert caught[0].filename == __file__
    chain = Sequence(transformations=[_linear()])
    with pytest.warns(DeprecationWarning) as caught:
        chain.to(Rotation, error=None)
    assert caught[0].filename == __file__
    with pytest.warns(DeprecationWarning) as caught:
        _linear().to_kind("rotation", error=None)
    assert caught[0].filename == __file__
    with pytest.warns(ConversionWarning) as caught:
        _linear().to(Rotation, lossy="warn")
    assert caught[0].filename == __file__


# ----------------------------------------------------------------------
#   WRITERS
# ----------------------------------------------------------------------


def _mgh_image() -> object:
    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.freesurfer.mgh import MghImage

    image = SingleScaleImage(
        np.zeros((2, 3, 4), np.float32), metadata=_lossy_metadata()
    )
    with metadata_loss_policy("ignore"):
        mgh = MghImage.from_instance(image)
    # MGH has no slot for a description, which the write loses.
    mgh.metadata.description = "T1w"
    return mgh


def test_a_writer_returns_what_the_policy_returns(
    tmp_path: Path, silent: None
) -> None:
    mgh = _mgh_image()
    # The value replaces the result of the write, and never reaches the
    # body of the writer as consolidated metadata.
    assert mgh.save(tmp_path / "a.mgz", on_error=None) is None
    assert mgh.to_bytes(on_error=7) == 7
    # A save of an object of the format returns what its writer returns.
    marker = object()
    assert io.save(mgh, tmp_path / "b.mgz", on_error=marker) is marker


def test_none_is_a_policy_of_a_save_that_converts(
    tmp_path: Path, silent: None
) -> None:
    from brainhops.datamodel.images import SingleScaleImage

    image = SingleScaleImage(
        np.zeros((2, 3, 4), np.float32), metadata=_lossy_metadata()
    )
    assert io.save(image, tmp_path / "a.mgz", on_error=None) is None
    assert _lossy_metadata().to(MghMetadata, on_error=None) is None
