"""Tests of the error policy of conversions (#590)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

from brainhops import io
from brainhops._core.errors import (
    apply_error_policy,
    error_policy,
    policy_action,
)
from brainhops.datamodel.metadata import (
    ConversionReport,
    Metadata,
    MetadataLossError,
    MetadataLossWarning,
    metadata_loss_policy,
)
from brainhops.datamodel.metadata._report import apply_loss_policy
from brainhops.datamodel.transformations import Linear, Rotation
from brainhops.errors import (
    UNSET,
    ConversionError,
    ConversionWarning,
    IncompatibleSystemError,
    LossyConversionError,
)
from brainhops.io.base.parsers import TextFileWriter
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
        error_policy("raise", on_loss={LossyConversionError: "warn"})


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
#   on_loss=
# ----------------------------------------------------------------------


def test_on_loss_sets_the_entry_of_a_loss() -> None:
    assert error_policy(on_loss="warn") == {LossyConversionError: "warn"}
    assert error_policy(on_loss=None) == {LossyConversionError: None}
    assert error_policy() == {}
    assert error_policy("raise", on_loss="ignore") == {
        LossyConversionError: "ignore",
        ConversionError: "raise",
    }
    for value in (True, False):
        with pytest.raises(TypeError):
            error_policy(on_loss=value)


def test_an_entry_of_on_error_wins_over_on_loss() -> None:
    policy = error_policy({LossyConversionError: "raise"}, on_loss="ignore")
    assert policy == {LossyConversionError: "raise"}


def test_on_loss_takes_precedence_for_a_loss(silent: None) -> None:
    lossy = _linear().to(Rotation, on_loss="ignore", on_error="raise")
    assert type(lossy) is Rotation
    with pytest.raises(LossyConversionError):
        _linear().to(Rotation, on_loss="raise", on_error="ignore")
    with pytest.raises(LossyConversionError):
        _linear().to(
            Rotation,
            on_loss="ignore",
            on_error={LossyConversionError: "raise"},
        )
    # Without on_loss=, a loss falls under on_error=, so "ignore" returns
    # the lossy result.
    assert type(_linear().to(Rotation, on_error="ignore")) is Rotation
    # None is an action, which returns None in place of the lossy result.
    assert _linear().to(Rotation, on_loss=None) is None


def test_to_takes_every_form_of_policy() -> None:
    lin = _linear()
    with pytest.raises(LossyConversionError):
        lin.to(Rotation)
    with pytest.warns(ConversionWarning):
        assert type(lin.to(Rotation, on_loss="warn")) is Rotation
    report = ConversionReport()
    assert type(lin.to(Rotation, on_loss=report)) is Rotation
    assert len(report.errors) == 1 and report.lossy
    assert lin.to(Rotation, on_error=None) is None
    # A loss is a conversion error, which the mapping finds by the order
    # of resolution.
    assert lin.to(Rotation, on_error={ConversionError: 0}) == 0
    with pytest.raises(_Refused):
        lin.to(Rotation, on_error=_Refused)
    # An error that is not a conversion error is acted on as well.
    with pytest.raises(TypeError):
        lin.to(Rotation, on_loss="ignore", nonsense=1)
    assert lin.to(Rotation, nonsense=1, on_error={TypeError: 0}) == 0


# ----------------------------------------------------------------------
#   on_loss= OF THE METADATA
# ----------------------------------------------------------------------


def test_on_loss_is_the_loss_half_of_a_metadata_policy(
    silent: None,
) -> None:
    meta = _lossy_metadata()
    with pytest.raises(MetadataLossError):
        meta.to(MghMetadata, on_loss="raise")
    report = ConversionReport()
    meta.to(MghMetadata, on_loss=report)
    assert set(report.lost) == {"description"}
    # on_loss= wins over on_error= for a loss.
    with pytest.raises(MetadataLossError):
        meta.to(MghMetadata, on_loss="raise", on_error="ignore")
    assert meta.to(MghMetadata, on_loss=None, on_error="raise") is None


def test_on_loss_fills_the_report_of_a_save(
    tmp_path: Path, silent: None
) -> None:
    from brainhops.datamodel.images import SingleScaleImage

    image = SingleScaleImage(
        np.zeros((2, 3, 4), np.float32), metadata=_lossy_metadata()
    )
    report = ConversionReport()
    io.save(image, tmp_path / "image.mgz", on_loss=report)
    assert "description" in report.lost


def test_on_loss_applies_to_a_consolidation(silent: None) -> None:
    with metadata_loss_policy("ignore"):
        xform = C3dAffine(np.eye(4)[:-1], metadata=Metadata(description="r"))
    with pytest.raises(MetadataLossError):
        xform.consolidate(on_loss="raise", on_error="ignore")
    assert xform.consolidate(on_loss=None) is None


# ----------------------------------------------------------------------
#   LOSSES OF METADATA
# ----------------------------------------------------------------------


def test_a_loss_of_metadata_is_a_lossy_conversion_error() -> None:
    assert issubclass(MetadataLossError, LossyConversionError)
    assert issubclass(MetadataLossError, TypeError)
    assert issubclass(MetadataLossWarning, ConversionWarning)


def test_on_loss_accepts_a_loss_of_metadata(silent: None) -> None:
    meta = _lossy_metadata()
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            meta.to(MghMetadata)
        assert type(meta.to(MghMetadata, on_loss="ignore")) is MghMetadata
        report = ConversionReport()
        meta.to(MghMetadata, on_loss=report)
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


def test_on_loss_accepts_a_loss_of_metadata_in_a_transformation(
    silent: None,
) -> None:
    # Convert3D stores no field, so the description that the edit sets is
    # lost by the conversion of the metadata into that of the format.
    xform = C3dAffine(np.eye(4)[:-1])
    metadata = Metadata(description="r")
    with metadata_loss_policy("raise"):
        with pytest.raises(MetadataLossError):
            xform.to(metadata=metadata)
        edited = xform.to(metadata=metadata, on_loss="ignore")
        assert type(edited) is C3dAffine
        report = ConversionReport()
        xform.to(metadata=metadata, on_loss=report)
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


# ----------------------------------------------------------------------
#   WHERE A WARNING POINTS
# ----------------------------------------------------------------------


def test_a_warning_points_at_the_call_of_the_user() -> None:
    from brainhops.datamodel.transformations import Sequence

    with pytest.warns(MetadataLossWarning) as caught:
        _lossy_metadata().to(MghMetadata, on_error="warn")
    assert caught[0].filename == __file__
    chain = Sequence(transformations=[_linear()])
    with pytest.warns(ConversionWarning) as caught:
        chain.to(Rotation, on_loss="warn")
    assert caught[0].filename == __file__
    with pytest.warns(ConversionWarning) as caught:
        _linear().to_rotation(on_loss="warn")
    assert caught[0].filename == __file__
    with pytest.warns(ConversionWarning) as caught:
        _linear().to(Rotation, on_loss="warn")
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


def test_a_nested_writer_follows_the_policy_of_the_outer_call(
    tmp_path: Path, silent: None
) -> None:
    mgh = _mgh_image()
    # `save` calls the other entry points of the writer without a
    # policy, so the loss follows the policy of `save`, and only `save`
    # returns the value in place of its result.
    marker = object()
    assert mgh.save(tmp_path / "a.mgz", on_error=marker) is marker
    report = ConversionReport()
    assert mgh.save(tmp_path / "b.mgz", on_loss=report) is None
    assert report.lost == {"description": "T1w"}


# ----------------------------------------------------------------------
#   WRITERS THAT YIELD THEIR LINES
# ----------------------------------------------------------------------


class _LineWriter(TextFileWriter):
    """A writer whose lines lose metadata after the first one."""

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        yield "first"
        apply_loss_policy(
            ConversionReport(lost={"description": "T1w"}), stacklevel=2
        )
        yield "second"


def test_the_policy_of_a_generator_applies_while_it_runs() -> None:
    writer = _LineWriter()
    # The body of a generator runs while its lines are read, under the
    # policy of the call.
    with pytest.raises(MetadataLossError):
        list(writer.to_lines(on_loss="raise"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = list(writer.to_lines(on_loss="ignore"))
    assert lines == ["first", "second"]
    report = ConversionReport()
    assert list(writer.to_lines(on_loss=report)) == ["first", "second"]
    assert report.lost == {"description": "T1w"}
    with pytest.warns(MetadataLossWarning):
        assert list(writer.to_lines()) == ["first", "second"]


def test_a_value_of_the_policy_ends_the_lines(silent: None) -> None:
    # The value cannot replace the result of a generator, so the lines
    # end where the body raised.
    assert list(_LineWriter().to_lines(on_error=None)) == ["first"]


def test_the_policy_of_a_generator_does_not_leak_to_its_consumer() -> None:
    lines = _LineWriter().to_lines(on_loss="raise")
    assert next(lines) == "first"
    # Between two lines, the code that reads them runs under the policy
    # in effect, which warns.
    with pytest.warns(MetadataLossWarning):
        apply_loss_policy(ConversionReport(lost={"echo_time": 0.03}))
    with pytest.raises(MetadataLossError):
        next(lines)


def test_the_policy_of_a_generator_is_checked_when_it_is_called() -> None:
    with pytest.raises(TypeError):
        _LineWriter().to_lines(on_error=True)


# ----------------------------------------------------------------------
#   THE FORMATS
# ----------------------------------------------------------------------

# The methods of the formats that apply a policy themselves. Every other
# method of a format runs under the policy of its caller.
_ENTRY_POINTS = {
    ("brainhops.io.base._base", "_FileBasedModelMixin", "consolidate"),
    ("brainhops.io.metadata._base", "MetadataFormat", "from_instance"),
}


def _subclasses(cls: type) -> tx.Set[type]:
    found = set()
    pending = [cls]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                pending.append(sub)
    return found


def test_no_method_of_a_format_declares_a_policy() -> None:
    import inspect

    import brainhops.io.images  # noqa: F401
    import brainhops.io.metadata  # noqa: F401
    import brainhops.io.transformations  # noqa: F401
    from brainhops.io.base import FileFormat
    from brainhops.io.base.parsers import FileReader, FileWriter

    classes = set()
    for base in (FileFormat, FileReader, FileWriter):
        for cls in _subclasses(base):
            classes.update(cls.__mro__)
    declared = []
    entry_points = set()
    for cls in classes:
        if not cls.__module__.startswith("brainhops.io"):
            continue
        for name, member in vars(cls).items():
            function = getattr(member, "__func__", member)
            if not inspect.isfunction(function):
                continue
            parameters = inspect.signature(function).parameters
            if not {"on_error", "on_loss"} & set(parameters):
                continue
            key = (cls.__module__, cls.__qualname__, name)
            if key in _ENTRY_POINTS:
                entry_points.add(key)
                continue
            declared.append(".".join(key))
    assert sorted(declared) == []
    assert entry_points == _ENTRY_POINTS
