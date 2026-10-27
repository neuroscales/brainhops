"""Tests for reading and writing Neuroglancer coordinate transforms.

A Neuroglancer coordinate transform is the JSON object that places a data
source of a layer in the viewer (#227). It is read from a `.json` file
whose keys are those of a transform, from any file with the hint
`"neuroglancer.affine"`, and from a `.ngtransform.json` file, which
`io.save` also writes, as it writes any name with the hint. The conventions
of the matrix, the dimensions and the units are those of
google/neuroglancer@abbdfc5ea97b81c0c0bf04a414f5c59f4d84d27f.
"""

import io as _io
import json

import numpy as np
import pytest
from bagof.magic import replace

import brainhops.io as io
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems
from brainhops.datamodel.transformations import Affine, Scaling, Sequence
from brainhops.datamodel.units import Unit
from brainhops.errors import ConversionError
from brainhops.io.base import format_hints
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat
from brainhops.io.transformations.base.affines import VoxelToRAS
from brainhops.io.transformations.neuroglancer import (
    NeuroglancerTransform,
    NeuroglancerTransformMetadata,
    NeuroglancerTransformRaw,
)

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# The transform of the Deep Zoom example of the Neuroglancer gallery
# (docs/gallery/index.rst at abbdfc5), with a channel dimension, written
# without a final newline, as a JavaScript program writes it.
GALLERY = (
    '{"outputDimensions":{"x":[1e-9,"m"],"y":[1e-9,"m"],"c^":[1,""]},'
    '"inputDimensions":{"x":[3.25e-7,"m"],"y":[3.25e-7,"m"],"c^":[1,""]}}'
)

# Entries with many decimals, so that a lossy round trip shows.
MATRIX = np.array(
    [
        [0.9876543210123456, 0.1, -0.05, 1.25],
        [-0.1, 0.99, 0.2, -2.5],
        [0.05, -0.2, 1.01, 0.3333333333333333],
    ]
)


def _system(unit: str, *names: str) -> _systems.CoordinateSystem:
    names = names or ("x", "y", "z")
    return _systems.CoordinateSystem(
        axes=[_axes.Axis(name=name, unit=Unit(unit)) for name in names]
    )


def _write(tmp_path, text: str, name: str = "t.json"):  # noqa: ANN001, ANN202
    file = tmp_path / name
    file.write_text(text)
    return file


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_a_gallery_transform_is_read_and_kept_byte_for_byte(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path, GALLERY)
    xform = io.load(file)
    assert type(xform) is NeuroglancerTransform
    # Without `matrix`, the transform is the identity, and the scales of
    # the input only say how the source is scaled.
    np.testing.assert_array_equal(xform.matrix, np.eye(4)[:-1])
    assert [a.name for a in xform.output.axes] == ["x", "y", "c^"]
    assert [a.unit for a in xform.output.axes] == [
        Unit("nm"),
        Unit("nm"),
        None,
    ]
    assert [a.unit for a in xform.input.axes] == [Unit("nm"), Unit("nm"), None]
    xform.save(tmp_path / "copy.json")
    assert (tmp_path / "copy.json").read_bytes() == file.read_bytes()


def test_the_linear_part_maps_physical_coordinates() -> None:
    # The linear coefficients map physical coordinates, so an identity
    # from micrometers to nanometers multiplies by 1000, and the
    # translation counts output scales of 4 nm.
    raw = NeuroglancerTransformRaw(
        content={
            "matrix": [[1, 0, 5], [0, 2, 0]],
            "outputDimensions": {"x": [4e-09, "m"], "y": [1, "nm"]},
            "inputDimensions": {"a": [1e-06, "m"], "b": [8, "um"]},
        }
    )
    xform = NeuroglancerTransform.from_raw(raw)
    np.testing.assert_array_equal(
        xform.matrix, [[1000.0, 0.0, 20.0], [0.0, 2000.0, 0.0]]
    )
    assert [a.unit for a in xform.input.axes] == [Unit("um"), Unit("um")]
    assert [a.unit for a in xform.output.axes] == [Unit("nm"), Unit("nm")]


def test_a_prefixed_unit_is_folded_into_the_scale() -> None:
    folded = NeuroglancerTransformRaw(
        content={"outputDimensions": {"x": [4, "nm"]}}
    )
    assert folded.output_dimensions == (
        NeuroglancerTransformRaw(
            content={"outputDimensions": {"x": [4e-09, "m"]}}
        ).output_dimensions
    )


def test_a_transform_without_input_dimensions_takes_the_output_units() -> None:
    raw = NeuroglancerTransformRaw(
        content={
            "matrix": [[2, 3]],
            "outputDimensions": {"t": [1, "ms"]},
        }
    )
    xform = NeuroglancerTransform.from_raw(raw)
    assert xform.input.axes[0].name is None
    assert xform.input.axes[0].unit == Unit("ms")
    np.testing.assert_array_equal(xform.matrix, [[2.0, 3.0]])


def test_the_decoded_matrix_is_read_only(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path, GALLERY))
    with pytest.raises(ValueError):
        xform.data[0, 0] = 2.0


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"matrix": [[1, 0]]},
        {"outputDimensions": {}},
        {"outputDimensions": {"x": [0, "m"]}},
        {"outputDimensions": {"x": [1, "inch"]}},
        {"outputDimensions": {"1x": [1, "m"]}},
        {"outputDimensions": {"0": [1, "m"]}},
        {"outputDimensions": {"x": [1, "m"]}, "matrix": [[1, 0, 0]]},
        {"outputDimensions": {"x": [1, "m"]}, "matrix": [[1, None]]},
        {"outputDimensions": {"x": [1, "m"]}, "sourceRank": 2},
        {
            "outputDimensions": {"x": [1, "m"]},
            "inputDimensions": {"x": [1, "m"], "y": [1, "m"]},
        },
    ],
)
def test_content_that_is_not_a_transform_is_refused(content: object) -> None:
    with pytest.raises(ParserContentError):
        NeuroglancerTransformRaw.from_text(json.dumps(content))


def test_an_input_dimension_may_be_named_by_an_integer() -> None:
    raw = NeuroglancerTransformRaw(
        content={
            "outputDimensions": {"x": [1, "m"]},
            "inputDimensions": {"0": [1, "m"]},
        }
    )
    assert raw.input_dimensions[0].name == "0"


def test_unknown_reader_options_raise(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path, GALLERY)
    with pytest.raises(TypeError):
        NeuroglancerTransform.load(file, invert=True)


# ----------------------------------------------------------------------
#   SNIFFING AND HINTS
# ----------------------------------------------------------------------


def test_only_the_keys_of_a_transform_are_sniffed(tmp_path) -> None:  # noqa: ANN001
    assert NeuroglancerTransformRaw.sniff_text(GALLERY) == Confidence.LIKELY
    content = json.loads(GALLERY)
    content["comment"] = "kept"
    other = json.dumps(content)
    # Neuroglancer ignores other keys, but such JSON is not sniffed.
    assert NeuroglancerTransformRaw.sniff_text(other) == Confidence.NO
    assert NeuroglancerTransformRaw.sniff_text('{"a": 1}') == Confidence.NO
    assert NeuroglancerTransformRaw.sniff_text("[1, 2]") == Confidence.NO
    file = _write(tmp_path, other)
    assert io.sniff(file) is None
    # A hint selects the format, which keeps the other key.
    xform = io.load(file, hint="neuroglancer.affine")
    assert type(xform) is NeuroglancerTransform
    assert xform.metadata.raw.content["comment"] == "kept"
    metadata = MetadataFormat.load(file, hint="neuroglancer.affine")
    assert isinstance(metadata, NeuroglancerTransformMetadata)


def test_the_hint_neuroglancer_is_left_to_precomputed_volumes() -> None:
    # The hint of the transform is dotted, so that it does not select a
    # transform where the hint `"neuroglancer"` of the precomputed volumes
    # is meant.
    for cls in (NeuroglancerTransform, NeuroglancerTransformMetadata):
        assert "neuroglancer.affine" in format_hints(cls)
        assert "neuroglancer" not in format_hints(cls)
        assert "ngtransform" not in format_hints(cls)


def test_the_metadata_is_read_by_dispatch(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path, GALLERY)
    metadata = MetadataFormat.load(file)
    assert isinstance(metadata, NeuroglancerTransformMetadata)
    assert metadata.to_text() == GALLERY


def test_metadata_without_a_record_is_not_written() -> None:
    with pytest.raises(WriterError):
        NeuroglancerTransformMetadata().to_text()


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_an_anatomical_affine_is_saved_in_millimeters(tmp_path) -> None:  # noqa: ANN001
    affine = Affine(MATRIX, input=_systems.RASmm(), output=_systems.RASmm())
    io.save(affine, tmp_path / "t.ngtransform.json")
    content = json.loads((tmp_path / "t.ngtransform.json").read_text())
    assert content["outputDimensions"] == {
        name: [0.001, "m"] for name in "xyz"
    }
    assert content["inputDimensions"] == content["outputDimensions"]
    # In the same units on both sides, the stored matrix is the matrix.
    np.testing.assert_array_equal(content["matrix"], MATRIX)
    back = io.load(tmp_path / "t.ngtransform.json")
    np.testing.assert_array_equal(back.matrix, MATRIX)
    assert back.output.compatible_with(_systems.RASmm())


def test_a_hint_saves_to_a_plain_json_name(tmp_path) -> None:  # noqa: ANN001
    affine = Affine(MATRIX, input=_systems.RASmm(), output=_systems.RASmm())
    file = tmp_path / "t.json"
    with pytest.raises(WriterError):
        io.save(affine, file)
    io.save(affine, file, hint="neuroglancer.affine")
    expected = NeuroglancerTransform.from_any(affine).to_text()
    assert file.read_text() == expected
    back = io.load(file, hint="neuroglancer.affine")
    np.testing.assert_array_equal(back.matrix, MATRIX)


def test_units_are_converted_exactly() -> None:
    # One millimeter is a million nanometers, and the stored linear part
    # is the ratio of physical lengths, 1.
    affine = Affine(
        [[1e6, 0, 0, 5], [0, 2e6, 0, 6], [0, 0, 1e6, 7]],
        input=_system("mm"),
        output=_system("nm"),
    )
    xform = NeuroglancerTransform.from_any(affine)
    content = json.loads(xform.to_text())
    assert content["matrix"] == [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 2.0, 0.0, 6.0],
        [0.0, 0.0, 1.0, 7.0],
    ]
    assert content["outputDimensions"]["x"] == [1e-09, "m"]
    back = NeuroglancerTransform.from_text(xform.to_text())
    np.testing.assert_array_equal(back.matrix, affine.matrix)


@pytest.mark.parametrize(
    ("unit", "dimension"),
    [
        ("Hz", [1, "Hz"]),
        ("kHz", [1000, "Hz"]),
        ("1/s", [1, "Hz"]),
        ("rad/s", [1, "rad/s"]),
        ("radian/second", [1, "rad/s"]),
        ("mrad/s", [0.001, "rad/s"]),
        ("rad/ms", [1000, "rad/s"]),
        ("rad*Hz", [1, "rad/s"]),
    ],
)
def test_a_frequency_is_written_in_hertz_or_radians_per_second(
    unit: str, dimension: list
) -> None:
    # pint converts hertz into radians per second, so the choice between
    # the two Neuroglancer units depends on the radian in the axis unit.
    affine = Affine(np.eye(4)[:-1], input=_system(unit), output=_system(unit))
    content = json.loads(NeuroglancerTransform.from_any(affine).to_text())
    assert content["outputDimensions"]["x"] == dimension
    assert content["inputDimensions"]["x"] == dimension


@pytest.mark.parametrize("unit", ["sr/s", "rad**2/s"])
def test_a_frequency_with_a_squared_radian_is_refused(unit: str) -> None:
    affine = Affine(np.eye(4)[:-1], input=_system(unit), output=_system(unit))
    with pytest.raises(ConversionError, match="no Neuroglancer equivalent"):
        NeuroglancerTransform.from_any(affine)


def test_a_chain_of_affines_is_reduced_to_one() -> None:
    first = Scaling(
        [2.0, 3.0, 4.0], input=_systems.RASmm(), output=_systems.RASmm()
    )
    chain = Sequence([first, first])
    xform = NeuroglancerTransform.from_any(chain)
    np.testing.assert_array_equal(
        xform.matrix, np.diag([4.0, 9.0, 16.0, 1.0])[:-1]
    )


@pytest.mark.parametrize(
    "affine, reason",
    [
        (Affine(MATRIX), "not a known system"),
        (VoxelToRAS(MATRIX), "counts voxels"),
        (Affine(MATRIX, input=_system("inch"), output=_system("mm")), "inch"),
        (
            Affine(
                MATRIX,
                input=_system("mm"),
                output=_system("mm", "x", "y", "x"),
            ),
            "distinct",
        ),
        (
            Affine(
                MATRIX,
                input=_system("mm"),
                output=_system("mm", "x", "y", "left-to-right"),
            ),
            "name",
        ),
    ],
    ids=["unknown", "voxel", "inch", "duplicate", "invalid-name"],
)
def test_an_affine_without_dimensions_is_refused(
    tmp_path,  # noqa: ANN001
    affine: Affine,
    reason: str,
) -> None:
    with pytest.raises(ConversionError, match=reason):
        NeuroglancerTransform.from_any(affine)
    with pytest.raises(WriterError, match=reason):
        io.save(affine, tmp_path / "t.ngtransform.json")
    assert not (tmp_path / "t.ngtransform.json").exists()


def test_a_transformation_without_systems_is_refused() -> None:
    xform = NeuroglancerTransform(MATRIX)
    with pytest.raises(UnrepresentableTransformationError):
        xform.consolidate()
    with pytest.raises(UnrepresentableTransformationError):
        NeuroglancerTransform(
            np.eye(4), input=_system("mm"), output=_system("mm")
        ).consolidate()


def test_a_plain_json_name_is_not_claimed(tmp_path) -> None:  # noqa: ANN001
    affine = Affine(MATRIX, input=_systems.RASmm(), output=_systems.RASmm())
    with pytest.raises(WriterError):
        io.save(affine, tmp_path / "t.json")
    # The format itself writes any name.
    NeuroglancerTransform.from_any(affine).save(tmp_path / "t.json")
    assert type(io.load(tmp_path / "t.json")) is NeuroglancerTransform


def test_a_transform_is_written_to_a_stream() -> None:
    affine = Affine(MATRIX, input=_systems.RASmm(), output=_systems.RASmm())
    stream = _io.StringIO()
    NeuroglancerTransform.from_any(affine).save(stream)
    assert stream.getvalue().endswith("}\n")
    back = NeuroglancerTransform.from_text(stream.getvalue())
    np.testing.assert_array_equal(back.matrix, MATRIX)


# ----------------------------------------------------------------------
#   OVERRIDES
# ----------------------------------------------------------------------


def _loaded(tmp_path):  # noqa: ANN001, ANN202
    text = json.dumps(
        {
            "sourceRank": 2,
            "matrix": [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]],
            "outputDimensions": {
                "x": [1e-09, "m"],
                "y": [1e-09, "m"],
                "z": [4e-08, "m"],
            },
            "comment": "kept",
        }
    )
    return NeuroglancerTransform.load(_write(tmp_path, text))


def test_an_untouched_transformation_keeps_its_record(tmp_path) -> None:  # noqa: ANN001
    xform = _loaded(tmp_path)
    assert xform.consolidate().metadata is xform.metadata
    assert replace(xform).metadata is xform.metadata
    copy = NeuroglancerTransform.from_instance(xform)
    assert copy.metadata is xform.metadata and copy._data is None


def test_an_assigned_matrix_keeps_the_dimensions(tmp_path) -> None:  # noqa: ANN001
    xform = _loaded(tmp_path)
    xform.data = np.diag([2.0, 3.0, 4.0, 1.0])[:-1]
    content = xform.consolidate().metadata.raw.content
    assert content["sourceRank"] == 2 and content["comment"] == "kept"
    assert content["outputDimensions"]["z"] == [4e-08, "m"]
    assert "inputDimensions" not in content
    np.testing.assert_array_equal(
        content["matrix"], np.diag([2.0, 3.0, 4.0, 1.0])[:-1]
    )


@pytest.mark.parametrize("side", ["input", "output"])
def test_an_assigned_system_is_encoded(tmp_path, side: str) -> None:  # noqa: ANN001
    xform = _loaded(tmp_path)
    setattr(xform, side, _system("um", "a", "b", "c"))
    consolidated = xform.consolidate()
    assert consolidated.metadata is not xform.metadata
    content = consolidated.metadata.raw.content
    key = side + "Dimensions"
    assert content[key] == {name: [1e-06, "m"] for name in "abc"}
    assert getattr(consolidated, side) == _system("um", "a", "b", "c")
    # The input of the record took the units of its output, which an
    # assigned output does not change.
    if side == "output":
        assert content["inputDimensions"]["z"] == [4e-08, "m"]


def test_assigned_metadata_is_read(tmp_path) -> None:  # noqa: ANN001
    xform = _loaded(tmp_path)
    # The output scale of z is 40 nm, so its translation is 7 * 40 nm.
    np.testing.assert_array_equal(
        xform.matrix, [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 280]]
    )
    xform.metadata = NeuroglancerTransformMetadata.from_text(GALLERY)
    np.testing.assert_array_equal(xform.matrix, np.eye(4)[:-1])
    assert xform.consolidate().metadata is xform.metadata


def test_a_matrix_of_another_rank_needs_new_systems(tmp_path) -> None:  # noqa: ANN001
    xform = _loaded(tmp_path)
    xform.data = np.eye(3)[:-1]
    with pytest.raises(UnrepresentableTransformationError, match="rank"):
        xform.consolidate()
    xform.input = _system("mm", "a", "b")
    xform.output = _system("mm", "a", "b")
    content = xform.consolidate().metadata.raw.content
    assert "sourceRank" not in content
    assert content["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
