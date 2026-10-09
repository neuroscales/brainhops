"""Tests for elastix and transformix transform parameter files."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from brainhops import io
from brainhops.datamodel import systems
from brainhops.datamodel import transformations as xforms
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations import itk
from brainhops.io.transformations.elastix import (
    ElastixParameterTransform,
    ElastixTomlTransform,
    ElastixTransform,
)
from brainhops.io.transformations.elastix._parser import (
    read_text_map,
    read_toml_map,
)

data_dir = Path(__file__).parent / "data" / "elastix"

# Cases with a stored transformix answer, keyed to the file read for each.
CASES = {
    path.name[: -len("_expected.npz")]: path
    for path in sorted(data_dir.glob("*_expected.npz"))
}
FILES = {name: data_dir / f"{name}.txt" for name in CASES}
FILES["chain"] = data_dir / "chain_TransformParameters.1.txt"
FILES["chain3"] = data_dir / "chain3_TransformParameters.2.txt"
FILES["euler3d_toml"] = data_dir / "euler3d.toml"


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map LPS points through an LPS-to-LPS transformation."""
    source = xforms.CoordinatesField(field=np.asarray(points, float))
    out = xforms.Sequence(transformations=[source, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _grid_points(vox2lps: np.ndarray, shape: tuple) -> np.ndarray:
    ndim = len(shape)
    ijk = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1)
    return ijk @ vox2lps[:, :ndim].T + vox2lps[:, ndim]


def _text(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


EULER = """
(Transform "EulerTransform")
(NumberOfParameters 6)
(TransformParameters 0.1 -0.2 0.3 1.5 -2 3.25)
(InitialTransformParameterFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(FixedImageDimension 3)
(MovingImageDimension 3)
(CenterOfRotationPoint 4.5 -2 7.25)
(ComputeZYX "false")
"""

TRANSLATION = """
(Transform "TranslationTransform")
(NumberOfParameters 3)
(TransformParameters 1 2 3)
(FixedImageDimension 3)
(MovingImageDimension 3)
"""


# ----------------------------------------------------------------------
#   TRANSFORMIX GROUND TRUTH
# ----------------------------------------------------------------------
# The fixtures come from data/elastix/generate_elastix_fixtures.py.


@pytest.mark.parametrize("name", sorted(CASES))
def test_fixed_geometry_is_the_grid_transformix_resamples_onto(
    name: str,
) -> None:
    """The fixed grid is read with a column-major Direction, as in elastix."""
    expected = np.load(CASES[name])
    geometry = io.load(FILES[name]).fixed_geometry
    np.testing.assert_allclose(
        np.asarray(geometry.transformation.matrix),
        expected["vox2lps"],
        atol=1e-12,
    )
    assert tuple(geometry.shape) == expected["disp"].shape[:-1]


@pytest.mark.parametrize("name", sorted(CASES))
def test_points_map_as_transformix_maps_them(name: str) -> None:
    """Every point of the fixed grid lands where transformix sends it."""
    expected = np.load(CASES[name])
    shape = expected["disp"].shape[:-1]
    points = _grid_points(expected["vox2lps"], shape)
    xform = io.load(FILES[name])
    # Transformix computes the deformation field in single precision.
    np.testing.assert_allclose(
        _apply(xform, points), points + expected["disp"], atol=1e-5
    )


def test_fixtures_still_match_transformix(tmp_path: Path) -> None:
    """The stored expectations still equal the output of transformix."""
    itk_ = pytest.importorskip("itk")
    if not hasattr(itk_, "ParameterObject"):
        pytest.skip("ITK is installed without elastix")
    for name in ("euler3d", "bspline3d"):
        parameter_object = itk_.ParameterObject.New()
        parameter_object.ReadParameterFile(str(FILES[name]))
        moving = itk_.image_from_array(np.zeros((6, 6, 6), np.float32))
        field = itk_.transformix_deformation_field(
            moving, parameter_object, output_directory=str(tmp_path)
        )
        disp = itk_.array_from_image(field).transpose(2, 1, 0, 3)
        np.testing.assert_allclose(
            disp, np.load(CASES[name])["disp"], atol=1e-5
        )


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, cls",
    [
        ("euler3d.txt", ElastixParameterTransform),
        ("bspline3d.txt", ElastixParameterTransform),
        ("euler3d.toml", ElastixTomlTransform),
    ],
)
def test_files_are_dispatched(filename: str, cls: type) -> None:
    path = data_dir / filename
    assert io.transformations.sniff(path) is cls
    assert io.sniff(path) is cls
    assert type(io.transformations.load(path)) is cls
    assert type(io.load(path)) is cls


@pytest.mark.parametrize(
    "hint, filename, cls",
    [
        ("elastix", "euler3d.txt", ElastixParameterTransform),
        ("transformix", "euler3d.txt", ElastixParameterTransform),
        ("elastix.params", "euler3d.txt", ElastixParameterTransform),
        ("elastix", "euler3d.toml", ElastixTomlTransform),
        ("elastix.toml", "euler3d.toml", ElastixTomlTransform),
    ],
)
def test_hints(hint: str, filename: str, cls: type) -> None:
    xform = io.transformations.load(data_dir / filename, hint=hint)
    assert type(xform) is cls


def test_a_registration_parameter_file_is_not_claimed(
    tmp_path: Path,
) -> None:
    """A registration parameter file names a Transform but is not claimed."""
    path = _text(
        tmp_path,
        "Parameters.Rigid.txt",
        '(Transform "EulerTransform")\n'
        '(Metric "AdvancedMattesMutualInformation")\n'
        "(MaximumNumberOfIterations 250)\n",
    )
    assert ElastixParameterTransform.sniff(path) == 0
    with pytest.raises(ParserContentError):
        ElastixParameterTransform.from_file(path)


def test_a_text_matrix_is_not_claimed(tmp_path: Path) -> None:
    path = _text(tmp_path, "matrix.txt", "1 0 0 0\n0 1 0 0\n0 0 1 0\n")
    assert ElastixParameterTransform.sniff(path) == 0
    assert ElastixTomlTransform.sniff_text("1 0 0\n") == 0


# ----------------------------------------------------------------------
#   SYNTAX
# ----------------------------------------------------------------------


def test_text_syntax() -> None:
    pmap = read_text_map(
        [
            "// a comment",
            "",
            '(Name "with space" "a//b")  // trailing',
            "\t(Numbers\t1 -2 3.5 1e-3)",
            "(Word true)",
            "(TransformParameters 1 2 3)",
        ]
    )
    assert pmap["Name"] == ("with space", "a//b")
    assert pmap["Numbers"] == (1, -2, 3.5, 1e-3)
    assert isinstance(pmap["Numbers"][0], int)
    assert pmap["Word"] == ("true",)
    assert pmap["TransformParameters"].dtype == np.float64


@pytest.mark.parametrize(
    "line",
    [
        "Transform EulerTransform",
        '(Transform "EulerTransform)',
        '("Transform" 1)',
        "(Bad-Name 1)",
    ],
)
def test_text_syntax_errors(line: str) -> None:
    with pytest.raises(ParserContentError):
        read_text_map([line])


def test_a_parameter_given_twice_is_refused() -> None:
    with pytest.raises(ParserContentError, match="more than once"):
        read_text_map(["(Size 1 2)", "(Size 3 4)"])


def test_toml_syntax() -> None:
    pmap = read_toml_map(
        [
            "# comment",
            'Transform = "EulerTransform" # trailing',
            "Size = [10, 12, 14]",
            "Spacing = [1, 1.5, 2,]",
            "ComputeZYX = false",
            'Path = "a#b"',
            "TransformParameters = [0.1, -0.2]",
        ]
    )
    assert pmap["Transform"] == ("EulerTransform",)
    assert pmap["Size"] == (10, 12, 14)
    assert pmap["Spacing"] == (1, 1.5, 2)
    assert pmap["ComputeZYX"] == ("false",)
    assert pmap["Path"] == ("a#b",)
    np.testing.assert_array_equal(pmap["TransformParameters"], [0.1, -0.2])


def test_toml_and_text_files_read_the_same() -> None:
    text = io.load(data_dir / "euler3d.txt")
    toml = io.load(data_dir / "euler3d.toml")
    assert text.parameter_map.keys() == toml.parameter_map.keys()
    np.testing.assert_array_equal(text[0].parameters, toml[0].parameters)


# ----------------------------------------------------------------------
#   BLOCKS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, cls",
    [
        ("translation3d", itk._common.ItkTranslationStruct),
        ("euler2d", itk._common.ItkEuler2DStruct),
        ("euler3d", itk._common.ItkEuler3DStruct),
        ("similarity2d", itk._common.ItkSimilarity2DStruct),
        ("similarity3d", itk._common.ItkSimilarity3DStruct),
        ("affine3d", itk._common.ItkAffineStruct),
        ("affinelog3d", itk._common.ItkAffineStruct),
        ("affinedti3d", itk._common.ItkAffineStruct),
        ("bspline3d", itk._common.ItkBSplineStruct),
    ],
)
def test_blocks_are_itk_blocks(name: str, cls: type) -> None:
    xform = io.load(FILES[name])
    assert len(xform) == 1
    assert isinstance(xform[0], cls)
    assert xform.block is xform[0]
    assert xform.input == xform.output == xform[0].input


def test_euler_center_and_order_are_read() -> None:
    plain = io.load(FILES["euler3d"])[0]
    zyx = io.load(FILES["euler3d_zyx"])[0]
    np.testing.assert_allclose(plain.center, [4.5, -2.0, 7.25])
    assert plain.compute_zyx is False
    assert zyx.compute_zyx is True


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_bspline_degree(degree: int) -> None:
    name = "bspline3d" if degree == 3 else f"bspline3d_order{degree}"
    block = io.load(FILES[name])[0]
    assert block.degree == degree
    assert block.displacement.degree == degree
    assert block.displacement.store == "coefficients"


def test_bspline_grid_index_moves_the_first_coefficient() -> None:
    """The first coefficient lies GridIndex voxels from GridOrigin."""
    xform = io.load(FILES["bspline3d"])
    pmap = xform.parameter_map
    spacing = np.asarray(pmap["GridSpacing"], float)
    index = np.asarray(pmap["GridIndex"], float)
    direction = np.asarray(pmap["GridDirection"], float).reshape(3, 3).T
    origin = np.asarray(pmap["GridOrigin"], float)
    assert index.any()
    vox2lps, shape = xform[0]._grid
    assert shape == tuple(int(v) for v in pmap["GridSize"])
    np.testing.assert_allclose(vox2lps[:, :3], direction * spacing)
    np.testing.assert_allclose(
        vox2lps[:, 3], origin + direction @ (spacing * index)
    )


def test_itk_transform_parameters_are_accepted(tmp_path: Path) -> None:
    """Parameters stored under their ITK names are also read, as in elastix."""
    path = _text(
        tmp_path,
        "itk.txt",
        '(Transform "AffineTransform")\n'
        "(ITKTransformParameters 1 0 0 0 1 0 0 0 1 1 2 3)\n"
        "(ITKTransformFixedParameters 4 5 6)\n"
        "(FixedImageDimension 3)\n",
    )
    block = io.load(path)[0]
    np.testing.assert_allclose(block.center, [4, 5, 6])
    np.testing.assert_allclose(block.parameters[-3:], [1, 2, 3])


def test_number_of_parameters_is_checked(tmp_path: Path) -> None:
    path = _text(
        tmp_path,
        "bad.txt",
        TRANSLATION.replace(
            "(NumberOfParameters 3)", "(NumberOfParameters 4)"
        ),
    )
    with pytest.raises(ParserContentError, match="NumberOfParameters"):
        ElastixParameterTransform.from_file(path)


@pytest.mark.parametrize(
    "replace, match",
    [
        (
            ('"TranslationTransform"', '"SplineKernelTransform"'),
            "not supported",
        ),
    ],
)
def test_unsupported_transforms_are_refused(
    tmp_path: Path, replace: tuple, match: str
) -> None:
    path = _text(tmp_path, "x.txt", TRANSLATION.replace(*replace))
    with pytest.raises(NotImplementedError, match=match):
        ElastixParameterTransform.from_file(path)


def test_cyclic_bsplines_are_refused(tmp_path: Path) -> None:
    text = (data_dir / "bspline2d.txt").read_text()
    text = text.replace(
        '(UseCyclicTransform "false")', '(UseCyclicTransform "true")'
    )
    with pytest.raises(NotImplementedError, match="Cyclic"):
        ElastixParameterTransform.from_file(
            _text(tmp_path, "cyclic.txt", text)
        )


def test_a_center_given_as_an_index_is_refused(tmp_path: Path) -> None:
    """A center given as a voxel index (elastix < 3.402) is refused."""
    text = EULER.replace("CenterOfRotationPoint", "CenterOfRotation")
    with pytest.raises(NotImplementedError, match="CenterOfRotation"):
        ElastixParameterTransform.from_file(_text(tmp_path, "old.txt", text))


def test_ignored_direction_cosines_warn(tmp_path: Path) -> None:
    path = _text(
        tmp_path, "nodir.txt", EULER + '(UseDirectionCosines "false")'
    )
    with pytest.warns(UserWarning, match="UseDirectionCosines"):
        io.load(path)


# ----------------------------------------------------------------------
#   CHAINS
# ----------------------------------------------------------------------


def test_chain_is_initial_first() -> None:
    """The initial transform applies first, so its blocks come first."""
    xform = io.load(FILES["chain"])
    assert len(xform) == 2
    assert isinstance(xform.initial, ElastixParameterTransform)
    assert xform[0] is xform.initial[0]
    assert isinstance(xform[0], itk._common.ItkAffineStruct)
    assert isinstance(xform[1], itk._common.ItkBSplineStruct)


def test_chain_follows_the_deprecated_key() -> None:
    """The deprecated key InitialTransformParametersFileName is followed."""
    xform = io.load(FILES["chain3"])
    assert "InitialTransformParametersFileName" in xform.parameter_map
    assert len(xform) == 3
    assert len(xform.initial) == 2
    assert len(xform.initial.initial) == 1


def test_chain_is_not_followed_on_request() -> None:
    xform = ElastixParameterTransform.from_file(FILES["chain"], initial=False)
    assert xform.initial is None
    assert len(xform) == 1
    assert xform.initial_filename == "chain_TransformParameters.0.txt"


def test_chain_initial_can_be_given(tmp_path: Path) -> None:
    text = TRANSLATION + '(InitialTransformParameterFileName "missing.txt")'
    path = _text(tmp_path, "t.txt", text)
    with pytest.raises(FileNotFoundError, match="initial"):
        ElastixParameterTransform.from_file(path)

    # An initial transform given by file name.
    xform = ElastixParameterTransform.from_file(
        path, initial=data_dir / "euler3d.txt"
    )
    assert len(xform) == 2
    # An initial transform given as a transformation.
    shift = xforms.Translation([1.0, 0.0, 0.0])
    xform = ElastixParameterTransform.from_file(path, initial=shift)
    assert xform[0] is shift


def test_chain_from_text_needs_a_location() -> None:
    """Text from memory resolves an initial transform from the cwd only."""
    text = TRANSLATION + '(InitialTransformParameterFileName "nowhere.txt")'
    with pytest.raises(FileNotFoundError):
        ElastixParameterTransform.from_text(text)
    assert len(ElastixParameterTransform.from_text(text, initial=False)) == 1


def test_chain_moved_absolute_path_is_found_by_name(tmp_path: Path) -> None:
    """A moved absolute path is found by base name beside the file."""
    shutil.copy(FILES["euler3d"], tmp_path / "TransformParameters.0.txt")
    text = TRANSLATION + (
        "(InitialTransformParameterFileName "
        '"/old/place/TransformParameters.0.txt")'
    )
    path = _text(tmp_path, "TransformParameters.1.txt", text)
    assert len(io.load(path)) == 2


def test_chain_loops_are_refused(tmp_path: Path) -> None:
    a = TRANSLATION + '(InitialTransformParameterFileName "b.txt")'
    b = TRANSLATION + '(InitialTransformParameterFileName "a.txt")'
    _text(tmp_path, "b.txt", b)
    with pytest.raises(ParserContentError, match="loop"):
        ElastixParameterTransform.from_file(_text(tmp_path, "a.txt", a))
    self_ = TRANSLATION + '(InitialTransformParameterFileName "s.txt")'
    with pytest.raises(ParserContentError, match="loop"):
        ElastixParameterTransform.from_file(_text(tmp_path, "s.txt", self_))


def test_addition_is_refused(tmp_path: Path) -> None:
    """HowToCombineTransforms "Add" is not a chain and is refused."""
    shutil.copy(FILES["euler3d"], tmp_path / "TransformParameters.0.txt")
    text = TRANSLATION + (
        '(InitialTransformParameterFileName "TransformParameters.0.txt")\n'
        '(HowToCombineTransforms "Add")'
    )
    path = _text(tmp_path, "TransformParameters.1.txt", text)
    with pytest.raises(NotImplementedError, match="Add"):
        ElastixParameterTransform.from_file(path)
    assert len(ElastixParameterTransform.from_file(path, initial=False)) == 1


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename", ["euler3d.txt", "bspline3d.txt", "euler3d.toml"]
)
def test_unmodified_files_round_trip(tmp_path: Path, filename: str) -> None:
    xform = io.load(data_dir / filename)
    out = tmp_path / filename
    xform.save(out)
    again = io.load(out)
    assert type(again) is type(xform)
    assert again.parameter_map.keys() == xform.parameter_map.keys()
    for key, value in xform.parameter_map.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(again.parameter_map[key], value)
        else:
            assert again.parameter_map[key] == value


def test_a_chain_is_written_with_its_initial_file_name() -> None:
    xform = io.load(FILES["chain"])
    text = xform.to_text()
    assert (
        '(InitialTransformParameterFileName "chain_TransformParameters.0.txt")'
        in text
    )


@pytest.mark.parametrize(
    "cls", [ElastixParameterTransform, ElastixTomlTransform]
)
def test_an_affine_is_written_centered_on_the_origin(
    tmp_path: Path, cls: type
) -> None:
    matrix = np.array(
        [[1.1, 0.2, 0.0, 2.0], [-0.1, 0.9, 0.1, -3.0], [0.0, 0.2, 1.0, 4.0]]
    )
    out = tmp_path / (
        "affine.toml" if cls is ElastixTomlTransform else "affine.txt"
    )
    cls([xforms.Affine(matrix)]).save(out)
    again = io.load(out)
    assert type(again) is cls
    assert again.parameter_map["Transform"] == ("AffineTransform",)
    assert again.parameter_map["CenterOfRotationPoint"] == (0, 0, 0)
    np.testing.assert_allclose(
        np.asarray(again.compute().to(xforms.Affine).matrix), matrix
    )


@pytest.mark.parametrize(
    "name",
    ["euler3d_zyx", "similarity3d", "translation3d", "bspline3d_order2"],
)
def test_a_block_is_written_as_its_elastix_transform(
    tmp_path: Path, name: str
) -> None:
    """A block keeps its class, center, grid and fixed geometry."""
    xform = io.load(FILES[name])
    block = xform[0]
    xform.transformations = [block]
    out = tmp_path / "out.txt"
    xform.save(out)
    again = io.load(out)
    assert again.parameter_map["Transform"] == xform.parameter_map["Transform"]
    assert type(again[0]) is type(block)
    np.testing.assert_allclose(again[0].parameters, block.parameters)
    np.testing.assert_allclose(
        again[0].fixed_parameters, block.fixed_parameters
    )
    np.testing.assert_allclose(
        np.asarray(again.fixed_geometry.transformation.matrix),
        np.asarray(xform.fixed_geometry.transformation.matrix),
    )
    # The fixed grid still maps where transformix maps it.
    expected = np.load(CASES[name])
    points = _grid_points(expected["vox2lps"], expected["disp"].shape[:-1])
    np.testing.assert_allclose(
        _apply(again, points), points + expected["disp"], atol=1e-5
    )


def test_an_affine_chain_is_written_as_one_affine(tmp_path: Path) -> None:
    xform = io.load(FILES["chain3"])
    expected = np.load(CASES["chain3"])
    xform.transformations = list(xform.transformations)
    out = tmp_path / "out.txt"
    xform.save(out)
    again = io.load(out)
    assert again.parameter_map["Transform"] == ("AffineTransform",)
    assert again.initial is None
    points = _grid_points(expected["vox2lps"], expected["disp"].shape[:-1])
    np.testing.assert_allclose(
        _apply(again, points), points + expected["disp"], atol=1e-5
    )


def test_unrepresentable_chains_are_refused(tmp_path: Path) -> None:
    xform = io.load(FILES["chain"])
    xform.transformations = list(xform.transformations)
    out = tmp_path / "out.txt"
    with pytest.raises(UnrepresentableTransformationError):
        xform.save(out)
    assert not out.exists()

    ras = xforms.Affine(
        np.eye(3, 4), input=systems.RASmm(), output=systems.RASmm()
    )
    with pytest.raises(UnrepresentableTransformationError, match="LPS"):
        ElastixParameterTransform([ras]).to_text()


def test_the_base_is_not_registered() -> None:
    """Only the text and TOML syntaxes take part in dispatch."""
    registry = io.transformations.FileBasedTransformation._REGISTRY
    assert ElastixParameterTransform in registry
    assert ElastixTomlTransform in registry
    assert ElastixTransform not in registry
