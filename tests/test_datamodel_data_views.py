"""Tests for the stored `data` of concrete transformations and their views.

Every concrete transformation stores one array, `data`, plus the flags
that say how it is encoded. Its named views (`field`, `matrix`, `scale`,
...) are always the map, as values. The views are checked under every
encoding, with cubic splines for the fields: at the default linear
degree, spline coefficients equal the values and would hide a view that
reads the stored array as values.
"""

import inspect

import numpy as np
import pytest
from bagof.magic import replace

from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._ext.invfield import inverse as inverse_disp
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    InverseCoordinatesField,
    InverseDisplacementField,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
    is_identity,
    is_translation,
)

DEGREE = 3
BOUND = "nearest"
FIELDS = (DisplacementField, CoordinatesField)


def _values(shape: tuple = (12, 13, 2), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _small(shape: tuple = (8, 9, 2), seed: int = 0) -> np.ndarray:
    # Small enough for the mesh inversion to be well behaved.
    return 0.1 * np.random.default_rng(seed).normal(size=shape)


def _coefficients(values: np.ndarray, **options) -> np.ndarray:
    options = {"degree": DEGREE, "bound": BOUND, **options}
    return np.asarray(value2coeff_field(values, **options))


def _grid(shape: tuple) -> np.ndarray:
    return np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1
    )


# The same matrix-family map, once per class: (class, view, values).
MATRIX_FAMILY = [
    (Affine, "matrix", np.array([[2.0, 0.5, 1.0], [0.0, 3.0, -2.0]])),
    (Linear, "matrix", np.array([[2.0, 0.5], [0.0, 3.0]])),
    (Rotation, "matrix", np.array([[0.0, -1.0], [1.0, 0.0]])),
    (Scaling, "scale", np.array([2.0, 3.0])),
    (Translation, "translation", np.array([1.0, -2.0])),
    (Permutation, "permutation", np.array([1, 0])),
]
MATRIX_IDS = [cls.__name__ for cls, _, _ in MATRIX_FAMILY]


# ----------------------------------------------------------------------
#   VIEWS ARE THE MAP, AS VALUES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("coeff", [False, True])
@pytest.mark.parametrize("build", ["data", "convert"])
def test_the_field_view_is_the_values(
    cls: type, coeff: bool, build: str
) -> None:
    values = _values()
    if build == "data":
        data = _coefficients(values) if coeff else values
        t = cls(data=data, degree=DEGREE, bound=BOUND, coeff=coeff)
    else:
        t = cls(field=values, degree=DEGREE, bound=BOUND).to(coeff=coeff)
    assert t.coeff is coeff
    np.testing.assert_allclose(np.asarray(t.field), values, atol=1e-10)
    if coeff:
        # Cubic coefficients are not the values: the view did decode.
        assert np.abs(np.asarray(t.data) - values).max() > 1e-3
    else:
        assert t.data is t.field


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_decoded_view_is_cached(cls: type) -> None:
    t = cls(data=_coefficients(_values()), degree=DEGREE, coeff=True)
    assert t.field is t.field


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_encoding_is_reached_by_conversion(cls: type) -> None:
    values = _values()
    t = cls(field=values, degree=DEGREE, bound=BOUND)
    np.testing.assert_allclose(
        np.asarray(t.to(coeff=True).data), _coefficients(values)
    )
    np.testing.assert_allclose(
        np.asarray(t.to(coeff=True).to(coeff=False).data), values, atol=1e-10
    )


@pytest.mark.parametrize("coeff", [False, True])
def test_the_grid_view_is_the_grid(coeff: bool) -> None:
    shape = (6, 7)
    t = CartesianField(shape=shape, degree=DEGREE, bound=BOUND, coeff=coeff)
    np.testing.assert_array_equal(np.asarray(t.field), _grid(shape))
    expected = _grid(shape).astype(float)
    if coeff:
        expected = _coefficients(expected)
    np.testing.assert_allclose(np.asarray(t.data), expected, atol=1e-10)


@pytest.mark.parametrize("cls, view, values", MATRIX_FAMILY, ids=MATRIX_IDS)
def test_the_matrix_family_views_are_the_values(
    cls: type, view: str, values: np.ndarray
) -> None:
    t = cls(values)
    np.testing.assert_array_equal(getattr(t, view), values)
    np.testing.assert_array_equal(t.data, values)


def test_the_homogeneous_matrix_is_read_from_the_matrix() -> None:
    matrix = np.array([[2.0, 0.5, 1.0], [0.0, 3.0, -2.0]])
    np.testing.assert_array_equal(
        Affine(data=matrix).homogeneous_matrix,
        np.vstack([matrix, [0.0, 0.0, 1.0]]),
    )


def test_the_identity_stores_nothing() -> None:
    assert Identity().data is None
    assert "data" not in inspect.signature(Identity).parameters


def test_an_unset_parameter_reads_as_none() -> None:
    for cls in FIELDS:
        assert cls(coeff=True, degree=DEGREE).field is None
    for cls, view, _ in MATRIX_FAMILY:
        assert getattr(cls(), view) is None


# ----------------------------------------------------------------------
#   INVERSES DERIVE THEIR DATA IN THE FORWARD'S ENCODING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("coeff", [False, True])
def test_a_displacement_inverse_keeps_the_encoding(coeff: bool) -> None:
    values = _small()
    forward = DisplacementField(field=values, degree=DEGREE).to(coeff=coeff)
    inverse = forward.inverse()
    assert isinstance(inverse, InverseDisplacementField)
    assert (inverse.coeff, inverse.degree, inverse.bound) == (
        forward.coeff,
        forward.degree,
        forward.bound,
    )
    expected = inverse_disp(np.asarray(forward.field))
    np.testing.assert_allclose(np.asarray(inverse.field), expected, atol=1e-8)
    if coeff:
        decoded = coeff2value_field(
            inverse.data, degree=inverse.degree, bound=inverse.bound
        )
        np.testing.assert_allclose(np.asarray(decoded), expected, atol=1e-8)
    computed = inverse.compute()
    assert type(computed) is DisplacementField
    assert computed.coeff is coeff
    np.testing.assert_allclose(np.asarray(computed.field), expected, atol=1e-8)


@pytest.mark.parametrize("coeff", [False, True])
def test_a_coordinates_inverse_keeps_the_encoding(coeff: bool) -> None:
    shape = (8, 9)
    coords = _grid(shape) + _small(shape + (2,))
    forward = CoordinatesField(field=coords, degree=DEGREE).to(coeff=coeff)
    inverse = forward.inverse()
    assert isinstance(inverse, InverseCoordinatesField)
    assert inverse.coeff is coeff
    grid = _grid(shape)
    expected = grid + inverse_disp(np.asarray(forward.field) - grid)
    np.testing.assert_allclose(np.asarray(inverse.field), expected, atol=1e-8)


def test_a_new_encoding_of_a_lazy_inverse_materializes_it() -> None:
    forward = DisplacementField(field=_small(), degree=DEGREE)
    inverse = forward.inverse()
    coefficients = inverse.to(coeff=True)
    assert type(coefficients) is DisplacementField
    assert coefficients.coeff is True
    np.testing.assert_allclose(
        np.asarray(coefficients.field), np.asarray(inverse.field), atol=1e-8
    )
    # An endpoint edit keeps the inverse lazy.
    system = CoordinateSystem(name="elsewhere")
    assert isinstance(inverse.to(input=system), InverseDisplacementField)


@pytest.mark.parametrize("cls, view, values", MATRIX_FAMILY, ids=MATRIX_IDS)
def test_a_matrix_family_inverse_derives_its_data(
    cls: type, view: str, values: np.ndarray
) -> None:
    if cls is Linear or cls is Affine:
        values = np.array(values, dtype=float)
        values[:, 1] += 0.25  # keep it invertible and not orthogonal
    inverse = cls(values).inverse()
    np.testing.assert_array_equal(getattr(inverse, view), inverse.data)
    assert "data" not in inspect.signature(type(inverse)).parameters
    assert view not in inspect.signature(type(inverse)).parameters


# ----------------------------------------------------------------------
#   CONSTRUCTORS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_field_values_are_stored_as_the_flags_say(cls: type) -> None:
    # The keyword is the map, as values; the flags describe its storage.
    values = _values()
    t = cls(field=values, degree=DEGREE, coeff=True)
    expected = cls(field=values, degree=DEGREE).to(coeff=True)
    assert (t.coeff, t.degree) == (True, DEGREE)
    np.testing.assert_array_equal(
        np.asarray(t.data), np.asarray(expected.data)
    )
    np.testing.assert_allclose(np.asarray(t.data), _coefficients(values))
    np.testing.assert_allclose(np.asarray(t.field), values, atol=1e-10)


@pytest.mark.parametrize(
    "cls, view, values",
    MATRIX_FAMILY + [(cls, "field", _values()) for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
def test_data_and_a_convenience_keyword_are_refused_together(
    cls: type, view: str, values: np.ndarray
) -> None:
    with pytest.raises(TypeError, match="both data= and"):
        cls(data=values, **{view: values})
    with pytest.raises(TypeError, match="both data= and"):
        cls(values, **{view: values})


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_field_and_data_build_the_same_field(cls: type) -> None:
    values = _values()
    a = cls(field=values, degree=DEGREE, bound=BOUND)
    b = cls(data=values, degree=DEGREE, bound=BOUND)
    c = cls(values, DEGREE, BOUND)
    for t in (a, b, c):
        assert t.data is values
        assert (t.degree, t.bound, t.coeff) == (DEGREE, BOUND, False)


@pytest.mark.parametrize("cls, view, values", MATRIX_FAMILY, ids=MATRIX_IDS)
def test_a_convenience_keyword_and_data_build_the_same_map(
    cls: type, view: str, values: np.ndarray
) -> None:
    for t in (cls(values), cls(data=values), cls(**{view: values})):
        assert type(t) is cls
        np.testing.assert_array_equal(t.data, values)
        np.testing.assert_array_equal(getattr(t, view), values)


@pytest.mark.parametrize(
    "cls, view",
    [(cls, view) for cls, view, _ in MATRIX_FAMILY]
    + [(cls, "field") for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
def test_data_is_first_and_the_convenience_is_keyword_only(
    cls: type, view: str
) -> None:
    parameters = list(inspect.signature(cls).parameters.values())
    assert parameters[0].name == "data"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    by_name = {p.name: p for p in parameters}
    assert by_name[view].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_grid_takes_neither_data_nor_field() -> None:
    parameters = inspect.signature(CartesianField).parameters
    assert "data" not in parameters
    assert "field" not in parameters
    with pytest.raises(TypeError):
        CartesianField(shape=(2, 3), field=np.zeros((2, 3, 2)))


@pytest.mark.parametrize(
    "cls, view, values",
    MATRIX_FAMILY + [(cls, "field", _values()) for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
def test_a_dictionary_names_the_map_or_its_data(
    cls: type, view: str, values: np.ndarray
) -> None:
    for key in (view, "data"):
        t = cls.from_dict({key: values})
        assert type(t) is cls
        np.testing.assert_array_equal(np.asarray(t.data), values)
        np.testing.assert_array_equal(np.asarray(getattr(t, view)), values)
        t = cls.from_other({key: values})
        np.testing.assert_array_equal(np.asarray(t.data), values)


def test_a_dictionary_of_field_values_is_stored_as_the_flags_say() -> None:
    values = _values()
    t = DisplacementField.from_dict(
        {"field": values, "degree": DEGREE, "coeff": True}
    )
    np.testing.assert_allclose(np.asarray(t.data), _coefficients(values))
    t = DisplacementField.from_dict(
        {"data": values, "degree": DEGREE, "coeff": True}
    )
    assert t.data is values


def test_an_unknown_key_names_the_convenience_keywords() -> None:
    with pytest.raises(TypeError, match="'matrix'") as error:
        Affine.from_other({"matrices": np.eye(3)[:2]})
    assert "'data'" in str(error.value)
    assert "'matrices'" in str(error.value)


def test_an_instance_is_read_through_its_data() -> None:
    t = DisplacementField(data=_coefficients(_values()), degree=3, coeff=True)
    copy = CoordinatesField.from_instance(t)
    assert copy.data is t.data
    assert (copy.coeff, copy.degree) == (True, 3)


@pytest.mark.parametrize(
    "cls, view, values",
    MATRIX_FAMILY + [(cls, "field", _values()) for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
def test_replace_with_a_convenience_keyword_points_to_to(
    cls: type, view: str, values: np.ndarray
) -> None:
    # `replace` carries `data` over, so the keyword meets it: the error
    # says why, and what to use instead.
    with pytest.raises(TypeError) as error:
        replace(cls(values), **{view: values})
    message = str(error.value)
    assert f"both data= and {view}=" in message
    assert "replace()" in message
    assert f"t.to({view}=...)" in message
    # On a transformation with no data yet there is nothing to meet.
    t = replace(cls(), **{view: values})
    np.testing.assert_array_equal(np.asarray(t.data), values)


# ----------------------------------------------------------------------
#   CONVERSION WITHIN A TYPE
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_field_is_stored_in_the_current_encoding(cls: type) -> None:
    t = cls(data=_coefficients(_values()), degree=DEGREE, coeff=True)
    values = _values(seed=1)
    u = t.to(field=values)
    assert u.coeff is True
    np.testing.assert_allclose(np.asarray(u.data), _coefficients(values))
    np.testing.assert_allclose(np.asarray(u.field), values, atol=1e-10)


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_new_data_is_stored_as_given(cls: type) -> None:
    t = cls(field=_values(), degree=DEGREE)
    data = _values(seed=2)
    u = t.to(data=data, coeff=True)
    assert u.data is data
    assert u.coeff is True


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_degree_refits_the_coefficients(cls: type) -> None:
    values = _values()
    t = cls(field=values, degree=DEGREE).to(coeff=True)
    u = t.to(degree=2)
    assert (u.degree, u.coeff) == (2, True)
    np.testing.assert_allclose(np.asarray(u.field), values, atol=1e-10)


def test_to_refuses_data_and_a_convenience_keyword_together() -> None:
    t = DisplacementField(field=_values())
    with pytest.raises(TypeError, match="both data= and field="):
        t.to(data=_values(), field=_values())


@pytest.mark.parametrize("cls, view, values", MATRIX_FAMILY, ids=MATRIX_IDS)
def test_a_convenience_keyword_overrides_the_map(
    cls: type, view: str, values: np.ndarray
) -> None:
    t = cls().to(**{view: values})
    np.testing.assert_array_equal(getattr(t, view), values)
    np.testing.assert_array_equal(t.to(data=values).data, values)


def test_a_linear_converts_to_a_rotation() -> None:
    matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotation = Linear(matrix).to(Rotation)
    assert type(rotation) is Rotation
    np.testing.assert_array_equal(rotation.matrix, matrix)


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("int64", "float32"),
        ("int32", "float32"),
        ("bool", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_the_coefficients_dtype(cls: type, dtype: str, expected: str) -> None:
    # Integer and boolean values are fitted in float32; floating values
    # keep their dtype.
    values = (_grid((7, 8)) % 2).astype(dtype)
    t = cls(field=values, degree=DEGREE).to(coeff=True)
    assert np.asarray(t.data).dtype == np.dtype(expected)
    np.testing.assert_allclose(
        np.asarray(t.field), values.astype(float), atol=1e-5
    )
    assert cls(field=values, degree=DEGREE, coeff=True).data.dtype == expected


@pytest.mark.parametrize("coeff", [False, True])
def test_the_grid_is_float64(coeff: bool) -> None:
    # A grid holds real coordinates: they, and their coefficients, are
    # float64, the default floating dtype of NumPy.
    t = CartesianField(shape=(7, 8), degree=DEGREE, coeff=coeff)
    assert np.asarray(t.field).dtype == np.float64
    assert np.asarray(t.data).dtype == np.float64


def test_an_unchanged_encoding_is_a_pass_through() -> None:
    t = DisplacementField(data=_values(), degree=DEGREE, coeff=True)
    assert t.to(coeff=True, degree=DEGREE, bound=BOUND) is t


# ----------------------------------------------------------------------
#   CHECKERS READ VALUES  (#294)
# ----------------------------------------------------------------------


def test_the_translation_check_reads_values() -> None:
    # Constant coefficients under a zero boundary are not a constant
    # field: the values fall off near the edges. On main, the check read
    # the coefficients as values, and called this a translation.
    data = np.ones((8, 9, 2))
    t = DisplacementField(
        data=data, degree=DEGREE, bound="constant", coeff=True
    )
    assert not is_translation(t, compute=True)
    assert is_translation(DisplacementField(field=data), compute=True)


def test_the_identity_check_reads_values() -> None:
    zeros = np.zeros((8, 9, 2))
    t = DisplacementField(data=zeros, degree=DEGREE, coeff=True)
    assert is_identity(t, compute=True)
    assert not is_identity(t, compute=False)
    assert is_identity(DisplacementField(coeff=True), compute=False)


# ----------------------------------------------------------------------
#   DISPLACEMENTS TO COORDINATES  (#294)
# ----------------------------------------------------------------------


def test_coordinates_from_coefficients_match_those_from_values() -> None:
    # The reproducer of #294: on main the two differed by 10.7, and the
    # result lost its encoding.
    u = np.random.default_rng(0).normal(size=(12, 13, 2))
    d = DisplacementField(field=u, degree=3)
    c = d.to(coeff=True)
    a = d.to(CoordinatesField)
    b = c.to(CoordinatesField)
    np.testing.assert_allclose(
        np.asarray(b.field), np.asarray(a.field), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(a.field), u + _grid((12, 13)))
    assert (a.coeff, a.degree, a.bound) == (False, d.degree, d.bound)
    assert (b.coeff, b.degree, b.bound) == (True, c.degree, c.bound)


@pytest.mark.parametrize("bound", ["nearest", "constant", "reflect"])
def test_coordinates_keep_the_encoding_of_the_displacements(
    bound: str,
) -> None:
    u = _values()
    c = DisplacementField(field=u, degree=DEGREE, bound=bound).to(coeff=True)
    b = c.to(CoordinatesField)
    assert type(b) is CoordinatesField
    assert (b.coeff, b.degree, b.bound) == (True, DEGREE, c.bound)
    np.testing.assert_allclose(
        np.asarray(b.field), u + _grid(u.shape[:-1]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(b.data),
        _coefficients(u + _grid(u.shape[:-1]), bound=bound),
        atol=1e-10,
    )


def test_an_unset_displacement_converts_to_unset_coordinates() -> None:
    d = DisplacementField(degree=DEGREE, coeff=True)
    c = d.to(CoordinatesField)
    assert c.data is None
    assert (c.coeff, c.degree) == (True, DEGREE)
