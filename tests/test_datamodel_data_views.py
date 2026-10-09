"""Tests of the stored `data` of concrete transformations and of its views.

The named views, such as `field` or `matrix`, always hold the map as
values, whatever the encoding of `data`. Fields use cubic splines, whose
coefficients differ from the values.
"""

import inspect

import numpy as np
import pytest
from bagof.magic import replace

from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops._ext.invfield import inverse as inverse_disp
from brainhops.datamodel._transformations import concrete as xconcrete
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
    # Small, so that the mesh inversion behaves well.
    return 0.1 * np.random.default_rng(seed).normal(size=shape)


def _coefficients(values: np.ndarray, **options) -> np.ndarray:
    options = {"degree": DEGREE, "bound": BOUND, **options}
    return np.asarray(value2coeff_field(values, **options))


def _grid(shape: tuple) -> np.ndarray:
    return np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1
    )


# One map per class of the matrix family, as (class, view, values).
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
@pytest.mark.parametrize("store", ["values", "coefficients"])
@pytest.mark.parametrize("build", ["data", "convert"])
def test_the_field_view_is_the_values(
    cls: type, store: str, build: str
) -> None:
    values = _values()
    if build == "data":
        data = _coefficients(values) if store == "coefficients" else values
        t = cls(data=data, degree=DEGREE, bound=BOUND, store=store)
    else:
        t = cls(field=values, degree=DEGREE, bound=BOUND).to(store=store)
    assert t.store == store
    np.testing.assert_allclose(np.asarray(t.field), values, atol=1e-10)
    if store == "coefficients":
        # Cubic coefficients differ from the values, so the view did decode
        # them.
        assert np.abs(np.asarray(t.data) - values).max() > 1e-3
    else:
        assert t.data is t.field


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_decoded_view_is_cached(cls: type) -> None:
    t = cls(data=_coefficients(_values()), degree=DEGREE, store="coefficients")
    assert t.field is t.field


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", ["data", "store", "degree", "bound"])
def test_assigning_data_or_a_flag_refreshes_the_view(
    cls: type, name: str
) -> None:
    # Assigning anything the cached view is decoded from clears the cache.
    coefficients = _coefficients(_values())
    t = cls(
        data=coefficients, degree=DEGREE, bound=BOUND, store="coefficients"
    )
    before = t.field
    new = {
        "data": 2 * coefficients,
        "store": "values",
        "degree": 2,
        "bound": "reflect",
    }[name]
    setattr(t, name, new)
    assert getattr(t, name) == new if name != "data" else t.data is new
    expected = coeff2value_field(
        np.asarray(t.data), degree=t.degree, bound=t.bound
    )
    if t.store == "values":
        expected = t.data
    assert t.field is not before
    np.testing.assert_allclose(np.asarray(t.field), expected, atol=1e-10)


def test_assigning_data_refreshes_a_cached_inverse() -> None:
    forward = DisplacementField(field=_small(), degree=DEGREE)
    inverse = forward.inverse()
    before = inverse.field
    forward.data = 2 * forward.data
    assert inverse.field is not before
    np.testing.assert_allclose(
        np.asarray(inverse.field),
        inverse_disp(np.asarray(forward.field)),
        atol=1e-8,
    )


@pytest.mark.parametrize("name", ["shape", "store"])
def test_assigning_the_shape_or_a_flag_refreshes_a_grid(name: str) -> None:
    t = CartesianField(shape=(6, 7), degree=DEGREE, bound=BOUND)
    before = t.field, t.data
    setattr(t, name, {"shape": (4, 5), "store": "coefficients"}[name])
    assert t.data is not before[1]
    np.testing.assert_array_equal(np.asarray(t.field), _grid(t.shape))
    expected = _grid(t.shape).astype(float)
    if t.store == "coefficients":
        expected = _coefficients(expected)
    np.testing.assert_allclose(np.asarray(t.data), expected, atol=1e-10)


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_encoding_is_reached_by_conversion(cls: type) -> None:
    values = _values()
    t = cls(field=values, degree=DEGREE, bound=BOUND)
    np.testing.assert_allclose(
        np.asarray(t.to(store="coefficients").data), _coefficients(values)
    )
    np.testing.assert_allclose(
        np.asarray(t.to(store="coefficients").to(store="values").data),
        values,
        atol=1e-10,
    )


@pytest.mark.parametrize("store", ["values", "coefficients"])
def test_the_grid_view_is_the_grid(store: str) -> None:
    shape = (6, 7)
    t = CartesianField(shape=shape, degree=DEGREE, bound=BOUND, store=store)
    np.testing.assert_array_equal(np.asarray(t.field), _grid(shape))
    expected = _grid(shape).astype(float)
    if store == "coefficients":
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
        assert cls(store="coefficients", degree=DEGREE).field is None
    for cls, view, _ in MATRIX_FAMILY:
        assert getattr(cls(), view) is None


# ----------------------------------------------------------------------
#   INVERSES DERIVE THEIR DATA IN THE FORWARD'S ENCODING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("store", ["values", "coefficients"])
def test_a_displacement_inverse_keeps_the_encoding(store: str) -> None:
    values = _small()
    forward = DisplacementField(field=values, degree=DEGREE).to(store=store)
    inverse = forward.inverse()
    assert isinstance(inverse, InverseDisplacementField)
    assert (inverse.store, inverse.degree, inverse.bound) == (
        forward.store,
        forward.degree,
        forward.bound,
    )
    expected = inverse_disp(np.asarray(forward.field))
    np.testing.assert_allclose(np.asarray(inverse.field), expected, atol=1e-8)
    if store == "coefficients":
        decoded = coeff2value_field(
            inverse.data, degree=inverse.degree, bound=inverse.bound
        )
        np.testing.assert_allclose(np.asarray(decoded), expected, atol=1e-8)
    computed = inverse.compute()
    assert type(computed) is DisplacementField
    assert computed.store == store
    np.testing.assert_allclose(np.asarray(computed.field), expected, atol=1e-8)


@pytest.mark.parametrize("store", ["values", "coefficients"])
def test_a_coordinates_inverse_keeps_the_encoding(store: str) -> None:
    shape = (8, 9)
    coords = _grid(shape) + _small(shape + (2,))
    forward = CoordinatesField(field=coords, degree=DEGREE).to(store=store)
    inverse = forward.inverse()
    assert isinstance(inverse, InverseCoordinatesField)
    assert inverse.store == store
    grid = _grid(shape)
    expected = grid + inverse_disp(np.asarray(forward.field) - grid)
    np.testing.assert_allclose(np.asarray(inverse.field), expected, atol=1e-8)


@pytest.mark.parametrize(
    "start, change",
    [
        (dict(store="values"), dict(store="coefficients")),
        (dict(store="coefficients"), dict(store="values")),
        (dict(store="coefficients"), dict(degree=1)),
    ],
    ids=["encode", "decode", "refit"],
)
def test_a_new_encoding_of_a_lazy_inverse_is_made_to_its_forward(
    start: dict, change: dict
) -> None:
    # The new encoding goes to the forward, and the inverse stays lazy but
    #
    # matches the inverse materialized first.
    forward = DisplacementField(field=_small(), degree=DEGREE).to(**start)
    inverse = forward.inverse()
    lazy = inverse.to(**change)
    assert isinstance(lazy, InverseDisplacementField)
    for name, value in change.items():
        assert getattr(lazy.forward, name) == value
        assert getattr(lazy, name) == value
    eager = inverse.compute().to(**change)
    np.testing.assert_allclose(
        np.asarray(lazy.field), np.asarray(eager.field), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(lazy.data), np.asarray(eager.data), atol=1e-12
    )
    # Changing an endpoint keeps the inverse lazy and its forward unchanged.
    system = CoordinateSystem(name="elsewhere")
    moved = inverse.to(input=system)
    assert isinstance(moved, InverseDisplacementField)
    assert moved.forward is forward


@pytest.mark.parametrize(
    "cls, view, values",
    MATRIX_FAMILY + [(cls, "field", _small()) for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
@pytest.mark.parametrize("keyword", ["data", "view"])
def test_the_map_of_a_lazy_inverse_cannot_be_set(
    cls: type, view: str, values: np.ndarray, keyword: str
) -> None:
    # The map of an inverse derives from its forward and cannot be set.
    if cls is CoordinatesField:
        values = values + _grid(values.shape[:-1])
    inverse = cls(values).inverse()
    name = view if keyword == "view" else "data"
    with pytest.raises(TypeError):
        inverse.to(**{name: values})


@pytest.mark.parametrize("cls, view, values", MATRIX_FAMILY, ids=MATRIX_IDS)
def test_a_matrix_family_inverse_derives_its_data(
    cls: type, view: str, values: np.ndarray
) -> None:
    if cls is Linear or cls is Affine:
        values = np.array(values, dtype=float)
        values[:, 1] += 0.25  # Keep the matrix invertible and non-orthogonal.
    inverse = cls(values).inverse()
    np.testing.assert_array_equal(getattr(inverse, view), inverse.data)
    assert "data" not in inspect.signature(type(inverse)).parameters
    assert view not in inspect.signature(type(inverse)).parameters


# ----------------------------------------------------------------------
#   CONSTRUCTORS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_field_values_are_stored_as_the_flags_say(cls: type) -> None:
    # The keyword gives the map as values; the flags describe the storage.
    values = _values()
    t = cls(field=values, degree=DEGREE, store="coefficients")
    expected = cls(field=values, degree=DEGREE).to(store="coefficients")
    assert (t.store, t.degree) == ("coefficients", DEGREE)
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
    with pytest.raises(TypeError):
        cls(data=values, **{view: values})
    with pytest.raises(TypeError):
        cls(values, **{view: values})


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_field_and_data_build_the_same_field(cls: type) -> None:
    values = _values()
    a = cls(field=values, degree=DEGREE, bound=BOUND)
    b = cls(data=values, degree=DEGREE, bound=BOUND)
    # `data` is the only positional parameter.
    c = cls(values, degree=DEGREE, bound=BOUND)
    for t in (a, b, c):
        assert t.data is values
        assert (t.degree, t.bound, t.store) == (DEGREE, BOUND, "values")


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
        t = cls.from_any({key: values})
        np.testing.assert_array_equal(np.asarray(t.data), values)


def test_a_dictionary_of_field_values_is_stored_as_the_flags_say() -> None:
    values = _values()
    t = DisplacementField.from_dict(
        {"field": values, "degree": DEGREE, "store": "coefficients"}
    )
    np.testing.assert_allclose(np.asarray(t.data), _coefficients(values))
    t = DisplacementField.from_dict(
        {"data": values, "degree": DEGREE, "store": "coefficients"}
    )
    assert t.data is values


def test_an_unknown_key_names_the_convenience_keywords() -> None:
    with pytest.raises(TypeError, match="'matrix'") as error:
        Affine.from_any({"matrices": np.eye(3)[:2]})
    assert "'data'" in str(error.value)
    assert "'matrices'" in str(error.value)


def test_an_instance_is_read_through_its_data() -> None:
    t = DisplacementField(
        data=_coefficients(_values()), degree=3, store="coefficients"
    )
    copy = CoordinatesField.from_instance(t)
    assert copy.data is t.data
    assert (copy.store, copy.degree) == ("coefficients", 3)


@pytest.mark.parametrize(
    "cls, view, values",
    MATRIX_FAMILY + [(cls, "field", _values()) for cls in FIELDS],
    ids=MATRIX_IDS + [cls.__name__ for cls in FIELDS],
)
def test_replace_with_a_convenience_keyword_meets_the_data(
    cls: type, view: str, values: np.ndarray
) -> None:
    # `replace` carries `data` over, so the keyword meets it and both are
    #
    # refused.
    with pytest.raises(TypeError):
        replace(cls(values), **{view: values})
    # Without data, there is nothing to meet.
    t = replace(cls(), **{view: values})
    np.testing.assert_array_equal(np.asarray(t.data), values)


# ----------------------------------------------------------------------
#   CONVERSION WITHIN A TYPE
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_field_is_stored_in_the_current_encoding(cls: type) -> None:
    t = cls(data=_coefficients(_values()), degree=DEGREE, store="coefficients")
    values = _values(seed=1)
    u = t.to(field=values)
    assert u.store == "coefficients"
    np.testing.assert_allclose(np.asarray(u.data), _coefficients(values))
    np.testing.assert_allclose(np.asarray(u.field), values, atol=1e-10)


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_new_data_is_stored_as_given(cls: type) -> None:
    t = cls(field=_values(), degree=DEGREE)
    data = _values(seed=2)
    u = t.to(data=data, store="coefficients")
    assert u.data is data
    assert u.store == "coefficients"


@pytest.mark.parametrize("cls", FIELDS, ids=lambda c: c.__name__)
def test_a_new_degree_refits_the_coefficients(cls: type) -> None:
    values = _values()
    t = cls(field=values, degree=DEGREE).to(store="coefficients")
    u = t.to(degree=2)
    assert (u.degree, u.store) == (2, "coefficients")
    np.testing.assert_allclose(np.asarray(u.field), values, atol=1e-10)


def test_to_refuses_data_and_a_convenience_keyword_together() -> None:
    t = DisplacementField(field=_values())
    with pytest.raises(TypeError):
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
    # Integer and boolean values are fitted in float32.
    values = (_grid((7, 8)) % 2).astype(dtype)
    t = cls(field=values, degree=DEGREE).to(store="coefficients")
    assert np.asarray(t.data).dtype == np.dtype(expected)
    np.testing.assert_allclose(
        np.asarray(t.field), values.astype(float), atol=1e-5
    )
    stored = cls(field=values, degree=DEGREE, store="coefficients")
    assert stored.data.dtype == expected


@pytest.mark.parametrize("store", ["values", "coefficients"])
def test_the_grid_is_float64(store: str) -> None:
    # The grid holds real coordinates, stored as float64.
    t = CartesianField(shape=(7, 8), degree=DEGREE, store=store)
    assert np.asarray(t.field).dtype == np.float64
    assert np.asarray(t.data).dtype == np.float64


def test_an_unchanged_encoding_is_a_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asking for the current encoding returns the same stored array.
    t = DisplacementField(data=_values(), degree=DEGREE, store="coefficients")

    def refuse(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("the stored coefficients were refitted")

    monkeypatch.setattr(xconcrete, "value2coeff_field", refuse)
    monkeypatch.setattr(xconcrete, "coeff2value_field", refuse)
    encoding = dict(store="coefficients", degree=DEGREE, bound=BOUND)
    assert t.to(**encoding).data is t.data


# ----------------------------------------------------------------------
#   CHECKERS READ VALUES  (#294)
# ----------------------------------------------------------------------


def test_the_translation_check_reads_values() -> None:
    # Constant coefficients under a zero boundary do not make a constant
    #
    # field.
    data = np.ones((8, 9, 2))
    t = DisplacementField(
        data=data, degree=DEGREE, bound="constant", store="coefficients"
    )
    assert not is_translation(t, compute=True)
    assert is_translation(DisplacementField(field=data), compute=True)


def test_the_identity_check_reads_values() -> None:
    zeros = np.zeros((8, 9, 2))
    t = DisplacementField(data=zeros, degree=DEGREE, store="coefficients")
    assert is_identity(t, compute=True)
    assert not is_identity(t, compute=False)
    velocity = DisplacementField(store="coefficients")
    assert is_identity(velocity, compute=False)


# ----------------------------------------------------------------------
#   DISPLACEMENTS TO COORDINATES  (#294)
# ----------------------------------------------------------------------


def test_coordinates_from_coefficients_match_those_from_values() -> None:
    # Regression (#294): the two results differed and the encoding was lost.
    u = np.random.default_rng(0).normal(size=(12, 13, 2))
    d = DisplacementField(field=u, degree=3)
    c = d.to(store="coefficients")
    a = d.to(CoordinatesField)
    b = c.to(CoordinatesField)
    np.testing.assert_allclose(
        np.asarray(b.field), np.asarray(a.field), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(a.field), u + _grid((12, 13)))
    assert (a.store, a.degree, a.bound) == ("values", d.degree, d.bound)
    assert (b.store, b.degree, b.bound) == ("coefficients", c.degree, c.bound)


@pytest.mark.parametrize("bound", ["nearest", "constant", "reflect"])
def test_coordinates_keep_the_encoding_of_the_displacements(
    bound: str,
) -> None:
    u = _values()
    c = DisplacementField(field=u, degree=DEGREE, bound=bound).to(
        store="coefficients"
    )
    b = c.to(CoordinatesField)
    assert type(b) is CoordinatesField
    assert (b.store, b.degree, b.bound) == ("coefficients", DEGREE, c.bound)
    np.testing.assert_allclose(
        np.asarray(b.field), u + _grid(u.shape[:-1]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(b.data),
        _coefficients(u + _grid(u.shape[:-1]), bound=bound),
        atol=1e-10,
    )


def test_an_unset_displacement_converts_to_unset_coordinates() -> None:
    d = DisplacementField(degree=DEGREE, store="coefficients")
    c = d.to(CoordinatesField)
    assert c.data is None
    assert (c.store, c.degree) == ("coefficients", DEGREE)
