"""
The dimensions of a Neuroglancer coordinate transform, and their units.

A Neuroglancer coordinate transform names the dimensions of its input and
output spaces, and gives each one a scale and a unit, as in
`{"x": [4e-09, "m"]}`. These helpers check the dimensions as Neuroglancer
reads them, decode them into the coordinate systems of the data model,
encode the systems of the data model back into dimensions, and convert the
matrix between the two.

The conventions, and the source of Neuroglancer that they were read
in, are given in the documentation of the package,
[`brainhops.io.transformations.neuroglancer`][].
"""

# stdlib
import math
import re
from decimal import Decimal

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems
from brainhops.datamodel.units import Unit, is_indexunit
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
)

KEYS = ("matrix", "outputDimensions", "inputDimensions", "sourceRank")
"""The keys that Neuroglancer reads in a coordinate transform."""

BASES = {"m": "meter", "s": "second", "Hz": "hertz", "rad/s": "radian/second"}
"""The units of Neuroglancer, without prefix, with their names in pint."""

PREFIXES = {
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "h": 2,
    "da": 1,
    "": 0,
    "d": -1,
    "c": -2,
    "m": -3,
    "µ": -6,
    "u": -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
}
"""The SI prefixes that Neuroglancer accepts, with their exponents."""

DISPLAYED = {
    24: "Y",
    21: "Z",
    18: "E",
    15: "P",
    12: "T",
    9: "G",
    6: "M",
    3: "k",
    0: "",
    -3: "m",
    -6: "µ",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
    -21: "z",
    -24: "y",
}
"""The prefixes that Neuroglancer displays, by exponent."""

_NAME = re.compile(r"^[a-zA-Z][a-zA-Z_0-9]*['^]?$")


class Dimension(tx.NamedTuple):
    """A dimension of a Neuroglancer coordinate space, as it is decoded."""

    name: str
    """The name of the dimension."""

    unit: str
    """The unit without prefix: `"m"`, `"s"`, `"Hz"`, `"rad/s"` or `""`."""

    scale: Decimal
    """The scale in that unit, as an exact decimal."""

    @property
    def exponent(self) -> int:
        """The exponent of the prefix of the axis that decodes the dimension.

        It is 0 for a dimension without a unit. Otherwise, it is the
        exponent of the prefix that Neuroglancer displays for the scale,
        the largest multiple of 3 that does not exceed it, between -24 and
        24.
        """
        if not self.unit:
            return 0
        exponent = 3 * math.floor(self.scale.adjusted() / 3)
        return min(max(exponent, -24), 24)

    @property
    def factor(self) -> Decimal:
        """The scale in the unit of the axis that decodes the dimension."""
        return self.scale.scaleb(-self.exponent)

    def axis(self, named: bool = True) -> _axes.Axis:
        """Return the axis of the data model that decodes the dimension.

        Parameters
        ----------
        named : bool, default=True
            Whether the axis takes the name of the dimension.

        Returns
        -------
        Axis
            The axis, with the unit of the dimension and the prefix of
            its scale, or without a unit.
        """
        unit = None
        if self.unit:
            unit = Unit(DISPLAYED[self.exponent] + self.unit)
        return _axes.Axis(name=self.name if named else None, unit=unit)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def check_content(content: tx.Any) -> None:
    """Check a coordinate transform as Neuroglancer reads it.

    Unknown keys are accepted, since Neuroglancer ignores them. The
    output dimensions are required, because a transform of rank 0 maps
    nothing.

    Raises
    ------
    ParserContentError
        If the content is not a coordinate transform.
    """
    if not isinstance(content, dict):
        raise ParserContentError(
            f"A Neuroglancer coordinate transform is a JSON object, not a "
            f"{type(content).__name__}."
        )
    if "outputDimensions" not in content:
        raise ParserContentError(
            "A Neuroglancer coordinate transform has outputDimensions."
        )
    rank = len(dimensions(content["outputDimensions"], "outputDimensions"))
    if rank == 0:
        raise ParserContentError(
            "A Neuroglancer coordinate transform has at least one output "
            "dimension."
        )
    if "inputDimensions" in content:
        found = len(dimensions(content["inputDimensions"], "inputDimensions"))
        if found != rank:
            raise ParserContentError(
                f"The inputDimensions of a Neuroglancer coordinate "
                f"transform are as many as its {rank} outputDimensions, "
                f"but there are {found}."
            )
    if "sourceRank" in content:
        source_rank = content["sourceRank"]
        if (
            isinstance(source_rank, bool)
            or not isinstance(source_rank, int)
            or not 0 <= source_rank <= rank
        ):
            raise ParserContentError(
                f"The sourceRank of a Neuroglancer coordinate transform is "
                f"an integer between 0 and {rank}, not {source_rank!r}."
            )
    if "matrix" in content:
        _check_matrix(content["matrix"], rank)


def dimensions(value: tx.Any, key: str) -> tx.Tuple[Dimension, ...]:
    """Decode the dimensions of a coordinate space, in their order.

    Parameters
    ----------
    value : Any
        The value of `outputDimensions` or `inputDimensions`.
    key : str
        The key that holds the value, which decides whether a dimension
        may be named by an integer.

    Returns
    -------
    tuple of Dimension
        The dimensions.

    Raises
    ------
    ParserContentError
        If the value is not a coordinate space.
    """
    if not isinstance(value, dict):
        raise ParserContentError(
            f"The {key} of a Neuroglancer coordinate transform are a JSON "
            f"object, not a {type(value).__name__}."
        )
    numeric = key == "inputDimensions"
    return tuple(
        _dimension(name, item, key, numeric) for name, item in value.items()
    )


def is_content(content: tx.Any) -> bool:
    """Tell whether parsed JSON is a coordinate transform and nothing else.

    The keys must be among the four that Neuroglancer reads, and the
    content must be a valid transform. Only such content is recognized
    by its content alone.

    Parameters
    ----------
    content : Any
        The parsed JSON.

    Returns
    -------
    bool
        Whether the content is a coordinate transform.
    """
    if not isinstance(content, dict) or not set(content) <= set(KEYS):
        return False
    try:
        check_content(content)
    except ParserContentError:
        return False
    return True


def decode_matrix(content: tx.Dict[str, tx.Any]) -> np.ndarray:
    """Return the matrix of the data model that a transform stores.

    The matrix maps the coordinates of the input axes to those of the
    output axes, in the units of [`Dimension.axis`][].

    Parameters
    ----------
    content : dict
        The checked content of the transform.

    Returns
    -------
    array, shape `(rank, rank + 1)`
        The matrix.
    """
    outputs, inputs = _spaces(content)
    rank = len(outputs)
    stored = content.get("matrix")
    matrix = np.zeros((rank, rank + 1))
    for row in range(rank):
        output = outputs[row]
        for column in range(rank + 1):
            if stored is None:
                value = 1.0 if row == column else 0.0
            else:
                value = stored[row][column]
            if column < rank:
                shift = inputs[column].exponent - output.exponent
                matrix[row, column] = _times(value, Decimal(1).scaleb(shift))
            else:
                matrix[row, column] = _times(value, output.factor)
    return matrix


def decode_system(
    content: tx.Dict[str, tx.Any], key: str
) -> _systems.CoordinateSystem:
    """Return the coordinate system of the data model for one side.

    A transform without `inputDimensions` has an input whose axes have
    no name and the units of the output, as described in the module.

    Parameters
    ----------
    content : dict
        The checked content of the transform.
    key : str
        `"outputDimensions"` or `"inputDimensions"`.

    Returns
    -------
    CoordinateSystem
        The system, with one axis for each dimension.
    """
    if key in content:
        axes = [item.axis() for item in dimensions(content[key], key)]
    else:
        stored = dimensions(content["outputDimensions"], "outputDimensions")
        axes = [item.axis(named=False) for item in stored]
    return _systems.CoordinateSystem(axes=axes)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def encode_dimensions(
    system: tx.Optional[_systems.CoordinateSystem], key: str, rank: int
) -> tx.Dict[str, tx.List[tx.Any]]:
    """Encode a coordinate system of the data model as dimensions.

    Each axis is written with a scale of one unit, so `millimeter` is
    written as `[0.001, "m"]`, which [`Dimension.axis`][] decodes as
    `millimeter` again. An axis without a unit is written without a
    unit.

    Parameters
    ----------
    system : CoordinateSystem or None
        The system.
    key : str
        `"outputDimensions"` or `"inputDimensions"`, which decides
        whether an axis may be named by an integer.
    rank : int
        The rank of the matrix.

    Returns
    -------
    dict
        The dimensions, in the order of the axes.

    Raises
    ------
    UnrepresentableTransformationError
        If the system is not known, if it does not have one named axis
        for each row of the matrix, or if a unit has no Neuroglancer
        equivalent.
    """
    side = "input" if key == "inputDimensions" else "output"
    if system is None or system.ndim is None:
        raise UnrepresentableTransformationError(
            f"A Neuroglancer coordinate transform names the dimensions of "
            f"its {side}, but the {side} of the affine is not a known "
            f"system of {rank} axes. Give the affine an {side} system "
            f"whose axes have names and units."
        )
    if system.ndim != rank:
        raise UnrepresentableTransformationError(
            f"The {side} of the affine has {system.ndim} axes, but its "
            f"matrix has {rank} rows, and a Neuroglancer coordinate "
            f"transform has as many dimensions on each side."
        )
    encoded = {}
    for axis in system.axes:
        name = axis.name
        if not isinstance(name, str) or not _valid_name(name, side):
            raise UnrepresentableTransformationError(
                f"The {side} axis {name!r} has no name that Neuroglancer "
                f"accepts. A name starts with a letter, continues with "
                f"letters, digits and underscores, and may end with ' or "
                f"^."
            )
        if name in encoded:
            raise UnrepresentableTransformationError(
                f"Two {side} axes are named {name!r}, and Neuroglancer "
                f"needs distinct names."
            )
        encoded[name] = _scale_and_unit(axis.unit, name, side)
    return encoded


def encode_matrix(
    matrix: np.ndarray,
    content: tx.Dict[str, tx.Any],
) -> tx.List[tx.List[float]]:
    """Return the rows that a transform stores for a matrix of the data model.

    This is the inverse of [`decode_matrix`][].

    Parameters
    ----------
    matrix : array, shape `(rank, rank + 1)`
        The matrix of the data model.
    content : dict
        The content that holds the dimensions of the transform.

    Returns
    -------
    list of list of float
        The rows of the `matrix` key.
    """
    outputs, inputs = _spaces(content)
    rank = len(outputs)
    rows = []
    for row in range(rank):
        output = outputs[row]
        values = []
        for column in range(rank + 1):
            value = matrix[row, column]
            if column < rank:
                shift = output.exponent - inputs[column].exponent
                values.append(_times(value, Decimal(1).scaleb(shift)))
            else:
                values.append(_divided(value, output.factor))
        rows.append(values)
    return rows


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _dimension(name: tx.Any, item: tx.Any, key: str, numeric: bool) -> Dimension:
    # One dimension, as `coordinateSpaceFromJson` reads it.
    if not _valid_name(name, "input" if numeric else "output"):
        raise ParserContentError(
            f"{name!r} is not a valid name of a dimension in the {key} of "
            f"a Neuroglancer coordinate transform."
        )
    if isinstance(item, dict):
        coordinates = item.get("coordinates")
        labels = item.get("labels")
        if (
            not isinstance(coordinates, list)
            or not isinstance(labels, list)
            or len(coordinates) != len(labels)
            or not all(_is_integer(c) for c in coordinates)
            or not all(isinstance(label, str) for label in labels)
        ):
            raise ParserContentError(
                f"The dimension {name!r} of the {key} is a coordinate "
                f"array, which needs as many integer coordinates as string "
                f"labels."
            )
        return Dimension(name=name, unit="", scale=Decimal(1))
    if not isinstance(item, list) or len(item) != 2:
        raise ParserContentError(
            f"The dimension {name!r} of the {key} is [scale, unit] or a "
            f"coordinate array, not {item!r}."
        )
    scale, unit = item
    if not _is_number(scale) or not math.isfinite(scale) or scale <= 0:
        raise ParserContentError(
            f"The scale of the dimension {name!r} of the {key} is a finite "
            f"positive number, not {scale!r}."
        )
    if unit == "":
        return Dimension(name=name, unit="", scale=_decimal(scale))
    for prefix, exponent in PREFIXES.items():
        base = unit[len(prefix) :] if unit.startswith(prefix) else None
        if base in BASES:
            scale = _decimal(scale).scaleb(exponent)
            return Dimension(name=name, unit=base, scale=scale)
    raise ParserContentError(
        f"The unit {unit!r} of the dimension {name!r} of the {key} is not a "
        f"unit of Neuroglancer, which are m, s, Hz and rad/s with an SI "
        f"prefix, or no unit."
    )


def _check_matrix(matrix: tx.Any, rank: int) -> None:
    # `rank` rows of `rank + 1` finite numbers.
    rows_ok = isinstance(matrix, list) and len(matrix) == rank
    if rows_ok:
        for row in matrix:
            if not isinstance(row, list) or len(row) != rank + 1:
                rows_ok = False
                break
            if not all(_is_number(v) and math.isfinite(v) for v in row):
                rows_ok = False
                break
    if not rows_ok:
        raise ParserContentError(
            f"The matrix of a Neuroglancer coordinate transform of rank "
            f"{rank} is a list of {rank} rows of {rank + 1} finite numbers."
        )


def _spaces(
    content: tx.Dict[str, tx.Any],
) -> tx.Tuple[tx.Tuple[Dimension, ...], tx.Tuple[Dimension, ...]]:
    # The output and input dimensions, the input taking the units of the
    # output when it is not given.
    outputs = dimensions(content["outputDimensions"], "outputDimensions")
    if "inputDimensions" in content:
        inputs = dimensions(content["inputDimensions"], "inputDimensions")
    else:
        inputs = outputs
    return outputs, inputs


def _scale_and_unit(
    unit: tx.Optional[Unit], name: str, side: str
) -> tx.List[tx.Any]:
    # The `[scale, unit]` of one unit of an axis, or the reason why
    # Neuroglancer has no such unit.
    if unit is None:
        return [1, ""]
    if is_indexunit(unit):
        raise UnrepresentableTransformationError(
            f"The {side} axis {name!r} counts voxels. A Neuroglancer "
            f"transform maps the coordinate space of its data source, "
            f"where the centre of a voxel is at a half integer and the "
            f"scale of a voxel is set by the source, so an affine on "
            f"voxels is not encoded. Compose it with the voxel-to-world "
            f"affine of the image first."
        )
    quantity = 1 * unit.to_pint()
    radian = "radian" in str(unit)
    for base, long_name in BASES.items():
        if (base == "rad/s") != radian:
            continue
        try:
            factor = quantity.to(long_name).magnitude
        except Exception:
            continue
        if factor <= 0:
            continue
        exponent = round(math.log10(factor))
        if exponent in PREFIXES.values() and math.isclose(
            factor, 10.0**exponent, rel_tol=1e-12
        ):
            return [_power_of_ten(exponent), base]
    raise UnrepresentableTransformationError(
        f"The unit {unit} of the {side} axis {name!r} has no Neuroglancer "
        f"equivalent. Neuroglancer has m, s, Hz and rad/s, with an SI "
        f"prefix, and no unit."
    )


def _valid_name(name: tx.Any, side: str) -> bool:
    # `isValidDimensionName`, which takes integers for input dimensions.
    if not isinstance(name, str):
        return False
    if side == "input" and name.isdigit() and name.isascii():
        return True
    return _NAME.match(name) is not None


def _is_number(value: tx.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: tx.Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decimal(value: tx.Union[int, float]) -> Decimal:
    # The exact decimal that the shortest representation of a number
    # spells, which is what the JSON text spelled.
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(value)


def _times(value: tx.Any, factor: Decimal) -> float:
    if factor == 1:
        return float(value)
    return float(_decimal(float(value)) * factor)


def _divided(value: tx.Any, factor: Decimal) -> float:
    if factor == 1:
        return float(value)
    return float(_decimal(float(value)) / factor)


def _power_of_ten(exponent: int) -> tx.Union[int, float]:
    # The scale of one prefixed unit, written as Neuroglancer writes it.
    return 1 if exponent == 0 else float(f"1e{exponent}")
