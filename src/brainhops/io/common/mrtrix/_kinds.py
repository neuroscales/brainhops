"""What the values along the fourth axis of an MRtrix image are.

An MRtrix image may hold, along its fourth axis, the components of a
vector or of a tensor at each voxel: the six coefficients of a diffusion
tensor that `dwi2tensor` writes, or the three components of a warp that
`warpconvert` writes. MRtrix expresses these components in scanner RAS
millimetres, the world of the `transform` of the header, which does not
depend on how the voxel axes are ordered or stored. The header has no
key that says what the volumes are, so this module reads it from two
clues, in this order.

1. brainhops records the kind of the values in the header key
   `brainhops_field_type`, which MRtrix keeps without reading it, and
   appends its own line to `command_history`. The key decides while
   brainhops wrote the file last, which the last line of
   `command_history` says. MRtrix copies the key into the images that
   its commands derive from the file, and such an image may hold other
   values under the same key, so the key alone decides nothing.
2. The last line of `command_history` names the command that wrote the
   file, and a few commands write one kind of values only, which the
   shape of the image confirms. These are listed in
   [`history_kind`][].

When neither clue decides, the axis stays a channel axis. The values of
the image are never read to decide.

The values of the key are these:

| Value           | Values along the fourth axis              |
| --------------- | ----------------------------------------- |
| `coordinates`   | scanner positions (a deformation)         |
| `displacements` | scanner displacements                     |
| `vectors`       | vectors of valence `"vector"`             |
| `covectors`     | vectors of valence `"covector"`           |
| `tensors`       | symmetric tensors in the order of MRtrix  |

The reader also accepts `coord`, `deformation`, `displacement` and
`disp`, as the readers of MRtrix warps do.
"""

# stdlib
import os

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.enums import TensorLayout, Valence

from ._raw import MrtrixRaw

HISTORY = "command_history"
"""The header key in which MRtrix records the commands that wrote a file."""

FIELD_TYPE = "brainhops_field_type"
"""The header key in which brainhops records what the volumes hold.

MRtrix keeps a header key that it does not know as free-form text
(`core/formats/mrtrix_utils.h`), and copies it into the images that its
commands derive from the file (`Header::create`, in `core/header.cpp`).
"""

BRAINHOPS = "brainhops"
"""The command name that brainhops writes in `command_history`."""

DEFORMATION = "deformation"
"""The kind of a field of scanner positions."""

DISPLACEMENT = "displacement"
"""The kind of a field of scanner displacements."""

VECTOR = "vector"
"""The kind of a field of vectors of valence `"vector"`."""

COVECTOR = "covector"
"""The kind of a field of vectors of valence `"covector"`."""

TENSOR = "tensor"
"""The kind of a field of symmetric tensors in the order of MRtrix."""

FIELD_TYPE_VALUES = {
    DEFORMATION: "coordinates",
    DISPLACEMENT: "displacements",
    VECTOR: "vectors",
    COVECTOR: "covectors",
    TENSOR: "tensors",
}
"""The value of `brainhops_field_type` that brainhops writes for a kind."""

FIELD_TYPE_KINDS = {
    "coordinates": DEFORMATION,
    "coord": DEFORMATION,
    "deformation": DEFORMATION,
    "displacement": DISPLACEMENT,
    "displacements": DISPLACEMENT,
    "disp": DISPLACEMENT,
    "vectors": VECTOR,
    "covectors": COVECTOR,
    "tensors": TENSOR,
}
"""The kind that each accepted value of `brainhops_field_type` names."""

TENSOR_LAYOUT = TensorLayout.symmetric_diagonal_first
"""The order in which MRtrix stores a symmetric matrix.

`dwi2tensor` stores the coefficients of a diffusion tensor as `D11`,
`D22`, `D33`, `D12`, `D13` and `D23` (`cmd/dwi2tensor.cpp`), which is
the diagonal first and then the upper triangle row by row.
"""

_SIZES = {
    DEFORMATION: 3,
    DISPLACEMENT: 3,
    VECTOR: 3,
    COVECTOR: 3,
    TENSOR: 6,
}
"""The number of volumes that each kind of values has in three dimensions."""

# A few commands write one kind of 4-D warp only (see `history_kind`).
_DEFORMATION_COMMANDS = ("warpinit", "warpcorrect", "mrregister")

_COMPONENT_AXIS = 3
"""The axis that holds the components, after the three spatial axes."""


def history_lines(header: tx.Optional[MrtrixRaw]) -> tx.List[str]:
    """Return the lines of the `command_history` of a header.

    Parameters
    ----------
    header : MrtrixRaw, optional
        The record of the header, or `None`.

    Returns
    -------
    list of str
        The lines that are not empty, oldest first.
    """
    if header is None:
        return []
    history = header.keyval.get(HISTORY)
    if not history:
        return []
    return [line for line in str(history).split("\n") if line.strip()]


def brainhops_history() -> str:
    """Return the `command_history` line that brainhops writes.

    MRtrix appends the command line and its version to the
    `command_history` of every file that it writes (`core/app.cpp`).

    Returns
    -------
    str
        The line, such as `"brainhops  (version=1.0)"`.
    """
    from brainhops import __version__

    return f"{BRAINHOPS}  (version={__version__})"


def written_by_brainhops(header: MrtrixRaw) -> bool:
    """Return whether the last `command_history` line names brainhops.

    Parameters
    ----------
    header : MrtrixRaw
        The record of the header.

    Returns
    -------
    bool
        Whether brainhops wrote the file last.
    """
    lines = history_lines(header)
    if not lines:
        return False
    tokens = lines[-1].split()
    return bool(tokens) and tokens[0] == BRAINHOPS


def stored_kind(header: MrtrixRaw) -> tx.Optional[str]:
    """Return the kind of values that `brainhops_field_type` records.

    Parameters
    ----------
    header : MrtrixRaw
        The record of the header.

    Returns
    -------
    str or None
        The kind, or `None` when the key is missing or not known.
    """
    value = header.keyval.get(FIELD_TYPE)
    if value is None:
        return None
    return FIELD_TYPE_KINDS.get(str(value).strip().lower())


def last_command(header: MrtrixRaw) -> tx.Tuple[str, tx.List[str]]:
    """Return the command that wrote a file last, and its arguments.

    MRtrix appends `argv` to `command_history` in every image that it
    writes (`core/app.cpp`), followed by `  (version=...)`, which is
    dropped here.

    Parameters
    ----------
    header : MrtrixRaw
        The record of the header.

    Returns
    -------
    tuple of (str, list of str)
        The base name of the command, or an empty string without a
        history, and its arguments.
    """
    lines = history_lines(header)
    if not lines:
        return "", []
    last = lines[-1].split("  (version=", 1)[0]
    tokens = [t.strip("'\"") for t in last.split()]
    if not tokens:
        return "", []
    return os.path.basename(tokens[0]), tokens[1:]


def history_kind(header: MrtrixRaw) -> tx.Optional[str]:
    """Return the kind of warp that the command that wrote a file outputs.

    The last line of `command_history` names the command that wrote the
    file, and a few commands output one kind of 4-D warp only.

    - `warpconvert <in> <type> <out>` writes a displacement or a
      deformation, which `*2displacement` or `*2deformation` says.
    - `warpinvert` writes a displacement with `-displacement`, and a
      deformation otherwise.
    - `warpinit`, `warpcorrect` and `mrregister -nl_warp` write a
      deformation.

    Anything else, such as a file processed further or written by
    another program, says nothing.

    Parameters
    ----------
    header : MrtrixRaw
        The record of the header.

    Returns
    -------
    str or None
        `"deformation"` or `"displacement"`, or `None`.
    """
    command, args = last_command(header)
    if command == "warpconvert":
        for arg in args:
            if arg.endswith("2" + DISPLACEMENT):
                return DISPLACEMENT
            if arg.endswith("2" + DEFORMATION):
                return DEFORMATION
        return None
    if command == "warpinvert":
        if any(arg.lstrip("-") == DISPLACEMENT for arg in args):
            return DISPLACEMENT
        return DEFORMATION
    if command in _DEFORMATION_COMMANDS:
        return DEFORMATION
    return None


def header_kind(header: MrtrixRaw) -> tx.Optional[str]:
    """Return what the fourth axis of an image holds, as its header says.

    The clues are those of the module docstring. The key
    `brainhops_field_type` decides while brainhops wrote the file last.
    Otherwise the last command of `command_history` decides when it
    writes one kind of values only, in the shape that it gives them:

    - `dwi2tensor` writes the six coefficients of a diffusion tensor in
      a 4-D image of six volumes (`cmd/dwi2tensor.cpp`). Its other
      outputs have another shape: the kurtosis tensor has 15 volumes,
      the predicted signal has one volume per diffusion-weighted volume,
      at least seven since the fit has seven unknowns, and the `b0`
      image is 3-D.
    - the warp commands of [`history_kind`][] write a 4-D image of three
      volumes. `mrregister` is left out, because it also writes the
      moved image, which may have three volumes as well.
    - `mrregister -nl_warp_full` writes a 5-D image of shape
      `(X, Y, Z, 3, 4)`, which holds four deformations along its last
      axis (`cmd/mrregister.cpp` and `src/registration/nonlinear.h`).

    The kind must agree with the number of volumes along the fourth
    axis: three for vectors and warps, and six for tensors. The key of
    brainhops may describe an image with more axes, such as a series of
    fields, whose components are still along the fourth axis. Any other
    image, including one whose key disagrees with its shape, gives
    `None`.

    Parameters
    ----------
    header : MrtrixRaw
        The record of the header.

    Returns
    -------
    str or None
        The kind of the values along the fourth axis, or `None` when the
        header does not tell.
    """
    dim = tuple(int(d) for d in header.dim)
    if len(dim) <= _COMPONENT_AXIS:
        return None
    count = dim[_COMPONENT_AXIS]
    warpfull = len(dim) == 5 and dim[3:] == (3, 4)
    if written_by_brainhops(header):
        kind = stored_kind(header)
        if kind is None or _SIZES[kind] != count:
            return None
        return kind
    command, _ = last_command(header)
    if len(dim) == 4 and command == "dwi2tensor" and count == 6:
        return TENSOR
    if len(dim) == 4 and count == 3 and command != "mrregister":
        return history_kind(header)
    if warpfull and command == "mrregister":
        return DEFORMATION
    return None


def component_axis(kind: str, name: str, unit: tx.Any) -> Axis:
    """Return the axis that holds the values of a kind.

    Parameters
    ----------
    kind : str
        The kind of the values, as [`header_kind`][] returns it.
    name : str
        The name of the axis.
    unit : Any
        The unit of the axis.

    Returns
    -------
    Axis
        A coordinate, displacement, vector or tensor axis.
    """
    if kind == DEFORMATION:
        return Axis(name, "coordinate", unit=unit)
    if kind == DISPLACEMENT:
        return Axis(name, "displacement", unit=unit)
    if kind == VECTOR:
        return Axis(name, "vector", unit=unit, valence=Valence.vector)
    if kind == COVECTOR:
        return Axis(name, "vector", unit=unit, valence=Valence.covector)
    return Axis(name, "tensor", unit=unit, layout=TENSOR_LAYOUT)


def axis_kind(axis: Axis) -> tx.Optional[str]:
    """Return the kind of values that MRtrix stores for an axis.

    A tensor axis is stored as such only when it holds a tensor of rank
    two whose indices are both of valence `"vector"`, as a diffusion
    tensor is, because MRtrix knows no other tensor.

    Parameters
    ----------
    axis : Axis
        A component axis.

    Returns
    -------
    str or None
        The kind, or `None` when MRtrix cannot store the values of the
        axis in a frame of its own.
    """
    kind = getattr(axis, "type", None)
    if kind in ("coordinate", "point"):
        return DEFORMATION
    if kind == "displacement":
        return DISPLACEMENT
    if kind == "vector":
        if axis.valences == (Valence.covector,):
            return COVECTOR
        return VECTOR
    if kind == "tensor":
        if tuple(axis.valences) == (Valence.vector, Valence.vector):
            return TENSOR
    return None


def kind_size(kind: str) -> int:
    """Return the number of volumes that a kind of values has.

    Parameters
    ----------
    kind : str
        The kind of the values.

    Returns
    -------
    int
        Three for vectors and warps, six for tensors.
    """
    return _SIZES[kind]
