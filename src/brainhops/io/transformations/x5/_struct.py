"""
The raw content of an X5 file, as stored, before any interpretation.

These structs mirror the HDF5 layout one to one, so that a file read and
written back keeps everything it said -- its JSON metadata included --
even when brainhops cannot turn a transform into a datamodel object.
"""

__all__ = [
    "X5Domain",
    "X5Header",
    "X5Node",
    "read_x5",
    "write_x5",
]

# stdlib
import json

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

from brainhops.io.base.parsers import ParserContentError

# io
from brainhops.io.common.hdf5 import delayed_dataset, read_string

X5_FORMAT = "X5"
"""The value of the root `Format` attribute of every X5 file."""

X5_VERSION = np.uint16(1)
"""The root `Version` attribute written, as nitransforms writes it."""

TRANSFORM_GROUP = "TransformGroup"
TRANSFORM_CHAIN = "TransformChain"

# Node attributes and datasets that have a field of their own; any other
# attribute is kept, as read, in `X5Node.attrs`.
_NODE_ATTRS = ("Type", "SubType", "Representation", "Metadata", "ArrayLength")


class X5Domain(Magic, repr=HIDE_IF_NONE):
    """
    The `Domain` group of a transform: the grid it is sampled on.

    REQUIRED for a `nonlinear` transform, RECOMMENDED for a `linear` one.
    """

    grid: bool = True
    """Whether the samples lie on a regular grid (`Grid` dataset)."""

    size: tx.Tuple[int, ...] = ()
    """The number of samples per dimension (`Size` dataset)."""

    mapping: tx.Any = None
    """
    The `(D + 1, D + 1)` affine from sample indices to world (RAS)
    coordinates (`Mapping` dataset): the voxel-to-RAS affine of the grid.
    """

    coordinates: tx.Optional[str] = None
    """The kind of world coordinates (`Coordinates` attribute), e.g.
    `"cartesian"`."""


class X5Node(Magic, repr=HIDE_IF_NONE):
    """
    One numbered group `/TransformGroup/<i>` of an X5 file.

    Every field mirrors an attribute or a dataset of the group. Large
    datasets (`Transform`, `Inverse`, `Jacobian`) are read lazily when
    the file is opened with `load=False`.
    """

    type: str = "linear"
    """`Type`: `"linear"`, `"nonlinear"` or `"composite"`. fslpy's
    `"affine"` is read as `"linear"`."""

    transform: tx.Any = None
    """`Transform`: the parameters -- a `(D + 1, D + 1)` matrix, a stack
    of them, or a dense field."""

    subtype: tx.Optional[str] = None
    """`SubType`, e.g. `"affine"`, `"densefield"` or `"bspline"`."""

    representation: tx.Optional[str] = None
    """`Representation`, e.g. `"matrix"`, `"displacements"` or
    `"deformations"`."""

    metadata: tx.Any = None
    """`Metadata`: the JSON attribute, decoded (a `dict`), or the raw
    string if it is not valid JSON."""

    dimension_kinds: tx.Optional[tx.Tuple[str, ...]] = None
    """`DimensionKinds`: what each axis of `transform` holds, e.g.
    `("space", "space", "space", "vector")`."""

    domain: tx.Optional[X5Domain] = None
    """The `Domain` group."""

    inverse: tx.Any = None
    """`Inverse`: an optional precomputed inverse."""

    jacobian: tx.Any = None
    """`Jacobian`: an optional cached Jacobian determinant."""

    additional_parameters: tx.Any = None
    """`AdditionalParameters`: subtype-specific parameters (for a
    `bspline`, the affine of the grid of knots)."""

    array_length: int = 1
    """`ArrayLength`: how many transforms `transform` stacks."""

    attrs: tx.Dict[str, tx.Any] = Factory(dict)
    """Any other attribute of the group, as read."""


class X5Header(Magic, repr=HIDE_IF_NONE):
    """The root of an X5 file: its attributes and its chains."""

    format: str = X5_FORMAT
    """The root `Format` attribute."""

    version: tx.Any = X5_VERSION
    """The root `Version` attribute, as read: `1` for the current draft,
    `"0.1.0"` for fslpy's earlier layout."""

    attrs: tx.Dict[str, tx.Any] = Factory(dict)
    """Any other root attribute, as read."""

    chains: tx.List[tx.Tuple[int, ...]] = Factory(list)
    """
    The chains of `/TransformChain`, in order: each lists, in the order
    they are applied, the indices of the transforms it chains.
    """

    legacy: bool = False
    """Whether the file used fslpy's earlier (`0.x`) layout."""


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def is_x5(h5file: h5py.File) -> bool:
    """Whether an open HDF5 file says it is an X5 file."""
    try:
        return read_string(h5file.attrs.get("Format")) == X5_FORMAT
    except (ValueError, UnicodeDecodeError):
        return False


def read_x5(
    h5file: h5py.File, load: bool = True, keep_open: bool = False
) -> tx.Tuple[X5Header, tx.List[X5Node]]:
    """Read the header and the transform nodes of an open X5 file."""
    if not is_x5(h5file):
        raise ParserContentError("Not an X5 file: no root Format='X5'.")
    if TRANSFORM_GROUP not in h5file and "Transform" in h5file:
        return _read_legacy(h5file)

    header = X5Header(
        format=X5_FORMAT,
        version=h5file.attrs.get("Version"),
        attrs={
            key: value
            for key, value in h5file.attrs.items()
            if key not in ("Format", "Version")
        },
    )
    group = h5file.get(TRANSFORM_GROUP)
    if group is None:
        raise ParserContentError("An X5 file has no /TransformGroup.")
    keys = sorted(group.keys(), key=_index)
    if [_index(key) for key in keys] != list(range(len(keys))):
        raise ParserContentError(
            f"The groups of /TransformGroup must be numbered 0, 1, ...; "
            f"found {keys}."
        )
    nodes = [
        _read_node(group[key], load=load, keep_open=keep_open) for key in keys
    ]

    chains = h5file.get(TRANSFORM_CHAIN)
    if chains is not None:
        for key in sorted(chains.keys(), key=_index):
            header.chains.append(_read_chain(chains[key], len(nodes)))
    return header, nodes


def _index(key: str) -> int:
    try:
        return int(key)
    except ValueError:
        raise ParserContentError(
            f"An X5 group is named {key!r}, not by its index."
        ) from None


def _read_chain(dataset: h5py.Dataset, count: int) -> tx.Tuple[int, ...]:
    """A chain, stored by nitransforms as a string `"0/1/2"`."""
    text = read_string(dataset)
    try:
        chain = tuple(int(i) for i in text.split("/"))
    except ValueError:
        raise ParserContentError(
            f"An X5 chain lists transform indices as '0/1/2', not {text!r}."
        ) from None
    if any(i < 0 or i >= count for i in chain):
        raise ParserContentError(
            f"An X5 chain refers to a transform that does not exist: "
            f"{text!r}, with {count} transforms."
        )
    return chain


def _read_dataset(
    group: h5py.Group, key: str, load: bool, keep_open: bool
) -> tx.Any:
    if key not in group:
        return None
    dataset = group[key]
    if load or dataset.ndim <= 2:  # matrices are always read
        return dataset[()]
    return delayed_dataset(group.file, dataset.name, keep_open)


def _read_node(group: h5py.Group, load: bool, keep_open: bool) -> X5Node:
    attrs = group.attrs
    if "Type" not in attrs:
        raise ParserContentError(f"X5 group {group.name} has no Type.")
    if "Transform" not in group:
        raise ParserContentError(
            f"X5 group {group.name} has no Transform dataset."
        )
    xtype = read_string(attrs["Type"])
    if xtype == "affine":  # fslpy
        xtype = "linear"
    node = X5Node(
        type=xtype,
        transform=_read_dataset(group, "Transform", load, keep_open),
        subtype=read_string(attrs.get("SubType")),
        representation=read_string(attrs.get("Representation")),
        metadata=_read_metadata(attrs.get("Metadata")),
        array_length=int(attrs.get("ArrayLength", 1)),
        inverse=_read_dataset(group, "Inverse", load, keep_open),
        jacobian=_read_dataset(group, "Jacobian", load, keep_open),
        additional_parameters=_read_dataset(
            group, "AdditionalParameters", True, keep_open
        ),
        attrs={k: v for k, v in attrs.items() if k not in _NODE_ATTRS},
    )
    if "DimensionKinds" in group:
        kinds = np.asarray(group["DimensionKinds"][()], dtype=object)
        kinds = tuple(read_string(kind) for kind in kinds.ravel())
        # nitransforms writes `np.asarray(None, dtype="S")`, i.e. b"None",
        # when a transform has no kinds.
        node.dimension_kinds = None if kinds == ("None",) else kinds
    if "Domain" in group:
        node.domain = _read_domain(group["Domain"])
    return node


def _read_metadata(value: tx.Any) -> tx.Any:
    if value is None:
        return None
    text = read_string(value)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _read_domain(group: h5py.Group) -> X5Domain:
    domain = X5Domain()
    if "Grid" in group:
        domain.grid = bool(int(np.asarray(group["Grid"][()])))
    if "Size" in group:
        domain.size = tuple(int(s) for s in np.ravel(group["Size"][()]))
    if "Mapping" in group:
        domain.mapping = np.asarray(group["Mapping"][()], dtype=np.float64)
    domain.coordinates = read_string(group.attrs.get("Coordinates"))
    return domain


# --- fslpy's earlier layout -------------------------------------------
#
# fslpy (`fsl.transform.x5`, 3.x) writes `Version="0.1.0"` files with a
# single transform at the root: a `Type` attribute ("linear" or
# "nonlinear"), a `/Transform` group holding a `Matrix` dataset, and the
# two image spaces `/A` and `/B`. A deformation also holds its grid in
# `/Transform/Mapping/Matrix`, and says in `SubType` whether it stores
# "relative" displacements or "absolute" coordinates. Both map world
# coordinates of space A to world coordinates of space B.


def _read_legacy(h5file: h5py.File) -> tx.Tuple[X5Header, tx.List[X5Node]]:
    header = X5Header(
        format=X5_FORMAT,
        version=h5file.attrs.get("Version"),
        attrs={
            key: value
            for key, value in h5file.attrs.items()
            if key not in ("Format", "Version", "Type", "Metadata")
        },
        legacy=True,
    )
    xtype = read_string(h5file.attrs.get("Type"))
    metadata = _read_metadata(h5file.attrs.get("Metadata"))
    group = h5file["Transform"]
    if "Matrix" not in group:
        raise ParserContentError("An fslpy X5 /Transform has no Matrix.")
    matrix = np.asarray(group["Matrix"][()])
    if xtype == "linear":
        inverse = group["Inverse"][()] if "Inverse" in group else None
        node = X5Node(
            type="linear",
            subtype="affine",
            representation="matrix",
            transform=matrix,
            inverse=inverse,
            metadata=metadata,
        )
    elif xtype == "nonlinear":
        subtype = read_string(group.attrs.get("SubType"))
        representation = {
            "relative": "displacements",
            "absolute": "deformations",
        }.get(subtype)
        if representation is None:
            raise ParserContentError(
                f"Unknown fslpy deformation type: {subtype!r}."
            )
        if "Mapping" not in group or "Matrix" not in group["Mapping"]:
            raise ParserContentError(
                "An fslpy X5 deformation has no /Transform/Mapping."
            )
        mapping = np.asarray(group["Mapping"]["Matrix"][()], np.float64)
        node = X5Node(
            type="nonlinear",
            subtype="densefield",
            representation=representation,
            transform=matrix,
            metadata=metadata,
            dimension_kinds=("space",) * (matrix.ndim - 1) + ("vector",),
            domain=X5Domain(
                grid=True,
                size=tuple(matrix.shape[:-1]),
                mapping=mapping,
                coordinates="cartesian",
            ),
        )
    else:
        raise ParserContentError(f"Unknown fslpy X5 type: {xtype!r}.")
    return header, [node]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def write_x5(
    h5file: h5py.File, header: X5Header, nodes: tx.Sequence[X5Node]
) -> None:
    """Write a header and transform nodes into an empty HDF5 file."""
    h5file.attrs["Format"] = X5_FORMAT
    version = header.version
    if header.legacy or version is None:
        # The nodes are always written in the current layout, so they
        # are given its version, not the one of fslpy's layout.
        version = X5_VERSION
    h5file.attrs["Version"] = version
    for key, value in header.attrs.items():
        h5file.attrs[key] = value
    group = h5file.create_group(TRANSFORM_GROUP)
    for i, node in enumerate(nodes):
        _write_node(group.create_group(str(i)), node)
    if header.chains:
        chains = h5file.create_group(TRANSFORM_CHAIN)
        for i, chain in enumerate(header.chains):
            chains.create_dataset(
                str(i), data="/".join(str(int(j)) for j in chain)
            )


def _write_node(group: h5py.Group, node: X5Node) -> None:
    for key, value in node.attrs.items():
        group.attrs[key] = value
    group.attrs["Type"] = node.type
    group.attrs["ArrayLength"] = int(node.array_length)
    if node.subtype is not None:
        group.attrs["SubType"] = node.subtype
    if node.representation is not None:
        group.attrs["Representation"] = node.representation
    if node.metadata is not None:
        metadata = node.metadata
        if not isinstance(metadata, str):
            metadata = json.dumps(metadata)
        group.attrs["Metadata"] = metadata
    group.create_dataset("Transform", data=np.asarray(node.transform))
    if node.dimension_kinds is not None:
        group.create_dataset(
            "DimensionKinds", data=np.asarray(node.dimension_kinds, dtype="S")
        )
    if node.domain is not None:
        domain = group.create_group("Domain")
        domain.create_dataset("Grid", data=np.uint8(bool(node.domain.grid)))
        domain.create_dataset("Size", data=np.asarray(node.domain.size))
        if node.domain.mapping is not None:
            domain.create_dataset(
                "Mapping", data=np.asarray(node.domain.mapping)
            )
        if node.domain.coordinates is not None:
            domain.attrs["Coordinates"] = node.domain.coordinates
    for key, value in (
        ("Inverse", node.inverse),
        ("Jacobian", node.jacobian),
        ("AdditionalParameters", node.additional_parameters),
    ):
        if value is not None:
            group.create_dataset(key, data=np.asarray(value))
