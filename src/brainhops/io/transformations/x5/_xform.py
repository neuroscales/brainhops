"""The chain of transformations stored in an X5 file."""

__all__ = ["X5Transform"]

from warnings import warn

import h5py
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.common.hdf5 import Hdf5ReaderWriter
from brainhops.io.transformations.base import TransformationFormat

from ._blocks import (
    _check_ras,
    node_to_transformation,
    transformation_to_nodes,
)
from ._metadata import X5Metadata
from ._raw import X5_VERSION, X5Header, X5Node, X5Raw

# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------


def _record(xform: "X5Transform") -> tx.Optional[X5Raw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _check_selection(
    record: tx.Optional[X5Raw],
    chain: tx.Optional[int],
    position: tx.Optional[int],
) -> None:
    """Refuse a chain or a node that a record does not have.

    Raises
    ------
    ValueError
        If both `chain` and `position` are given.
    ParserContentError
        If `chain` or `position` is out of range.
    """
    if chain is not None and position is not None:
        raise ValueError("Select a chain or a position, not both.")
    chains = [] if record is None else record.header.chains
    nodes = [] if record is None else record.nodes
    if chain is not None and not 0 <= chain < len(chains):
        raise ParserContentError(
            f"This X5 file has {len(chains)} chain(s), so it has no chain "
            f"{chain}."
        )
    if position is not None and not 0 <= position < len(nodes):
        raise ParserContentError(
            f"This X5 file has {len(nodes)} transform(s), so it has no "
            f"transform {position}."
        )


def _set_metadata(self: "X5Transform", value: tx.Optional[X5Metadata]) -> None:
    """Store the metadata and forget the nodes decoded from the old record.

    The chain decoded from the old record is deleted by the property, which
    names `transformations` among the values that it invalidates.
    """
    self._metadata = value
    self.__dict__.pop("_decoded_nodes", None)


# ----------------------------------------------------------------------
#   FORMAT
# ----------------------------------------------------------------------


@register_format
class X5Transform(
    _xforms.Sequence,
    TransformationFormat,
    Hdf5ReaderWriter,
):
    """A transformation stored in a BIDS X5 (`.x5`) file.

    The transformation is a
    [`Sequence`][brainhops.datamodel.transformations.Sequence]
    of the transforms that the file chains, in the order they are applied
    (see [`X5Transform.selection`][]). Each transform maps RAS millimetre
    coordinates to RAS millimetre coordinates.

    | X5 node                                | Transformation              |
    | -------------------------------------- | --------------------------- |
    | `linear`, a 4x4 matrix                 | `Affine` (`RASmm` to `RASmm`) |
    | `nonlinear`, `displacements`           | [`X5DisplacementField`][]   |
    | `nonlinear`, `deformations`            | [`X5CoordinatesField`][]    |
    | `nonlinear`, `bspline`                 | [`X5BSplineField`][]        |

    The content of the file, including each node's JSON metadata, `Domain`,
    `Inverse` and other attributes, is held as an [`X5Raw`][] record by the
    [`X5Metadata`][] of the transformation. The fields of the record are
    lazy arrays, which read the file again by its name when they are used,
    so reading a file reads its matrices but none of its fields. The chain
    of transformations is decoded from the record when it is first used,
    and it is then cached.

    !!! note "What is written"
        A transformation read from a file, whose `transformations` were never
        assigned, is written back as read, with every node and chain, in the
        current layout. Otherwise, the nodes that encode `transformations` are
        written, with one `/TransformChain` that applies them in order when
        there are several. An element that was decoded from a node is written
        as that node, so its metadata survives. The package documentation lists
        what can be encoded.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".x5",)
    HINTS = ("x5", "bids")

    _metadata: KwOnly[NoRepr[tx.Optional[X5Metadata]]] = None

    metadata = smartproperty(
        "metadata", fset=_set_metadata, invalidates=("transformations",)
    )
    """The metadata of the file, which holds the file as a record.

    A transformation built from a chain has no metadata. Assigning other
    metadata drops the chain decoded from the old record.
    """

    _chain: KwOnly[tx.Optional[int]] = None

    chain = smartproperty("chain", invalidates=("transformations",))
    """The index of the chain of `/TransformChain` that is read.

    See [`X5Transform.selection`][]. Assigning another index drops the
    chain decoded from the record.
    """

    _position: KwOnly[tx.Optional[int]] = None

    position = smartproperty("position", invalidates=("transformations",))
    """The index of the single node of `/TransformGroup` that is read.

    See [`X5Transform.selection`][]. Assigning another index drops the
    chain decoded from the record.
    """

    # --- chain --------------------------------------------------------

    @property
    def selection(self) -> tx.Tuple[int, ...]:
        """The indices of the nodes that this transformation chains, in order.

        The indices are resolved as follows:

        1. The chain `chain` of `/TransformChain`, if one was asked for.
        2. The node `position`, if one was asked for.
        3. Otherwise the first chain of `/TransformChain`, if the file
           has one, as nitransforms' `TransformChain.from_filename` does.
        4. Otherwise the file's single node.
        5. Otherwise, in a file with several nodes and no chain, the first
           node, as nitransforms' `Affine.from_filename` and
           `DenseFieldTransform.from_filename` do. A warning is issued in
           this case, because the draft does not say how nodes that no
           chain lists relate to each other.

        A transformation without a record selects no node.

        Raises
        ------
        ValueError
            If both `chain` and `position` are set.
        ParserContentError
            If `chain` or `position` is out of range.
        """
        record = _record(self)
        _check_selection(record, self.chain, self.position)
        if record is None:
            return ()
        chains = record.header.chains
        if self.chain is not None:
            return tuple(chains[self.chain])
        if self.position is not None:
            return (int(self.position),)
        if chains:
            return tuple(chains[0])
        if len(record.nodes) > 1:
            warn(
                f"This X5 file holds {len(record.nodes)} transforms and no "
                f"chain, so only the first one is read. Pass `position=` "
                f"to read another one.",
                stacklevel=2,
            )
        return (0,) if record.nodes else ()

    def node_transformation(self, index: int) -> _xforms.Transformation:
        """Decode node `index` of the record into a transformation.

        The node is decoded once and the same object is returned afterwards,
        so that the writer can recognise it by identity. Assigning other
        metadata forgets the decoded nodes.

        Raises
        ------
        ParserContentError
            If the transformation has no record.
        """
        record = _record(self)
        if record is None:
            raise ParserContentError("This X5 transformation has no record.")
        decoded = self.__dict__.setdefault("_decoded_nodes", {})
        if index not in decoded:
            decoded[index] = node_to_transformation(record.nodes[index])
        return decoded[index]

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations, in the order they are applied.

        The chain is decoded from the record when it is first used, and it
        is then cached. Assigning a chain replaces the decoded one, and the
        assigned chain is what the writer encodes.
        """
        return tuple(self.node_transformation(i) for i in self.selection)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file by its root `Format` attribute.

        The file is scored by [`X5Metadata.sniff_h5`][].
        """
        return X5Metadata.sniff_h5(h5file, error=error)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        h5file: h5py.File,
        load: bool = False,
        chain: tx.Optional[int] = None,
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Read an X5 transformation from an open HDF5 file.

        The record is read by [`X5Metadata.from_h5`][], and the chain is
        decoded from it when it is first used.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the fields into memory at once, rather than
            when they are used. The fields of a file opened from a stream
            are always read at once.
        chain : int, optional
            The index of the chain of `/TransformChain` to read.
        position : int, optional
            The index of the single node of `/TransformGroup` to read.
        **kwargs : Any
            Other fields of the transformation.

        Returns
        -------
        X5Transform
            The transformation.

        Raises
        ------
        ValueError
            If both `chain` and `position` are given.
        ParserContentError
            If `chain` or `position` is out of range.
        """
        metadata = X5Metadata.from_h5(h5file, load=load)
        _check_selection(metadata.raw, chain, position)
        return cls(metadata=metadata, chain=chain, position=position, **kwargs)

    @classmethod
    def from_raw(cls, raw: X5Raw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`X5Raw`][].

        The record is held by new metadata, without a copy. Keyword
        arguments such as `chain` or `position` go to the constructor.
        """
        return cls(metadata=X5Metadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        An [`X5Raw`][] is read with [`from_raw`][], and any other value as
        the bases read it.
        """
        if isinstance(other, X5Raw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def to_raw(self) -> X5Raw:
        """Return the [`X5Raw`][] that encodes this transformation.

        If neither `transformations` nor the `input` and `output` systems
        were assigned, the record that the metadata holds is returned as it
        is, without a copy, so that the file is written back as it was
        read, with every node and chain. The options `chain` and `position`
        only select what is decoded, so they do not change the record.
        Otherwise, a new record is built from the chain, whose header keeps
        the root attributes of the current record. A transformation without
        a record and without a chain gives an empty record.

        Raises
        ------
        UnrepresentableTransformationError
            If an element cannot be encoded in X5, or if the systems that
            were assigned are not `RASmm`.
        """
        record = _record(self)
        assigned = getattr(self, "_transformations", None) is not None
        declared = self._input is not None or self._output is not None
        if not assigned and not declared:
            return X5Raw() if record is None else record
        if declared:
            _check_ras(self, "chain")
        decoded = self.__dict__.get("_decoded_nodes", {})
        return _encode(self.transformations, record, decoded)

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the transformation into an empty HDF5 file.

        The record is written by [`X5Raw.to_h5`][].
        """
        self.to_raw().to_h5(h5file, **kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the transformation to the file at a path, replacing it.

        The record is built before the file is opened, so a transformation
        that cannot be encoded leaves the file untouched.
        """
        self.to_raw().to_filename(filename, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the X5 file that encodes the transformation."""
        return self.to_raw().to_bytes(**kwargs)


# ----------------------------------------------------------------------
#   ENCODING A CHAIN AS A RECORD
# ----------------------------------------------------------------------


def _encode(
    chain: tx.Sequence[_xforms.Transformation],
    record: tx.Optional[X5Raw],
    decoded: tx.Dict[int, _xforms.Transformation],
) -> X5Raw:
    """Encode a chain of transformations as a new record.

    An element that was decoded from a node of `record`, which `decoded`
    lists by node index, is written as that node, so its metadata survives.
    Any other element is encoded by `transformation_to_nodes`. Several
    nodes are chained by one `/TransformChain`. The root attributes of
    `record` are kept, and the header is written in the current layout.
    """
    reused = {id(xform): index for index, xform in decoded.items()}
    nodes: tx.List[X5Node] = []
    indices: tx.Dict[int, int] = {}
    order: tx.List[int] = []
    for xform in chain:
        if id(xform) in reused:
            encoded = [record.nodes[reused[id(xform)]]]
        else:
            encoded = transformation_to_nodes(xform)
        for node in encoded:
            if id(node) not in indices:
                indices[id(node)] = len(nodes)
                nodes.append(node)
            order.append(indices[id(node)])
    header = X5Header() if record is None else record.header
    version = X5_VERSION if header.legacy else header.version
    header = replace(
        header,
        attrs=dict(header.attrs),
        chains=[tuple(order)] if len(order) > 1 else [],
        legacy=False,
        version=version,
    )
    return X5Raw(header=header, nodes=nodes)
