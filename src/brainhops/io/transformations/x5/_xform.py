__all__ = ["X5Transform", "X5TransformParser"]

# stdlib
from warnings import warn

# dependencies
import h5py
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic, replace

from brainhops._core.properties import smartproperty

# datamodel
from brainhops.datamodel import transformations as _xforms

# core
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._field import MetadataField
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.hdf5 import Hdf5ParserWriter
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.metadata._sync import sync_metadata
from brainhops.io.transformations.base import WritableFileBasedTransformation

# locals
from ._blocks import node_to_transformation, transformation_to_nodes
from ._metadata import X5Metadata, X5Raw, metadata_index
from ._struct import (
    X5_VERSION,
    X5Header,
    X5Node,
    is_x5,
    read_x5,
    write_x5,
)


class X5TransformParser(
    Magic,
    Hdf5ParserWriter,
    repr=HIDE_IF_NONE,
):
    """Reads and writes the raw content of a BIDS X5 file."""

    header: X5Header = Factory(X5Header, repr=False)
    """The root attributes and the chains of the file."""

    nodes: tx.List[X5Node] = Factory(list, repr=False)
    """Every transform of `/TransformGroup`, as stored."""

    chain: tx.Optional[int] = None
    """
    Which chain of `/TransformChain` the transformation is, if any.
    See `X5Transform.selection`.
    """

    position: tx.Optional[int] = None
    """
    Which single transform of `/TransformGroup` the transformation is,
    if any. See `X5Transform.selection`.
    """

    file: tx.Optional[h5py.File] = None
    """The open HDF5 file, when read with `keep_open=True`."""

    metadata: MetadataField[
        X5Metadata,
        Factory(),
        tx.Doc(
            """
            The metadata of the node the transformation was read from,
            decoded from its JSON `Metadata`, with `(header, node)` as its
            record. A chain of several nodes has none of its own: its
            nodes keep theirs, and write them back. See
            [`X5Metadata`][brainhops.io.transformations.x5.X5Metadata].
            """
        ),
    ]

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        index = self._metadata_index() if self.nodes else None
        node = None if index is None else self.nodes[index]
        header = self.header
        sync_metadata(
            self,
            X5Metadata,
            X5Raw(header, node),
            same=lambda held: held.is_record_of(header, node),
            image=self,
        )

    def _metadata_index(self) -> tx.Optional[int]:
        """The node the metadata is that of: the single node read, or
        `None` for a chain of several (composition does not merge)."""
        return metadata_index(self.header, self.chain, self.position)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file: an X5 file says so in its root
        `Format` attribute."""
        if is_x5(h5file):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error("HDF5 file is not an X5 file: no Format='X5'.")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        h5file: h5py.File,
        keep_open: bool = False,
        load: bool = True,
        chain: tx.Optional[int] = None,
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Build an object from an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Input HDF5 file.
        keep_open : bool
            Keep the file open after reading it.
        load : bool
            Read the fields into memory. If `False`, they are read
            lazily, when used.
        chain : int, optional
            Which chain of `/TransformChain` to read.
        position : int, optional
            Which single transform of `/TransformGroup` to read.
        """
        header, nodes = read_x5(h5file, load=load, keep_open=keep_open)
        if chain is not None and position is not None:
            raise ValueError("Select a chain or a position, not both.")
        if chain is not None and not 0 <= chain < len(header.chains):
            raise ParserContentError(
                f"This X5 file has {len(header.chains)} chain(s), so it "
                f"has no chain {chain}."
            )
        if position is not None and not 0 <= position < len(nodes):
            raise ParserContentError(
                f"This X5 file has {len(nodes)} transform(s), so it has "
                f"no transform {position}."
            )
        return cls(
            header=header,
            nodes=nodes,
            chain=chain,
            position=position,
            file=h5file if keep_open else None,
            **kwargs,
        )

    # --- to -----------------------------------------------------------

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        on_loss = kwargs.pop("on_loss", None)
        report = ConversionReport(target="x5")
        header, nodes = self._to_struct(report)
        apply_loss_policy(report, on_loss, stacklevel=4)
        return lambda h5file: write_x5(h5file, header, nodes)

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write this transformation into an empty HDF5 file."""
        self._h5_writer(**kwargs)(h5file)

    def to_struct(
        self, *, on_loss: tx.Optional[OnLoss] = None
    ) -> tx.Tuple[X5Header, tx.List[X5Node]]:
        """
        The header and the nodes that encode this object.

        What the metadata cannot carry is handed to `on_loss`: the
        policy in effect by default, `"ignore"`, `"warn"`, `"raise"`, or
        a `ConversionReport` to fill.
        """
        report = ConversionReport(target="x5")
        out = self._to_struct(report)
        apply_loss_policy(report, on_loss, stacklevel=2)
        return out

    def _to_struct(
        self, report: ConversionReport
    ) -> tx.Tuple[X5Header, tx.List[X5Node]]:
        """`to_struct`, recording what is lost in `report`."""
        return self.header, list(self.nodes)

    def _close(self) -> None:
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self) -> None:
        """Close the underlying HDF5 file, if one is still open."""
        self._close()


@register_format
class X5Transform(
    X5TransformParser,
    _xforms.Sequence,
    WritableFileBasedTransformation,
):
    """
    A transformation stored in a BIDS X5 (`.x5`) file.

    It is the [`Sequence`][brainhops.datamodel.transformations.Sequence]
    of the transforms that the file chains, in the order in which they
    are applied (see [`selection`][.selection]). Every transform maps
    RAS world coordinates, in millimetres, to RAS world coordinates:

    | X5 node                                | Transformation              |
    | -------------------------------------- | --------------------------- |
    | `linear`, a 4x4 matrix                 | `Affine` (`RASmm` to `RASmm`) |
    | `nonlinear`, `displacements`           | [`X5DisplacementField`][]   |
    | `nonlinear`, `deformations`            | [`X5CoordinatesField`][]    |
    | `nonlinear`, `bspline`                 | [`X5BSplineField`][]        |

    The raw content of the file -- every node, with its JSON
    `Metadata`, its `Domain`, its precomputed `Inverse` and any other
    attribute -- is kept in `header` and `nodes`.

    !!! note "What is written"
        A transformation read from a file, whose chain has not been
        assigned, is written back as it was read: every node and chain
        of the file, in the current layout. One whose `transformations`
        were assigned -- including one built from scratch -- is written
        as the nodes those transformations encode, plus one
        `/TransformChain` that applies them in order when there are
        several. An element that is one of the transformations read
        from a node is written as that node, unchanged, so its metadata
        survives. The package documentation lists what can be encoded.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".x5",)
    HINTS = ("x5", "bids")

    # --- chain --------------------------------------------------------

    @property
    def selection(self) -> tx.Tuple[int, ...]:
        """
        The indices of the nodes this transformation chains, in order.

        1. The chain `chain` of `/TransformChain`, if one was asked for.
        2. The node `position`, if one was asked for.
        3. Otherwise the first chain of `/TransformChain`, if the file
           has one, as nitransforms' `TransformChain.from_filename` does.
        4. Otherwise the file's single node.
        5. Otherwise, in a file with several nodes and no chain, the first
           node, as nitransforms' `Affine.from_filename` and
           `DenseFieldTransform.from_filename` do, with a warning: the
           draft says nothing of how unchained nodes relate.
        """
        chains = self.header.chains
        if self.chain is not None:
            return tuple(chains[self.chain])
        if self.position is not None:
            return (int(self.position),)
        if chains:
            return tuple(chains[0])
        if len(self.nodes) > 1:
            warn(
                f"This X5 file holds {len(self.nodes)} transforms and no "
                f"chain, so only the first one is read. Pass `position=` "
                f"to read another one.",
                stacklevel=2,
            )
        return (0,) if self.nodes else ()

    def node_transformation(self, index: int) -> _xforms.Transformation:
        """
        The transformation that node `index` encodes.

        It is decoded once, and the same object is returned afterwards,
        which is how the writer recognises it. The node the metadata is
        that of (a single node read) gives its block a copy of it, as
        it is when the block is decoded.
        """
        blocks = self.__dict__.setdefault("_x5_blocks", {})
        if index not in blocks:
            block = node_to_transformation(self.nodes[index])
            if (
                self.metadata is not None
                and index == self._metadata_index()
                and hasattr(block, "metadata")
            ):
                # A single-node file: the metadata is the node's, so the
                # block carries it too (a copy, as the data model's own
                # metadata), and `from_other(block)` converts it.
                block.metadata = self.metadata
            blocks[index] = block
        return blocks[index]

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """
        The chain of transformations, in the order they are applied.

        It is decoded from the nodes on first access, and cached, as a
        tuple. Assigning to it overrides the decoded chain, and is what
        the writer then encodes.
        """
        return tuple(self.node_transformation(i) for i in self.selection)

    # --- to -----------------------------------------------------------

    def _to_struct(
        self, report: ConversionReport
    ) -> tx.Tuple[X5Header, tx.List[X5Node]]:
        """
        The header and the nodes that encode this transformation.

        The metadata is written into the JSON `Metadata` of the node it
        belongs to (see `X5Metadata`); what cannot be written is
        recorded in `report`.

        Raises
        ------
        UnrepresentableTransformationError
            If an element of the chain cannot be encoded in X5.
        """
        if getattr(self, "_transformations", None) is None:
            nodes = list(self.nodes)
            index = self._metadata_index() if nodes else None
            return self.header, self._write_metadata(nodes, index, report)

        blocks = self.__dict__.get("_x5_blocks", {})
        reused = {id(block): index for index, block in blocks.items()}
        nodes: tx.List[X5Node] = []
        indices: tx.Dict[int, int] = {}
        chain: tx.List[int] = []
        for xform in self._transformations:
            if id(xform) in reused:
                encoded = [self.nodes[reused[id(xform)]]]
            else:
                encoded = transformation_to_nodes(xform)
            for node in encoded:
                if id(node) not in indices:
                    indices[id(node)] = len(nodes)
                    nodes.append(node)
                chain.append(indices[id(node)])
        chains = [tuple(chain)] if len(chain) > 1 else []
        version = X5_VERSION if self.header.legacy else self.header.version
        header = replace(
            self.header, chains=chains, legacy=False, version=version
        )
        index = 0 if len(nodes) == 1 else None
        return header, self._write_metadata(nodes, index, report)

    def _write_metadata(
        self,
        nodes: tx.List[X5Node],
        index: tx.Optional[int],
        report: ConversionReport,
    ) -> tx.List[X5Node]:
        """Encode the metadata into node `index` (`None`: a chain of
        several nodes, which has no node of its own)."""
        metadata = self.metadata
        if metadata is None:
            return nodes
        metadata, found = X5Metadata.writable(metadata)
        report.source = found.source
        report.merge(found)
        node = None if index is None else nodes[index]
        if node is None or metadata.node is not node:
            # Not the node it was read from: everything is written.
            metadata = replace(metadata, snapshot={})
        raw = metadata.update_raw(
            X5Raw(self.header, node), image=self, on_loss=report
        )
        metadata.check_raw(raw, image=self, on_loss=report)
        node = raw.node
        if index is not None:
            nodes[index] = node
        return nodes
