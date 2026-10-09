__all__ = ["X5Transform", "X5TransformParser"]

from warnings import warn

import h5py
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Factory, Magic, replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.common.hdf5 import Hdf5ParserWriter
from brainhops.io.transformations.base import TransformationFormat

from ._blocks import node_to_transformation, transformation_to_nodes
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
    """A parser that reads and writes the raw content of a BIDS X5 file."""

    header: X5Header = Factory(X5Header, repr=False)
    """The root attributes and chains."""

    nodes: tx.List[X5Node] = Factory(list, repr=False)
    """Every transform of `/TransformGroup`, as stored."""

    chain: tx.Optional[int] = None
    """The index of the chain of `/TransformChain` that is read.

    See [`X5Transform.selection`][].
    """

    position: tx.Optional[int] = None
    """The index of the single node of `/TransformGroup` that is read.

    See [`X5Transform.selection`][].
    """

    file: tx.Optional[h5py.File] = None
    """The open HDF5 file, when it was read with `keep_open=True`."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file by its root `Format` attribute."""
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
        """Read an X5 transformation from an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        keep_open : bool, default=False
            Whether to keep the file open after reading.
        load : bool, default=True
            Whether to load fields into memory, rather than lazily on use.
        chain : int, optional
            The index of the chain of `/TransformChain` to read.
        position : int, optional
            The index of the single node of `/TransformGroup` to read.

        Raises
        ------
        ValueError
            If both `chain` and `position` are given.
        ParserContentError
            If `chain` or `position` is out of range.
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
        header, nodes = self.to_struct()
        return lambda h5file: write_x5(h5file, header, nodes)

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write into an empty HDF5 file."""
        self._h5_writer(**kwargs)(h5file)

    def to_struct(self) -> tx.Tuple[X5Header, tx.List[X5Node]]:
        """Return the header and nodes that encode this object, unchanged."""
        return self.header, list(self.nodes)

    def _close(self) -> None:
        if isinstance(self.file, h5py.File):
            self.file.close()

    def __del__(self) -> None:
        """Close the HDF5 file if it is still open."""
        self._close()


@register_format
class X5Transform(
    X5TransformParser,
    _xforms.Sequence,
    TransformationFormat,
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

    The raw content of the file, including each node's JSON metadata, `Domain`,
    `Inverse` and other attributes, is kept in `header` and `nodes`.

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
        """Decode node `index` into a transformation.

        The node is decoded once and the same object is returned afterwards,
        so that the writer can recognise it by identity.
        """
        blocks = self.__dict__.setdefault("_x5_blocks", {})
        if index not in blocks:
            blocks[index] = node_to_transformation(self.nodes[index])
        return blocks[index]

    @smartproperty(cache=True)
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain of transformations, in the order they are applied.

        The chain is decoded lazily from the nodes. Assigning it replaces the
        decoded chain, and the assigned chain is what the writer encodes.
        """
        return tuple(self.node_transformation(i) for i in self.selection)

    # --- to -----------------------------------------------------------

    def to_struct(self) -> tx.Tuple[X5Header, tx.List[X5Node]]:
        """Return the header and nodes that encode this transformation.

        Raises
        ------
        UnrepresentableTransformationError
            If an element cannot be encoded in X5.
        """
        if getattr(self, "_transformations", None) is None:
            return self.header, list(self.nodes)

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
        return header, nodes
