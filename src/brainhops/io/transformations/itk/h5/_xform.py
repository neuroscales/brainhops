"""The chain of transformations stored in an ITK HDF5 (`.h5`) file."""

__all__ = ["H5Transform"]

import h5py
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.common.hdf5 import Hdf5Reader

from .._chains import (
    _check_position,
    _check_systems,
    _decode_chain,
    _encode_chain,
    _same_chain,
)
from .._xform import ItkTransform
from ._metadata import H5Metadata
from ._raw import H5Header, H5Raw


def _record(xform: "H5Transform") -> tx.Optional[H5Raw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


@register_format
class H5Transform(Hdf5Reader, ItkTransform):
    """Transformation stored in an ITK HDF5 (`.h5`) file.

    The transformation is the chain of blocks of one top-level transform
    of the file, in the order in which they apply (see
    [`brainhops.io.transformations.itk.h5`][]). The root and the blocks of
    the file are held as an [`H5Raw`][] record by the [`H5Metadata`][] of
    the transformation, and the chain is decoded from the record when it
    is first used. The parameters of the warps are lazy arrays, which read
    the file again by its name when they are used. The decoded chain is a
    list that is stored in the transformation, so a change made to it in
    place, such as an appended block, is kept.

    The record describes the file as it was read, or as the transformation
    was last consolidated. A chain that is assigned or changed, and an
    `input` or `output` that is set, are pending until [`consolidate`][]
    encodes them into a new record. brainhops does not write ITK `.h5`
    files, so the record is only held in memory.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")
    HINTS = ("h5",)

    _metadata: NoRepr[tx.Optional[H5Metadata]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds the blocks as a record.

    A transformation built from a chain has no metadata. Assigning other
    metadata drops the chain that the transformation holds, read or
    assigned, so that it is decoded from the new record.
    """

    @metadata.setter
    def metadata(self, value: tx.Optional[H5Metadata]) -> None:
        self._metadata = value
        self._transformations = None

    _position: KwOnly[tx.Optional[int]] = None

    position = smartproperty("position", invalidates=("transformations",))
    """The index of the top-level transform of the file that is read.

    A file that starts with a `CompositeTransform` header holds a single
    composite transform, and any other file holds one transform per block.
    By default, the first transform is read, with a warning if the file
    holds several. Assigning another index drops the chain that the
    transformation holds, read or assigned. The index only selects what is
    read, so it does not change the record.
    """

    @position.setter
    def position(self, value: tx.Optional[int]) -> None:
        self._position = value
        self._transformations = None

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The blocks of the transform, in the order in which they apply.

        The chain is decoded from the record on first access and stored in
        the transformation, so a change made to the list in place is kept.
        Its blocks are the blocks of the record, whose parameters are
        read-only. Assigning a list replaces the stored one. In both cases,
        the list is compared at [`consolidate`][] with the chain that the
        record decodes to, block by block. A transformation without a
        record has no chain.

        Raises
        ------
        ParserContentError
            If the record has no transform at `position`.
        """
        record = _record(self)
        chain = [] if record is None else _decode(record, self.position)
        self._transformations = chain
        return chain

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score an open HDF5 file as an ITK transform file.

        The file is scored by [`H5Metadata.sniff_h5`][].

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an ITK
            transform file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return H5Metadata.sniff_h5(h5file, error=error)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        h5file: h5py.File,
        load: bool = False,
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Read a transformation from an open ITK HDF5 file.

        The record is read by [`H5Metadata.from_h5`][], and the chain is
        decoded from it when it is first used.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        load : bool, default=False
            Whether to read the parameters of the warps into memory at
            once, rather than when they are used. The parameters of a file
            opened from a stream are always read at once.
        position : int, optional
            The index of the top-level transform to read: the composite
            when the file has one, otherwise one of its blocks. By default
            the first one is read, with a warning when there are several.
        **kwargs : Any
            Other fields of the transformation. Any other option is
            refused.

        Returns
        -------
        H5Transform
            The transformation.

        Raises
        ------
        ParserContentError
            If the file has no transform at `position`, or if a composite
            header is not its first block.
        """
        metadata = H5Metadata.from_h5(h5file, load=load)
        _check_position(metadata.raw.blocks, position)
        return cls(metadata=metadata, position=position, **kwargs)

    @classmethod
    def from_raw(cls, raw: H5Raw, **kwargs) -> tx.Self:
        """Build a transformation from an already read [`H5Raw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : H5Raw
            The record of the file.
        **kwargs : Any
            Field values, such as `position`, which go to the constructor.

        Returns
        -------
        H5Transform
            The transformation.

        Raises
        ------
        ParserContentError
            If the record has no transform at `position`.
        """
        _check_position(raw.blocks, kwargs.get("position"))
        return cls(metadata=H5Metadata.from_raw(raw), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`H5Raw`][] is read with [`from_raw`][], and any other value as
        the bases read it.

        Parameters
        ----------
        other : object
            A record, a file, its content, a transformation, a mapping, or
            an instance of a similar class.
        *args : Any
            Constructor arguments. A record and a file take keyword
            arguments only.
        **kwargs : Any
            Options of the reader, and field values otherwise.

        Returns
        -------
        H5Transform
            The new transformation.
        """
        if isinstance(other, H5Raw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def consolidate(self, *, inplace: bool = False) -> tx.Self:
        """Encode the pending chain and systems into a new record.

        A transformation that holds a record is left as it is when its
        chain was never read, or when it still holds the blocks that the
        record decodes to, in the same order. An `input` or `output` that
        was set must be the system that the chain maps from or to, since
        the file stores no system, and it is then dropped. The option
        `position` only selects what is read, so it does not change the
        record.

        Otherwise, a new record holds the root of the current record and
        the blocks of the chain: an ITK
        block as it is, and an affine between ITK spaces as an
        `AffineTransform` centered on the origin. A chain of several
        blocks is stored as a composite, as ITK writes it. A
        transformation without a record and without a chain gives an
        empty record. The new record is held by new metadata, and the
        option `position`, which indexed the old record, is reset.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.

        Returns
        -------
        H5Transform
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        UnrepresentableTransformationError
            If a transformation of the chain is neither an ITK block nor
            an affine between ITK spaces, or if a system that was set
            differs from the system of the chain.
        """
        record = _record(self)
        decoded = (
            [] if record is None else _decode(record, self.position, True)
        )
        chain = self._transformations
        _check_systems(self, decoded if chain is None else chain)
        if record is not None and (
            chain is None or _same_chain(chain, decoded)
        ):
            if not inplace:
                return replace(
                    self, transformations=None, input=None, output=None
                )
            self.transformations = None
            self.input = None
            self.output = None
            return self
        header = H5Header() if record is None else record.header
        record = H5Raw(header=header, blocks=_encode_chain(chain or []))
        if self.metadata is None:
            metadata = H5Metadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
        if not inplace:
            return replace(
                self,
                metadata=metadata,
                transformations=None,
                input=None,
                output=None,
                position=None,
            )
        self.metadata = metadata
        self.input = None
        self.output = None
        self.position = None
        return self


def _decode(
    record: H5Raw, position: tx.Optional[int], quiet: bool = False
) -> tx.List[_xforms.Transformation]:
    """Decode the chain of a record, as [`_decode_chain`][] does."""
    return _decode_chain(record.blocks, position, quiet=quiet)
