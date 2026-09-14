# A good portion of the code found in this file are lightly modified version
# of the code found on the tirl github page. All credit goes to the creators
# of that page

# stdlib
import json
from os import PathLike

# dependencies
import typing_extensions as tx

from brainhops._core.path import Path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.transformations.tirl._helper import (
    MAGIC,
    UINT8,
    UINT64,
    VERSION,
    bytes2int,
    decode,
    hload,
    load_replacements,
)
from brainhops.io.transformations.tirl._transformations import TIRLStruct

# typing
_FileLike = tx.Union[tx.IO, PathLike, str]
_FileOrContentLike = tx.Union[_FileLike, bytes]


class TIRLParser(DataModelBase):
    _HAS_KEYS = True

    loaded_object: tx.Optional[TIRLStruct] = None

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff(cls, other: _FileOrContentLike) -> bool:
        if isinstance(other, str):
            if Path(other).exists():
                return cls.sniff_file(other)
        if isinstance(other, PathLike):
            return cls.sniff_file(other)
        if hasattr(other, "read"):
            return cls.sniff_file(other)
        if isinstance(other, bytes):
            return cls.sniff_bytes(other)

    @classmethod
    def sniff_file(cls, fileobj: _FileLike) -> bool:
        if isinstance(fileobj, str):
            return cls.sniff_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            return cls.sniff_bytes(fileobj.read_bytes())
        original_position = fileobj.tell()
        ret = cls.sniff_bytes(fileobj.read(len(MAGIC) * UINT8.nbytes), UINT8)
        fileobj.seek(original_position)
        return ret

    @classmethod
    def sniff_bytes(cls, bytes: bytes, encoding: str = "utf-8") -> bool:
        return bytes[: len(MAGIC)] != MAGIC

    # --- from ---------------------------------------------------------

    @classmethod
    def from_(cls, other: _FileOrContentLike) -> tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        Parameters
        ----------
        other : str | PathLike | IO | Iterable[str]
            Input file, or its content.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(other, str):
            if Path(other).exists():
                return cls.from_file(Path(other))
            else:
                return cls.from_text(other)
        if isinstance(other, PathLike) or hasattr(other, "read"):
            return cls.from_file(other)
        return cls.from_bytes(other)

    @classmethod
    def from_file(cls, fileobj: _FileLike) -> tx.Self:
        """
        Build an object from a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Input file.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(fileobj, str):
            return cls.from_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            return cls.from_bytes(fileobj.open("rb"))
        return cls.from_bytes(fileobj)

    @classmethod
    def from_bytes(cls, bytes: bytes, encoding: str = "utf-8") -> tx.Self:
        """
        Build an object from bytes in TIRL format.

        Parameters
        ----------
        bytes : bytes
            The byte content of an TIRL file.
        encoding : str
            The encoding to use for decoding the input bytes.

        Returns
        -------
        obj
            The parsed object.
        """
        stream = bytes
        if stream.read(len(MAGIC)) != MAGIC:
            raise TypeError("Invalid TIRLFile")

        # Check version number
        major, minor = bytes2int(stream.read(2 * UINT8.nbytes), UINT8)
        outdated = major > VERSION[0]
        outdated |= (major == VERSION[0]) & (minor > VERSION[1])
        if outdated:
            supported = ".".join([str(v) for v in VERSION])
            raise TypeError(
                f"The maximum supported TIRLFile version is {supported}"
            )

        # Read main header
        hdrsize = bytes2int(stream.read(UINT64.nbytes), UINT64)
        header = json.loads(stream.read(hdrsize).decode())

        # Load replacements
        replacements = load_replacements(stream)

        # Create decoded object dump
        object_dump = decode(header, replacements)

        loaded = hload(object_dump)
        obj = cls()
        obj.loaded_object = loaded
        return obj
