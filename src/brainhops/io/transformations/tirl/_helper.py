# A good portion of the code found in this file are lightly modified version
# of the code found on the tirl github page. All credit goes to the
# creators of that page

# stdlib
import json
from collections import namedtuple
from itertools import count
from pydoc import locate
from warnings import warn

# dependencies
import numpy as np
import numpy.lib.format as nplib
import typing_extensions as _tx

# internals
from brainhops.io.transformations.tirl._transformations import TIRLStruct

ctype = namedtuple("ctype", ["nbytes", "byteorder", "signed"])

MAGIC = b"TIRLFile"
UINT8 = ctype(1, "little", False)
UINT64 = ctype(8, "little", False)
ARRAY_TAG = b"NDArray"
BYTES_TAG = b"Bytes"
MMAP_TAG = b"MemoryMap"
LH_TAG = b"LookupHeader"
BUFSIZE = 100 * 1024 ** 2  # 100 MB
UNIQUE_INDEX = count(0, 1)
VERSION = [3, 7]

_IO = _tx.IO[bytes]


# ----------------------------------------------------------------------
#   Binary helpers
# ----------------------------------------------------------------------

def bytes2int(data: bytes, int_type: ctype) -> _tx.Union[int, tuple[int, ...]]:
    """
    Converts a byte buffer into one or more integers.

    Reads the buffer in chunks of ``int_type.nbytes``, interpreting each
    chunk according to the specified byte order and signedness. Returns a
    single int if the buffer holds exactly one value, otherwise a tuple.
    """
    itemsize, byteorder, signed = int_type
    assert len(data) % itemsize == 0
    result = [
        int.from_bytes(data[start:start + itemsize], byteorder, signed=signed)
        for start in range(0, len(data), itemsize)
    ]
    return result[0] if len(result) == 1 else tuple(result)


def istagged(item: object, tag: str) -> bool:
    """
    Returns True if *item* is an XML-style tagged string,
    e.g. ``<tag>value</tag>``.
    """
    if isinstance(item, str):
        return item.startswith(f"<{tag}>") and item.endswith(f"</{tag}>")
    return False


def getval(item: object, tag: str) -> object:
    """
    Strips the XML-style tag from a tagged string and returns the inner value.
    """
    if isinstance(item, str):
        return item[len(f"<{tag}>"):-len(f"</{tag}>")]
    return item


# ----------------------------------------------------------------------
#   Dict / list comparison (ndarray-safe)
# ----------------------------------------------------------------------

def dictcmp(a: dict, b: dict) -> bool:
    """
    Recursively compares two dicts whose values may include ndarrays.
    Plain ``==`` would raise on arrays due to ambiguous truth values.
    """
    assert isinstance(a, dict) and isinstance(b, dict), \
        "Dict inputs are required for comparison."
    a_items = sorted(a.items(), key=lambda e: e[0])
    b_items = sorted(b.items(), key=lambda e: e[0])
    results = []
    for (_, v1), (_, v2) in zip(a_items, b_items):
        if type(v1) is not type(v2):
            return False
        if isinstance(v1, dict):
            results.append(dictcmp(v1, v2))
        elif isinstance(v1, (tuple, list)):
            results.append(lstcmp(v1, v2))
        elif isinstance(v1, np.ndarray):
            equal = v1.size == v2.size
            if equal and v1.size > 0:
                equal = bool(np.all(v1 == v2))
            results.append(equal)
        else:
            results.append(v1 == v2)
    return all(results)


def lstcmp(a: _tx.Union[list, tuple], b: _tx.Union[list, tuple]) -> bool:
    """
    Recursively compares two lists or tuples whose elements may
    include ndarrays.
    """
    assert isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)), \
        "List/tuple inputs are required for comparison."
    results = []
    for v1, v2 in zip(a, b):
        if type(v1) is not type(v2):
            return False
        if isinstance(v1, dict):
            results.append(dictcmp(v1, v2))
        elif isinstance(v1, (tuple, list)):
            results.append(lstcmp(v1, v2))
        elif isinstance(v1, np.ndarray):
            equal = v1.size == v2.size
            if equal and v1.size > 0:
                equal = bool(np.all(v1 == v2))
            results.append(equal)
        else:
            results.append(v1 == v2)
    return all(results)


# ----------------------------------------------------------------------
#   Block loaders
# ----------------------------------------------------------------------

def load_array(f: _IO) -> np.memmap:
    """
    Loads an ndarray block from the file, returning a memory-mapped handle.
    """
    # TODO: consider returning a dask array instead of np.memmap
    _blocksize = bytes2int(f.read(UINT64.nbytes), UINT64)
    version = nplib.read_magic(f)
    if version == (1, 0):
        shape, fortran_order, dtype = nplib.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran_order, dtype = nplib.read_array_header_2_0(f)
    else:
        raise ValueError(f"Invalid ndarray header version: {version}")
    order = "F" if fortran_order else "C"
    try:
        return np.memmap(f.name, dtype=dtype, mode="r+",
                         offset=f.tell(), shape=shape, order=order)
    except PermissionError:
        warn("Data object is read-only.", stacklevel=1)
        return np.memmap(f.name, dtype=dtype, mode="r",
                         offset=f.tell(), shape=shape, order=order)


def load_bytes(f: _IO) -> bytes:
    """Loads a raw bytes block from the file."""
    blocksize = bytes2int(f.read(UINT64.nbytes), UINT64)
    return f.read(blocksize)


def load_memmap(f: _IO, mode: str = "r+") -> np.memmap:
    """
    Returns a memory-mapped array from the file.
    Falls back to read-only if the file is not writable.
    """
    js_descr = f.read(bytes2int(f.read(UINT64.nbytes), UINT64)).decode()
    shape, dtype, order = json.loads(js_descr)
    order = "C" if order else "F"
    try:
        return np.memmap(f.name, dtype=np.dtype(dtype), mode=mode,
                         offset=f.tell(), shape=tuple(shape), order=order)
    except PermissionError:
        warn("Data array is read-only.", stacklevel=1)
        return np.memmap(f.name, dtype=np.dtype(dtype), mode="r",
                         offset=f.tell(), shape=tuple(shape), order=order)


def load_replacements(f: _IO) -> dict:
    """
    Reads the LookupHeader and all replacement blocks from a TIRLFile.

    The LookupHeader encodes a sorted key list and a matching table of byte
    offsets. Each offset points to a tagged block (NDArray, MemoryMap, or
    Bytes) that replaces a placeholder in the JSON header.
    """
    replacements: dict = {}

    if f.read(len(LH_TAG)) != LH_TAG:
        raise IndexError("Invalid LookupHeader in TIRLFile.")

    js_sorted_keys = f.read(bytes2int(f.read(UINT64.nbytes), UINT64)).decode()
    sorted_keys = json.loads(js_sorted_keys)
    offsets = bytes2int(f.read(len(sorted_keys) * UINT64.nbytes), UINT64)
    offsets = (offsets,) if not hasattr(offsets, "__iter__") else offsets

    for key, offset in zip(sorted_keys, offsets):
        f.seek(offset)

        # Read the null-terminated block label
        label_bytes = []
        while (ch := f.read(1)) != b"\0":
            label_bytes.append(ch)
        label = b"".join(label_bytes)

        if label == MMAP_TAG:
            item = load_memmap(f)
        elif label == ARRAY_TAG:
            item = load_array(f)
        elif label == BYTES_TAG:
            item = load_bytes(f)
        else:
            raise NotImplementedError(
                f"Unknown replacement block label: {label}")

        if isinstance(key, list):
            key = tuple(key)
        replacements[key] = item

    return replacements


# ----------------------------------------------------------------------
#   Decode / hload
# ----------------------------------------------------------------------

def decode(encoded_dump: _tx.Union[dict, list, tuple, object],
           objects: dict) -> object:
    """
    Recursively restores a TIRL object dump into a Python data structure.

    Handles four special cases found in serialised dumps:
    - ``<complex>...</complex>`` tagged strings → Python complex numbers
    - ``<class>...</class>`` tagged strings     → class objects (via pydoc.locate)
    - ``<obj>...</obj>`` tagged strings         → back-references into *objects*
    - Dicts with both ``"type"`` and ``"id"``   → registered in *objects* for
      later back-references, with collision detection via a unique index
    """  # noqa: E501
    assert isinstance(objects, dict)

    if isinstance(encoded_dump, dict):
        restored: _tx.Any = {}
        iterator = tuple((k, encoded_dump[k])
                         for k in sorted(encoded_dump.keys()))
    elif hasattr(encoded_dump, "__iter__"):
        restored = [None] * len(encoded_dump)
        iterator = enumerate(encoded_dump)
    else:
        if not objects:
            return encoded_dump
        raise AssertionError("Invalid TIRLObject dump encoding.")

    for key, item in iterator:
        if istagged(item, "complex"):
            restored[key] = complex(getval(item, "complex"))

        elif istagged(item, "class"):
            restored[key] = locate(getval(item, "class"))

        elif istagged(item, "obj"):
            objid = getval(item, "obj")
            # Backwards compatibility: legacy tuple
            # IDs stored as repr strings (<3.0)
            if objid.startswith("(") and objid.endswith(")"):
                objid = eval(objid)
                assert isinstance(objid, tuple)
            restored[key] = objects[objid]

        elif isinstance(item, dict):
            decoded = decode(item, objects)
            if decoded.get("type") and decoded.get("id"):
                raw_id = decoded["id"]
                objid = (raw_id + "_src") if isinstance(raw_id, str) \
                    else (tuple(raw_id) + ("src",))
                if objid not in objects:
                    objects[objid] = decoded
                elif not dictcmp(objects[objid], decoded):
                    objid = "reassigned-" + str(next(UNIQUE_INDEX))
                    decoded["id"] = objid
                    objects[objid] = decoded
                restored[key] = objects[objid]
            else:
                restored[key] = decoded

        elif isinstance(item, list):
            restored[key] = decode(item, objects)

        elif isinstance(item, tuple):
            restored[key] = tuple(decode(list(item), objects))

        else:
            restored[key] = item

    if isinstance(encoded_dump, tuple):
        restored = tuple(restored)

    return restored


def hload(node: object, objects: _tx.Optional[dict] = None) -> object:
    """
    Recursively instantiates TIRLStruct objects from a decoded object dump.

    Walks the node tree depth-first. Any dict with both ``"type"`` and
    ``"id"`` fields is constructed as a ``TIRLStruct`` (or the appropriate
    subclass via its registry) and cached in *objects* to avoid duplication.
    """
    if objects is None:
        objects = {}

    if isinstance(node, dict):
        for key, item in node.items():
            res = hload(item, objects)
            if isinstance(res, TIRLStruct):
                node[key] = res

        if "type" in node and "id" in node:
            raw_id = node["id"]
            if isinstance(raw_id, str):
                objid = node["type"] + raw_id + node.get("signature", "")
            else:
                objid = tuple(raw_id)

            if objid not in objects:
                node["type"] = node["type"].split(".")[-1]
                try:
                    obj = TIRLStruct(**node)
                except TypeError:
                    if node["type"] == "TImage":
                        obj = TIRLStruct(**node)
                    obj = node
                objects[objid] = obj

            return objects[objid]

    elif isinstance(node, list):
        for key, item in enumerate(node):
            res = hload(item, objects)
            if isinstance(res, TIRLStruct):
                node[key] = res
        return node

    else:
        return node
