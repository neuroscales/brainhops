# stdlib
import typing_extensions as _tx
from os import PathLike
from pathlib import Path as LocalPath

import numpy as np
import h5py

from brainhops._ext.struct import Struct
from brainhops.io.transformations.itk._utils import TransformBlock

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

_FileLike = _tx.Union[_tx.IO, PathLike, str]
_HEADER = "#Insight Transform File V1.0"

# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------


class H5TransformParser(Struct):
    VERSION = _HEADER

    # ---------------------------------------------------------
    # Sniffing
    # ---------------------------------------------------------

    @classmethod
    def sniff(cls, other: _FileLike) -> bool:
        try:
            obj = cls.from_(other)
            return len(obj.transform_blocks) > 0
        except Exception:
            return False

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    @classmethod
    def from_(cls, other: _FileLike) -> _tx.Self:
        """
        Build an object from a file (path, file-like object).

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
            p = LocalPath(other)
            return cls.from_file(p)
        if isinstance(other, PathLike):
            return cls.from_file(other)
        if hasattr(other, "read"):
            return cls.from_file(other)
        raise TypeError("Unsupported input type for H5TransformParser")

    @classmethod
    def from_file(cls, fileobj: _FileLike) -> _tx.Self:
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
            fileobj = LocalPath(fileobj)

        obj = cls()

        with h5py.File(fileobj, "r") as f:

            # ---- detect real ITK structure ----
            if "TransformGroup" in f:
                root = f["TransformGroup"]
            else:
                root = f

            for key in sorted(root.keys()):
                grp = root[key]

                if not isinstance(grp, h5py.Group):
                    continue

                tb = TransformBlock(transform="")

                # ---- ITK naming ----
                if "TransformType" in grp:
                    val = grp["TransformType"][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    elif isinstance(val, np.ndarray):
                        val = val[0].decode()
                    tb.transform = val

                if "TransformParameters" in grp:
                    tb.parameters = list(grp["TransformParameters"][()])

                if "TransformFixedParameters" in grp:
                    tb.fixed_parameters = list(
                        grp["TransformFixedParameters"][()])

                # ---- fallback (your custom format) ----
                if not tb.transform:
                    tb.transform = grp.attrs.get("transform", "")

                if not tb.parameters and "parameters" in grp:
                    tb.parameters = list(grp["parameters"])

                if not tb.fixed_parameters and "fixed_parameters" in grp:
                    tb.fixed_parameters = list(grp["fixed_parameters"])

                # ---- extras ----
                for k, v in grp.attrs.items():
                    if k != "transform":
                        tb.extras[k] = v

                obj.transform_blocks.append(tb)

        return obj

    # ---------------------------------------------------------
    # Object
    # ---------------------------------------------------------

    def __init__(self):
        self.transform_blocks = []

    # ---------------------------------------------------------
    # Writing
    # ---------------------------------------------------------

    def to_file(self, fileobj: _FileLike) -> None:
        """
        Write the object to a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Output file.

        Returns
        -------
        None
        """
        if isinstance(fileobj, str):
            fileobj = LocalPath(fileobj)

        with h5py.File(fileobj, "w") as f:
            f.attrs["version"] = self.VERSION

            for i, transform in enumerate(self.transform_blocks):
                grp = f.create_group(f"Transform_{i}")

                grp.attrs["transform"] = transform.transform

                grp.create_dataset(
                    "parameters",
                    data=np.array(transform.parameters, dtype=float),
                )

                grp.create_dataset(
                    "fixed_parameters",
                    data=np.array(transform.fixed_parameters, dtype=float),
                )

                for k, v in transform.extras.items():
                    grp.attrs[k] = v
