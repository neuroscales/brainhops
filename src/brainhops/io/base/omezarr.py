import os
from os import PathLike

import numpy as np
import typing_extensions as _tx

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Identity,
    Scaling,
    Sequence,
    Transformation,
    Translation,
)
from brainhops.datamodel.units import Unit
from brainhops.io.base.parsers import FileParser

# optionals
if _tx.TYPE_CHECKING:
    from abczarr import ZarrNode, open
else:
    try:
        from abczarr import ZarrNode, open
    except ImportError:
        open = None
        ZarrNode = None

_OmeZarrLike = _tx.Union[
    ZarrNode,
    str,
    PathLike,
]


class OmeZarrParser(FileParser, DataModelBase):
    group: _tx.Optional[ZarrNode] = None

    @property
    def _metadata(self) -> _tx.Optional[dict]:
        if self.group is None:
            return None
        metadata = dict(self.group.attrs)
        # NGFF 0.5+ (zarr v3) nests everything under "ome"; earlier
        # versions (zarr v2, .zattrs) put `multiscales` at the top level.
        ome = metadata.get("ome", metadata)
        multiscales = ome.get("multiscales")
        if not multiscales:
            return None
        return multiscales[0]

    @property
    def _axes(self) -> _tx.Optional[_tx.List[Axis]]:
        """The Axis objects parsed from this zarr's metadata."""
        metadata = self._metadata
        if metadata is None:
            return None
        return [self._axis_from_ngff(a) for a in metadata["axes"]]

    @classmethod
    def from_(cls, other: _OmeZarrLike) -> _tx.Self:
        """Create an OmeZarrParser from a zarr group or a path to one."""
        cls._require_zarr()
        if isinstance(other, ZarrNode):
            return cls(group=other, mode="r")
        return cls(group=open(other, mode="r"))

    @classmethod
    def from_file(cls, path: _tx.Union[str, PathLike]) -> _tx.Self:
        """Create an OmeZarrParser from a zarr group or a path to one."""
        cls._require_zarr()
        return cls(group=open(path, mode="r"))

    @classmethod
    def sniff(cls, other: _OmeZarrLike) -> bool:
        """
        Check if other is already a ZarrNode or if other has a valid json file
        """
        cls._require_zarr()
        if isinstance(other, ZarrNode):
            return True
        return cls.sniff_file(other)

    @classmethod
    def sniff_file(cls, path: _tx.Union[str, PathLike]) -> bool:
        """
        Check if path has a valid zarr json file
        """
        cls._require_zarr()
        if isinstance(path, PathLike):
            path = str(path)
        path = path.removesuffix("/")
        zarray = path + "/.zarray"
        zattrs = path + "/.zattrs"
        zjson = path + "/zarr.json"
        if (
            os.path.exists(zarray)
            or os.path.exists(zattrs)
            or os.path.exists(zjson)
        ):
            return True
        return True

    # --- helpers -----------------------------------------------------

    @staticmethod
    def _require_zarr() -> None:
        if ZarrNode is None:
            raise ImportError(
                "The `zarr` package is required to read OME-Zarr images. "
                "Install it with `pip install zarr`."
            )

    @staticmethod
    def _axis_from_ngff(axis_meta: dict) -> Axis:
        """Build an Axis from one entry of axis metadata."""
        return Axis(
            name=axis_meta["name"],
            type=axis_meta.get("type"),
            unit=Unit(axis_meta.get("unit")),
        )

    @classmethod
    def _transform_from_metadata(
        cls, metadata: dict, level: int
    ) -> Transformation:
        """
        Build the Transformation for one resolution level of metadatas entry.
        """
        axes = [cls._axis_from_ngff(a) for a in metadata["axes"]]
        voxel_space = CoordinateSystem(axes=axes)
        world_space = CoordinateSystem(axes=axes)

        dataset = metadata["datasets"][level]
        # a metadata-wide transform (applied to every level) may also
        # be declared alongside each per-dataset one; apply it first.
        raw_transforms = list(
            metadata.get("coordinateTransformations", [])
        ) + list(dataset.get("coordinateTransformations", []))

        pieces: _tx.List[Transformation] = []
        for t in raw_transforms:
            kind = t.get("type")
            if kind == "scale":
                pieces.append(
                    Scaling(
                        scale=np.asarray(t["scale"], dtype=float),
                        input=voxel_space,
                        output=world_space,
                    )
                )
            elif kind == "translation":
                pieces.append(
                    Translation(
                        translation=np.asarray(t["translation"], dtype=float),
                        input=voxel_space,
                        output=world_space,
                    )
                )
            elif kind == "identity":
                continue
            else:
                raise NotImplementedError(
                    "Unsupported NGFF coordinateTransformation type: "
                    f"{kind!r}. Only 'scale' and 'translation' are "
                    "currently handled."
                )

        if not pieces:
            return Identity(input=voxel_space, output=world_space)
        if len(pieces) == 1:
            return pieces[0]
        return Sequence(
            transformations=pieces, input=voxel_space, output=world_space
        )
