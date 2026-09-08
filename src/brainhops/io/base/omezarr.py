import os
from os import PathLike

import numpy as np
import typing_extensions as _tx

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
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
        cls, metadata: dict, level: _tx.Optional[int] = None
    ) -> _tx.List[Transformation]:
        """
        Build the Transformation for one resolution level of metadatas entry.
        """
        axes = [cls._axis_from_ngff(a) for a in metadata["axes"]]
        voxel_space = CoordinateSystem(axes=axes)
        world_space = CoordinateSystem(axes=axes)

        dataset = (
            metadata["datasets"][level] if level is not None else metadata
        )
        # a metadata-wide transform (applied to every level) may also
        # be declared alongside each per-dataset one; apply it first.
        raw_transforms = list(dataset.get("coordinateTransformations", []))

        pieces: _tx.List[Transformation] = [
            cls._json_to_transformation(t, voxel_space, world_space)
            for t in raw_transforms
        ]

        if not pieces:
            return [Identity(input=voxel_space, output=world_space)]
        return pieces

    @classmethod
    def _json_to_transformation(
        cls,
        transformation: dict,
        voxel_space: CoordinateSystem,
        world_space: CoordinateSystem,
    ) -> Transformation:
        kind = transformation.get("type")
        if kind == "scale":
            return Scaling(
                scale=np.asarray(transformation["scale"], dtype=float),
                input=voxel_space,
                output=world_space,
            )
        elif kind == "translation":
            return Translation(
                translation=np.asarray(
                    transformation["translation"], dtype=float
                ),
                input=voxel_space,
                output=world_space,
            )

        elif kind == "affine":
            return Affine(
                matrix=np.asarray(transformation["affine"], dtype=float),
                input=voxel_space,
                output=world_space,
            )
        elif kind == "rotation":
            return Affine(
                matrix=np.asarray(transformation["rotation"], dtype=float),
                input=voxel_space,
                output=world_space,
            )
        elif kind == "sequence":
            return (
                Sequence(
                    transformations=[
                        cls._json_to_transformation(
                            t, voxel_space, voxel_space
                        )
                        for t in transformation["transformations"][:-1]
                    ]
                    + [
                        cls._json_to_transformation(
                            transformation["transformation"][-1],
                            voxel_space,
                            world_space,
                        )
                    ],
                    input=voxel_space,
                    output=world_space,
                )
                if len(transformation["transformations"]) > 0
                else Identity(input=voxel_space, output=world_space)
            )
        elif kind == "displacements":
            raise NotImplementedError(
                "need to figure out how to do this without circular imports"
            )
        elif kind == "coordinates":
            raise NotImplementedError(
                "need to figure out how to do this without circular imports"
            )
        elif kind == "bijection":
            return Bijection(
                input=voxel_space,
                output=world_space,
                forward=cls._json_to_transformation(transformation["forward"]),
                backward=cls._json_to_transformation(
                    transformation["inverse"]
                ),
            )
        elif kind == "byDimension":
            # TODO: Need to use the coordinate systems provided by byDimension
            return (
                Sequence(
                    transformations=[
                        cls._json_to_transformation(
                            t, voxel_space, voxel_space
                        )
                        for t in transformation["transformations"][:-1]
                    ]
                    + [
                        cls._json_to_transformation(
                            transformation["transformation"][-1],
                            voxel_space,
                            world_space,
                        )
                    ],
                    input=voxel_space,
                    output=world_space,
                )
                if len(transformation["transformations"]) > 0
                else Identity(input=voxel_space, output=world_space)
            )
        elif kind == "identity":
            return Identity(input=voxel_space, output=world_space)
        else:
            raise NotImplementedError(
                "Unsupported NGFF coordinateTransformation type: "
                f"{kind!r}. Only 'scale' and 'translation' are "
                "currently handled."
            )
