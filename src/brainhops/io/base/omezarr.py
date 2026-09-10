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
    Permutation,
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
    from abczarr.ome import v0_6rc0
else:
    try:
        from abczarr import ZarrNode, open
        from abczarr.ome import v0_6rc0
    except ImportError:
        open = None
        ZarrNode = None
        v0_6rc0 = None

_OmeZarrLike = _tx.Union[
    ZarrNode,
    str,
    PathLike,
]


class OmeZarrParser(FileParser, DataModelBase):
    group: _tx.Optional[ZarrNode] = None

    @property
    def _metadata(self) -> v0_6rc0.Multiscale:
        if self.group is None:
            return None
        ome = self.group.ome.to_version("0.6rc0")
        multiscales = ome.multiscales
        if not multiscales:
            return None
        return multiscales[0]

    @property
    def _axes(self) -> _tx.Optional[_tx.List[Axis]]:
        """The Axis objects parsed from this zarr's metadata."""
        metadata = self._metadata
        if metadata is None:
            return None
        return [self._convert_axis(a) for a in metadata["axes"]]

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
    def _convert_axis(axis_meta: v0_6rc0.Axis) -> Axis:
        """Build an Axis from one entry of axis metadata."""
        return Axis(
            name=axis_meta.name if axis_meta.name else None,
            type=axis_meta.type if axis_meta.type else None,
            unit=Unit(axis_meta.unit) if axis_meta.unit else None,
        )

    @classmethod
    def _transform_from_metadata(
        cls, metadata: v0_6rc0.Multiscale, level: _tx.Optional[int] = None
    ) -> _tx.List[Transformation]:
        """
        Build the Transformation for one resolution level of metadatas entry.
        """
        coordinateSystems = {}
        for system in metadata.coordinateSystems:
            coordinateSystems[system.name] = CoordinateSystem(
                name=system.name,
                axes=[cls._convert_axis(a) for a in system.axes],
            )

        # If level is none use the base file's transformations
        dataset = metadata.datasets[level] if level is not None else metadata

        raw_transforms = (
            dataset.coordinateTransformations
            if dataset.coordinateTransformations
            else []
        )

        pieces: _tx.List[Transformation] = [
            cls._json_to_transformation(t, coordinateSystems)
            for t in raw_transforms
        ]

        if not pieces:
            return [Identity()]
        return pieces

    @staticmethod
    def get_system(systems: dict, space: v0_6rc0.Space):
        if space:
            return systems.get(space.name, None)
        return None

    @classmethod
    def _json_to_transformation(
        cls, transformation: v0_6rc0.CoordinateSystem, systems: dict
    ) -> Transformation:
        kind = transformation.type
        if kind == "scale":
            return Scaling(
                scale=np.asarray(transformation.scale),
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        elif kind == "translation":
            return Translation(
                translation=np.asarray(transformation.translation),
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        elif kind == "mapAxis":
            return Permutation(
                permutation=np.asarray(transformation.mapAxis),
                input=systems.get(transformation.input.name, None),
                output=cls.get_system(systems, transformation.output),
            )

        elif kind == "affine":
            return Affine(
                matrix=np.asarray(transformation.affine),
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        elif kind == "rotation":
            return Affine(
                matrix=np.asarray(transformation.rotation),
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        elif kind == "sequence":
            return Sequence(
                transformations=[
                    cls._json_to_transformation(t, systems)
                    for t in transformation.transformations
                ],
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        elif kind == "displacements":
            raise NotImplementedError("displacements not implemented yet")
        elif kind == "coordinates":
            raise NotImplementedError("coordinates not implemented yet")
        elif kind == "bijection":
            return Bijection(
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
                forward=cls._json_to_transformation(transformation["forward"]),
                backward=cls._json_to_transformation(
                    transformation["inverse"]
                ),
            )
        elif kind == "projectAxis":
            raise NotImplementedError("projectAxis not implemented yet")
        elif kind == "byDimension":
            raise NotImplementedError("byDimension not implemented yet")
        elif kind == "identity":
            return Identity(
                input=cls.get_system(systems, transformation.input),
                output=cls.get_system(systems, transformation.output),
            )
        else:
            raise NotImplementedError("Unsupported transformation")
