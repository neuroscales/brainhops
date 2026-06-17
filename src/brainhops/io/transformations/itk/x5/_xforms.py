# stdlib
from os import PathLike

# externals
import numpy as np
import typing_extensions as _tx


# internals
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.itk._utils import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.itk.x5._struct import X5TransformStruct
from pathlib import Path as LocalPath


_X5TransformLike = _tx.Union[
    X5TransformStruct,
    str,
    PathLike
]


class X5TransformBasedTransformation(_xforms.Transformation):

    x5Transformation: _tx.Optional[X5TransformStruct] = None

    def new_transform(
        self,
        transform: str,
        parameters=(),
        fixed_parameters=(),
    ):
        return self.x5Transformation.new_transform(
            transform=transform, parameters=parameters, fixed_parameters=fixed_parameters)

    def to_file(self, fileobj: _tx.Union[_tx.IO, PathLike, str]) -> None:
        self.x5Transformation.to_file(fileobj)

    @property
    def image(self) -> _tx.Optional[X5TransformStruct]:
        """The X5Transform image from which the transformation was derived."""
        return getattr(self, "_image", None)

    @classmethod
    def from_(cls, other: _X5TransformLike) -> _tx.Self:
        """Create a X5TransformVoxelToLPS transformation from an X5Transform image."""
        return cls.from_x5(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        return cls.from_x5(X5TransformStruct.from_file(str(fileobj)))

    @classmethod
    def from_x5(cls, x5_obj: _X5TransformLike) -> _tx.Self:
        if isinstance(x5_obj, X5TransformStruct):
            return cls(x5Transformation=x5_obj)
        return cls.from_x5(X5TransformStruct.from_(str(x5_obj)))


class X5TransformVoxelToLPS(VoxelToLPS, X5TransformBasedTransformation):

    @property
    def transformations(self) -> _tx.List[_xforms.Transformation]:
        tf = []

        for block in self.x5Transformation.transform_blocks:

            # ---- displacement case ----
            if getattr(block, "is_displacement", False):
                voxel2world, disp = block.to_displacement()
                world2voxel = np.linalg.inv(voxel2world)

                tf.append(_xforms.Affine(matrix=world2voxel))
                tf.append(_xforms.DisplacementField(field=disp))
                tf.append(_xforms.Affine(matrix=voxel2world))

            # ---- affine case ----
            else:
                tf.append(_xforms.Affine(matrix=block.to_affine()))

        return tf

    @transformations.setter
    def transformations(self, value):
        self.transformation_blocks = value

    def inverse(self) -> LPSToVoxel:
        return super().inverse().to(LPSToVoxel)


class X5TransformLPSToVoxel(LPSToVoxel, X5TransformBasedTransformation):

    @property
    def transformations(self) -> _tx.List[_xforms.Transformation]:
        tf = []

        for block in self.x5Transformation.transform_blocks:

            # ---- displacement case ----
            if getattr(block, "is_displacement", False):
                voxel2world, disp = block.to_displacement()
                world2voxel = np.linalg.inv(voxel2world)

                tf.append(_xforms.Affine(matrix=world2voxel))
                tf.append(_xforms.DisplacementField(field=disp))
                tf.append(_xforms.Affine(matrix=voxel2world))

            # ---- affine case ----
            else:
                tf.append(_xforms.Affine(matrix=block.to_affine()))

        return tf

    @transformations.setter
    def transformations(self, value):
        self.transformation_blocks = value

    def inverse(self) -> VoxelToLPS:
        return super().inverse().to(VoxelToLPS)
