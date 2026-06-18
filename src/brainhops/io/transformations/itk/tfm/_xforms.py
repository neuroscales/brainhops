# stdlib
from os import PathLike

# externals
import numpy as np
import typing_extensions as _tx


# internals
from brainhops.datamodel import transformations as _xforms
from pathlib import Path as LocalPath

from brainhops.io.transformations.itk._utils import LPSToLPS, LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.itk.tfm._struct import TxtTransformStruct


_TxtTransformLike = _tx.Union[
    TxtTransformStruct,
    str,
    PathLike
]


class TxtTransformBasedTransformation(_xforms.Transformation):

    txtTransformation: _tx.Optional[TxtTransformStruct] = None

    def new_transform(
        self,
        transform: str,
        parameters=(),
        fixed_parameters=(),
    ):
        return self.txtTransformation.new_transform(
            transform=transform, parameters=parameters, fixed_parameters=fixed_parameters)

    def to_file(self, fileobj: _tx.Union[_tx.IO, PathLike, str]) -> None:
        text = self.txtTransformation.to_text()
        if isinstance(fileobj, str):
            fileobj = LocalPath(fileobj)
        if isinstance(fileobj, PathLike):
            LocalPath(fileobj).write_text(text)
            return
        fileobj.write(text)

    @property
    def image(self) -> _tx.Optional[TxtTransformStruct]:
        """The TxtTransform image from which the transformation was derived."""
        return getattr(self, "_image", None)

    @classmethod
    def from_(cls, other: _TxtTransformLike) -> _tx.Self:
        """Create a TxtTransformVoxelToLPS transformation from an TxtTransform image."""
        return cls.from_txt(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        return cls.from_txt(TxtTransformStruct.from_file(str(fileobj)))

    @classmethod
    def from_txt(cls, txt_obj: _TxtTransformLike) -> _tx.Self:
        if isinstance(txt_obj, TxtTransformStruct):
            return cls(txtTransformation=txt_obj)
        return cls.from_txt(TxtTransformStruct.from_(str(txt_obj)))


class TxtTransformVoxelToLPS(VoxelToLPS, TxtTransformBasedTransformation):

    @property
    def transformations(self) -> _tx.List[_xforms.Transformation]:
        tf = []

        for block in self.txtTransformation.transform_blocks:

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


class TxtTransformLPSToVoxel(LPSToVoxel, TxtTransformBasedTransformation):

    @property
    def transformations(self) -> _tx.List[_xforms.Transformation]:
        tf = []

        for block in self.txtTransformation.transform_blocks:

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


class TxtTransformLPSToLPS(LPSToLPS, TxtTransformBasedTransformation):

    @property
    def transformations(self) -> _tx.List[_xforms.Transformation]:
        tf = []

        for block in self.txtTransformation.transform_blocks:

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

    def inverse(self) -> LPSToLPS:
        return super().inverse().to(LPSToLPS)
