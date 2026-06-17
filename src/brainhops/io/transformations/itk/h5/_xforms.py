# stdlib
from os import PathLike

# externals
import numpy as np
import typing_extensions as _tx


# internals
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.itk._utils import LPSToVoxel, VoxelToLPS
from brainhops.io.transformations.itk.h5._struct import H5TransformStruct
from pathlib import Path as LocalPath


_H5TransformLike = _tx.Union[
    H5TransformStruct,
    str,
    PathLike
]


class H5TransformBasedTransformation(_xforms.Transformation):

    txtTransformation: _tx.Optional[H5TransformStruct] = None

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
    def image(self) -> _tx.Optional[H5TransformStruct]:
        """The H5Transform image from which the transformation was derived."""
        return getattr(self, "_image", None)

    @classmethod
    def from_(cls, other: _H5TransformLike) -> _tx.Self:
        """Create a H5TransformVoxelToLPS transformation from an H5Transform image."""
        return cls.from_txt(other)

    @classmethod
    def from_file(cls, fileobj: _tx.Union[str, PathLike]) -> _tx.Self:
        return cls.from_txt(H5TransformStruct.from_file(str(fileobj)))

    @classmethod
    def from_txt(cls, txt_obj: _H5TransformLike) -> _tx.Self:
        if isinstance(txt_obj, H5TransformStruct):
            return cls(txtTransformation=txt_obj)
        return cls.from_txt(H5TransformStruct.from_(str(txt_obj)))


class H5TransformVoxelToLPS(VoxelToLPS, H5TransformBasedTransformation):

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


class H5TransformLPSToVoxel(LPSToVoxel, H5TransformBasedTransformation):

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
