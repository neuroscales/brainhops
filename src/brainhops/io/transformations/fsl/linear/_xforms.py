# stdlib
from numbers import Real
from os import PathLike

# externals
import typing_extensions as _tx


# internals
from brainhops.datamodel import transformations as _xforms
from pathlib import Path as LocalPath

from brainhops.datamodel.typing import npmatrix
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.fsl.linear._parser import FslLinearTransformParser


class FslLinearTransformBasedTransformation(_xforms.Affine, FslLinearTransformParser):
    ...


class FslLinearTransformVoxelToRAS(VoxelToRAS, FslLinearTransformBasedTransformation):

    def inverse(self) -> RASToVoxel:
        return super().inverse().to(RASToVoxel)


class FslLinearTransformRASToVoxel(RASToVoxel, FslLinearTransformBasedTransformation):

    def inverse(self) -> VoxelToRAS:
        return super().inverse().to(VoxelToRAS)
