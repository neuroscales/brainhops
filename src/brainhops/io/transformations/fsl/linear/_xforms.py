# internals
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.fsl.linear._parser import (
    FslLinearTransformParser,
)


class FslLinearTransformBasedTransformation(_xforms.Affine, FslLinearTransformParser):
    ...


class FslLinearTransformVoxelToRAS(VoxelToRAS, FslLinearTransformBasedTransformation):

    def inverse(self) -> RASToVoxel:
        return super().inverse().to(RASToVoxel)


class FslLinearTransformRASToVoxel(RASToVoxel, FslLinearTransformBasedTransformation):

    def inverse(self) -> VoxelToRAS:
        return super().inverse().to(VoxelToRAS)
