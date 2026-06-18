__all__ = ["TxtTransformStruct"]


import typing_extensions as _tx

# _ext
from brainhops.io.transformations.itk._utils import TransformBlock

# internals
from ._parser import TxtTransformParser


class TxtTransformStruct(TxtTransformParser):
    """
    In-memory representation of an ITK TxtTransformIO file.

    Parsing and writing are implemented in `TxtTransformParser`.

    Example
    -------
    ```
    #Insight Transform File V1.0

    #Transform 0
    TransformType: Euler2DTransform_double_2_2
    TransformParameters: 0.2 10 -5
    TransformFixedParameters: 128 128
    ```
    """

    transform_blocks: _tx.List[TransformBlock] = []

    @property
    def n_transforms(self) -> int:
        return len(self.transform_blocks)

    @property
    def is_composite(self) -> bool:
        return self.n_transforms > 1

    @property
    def first_transform(self) -> _tx.Optional[TransformBlock]:

        if not self.transform_blocks:
            return None

        return self.transform_blocks[0]

    @property
    def last_transform(self) -> _tx.Optional[TransformBlock]:

        if not self.transform_blocks:
            return None

        return self.transform_blocks[-1]

    def append_transform(
        self,
        transform: TransformBlock,
    ):

        self.transform_blocks = (
            *self.transform_blocks,
            transform,
        )

    def new_transform(
        self,
        transform: str,
        parameters=(),
        fixed_parameters=(),
    ):

        obj = TransformBlock()
        obj.TransformType = transform
        obj.TransformParameters = parameters
        obj.TransformFixedParameters = fixed_parameters

        self.append_transform(
            obj
        )
