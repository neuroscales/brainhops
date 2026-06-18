__all__ = ["X5TransformStruct"]

# externals
import typing_extensions as _tx

# _ext
from brainhops.io.transformations.itk._utils import TransformBlock

# internals
from brainhops.io.transformations.itk.h5._parser import H5TransformParser


class X5TransformStruct(H5TransformParser):
    """
    In-memory representation of an ITK X5TransformIO file.

    Parsing and writing are implemented in `X5TransformParser`.
    """

    transform_blocks: _tx.List[TransformBlock] = []
    # TODO create a system automatically parse to and from from file
    from_space: _tx.Optional[str] = None
    to_space: _tx.Optional[str] = None
    mode: _tx.Optional[str] = None
    types: _tx.List[str] = []
    multiplexed: _tx.List[bool] = []
    invertible: bool = True

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
