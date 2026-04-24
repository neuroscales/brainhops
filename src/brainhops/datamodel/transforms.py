import typing_extensions as _tx
import dask.array as da
import numpy as np
from scipy.ndimage import map_coordinates


from functools import partial

from brainhops.struct import ClassVar
from .struct import SpecializedStruct
from .systems import CoordinateSystem

ArrayLike = _tx.Union[np.ndarray, da.Array]


class Transform(SpecializedStruct):
    input: _tx.Optional[CoordinateSystem] = None
    output: _tx.Optional[CoordinateSystem] = None

    def sample_tensor(self, grid: ArrayLike, sampled_tensor: ArrayLike):
        """
        Sample a tensor at arbitrary grid coordinates using linear interpolation.
        """
        orig_grid = grid
        new_grid = grid.copy()
        slices = []

        for i in range(grid.shape[-1]):
            gmin = da.min(orig_grid[..., i])
            gmax = da.max(orig_grid[..., i])

            lo = max(0, int(np.floor(gmin)))
            hi = min(sampled_tensor.shape[i], int(np.ceil(gmax)) + 1)

            slices.append(slice(lo, hi))
            new_grid[..., i] -= lo

        new_sampled = sampled_tensor[tuple(slices)]

        r_values = [new_grid[..., i] for i in range(new_grid.shape[-1])]
        coords = np.vstack([
            r.ravel() for r in r_values
        ])

        if isinstance(new_sampled, da.Array):
            new_sampled = new_sampled.compute()
        if isinstance(new_grid, da.Array):
            new_grid = new_grid.compute()

        # Case 1: sampling grayscale tensor
        if new_sampled.ndim == new_grid.ndim - 1:
            warped = map_coordinates(
                new_sampled,
                coords,
                order=1,
                mode="nearest"
            )
            return warped.reshape(new_grid.shape[:-1])

        # Case 2: sampling displacement tensor or multichannel image
        elif new_sampled.ndim == new_grid.ndim:
            channels = new_sampled.shape[-1]
            warped = np.empty((*new_grid.shape[:-1], channels),
                              dtype=new_sampled.dtype)

            for c in range(channels):
                warped_c = map_coordinates(
                    new_sampled[..., c],
                    coords,
                    order=1,
                    mode="nearest"
                )
                warped[..., c] = warped_c.reshape(new_grid.shape[:-1])

            return warped

        else:
            raise ValueError(
                f"Grid Shape: {grid.shape}, Sampled Shape: {sampled_tensor.shape}. Grid and sampled tensor must have same shape or grid must have one less dimension")

    @_tx.overload
    def __call__(self, x: "CoordinatesField", compute: _tx.Literal[True]) -> "CoordinatesField":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.
        """
        ...

    @_tx.overload
    def __call__(self, x: "CoordinatesField", compute: _tx.Literal[False]) -> "Sequence":
        """
        Transform coordinates from the input coordinate system to the
        output coordinate system.

        This variant does not compute the transformed coordinates, but
        instead returns an object that holds the original coordinates
        and the transform, and can be computed later when needed.
        """
        ...

    @_tx.overload
    def __call__(self, x: "Transform", compute: _tx.Literal[True]) -> "Transform":
        """
        Compose this transform with another transform.

        The resulting transform will have the input coordinate system of
        the other transform and the output coordinate system of this
        transform.

        If the output coordinate system of the other transform does not
        match the input coordinate system of this transform, it will
        try to guess an intermediate adapter transform, and raise an
        error if it cannot find one.
        """
        ...

    @_tx.overload
    def __call__(self, x: "Transform", compute: _tx.Literal[False]) -> "Sequence":
        """
        Compose this transform with another transform, without computing
        the resulting transform, but instead returning a sequence of the
        two transforms that can be computed later when needed.
        """
        ...

    def __call__(self, x, compute: bool = False):
        if compute:
            if self.type == "displacement":
                g_vals = np.meshgrid(
                    *[np.arange(s) for s in self.field.shape[:-1]],
                    indexing="ij"
                )
                identity = np.ones(
                    (*self.field.shape[:-1], len(self.field.shape[-1])))
                for i in range(len(g_vals)):
                    identity[:, :, :, i] = g_vals[i]
                cords = (self.field + identity)
                return CoordinatesField(field=self.sample_tensor(cords, x.field),
                                        input=x.input,
                                        output=self.output)
            elif self.type == "coordinates":
                return CoordinatesField(field=self.sample_tensor(self.field, x.field),
                                        input=x.input,
                                        output=self.output)
            elif self.type == "sequence":
                return self.compute()(x, True)
            else:
                raise ValueError(
                    f"can not apply transformation of type: {self.type}")
        else:
            if self.type == "sequence":
                return Sequence(transforms=[x, *self.transforms], input=x.input, output=self.output).compute(False)
            else:
                return Sequence(transforms=[x, self], input=x.input, output=self.output).compute(False)

    def apply_transformation(self, x: "CoordinatesField") -> "CoordinatesField":
        """
        Apply x to an output coordinate field to generate an input coordinate field.
        """
        if self.type == "affine":
            return CoordinatesField(field=x.field @ np.array(self.matrix)[:-1, :-1].T + self.matrix[:-1, -1],
                                    input=self.input,
                                    output=x.output).compute(False)
        elif self.type == "linear":
            return CoordinatesField(field=x.field @ self.matrix.T,
                                    input=self.input,
                                    output=x.output)
        elif self.type == "displacement":
            return CoordinatesField(field=self.sample_tensor(x.field, self.field) + x.field,
                                    input=self.input,
                                    output=x.output)
        elif self.type == "coordinates":
            return CoordinatesField(field=self.sample_tensor(x.field, self.field),
                                    input=self.input,
                                    output=x.output)
        elif self.type == "scale":
            return CoordinatesField(field=x.field*self.scale,
                                    input=self.input,
                                    output=x.output)
        elif self.type == "translation":
            return CoordinatesField(field=x.field+self.translation,
                                    input=self.input,
                                    output=x.output)
        elif self.type == "permutation":
            return CoordinatesField(field=da.transpose(x.field, (*self.permutation, len(self.permutation))),
                                    input=self.input,
                                    output=x.output)
        elif self.type == "sequence":
            transform_output = x
            for transform in reversed(self.transforms):
                transform_output = transform.combine(transform_output)
            return transform_output

    @_tx.overload
    def to(self, type: _tx.Type["Linear"]) -> "Linear":
        """
        Convert linearish transformation into Linear transformation.

        raises ValueError if self is not linearish.
        """

    @_tx.overload
    def to(self, type: _tx.Type["Affine"]) -> "Affine":
        """
        Convert affineish transformation into Affine transformation.

        raises ValueError if self is not affineish.
        """

    def to(self, type: _tx.Type["Transform"]) -> "Transform":
        if type is Linear:
            if self.type == "linear":
                return self
            elif self.type == "scale":
                mat = np.zeros((len(self.scale), len(self.scale)))
                for i in range(len(self.scale)):
                    mat[i, i] = self.scale[i]
                return Linear(matrix=mat, input=self.input, output=self.output)
            elif self.type == "permutation":
                mat = np.zeros(
                    (len(self.permutation), len(self.permutation)))
                for i in range(len(self.permutation)):
                    mat[i, self.permutation[i]] = 1
                return Linear(matrix=mat, input=self.input, output=self.output)
            else:
                raise ValueError(
                    f"Transformation must be linear, scale, or permutation. Instead was: {self.type}")
        if type is Affine:
            if is_linearish(self):
                linear = self.to(Linear)
                mat = np.zeros(
                    (linear.matrix.shape[0]+1, linear.matrix.shape[1]+1))
                mat[:-1, :-1] = linear.matrix
                mat[-1, -1] = 1
                return Affine(matrix=mat, input=self.input, output=self.output)
            elif self.type == "translation":
                mat = np.identity(len(self.translation)+1)
                mat[:-1, -1] = self.translation
                return Affine(matrix=mat, input=self.input, output=self.output)
            elif self.type == "affine":
                return self
            else:
                raise ValueError(
                    f"transformation must be Affineish. Instead was: {self.type}")
        raise ValueError(f"type must be Affine or Linear. Instead was: {type}")

    @_tx.overload
    def compute(self, hard: _tx.Literal[False]) -> "Transform":
        """
        Combines Transforms of a sequence into a smaller sequence by combining Affineish values that are next to each other.
        If the resulting sequence would be 1 long instead return that transform.

        Raises ValueError if self is not a sequence
        """

    @_tx.overload
    def compute(self, hard: _tx.Literal[True]) -> "CoordinatesField":
        """
        Combines Transforms of a sequence into a CoordinatesField.

        Raises ValueError if self is not a sequence or does not end with a CoordinateField
        """

    def compute(self, hard: bool = True) -> "Transform":
        if not hard:
            if self.type == "sequence":
                transforms = []
                i = 0
                while i < len(self.transforms):
                    if is_affineish(self.transforms[i]):
                        affine_trans = self.transforms[i]
                        j = i + 1
                        while j < len(self.transforms) and is_affineish(self.transforms[j]):
                            affine_trans = _compose(
                                affine_trans, self.transforms[j])
                            j += 1
                        transforms.append(affine_trans)
                        i = j-1
                    elif self.transforms[i].type == "sequence":
                        softSequence = self.transforms[i].compute(False)
                        transforms = transforms + softSequence.transforms
                    else:
                        transforms.append(self.transforms[i])
                    i += 1
                if len(transforms) == 1:
                    return transforms[0]
                return Sequence(transforms=transforms, input=self.input, output=self.output)
            else:
                return self
        else:
            if self.type == "sequence":
                softSequence = self.compute(False)
                if softSequence.transforms[-1].type != "coordinates":
                    raise ValueError(
                        "right most transformation must be a coordinate feild")
                transform_output = softSequence.transforms[-1]
                for transform in reversed(softSequence.transforms[:-1]):
                    transform_output = transform.apply_transformation(
                        transform_output)
                return transform_output
            else:
                raise ValueError(
                    "can only compute on sequences")

    def __matmul__(self, other):
        return self(other, compute=False)


class Affine(Transform):
    matrix: _tx.Optional[ArrayLike] = None
    type: ClassVar[str] = "affine"


class Linear(Transform):
    matrix: _tx.Optional[ArrayLike] = None
    type: ClassVar[str] = "linear"


class DisplacementField(Transform):
    field: _tx.Optional[ArrayLike] = None
    type: ClassVar[str] = "displacement"


class CoordinatesField(Transform):
    field: _tx.Optional[ArrayLike] = None
    type: ClassVar[str] = "coordinates"


class CartesianCoordinatesField(CoordinatesField):
    def __init__(self, shape):
        self._shape = None
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self._update_field()

    def _update_field(self):
        self.field = da.meshgrid(
            *[da.arange(s) for s in self.shape],
            indexing="ij"
        )


class CartesianSliceField(CoordinatesField):
    def __init__(self, slices):
        self._slices = None
        self.slices = slices

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, value):
        self._slices = value
        self._update_field()

    def _update_field(self):
        if self._slices is None:
            self.field = None
            return

        idx_arrays = []
        for s in self._slices:
            start, stop, step = s.indices(s.stop)
            idx_arrays.append(np.arange(start, stop, step))

        grids = np.meshgrid(*idx_arrays, indexing='ij')
        self.field = np.stack(grids, axis=-1)


class Permutation(Transform):
    permutation: _tx.Optional[_tx.List[int]] = None
    type: ClassVar[str] = "permutation"


class Scale(Transform):
    scale: _tx.Optional[_tx.List[float]] = None
    type: ClassVar[str] = "scale"


class Translation(Transform):
    translation: _tx.Optional[_tx.List[float]] = None
    type: ClassVar[str] = "translation"


class Sequence(Transform):
    transforms: _tx.Optional[_tx.List[Transform]] = None
    type: ClassVar[str] = "sequence"


_COMPOSERS = {}
_COMPOSERS_FASTMAP = {}
_CONVERTERS = {}
_CONVERTERS_FASTMAP = {}


def _composer(func: _tx.Callable) -> _tx.Callable:
    """Decorator to register a function as a composer of two transforms."""
    types = tuple(_tx.get_type_hints(func).values())[:2]
    _COMPOSERS[types] = func
    _COMPOSERS_FASTMAP.clear()
    return func


def _distance(t1: type, t2: type, oriented: bool = True) -> int:
    """Compute the distance between two types in the class hierarchy."""
    # TODO: handle type hints (Union, Any)
    if t1 == t2:
        return 0
    if issubclass(t1, t2):
        return 1 + min(map(partial(_distance, t2=t2), t1.__bases__))
    if issubclass(t2, t1) and not oriented:
        return 1 + min(map(partial(_distance, t2=t1), t2.__bases__))
    return float("inf")


def _compose(x1: Transform, x2: Transform) -> Transform:
    """
    Dispatch the composition of two transforms to the appropriate 
    composer function.
    """
    t1, t2 = type(x1), type(x2)
    if (t1, t2) in _COMPOSERS_FASTMAP:
        return _COMPOSERS_FASTMAP[(t1, t2)](x1, x2)
    best_distance, best_func = float("inf"), None
    for (T1, T2), FUNC in _COMPOSERS.items():
        distance = _distance(t1, T1) + _distance(t2, T2)
        if distance < best_distance:
            best_distance, best_func = distance, FUNC
    if best_distance < float("inf"):
        _COMPOSERS_FASTMAP[(t1, t2)] = best_func
        return best_func(x1, x2)
    raise TypeError(f"No composer found for types: {t1}, {t2}")


@_composer
def _(t1: Sequence, t2: Transform) -> Affine:
    return Sequence(
        transforms=[t2] + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute(False)


@_composer
def _(t1: Transform, t2: Sequence) -> Sequence:
    return Sequence(
        transforms=list(t2.transforms) + [t1],
        input=t2.input,
        output=t1.output
    ).compute(False)


@_composer
def _(t1: Sequence, t2: Sequence) -> Sequence:
    return Sequence(
        transforms=list(t2.transforms) + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute(False)


@_composer
def _(t1: Affine, t2: Affine) -> Affine:
    return Affine(
        matrix=t1.matrix @ t2.matrix,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Linear, t2: Linear) -> Linear:
    return Linear(
        matrix=t1.matrix @ t2.matrix,
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Permutation, t2: Permutation) -> Permutation:
    return Permutation(
        permutation=[t1.permutation[i] for i in t2.permutation],
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Scale, t2: Scale) -> Scale:
    return Scale(
        scale=[s1 * s2 for s1, s2 in zip(t1.scale, t2.scale)],
        input=t2.input,
        output=t1.output
    )


@_composer
def _(t1: Translation, t2: Translation) -> Translation:
    return Translation(
        translation=[t1 + t2 for t1,
                     t2 in zip(t1.translation, t2.translation)],
        input=t2.input,
        output=t1.output
    )


def is_linearish(obj) -> bool:
    return isinstance(obj, (Linear, Scale, Permutation))


def is_affineish(obj) -> bool:
    return isinstance(obj, (Linear, Scale, Permutation, Affine, Translation))


@_composer
def _(t1: Transform, t2: Transform) -> Transform:
    if is_linearish(t1) and is_linearish(t2):
        return t1.to(Linear) @ t2.to(Linear)

    if is_affineish(t1) and is_affineish(t2):
        return t1.to(Affine) @ t2.to(Affine)

    return Sequence(transforms=[t2, t1], input=t2.input, output=t1.output)
