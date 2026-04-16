import typing_extensions as _tx
from functools import partial

from brainhops.struct import ClassVar
from .struct import SpecializedStruct
from .systems import CoordinateSystem


class Transform(SpecializedStruct):
    input: _tx.Optional[CoordinateSystem] = None
    output: _tx.Optional[CoordinateSystem] = None

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
        if isinstance(x, Transform):
            x = Sequence([x, self])
        else:
            raise TypeError(f"Unsupported type for transformation: {type(x)}")
        if compute:
            x = self.compute()
        return x

    @_tx.overload
    def __matmul__(self, other):
        return self(other, compute=False)


class Affine(Transform):
    matrix: _tx.Optional["ArrayLike"] = None
    type: ClassVar[str] = "affine"


class Linear(Transform):
    matrix: _tx.Optional["ArrayLike"] = None
    type: ClassVar[str] = "linear"


class DisplacementField(Transform):
    field: _tx.Optional["ArrayLike"] = None
    type: ClassVar[str] = "displacement"


class CoordinatesField(Transform):
    field: _tx.Optional["ArrayLike"] = None
    type: ClassVar[str] = "coordinates"


class CartesianCoordinatesField(CoordinatesField):
    shape: _tx.Optional[_tx.Tuple[int, ...]] = None

    @property
    def field(self) -> "ArrayLike":
        if getattr(self, "_field", None) is None:
            import dask.array as da  # implement `get_array_backend()`
            self._field = da.meshgrid(
                *[da.arange(s) for s in self.shape], 
                indexing="ij"
            )
        return self._field

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
        return _COMPOSERS_FASTMAP[(t1, t2)]
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
    ).compute()


@_composer
def _(t1: Transform, t2: Sequence) -> Affine:
    return Sequence(
        transforms=list(t2.transforms) + [t1],
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Sequence, t2: Sequence) -> Affine:
    return Sequence(
        transforms=list(t2.transforms) + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute()


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
        translation=[t1 + t2 for t1, t2 in zip(t1.translation, t2.translation)],
        input=t2.input,
        output=t1.output
    )


_LinearIsh = _tx.Union[Linear, Scale, Permutation]
_AffineIsh = _tx.Union[_LinearIsh, Affine, Translation]


@_composer
def _(t1: _LinearIsh, t2: _LinearIsh) -> Linear:
    return (t1.to(Linear) @ t2.to(Linear)).compute()


@_composer
def _(t1: _AffineIsh, t2: _AffineIsh) -> Affine:
    return (t1.to(Affine) @ t2.to(Affine)).compute()


