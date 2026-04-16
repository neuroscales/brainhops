import typing_extensions as _tx
from functools import partial

from .struct import SpecializedStruct
from .systems import CoordinateSystem
from .typing import HiddenConst


class Transform(SpecializedStruct):
    input: _tx.Optional[CoordinateSystem] = None
    output: _tx.Optional[CoordinateSystem] = None

    def compute(self, *args, **kwargs) -> _tx.Self:
        return self

    def to(self, cls: _tx.Type[_tx.Self]) -> _tx.Self:
        return _to(self, cls)

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


class CoordinatesField(Transform):
    field: _tx.Optional["ArrayLike"] = None
    type: HiddenConst[str] = "coordinates"


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


class DisplacementField(Transform):
    field: _tx.Optional["ArrayLike"] = None
    type: HiddenConst[str] = "displacement"


class Affine(Transform):
    matrix: _tx.Optional["ArrayLike"] = None
    type: HiddenConst[str] = "affine"


class Linear(Transform):
    matrix: _tx.Optional["ArrayLike"] = None
    type: HiddenConst[str] = "linear"

class Permutation(Transform):
    permutation: _tx.Optional[_tx.List[int]] = None
    type: HiddenConst[str] = "permutation"


class Scale(Transform):
    scale: _tx.Optional[_tx.List[float]] = None
    type: HiddenConst[str] = "scale"


class Translation(Transform):
    translation: _tx.Optional[_tx.List[float]] = None
    type: HiddenConst[str] = "translation"


class Identity(Transform):
    type: HiddenConst[str] = "identity"


class Sequence(Transform):
    # NOTE: transforms in a sequence are listed in the order in which 
    # they are applied. It reads as the opposite order to function
    # composition (or matrix multiplication), which may be confusing.
    #
    # `Sequence([t1, t2, t3])(x)` is equivalent to `t3(t2(t1(x)))`.
    # `Sequence([a1, a2, a3]) @ x` is equivalent to `a3 @ a2 @ a1 @ x`.

    transforms: _tx.Optional[_tx.List[Transform]] = None
    type: HiddenConst[str] = "sequence"

    def compute(self, mode=None) -> Transform:
        """
        Compute the resulting transform of the sequence of transforms.

        If all transforms in the sequence are affine-like transforms, 
        `compute()` returns an affine-like transform.

        If the first (= rightmost) transform in the sequence is a 
        coordinate field, `compute()` returns a coordinate field.

        If the first (= rightmost) transform in the sequence is an 
        affine-like transform, and the sequence contains at least one 
        non-affine-like transform, `compute()` returns a sequence of two
        transforms: 
        1. the composition of all affine-like transforms that appear
           before the first non-affine-like transform in the sequence, and
        2. the composition of all transforms in the sequence, starting
           from the first non-affine-like transform in the sequence.

        Parameters
        ----------
        mode : [list of] str, optional
            Types of transforms to compute. 
            * If `None` (default): compute all transforms in the sequence.
            * If the name of a transformation type: compute only consecutive
              sequences of transformations that match the specified type.
        """
        return _compute_sequence(self, mode=mode)
    
    def _flatten(self) -> _tx.Self:
        # Flatten nested sequences of transforms into a single sequence.
        if self.transforms is None:
            return self
        flattened = []
        for t in self.transforms:
            if isinstance(t, Sequence):
                flattened.extend(t._flatten().transforms or [])
            else:
                flattened.append(t)
        return Sequence(self, transforms=flattened)

    def _is_flat(self) -> bool:
        # Check if the sequence is flat (i.e., does not contain any nested sequences).
        if self.transforms is None:
            return True
        return all(not isinstance(t, Sequence) for t in self.transforms)


_IDENTITIES = {'identity'}
_TRANSLATIONS = {*_IDENTITIES, 'translation'}
_SCALES = {*_IDENTITIES, 'scale'}
_PERMUTATIONS = {*_IDENTITIES, 'permutation'}
_LINEARS = {*_SCALES, *_PERMUTATIONS, 'linear'}
_AFFINES = {*_LINEARS, *_TRANSLATIONS, 'affine'}
_NONLINEARS = {*_AFFINES, 'displacements', 'coordinates'}
_XFORMHIERARCHY = {
    'identity': _IDENTITIES,
    'translation': _TRANSLATIONS,
    'scale': _SCALES,
    'permutation': _PERMUTATIONS,
    'linear': _LINEARS,
    'affine': _AFFINES,
    'nonlinear': _NONLINEARS
}


def _compute_sequence(self: Sequence, mode=None) -> Transform:
    # Totally UNTESTED, so raise for now.
    raise NotImplementedError

    # 0. check if already flat and composed:
    if self.transforms is None or len(self.transforms) == 0:
        return Identity(input=self.input, output=self.output)
    if len(self.transforms) == 1:
        return self.transforms[0]
    # 1. flatten sequence:
    if not self._is_flat():
        return self._flatten().compute(mode=mode)
    # 2. compose consecutive transforms of the same type
    if mode is None:
        mode = (
            "translation", 
            "scale", 
            "permutation", 
            "linear", 
            "affine", 
            "nonlinear"
        )
    if isinstance(mode, str):
        mode = (mode,)
    for mode1 in mode:
        types1 = _XFORMHIERARCHY[mode1]
        for i in range(len(self.transforms) - 1):
            t1, t2 = self.transforms[i], self.transforms[i + 1]
            if t1.type != "coordinates" and t2.type == "coordinates":
                # Cannot `compose(coordinates, affine)`
                continue
            if t1.type in types1 and t2.type in types1:
                composed = _compose(t2, t1)  # ! reversed order !
                # Then: insert composed transform in place of the two 
                # original transforms, and repeat until no more 
                # consecutive transforms of the same type are found.
                transforms = (
                    self.transforms[:i] + [composed] + 
                    self.transforms[i + 2:]
                )
                return Sequence(self, transforms=transforms).compute(mode)
    return self



_COMPOSERS = {}
_COMPOSERS_FASTMAP = {}
_CONVERTERS = {}
_CONVERTERS_FASTMAP = {}


def _to(x: Transform, cls: _tx.Type[Transform]) -> Transform:
    """Convert a transform to a different type."""
    # TODO: implement using the CONVERTERS map, 
    #       similarly to _compose() and _COMPOSERS.
    raise NotImplementedError  


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
def _(t1: Sequence, t2: Transform) -> Transform:
    return Sequence(
        transforms=[t2] + list(t1.transforms),
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Transform, t2: Sequence) -> Transform:
    return Sequence(
        transforms=list(t2.transforms) + [t1],
        input=t2.input,
        output=t1.output
    ).compute()


@_composer
def _(t1: Sequence, t2: Sequence) -> Sequence:
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


