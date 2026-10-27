"""Conversion of images and transformations to another array backend.

Every class that holds data, images and transformations alike, has two
families of methods. The `to_*` methods return the data of an object as a
bare array of a backend, and the `as_*` methods return an object of the
same class whose data are arrays of that backend. The backend settings
are described in [`brainhops.backends`][].

The mixin below gives every class both families. A class whose data
need more than the default conversion overrides `to_backend` or
`as_backend`, and the methods named after each backend call them.
"""

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.backends import (
    BackendLike,
    DeviceLike,
    to_backend,
)
from brainhops._core.magic import smart_replace
from brainhops._core.typing import ArrayProtocol

_T = tx.TypeVar("_T")


class _BackendsMixin:
    """
    The `to_*` and `as_*` methods of the classes that hold data.
    """

    def to_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
    ) -> tx.Optional[ArrayProtocol]:
        """Return the data as a bare array of the backend of a setting.

        The object does not change. Small parameters, such as the matrix
        of an affine transformation, are converted too, since the caller
        asks for them. An object without data, such as the identity,
        gives `None`.

        Parameters
        ----------
        backend : str | module, optional
            The setting. By default, the current setting is used.
        device : str | int, optional
            The device for converted arrays. The backend supplies its default.

        Returns
        -------
        array | None
            The data.
        """
        data = self.data
        if data is None:
            return None
        return to_backend(data, backend, device=device)

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        """
        Return the object with its data converted to the requested backend.

        By default, the object holds no voxels and is returned as it is.
        Small parameters stay NumPy arrays where they are stored, so that
        code can convert a list of objects without testing their type.

        Parameters
        ----------
        backend : str | module, optional
            The setting. By default, the current setting is used.
        device : str | int, optional
            The device for converted arrays. The backend supplies its default.
        inplace : bool, default=False
            Whether this object holds the converted data and is returned,
            instead of a new object.

        Returns
        -------
        object
            An object of the same class.
        """
        return _as_backend_data(self, backend, inplace, device=device)

    def to_numpy(self) -> ArrayProtocol:
        """Return the data as a NumPy array, reading or computing them."""
        return self.to_backend("numpy")

    def to_cupy(self, device: tx.Optional[DeviceLike] = None) -> ArrayProtocol:
        """Return the data as a CuPy array on the GPU."""
        return self.to_backend("cupy", device=device)

    def to_dask(self, device: tx.Optional[DeviceLike] = None) -> ArrayProtocol:
        """Return the data as a Dask array, without reading them."""
        return self.to_backend("dask", device=device)

    def as_numpy(self, *, inplace: bool = False) -> tx.Self:
        """Return the object with its data as NumPy arrays.

        On an object read from a file, `as_numpy(inplace=True)` reads the
        file once and keeps its values.

        Parameters
        ----------
        inplace : bool, default=False
            Whether this object holds the converted data and is returned.

        Returns
        -------
        object
            An object of the same class.
        """
        return self.as_backend("numpy", inplace=inplace)

    def as_cupy(
        self, device: tx.Union[str, int, None] = None, *, inplace: bool = False
    ) -> tx.Self:
        """Return the object with its data as CuPy arrays on the GPU.

        Parameters
        ----------
        device : str | int | None, optional
            The device, such as `"cuda:0"` or `0`. By default, the
            backend supplies its device.
        inplace : bool, default=False
            Whether this object holds the converted data and is returned.

        Returns
        -------
        object
            An object of the same class.
        """
        return self.as_backend("cupy", device=device, inplace=inplace)

    def as_dask(
        self, device: tx.Union[str, int, None] = None, *, inplace: bool = False
    ) -> tx.Self:
        """Return the object with its data as Dask arrays.

        Parameters
        ----------
        device : str | int | None, optional
            The device, such as `"cuda:0"` or `0`.
            By default, the backend supplies its device.
        inplace : bool, default=False
            Whether this object holds the converted data and is returned.

        Returns
        -------
        object
            An object of the same class.
        """
        return self.as_backend("dask", device=device, inplace=inplace)


def as_backend(
    objects: tx.Union[tx.Sequence[_BackendsMixin], _BackendsMixin, None],
    backend: tx.Optional[BackendLike] = None,
    *,
    device: tx.Optional[DeviceLike] = None,
    inplace: bool = False,
) -> tx.Any:
    """
    Apply `as_backend` to an object or a list of objects.

    A list or tuple is returned as it is when no object changes.

    Parameters
    ----------
    objects : object | sequence, optional
        The object(s), such as the members of a sequence.
    backend : str | module
        The setting.
    device : str | int, optional
        The device for converted arrays. The backend supplies its default.
    inplace : bool, default=False
        Whether to assign the converted arrays in the objects themselves,
        or to return new objects with the converted arrays.

    Returns
    -------
    object | sequence | None
        The converted object(s).
    """
    if not objects:
        return objects
    if isinstance(objects, _BackendsMixin):
        return objects.as_backend(backend, device=device, inplace=inplace)
    converted = [
        obj.as_backend(backend, device=device, inplace=inplace)
        for obj in objects
    ]
    if all(new is old for new, old in zip(converted, objects)):
        return objects
    return type(objects)(converted)


class _FrozenBackendMixin(_BackendsMixin):
    """
    Mixin for classes that only support one backend, such as concrete
    transformations with small parameters.
    """

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        return self


class _MetaBackendMixin(_BackendsMixin):
    """
    Mixin for classes that do not hold their own data, but instead
    hold references to other objects that do.
    """

    def as_backend(
        self,
        backend: tx.Optional[BackendLike] = None,
        *,
        device: tx.Optional[DeviceLike] = None,
        inplace: bool = False,
    ) -> tx.Self:
        opt = dict(backend=backend, device=device, inplace=inplace)
        changes = {}
        for field in getattr(self, "meta_fields", ()):
            value = getattr(self, field)
            if value is None:
                continue
            converted = as_backend(value, **opt)
            if converted is not value:
                changes[field] = converted
        return smart_replace(self, "inplace" if inplace else None, **changes)


def _as_backend_data(
    obj: _T,
    backend: tx.Optional[BackendLike],
    inplace: bool,
    *,
    device: tx.Optional[DeviceLike] = None,
) -> _T:
    """Return an object with its `data` converted, for the overrides.

    The object is returned as it is when the conversion keeps the array,
    so that an object that needs no conversion stays the same object.

    Parameters
    ----------
    obj : object
        An image or a transformation that stores `data`.
    backend : str or module
        The setting.
    inplace : bool
        Whether the object itself holds the converted array.
    device : str or int, optional
        The device for converted arrays. The backend supplies its default.

    Returns
    -------
    object
        The object, or a copy of it.
    """
    data = obj.data
    if data is None:
        return obj
    converted = to_backend(data, backend, device=device)
    if converted is data:
        return obj
    return smart_replace(obj, "inplace" if inplace else None, data=converted)
