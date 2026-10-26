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
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.backends import (
    BackendLike,
    DeviceLike,
    _is_loaded,
    backend,
    to_backend,
)
from brainhops._core.magic import smart_replace
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

_T = tx.TypeVar("_T")


class _BackendsMixin:
    """
    The `to_*` and `as_*` methods of the classes that hold data.
    """

    @property
    def dataobj(self) -> ArrayProtocol:
        """The array of the object, without reading it.

        By default, it is the data that the object holds. An object read
        from a file returns instead a lazy array of the file, in the
        layout of `data`, which reads nothing until it is converted, so
        that a part of a large file can be read alone. The shape and the
        type of the values are read from it without reading the file.
        """
        return _stored_data(self)

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
        data = _stored_data(self)
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


def dataproperty(**kwargs: tx.Any) -> property:
    """Declare the `data` of an image or a field, a view of its `dataobj`.

    An object stores its array in `dataobj`, and `data` is the same
    array, loaded and in the backend of the setting in effect. Assigning
    `data` stores the new array in `dataobj`.

    A NumPy, CuPy or Dask array in `dataobj` is read as it is. Any other
    array, such as the lazy array of a file or a zarr array that was
    assigned, is loaded at the first read with the setting in effect, and
    the result is cached: under the default setting, a small array is
    read with NumPy and a large one is kept lazy as a Dask array, and
    under `with backend("dask"):` the array is kept lazy whatever its
    size. Each read then converts the loaded array with the setting in
    effect at that read, as [`to_backend`][] does, and the loaded array
    does not change. An object that should hold another backend for good
    is converted with an `as_*` method, or its backend is chosen before
    its first read. Assigning `data` or `dataobj` clears the cache.

    Parameters
    ----------
    **kwargs : Any
        Other options of [`smartproperty`][], such as `invalidates`.

    Returns
    -------
    property
        The property `data`, stored in `_data`.
    """

    # The getter runs when `_data` holds nothing or an array that is not
    # loaded, so `dataobj` is then the lazy array of a file or the array
    # that was assigned. The property is named after its getter, so the
    # getter is `data`.
    def data(self: tx.Any) -> tx.Any:
        dataobj = self.dataobj
        if dataobj is None:
            return None
        return _load(dataobj)

    return smartproperty(
        data, cache=True, convert=to_backend, unset=_not_loaded, **kwargs
    )


def dataobjproperty(
    fget: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None,
) -> property:
    """Declare the `dataobj` of an image or a field, where its array lives.

    `dataobj` is the array that the object stores, without reading it.
    By default, it is the array that was assigned as `data` or `dataobj`.
    A file format passes its own getter, which returns the assigned array
    if there is one and the lazy array of its file otherwise. Assigning
    `dataobj` is the same as assigning `data`: the array is stored, and
    the loaded array that `data` caches is cleared.

    !!! example
        ```python
        class Format(SingleScaleImage):
            @dataobjproperty
            def dataobj(self):
                if self._data is not None:
                    return self._data
                return lazy_array_of_the_file(self)
        ```

    Parameters
    ----------
    fget : callable, optional
        The getter. By default, the getter returns `_data`.

    Returns
    -------
    property
        The property `dataobj`.
    """
    if fget is None:

        def fget(self: tx.Any) -> ArrayProtocol:
            return getattr(self, "_data", None)

    def dataobj(self: tx.Any, value: tx.Any) -> None:
        self.data = value

    return property(fget, dataobj, doc=fget.__doc__)


def _not_loaded(value: tx.Any) -> bool:
    # A stored value that is not a NumPy, CuPy or Dask array reads as
    # unset, so that `data` loads it once and caches the result.
    return value is None or not _is_loaded(value)


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
    data = _stored_data(obj)
    if data is None:
        return obj
    converted = to_backend(data, backend, device=device)
    if converted is data:
        return obj
    return smart_replace(obj, "inplace" if inplace else None, data=converted)


def _stored_data(obj: tx.Any) -> tx.Any:
    """Return the `data` of an object as it is stored, without converting it.

    The `data` of an image or a field follows the backend setting in
    effect when it is read (see [`dataproperty`][]). The conversion
    methods read it under the setting `"native"`, which returns the stored or
    cached array itself, so that they convert it once, from what the
    object holds.
    """
    with backend("native"):
        return obj.data


def _load(dataobj: tx.Any) -> tx.Any:
    """Read the values of a lazy array as the setting in effect says.

    Under the default setting, a lazy array of a file is read with NumPy
    when it is small and kept lazy as a Dask array when it is large, and
    an explicit setting chooses the backend (see [`to_backend`][]). A
    NumPy array that is read from a file is made read-only, because the
    writers encode what the object holds, not this array, so a change
    made to it in place would be lost. A loaded array is returned as it is.
    """
    if _is_loaded(dataobj):
        return dataobj
    return _read_only(to_backend(dataobj), dataobj)


def _read_only(data: tx.Any, dataobj: tx.Any) -> tx.Any:
    # A NumPy array read from a lazy array of a file is made read-only,
    # as `_load` explains. A loaded array is the array itself, or a view
    # of it, so its flag is left as it is.
    if isinstance(data, np.ndarray) and not _is_loaded(dataobj):
        data.flags.writeable = False
    return data
