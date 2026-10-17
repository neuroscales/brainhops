"""Conversion of images and transformations to another array backend.

Every class that holds data, images and transformations alike, has two
families of methods. The `to_*` methods return the data of an object as a
bare array of a backend, and the `as_*` methods return an object of the
same class whose data are arrays of that backend. The backend settings
are described in [`brainhops.backends`][], and the design in the memo
`spec/design/array-backends.md`.

The mixin below gives every class both families. A class whose data
need more than the default conversion overrides `to_backend` or
`as_backend`, and the methods named after each backend call them.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops import backends
from brainhops._core.properties import smartproperty
from brainhops.backends import _CURRENT, Setting

_Default = tx.Union[Setting, tx.Any]
"""The type of a backend argument, whose default is the current setting."""

_T = tx.TypeVar("_T")


class _BackendsMixin:
    """The `to_*` and `as_*` methods of the classes that hold data.

    By default, `to_backend` converts `data`, and `as_backend` returns
    the object as it is, because most transformations hold only small
    parameters, which stay NumPy arrays. Images, transformation fields
    and the transformations that hold other transformations override
    `as_backend`.
    """

    def to_backend(self, backend: _Default = _CURRENT) -> tx.Any:
        """Return the data as a bare array of the backend of a setting.

        The object does not change. Small parameters, such as the matrix
        of an affine transformation, are converted too, since the caller
        asks for them. An object without data, such as the identity,
        gives `None`.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.

        Returns
        -------
        array or None
            The data.
        """
        data = _stored_data(self)
        if data is None:
            return None
        return backends.to_backend(data, backend)

    def as_backend(
        self, backend: _Default = _CURRENT, *, inplace: bool = False
    ) -> tx.Self:
        """Return the object with its voxels converted as a setting says.

        By default, the object holds no voxels and is returned as it is.
        Small parameters stay NumPy arrays where they are stored, so that
        code can convert a list of objects without testing their type.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.
        inplace : bool, default=False
            Whether this object holds the converted data and is returned,
            instead of a new object.

        Returns
        -------
        object
            An object of the same class.
        """
        return self

    def to_numpy(self) -> tx.Any:
        """Return the data as a NumPy array, reading or computing them."""
        return self.to_backend("numpy")

    def to_cupy(self) -> tx.Any:
        """Return the data as a CuPy array on the GPU."""
        return self.to_backend("cupy")

    def to_dask(self) -> tx.Any:
        """Return the data as a Dask array, without reading them."""
        return self.to_backend("dask")

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

    def as_cupy(self, *, inplace: bool = False) -> tx.Self:
        """Return the object with its data as CuPy arrays on the GPU.

        Parameters
        ----------
        inplace : bool, default=False
            Whether this object holds the converted data and is returned.

        Returns
        -------
        object
            An object of the same class.
        """
        return self.as_backend("cupy", inplace=inplace)

    def as_dask(self, *, inplace: bool = False) -> tx.Self:
        """Return the object with its data as Dask arrays.

        Parameters
        ----------
        inplace : bool, default=False
            Whether this object holds the converted data and is returned.

        Returns
        -------
        object
            An object of the same class.
        """
        return self.as_backend("dask", inplace=inplace)


def as_backend_all(
    objects: tx.Optional[tx.Sequence[_BackendsMixin]], backend: _Default
) -> tx.Any:
    """Convert every object of a list or a tuple with its `as_backend`.

    The list or the tuple is returned as it is when no object changes.

    Parameters
    ----------
    objects : list or tuple, optional
        The objects, such as the members of a sequence.
    backend : str or module
        The setting.

    Returns
    -------
    list or tuple or None
        The converted objects.
    """
    if not objects:
        return objects
    converted = [obj.as_backend(backend) for obj in objects]
    if all(new is old for new, old in zip(converted, objects)):
        return objects
    return type(objects)(converted)


def _as_backend_data(obj: _T, backend: _Default, inplace: bool) -> _T:
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

    Returns
    -------
    object
        The object, or a copy of it.
    """
    data = _stored_data(obj)
    if data is None:
        return obj
    converted = backends.to_backend(data, backend)
    if converted is data:
        return obj
    return _with_fields(obj, inplace, data=converted)


def _with_fields(obj: _T, inplace: bool, **changes: tx.Any) -> _T:
    """Assign new values of fields to an object, or to a copy of it.

    Parameters
    ----------
    obj : object
        An image or a transformation.
    inplace : bool
        Whether the object itself is changed.
    **changes : Any
        The new value of each field, by name.

    Returns
    -------
    object
        The object, or the copy.
    """
    if not changes:
        return obj
    if inplace:
        for name, value in changes.items():
            setattr(obj, name, value)
        return obj
    return replace(obj, **changes)


def _stored_data(obj: tx.Any) -> tx.Any:
    """Return the `data` of an object as it is stored, without converting it.

    The `data` of an image or a field follows the backend setting in
    effect when it is read (see [`dataproperty`][]). The conversion
    methods read it under the setting `None`, which returns the stored or
    cached array itself, so that they convert it once, from what the
    object holds.
    """
    with backends.backend(None):
        return obj.data


def _follow_setting(value: tx.Any) -> tx.Any:
    """Convert an array with the backend setting in effect, if any.

    Under the default setting, `None`, the array is returned as it is.
    Under any other setting, it is converted as
    [`to_backend`][brainhops.backends.to_backend] says, and a new array is
    returned at each call.
    """
    if backends.get_backend() is None:
        return value
    return backends.to_backend(value)


def dataproperty(
    fget: tx.Optional[tx.Callable] = None, **kwargs: tx.Any
) -> tx.Any:
    """Declare the `data` of an image or a field, which follows the setting.

    A format reads its `data` from the array of its file the first time
    that `data` is read, and caches the result. At each read, the cached
    array, or an array that was assigned, is returned converted with the
    backend setting in effect: as it is under the default setting, and as
    a new array of the requested backend under any other setting, such as
    `with backend("dask"):`. The cached array does not change, so an
    object that should hold another backend for good is converted with
    an `as_*` method instead.

    Parameters
    ----------
    fget : callable, optional
        The getter that decodes the data from the file. Without it, the
        function returns a decorator.
    **kwargs : Any
        Other options of [`smartproperty`][], such as `invalidates`.

    Returns
    -------
    property
        The property, or a decorator that makes one.
    """
    kwargs.setdefault("cache", True)
    kwargs["convert"] = _follow_setting
    if fget is None:
        return smartproperty(**kwargs)
    return smartproperty(fget, **kwargs)
