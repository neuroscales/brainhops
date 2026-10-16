"""Conversion of images and transformations to another array backend.

Every class that holds data, images and transformations alike, has two
families of methods. The `to_*` methods return the data of an object as a
bare array of a backend, and the `as_*` methods return an object of the
same class whose data are arrays of that backend. The backend settings
are described in [`brainhops.backends`][], and the design in the memo
`docs/design/array-backends.md`.

A class defines `to_backend` and `as_backend`, and the mixin below
derives the methods named after each backend from them.
"""

# dependencies
import typing_extensions as tx

# internals
from brainhops import backends
from brainhops.backends import _CURRENT, Setting, to_backend

if tx.TYPE_CHECKING:
    from .base import DataModelBase

_Default = tx.Union[Setting, tx.Any]
"""The type of a backend argument, whose default is the current setting."""


class BackendConversions:
    """The `to_*` and `as_*` methods of the classes that hold data.

    A subclass defines `to_backend(backend)`, which returns its data as a
    bare array, and `as_backend(backend, inplace=False)`, which returns an
    object of the same class whose data are converted. The other methods
    call them with the name of a backend.
    """

    def to_backend(self, backend: _Default = _CURRENT) -> tx.Any:
        """Return the data as a bare array of the backend of a setting.

        Parameters
        ----------
        backend : str or module, optional
            The setting. By default, the current setting is used.

        Returns
        -------
        array
            The data.
        """
        raise NotImplementedError

    def as_backend(
        self, backend: _Default = _CURRENT, *, inplace: bool = False
    ) -> tx.Self:
        """Return the object with its data converted as a setting says.

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
        raise NotImplementedError

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


def convert_member(value: tx.Any, backend: _Default) -> tx.Any:
    """Convert an object that another object holds, or return it as it is.

    A transformation or an image is converted with its `as_backend`
    method, a list or a tuple of them element by element, and any other
    value is returned as it is. The value itself is never changed, so a
    member that needs no conversion is the same object afterwards.

    Parameters
    ----------
    value : object
        The member.
    backend : str or module
        The setting.

    Returns
    -------
    object
        The converted member.
    """
    if isinstance(value, BackendConversions):
        return value.as_backend(backend)
    if isinstance(value, (list, tuple)) and value:
        converted = [convert_member(item, backend) for item in value]
        if all(new is old for new, old in zip(converted, value)):
            return value
        return type(value)(converted)
    return value


def convert_array(value: tx.Any, backend: _Default) -> tx.Any:
    """Convert an array as a setting says, or return `None` as it is."""
    if value is None:
        return None
    if isinstance(backend, backends._Current):
        return to_backend(value)
    return to_backend(value, backend)


def stores_field(cls: type, name: str) -> bool:
    """Tell whether a class stores a field under a constructor name.

    A field derived from others, such as the `data` of an inverse, is not
    stored, so converting it would compute it.

    Parameters
    ----------
    cls : type
        A class of the data model.
    name : str
        The name of a constructor argument.

    Returns
    -------
    bool
        Whether a field of the class takes that name.
    """
    from bagof.magic import fields

    try:
        declared = fields(cls)
    except TypeError:
        return False
    return any(
        field.init and name in (field.name, *(field.aliases or ()))
        for field in declared
    )


def apply_changes(
    obj: "DataModelBase", changes: tx.Dict[str, tx.Any], inplace: bool
) -> tx.Any:
    """Assign the converted members to an object, or to a copy of it.

    Parameters
    ----------
    obj : DataModelBase
        The object.
    changes : dict
        The new value of each converted member, by name.
    inplace : bool
        Whether the object itself is changed.

    Returns
    -------
    DataModelBase
        The object, or the copy.
    """
    from bagof.magic import replace

    if inplace:
        for name, value in changes.items():
            setattr(obj, name, value)
        return obj
    return replace(obj, **changes)
