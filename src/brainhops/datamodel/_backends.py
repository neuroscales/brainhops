"""Conversion of images and transformations to another array backend.

Every class that holds data, images and transformations alike, has two
families of methods. The `to_*` methods return the data of an object as a
bare array of a backend, and the `as_*` methods return an object of the
same class whose data are arrays of that backend. The backend settings
are described in [`brainhops.backends`][], and the design in the memo
`docs/design/array-backends.md`.

The mixin below gives every class both families. A class whose data
need more than the default conversion overrides `to_backend` or
`as_backend`, and the methods named after each backend call them.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops import backends
from brainhops.backends import _CURRENT, Setting

_Default = tx.Union[Setting, tx.Any]
"""The type of a backend argument, whose default is the current setting."""


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
        data = self.data
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

    def _as_backend_data(self, backend: _Default, inplace: bool) -> tx.Self:
        """Return the object with `data` converted, for the overrides.

        The object is returned as it is when the conversion keeps the
        array, so that an object that needs no conversion stays the same
        object.
        """
        data = self.data
        if data is None:
            return self
        converted = backends.to_backend(data, backend)
        if converted is data:
            return self
        return self._with_fields(inplace, data=converted)

    def _with_fields(self, inplace: bool, **changes: tx.Any) -> tx.Self:
        """Assign new values of fields to the object, or to a copy of it."""
        if not changes:
            return self
        if inplace:
            for name, value in changes.items():
                setattr(self, name, value)
            return self
        return replace(self, **changes)


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
