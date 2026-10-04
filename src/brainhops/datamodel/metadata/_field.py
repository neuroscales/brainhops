"""`MetadataField`: the `metadata` field of images and transformations."""

__all__ = ["MetadataField"]

# externals
import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import ConvertTo, KwOnly, NoEq, NoRepr

# internals
from ._base import Metadata, _fits

# ----------------------------------------------------------------------
#   THE `metadata` FIELD
# ----------------------------------------------------------------------


def _target_class(hint: tx.Any) -> tx.Optional[type]:
    """The `Metadata` class of a field hint (`Optional[X]` gives `X`)."""
    if isinstance(hint, type) and issubclass(hint, Metadata):
        return hint
    for arg in tx.get_args(hint):
        found = _target_class(arg)
        if found is not None:
            return found
    return None


class _EnsureCopy:
    """
    The converter of a `metadata` field: converts what it is given into
    the field's class, as `bagof` would (and a metadata object of another
    class, even a subclass of a generic field's `Metadata`, by
    conversion, which reports the loss), and copies a metadata object
    that already is of that class (`Metadata.copy`), so that two images
    or transformations never hold the same one.
    """

    def __init__(self, hint: tx.Any) -> None:
        self.hint = hint
        self._convert: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
        self._target: tx.Optional[type] = None

    def __call__(self, value: tx.Any) -> tx.Any:
        if self._convert is None:
            self._convert = Converter.get(self.hint)
            self._target = _target_class(self.hint)
        target = self._target
        if (
            isinstance(value, Metadata)
            and target is not None
            and not _fits(value, target)
        ):
            return target.from_instance(value)
        out = self._convert(value)
        if out is value and isinstance(out, Metadata):
            out = out.copy()
        return out


class MetadataField:
    """
    The annotation of a `metadata` field:
    `MetadataField[hint, *annotations]` is
    `Annotated[hint, ConvertTo(...), KwOnly(), NoRepr(), NoEq(),
    *annotations]`.

    The field is keyword-only, out of `repr` and `==`, converted on
    assignment (even on a class that does not convert its fields) and
    copied rather than shared. `hint` is a `Metadata` class, or
    `Optional` of one; the annotations add the rest:

    ```python
    metadata: MetadataField[tx.Optional[Metadata], tx.Doc("...")] = None
    metadata: MetadataField[
        NiftiMetadata, Factory(NiftiMetadata), tx.Doc("...")
    ]
    ```

    A narrowed field must be declared on the class itself or on its
    first base (see the module documentation).
    """

    def __class_getitem__(cls, params: tx.Any) -> tx.Any:
        if not isinstance(params, tuple):
            params = (params,)
        hint, *extras = params
        return tx.Annotated[
            (
                hint,
                ConvertTo(_EnsureCopy(hint)),
                KwOnly(),
                NoRepr(),
                NoEq(),
                *extras,
            )
        ]
