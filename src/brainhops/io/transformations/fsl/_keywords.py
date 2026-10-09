import typing_extensions as tx
from bagof.magic import fields_dict


def pop_aliased(cls: type, kwargs: tx.Dict[str, tx.Any], name: str) -> tx.Any:
    """Remove a field and its aliases from keyword arguments.

    The FSL readers receive the moving and reference images as keyword
    arguments and take them out of the keywords before parsing the
    file, either because the parsing code would reject them or because
    the reader passes them to the constructor itself. This function
    takes out every spelling that the field `name` of `cls` declares
    with `Alias`, so that the readers accept exactly the keywords that
    the constructor accepts.

    Every alias is removed from `kwargs`, and the value of the one that
    is not `None` is returned. As in the constructor, giving a value
    under more than one alias is an error.

    Parameters
    ----------
    cls : type
        The class that declares the field, which must be a `Magic`
        class.
    kwargs : dict[str, Any]
        The keyword arguments, which are modified in place.
    name : str
        The name of the field.

    Returns
    -------
    value : Any
        The value given under one of the aliases, or `None` if no alias
        was given a value.

    Raises
    ------
    TypeError
    """
    aliases = fields_dict(cls)[name].alias or (name,)
    if isinstance(aliases, str):
        aliases = (aliases,)
    given = {}
    for alias in aliases:
        value = kwargs.pop(alias, None)
        if value is not None:
            given[alias] = value
    if len(given) > 1:
        raise TypeError(
            f"{cls.__name__} got multiple values for argument {name!r} "
            f"(given as {', '.join(map(repr, given))})."
        )
    return next(iter(given.values()), None)
