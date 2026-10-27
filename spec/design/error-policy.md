# Design: one policy for errors and losses

Status: decided, not implemented (#590). This memo records how a caller
says what should happen when a conversion fails or loses information. It
replaces three conventions with one keyword, `on_error=`, and keeps
`lossy=` as a shorter way to set the part of that policy that concerns
losses. Decisions are tagged `E1` to `E7`.

## 1. The problem

Three conventions exist side by side.

- `Transformation.to()` takes `lossy=`, a boolean that accepts a lossy
  result, and `error=`, which is `True` to raise the error, an
  exception to raise in its place, or any other value to return
  instead of a result.
- The `to()` and `from_instance()` methods of metadata take `on_loss=`,
  which is a `LossPolicy` (`"ignore"`, `"warn"` or `"raise"`) or a
  `ConversionReport` to fill with what was lost.
- `errors.raise_or_warn()` takes `on_error=`, with the values
  `"raise"`, `"warn"`, `"ignore"`, an exception or warning, or a value
  to return. Nothing in `src/` calls it.

A caller who converts an image to a file format meets two of these
conventions in one call, and none of them can say "raise on this kind
of error and return `None` on that one".

## 2. The policy

E1. A policy is one of the values in the table below, or a mapping from
exception classes to such values. The keyword that takes a policy is
`on_error=`.

| Value                    | Effect                                     |
| ------------------------ | ------------------------------------------ |
| `"raise"`                | Raise the error again.                     |
| `"warn"`                 | Warn, then return the result (see E2).     |
| `"ignore"`               | Return the result (see E2) without warning. |
| a `ConversionReport`     | Record the error, then return the result.  |
| an exception class       | Raise it, from the original error.         |
| an exception instance    | Raise it, from the original error.         |
| any other value          | Return the value.                          |

E2. The result that `"warn"`, `"ignore"` and a report return is the
lossy result when the error carries one, as a `LossyConversionError`
does in its `result` attribute. Any other error has no result, and the
value returned is `None`.

E3. A mapping chooses an action by the class of the error, for example
`on_error={LossyConversionError: "warn", ConversionError: None}`. A
single value `v` is read as the mapping `{Exception: v}`. When an error
is raised, the policy walks the classes of the error in method
resolution order and applies the first one that has an entry. An error
whose classes have no entry is raised.

E4. A string is always read as an action, so a policy cannot return the
string `"raise"`, `"warn"` or `"ignore"` as a value. No conversion
needs to, and this rule keeps the common spellings short.

## 3. The `lossy=` shortcut

E5. `lossy=` stays on every conversion, because accepting a loss is the
most common request and `lossy=True` is short. It takes the same values
as `on_error=` and sets the entry for `LossyConversionError`. The
policy that a call applies is

```python
{LossyConversionError: lossy, **on_error}
```

after a single value of `on_error` has been read as a mapping (E3). The
value `lossy=True` means `"ignore"`. The default is `lossy=None`, which
adds no entry, and `lossy=False` is kept with the same meaning. Because
the entry of `lossy=` names a more specific class than a single value
of `on_error=`, `lossy=True, on_error="raise"` accepts losses and raises
every other error. An explicit entry for `LossyConversionError` in
`on_error=` is merged last and wins over `lossy=`.

## 4. The classes of errors

E6. `MetadataLossError` becomes a subclass of `LossyConversionError`,
so that `lossy=` and `on_error=` treat a loss of metadata as they treat
any other loss. `ConversionError` is a subclass of `TypeError`, and the
field converters of the data model turn a `TypeError` or a
`ValueError` into a conversion error of their own. `MetadataLossError`
was made a plain `Exception` to escape that rule, as the docstring of
the class explains. The field converters must therefore re-raise any
`ConversionError` unchanged before they handle a `TypeError`.
`ConversionReport` becomes the report of every conversion, not only of
metadata, so that it can record a loss of a transformation as well.

## 5. Migration

E7. `error=` on `Transformation.to()` and `on_loss=` on the metadata
methods become deprecated aliases of `on_error=`, through
`_core/deprecation.py`. A `LossPolicy` keeps its three values, which
are valid policies. `errors.raise_or_warn()` is rewritten on top of
the new policy or removed, since it has no caller.
