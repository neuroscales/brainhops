# Design: one policy for errors and losses

Status: implemented (#590). This memo records how a caller
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
`on_error={LossyConversionError: "warn", ConversionError: None}`. Each
call that takes a policy declares a scope, which is the tuple of the
exception classes that the call raises on purpose when it cannot
convert. A single value `v` is read as the mapping
`{cls: v for cls in scope}`. The scope of `Transformation.to()` and of
its overrides is `(ConversionError,)`, which covers a
`LossyConversionError` and a `MetadataLossError`. The metadata
methods, the writers, `consolidate` and `io.save` have the same scope,
so that a single value never applies to an `OSError`. An error outside
the scope, such as an `OSError` or a bug, is raised unchanged, because
a caller who writes `on_error=None` asks for `None` when the conversion
fails, not for every error to be hidden. A mapping is taken as written,
and may name a class outside the scope, as `{Exception: None}` does to
catch every error. The value `"raise"` needs no scope, since raising is
what happens without a policy. An exception class or instance given as
a single value stays an action, which raises that exception in place of
an error of the scope. The value `True` or `False` is refused as a
single value with a `TypeError`, because it reads as a yes or a no
rather than as an action, and the deprecated `error=True` still reads
as `"raise"` (E7). When an error is raised, the policy walks the
classes of the error in method resolution order and applies the first
one that has an entry. An error whose classes have no entry is raised.

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
the entry of `lossy=` names a more specific class than the scope of a
single value of `on_error=`, `lossy=True, on_error="raise"` accepts
losses and raises every other error. An explicit entry for `LossyConversionError` in
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

## 6. Implementation

The policy lives in `brainhops.errors`. `ErrorPolicy` and `ErrorAction`
are its types, `error_policy()` reads `on_error=` and `lossy=` into one
mapping (E3, E5), `policy_action()` walks the method resolution order,
and `apply_error_policy()` acts on one error (E1, E2). Both functions
take the scope of the call as `scope=`, whose default is
`(ConversionError,)`. The decorator `takes_error_policy` takes a
`scope=` argument with the same default, and `Transformation.to()`
passes `(ConversionError,)` to `call_under_policy()`. The default of
`on_error=` is the sentinel `UNSET` rather than `None`, because `None`
is a policy that returns `None` (E1). Without a policy, every error is
raised and a loss of metadata follows the policy in effect, which
`metadata_loss_policy()` sets, so that a call that names no policy
behaves as it did before. `ConversionWarning` is the warning of the
`"warn"` action, and `MetadataLossWarning` is a subclass of it.

`ConversionReport` gains an `errors` list and a `record()` method. A
`MetadataLossError` adds its fields to `lost` and `approximated`, as
before, and any other error is appended to `errors` (E6).

The metadata methods that took `on_loss=` take `on_error=`, and the
conversions among them, `Metadata.to()` and the `from_instance()` of
metadata, also take `lossy=` (E5). The decorator `takes_error_policy`
applies the policy around such a method. The body of the method acts on
a loss where the loss happens, because a warning or a report lets the
method go on with its result, so the decorator passes it only the
action for a `MetadataLossError`, and an action that the body cannot
take there, such as a value to return, is passed as `"raise"` and then
applied by the decorator. Because the body receives a mapping that only
concerns losses, it can pass its `on_error` on to another method
without widening the policy. `Transformation.to()` runs its conversion
with `call_under_policy()`, which makes the action for a loss the policy
in effect while the conversion runs, so that the conversions of metadata
that it triggers follow the policy of the call.

The field converter of the data model, `DataModelConverter`, lets a
`LossyConversionError` through before it handles a `TypeError` (E6).
It narrows E6, which names any `ConversionError`, because the converter
of a union and the callers of the converter rely on the wrapping of any
other conversion error: a class refuses a sibling instance with a
`ConversionError` of brainhops, and the converter of a union moves on to
its next member only after a conversion error of bagof. A refused loss
is the error that a policy must see as itself, and it still does. The
walk of a chain of converters raises a loss that carries no
result instead of walking it through. `io.save` acts on its report once
the file is written, outside of the loops that try the candidate
formats, because a refused loss is a `ConversionError`, which those
loops take for a format that refuses the object.

A deprecated alias keeps its old meaning, so that code that uses it
does not change behaviour (E7). The value `v` of `error=` reads as
`on_error={ConversionError: v}`, with `True` read as `"raise"`, and the
value `v` of `on_loss=` reads as `on_error={MetadataLossError: v}`.
`errors.raise_or_warn()` is removed, and its type `errors.OnError`
imports as `ErrorPolicy` with a `DeprecationWarning`.
