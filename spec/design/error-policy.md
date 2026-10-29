# Design: one policy for errors and losses

Status: implemented (#590). This memo records how a caller says what
should happen when a conversion fails or loses information. It replaces
three conventions with one policy, given as `on_error=`, whose half
that concerns losses is also given as `on_loss=`. Decisions are tagged
`E1` to `E8`.

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
rather than as an action. When an error is raised, the policy walks
the classes of the error in method resolution order and applies the first
one that has an entry. An error whose classes have no entry is raised.

E4. A string is always read as an action, so a policy cannot return the
string `"raise"`, `"warn"` or `"ignore"` as a value. No conversion
needs to, and this rule keeps the common spellings short.

## 3. The loss half of the policy

E5. Every conversion and `io.save` take `on_loss=`, which is the half
of the policy that concerns losses. It takes the same values as a
single action of `on_error=`, and sets the entry for
`LossyConversionError`, which also covers a `MetadataLossError`. The
policy that a call applies is

```python
{LossyConversionError: on_loss, **on_error}
```

after a single value of `on_error` has been read as a mapping (E3).
By default, `on_loss=` adds no entry, so that a loss falls under
`on_error=`. The value `None` is an action, which returns `None` in
place of a lossy result. Because the entry of `on_loss=` names a more
specific class than the scope of a single value of `on_error=`,
`on_loss="raise", on_error="ignore"` raises on a loss and ignores the
other conversion errors, and `on_loss="ignore", on_error="raise"`
accepts a loss and raises every other error. An explicit entry for
`LossyConversionError` in `on_error=` is merged last and wins over
`on_loss=`. One consequence is a sharp edge: `on_error="ignore"` alone
also returns the lossy result of a conversion that loses information,
because a loss is a conversion error. A loss is accepted with
`on_loss="ignore"`, and there is no shorter spelling.

In `io.save`, a format that holds a transformation only approximately
is a loss. `"ignore"` keeps such a format among the candidates, ranked
after those that hold the transformation exactly, `"warn"` keeps it and
warns when it is chosen, and a report keeps it and records the loss of
the one that is chosen. Any other action drops such a format, which is
also what happens by default, and a value to return, or an exception
to raise, takes the place of the loss when only such formats remain.

## 4. The classes of errors

E6. `MetadataLossError` becomes a subclass of `LossyConversionError`,
so that `on_loss=` and `on_error=` treat a loss of metadata as they treat
any other loss. `ConversionError` is a subclass of `TypeError`, and the
field converters of the data model turn a `TypeError` or a
`ValueError` into a conversion error of their own. `MetadataLossError`
was made a plain `Exception` to escape that rule, as the docstring of
the class explains. The field converters must therefore re-raise any
`ConversionError` unchanged before they handle a `TypeError`.
`ConversionReport` becomes the report of every conversion, not only of
metadata, so that it can record a loss of a transformation as well.

## 5. Migration

E7. The old keywords are removed rather than deprecated, because the
package has no users yet. `error=` on `Transformation.to()` and its
overrides, and `lossy=` on the conversions and on `io.save`, are gone,
and a call that still passes one fails as it fails for any other
unknown keyword. `on_loss=` on the metadata methods is kept, with the
meaning of E5, which is wider than its old one because it covers every
`LossyConversionError`. A `LossPolicy` keeps its three values, which
are valid actions. The strings `"raise"`, `"warn"` and `"ignore"` stay
reserved for the actions (E4). `errors.raise_or_warn()` and its type
`errors.OnError` are removed, since nothing calls them.

## 6. Where the policy is applied

E8. A policy is applied in one place, by `call_under_policy()`, and
only by the call that the user made. That function merges `on_error=`
and `on_loss=` into one policy, makes the action that the policy takes
for a loss of metadata the policy in effect (the context variable that
`metadata_loss_policy()` sets) while the call runs, and hands an error
that escapes the call to the policy. An action that the body of the
call can take where a loss happens, which is `"ignore"`, `"warn"`,
`"raise"` or a report, is set as it is. Any other action, such as a
value to return or an exception to raise, is set as `"raise"`, so that
the body raises a `MetadataLossError`, and the policy then returns its
value or raises its exception. The body acts on a loss with
`apply_loss_policy()`, which reads the policy in effect, and it passes
no policy to the methods that it calls. A method called without a
policy runs as it is, under the policy of its caller, so that only the
outermost call returns a value in place of a result, and no policy is
applied twice.

The reason is maintenance. If the body received the policy as
`on_error=`, every method of every format would have to declare both
keywords, document them and forward them to the methods that it calls,
and a method that did not forward them would escape the policy of its
caller.

The policy is applied in three ways.

- The explicit entry points declare `on_error=` and `on_loss=` and call
  `call_under_policy()` in their body: `Transformation.to()`, the
  `to_*` methods of a transformation through `_to_class()`,
  `Metadata.to()`, `Metadata.from_instance()`,
  `MetadataFormat.from_instance()`, `io.save()` and `to_bids()`.
- The common base of the formats, `FileBasedModelMixin`, defines the
  one public `consolidate(*, inplace=False, on_error, on_loss,
  **fields)`, in which both keywords are optional, and which calls the protected hook
  `_consolidate(inplace=inplace, **fields)` under the policy. A format
  overrides only the hook, which takes no policy. The hook of the base
  class, which knows no record, has nothing to consolidate, so it
  returns the object itself with `inplace=True` and an unchanged copy
  otherwise. It reports a field given by keyword as lost, with
  `apply_loss_policy()`, so that the policy decides what happens to
  it.
- Each entry point of a writer (`save`, `to_file`, `to_filename`,
  `to_fileobj`, `to_bytes`, `to_text`, `to_lines` and `to_h5`) carries
  the decorator `takes_error_policy()`, written above the method in the
  base class of the writers, `_WriterMixin`, and in every class that
  defines or overrides one. The decorator is explicit, so that a reader
  of a format sees which methods take a policy. Its wrapper removes the
  two keywords and calls the method with `call_under_policy()`. An
  override that calls its base class with `super()` passes no policy,
  so the inner call runs under the policy of the outer one, and a
  method that is already decorated is not decorated again. The body of a generator, such as a
  `to_lines` written with `yield`, runs only while its items are read,
  so its wrapper is a generator that sets the policy in effect while
  the body computes each item, and not while the caller handles it. An
  error on which the policy does not raise ends the iteration, because
  a generator has no result that a value could replace.

A test checks that no method of a format under `brainhops.io` declares
`on_error=` or `on_loss=`, except the `consolidate` of the base class
and `MetadataFormat.from_instance()`. Another test checks that every
entry point of a writer that a class under `brainhops.io` defines
carries the decorator, so that a forgotten decorator fails the suite.
One consequence of E8 is that a
conversion that the body of a format makes without a policy, such as
the conversion of a field, follows the policy of the caller rather than
the default `"warn"`.

## 7. Implementation

The types of the policy, `ErrorPolicy` and `ErrorAction`, and the
classes of errors and warnings live in `brainhops.errors`, which lists
them in its `__all__`. The functions that read a policy are internal,
in `brainhops._core.errors`: `error_policy()` reads `on_error=` and
`on_loss=` into one mapping (E3, E5), `policy_action()` walks the
method resolution order, and `apply_error_policy()` acts on one error
(E1, E2). `error_policy()` and `apply_error_policy()` take the scope of
the call as `scope=`, whose default is `(ConversionError,)`, and
`call_under_policy()` takes the same argument. The default of
`on_error=` and of `on_loss=` is the sentinel `UNSET` rather than
`None`, because `None` is a policy that returns `None` (E1). `UNSET`
is an internal name of `brainhops._core.errors`, so that the public
docstrings describe both keywords as optional and never name the
sentinel. Without a policy, every error is
raised and a loss of metadata follows the policy in effect, which
`metadata_loss_policy()` sets, so that a call that names no policy
behaves as it did before. `ConversionWarning` is the warning of the
`"warn"` action, and `MetadataLossWarning` is a subclass of it.
`call_under_policy()`, `apply_loss_policy()` and the decorator of the
writers, `takes_error_policy()`, are internal names of
`brainhops.datamodel.metadata._report`.

`ConversionReport` gains an `errors` list and a `record()` method. A
`MetadataLossError` adds its fields to `lost` and `approximated`, as
before, and any other error is appended to `errors` (E6).

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
