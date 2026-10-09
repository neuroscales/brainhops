# Reading and writing capabilities

Reading, writing and format dispatch have separate APIs. `FileReader`
provides `sniff*`, `load` and `from_*` for one format; `FileWriter` provides
`save` and `to_*`. `FormatDispatcher` selects among registered formats.
Readers and dispatchers share private input adapter mixins, but neither
inherits the other's public API. In particular, a reader's `sniff*` returns
a confidence score; a dispatcher's `sniff*` returns a selected class or
`None`. These capabilities add no fields to the object's representation.
Text and binary variants supply the corresponding stream modes and adapters.

An internal format representation may use any fields or third-party type.
When implementing its own I/O, it can inherit `FileReader` (a `FileReader`)
and a writer independently:

```python
class LtaStruct(TextFileReader, TextFileWriter):
    # Native fields and implementations of from_lines / to_lines.
    ...
```

A write-only representation inherits only `TextFileWriter` or
`BinaryFileWriter`. It does not acquire `load`, `from_*`, or `sniff*`.
Conversely, a parser does not acquire writing methods.

Public format objects implement the Image/Transformation API and hold their
internal representation. They can use the shared readers and writers without
inheriting a parser. `FormatDispatcher` owns the format-selection methods
and reuses the input adapter mixins; `FileBasedObject` combines it with `Format`
membership. A family owns a registry through `@format_registry`; a concrete
format joins its ancestors' registries through `@register_format`.

```python
@register_format
class MyTransformation(
    TextFileReader, TextFileWriter, Affine, FileBasedTransformation
):
    # Model properties wrap a private native representation.
    # from_lines reads that representation, then wraps it.
    # to_lines delegates to the representation of the current model.
    ...
```

These examples illustrate the bases rather than complete format
implementations. List the specialized text/binary adapters before generic
I/O bases so that their encoding and stream modes take precedence. Readers
and writers share no inheritance dependency; when both occur in a hierarchy,
keep their order consistent across its bases.

`Format` itself supplies membership and file-name matching attributes, with
no reading or writing API. A public export-only object can therefore combine
its model with `Format` and `FileWriter`. Generic `io.save` filters the public
format registry for writer capability, so this object is eligible without
inheriting `WritableFileBasedObject`. Generic `io.load` uses the readable
`FileBasedObject` registry and never attempts export-only objects.

A standalone dispatcher such as the proposed `FileBasedMetadata` can inherit
`FormatDispatcher` and own its own registry without inheriting `Format` or
`FileBasedObject`. Its concrete formats then participate only in that
dispatcher, not in generic object loading or saving. An object's metadata
view may share its native representation; this capability split imposes no
metadata schema or mutation policy.

The generic `FileParser`, `TextFileParser`, and `BinaryFileParser` bases are
replaced by `FileReader`, `TextFileReader`, and `BinaryFileReader`. Replace
the combined `*FileParserWriter` bases with the corresponding reader and
writer bases explicitly. Format-specific parser names, such as `LtaParser`
and `NiftiParser`, remain: a format parser may implement both routes.

Existing `WritableFileBased*` classes still provide both routes. Code that
checks writing support should test `FileWriter`. Dispatchers are neither
`FileReader`s nor `FileSniffer`s; concrete public objects can additionally
compose readers for their own format.

# ::: brainhops.io.base.parsers
