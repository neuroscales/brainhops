# convert

=== ":octicons-terminal-24:"
    ```shell
    brainhops convert <input> --output FILE
    ```

Read an object from one file format and write it back in another, for
example an ITK transform to a NIfTI displacement field. The kind of
object is detected from the input, and the target format is chosen from
the output file's extension.

## Positional arguments

| Name    | Type   | Description                  |
| ------- | ------ | ------------------------------- |
| `input` | `path` | Path to the object to convert. |

## Options

| Flag             | Type   | Description                                                          | Default    |
| ----------------- | ------ | ----------------------------------------------------------------------- | ---------- |
| `-h`, `--help`     |        | Show the help message and exit.                                         |            |
| `-o`, `--output`   | `path` | Path to write the converted object to. Its extension selects the target format. | *required* |

!!! warning "Not implemented yet"
    Running this command raises an error rather than writing anything.
    It depends on writable formats (issue #41), and on the policy for a
    feature that the source format records but the target format cannot
    represent.
