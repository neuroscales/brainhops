# convert

!!! warning "Not implemented yet"
    Running `brainhops convert` reports an error and writes nothing. The
    command is blocked on writable formats (issue #41), and on a policy for
    features that the source records but the target format cannot
    represent.

=== ":octicons-terminal-24:"

    ```shell
    brainhops convert <input> --output FILE
    ```

The `convert` command is meant to read an object stored in one format and
to write it in another, for example to turn an ITK transform into a NIfTI
displacement field. The kind of object is detected from the input file, and
the target format is chosen from the extension of the output file.

## Positional arguments

| Name    | Type   | Description                    |
| ------- | ------ | ------------------------------ |
| `input` | `path` | Path to the object to convert. |

## Options

| Flag             | Type   | Description                                                                      | Default    |
| ---------------- | ------ | -------------------------------------------------------------------------------- | ---------- |
| `-h`, `--help`   |        | Show the help message and exit.                                                  |            |
| `-o`, `--output` | `path` | Path to write the converted object to. Its extension selects the target format. | *required* |
