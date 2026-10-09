# compose

!!! warning "Not implemented yet"
    Running `brainhops compose` reports an error and combines nothing. The
    command is blocked on a writable transformation format (issue #41), and
    on a decision about how per-operand operations, such as inversion, the
    matrix square and square root, and the exponential, should be written
    on the command line.

=== ":octicons-terminal-24:"

    ```shell
    brainhops compose [FILE ...] [--output FILE]
    ```

The `compose` command is meant to combine several transformations into one
and to write the result to a file.

The composite follows the composition operator of the data model: for two
transformations `A` and `B`, the composite `A @ B` maps a coordinate
through `B` first and then through `A`. The files are listed in
composition order, so the last file listed is applied first. This is the
opposite of [`reslice`](reslice.md), whose `--transform` options are
applied in the order they are given.

## Positional arguments

| Name         | Type         | Description                                            |
| ------------ | ------------ | ------------------------------------------------------ |
| `transforms` | `list[path]` | Transformation files to combine, in composition order. |

## Options

| Flag             | Type   | Description                                   | Default |
| ---------------- | ------ | --------------------------------------------- | ------- |
| `-h`, `--help`   |        | Show the help message and exit.               |         |
| `-o`, `--output` | `path` | Path to write the combined transformation to. |         |
