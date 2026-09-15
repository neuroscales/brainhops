# compose

=== ":octicons-terminal-24:"
    ```shell
    brainhops compose [FILE ...] [--output FILE]
    ```

Combine several transformations into one and write the result. The
transformations are combined following the composition operator: given
transformations `A` and `B`, the composite maps a coordinate through `B`
first and then through `A`, matching `A @ B` in the data model. The
files are listed in this composition order, so the last file listed is
applied first.

## Positional arguments

| Name         | Type          | Description                                                    |
| ------------ | ------------- | ------------------------------------------------------------------ |
| `transforms` | `list[path]`  | Transformation files to combine, in composition order.        |

## Options

| Flag             | Type   | Description                                    | Default |
| ----------------- | ------ | ------------------------------------------------ | ------- |
| `-h`, `--help`     |        | Show the help message and exit.                  |         |
| `-o`, `--output`   | `path` | Path to write the combined transformation to.    |         |

!!! warning "Not implemented yet"
    Running this command raises an error rather than combining anything.
    It depends on a writable transformation format (issue #41), and on
    the still-undecided command-line syntax for per-operand operations
    such as inversion and the matrix square root.
