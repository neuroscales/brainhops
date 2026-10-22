---
icon: octicons/terminal-24
---

# Command-line API

brainhops exposes a subset of its functionality as subcommands of a single
program.

=== ":octicons-terminal-24:"

    ```shell
    brainhops <command> [options]
    ```

## Positional arguments

| Name      | Type                        | Description                                                                                       |
| --------- | --------------------------- | ------------------------------------------------------------------------------------------------- |
| `command` | `{reslice,compose,convert}` | Subcommand to run: [`reslice`](reslice.md), [`compose`](compose.md) or [`convert`](convert.md). |

The `reslice` and `convert` commands are implemented. The `compose`
command is already registered, so that its options can be reviewed, but
running it reports that it is not implemented yet and exits with an
error.

## Options

| Flag           | Type | Description                     | Default |
| -------------- | ---- | ------------------------------- | ------- |
| `-h`, `--help` |      | Show the help message and exit. |         |

Running `brainhops` without a command prints the help message and exits
with a non-zero status. Running `brainhops <command> --help` prints the
help of that command only.
