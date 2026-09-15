---
icon: octicons/terminal-24
---

# Command-line API

`brainhops` exposes a subset of its functionality as subcommands of a
single program.

=== ":octicons-terminal-24:"
    ```shell
    brainhops <command> [options]
    ```

### Positional arguments

| Name      | Type                        | Description                                                          |
| --------- | --------------------------- | ---------------------------------------------------------------------- |
| `command` | `{reslice,compose,convert}` | Subcommand to run. See the [command reference](commands.md) for the arguments each one accepts. |

### Options

| Flag           | Type | Description                 | Default |
| -------------- | ---- | ---------------------------- | ------- |
| `-h`, `--help` |      | Show the help message and exit. |         |

Running `brainhops` with no command prints this help message and exits
with a non-zero status. `brainhops <command> --help` prints the help
message for that command alone.
