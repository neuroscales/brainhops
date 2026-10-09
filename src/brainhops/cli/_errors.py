"""Errors raised by the command-line interface.

Commands never call `sys.exit`. They raise a [`CliError`][], whose
message the dispatcher prints to standard error before exiting with the
exit code of the error.
"""

from __future__ import annotations


class CliError(Exception):
    """Error that ends a command with a message and an exit code.

    Raising instead of exiting keeps commands testable. The default exit
    code is 1.
    """

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class WritingUnavailable(CliError):
    """Error raised when a computed result cannot be written.

    The output name may match no writable format, no format may hold the
    object, or the chosen format may fail to write it. The exit code 3 sets
    this case apart from ordinary errors.
    """

    exit_code: int = 3

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=self.exit_code)
