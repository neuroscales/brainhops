"""Errors raised by the command-line interface.

A `CliError` carries the exit code that the process should return. The
top-level dispatcher catches it, prints the message to standard error and
exits with that code, so a command never has to call `sys.exit` itself.
"""

from __future__ import annotations


class CliError(Exception):
    """An error that ends a command with a message and an exit code.

    The dispatcher prints the message to standard error and returns
    `exit_code` from the process. A command raises this rather than
    printing and exiting, so that the same command function can be called
    from a test and inspected.
    """

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class WritingUnavailable(CliError):
    """A resampled or composed object could not be written to disk.

    No writable format is registered for the output file name, or none
    of those that are can hold the object, or the one chosen cannot write
    it. The computation itself succeeded, so this is distinct from a
    failure to compute. The dedicated exit code lets a caller tell "the
    result could not be written" apart from an ordinary error.
    """

    exit_code: int = 3

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=self.exit_code)
