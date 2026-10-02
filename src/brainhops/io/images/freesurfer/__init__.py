"""Readers and writers for FreeSurfer image formats."""

__all__ = ["mgh", "MghImage"]

from . import mgh
from .mgh import MghImage
