"""Readers and writers for FreeSurfer image formats."""

__all__ = ["mgh", "MGHImage"]

from . import mgh
from .mgh import MGHImage
