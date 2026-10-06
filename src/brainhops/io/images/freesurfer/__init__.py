"""Readers and writers for FreeSurfer image formats."""

__all__ = ["mgh", "MghImage", "MghMetadata"]

from . import mgh
from .mgh import MghImage, MghMetadata
