
"""
Spectre Dialect Netlister (Placeholder)
"""
from typing import IO
from ..data import Program
from .base import Netlister, ErrorMode
from .. import NetlistDialects

class SpectreNetlister(Netlister):
    """Placeholder for Spectre Netlister if needed separate from base"""
    
    @property
    def enum(self):
        return NetlistDialects.SPECTRE
