"""
Netlist Writing Module
"""

from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, HspiceNetlister, NgspiceNetlister, CdlNetlister
from .xyce import XyceNetlister

class WriteOptions:
    """Options for writing a netlist"""
    def __init__(self, fmt, file_type: str = ""):
        self.fmt = fmt
        self.file_type = file_type

def netlist(src, dest, options):
    """Write a netlist"""
    # Dispatch to the appropriate netlister
    from .. import NetlistDialects
    
    if options.fmt == NetlistDialects.XYCE:
        netlister = XyceNetlister(src, dest, file_type=options.file_type)
    else:
        netlister = SpiceNetlister(src, dest, file_type=options.file_type)
        
    netlister.netlist()
