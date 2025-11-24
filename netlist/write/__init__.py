"""
Netlist Writing Module
"""

from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, HspiceNetlister, NgspiceNetlister, CdlNetlister
from .xyce import XyceNetlister

class WriteOptions:
    """Options for writing a netlist"""
    def __init__(self, fmt, file_type: str = "", includes: list = None, model_file: str = None):
        self.fmt = fmt
        self.file_type = file_type
        self.includes = includes or []
        self.model_file = model_file

def netlist(src, dest, options):
    """Write a netlist"""
    # Dispatch to the appropriate netlister
    from .. import NetlistDialects
    
    if options.fmt == NetlistDialects.XYCE:
        netlister = XyceNetlister(src, dest, file_type=options.file_type, includes=options.includes, model_file=options.model_file)
    else:
        netlister = SpiceNetlister(src, dest, file_type=options.file_type)
        
    netlister.netlist()
