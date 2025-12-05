"""
Netlist Writing Module
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, IO
from datetime import datetime

from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, HspiceNetlister, NgspiceNetlister, CdlNetlister
from .xyce import XyceNetlister

class WriteOptions:
    """Options for writing a netlist"""
    def __init__(self, fmt, file_type: str = "", includes: list = None, model_file: str = None, model_level_mapping: Optional[Dict[str, List[Tuple[int, int]]]] = None, log_file: Optional[str] = None):
        self.fmt = fmt
        self.file_type = file_type
        self.includes = includes or []  # List of (file, section) tuples for .lib statements
        self.model_file = model_file  # Path to model file for .include
        self.model_level_mapping = model_level_mapping
        self.log_file = log_file  # Optional path to log file for warnings

def _write_log_file(warnings: List[Tuple[str, Optional[str]]], log_path: str, src_info: Optional[str] = None, dest_info: Optional[str] = None) -> None:
    """Write collected warnings to a log file.
    
    Args:
        warnings: List of (message, context) tuples
        log_path: Path to the log file
        src_info: Optional source file/dialect information
        dest_info: Optional destination file/dialect information
    """
    with open(log_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("Netlist Translation Warning Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if src_info:
            f.write(f"Source: {src_info}\n")
        if dest_info:
            f.write(f"Destination: {dest_info}\n")
        f.write("=" * 80 + "\n\n")
        
        if not warnings:
            f.write("No warnings generated during translation.\n")
            return
        
        # Group warnings by type
        bjt_warnings = []
        other_warnings = []
        
        for msg, context in warnings:
            if "BJT" in msg or "MEXTRAM" in msg or "Level 504" in msg or "Level 1" in msg:
                bjt_warnings.append((msg, context))
            else:
                other_warnings.append((msg, context))
        
        # Write BJT warnings section
        if bjt_warnings:
            f.write("BJT Parameter Compatibility Warnings\n")
            f.write("-" * 80 + "\n")
            for msg, context in bjt_warnings:
                if context:
                    f.write(f"[{context}] {msg}\n")
                else:
                    f.write(f"{msg}\n")
                f.write("\n")
            f.write("\n")
        
        # Write other warnings section
        if other_warnings:
            f.write("Other Translation Warnings\n")
            f.write("-" * 80 + "\n")
            for msg, context in other_warnings:
                if context:
                    f.write(f"[{context}] {msg}\n")
                else:
                    f.write(f"{msg}\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total warnings: {len(warnings)}\n")
        f.write("=" * 80 + "\n")

def netlist(src, dest, options):
    """Write a netlist
    
    Args:
        src: Source Program
        dest: Destination IO stream or file path
        options: WriteOptions instance
        
    Returns:
        Optional[str]: Path to log file if one was generated, None otherwise
    """
    # Dispatch to the appropriate netlister
    from .. import NetlistDialects
    
    if options.fmt == NetlistDialects.XYCE:
        netlister = XyceNetlister(src, dest, file_type=options.file_type, 
                                  includes=options.includes, model_file=options.model_file,
                                  model_level_mapping=options.model_level_mapping)
    else:
        netlister = SpiceNetlister(src, dest, file_type=options.file_type)
        
    netlister.netlist()
    
    # Generate log file if requested or if warnings were generated
    warnings = netlister.get_warnings()
    log_file_path = None
    
    if options.log_file:
        # Use provided log file path
        log_file_path = options.log_file
        _write_log_file(warnings, log_file_path)
    elif warnings:
        # Auto-generate log file if warnings exist
        # Create temporary log file
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        log_file_path = log_file.name
        log_file.close()
        
        # Determine source/dest info for log header
        src_info = None
        dest_info = None
        if hasattr(dest, 'name'):
            dest_info = f"File: {dest.name}, Format: {options.fmt.value if hasattr(options.fmt, 'value') else str(options.fmt)}"
        elif isinstance(dest, (str, Path)):
            dest_info = f"File: {dest}, Format: {options.fmt.value if hasattr(options.fmt, 'value') else str(options.fmt)}"
        else:
            dest_info = f"Format: {options.fmt.value if hasattr(options.fmt, 'value') else str(options.fmt)}"
        
        _write_log_file(warnings, log_file_path, dest_info=dest_info)
        print(f"Translation warnings logged to: {log_file_path}")
    
    return log_file_path
