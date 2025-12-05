"""
# Netlist Conversion 

Interleaved parsing and writing of netlists, designed to speed up large inputs. 
"""

# Std-Lib Imports
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Optional
from datetime import datetime

# PyPi
from pydantic.dataclasses import dataclass

# Local Imports
from .data import NetlistDialects, Program


@dataclass
class ConversionIO:
    """Input and Output Datatype for Nelist-Format Conversion."""

    path: Path
    base_dir: Path
    dialect: Optional[NetlistDialects]


def convert(src: ConversionIO, dest: ConversionIO) -> Optional[str]:
    """Convert a potentially multi-file netlist-program between dialects.
    
    Returns:
        Optional[str]: Path to log file if warnings were generated, None otherwise
    """
    from .parse import Parser, ErrorMode as ParserErrorMode
    from .write import writer, ErrorMode as WriterErrorMode
    from .write import _write_log_file

    # Create the Parser
    parser = Parser(
        src=[src.path], dialect=src.dialect, errormode=ParserErrorMode.STORE
    )

    # Collect all warnings across all files
    all_warnings = []

    # For each source-file parsed, write and drop it from memory
    while parser.pending:
        sourcefile = parser.parse_one()

        if src.base_dir not in sourcefile.path.parents:
            raise TabError

        #
        dest_path = Path(
            str(sourcefile.path).replace(str(src.base_dir), str(dest.base_dir))
        )
        if not dest_path.parent.exists():
            os.makedirs(dest_path.parent)

        # Create a Program from the source file
        from .data import Program
        source_program = Program(files=[sourcefile])
        
        # Create a new netlister for this file
        netlister_cls = writer(dest.dialect)
        netlister = netlister_cls(
            src=source_program, dest=open(dest_path, "w"), errormode=WriterErrorMode.COMMENT
        )
        
        # Write the program
        netlister.netlist()
        netlister.dest.flush()
        netlister.dest.close()
        
        # Collect warnings from this file
        file_warnings = netlister.get_warnings()
        if file_warnings:
            # Add file context to warnings
            for msg, context in file_warnings:
                file_context = f"File: {sourcefile.path.name}"
                if context:
                    file_context = f"{file_context}, {context}"
                all_warnings.append((msg, file_context))

        print(f"Converted {sourcefile.path} to {dest_path}")

    # Generate log file if warnings were collected
    log_file_path = None
    
    if all_warnings:
        # Create temporary log file
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        log_file_path = log_file.name
        log_file.close()
        
        # Determine source/dest info for log header
        src_info = f"File: {src.path}, Dialect: {src.dialect.value if src.dialect and hasattr(src.dialect, 'value') else str(src.dialect)}"
        dest_info = f"Base directory: {dest.base_dir}, Dialect: {dest.dialect.value if dest.dialect and hasattr(dest.dialect, 'value') else str(dest.dialect)}"
        
        _write_log_file(all_warnings, log_file_path, src_info=src_info, dest_info=dest_info)
        print(f"Translation warnings logged to: {log_file_path}")
    
    return log_file_path
