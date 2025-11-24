""" 
Netlists 

Parsing and generating popular formats of circuit netlist. 
"""

__version__ = "0.1.0"

import warnings
from pathlib import Path

# Configure warning format to be more concise (single line, no source code repetition)
# This applies globally whenever the netlist package is imported
def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f'{Path(filename).name}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = _warning_on_one_line


from .data import *
from .dialects import *
from .write import *
from .convert import convert, ConversionIO
from .parse import parse_str, parse_files, ParseOptions
from .ast_to_cst import ast_to_cst, has_external_refs, get_external_refs, Scope
from .compile import compile
