"""Convert Spectre bsource examples to Xyce format."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running this example without installing the package.
#
# This file lives at: Netlist/examples/spectre_to_xyce/bsources/convert.py
# Netlist repo root is two parents up from bsources/ (examples/ -> Netlist/).
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[2]
sys.path.insert(0, str(repo_root))

from netlist import NetlistDialects, ParseOptions, WriteOptions, netlist, parse_files  # noqa: E402

# Parse the Spectre input file
input_file = script_dir / "in.scs"
options_parse = ParseOptions(dialect=NetlistDialects.SPECTRE)
program = parse_files(str(input_file), options=options_parse)

# Write to Xyce format
output_file = script_dir / "out.cir"
options_write = WriteOptions(fmt=NetlistDialects.XYCE)
with open(output_file, "w") as f:
    netlist(src=program, dest=f, options=options_write)

print(f"Conversion complete: {input_file} -> {output_file}")

