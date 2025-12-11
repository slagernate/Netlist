"""Convert Spectre subcircuit file to ngspice format."""

from netlist import parse_files, netlist, NetlistDialects, ParseOptions, WriteOptions
from netlist.data import write_json
from netlist.ast_to_cst import ast_to_cst
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Parse the Spectre input file
input_file = script_dir / "in.scs"
options_parse = ParseOptions(dialect=NetlistDialects.SPECTRE)
program = parse_files(str(input_file), options=options_parse)

# Write AST Program to JSON (this is what the writer actually uses)
write_json(program, script_dir / "ir.json")

# Write to ngspice format
output_file = script_dir / "out.spice"
options_write = WriteOptions(fmt=NetlistDialects.NGSPICE)
with open(output_file, "w") as f:
    netlist(src=program, dest=f, options=options_write)

print(f"Conversion complete: {input_file} -> {output_file}")
