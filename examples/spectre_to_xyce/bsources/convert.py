"""Convert Spectre bsource examples to Xyce format."""

from pathlib import Path

from netlist import NetlistDialects, ParseOptions, WriteOptions, netlist, parse_files

# Get the directory where this script is located
script_dir = Path(__file__).parent

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
