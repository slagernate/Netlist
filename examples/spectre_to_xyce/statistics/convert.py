"""Convert Spectre statistics file to Xyce format."""

from netlist import parse_files, netlist, NetlistDialects, ParseOptions, WriteOptions
from netlist.write.spice import apply_statistics_variations

# Parse the Spectre input file
input_file = "in.scs"
options_parse = ParseOptions(dialect=NetlistDialects.SPECTRE)
program = parse_files(input_file, options=options_parse)

# Apply statistics variations for Xyce compatibility
apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

# Write to Xyce format
output_file = "out.cir"
options_write = WriteOptions(fmt=NetlistDialects.XYCE)
with open(output_file, "w") as f:
    netlist(src=program, dest=f, options=options_write)

print(f"Conversion complete: {input_file} -> {output_file}")

