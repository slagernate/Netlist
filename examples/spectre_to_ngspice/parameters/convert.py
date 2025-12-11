"""Convert Spectre parameters file to ngspice format."""

from netlist import parse_files, netlist, NetlistDialects, ParseOptions, WriteOptions

# Parse the Spectre input file
input_file = "in.scs"
options_parse = ParseOptions(dialect=NetlistDialects.SPECTRE)
program = parse_files(input_file, options=options_parse)

# Write to ngspice format
output_file = "out.spice"
options_write = WriteOptions(fmt=NetlistDialects.NGSPICE)
with open(output_file, "w") as f:
    netlist(src=program, dest=f, options=options_write)

print(f"Conversion complete: {input_file} -> {output_file}")
