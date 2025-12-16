#!/usr/bin/env python3
"""
Example: Xschem Bridge File Generation

Demonstrates generating xschem symbols and bridge files for multiple spice dialects
from a simple IR containing primitives (nmos, pmos, bjt, resistor, capacitor, diode).
"""

from pathlib import Path
from netlist.data import (
    Program,
    SourceFile,
    SubcktDef,
    Ident,
    ParamDecl,
    Float,
    ModelDef,
)
from netlist import netlist, NetlistDialects, WriteOptions


def create_simple_ir() -> Program:
    """Create a simple IR with 6 primitives, each with their own model."""
    
    # NMOS with model
    nmos_subckt = SubcktDef(
        name=Ident("nmos_1v0"),
        ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[
            ParamDecl(name=Ident("l"), default=Float(0.15)),
            ParamDecl(name=Ident("w"), default=Float(1.0)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    nmos_model = ModelDef(
        name=Ident("nmos_1v0_model"),
        mtype=Ident("nmos"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Float(54)),
        ]
    )
    
    # PMOS with model
    pmos_subckt = SubcktDef(
        name=Ident("pmos_1v0"),
        ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[
            ParamDecl(name=Ident("l"), default=Float(0.15)),
            ParamDecl(name=Ident("w"), default=Float(1.0)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    pmos_model = ModelDef(
        name=Ident("pmos_1v0_model"),
        mtype=Ident("pmos"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Float(54)),
        ]
    )
    
    # BJT with model
    bjt_subckt = SubcktDef(
        name=Ident("npn_bjt_vert"),
        ports=[Ident("c"), Ident("b"), Ident("e"), Ident("s")],
        params=[
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    bjt_model = ModelDef(
        name=Ident("npn_bjt_vert_model"),
        mtype=Ident("npn"),
        args=[],
        params=[]
    )
    
    # Resistor with model
    res_subckt = SubcktDef(
        name=Ident("res_poly"),
        ports=[Ident("p"), Ident("n")],
        params=[
            ParamDecl(name=Ident("r"), default=Float(1000.0)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    res_model = ModelDef(
        name=Ident("res_poly_model"),
        mtype=Ident("resistor"),
        args=[],
        params=[]
    )
    
    # Capacitor with model
    cap_subckt = SubcktDef(
        name=Ident("mimcap_m7m8"),
        ports=[Ident("p"), Ident("n")],
        params=[
            ParamDecl(name=Ident("c"), default=Float(1e-12)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    cap_model = ModelDef(
        name=Ident("mimcap_m7m8_model"),
        mtype=Ident("capacitor"),
        args=[],
        params=[]
    )
    
    # Diode with model
    diode_subckt = SubcktDef(
        name=Ident("diode_nw"),
        ports=[Ident("p"), Ident("n")],
        params=[
            ParamDecl(name=Ident("area"), default=Float(1e-12)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    diode_model = ModelDef(
        name=Ident("diode_nw_model"),
        mtype=Ident("diode"),
        args=[],
        params=[]
    )
    
    # Create program with all subcircuits and models
    return Program(files=[
        SourceFile(
            path=Path("simple_primitives.sp"),
            contents=[
                nmos_subckt,
                nmos_model,
                pmos_subckt,
                pmos_model,
                bjt_subckt,
                bjt_model,
                res_subckt,
                res_model,
                cap_subckt,
                cap_model,
                diode_subckt,
                diode_model,
            ]
        )
    ])


def main():
    """Generate xschem symbols and bridge files for multiple dialects."""
    
    # Create output directory (relative to script location)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create simple IR
    program = create_simple_ir()
    
    # Write IR to JSON file for reference
    ir_json_file = script_dir / "simple_ir.json"
    from netlist.data import write_json
    write_json(program, ir_json_file)
    print(f"IR written to: {ir_json_file}")
    print()
    
    print("Generating xschem symbols and bridge files...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate xschem symbols (no bridge file)
    print("1. Generating xschem symbols (dialect-agnostic)...")
    options = WriteOptions(fmt=NetlistDialects.XSCHEM)
    # Use output directory as dest so symbols are written there
    with open(output_dir / "dummy.txt", 'w') as f:
        netlist(src=program, dest=f, options=options)
    # Remove the dummy file
    (output_dir / "dummy.txt").unlink(missing_ok=True)
    print("   Symbols generated")
    print()
    
    # Generate xschem symbols + xyce bridge file
    print("2. Generating xschem symbols + xyce bridge file...")
    options_xyce = WriteOptions(
        fmt=NetlistDialects.XSCHEM,
        subcircuit_dialect=NetlistDialects.XYCE
    )
    with open(output_dir / "dummy.txt", 'w') as f:
        netlist(src=program, dest=f, options=options_xyce)
    (output_dir / "dummy.txt").unlink(missing_ok=True)
    bridge_xyce = output_dir / "bridge_xyce.spice"
    if bridge_xyce.exists():
        print(f"   Bridge file written to: {bridge_xyce.name}")
    print()
    
    # Generate xschem symbols + ngspice bridge file
    print("3. Generating xschem symbols + ngspice bridge file...")
    options_ngspice = WriteOptions(
        fmt=NetlistDialects.XSCHEM,
        subcircuit_dialect=NetlistDialects.NGSPICE
    )
    with open(output_dir / "dummy.txt", 'w') as f:
        netlist(src=program, dest=f, options=options_ngspice)
    (output_dir / "dummy.txt").unlink(missing_ok=True)
    bridge_ngspice = output_dir / "bridge_ngspice.spice"
    if bridge_ngspice.exists():
        print(f"   Bridge file written to: {bridge_ngspice.name}")
    print()
    
    # List generated symbol files
    print("Generated symbol files:")
    for sym_file in sorted(output_dir.glob("*.sym")):
        print(f"   - {sym_file.name}")
    print()
    
    print("Example complete!")
    print()
    print("To use in xschem:")
    print("  1. Place the .sym files in your xschem symbol library")
    print("  2. Include the appropriate bridge file in your testbench:")
    print(f"     - For xyce: .include \"{bridge_xyce.name}\"")
    print(f"     - For ngspice: .include \"{bridge_ngspice.name}\"")


if __name__ == "__main__":
    main()

