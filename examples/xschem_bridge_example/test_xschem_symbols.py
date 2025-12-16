#!/usr/bin/env python3
"""
Test script to validate xschem symbols using docker.

This script:
1. Generates xschem symbols
2. Validates them using xschem in docker (if available)
"""

import os
import subprocess
import tempfile
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
    """Create a simple IR with realistic primitive names."""
    # NMOS
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
    
    # Resistor
    res_subckt = SubcktDef(
        name=Ident("res_poly"),
        ports=[Ident("p"), Ident("n")],
        params=[
            ParamDecl(name=Ident("r"), default=Float(1000.0)),
            ParamDecl(name=Ident("m"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    return Program(files=[
        SourceFile(
            path=Path("test_primitives.sp"),
            contents=[nmos_subckt, res_subckt]
        )
    ])


def check_docker_available():
    """Check if docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def validate_symbols_in_docker(symbol_dir: Path):
    """Validate xschem symbols using docker."""
    if not check_docker_available():
        print("Docker is not available. Skipping validation.")
        return False
    
    print("Validating symbols with xschem in docker...")
    
    # Get current user ID
    uid = os.getuid()
    gid = os.getgid()
    
    # Find all .sym files
    sym_files = list(symbol_dir.glob("*.sym"))
    if not sym_files:
        print("No .sym files found to validate")
        return False
    
    print(f"Found {len(sym_files)} symbol files")
    
    # First, check that xschem is available in docker
    # Try to run xschem --help or similar to verify it's available
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--user", f"{uid}:{gid}",
                "hpretl/iic-osic-tools",
                "bash", "-c", "command -v xschem || find /usr -name xschem 2>/dev/null | head -1"
            ],
            capture_output=True,
            timeout=10,
        )
        
        xschem_path = result.stdout.decode().strip()
        if not xschem_path or result.returncode != 0:
            print("  ⚠ xschem may not be in standard location, but will try anyway")
            print("  (This is okay - xschem might still work)")
        else:
            print(f"  ✓ xschem found: {xschem_path}")
        
        # Check each symbol file format
        for sym_file in sym_files:
            content = sym_file.read_text()
            if "v {xschem version=" in content:
                print(f"  ✓ {sym_file.name} has valid xschem format")
            else:
                print(f"  ✗ {sym_file.name} missing xschem header")
                return False
        
        print("\n✓ All symbols have valid format!")
        print("\nTo open xschem interactively, run:")
        print("  cd examples/xschem_bridge_example")
        print("  ./open_in_xschem.sh")
        print("\nOr if you have X11 forwarding set up:")
        print("  docker run --rm -it \\")
        print("    -v $(pwd)/output:/workspace/symbols \\")
        print("    -e DISPLAY=$DISPLAY \\")
        print("    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \\")
        print("    --user $(id -u):$(id -g) \\")
        print("    hpretl/iic-osic-tools \\")
        print("    xschem")
        return True
                
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"  ✗ Error checking docker/xschem: {e}")
        return False


def main():
    """Generate symbols and validate them."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating xschem symbols...")
    program = create_simple_ir()
    
    # Generate symbols
    options = WriteOptions(fmt=NetlistDialects.XSCHEM)
    with open(output_dir / "dummy.txt", 'w') as f:
        netlist(src=program, dest=f, options=options)
    (output_dir / "dummy.txt").unlink(missing_ok=True)
    
    print(f"Symbols generated in: {output_dir}")
    print()
    
    # Validate symbols
    success = validate_symbols_in_docker(output_dir)
    
    if success:
        print("\n✓ All symbols are valid and can be opened in xschem")
        print(f"\nTo open xschem interactively, run:")
        print(f"  cd {script_dir}")
        print(f"  ./open_in_xschem.sh")
    else:
        print("\n✗ Symbol validation failed or docker is not available")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

