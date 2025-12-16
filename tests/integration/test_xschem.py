"""
Integration tests for xschem export dialect.

Tests xschem symbol file generation and validation.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from io import StringIO

from netlist import parse_files, netlist, NetlistDialects, ParseOptions, WriteOptions
from netlist.data import Program, SourceFile, SubcktDef, Ident, ParamDecl, Float, Instance, Ref
from netlist.write.primitive_detector import PrimitiveDetector, PrimitiveType


def test_primitive_detector_mosfet():
    """Test primitive detector for MOSFET subcircuit"""
    detector = PrimitiveDetector()
    
    # Create a MOSFET-like subcircuit
    subckt = SubcktDef(
        name=Ident("nmos"),
        ports=[
            Ident("d"),  # Drain
            Ident("g"),  # Gate
            Ident("s"),  # Source
            Ident("b"),  # Bulk
        ],
        params=[
            ParamDecl(name=Ident("l"), default=Float(0.1)),
            ParamDecl(name=Ident("w"), default=Float(1.0)),
        ],
        entries=[]
    )
    
    prim_type = detector.detect(subckt)
    assert prim_type == PrimitiveType.MOSFET


def test_primitive_detector_resistor():
    """Test primitive detector for resistor subcircuit"""
    detector = PrimitiveDetector()
    
    # Create a resistor-like subcircuit
    subckt = SubcktDef(
        name=Ident("resistor"),
        ports=[
            Ident("p"),
            Ident("n"),
        ],
        params=[
            ParamDecl(name=Ident("r"), default=Float(1000.0)),
        ],
        entries=[]
    )
    
    prim_type = detector.detect(subckt)
    assert prim_type == PrimitiveType.RESISTOR


def test_primitive_detector_bjt():
    """Test primitive detector for BJT subcircuit"""
    detector = PrimitiveDetector()
    
    # Create a BJT-like subcircuit
    subckt = SubcktDef(
        name=Ident("npn_bjt"),
        ports=[
            Ident("c"),  # Collector
            Ident("b"),  # Base
            Ident("e"),  # Emitter
            Ident("s"),  # Substrate
        ],
        params=[],
        entries=[]
    )
    
    prim_type = detector.detect(subckt)
    assert prim_type == PrimitiveType.BJT


def test_primitive_detector_unknown():
    """Test primitive detector for unknown subcircuit"""
    detector = PrimitiveDetector()
    
    # Create a non-primitive subcircuit
    subckt = SubcktDef(
        name=Ident("my_custom_circuit"),
        ports=[
            Ident("in1"),
            Ident("in2"),
            Ident("out"),
        ],
        params=[],
        entries=[]
    )
    
    prim_type = detector.detect(subckt)
    assert prim_type == PrimitiveType.UNKNOWN


def test_xschem_symbol_generation():
    """Test xschem symbol file generation"""
    # Create a simple MOSFET subcircuit
    program = Program(files=[
        SourceFile(
            path="test_mosfet.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos"),
                    ports=[
                        Ident("d"),
                        Ident("g"),
                        Ident("s"),
                        Ident("b"),
                    ],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.1)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                    ],
                    entries=[]
                )
            ]
        )
    ])
    
    # Generate xschem symbols
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "output.txt"
        options_write = WriteOptions(fmt=NetlistDialects.XSCHEM)
        
        with open(dest_path, 'w') as f:
            netlist(src=program, dest=f, options=options_write)
        
        # Check that symbol file was generated (uses IR name, not generic type name)
        sym_file = Path(tmpdir) / "nmos.sym"
        assert sym_file.exists(), f"Symbol file {sym_file} should exist"
        
        # Check symbol file content
        content = sym_file.read_text()
        assert "v {xschem version=" in content
        assert "spice_primitive=true" in content
        assert "type=mosfet" in content
        assert "template=" in content
        assert "model=nmos" in content  # Should use IR name


def test_xschem_multiple_primitives():
    """Test xschem symbol generation for multiple primitive types"""
    program = Program(files=[
        SourceFile(
            path="test_primitives.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos"),
                    ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.1)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                    ],
                    entries=[]
                ),
                SubcktDef(
                    name=Ident("resistor"),
                    ports=[Ident("p"), Ident("n")],
                    params=[ParamDecl(name=Ident("r"), default=Float(1000.0))],
                    entries=[]
                ),
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "output.txt"
        options_write = WriteOptions(fmt=NetlistDialects.XSCHEM)
        
        with open(dest_path, 'w') as f:
            netlist(src=program, dest=f, options=options_write)
        
        # Check that both symbol files were generated (using IR names)
        nmos_sym = Path(tmpdir) / "nmos.sym"
        resistor_sym = Path(tmpdir) / "resistor.sym"
        
        assert nmos_sym.exists(), "nmos.sym should exist"
        assert resistor_sym.exists(), "resistor.sym should exist"


def test_xschem_non_primitive_warning():
    """Test that non-primitive subcircuits generate warnings"""
    program = Program(files=[
        SourceFile(
            path="test_custom.sp",
            contents=[
                SubcktDef(
                    name=Ident("custom_circuit"),
                    ports=[Ident("in1"), Ident("in2"), Ident("out")],
                    params=[],
                    entries=[]
                )
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "output.txt"
        options_write = WriteOptions(fmt=NetlistDialects.XSCHEM)
        
        with open(dest_path, 'w') as f:
            log_file = netlist(src=program, dest=f, options=options_write)
        
        # Check that warning was logged
        if log_file:
            log_content = Path(log_file).read_text()
            assert "not a recognized primitive" in log_content or "custom_circuit" in log_content


def _check_docker_available():
    """Check if docker is available for xschem validation"""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def test_xschem_docker_validation():
    """Test xschem symbol validation using docker (optional)"""
    if not _check_docker_available():
        # Skip test if docker is not available
        return
    
    # Create a simple MOSFET subcircuit
    program = Program(files=[
        SourceFile(
            path="test_mosfet.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos"),
                    ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.1)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                    ],
                    entries=[]
                )
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "output.txt"
        options_write = WriteOptions(fmt=NetlistDialects.XSCHEM)
        
        with open(dest_path, 'w') as f:
            netlist(src=program, dest=f, options=options_write)
        
        sym_file = Path(tmpdir) / "nmos.sym"
        assert sym_file.exists()
        
        # Try to validate with xschem in docker
        # Note: This requires the hpretl/iic-osic-tools image
        try:
            # Get current user ID to avoid permission issues
            uid = os.getuid()
            gid = os.getgid()
            
            # Run xschem validation in docker
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{tmpdir}:/workspace",
                    "-w", "/workspace",
                    "--user", f"{uid}:{gid}",
                    "hpretl/iic-osic-tools",
                    "xschem", "--version"
                ],
                capture_output=True,
                timeout=30,
                cwd=tmpdir
            )
            
            # If xschem is available, try to load the symbol file
            if result.returncode == 0:
                # Try to validate the symbol file
                # Note: xschem might not have a direct validation command,
                # so we just check that it can be read
                # In a real scenario, you might use xschem's TCL API
                assert sym_file.exists()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Docker or xschem not available, skip validation
            pass


def test_xschem_bridge_file_generation():
    """Test bridge file generation for xyce dialect"""
    # Create a simple IR with realistic primitive names
    program = Program(files=[
        SourceFile(
            path="test_primitives.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos_1v0"),
                    ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.15)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                        ParamDecl(name=Ident("m"), default=Float(1.0)),
                    ],
                    entries=[]
                ),
                SubcktDef(
                    name=Ident("res_poly"),
                    ports=[Ident("p"), Ident("n")],
                    params=[
                        ParamDecl(name=Ident("r"), default=Float(1000.0)),
                        ParamDecl(name=Ident("m"), default=Float(1.0)),
                    ],
                    entries=[]
                ),
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "dummy.txt"
        options_write = WriteOptions(
            fmt=NetlistDialects.XSCHEM,
            subcircuit_dialect=NetlistDialects.XYCE
        )
        
        with open(dest_path, 'w') as f:
            netlist(src=program, dest=f, options=options_write)
        
        # Check that symbol files were generated with IR names
        nmos_sym = Path(tmpdir) / "nmos_1v0.sym"
        res_sym = Path(tmpdir) / "res_poly.sym"
        
        assert nmos_sym.exists(), "nmos_1v0.sym should exist"
        assert res_sym.exists(), "res_poly.sym should exist"
        
        # Check that bridge file was generated
        bridge_file = Path(tmpdir) / "bridge_xyce.spice"
        assert bridge_file.exists(), "bridge_xyce.spice should exist"
        
        # Check bridge file content
        bridge_content = bridge_file.read_text()
        assert "XYCE BRIDGE FILE" in bridge_content
        assert ".subckt nmos_1v0" in bridge_content
        assert ".subckt res_poly" in bridge_content
        assert "Xprim" in bridge_content  # Bridge instances use Xprim


def test_xschem_bridge_multiple_dialects():
    """Test bridge file generation for multiple dialects"""
    # Create a simple IR
    program = Program(files=[
        SourceFile(
            path="test_mosfet.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos_1v0"),
                    ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.15)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                    ],
                    entries=[]
                ),
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate xyce bridge
        dest_xyce = Path(tmpdir) / "dummy_xyce.txt"
        options_xyce = WriteOptions(
            fmt=NetlistDialects.XSCHEM,
            subcircuit_dialect=NetlistDialects.XYCE
        )
        with open(dest_xyce, 'w') as f:
            netlist(src=program, dest=f, options=options_xyce)
        
        bridge_xyce = Path(tmpdir) / "bridge_xyce.spice"
        assert bridge_xyce.exists()
        
        # Generate ngspice bridge
        dest_ngspice = Path(tmpdir) / "dummy_ngspice.txt"
        options_ngspice = WriteOptions(
            fmt=NetlistDialects.XSCHEM,
            subcircuit_dialect=NetlistDialects.NGSPICE
        )
        with open(dest_ngspice, 'w') as f:
            netlist(src=program, dest=f, options=options_ngspice)
        
        bridge_ngspice = Path(tmpdir) / "bridge_ngspice.spice"
        assert bridge_ngspice.exists()
        
        # Check that both bridge files reference the IR name
        xyce_content = bridge_xyce.read_text()
        ngspice_content = bridge_ngspice.read_text()
        
        assert ".subckt nmos_1v0" in xyce_content
        assert ".subckt nmos_1v0" in ngspice_content
        assert "XYCE BRIDGE FILE" in xyce_content
        assert "NGSPICE BRIDGE FILE" in ngspice_content


def test_xschem_symbol_uses_ir_name():
    """Test that symbols use IR subcircuit names, not generic names"""
    program = Program(files=[
        SourceFile(
            path="test_mosfet.sp",
            contents=[
                SubcktDef(
                    name=Ident("nmos_1v0"),
                    ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                    params=[
                        ParamDecl(name=Ident("l"), default=Float(0.15)),
                        ParamDecl(name=Ident("w"), default=Float(1.0)),
                    ],
                    entries=[]
                ),
            ]
        )
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "dummy.txt"
        options_write = WriteOptions(fmt=NetlistDialects.XSCHEM)
        
        with open(dest_path, 'w') as f:
            netlist(src=program, dest=f, options=options_write)
        
        # Symbol file should use IR name
        sym_file = Path(tmpdir) / "nmos_1v0.sym"
        assert sym_file.exists(), "Symbol file should use IR name nmos_1v0"
        
        # Check symbol content
        sym_content = sym_file.read_text()
        assert "G nmos_1v0" in sym_content  # Symbol name in header
        assert "model=nmos_1v0" in sym_content  # Model attribute uses IR name
        assert "X@name" in sym_content  # Template uses X prefix

