
"""
Integration tests for end-to-end conversions or complex scenarios.
Currently just a placeholder for future expansion as most existing tests
fit well into unit tests for parser/writer.
"""

from io import StringIO
from netlist import parse_files, netlist, NetlistDialects, ParseOptions, WriteOptions
from netlist.data import Program, SourceFile, ParamDecls, ParamDecl, Ident, Float, StatisticsBlock, Variation

def test_dummy_integration():
    assert True


def test_spectre_to_ngspice_basic_conversion():
    """Test basic Spectre to ngspice conversion"""
    # Create a simple Spectre-style program
    program = Program(files=[
        SourceFile(
            path="test.scs",
            contents=[
                ParamDecls(params=[
                    ParamDecl(name=Ident("vdd"), default=Float(1.8), distr=None),
                    ParamDecl(name=Ident("vss"), default=Float(0.0), distr=None),
                ])
            ]
        )
    ])
    
    # Convert to ngspice format
    dest = StringIO()
    options_write = WriteOptions(fmt=NetlistDialects.NGSPICE)
    netlist(src=program, dest=dest, options=options_write)
    output = dest.getvalue()
    
    # Verify ngspice syntax
    assert ".param" in output
    assert "vdd" in output
    assert "vss" in output
    assert "PARAMS:" not in output  # ngspice doesn't use PARAMS: keyword
    assert "*" in output or output.strip().startswith(".param")  # Should have comments or params


def test_spectre_to_ngspice_statistics_conversion():
    """Test Spectre to ngspice conversion with statistics blocks"""
    from netlist.write.spice import apply_statistics_variations
    
    # Create a program with statistics
    program = Program(files=[
        SourceFile(
            path="test.scs",
            contents=[
                ParamDecls(params=[
                    ParamDecl(name=Ident("vth0_nom"), default=Float(0.45), distr=None),
                ]),
                StatisticsBlock(
                    process=[
                        Variation(name=Ident("vth0_nom"), dist="gauss", std=Float(0.01), mean=None),
                    ],
                    mismatch=None
                )
            ]
        )
    ])
    
    # Apply statistics variations
    apply_statistics_variations(program, output_format=NetlistDialects.NGSPICE)
    
    # Convert to ngspice format
    dest = StringIO()
    options_write = WriteOptions(fmt=NetlistDialects.NGSPICE)
    netlist(src=program, dest=dest, options=options_write)
    output = dest.getvalue()
    
    # Verify statistics were processed
    assert ".param" in output
    assert "__process__" in output  # Process variation parameter should be created
    assert "{" in output  # Should use braces for expressions

