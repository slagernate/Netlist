
from io import StringIO
from pathlib import Path

from netlist import (
    netlist as write_netlist,
    Program,
    SourceFile,
    Options,
    Option,
    Ident,
    MetricNum,
    SourceInfo,
    NetlistDialects,
    ParamDecl,
    ParamDecls,
    Float,
    Int,
    Ref,
    BinaryOp,
    BinaryOperator,
    Call,
    FunctionDef,
    ArgType,
    TypedArg,
    Return,
    StartSubckt,
    EndSubckt,
    StatisticsBlock,
    Variation,
    SubcktDef,
    Instance,
    ParamVal,
    ModelDef,
    ModelFamily,
    ModelVariant,
    LibSectionDef,
    Library,
    Expr,
)
from netlist.write import WriteOptions
from netlist.write.spice import apply_statistics_variations, debug_find_all_param_refs
from netlist.dialects.spectre import SpectreDialectParser


def test_write1():
    """Test writing an empty netlist `Program`"""
    src = Program(files=[SourceFile(path="/", contents=[])])
    write_netlist(src=src, dest=StringIO())


def test_write2():
    """Test writing some actual content"""

    src = Program(
        files=[
            SourceFile(
                path="/",
                contents=[
                    Options(
                        name=None,
                        vals=[
                            Option(
                                name=Ident(name="scale"),
                                val=MetricNum(val="1.0u"),
                            )
                        ],
                        source_info=SourceInfo(
                            line=15, dialect=NetlistDialects.SPECTRE_SPICE
                        ),
                    )
                ],
            )
        ]
    )
    write_netlist(src=src, dest=StringIO())


def test_write_xyce_func():
    """Test writing Xyce .FUNC definitions"""
    
    # Create a simple function: test_func_a() {gauss(0, 0.1, 1)}
    func = FunctionDef(
        name=Ident("test_func_a"),
        rtype=ArgType.REAL,
        args=[],  # No arguments
        stmts=[
            Return(
                val=Call(
                    func=Ref(ident=Ident("gauss")),
                    args=[Int(0), Float(0.1), Int(1)]
                )
            )
        ]
    )
    
    src = Program(
        files=[
            SourceFile(
                path="/",
                contents=[func]
            )
        ]
    )
    
    # Write using Xyce format
    dest = StringIO()
    write_netlist(src=src, dest=dest, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output = dest.getvalue()
    
    # Verify output format: .FUNC test_func_a() { gauss(0,0.1,1) }
    assert ".FUNC" in output
    assert "test_func_a()" in output
    assert "gauss" in output
    # Check that it's properly formatted
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    func_line = next((line for line in lines if ".FUNC" in line), None)
    assert func_line is not None
    assert func_line.startswith(".FUNC")
    assert "test_func_a()" in func_line
    assert "{" in func_line
    assert "}" in func_line


def test_write_xyce_func_with_args():
    """Test writing Xyce .FUNC definitions with arguments"""
    
    # Create a function with arguments: lnorm(mu, sigma, seed) { exp(gauss(mu, sigma, seed)) }
    func = FunctionDef(
        name=Ident("lnorm"),
        rtype=ArgType.REAL,
        args=[
            TypedArg(tp=ArgType.REAL, name=Ident("mu")),
            TypedArg(tp=ArgType.REAL, name=Ident("sigma")),
            TypedArg(tp=ArgType.REAL, name=Ident("seed")),
        ],
        stmts=[
            Return(
                val=Call(
                    func=Ref(ident=Ident("exp")),
                    args=[
                        Call(
                            func=Ref(ident=Ident("gauss")),
                            args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma")), Ref(ident=Ident("seed"))]
                        )
                    ]
                )
            )
        ]
    )
    
    src = Program(
        files=[
            SourceFile(
                path="/",
                contents=[func]
            )
        ]
    )
    
    # Write using Xyce format
    dest = StringIO()
    write_netlist(src=src, dest=dest, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output = dest.getvalue()
    
    # Verify output format: .FUNC lnorm(mu,sigma,seed) { exp(gauss(mu,sigma,seed)) }
    assert ".FUNC" in output
    assert "lnorm" in output
    assert "exp" in output
    assert "gauss" in output
    # Check that arguments are included
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    func_line = next((line for line in lines if ".FUNC" in line), None)
    assert func_line is not None
    assert "lnorm(" in func_line
    assert "mu" in func_line or "sigma" in func_line  # Arguments should be present


def test_writer_parentheses_precedence():
    """Test that the writer adds parentheses for operator precedence in expressions"""
    from netlist.write.spice import XyceNetlister
    
    # Parse the expression that exposes the precedence issue
    expr_str = "a * b * (c + d * e)"
    parser = SpectreDialectParser.from_str(expr_str)
    parsed_expr = parser.parse(parser.parse_expr)

    # Create a minimal XyceNetlister to test formatting
    netlister = XyceNetlister(src=None, dest=StringIO())  # src can be None for this isolated test
    formatted = netlister.format_expr(parsed_expr)

    # Expected: {a*b*(c+d*e)} (with parentheses around the addition)
    # Current buggy output: {a*b*c+d*e} (missing parentheses, which changes meaning)
    expected = "{a*b*(c+d*e)}"
    assert formatted == expected, f"Writer failed to add parentheses for precedence: got {formatted}, expected {expected}"


def test_apply_statistics_vary_mismatch():
    """Test mismatch variations (Gauss and Lognorm) are applied to corresponding parameters."""
    
    # Create a program with multiple parameters and statistics blocks
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None),
    ]
    
    stats = StatisticsBlock(
        process=None,
        mismatch=[
            Variation(name=Ident("width"), dist="gauss", std=Float(0.1), mean=None),  # Gaussian mismatch
            Variation(name=Ident("length"), dist="gauss", std=Float(0.05), mean=None),  # Gaussian mismatch
            Variation(name=Ident("tox"), dist="lnorm", std=Float(0.15), mean=None),  # Lognormal mismatch
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Apply variations
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # NEW BEHAVIOR: Parameters with mismatch variations are REMOVED and replaced with function calls
    # Verify enable_mismatch parameter was NOT added to the generated library
    enable_mismatch_param = next(
        (param for file in program.files
         for entry in file.contents
         if isinstance(entry, ParamDecls)
         for param in entry.params
         if param.name.name == "enable_mismatch"),
        None
    )
    assert enable_mismatch_param is None
    
    # Verify width, length, and tox parameters were REMOVED
    all_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
    ]
    param_names = [p.name.name for p in all_params]
    assert "width" not in param_names
    assert "length" not in param_names
    assert "tox" not in param_names
    
    # Verify that mismatch param functions were added to the program
    mismatch_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
        if param.name.name in ["width__mismatch__(dummy_param)", "length__mismatch__(dummy_param)", "tox__mismatch__(dummy_param)"]
    ]
    assert len(mismatch_params) == 3
    
    # Verify width param function uses gauss with enable_mismatch multiplier
    width_param = next(p for p in mismatch_params if p.name.name == "width__mismatch__(dummy_param)")
    width_expr = width_param.default
    assert isinstance(width_expr, BinaryOp)
    assert width_expr.tp == BinaryOperator.ADD
    assert isinstance(width_expr.left, Float) and width_expr.left.val == 1.0
    
    # Verify tox param function uses lnorm with enable_mismatch multiplier
    tox_param = next(p for p in mismatch_params if p.name.name == "tox__mismatch__(dummy_param)")
    tox_expr = tox_param.default
    assert isinstance(tox_expr, BinaryOp)
    assert tox_expr.tp == BinaryOperator.ADD
    assert isinstance(tox_expr.left, Float) and tox_expr.left.val == 2.0e-9


def test_apply_statistics_vary_process():
    """Test process variations are applied to corresponding parameters."""
    
    params = [
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None),
        ParamDecl(name=Ident("mobility"), default=Float(300.0), distr=None),
        ParamDecl(name=Ident("junction_depth"), default=Float(0.1e-6), distr=None),
        ParamDecl(name=Ident("alias_vth"), default=Ref(ident=Ident("vth")), distr=None),  # Alias for vth
        ParamDecl(name=Ident("uses_alias"), default=Ref(ident=Ident("alias_vth")), distr=None),  # Uses the alias
    ]
    
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.05), mean=Float(0.1)),  # Fast corner: +0.1V
            Variation(name=Ident("mobility"), dist="gauss", std=Float(10.0), mean=Float(-50.0)),  # Slow corner: -50 cm²/V·s
            Variation(name=Ident("junction_depth"), dist="gauss", std=Float(0.01e-6), mean=Float(0.02e-6)),  # Deep junction corner
        ],
        mismatch=None
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # For Xyce format, original parameters are REMOVED and all references are replaced with __process__ version
    all_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
    ]
    param_names = [p.name.name for p in all_params]
    assert "vth" not in param_names
    assert "mobility" not in param_names
    assert "junction_depth" not in param_names
    assert "alias_vth" not in param_names
    
    # Verify alias uses are replaced with __process__ version
    uses_alias_param = next((p for p in all_params if p.name.name == "uses_alias"), None)
    assert uses_alias_param is not None
    assert isinstance(uses_alias_param.default, Ref)
    assert uses_alias_param.default.ident.name == "vth__process__"


def test_apply_statistics_vary_no_match():
    """Test that variations are ignored if no corresponding parameter exists."""
    
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None),
    ]
    
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("non_existent_param1"), dist="gauss", std=Float(0.1), mean=Float(2.0)),
            Variation(name=Ident("non_existent_param2"), dist="lnorm", std=Float(0.2), mean=None),
        ],
        mismatch=[
            Variation(name=Ident("non_existent_param3"), dist="gauss", std=Float(0.15), mean=None),
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Store original values
    original_width = params[0].default
    original_length = params[1].default
    
    import pytest
    with pytest.raises(RuntimeError, match="Process variation parameters not found"):
        apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Parameters should remain unchanged since error was raised before applying
    assert params[0].default == original_width
    assert params[1].default == original_length


def test_apply_statistics_vary_spectre_no_xyce_transformations():
    """Test that Xyce-specific transformations are NOT applied when output format is Spectre."""
    
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None),
    ]
    
    stats = StatisticsBlock(
        process=None,
        mismatch=[
            Variation(name=Ident("width"), dist="gauss", std=Float(0.1), mean=None),
            Variation(name=Ident("tox"), dist="lnorm", std=Float(0.15), mean=None),
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    apply_statistics_variations(program, output_format=NetlistDialects.SPECTRE)

    # Verify enable_mismatch parameter was NOT added
    enable_mismatch_param = next(
        (param for file in program.files
         for entry in file.contents
         if isinstance(entry, ParamDecls)
         for param in entry.params
         if param.name.name == "enable_mismatch"),
        None
    )
    assert enable_mismatch_param is None

    # Verify mismatch functions were NOT created
    mismatch_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name.endswith("_mismatch")
    ]
    assert len(mismatch_funcs) == 0


def test_param_process_and_mismatch():
    """Test that parameters with both process and mismatch variations are handled correctly."""
    
    params = [
        ParamDecl(name=Ident("vth0"), default=Ref(ident=Ident("vth0_nom")), distr=None),
    ]
    
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("vth0"), dist="gauss", std=Float(0.01), mean=None),
        ],
        mismatch=[
            Variation(name=Ident("vth0"), dist="gauss", std=Float(0.002), mean=None),
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])
    
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)
    
    # Verify the original parameter was removed (process removes it first, then mismatch removes __process__)
    all_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
    ]
    param_names = [p.name.name for p in all_params]
    assert "vth0" not in param_names
    assert "vth0__process__" not in param_names
    
    # Verify process__mismatch param function was created
    mismatch_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
        if param.name.name == "vth0__process____mismatch__(dummy_param)"
    ]
    assert len(mismatch_params) == 1


def test_param_mismatch_and_corner():
    """Test that parameters with both process corner and mismatch variations are handled correctly."""
    
    params = [
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None),
    ]
    
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.05), mean=Float(0.1)),  # Process variation
        ],
        mismatch=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.02), mean=None),  # Mismatch variation
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Verify vth parameter was REMOVED (mismatch variation removes it)
    all_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
    ]
    param_names = [p.name.name for p in all_params]
    assert "vth" not in param_names
    
    # Verify mismatch param function was created with both process and mismatch suffixes
    mismatch_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
        if param.name.name == "vth__process____mismatch__(dummy_param)"
    ]
    assert len(mismatch_params) == 1


def test_mismatch_parameter_reference_replacement():
    """Test that mismatch variations replace ALL parameter references and remove the declaration."""

    mismatch_var_a_param = ParamDecl(name=Ident("mismatch_var_a"), default=Int(0), distr=None)
    other_param = ParamDecl(name=Ident("other_param"), default=Float(1.0), distr=None)
    
    instance = Instance(
        name=Ident("X1"),
        module=Ref(ident=Ident("subckt1")),
        conns=[],
        params=[
            ParamVal(name=Ident("param1"), val=Ref(ident=Ident("mismatch_var_a"))),  # Reference to mismatch_var_a
            ParamVal(name=Ident("param2"), val=Float(2.0)),
        ]
    )
    
    subckt = SubcktDef(
        name=Ident("subckt1"),
        ports=[],
        params=[
            ParamDecl(name=Ident("sub_param"), default=BinaryOp(
                tp=BinaryOperator.ADD,
                left=Ref(ident=Ident("mismatch_var_a")),  # Reference to mismatch_var_a in expression
                right=Float(1.0)
            ), distr=None),
        ],
        entries=[]
    )
    
    param_with_ref = ParamDecl(
        name=Ident("param_ref"),
        default=BinaryOp(
            tp=BinaryOperator.MUL,
            left=Ref(ident=Ident("mismatch_var_a")),  # Reference to mismatch_var_a
            right=Float(2.0)
        ),
        distr=None
    )
    
    stats = StatisticsBlock(
        process=None,
        mismatch=[
            Variation(name=Ident("mismatch_var_a"), dist="gauss", std=Float(0.1), mean=None),
        ]
    )
    
    program = Program(files=[SourceFile(
        path="test",
        contents=[
            ParamDecls(params=[mismatch_var_a_param, other_param, param_with_ref]),
            stats,
            instance,
            subckt,
        ]
    )])
    
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)
    
    # Verify the parameter declaration was REMOVED
    all_params = [
        param for file in program.files
        for entry in file.contents
        if isinstance(entry, ParamDecls)
        for param in entry.params
    ]
    param_names = [p.name.name for p in all_params]
    assert "mismatch_var_a" not in param_names
    
    # Verify instance parameter reference was replaced
    instance_param1 = next(p for p in instance.params if p.name.name == "param1")
    assert isinstance(instance_param1.val, Call)
    assert instance_param1.val.func.ident.name == "mismatch_var_a__mismatch__"


def test_parameter_replacement_coverage_and_verification():
    """Test that parameter replacement covers all entry types and verification catches unreplaced refs."""
    
    params = [ParamDecl(name=Ident("test_param"), default=Float(1.0), distr=None)]
    model = ModelDef(name=Ident("test_model"), mtype=Ident("nmos"), args=[], params=[
        ParamDecl(name=Ident("rsh"), default=Ref(ident=Ident("test_param")), distr=None)
    ])
    lib_section = LibSectionDef(name=Ident("test_section"), entries=[
        ParamDecls(params=[ParamDecl(name=Ident("lib_param"), default=Ref(ident=Ident("test_param")), distr=None)])
    ])
    library = Library(name=Ident("test_lib"), sections=[lib_section])
    
    stats = StatisticsBlock(process=[Variation(name=Ident("test_param"), dist="gauss", std=Float(0.1), mean=None)], mismatch=None)
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), model, library, stats])])
    
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)
    
    # Verify no unreplaced references remain
    remaining = debug_find_all_param_refs(program, "test_param")
    assert len(remaining) == 0


def test_process_variation_finds_param_in_library_section():
    """Test that process variations can find and apply to parameters in library sections."""

    lib_param = ParamDecl(name=Ident("process_var_a"), default=Float(4.148e-09), distr=None)
    lib_section = LibSectionDef(
        name=Ident("fet_tt"),
        entries=[
            ParamDecls(params=[lib_param])
        ]
    )
    
    global_param = ParamDecl(name=Ident("other_param"), default=Float(1.0), distr=None)
    
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("process_var_a"), dist="gauss", std=Float(0.05), mean=Float(0.1)),
        ],
        mismatch=None
    )
    
    program = Program(files=[SourceFile(
        path="test",
        contents=[
            ParamDecls(params=[global_param]),
            lib_section,
            stats,
        ]
    )])
    
    from netlist.write.spice import XyceNetlister
    
    output = StringIO()
    netlister = XyceNetlister(program, output)
    netlister.netlist()
    output_str = output.getvalue()
    
    # Verify process variation was written AFTER all library sections
    assert ".endl fet_tt" in output_str
    endl_pos = output_str.find(".endl fet_tt")
    param_pos = output_str.find(".param", endl_pos)
    assert param_pos > endl_pos


def test_spectre_statistics_param_generation():
    """Test that Spectre statistics blocks generate correct .param declarations for Xyce"""

    stats_block = StatisticsBlock(
        process=[
            Variation(name=Ident("param_a"), dist="gauss", std=Float(0.1), mean=None),
            Variation(name=Ident("param_b"), dist="gauss", std=Float(0.05), mean=None),
        ],
        mismatch=[
            Variation(name=Ident("param_c"), dist="lnorm", std=Float(0.05), mean=None),
            Variation(name=Ident("param_d"), dist="gauss", std=Float(0.03), mean=None),
        ]
    )

    param_decls = ParamDecls(params=[
        ParamDecl(name=Ident("param_a"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("param_b"), default=Float(2.0), distr=None),
        ParamDecl(name=Ident("param_c"), default=Float(3.0), distr=None),
        ParamDecl(name=Ident("param_d"), default=Float(4.0), distr=None),
    ])

    program = Program(files=[
        SourceFile(path=Path("test.scs"), contents=[param_decls, stats_block])
    ])

    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    lnorm_params = []
    mismatch_params = []

    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                for param in entry.params:
                    if 'lnorm' in param.name.name:
                        lnorm_params.append(param)
                    elif 'mismatch' in param.name.name:
                        mismatch_params.append(param)

    assert len(lnorm_params) == 2
    assert len(mismatch_params) == 2


def test_primitive_instance_no_params_keyword():
    """Test that primitive instances (MOS devices) never have PARAMS: keyword in Xyce output"""
    
    subckt = SubcktDef(
        name=Ident("pmos_lvt"),
        ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[],
        entries=[
            Instance(
                name=Ident("Mpmos_lvt"),
                module=Ref(ident=Ident("plowvt_model")),
                conns=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                params=[
                    ParamVal(name=Ident("l"), val=Float(1.0)),
                    ParamVal(name=Ident("w"), val=Float(2.0)),
                    ParamVal(name=Ident("nf"), val=Int(1)),
                ]
            )
        ]
    )

    program = Program(files=[
        SourceFile(path="test.cir", contents=[subckt])
    ])

    output = StringIO()
    write_netlist(src=program, dest=output, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()

    assert "Mpmos_lvt" in output_str
    assert "PARAMS:" not in output_str
    
    # Also verify that regular subcircuit instances DO have PARAMS:
    subckt2 = SubcktDef(
        name=Ident("test_subckt"),
        ports=[Ident("in"), Ident("out")],
        params=[],
        entries=[
            Instance(
                name=Ident("X1"),
                module=Ref(ident=Ident("other_subckt")),
                conns=[Ident("in"), Ident("out")],
                params=[
                    ParamVal(name=Ident("param1"), val=Float(1.0)),
                ]
            )
        ]
    )

    program2 = Program(files=[
        SourceFile(path="test2.cir", contents=[subckt2])
    ])

    output2 = StringIO()
    write_netlist(src=program2, dest=output2, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str2 = output2.getvalue()

    assert "PARAMS:" in output_str2


def test_bsim4_model_translation():
    """Test that Spectre BSIM4 model cards get translated properly to Xyce format."""
    
    # Test 1: type={p} → pmos with level=54
    pmos = ModelDef(name=Ident("pmos_model"), mtype=Ident("bsim4"), args=[], params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None), ParamDecl(name=Ident("version"), default=Float(4.5), distr=None)])
    output = StringIO()
    write_netlist(src=Program(files=[SourceFile(path="test.cir", contents=[pmos])]), dest=output, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()
    assert ".model pmos_model pmos" in output_str and ("level=54.0" in output_str or "level=54" in output_str)
    
    # Test 4: deltox in model → mapped to dtox
    with_deltox = ModelDef(name=Ident("model_with_deltox"), mtype=Ident("bsim4"), args=[], params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("n")), distr=None), ParamDecl(name=Ident("deltox"), default=Float(1e-9), distr=None), ParamDecl(name=Ident("version"), default=Float(4.5), distr=None)])
    output4 = StringIO()
    write_netlist(src=Program(files=[SourceFile(path="test4.cir", contents=[with_deltox])]), dest=output4, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str4 = output4.getvalue()
    assert ".model model_with_deltox nmos" in output_str4 and "deltox=" not in output_str4 and "dtox=" in output_str4


def test_bsim4_deltox_filtering_in_subckt():
    """Test that deltox is filtered from instances in subcircuits when ModelFamily is inside subcircuit."""
    
    model_family = ModelFamily(name=Ident("test_model"), mtype=Ident("bsim4"), variants=[
        ModelVariant(model=Ident("test_model"), variant=Ident("1"), mtype=Ident("bsim4"), args=[], 
                   params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None)]),
        ModelVariant(model=Ident("test_model"), variant=Ident("2"), mtype=Ident("bsim4"), args=[], 
                   params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None)])
    ])
    subckt = SubcktDef(name=Ident("test_pmos"), ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")], 
                      params=[], entries=[
        Instance(name=Ident("test_pmos"), module=Ref(ident=Ident("test_model")), 
                conns=[Ident("d"), Ident("g"), Ident("s"), Ident("b")], 
                params=[ParamVal(name=Ident("l"), val=Float(1.0)), ParamVal(name=Ident("w"), val=Float(2.0)), 
                       ParamVal(name=Ident("deltox"), val=Float(1e-9)), ParamVal(name=Ident("delvto"), val=Float(0.1))]),
        model_family
    ])
    output = StringIO()
    write_netlist(src=Program(files=[SourceFile(path="test.cir", contents=[subckt])]), dest=output, 
                 options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()
    instance_section = output_str.split("Mtest_pmos")[1].split("\n\n")[0]
    assert "deltox=" not in instance_section and "dtox=" not in instance_section
    assert "l=" in instance_section and "w=" in instance_section
    assert "delvto=" in instance_section

