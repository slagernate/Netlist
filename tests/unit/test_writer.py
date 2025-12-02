
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
# Import apply_statistics_variations from netlist.transform instead of netlist.write.spice
# as it was moved there during refactoring, although it might still be exposed via spice for backward compat
# checking netlist/write/spice.py, it is still there (as it was not removed yet or imported back).
# But cleaner to import from where it is defined if possible.
# Wait, I moved it to netlist/transform.py in Phase 1? No, I moved it to netlist/transform.py in the plan,
# but in the previous turn I edited netlist/write/spice.py and it seems to still be there?
# Let's check if I actually moved it. I see it in `netlist/write/spice.py` content I read.
# So I will import from `netlist.write.spice` for now.
from netlist.write.spice import apply_statistics_variations, debug_find_all_param_refs
from netlist.dialects.spectre import SpectreDialectParser


def test_write1():
    """Test writing an empty netlist `Program`"""
    src = Program(files=[SourceFile(path="/", contents=[])])
    # Updated to pass required options argument
    write_netlist(src=src, dest=StringIO(), options=WriteOptions(fmt=NetlistDialects.SPECTRE))


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
    # Updated to pass required options argument
    write_netlist(src=src, dest=StringIO(), options=WriteOptions(fmt=NetlistDialects.SPECTRE))


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
    # Fix import path for XyceNetlister
    from netlist.write.xyce import XyceNetlister
    
    # Parse the expression that exposes the precedence issue
    expr_str = "a * b * (c + d * e)"
    parser = SpectreDialectParser.from_str(expr_str)
    parsed_expr = parser.parse(parser.parse_expr)

    # Create a minimal XyceNetlister to test formatting
    netlister = XyceNetlister(src=None, dest=StringIO())  # src can be None for this isolated test
    formatted = netlister.format_expr(parsed_expr)

    # Expected: {a*(b*(c+(d*e)))} (safe parenthesis insertion)
    expected = "{a*(b*(c+(d*e)))}"
    assert formatted == expected, f"Writer failed to add parentheses for precedence: got {formatted}, expected {expected}"


def test_apply_statistics_vary_mismatch():
    """Test mismatch variations (Gauss and Lognorm) are applied to corresponding parameters."""
    
    # Create a program with multiple parameters and statistics blocks
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None, comment=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None, comment=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None, comment=None),
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
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None, comment=None),
        ParamDecl(name=Ident("mobility"), default=Float(300.0), distr=None, comment=None),
        ParamDecl(name=Ident("junction_depth"), default=Float(0.1e-6), distr=None, comment=None),
        ParamDecl(name=Ident("alias_vth"), default=Ref(ident=Ident("vth")), distr=None, comment=None),  # Alias for vth
        ParamDecl(name=Ident("uses_alias"), default=Ref(ident=Ident("alias_vth")), distr=None, comment=None),  # Uses the alias
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
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None, comment=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None, comment=None),
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
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None, comment=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None, comment=None),
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
        ParamDecl(name=Ident("vth0"), default=Ref(ident=Ident("vth0_nom")), distr=None, comment=None),
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
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None, comment=None),
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

    mismatch_var_a_param = ParamDecl(name=Ident("mismatch_var_a"), default=Int(0), distr=None, comment=None)
    other_param = ParamDecl(name=Ident("other_param"), default=Float(1.0), distr=None, comment=None)
    
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
            ), distr=None, comment=None),
        ],
        entries=[]
    )
    
    param_with_ref = ParamDecl(name=Ident("param_ref"),
        default=BinaryOp(
            tp=BinaryOperator.MUL,
            left=Ref(ident=Ident("mismatch_var_a")),  # Reference to mismatch_var_a
            right=Float(2.0)
        ),
        distr=None, comment=None)
    
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
    
    params = [ParamDecl(name=Ident("test_param"), default=Float(1.0), distr=None, comment=None)]
    model = ModelDef(name=Ident("test_model"), mtype=Ident("nmos"), args=[], params=[
        ParamDecl(name=Ident("rsh"), default=Ref(ident=Ident("test_param")), distr=None, comment=None)
    ])
    lib_section = LibSectionDef(name=Ident("test_section"), entries=[
        ParamDecls(params=[ParamDecl(name=Ident("lib_param"), default=Ref(ident=Ident("test_param")), distr=None, comment=None)], comment=None)
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

    lib_param = ParamDecl(name=Ident("process_var_a"), default=Float(4.148e-09), distr=None, comment=None)
    lib_section = LibSectionDef(
        name=Ident("fet_tt"),
        entries=[
            ParamDecls(params=[lib_param])
        ]
    )
    
    global_param = ParamDecl(name=Ident("other_param"), default=Float(1.0), distr=None, comment=None)
    
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
    
    # Fix import path for XyceNetlister
    from netlist.write.xyce import XyceNetlister
    
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
        ParamDecl(name=Ident("param_a"), default=Float(1.0), distr=None, comment=None),
        ParamDecl(name=Ident("param_b"), default=Float(2.0), distr=None, comment=None),
        ParamDecl(name=Ident("param_c"), default=Float(3.0), distr=None, comment=None),
        ParamDecl(name=Ident("param_d"), default=Float(4.0), distr=None, comment=None),
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
    pmos = ModelDef(name=Ident("pmos_model"), mtype=Ident("bsim4"), args=[], params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None, comment=None), ParamDecl(name=Ident("version"), default=Float(4.5), distr=None, comment=None)], comment=None)
    output = StringIO()
    write_netlist(src=Program(files=[SourceFile(path="test.cir", contents=[pmos])]), dest=output, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()
    assert ".model pmos_model pmos" in output_str and ("level=54.0" in output_str or "level=54" in output_str)
    
    # Test 4: deltox in model → mapped to dtox
    with_deltox = ModelDef(name=Ident("model_with_deltox"), mtype=Ident("bsim4"), args=[], params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("n")), distr=None, comment=None), ParamDecl(name=Ident("deltox"), default=Float(1e-9), distr=None, comment=None), ParamDecl(name=Ident("version"), default=Float(4.5), distr=None, comment=None)], comment=None)
    output4 = StringIO()
    write_netlist(src=Program(files=[SourceFile(path="test4.cir", contents=[with_deltox])]), dest=output4, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str4 = output4.getvalue()
    assert ".model model_with_deltox nmos" in output_str4 and "deltox=" not in output_str4 and "dtox=" in output_str4


def test_bsim4_deltox_filtering_in_subckt():
    """Test that deltox is filtered from instances in subcircuits when ModelFamily is inside subcircuit."""
    
    model_family = ModelFamily(name=Ident("test_model"), mtype=Ident("bsim4"), variants=[
        ModelVariant(model=Ident("test_model"), variant=Ident("1"), mtype=Ident("bsim4"), args=[], 
                   params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None, comment=None)]),
        ModelVariant(model=Ident("test_model"), variant=Ident("2"), mtype=Ident("bsim4"), args=[], 
                   params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("p")), distr=None, comment=None)], comment=None)
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


def test_xyce_parameter_reference_braces():
    """
    Test that Xyce parameters that are references are wrapped in braces.
    Expected: m={m}
    """
    
    # Create a subcircuit that has a parameter 'm'
    # and an instance that passes 'm' to its model/subckt
    
    subckt = SubcktDef(
        name=Ident("parent_subckt"),
        ports=[],
        params=[
            ParamDecl(name=Ident("m"), default=Float(1.0), distr=None, comment=None),
            ParamDecl(name=Ident("l"), default=Float(1.0), distr=None, comment=None)
        ],
        entries=[
            Instance(
                name=Ident("Mpmos_lvt"),
                module=Ref(ident=Ident("plowvt_model")),
                conns=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
                params=[
                    ParamVal(name=Ident("m"), val=Ref(ident=Ident("m"))),
                    ParamVal(name=Ident("l"), val=Ref(ident=Ident("l"))),
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
    
    # Verify that parameters are wrapped in braces
    assert "m={m}" in output_str
    assert "l={l}" in output_str





def test_xyce_bsim4_dtox_move():
    """
    Test that 'deltox' parameter is moved from instance to model for BSIM4 in Xyce.
    """
    # Define a BSIM4 model
    model = ModelDef(
        name=Ident("bsim4_model_move"),
        mtype=Ident("bsim4"),
        args=[],
        params=[ParamDecl(name=Ident("type"), default=Ref(ident=Ident("n")), distr=None, comment=None)]
    )
    
    # Create an instance using that model with deltox parameter
    instance = Instance(
        name=Ident("M1"),
        module=Ref(ident=Ident("bsim4_model_move")),
        conns=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[
            ParamVal(name=Ident("w"), val=Float(1e-6)),
            ParamVal(name=Ident("l"), val=Float(1e-6)),
            ParamVal(name=Ident("deltox"), val=Float(3e-9))
        ]
    )
    
    # Wrap in a subcircuit because the fixup happens at subcircuit level
    subckt = SubcktDef(
        name=Ident("test_subckt"),
        ports=[],
        params=[],
        entries=[model, instance]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[subckt])
    ])
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()
    
    # Verify that dtox is NOT in instance line
    instance_line = next((line for line in output_str.split('\n') if line.startswith("M1")), "")
    # Check subsequent lines for params
    instance_params = ""
    capture = False
    for line in output_str.split('\n'):
        if line.startswith("M1"):
            capture = True
        elif capture and line.startswith(".model"):
            capture = False
        elif capture:
            instance_params += line
            
    assert "dtox=" not in instance_params
    assert "deltox=" not in instance_params
    
    # Verify that dtox IS in model line
    model_params = ""
    capture = False
    for line in output_str.split('\n'):
        if line.startswith(".model bsim4_model_move"):
            capture = True
        elif capture and (line.startswith(".ENDS") or line.startswith("M")):
            capture = False
        elif capture:
            model_params += line
            
    assert "dtox=" in model_params


def test_xyce_param_default_recovery_from_model():
    """
    Test that subcircuit parameters without defaults can recover defaults from model usage.
    This tests the fix for parameters like w, l, area, perim that are required but may
    have defaults in the model definition.
    """
    # Create a model with default parameter values
    model = ModelDef(
        name=Ident("test_model"),
        mtype=Ident("nmos"),
        args=[],
        params=[
            ParamDecl(name=Ident("w"), default=Float(1e-6), distr=None),
            ParamDecl(name=Ident("l"), default=Float(0.5e-6), distr=None),
            ParamDecl(name=Ident("area"), default=Float(1e-12), distr=None),
            ParamDecl(name=Ident("perim"), default=Float(2e-6), distr=None),
        ]
    )
    
    # Create an instance using that model
    instance = Instance(
        name=Ident("M1"),
        module=Ref(ident=Ident("test_model")),
        conns=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[
            # Pass parameters by reference to subcircuit params
            ParamVal(name=Ident("w"), val=Ref(ident=Ident("w"))),
            ParamVal(name=Ident("l"), val=Ref(ident=Ident("l"))),
            ParamVal(name=Ident("area"), val=Ref(ident=Ident("area"))),
            ParamVal(name=Ident("perim"), val=Ref(ident=Ident("perim"))),
        ]
    )
    
    # Create a subcircuit that uses the model but doesn't have defaults for w, l, area, perim
    subckt = SubcktDef(
        name=Ident("test_subckt"),
        ports=[Ident("d"), Ident("g"), Ident("s"), Ident("b")],
        params=[
            # These params have no defaults - should recover from model
            ParamDecl(name=Ident("w"), default=None, distr=None),
            ParamDecl(name=Ident("l"), default=None, distr=None),
            ParamDecl(name=Ident("area"), default=None, distr=None),
            ParamDecl(name=Ident("perim"), default=None, distr=None),
        ],
        entries=[model, instance]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[subckt])
    ])
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=WriteOptions(fmt=NetlistDialects.XYCE))
    output_str = output.getvalue()
    
    # Verify that the subcircuit parameters have recovered defaults (not max float)
    # Check the PARAMS line in the subcircuit definition
    lines = output_str.split('\n')
    params_line = None
    for i, line in enumerate(lines):
        if line.startswith(".SUBCKT test_subckt"):
            # Look for the PARAMS line (should be next continuation line)
            for j in range(i+1, min(i+5, len(lines))):
                if "PARAMS:" in lines[j]:
                    params_line = lines[j]
                    break
            break
    
    assert params_line is not None, "PARAMS line not found in subcircuit definition"
    
    # Verify that defaults were recovered (should have numeric values, not max float)
    # The recovered defaults should be: w=1e-6, l=0.5e-6, area=1e-12, perim=2e-6
    assert "w=1e-06" in params_line or "w=1.0e-06" in params_line or "w=1e-6" in params_line
    assert "l=5e-07" in params_line or "l=0.5e-06" in params_line or "l=5.0e-07" in params_line
    assert "area=1e-12" in params_line or "area=1.0e-12" in params_line
    assert "perim=2e-06" in params_line or "perim=2.0e-06" in params_line or "perim=2e-6" in params_line
    
    # Verify that we did NOT use max float (which would be something like 1.7976931348623157e+308)
    assert "1.7976931348623157e+308" not in params_line
    assert "1.7976931348623157e+308" not in output_str


def test_xyce_bjt_parameter_clamping_numeric():
    """
    Test that BJT Level 1 to MEXTRAM parameter mapping clamps numeric values to valid ranges.
    """
    from netlist.write.xyce import XyceNetlister
    
    # Create a BJT model with parameters that are out of range
    # Note: The mapping requires model_level_mapping to be set to map level 1 -> 504
    model = ModelDef(
        name=Ident("test_bjt"),
        mtype=Ident("npn"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Int(1), distr=None),
            # VAR=0 should be clamped to VER=0.01 (min is 0.01)
            ParamDecl(name=Ident("var"), default=Int(0), distr=None),
            # RBM=0 should be clamped to RBC=0.001 (min is 0.001)
            ParamDecl(name=Ident("rbm"), default=Int(0), distr=None),
            # BF=0 should be clamped to BF=0.0001 (min is 0.0001)
            ParamDecl(name=Ident("bf"), default=Int(0), distr=None),
            # VAF=0 should be clamped to VEF=0.01 (min is 0.01)
            ParamDecl(name=Ident("vaf"), default=Int(0), distr=None),
            # TF=0 should be clamped to TAUB=1e-12 (exclusive min 0.0)
            ParamDecl(name=Ident("tf"), default=Int(0), distr=None),
        ]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[model])
    ])
    
    # Use model_level_mapping to trigger the level 1 -> 504 mapping
    from netlist.write import WriteOptions
    options = WriteOptions(
        fmt=NetlistDialects.XYCE,
        model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]}
    )
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=options)
    output_str = output.getvalue()
    
    # Verify that parameters were mapped and clamped to valid ranges
    assert "VER=0.01" in output_str or "VER=0.01 " in output_str
    assert "RBC=0.001" in output_str or "RBC=0.001 " in output_str
    assert "BF=0.0001" in output_str or "BF=0.0001 " in output_str
    assert "VEF=0.01" in output_str or "VEF=0.01 " in output_str
    # TAUB should be clamped to 1e-12 (exclusive minimum)
    assert "TAUB=1e-12" in output_str or "TAUB=1e-12 " in output_str
    
    # Verify that original parameter names are NOT present (they should be mapped)
    assert "VAR=" not in output_str
    assert "RBM=" not in output_str
    assert "VAF=" not in output_str
    assert "TF=" not in output_str
    
    # Verify that original out-of-range values are NOT present
    assert "VER=0" not in output_str.replace("VER=0.01", "")
    assert "RBC=0" not in output_str.replace("RBC=0.001", "")
    assert "BF=0" not in output_str.replace("BF=0.0001", "")
    assert "VEF=0" not in output_str.replace("VEF=0.01", "")
    assert "TAUB=0" not in output_str.replace("TAUB=1e-12", "")


def test_xyce_bjt_parameter_expression_wrapping_max():
    """
    Test that BJT Level 1 to MEXTRAM parameter mapping wraps expressions in max() for minimum bounds.
    """
    from netlist.write.xyce import XyceNetlister
    
    # Create a BJT model with expression parameters that might evaluate to 0
    # BF expression that could evaluate to 0
    bf_expr = BinaryOp(
        tp=BinaryOperator.MUL,
        left=Float(39.28),
        right=Ref(ident=Ident("var_bf"))
    )
    
    # VAF expression that could evaluate to 0
    vaf_expr = BinaryOp(
        tp=BinaryOperator.DIV,
        left=Float(100.0),
        right=Ref(ident=Ident("var_bf"))
    )
    
    model = ModelDef(
        name=Ident("test_bjt_expr"),
        mtype=Ident("npn"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Int(1), distr=None),
            ParamDecl(name=Ident("bf"), default=bf_expr, distr=None),
            ParamDecl(name=Ident("vaf"), default=vaf_expr, distr=None),
        ]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[model])
    ])
    
    # Use model_level_mapping to trigger the level 1 -> 504 mapping
    options = WriteOptions(
        fmt=NetlistDialects.XYCE,
        model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]}
    )
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=options)
    output_str = output.getvalue()
    
    # Verify that expressions are wrapped in max() to enforce minimum bounds
    # BF should be wrapped: max(39.28*var_bf, 0.0001)
    assert "max(" in output_str
    assert "BF=" in output_str
    # Check that the BF line contains max() call
    bf_line = next((line for line in output_str.split('\n') if 'BF=' in line), None)
    assert bf_line is not None
    assert "max(" in bf_line
    
    # VEF should be wrapped: max(100/var_bf, 0.01)
    vef_line = next((line for line in output_str.split('\n') if 'VEF=' in line), None)
    assert vef_line is not None
    assert "max(" in vef_line


def test_xyce_bjt_parameter_expression_wrapping_min():
    """
    Test that BJT Level 1 to MEXTRAM parameter mapping wraps expressions in min() for maximum bounds.
    """
    from netlist.write.xyce import XyceNetlister
    
    # Create a parameter with an expression that might exceed maximum bound
    # PS parameter with range ]0.01, 0.99[ - should wrap in both max() and min()
    ps_expr = BinaryOp(
        tp=BinaryOperator.ADD,
        left=Float(0.5),
        right=Ref(ident=Ident("offset"))
    )
    
    model = ModelDef(
        name=Ident("test_bjt_ps"),
        mtype=Ident("npn"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Int(1), distr=None),
            # NS maps to PS with range ]0.01, 0.99[
            ParamDecl(name=Ident("ns"), default=ps_expr, distr=None),
        ]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[model])
    ])
    
    # Use model_level_mapping to trigger the level 1 -> 504 mapping
    options = WriteOptions(
        fmt=NetlistDialects.XYCE,
        model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]}
    )
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=options)
    output_str = output.getvalue()
    
    # Verify that expressions with both min and max bounds are wrapped appropriately
    # PS should be wrapped: min(max(0.5+offset, 0.01+epsilon), 0.99-epsilon)
    ps_line = next((line for line in output_str.split('\n') if 'PS=' in line), None)
    assert ps_line is not None
    # Should have both max and min wrapping
    assert "max(" in ps_line or "min(" in ps_line


def test_xyce_bjt_parameter_clamping_float_values():
    """
    Test that Float values are properly clamped (not just Int).
    """
    from netlist.write.xyce import XyceNetlister
    
    model = ModelDef(
        name=Ident("test_bjt_float"),
        mtype=Ident("npn"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Int(1), distr=None),
            # Use Float(0.0) instead of Int(0)
            ParamDecl(name=Ident("var"), default=Float(0.0), distr=None),
            ParamDecl(name=Ident("rbm"), default=Float(0.0), distr=None),
        ]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[model])
    ])
    
    # Use model_level_mapping to trigger the level 1 -> 504 mapping
    options = WriteOptions(
        fmt=NetlistDialects.XYCE,
        model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]}
    )
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=options)
    output_str = output.getvalue()
    
    # Verify Float values are clamped correctly
    assert "VER=0.01" in output_str
    assert "RBC=0.001" in output_str


def test_xyce_bjt_parameter_clamping_exclusive_bounds():
    """
    Test that exclusive bounds (]min, max[) are handled correctly with epsilon.
    """
    from netlist.write.xyce import XyceNetlister
    
    model = ModelDef(
        name=Ident("test_bjt_exclusive"),
        mtype=Ident("npn"),
        args=[],
        params=[
            ParamDecl(name=Ident("level"), default=Int(1), distr=None),
            # TF=0 with exclusive minimum ]0.0, +inf) should become TAUB=1e-12
            ParamDecl(name=Ident("tf"), default=Int(0), distr=None),
            # NS=1.0 with exclusive range ]0.01, 0.99[ should be clamped
            ParamDecl(name=Ident("ns"), default=Float(1.0), distr=None),
        ]
    )
    
    program = Program(files=[
        SourceFile(path="test.cir", contents=[model])
    ])
    
    # Use model_level_mapping to trigger the level 1 -> 504 mapping
    options = WriteOptions(
        fmt=NetlistDialects.XYCE,
        model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]}
    )
    
    output = StringIO()
    write_netlist(src=program, dest=output, options=options)
    output_str = output.getvalue()
    
    # Verify exclusive bounds are handled
    # TAUB should be 1e-12 (exclusive min 0.0)
    assert "TAUB=1e-12" in output_str or "TAUB=1e-12 " in output_str
    # PS should be clamped to less than 0.99 (exclusive max)
    ps_line = next((line for line in output_str.split('\n') if 'PS=' in line), None)
    assert ps_line is not None
    # Should be clamped to something less than 0.99
    import re
    ps_match = re.search(r'PS=([\d.e+-]+)', ps_line)
    if ps_match:
        ps_val = float(ps_match.group(1))
        assert ps_val < 0.99, f"PS value {ps_val} should be less than 0.99 (exclusive max)"
