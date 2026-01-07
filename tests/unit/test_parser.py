
from textwrap import dedent
from io import StringIO
from pathlib import Path

import netlist
from netlist import (
    parse_files,
    parse_str,
    ParseOptions,
    Program,
    SourceFile,
    ParamDecl,
    ModelDef,
    ModelVariant,
    ModelFamily,
    BinaryOperator,
    UnaryOperator,
    ParamVal,
    Ident,
    MetricNum,
    NetlistDialects,
    SpiceDialectParser,
    BinaryOp,
    SpectreDialectParser,
    SpectreSpiceDialectParser,
    Int,
    Float,
    UnaryOp,
    Call,
    Ref,
    StartProtectedSection,
    EndProtectedSection,
    StartSubckt,
    Primitive,
    Instance,
    Include,
    ParamDecls,
    FunctionDef,
    ArgType,
    TypedArg,
    Return,
    TernOp,
    Scope,
    has_external_refs,
    get_external_refs,
    StatisticsBlock,
    Variation,
    SubcktDef,
)


def test_spice_exprs():
    """Test parsing spice-format expressions"""

    def parse_expression(s: str) -> SpiceDialectParser:
        """Parse a string expression, including placing the parser in EXPR mode first,
        particularly so that elements like "*" are interpreted as multiplication."""
        from netlist.dialects.base import ParserState

        parser = SpiceDialectParser.from_str(s)
        parser.state = ParserState.EXPR
        return parser.parse(parser.parse_expr)

    p = parse_expression(" ' a + b ' ")  # SPICE-style ticked-expression
    assert p == BinaryOp(
        tp=BinaryOperator.ADD,
        left=Ref(ident=Ident(name="a")),
        right=Ref(ident=Ident(name="b")),
    )


def test_spectre_exprs():
    def parse_expression(s: str) -> SpectreDialectParser:
        """Parse a string expression"""
        from netlist.dialects.base import ParserState

        parser = SpectreDialectParser.from_str(s)
        parser.state = ParserState.EXPR
        return parser.parse(parser.parse_expr)

    p = parse_expression("1")
    assert p == Int(1)
    p = parse_expression("1+2")
    assert p == BinaryOp(BinaryOperator.ADD, Int(1), Int(2))
    p = parse_expression("1+2*3")
    assert p == BinaryOp(
        BinaryOperator.ADD, Int(1), BinaryOp(BinaryOperator.MUL, Int(2), Int(3))
    )
    p = parse_expression("1*2+3")
    assert p == BinaryOp(
        BinaryOperator.ADD,
        left=BinaryOp(BinaryOperator.MUL, Int(1), Int(2)),
        right=Int(3),
    )

    p = parse_expression("(1+2)*(3+4)")
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=BinaryOp(tp=BinaryOperator.ADD, left=Int(val=1), right=Int(val=2)),
        right=BinaryOp(tp=BinaryOperator.ADD, left=Int(val=3), right=Int(val=4)),
    )

    p = parse_expression("a     ")
    assert p == Ref(ident=Ident("a"))
    p = parse_expression("   b + 1     ")
    assert p == BinaryOp(BinaryOperator.ADD, Ref(ident=Ident("b")), Int(1))

    p = parse_expression("1e-3")
    assert p == Float(1e-3)
    p = parse_expression("1.0")
    assert p == Float(1.0)
    p = parse_expression("1.")
    assert p == Float(1.0)
    p = parse_expression(".1")
    assert p == Float(0.1)
    p = parse_expression("1e-3 + 2. * .3")
    assert p == BinaryOp(
        BinaryOperator.ADD,
        Float(1e-3),
        BinaryOp(BinaryOperator.MUL, Float(2.0), Float(0.3)),
    )

    p = parse_expression("r*l/w")
    # Multiplication and division are same precedence and left-associative: (r*l)/w
    assert p == BinaryOp(
        tp=BinaryOperator.DIV,
        left=BinaryOp(
            tp=BinaryOperator.MUL,
            left=Ref(ident=Ident(name="r")),
            right=Ref(ident=Ident(name="l")),
        ),
        right=Ref(ident=Ident(name="w")),
    )

    p = parse_expression("(0.5f * p)")  # SPICE metric-suffixed number
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=MetricNum(val="0.5f"),
        right=Ref(ident=Ident(name="p")),
    )

    p = parse_expression(" a + func(b, c) ")  # Function call
    assert p == BinaryOp(
        tp=BinaryOperator.ADD,
        left=Ref(ident=Ident(name="a")),
        right=Call(
            func=Ref(ident=Ident(name="func")),
            args=[Ref(ident=Ident(name="b")), Ref(ident=Ident(name="c"))],
        ),
    )

    p = parse_expression(" - a ")  # Unary operator
    assert p == UnaryOp(tp=UnaryOperator.NEG, targ=Ref(ident=Ident(name="a")))

    p = parse_expression(" - + + - a ")  # Unary operator(s!)
    assert p == UnaryOp(
        tp=UnaryOperator.NEG,
        targ=UnaryOp(
            tp=UnaryOperator.PLUS,
            targ=UnaryOp(
                tp=UnaryOperator.PLUS,
                targ=UnaryOp(tp=UnaryOperator.NEG, targ=Ref(ident=Ident(name="a"))),
            ),
        ),
    )

    p = parse_expression(" -5 * -3 ")  # Mixture of unary & binary ops
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=UnaryOp(tp=UnaryOperator.NEG, targ=Int(val=5)),
        right=UnaryOp(tp=UnaryOperator.NEG, targ=Int(val=3)),
    )

    p = parse_expression(" 3 ** 4 * 2  ")  # Mixture of unary & binary ops
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=BinaryOp(tp=BinaryOperator.POW, left=Int(val=3), right=Int(val=4)),
        right=Int(val=2),
    )
    p = parse_expression(" 2 * 3 ** 4 ")  # Mixture of unary & binary ops
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=Int(val=2),
        right=BinaryOp(tp=BinaryOperator.POW, left=Int(val=3), right=Int(val=4)),
    )


import pytest

def test_primitive():
    txt = dedent(
        """ r1 1 0
        + fun_param='((0.5*(x-2*y))+z)/(2*(a-2*b))'
        * A mid-stream line comment
        + funner_param=11e-21
        """
    )
    p = SpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_primitive)
    # The continuation handling should work correctly for parameters
    # (both fun_param and funner_param are parsed)
    assert i == Primitive(
        name=Ident(name="r1"),
        args=[Int(val=1), Int(val=0)],
        kwargs=[
            ParamVal(
                name=Ident(name="fun_param"),
                val=BinaryOp(
                    tp=BinaryOperator.DIV,
                    left=BinaryOp(
                        tp=BinaryOperator.ADD,
                        left=BinaryOp(
                            tp=BinaryOperator.MUL,
                            left=Float(val=0.5),
                            right=BinaryOp(
                                tp=BinaryOperator.SUB,
                                left=Ref(ident=Ident(name="x")),
                                right=BinaryOp(
                                    tp=BinaryOperator.MUL,
                                    left=Int(val=2),
                                    right=Ref(ident=Ident(name="y")),
                                ),
                            ),
                        ),
                        right=Ref(ident=Ident(name="z")),
                    ),
                    right=BinaryOp(
                        tp=BinaryOperator.MUL,
                        left=Int(val=2),
                        right=BinaryOp(
                            tp=BinaryOperator.SUB,
                            left=Ref(ident=Ident(name="a")),
                            right=BinaryOp(
                                tp=BinaryOperator.MUL,
                                left=Int(val=2),
                                right=Ref(ident=Ident(name="b")),
                            ),
                        ),
                    ),
                ),
            ),
            ParamVal(name=Ident(name="funner_param"), val=Float(val=1.1e-20)),
        ],
    )


def test_instance():
    p = SpectreDialectParser.from_str(
        "xxx (d g s b) mymos l=11 w=global_w",
    )
    i = p.parse(p.parse_instance)
    assert i == Instance(
        name=Ident(name="xxx"),
        module=Ref(ident=Ident(name="mymos")),
        conns=[Ident(name="d"), Ident(name="g"), Ident(name="s"), Ident(name="b")],
        params=[
            ParamVal(name=Ident(name="l"), val=Int(val=11)),
            ParamVal(name=Ident(name="w"), val=Ref(ident=Ident(name="global_w"))),
        ],
    )

    txt = """rend  (r1 ra) resistor r=rend *(1 + vc1_raw_end*(1 - exp(-abs(v(r2,r1))))
        +                            + vc2_raw_end*(1 - exp(-abs(v(r2,r1)))) * (1 - exp(-abs(v(r2,r1))))        )
        +     + vc3_raw_end*(1 - exp(-abs(v(r2,r1)))) * (1 - exp(-abs(v(r2,r1)))) * (1 - exp(-abs(v(r2,r1))))       """  # The question: adding these (((
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_instance)


def test_instance_parens():
    """
    Spectre has a fun behavior with dangling close-parens at the end of instance statements -
    it accepts as many as you care to provide.
    """

    txt = "rsad 1 0 resistor r=1  ))))))"
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_instance)
    assert i == Instance(
        name=Ident(name="rsad"),
        module=Ref(ident=Ident(name="resistor")),
        conns=[Ident(name="1"), Ident(name="0")],
        params=[ParamVal(name=Ident(name="r"), val=Int(val=1))],
    )


def test_subckt_def():
    p = SpectreDialectParser.from_str("subckt mymos (d g s b) l=11 w=global_w")
    i = p.parse(p.parse_subckt_start)
    assert i == StartSubckt(
        name=Ident(name="mymos"),
        ports=[Ident(name="d"), Ident(name="g"), Ident(name="s"), Ident(name="b")],
        params=[
            ParamDecl(name=Ident(name="l"), default=Int(val=11), distr=None, comment=None),
            ParamDecl(name=Ident(name="w"),
                default=Ref(ident=Ident(name="global_w")),
                distr=None,
                comment=None,
            ),
        ],
    )


def test_model_family():
    txt = dedent(
        """model npd_model bsim3 {
        0: type=n
        //
        + lmin = 1.0 lmax = 2.0 wmin = 1.2 wmax = 1.4
        + level = 999
        + // some commentary
        // plus some blank lines
        + tnom = 30
        1: type=n
        + version = 3.2
        + xj = 1.2e-7
        + lln = 1
        //
        //  Plus More Commentary
        //
        + lwn = 1
        }
        """
    )

    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_model)
    assert i == ModelFamily(
        name=Ident(name="npd_model"),
        mtype=Ident(name="bsim3"),
        variants=[
            ModelVariant(
                model=Ident(name="npd_model"),
                variant=Ident(name="0"),
                mtype=Ident(name="bsim3"),
                args=[],
                params=[
                    ParamDecl(name=Ident(name="type"),
                        default=Ref(ident=Ident(name="n")),
                        distr=None,
                            comment=None,
                    ),
                    ParamDecl(name=Ident(name="lmin"),
                        default=Float(val=1.0),
                        distr=None,
                        comment=None,
                    ),
                    ParamDecl(name=Ident(name="lmax"),
                        default=Float(val=2.0),
                        distr=None,
                        comment=None,
                    ),
                    ParamDecl(name=Ident(name="wmin"),
                        default=Float(val=1.2),
                        distr=None,
                        comment=None,
                    ),
                    ParamDecl(name=Ident(name="wmax"),
                        default=Float(val=1.4),
                        distr=None,
                        comment=None,
                    ),
                    ParamDecl(name=Ident(name="level"),
                        default=Int(val=999),
                        distr=None,
                        comment=None,  # Comments on separate lines aren't associated with parameters
                    ),
                    ParamDecl(name=Ident(name="tnom"),
                        default=Int(val=30),
                        distr=None,
                        comment=None,
                    ),
                ],
            ),
            ModelVariant(
                model=Ident(name="npd_model"),
                variant=Ident(name="1"),
                mtype=Ident(name="bsim3"),
                args=[],
                params=[
                        ParamDecl(name=Ident(name="type"),
                            default=Ref(
                                ident=Ident(name="n")
                            ),  # FIXME: this is a "ref", but to a kinda behind-the-scenes "thing" `n`
                            distr=None,
                            comment=None,
                        ),
                        ParamDecl(name=Ident(name="version"),
                            default=Float(val=3.2),
                            distr=None,
                            comment=None,
                        ),
                        ParamDecl(name=Ident(name="xj"),
                            default=Float(val=1.2e-07),
                            distr=None,
                            comment=None,
                        ),
                        ParamDecl(name=Ident(name="lln"),
                            default=Int(val=1),
                            distr=None,
                            comment=None,
                        ),
                        ParamDecl(name=Ident(name="lwn"),
                            default=Int(val=1),
                            distr=None,
                            comment=None,
                        ),
                ],
            ),
        ],
    )


def test_spectre_midstream_comment():
    """Test for mid-stream full-line comments, which do not break up statements such as `model`
    from being line-continued."""

    txt = dedent(
        """model whatever diode
        + level      =        3
        *
        * This commentary here does not break up the statement.
        *
        + area       =        1.1e11
        """
    )

    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_model)

    # Check that parsed to a `ModelDef`
    assert i == ModelDef(
        name=Ident(name="whatever"),
        mtype=Ident(name="diode"),
        args=[],
        params=[
            ParamDecl(name=Ident(name="level"), default=Int(val=3), distr=None, comment=None),
            ParamDecl(name=Ident(name="area"), default=Float(val=1.1e11), distr=None, comment=None),
        ],
    )


def test_parse_capital_param():
    txt = ".PARAM a = 3 \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)

    assert i == ParamDecls(
        params=[ParamDecl(name=Ident(name="a"), default=Int(val=3), distr=None, comment=None)]
    )


def test_spice_include():
    txt = '.include "/path/to/file" \n'
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == Include(path=Path("/path/to/file"))


def test_protection():
    """Test the `protect` / `unprotect` encryption features"""

    txt = ".protect \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == StartProtectedSection()

    txt = ".prot \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == StartProtectedSection()

    txt = ".unprotect \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == EndProtectedSection()

    txt = ".unprot \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == EndProtectedSection()

    txt = "protect \n"
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == StartProtectedSection()

    txt = "prot \n"
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == StartProtectedSection()

    txt = "unprotect \n"
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == EndProtectedSection()

    txt = "unprot \n"
    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == EndProtectedSection()


def test_names_including_keywords():
    """Test parsing objects whose names include keywords, such as `my_favorite_subckt`."""

    txt = ".param my_favorite_model = model_that_works_best \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)

    assert i == ParamDecls(
        params=[
            ParamDecl(name=Ident(name="my_favorite_model"),
                default=Ref(ident=Ident(name="model_that_works_best")),
                distr=None, comment=None)
        ],
    )


def test_model_with_parens():
    txt = ".model mymodel mtype arg1 arg2 arg3 (key1=val1 key2=val2) \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    m = p.parse(p.parse_statement)

    golden = ModelDef(
        name=Ident(name="mymodel"),
        mtype=Ident(name="mtype"),
        args=[
            Ident(name="arg1"),
            Ident(name="arg2"),
            Ident(name="arg3"),
        ],
        params=[
            ParamDecl(name=Ident(name="key1"),
                default=Ref(ident=Ident(name="val1")),
                distr=None,
            ),
            ParamDecl(name=Ident(name="key2"),
                default=Ref(ident=Ident(name="val2")),
                distr=None,
            ),
        ],
    )
    assert m == golden

    # Run the same thing without the parens, check we get the same result
    txt = ".model mymodel mtype arg1 arg2 arg3 key1=val1 key2=val2 \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    m = p.parse(p.parse_statement)
    assert m == golden


def test_spice_function_def():
    """Test parsing a SPICE-syntax function-definition"""

    txt = ".param f1(p1, p2) = 'p1 > p2 ? a : b' \n"

    p = SpectreSpiceDialectParser.from_str(txt)
    m = p.parse(p.parse_statement)

    assert isinstance(m, FunctionDef)
    assert m == FunctionDef(
        name=Ident(name="f1"),
        rtype=ArgType.UNKNOWN,
        args=[
            TypedArg(
                tp=ArgType.UNKNOWN,
                name=Ident(name="p1"),
            ),
            TypedArg(
                tp=ArgType.UNKNOWN,
                name=Ident(name="p2"),
            ),
        ],
        stmts=[
            Return(
                val=BinaryOp(
                    tp=BinaryOperator.GT,
                    left=Ref(ident=Ident(name="p1"), resolved=None),
                    right=TernOp(
                        cond=Ref(ident=Ident(name="p2"), resolved=None),
                        if_true=Ref(ident=Ident(name="a"), resolved=None),
                        if_false=Ref(ident=Ident(name="b"), resolved=None),
                    ),
                ),
            )
        ],
    )


def test_nested_subckt_def():
    """Test parsing nested sub-circuit definitions"""

    txt = dedent(
        """.subckt a
            .subckt b
                .subckt c
                .ends
                .subckt d
                .ends
                xc c * Instance of `c`
                xd d * Instance of `d`
            .ends
        .ends
        """
    )

    scope = netlist.compile(txt)
    assert isinstance(scope, Scope)
    assert get_external_refs(scope) == []
    assert not has_external_refs(scope)


def test_subckt_params_not_promoted_from_body():
    """Test that parameters on subckt line are correctly captured.
    
    This test verifies basic subcircuit parameter parsing.
    The key distinction is tested in test_spectre_parameters_statement_promoted_to_subckt_params:
    - 'parameters' statements in body ARE promoted (even with defaults)
    - Parameters on subckt line are always in params
    """
    
    # Use Spectre syntax - parameters on subckt line
    txt = dedent("""
        subckt test_sub ( d g s b ) w=10u ad='w*5' pd='2*w'
        r1 d g r=1k
        ends test_sub
    """)
    
    program = parse_str(txt, options=ParseOptions(dialect=NetlistDialects.SPECTRE))
    subckt = program.files[0].contents[0]
    assert isinstance(subckt, SubcktDef)
    
    # Parameters from subckt line should be in params
    param_names = [p.name.name for p in subckt.params]
    assert set(param_names) == {'w', 'ad', 'pd'}, f"Expected ['w', 'ad', 'pd'], got {param_names}"


def test_spectre_parameters_statement_promoted_to_subckt_params():
    """Test that Spectre 'parameters' statements in subcircuit body are promoted to subcircuit parameters.
    
    In Spectre, when a 'parameters' statement appears in a subcircuit body (on a separate line after
    the subckt declaration), ALL parameters from that statement should be promoted to subcircuit parameters,
    even if they have default values. This is different from .param statements which remain local.
    """
    
    txt = dedent("""
        subckt ndio_hia_rf ( anode cathode ) 
        parameters aw=0.8u al=20u fn=1 factor=factor_esd hiaflag=0 dw=0
        r1 anode cathode r=1k
        ends ndio_hia_rf
    """)
    
    program = parse_str(txt, options=ParseOptions(dialect=NetlistDialects.SPECTRE))
    subckt = program.files[0].contents[0]
    assert isinstance(subckt, SubcktDef)
    assert subckt.name.name == "ndio_hia_rf"
    
    # All parameters from 'parameters' statement should be in subcircuit params
    param_names = [p.name.name for p in subckt.params]
    expected_params = ['aw', 'al', 'fn', 'factor', 'hiaflag', 'dw']
    assert set(param_names) == set(expected_params), \
        f"Expected params {expected_params}, got {param_names}"
    
    # Verify parameter defaults are preserved
    param_dict = {p.name.name: p.default for p in subckt.params}
    assert isinstance(param_dict['aw'], MetricNum) and param_dict['aw'].val == '0.8u', \
        f"Expected aw=MetricNum('0.8u'), got {param_dict.get('aw')}"
    assert isinstance(param_dict['al'], MetricNum) and param_dict['al'].val == '20u', \
        f"Expected al=MetricNum('20u'), got {param_dict.get('al')}"
    assert isinstance(param_dict['fn'], Int) and param_dict['fn'].val == 1, \
        f"Expected fn=Int(1), got {param_dict.get('fn')}"
    
    # Parameters should NOT be in entries (they were promoted)
    param_decls_in_entries = [e for e in subckt.entries if isinstance(e, ParamDecls)]
    assert len(param_decls_in_entries) == 0, \
        f"Parameters should be promoted, not in entries. Found {len(param_decls_in_entries)} ParamDecls entries"


def test_spectre_multiply_starting_continuation():
    """Test the case of a multiply starting a continuation-line
    This can prove confusing to parsing, as the state of whether "*" means "multiply" or "comment"
    changes depending on whether it occurs at the start of a new line."""

    # Spectre format
    txt = dedent(
        """
        r1 ( p n ) resistor r=5
        + * 10
    """
    )
    parse_str(txt, options=ParseOptions(dialect=NetlistDialects.SPECTRE))

    # Spice format
    txt = dedent(
        """
        r1  p n  r=5
        + * 10
    """
    )
    parse_str(txt, options=ParseOptions(dialect=NetlistDialects.SPECTRE_SPICE))
    parse_str(txt, options=ParseOptions(dialect=NetlistDialects.NGSPICE))


def test_nested_expression_parens():
    """Test parsing and writing of nested expressions with parentheses, specifically ensuring parentheses are added for operator precedence in output."""

    # This should parse with nested ops and require parentheses around (c + d * e) in output.
    expr_str = "a * b * (c + d * e)"
    parser = SpectreDialectParser.from_str(expr_str)
    parsed = parser.parse(parser.parse_expr)

    # Assertions: Verify left-associative parsing ((a * b) * (c + (d * e)))
    assert isinstance(parsed, BinaryOp)
    assert parsed.tp == BinaryOperator.MUL
    assert isinstance(parsed.left, BinaryOp) and parsed.left.tp == BinaryOperator.MUL
    assert isinstance(parsed.right, BinaryOp) and parsed.right.tp == BinaryOperator.ADD
    assert isinstance(parsed.right.left, Ref) and parsed.right.left.ident.name == "c"
    assert isinstance(parsed.right.right, BinaryOp) and parsed.right.right.tp == BinaryOperator.MUL


def test_mixed_spectre_spice_dialect():
    """Test parsing of netlists with mixed Spectre/Spice dialects, including dialect switches."""

    # Sample netlist: Start in SPICE, use .subckt (SPICE syntax), then params: triggers switch to Spectre,
    # then switch back to SPICE for final statements
    txt = dedent("""
        simulator lang=spice
        .subckt my_subckt (t1 t2 b)
           params: width length
           r0 (t1 t2) resistor width=width length=length
           d0 (b t1) diode area='width*length*0.5' perim='width+length'
           d1 (b t2) diode area='width*length*0.5' perim='width+length'
        .ends
        simulator lang=spice
        .model my_model resistor
        x1 (in out) my_subckt
    """)

    # Parse using parse_str which properly handles mixed dialects via FileParser
    program = parse_str(txt, options=ParseOptions(dialect=NetlistDialects.SPECTRE_SPICE))

    # Collect all entries from the program
    entries = []
    for file in program.files:
        for entry in file.contents:
            entries.append(entry)

    # Verify the parsed program has expected elements
    assert len(entries) > 0  # Program contains entries
    # Check for SubcktDef with Spectre params
    subckt = next((entry for entry in entries if isinstance(entry, SubcktDef)), None)
    assert subckt is not None
    assert subckt.name.name == "my_subckt"
    assert len(subckt.params) == 2  # width and length parameters
    # Check for Spice-style instance (after dialect switch) - instances starting with 'x' in Spice
    instance = next((entry for entry in entries if isinstance(entry, Instance)), None)
    assert instance is not None
    assert instance.name.name == "x1"
    assert instance.module.ident.name == "my_subckt"


def test_statistics_blocks():
    """Test parsing of Spectre statistics blocks with process and mismatch variations."""

    # Sample Spectre statistics block with multiple parameters and distributions
    # Expected: parse process variations (param_a, param_b) and mismatch variations (param_c, param_d)
    txt = dedent("""\
        statistics {
           process {
              vary param_a dist=gauss std=1.5
              vary param_b dist=lnorm std=0.8 mean=2.0
              vary param_e dist=gauss std=0.3 mean=0.5
           }
           mismatch {
              vary param_c dist=gauss std=0.5
              vary param_d dist=lnorm std=0.2
           }
        }
    """)

    # Parse using SpectreDialectParser
    parser = SpectreDialectParser.from_str(txt)
    stats_block = parser.parse(parser.parse_statistics_block)

    # Verify the StatisticsBlock structure
    assert isinstance(stats_block, StatisticsBlock)
    
    # Check process variations - expected: 3 variations
    assert len(stats_block.process) == 3, "Expected 3 process variations"
    
    # First process variation: param_a with gauss distribution
    assert stats_block.process[0].name.name == "param_a"
    assert stats_block.process[0].dist == "gauss"
    assert isinstance(stats_block.process[0].std, Float) and stats_block.process[0].std.val == 1.5
    assert stats_block.process[0].mean is None  # No mean specified
    
    # Second process variation: param_b with lnorm distribution and mean
    assert stats_block.process[1].name.name == "param_b"
    assert stats_block.process[1].dist == "lnorm"
    assert isinstance(stats_block.process[1].std, Float) and stats_block.process[1].std.val == 0.8
    assert isinstance(stats_block.process[1].mean, Float) and stats_block.process[1].mean.val == 2.0
    
    # Third process variation: param_e with gauss distribution and mean
    assert stats_block.process[2].name.name == "param_e"
    assert stats_block.process[2].dist == "gauss"
    assert isinstance(stats_block.process[2].std, Float) and stats_block.process[2].std.val == 0.3
    assert isinstance(stats_block.process[2].mean, Float) and stats_block.process[2].mean.val == 0.5
    
    # Check mismatch variations - expected: 2 variations
    assert len(stats_block.mismatch) == 2, "Expected 2 mismatch variations"
    
    # First mismatch variation: param_c with gauss distribution
    assert stats_block.mismatch[0].name.name == "param_c"
    assert stats_block.mismatch[0].dist == "gauss"
    assert isinstance(stats_block.mismatch[0].std, Float) and stats_block.mismatch[0].std.val == 0.5
    assert stats_block.mismatch[0].mean is None  # No mean specified
    
    # Second mismatch variation: param_d with lnorm distribution
    assert stats_block.mismatch[1].name.name == "param_d"
    assert stats_block.mismatch[1].dist == "lnorm"
    assert isinstance(stats_block.mismatch[1].std, Float) and stats_block.mismatch[1].std.val == 0.2
    assert stats_block.mismatch[1].mean is None  # No mean specified


def test_parameters_multiple_continuation_lines_with_expressions():
    """Test parsing multiple continuation lines with expressions containing division operators.
    
    This test is based on the real input from MODELS/common/spectre/models.scs around line 94-99,
    where we have multiple continuation lines with parameter assignments that include division operations.
    """
    txt = dedent("""
        parameters 
        // Separate Tox Corners from Monte Carlo for fet ff and ss
        + 	sw_tox_lv                = sw_tox_lv_corner 
        + 	sw_tox_hv                = sw_tox_hv_corner
        + 	sw_func_tox_lv_ratio     = sw_tox_lv             / sw_tox_lv_nom
        + 	sw_func_tox_hv_ratio     = sw_tox_hv             / sw_tox_hv_nom
    """)
    
    parser = SpectreDialectParser.from_str(txt)
    result = parser.parse(parser.parse_statement)
    
    # Should parse to a ParamDecls with 4 parameters
    assert isinstance(result, ParamDecls)
    assert len(result.params) == 4
    
    # Check first parameter: sw_tox_lv = sw_tox_lv_corner
    assert result.params[0].name.name == "sw_tox_lv"
    assert isinstance(result.params[0].default, Ref)
    assert result.params[0].default.ident.name == "sw_tox_lv_corner"
    
    # Check second parameter: sw_tox_hv = sw_tox_hv_corner
    assert result.params[1].name.name == "sw_tox_hv"
    assert isinstance(result.params[1].default, Ref)
    assert result.params[1].default.ident.name == "sw_tox_hv_corner"
    
    # Check third parameter: sw_func_tox_lv_ratio = sw_tox_lv / sw_tox_lv_nom
    assert result.params[2].name.name == "sw_func_tox_lv_ratio"
    assert isinstance(result.params[2].default, BinaryOp)
    assert result.params[2].default.tp == BinaryOperator.DIV
    assert isinstance(result.params[2].default.left, Ref)
    assert result.params[2].default.left.ident.name == "sw_tox_lv"
    assert isinstance(result.params[2].default.right, Ref)
    assert result.params[2].default.right.ident.name == "sw_tox_lv_nom"
    
    # Check fourth parameter: sw_func_tox_hv_ratio = sw_tox_hv / sw_tox_hv_nom
    assert result.params[3].name.name == "sw_func_tox_hv_ratio"
    assert isinstance(result.params[3].default, BinaryOp)
    assert result.params[3].default.tp == BinaryOperator.DIV
    assert isinstance(result.params[3].default.left, Ref)
    assert result.params[3].default.left.ident.name == "sw_tox_hv"
    assert isinstance(result.params[3].default.right, Ref)
    assert result.params[3].default.right.ident.name == "sw_tox_hv_nom"


def test_parameters_continuation_lines_two_expr_minimal():
    """Minimal repro: 4 continuation lines where the last two are division expressions.

    This is intentionally smaller than the sky130 model snippet, but keeps the same shape:
    simple refs, then an expr containing '/', then another expr containing '/'.
    """
    txt = dedent("""
        parameters
        + a = b
        + c = d
        + e = f / g
        + h = i / j
    """)

    parser = SpectreDialectParser.from_str(txt)
    result = parser.parse(parser.parse_statement)

    assert isinstance(result, ParamDecls)
    assert [p.name.name for p in result.params] == ["a", "c", "e", "h"]

    assert isinstance(result.params[2].default, BinaryOp)
    assert result.params[2].default.tp == BinaryOperator.DIV
    assert isinstance(result.params[3].default, BinaryOp)
    assert result.params[3].default.tp == BinaryOperator.DIV


def test_parameters_continuation_lines_mixed_expressions():
    """Test parsing continuation lines with a mix of simple assignments and complex expressions.
    
    This ensures the parser correctly handles transitions between simple ref assignments
    and expressions with operators across continuation lines.
    """
    txt = dedent("""
        parameters 
        +  param1 = value1
        +  param2 = value2
        +  param3 = value3 / value4
        +  param4 = value5 * value6
        +  param5 = value7
    """)
    
    parser = SpectreDialectParser.from_str(txt)
    result = parser.parse(parser.parse_statement)
    
    # Should parse to a ParamDecls with 5 parameters
    assert isinstance(result, ParamDecls)
    assert len(result.params) == 5
    
    # Check simple assignments
    assert result.params[0].name.name == "param1"
    assert isinstance(result.params[0].default, Ref)
    assert result.params[0].default.ident.name == "value1"
    
    assert result.params[1].name.name == "param2"
    assert isinstance(result.params[1].default, Ref)
    assert result.params[1].default.ident.name == "value2"
    
    # Check division expression
    assert result.params[2].name.name == "param3"
    assert isinstance(result.params[2].default, BinaryOp)
    assert result.params[2].default.tp == BinaryOperator.DIV
    
    # Check multiplication expression
    assert result.params[3].name.name == "param4"
    assert isinstance(result.params[3].default, BinaryOp)
    assert result.params[3].default.tp == BinaryOperator.MUL
    
    # Check simple assignment after expression
    assert result.params[4].name.name == "param5"
    assert isinstance(result.params[4].default, Ref)
    assert result.params[4].default.ident.name == "value7"

