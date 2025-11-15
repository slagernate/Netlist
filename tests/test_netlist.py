"""
# Netlist Unit Tests

"""

from io import StringIO
from pathlib import Path
from textwrap import dedent

# DUT Imports
import netlist
from netlist import (
    __version__,
    parse_files,
    parse_str,
    netlist as write_netlist,
    ParseOptions,
    Program,
    SourceFile,
    ParamDecl,
    ModelDef,
    ModelVariant,
    ModelFamily,
    BinaryOperator,
    UnaryOperator,
    Options,
    Option,
    ParamVal,
    Ident,
    MetricNum,
    SourceInfo,
    NetlistDialects,
    SpiceDialectParser,
    BinaryOp,
    SpectreDialectParser,
    Int,
    Float,
    MetricNum,
    UnaryOp,
    Call,
    Ref,
    StartProtectedSection,
    EndProtectedSection,
)


def test_version():
    assert __version__ == "0.1.0"


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
    assert p == BinaryOp(
        tp=BinaryOperator.MUL,
        left=Ref(ident=Ident(name="r")),
        right=BinaryOp(
            tp=BinaryOperator.DIV,
            left=Ref(ident=Ident(name="l")),
            right=Ref(ident=Ident(name="w")),
        ),
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


def test_param_values():
    from netlist import Ident, ParamDecl, Float, Expr, BinaryOp

    p = ParamDecl(Ident("a"), Float(5))
    p = ParamDecl(
        Ident("b"),
        BinaryOp(
            BinaryOperator.ADD,
            Float(1e-3),
            BinaryOp(BinaryOperator.MUL, Float(2.0), Float(0.3)),
        ),
    )


def test_primitive():
    from netlist import SpiceDialectParser
    from netlist.data import Ident, BinaryOp, Primitive, Float, Int, ParamVal

    txt = dedent(
        """ r1 1 0
        + fun_param='((0.5*(x-2*y))+z)/(2*(a-2*b))'
        * A mid-stream line comment
        + funner_param=11e-21
        """
    )
    p = SpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_primitive)
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
    from netlist import SpectreDialectParser
    from netlist import Ident, ParamVal, Int, Instance, Ref

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

    So it will accept this is a valid instance:
    ```
    rsad 1 0 resistor r=1  )))))) // really, with all those parentheses
    ```

    The same close-paren behavior does not apply to parameter-declaration statements.
    It may apply to other types.

    You may ask, why should `netlist` inherit what is almost certainly a Spectre bug?
    Because, sadly, notable popular commercial netlists and models include some of these errant parentheses,
    and therefore only work *because* of the Spectre-bug. So, if we want to parse them, we need that bug too.
    """

    txt = "rsad 1 0 resistor r=1  ))))))"
    from netlist import SpectreDialectParser
    from netlist import Ident, ParamVal, Int, Instance, Ref

    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_instance)
    assert i == Instance(
        name=Ident(name="rsad"),
        module=Ref(ident=Ident(name="resistor")),
        conns=[Ident(name="1"), Ident(name="0")],
        params=[ParamVal(name=Ident(name="r"), val=Int(val=1))],
    )


def test_subckt_def():
    from netlist import SpectreDialectParser
    from netlist import Ident, ParamDecl, Int, StartSubckt, Ref

    p = SpectreDialectParser.from_str("subckt mymos (d g s b) l=11 w=global_w")
    i = p.parse(p.parse_subckt_start)
    assert i == StartSubckt(
        name=Ident(name="mymos"),
        ports=[Ident(name="d"), Ident(name="g"), Ident(name="s"), Ident(name="b")],
        params=[
            ParamDecl(name=Ident(name="l"), default=Int(val=11), distr=None),
            ParamDecl(
                name=Ident(name="w"),
                default=Ref(ident=Ident(name="global_w")),
                distr=None,
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
                    ParamDecl(
                        name=Ident(name="type"),
                        default=Ref(ident=Ident(name="n")),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="lmin"),
                        default=Float(val=1.0),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="lmax"),
                        default=Float(val=2.0),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="wmin"),
                        default=Float(val=1.2),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="wmax"),
                        default=Float(val=1.4),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="level"),
                        default=Int(val=999),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="tnom"),
                        default=Int(val=30),
                        distr=None,
                    ),
                ],
            ),
            ModelVariant(
                model=Ident(name="npd_model"),
                variant=Ident(name="1"),
                mtype=Ident(name="bsim3"),
                args=[],
                params=[
                    ParamDecl(
                        name=Ident(name="type"),
                        default=Ref(
                            ident=Ident(name="n")
                        ),  # FIXME: this is a "ref", but to a kinda behind-the-scenes "thing" `n`
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="version"),
                        default=Float(val=3.2),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="xj"),
                        default=Float(val=1.2e-07),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="lln"),
                        default=Int(val=1),
                        distr=None,
                    ),
                    ParamDecl(
                        name=Ident(name="lwn"),
                        default=Int(val=1),
                        distr=None,
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
    from netlist import SpectreDialectParser

    p = SpectreDialectParser.from_str(txt)
    i = p.parse(p.parse_model)

    # Check that parsed to a `ModelDef`
    from netlist.data import ModelDef, Ident, ParamDecl, Int, Float

    assert i == ModelDef(
        name=Ident(name="whatever"),
        mtype=Ident(name="diode"),
        args=[],
        params=[
            ParamDecl(name=Ident(name="level"), default=Int(val=3), distr=None),
            ParamDecl(name=Ident(name="area"), default=Float(val=1.1e11), distr=None),
        ],
    )


def test_parse_capital_param():
    from netlist import SpectreSpiceDialectParser, Ident, ParamDecls, ParamDecl, Int

    txt = ".PARAM a = 3 \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)

    assert i == ParamDecls(
        params=[ParamDecl(name=Ident(name="a"), default=Int(val=3), distr=None)]
    )


def test_spice_include():
    from netlist import SpectreSpiceDialectParser, Include

    txt = '.include "/path/to/file" \n'
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)
    assert i == Include(path=Path("/path/to/file"))


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
    from netlist.data import FunctionDef, ArgType, TypedArg, Return, Call, Ref, Int, Float
    from netlist.write import WriteOptions, NetlistDialects
    
    # Create a simple function: mm_p1() {gauss(0, 0.1, 1)}
    func = FunctionDef(
        name=Ident("mm_p1"),
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
    
    # Verify output format: .FUNC mm_p1() { gauss(0,0.1,1) }
    assert ".FUNC" in output
    assert "mm_p1()" in output
    assert "gauss" in output
    # Check that it's properly formatted
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    func_line = next((line for line in lines if ".FUNC" in line), None)
    assert func_line is not None
    assert func_line.startswith(".FUNC")
    assert "mm_p1()" in func_line
    assert "{" in func_line
    assert "}" in func_line


def test_write_xyce_func_with_args():
    """Test writing Xyce .FUNC definitions with arguments"""
    from netlist.data import FunctionDef, ArgType, TypedArg, Return, Call, Ref, Ident, BinaryOp, BinaryOperator
    from netlist.write import WriteOptions, NetlistDialects
    
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


def test_protection():
    """Test the `protect` / `unprotect` encryption features"""
    from netlist import SpectreDialectParser, SpectreSpiceDialectParser

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
    from netlist import SpectreSpiceDialectParser, Ident, ParamDecls, ParamDecl, Int

    txt = ".param my_favorite_model = model_that_works_best \n"
    p = SpectreSpiceDialectParser.from_str(txt)
    i = p.parse(p.parse_statement)

    assert i == ParamDecls(
        params=[
            ParamDecl(
                name=Ident(name="my_favorite_model"),
                default=Ref(ident=Ident(name="model_that_works_best")),
                distr=None,
            )
        ],
    )


def test_model_with_parens():
    from netlist import SpectreSpiceDialectParser

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
            ParamDecl(
                name=Ident(name="key1"),
                default=Ref(ident=Ident(name="val1")),
                distr=None,
            ),
            ParamDecl(
                name=Ident(name="key2"),
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
    from netlist import SpectreSpiceDialectParser
    from netlist.data import (
        Ident,
        FunctionDef,
        ArgType,
        TypedArg,
        Return,
        BinaryOp,
        TernOp,
    )

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
    from netlist import has_external_refs, get_external_refs, Scope

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
    from netlist.dialects.spectre import SpectreDialectParser
    from netlist.data import BinaryOp, BinaryOperator, Int, Float, Ref, Ident

    # This should parse with nested ops and require parentheses around (c + d * e) in output.
    expr_str = "a * b * (c + d * e)"
    parser = SpectreDialectParser.from_str(expr_str)
    parsed = parser.parse(parser.parse_expr)

    # Debug: Print the parsed AST to verify structure (optional, for troubleshooting)
    print(f"Parsed AST: {parsed}")
    print(f"Type: {type(parsed)}")
    if isinstance(parsed, BinaryOp):
        print(f"Root op: {parsed.tp}")
        print(f"Left: {parsed.left}")
        print(f"Right: {parsed.right}")
        if isinstance(parsed.right, BinaryOp):
            print(f"Right op: {parsed.right.tp}")
            print(f"Right.Right: {parsed.right.right}")
            if isinstance(parsed.right.right, BinaryOp):
                print(f"Right.Right op: {parsed.right.right.tp}")
                print(f"Right.Right.Right: {parsed.right.right.right}")

    # Assertions: Verify left-associative parsing (a * (b * (c + (d * e))))
    assert isinstance(parsed, BinaryOp)
    assert parsed.tp == BinaryOperator.MUL
    assert isinstance(parsed.right, BinaryOp)  # b * (...)
    assert parsed.right.tp == BinaryOperator.MUL
    assert isinstance(parsed.right.right, BinaryOp)  # The (c + ...)
    assert parsed.right.right.tp == BinaryOperator.ADD  # Now this should match
    assert isinstance(parsed.right.right.left, Ref) and parsed.right.right.left.ident.name == "c"
    assert isinstance(parsed.right.right.right, BinaryOp) and parsed.right.right.right.tp == BinaryOperator.MUL


def test_writer_parentheses_precedence():
    """Test that the writer adds parentheses for operator precedence in expressions"""
    from netlist.dialects.spectre import SpectreDialectParser
    from netlist.write.spice import XyceNetlister
    from io import StringIO

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


def test_mixed_spectre_spice_dialect():
    """Test parsing of netlists with mixed Spectre/Spice dialects, including dialect switches."""
    from netlist import parse_str, ParseOptions, NetlistDialects, SubcktDef, Instance, Primitive

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
    from netlist import SpectreDialectParser, StatisticsBlock, Variation, Ident, Float

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


def test_apply_statistics_vary_mismatch():
    """Test mismatch variations (Gauss and Lognorm) are applied to corresponding parameters.
    
    Mismatch variations use Monte Carlo (implied). Both gauss and lnorm distributions
    are tested to verify proper handling of different distribution types.
    """
    from netlist.write.spice import apply_statistics_variations
    from netlist.data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float, BinaryOp, BinaryOperator, Ref, Ident, Call, FunctionDef, Return, Int

    # Create a program with multiple parameters and statistics blocks
    # Parameters: width, length, tox (oxide thickness)
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None),
    ]
    
    # Statistics block with mismatch variations (Monte Carlo implied)
    # Expected: gauss variations multiply by (1 + std * gauss(...))
    # Expected: lnorm variations multiply by (1 + std * lnorm(...)) where lnorm is exp(gauss(...))
    stats = StatisticsBlock(
        process=None,
        mismatch=[
            Variation(name=Ident("width"), dist="gauss", std=Float(0.1), mean=None),  # Gaussian mismatch
            Variation(name=Ident("length"), dist="gauss", std=Float(0.05), mean=None),  # Gaussian mismatch
            Variation(name=Ident("tox"), dist="lnorm", std=Float(0.15), mean=None),  # Lognormal mismatch
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Apply variations (mismatch block detected, functions created automatically)
    # Pass XYCE format to enable Xyce-specific transformations (mismatch functions, enable_mismatch, lnorm definitions)
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Check width parameter: mismatch creates function call
    # Expected: original + width_mismatch() where width_mismatch() {enable_mismatch * gauss(0, std, seed)}
    # Also verify enable_mismatch parameter was added
    enable_mismatch_param = next(
        (param for file in program.files
         for entry in file.contents
         if isinstance(entry, ParamDecls)
         for param in entry.params
         if param.name.name == "enable_mismatch"),
        None
    )
    assert enable_mismatch_param is not None, "Expected enable_mismatch parameter to be added"
    assert isinstance(enable_mismatch_param.default, Float) and enable_mismatch_param.default.val == 1.0
    width_param = params[0]
    assert isinstance(width_param.default, BinaryOp)
    assert width_param.default.tp == BinaryOperator.ADD  # Expected: original + function_call
    assert isinstance(width_param.default.left, Float) and width_param.default.left.val == 1.0  # Original value
    assert isinstance(width_param.default.right, Call)
    assert width_param.default.right.func.ident.name == "width_mismatch"  # Function call: width_mismatch()
    assert len(width_param.default.right.args) == 0  # No arguments

    # Check length parameter: mismatch creates function call
    length_param = params[1]
    assert isinstance(length_param.default, BinaryOp)
    assert length_param.default.tp == BinaryOperator.ADD
    assert isinstance(length_param.default.left, Float) and length_param.default.left.val == 0.5
    assert isinstance(length_param.default.right, Call)
    assert length_param.default.right.func.ident.name == "length_mismatch"  # Function call: length_mismatch()

    # Check tox parameter: mismatch creates function call with lnorm
    tox_param = params[2]
    assert isinstance(tox_param.default, BinaryOp)
    assert tox_param.default.tp == BinaryOperator.ADD
    assert isinstance(tox_param.default.left, Float) and tox_param.default.left.val == 2.0e-9
    assert isinstance(tox_param.default.right, Call)
    assert tox_param.default.right.func.ident.name == "tox_mismatch"  # Function call: tox_mismatch()
    
    # Verify that mismatch functions were added to the program
    mismatch_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name in ["width_mismatch", "length_mismatch", "tox_mismatch"]
    ]
    assert len(mismatch_funcs) == 3, f"Expected 3 mismatch functions, got {len(mismatch_funcs)}"
    
    # Verify width function uses gauss with enable_mismatch multiplier
    width_func = next(f for f in mismatch_funcs if f.name.name == "width_mismatch")
    assert len(width_func.stmts) == 1
    assert isinstance(width_func.stmts[0], Return)
    width_return_val = width_func.stmts[0].val
    # Expected: enable_mismatch * gauss(...)
    assert isinstance(width_return_val, BinaryOp)
    assert width_return_val.tp == BinaryOperator.MUL
    assert isinstance(width_return_val.left, Ref) and width_return_val.left.ident.name == "enable_mismatch"
    width_call = width_return_val.right
    assert isinstance(width_call, Call)
    assert width_call.func.ident.name == "gauss"  # Expected: gauss for mismatch
    assert len(width_call.args) == 3
    assert isinstance(width_call.args[0], Int) and width_call.args[0].val == 0  # mean = 0
    assert isinstance(width_call.args[1], Float) and width_call.args[1].val == 0.1  # std
    assert isinstance(width_call.args[2], Int) and width_call.args[2].val == 1  # seed
    
    # Verify tox function uses lnorm with enable_mismatch multiplier
    tox_func = next(f for f in mismatch_funcs if f.name.name == "tox_mismatch")
    assert len(tox_func.stmts) == 1
    assert isinstance(tox_func.stmts[0], Return)
    tox_return_val = tox_func.stmts[0].val
    # Expected: enable_mismatch * lnorm(...)
    assert isinstance(tox_return_val, BinaryOp)
    assert tox_return_val.tp == BinaryOperator.MUL
    assert isinstance(tox_return_val.left, Ref) and tox_return_val.left.ident.name == "enable_mismatch"
    tox_call = tox_return_val.right
    assert isinstance(tox_call, Call)
    assert tox_call.func.ident.name == "lnorm"  # Expected: lnorm for lognormal mismatch
    assert len(tox_call.args) == 3
    assert isinstance(tox_call.args[0], Int) and tox_call.args[0].val == 0  # mean = 0
    assert isinstance(tox_call.args[1], Float) and tox_call.args[1].val == 0.15  # std
    assert isinstance(tox_call.args[2], Int) and tox_call.args[2].val == 3  # seed (third mismatch param, so idx=2, seed=3)
    
    # Verify that lnorm function definition was added to the program
    # (needed for the tox() function to work, since it calls lnorm)
    lnorm_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name == "lnorm"
    ]
    assert len(lnorm_funcs) == 1, "Expected lnorm function to be added for lognormal mismatch"
    lnorm_func = lnorm_funcs[0]
    assert len(lnorm_func.stmts) == 1
    assert isinstance(lnorm_func.stmts[0], Return)
    # Verify lnorm is defined as exp(gauss(...))
    lnorm_body = lnorm_func.stmts[0].val
    assert isinstance(lnorm_body, Call)
    assert lnorm_body.func.ident.name == "exp"  # lnorm = exp(gauss(...))
    assert len(lnorm_body.args) == 1
    gauss_call = lnorm_body.args[0]
    assert isinstance(gauss_call, Call)
    assert gauss_call.func.ident.name == "gauss"  # lnorm uses gauss internally

def test_apply_statistics_vary_process():
    """Test process variations are applied to corresponding parameters.
    
    Process variations use Monte Carlo (implied) and may include mean values for corner analysis.
    """
    from netlist.write.spice import apply_statistics_variations
    from netlist.data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float, BinaryOp, BinaryOperator

    # Create a program with multiple parameters for process variations
    # Parameters: vth (threshold voltage), mobility, junction_depth
    params = [
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None),
        ParamDecl(name=Ident("mobility"), default=Float(300.0), distr=None),
        ParamDecl(name=Ident("junction_depth"), default=Float(0.1e-6), distr=None),
    ]
    
    # Statistics block with process variations (Monte Carlo implied, mean values for corners)
    # Expected: process variations apply Monte Carlo and add mean value if present
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.05), mean=Float(0.1)),  # Fast corner: +0.1V
            Variation(name=Ident("mobility"), dist="gauss", std=Float(10.0), mean=Float(-50.0)),  # Slow corner: -50 cm²/V·s
            Variation(name=Ident("junction_depth"), dist="gauss", std=Float(0.01e-6), mean=Float(0.02e-6)),  # Deep junction corner
        ],
        mismatch=None
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Apply variations (process block detected, Monte Carlo and mean applied automatically)
    # Pass XYCE format (though process variations are format-agnostic, this ensures consistency)
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Check vth parameter: process variations apply Monte Carlo first, then add mean
    # Expected structure: (original * (1 + std * gauss(...))) + mean
    vth_param = params[0]
    assert isinstance(vth_param.default, BinaryOp)
    assert vth_param.default.tp == BinaryOperator.ADD  # Expected: (Monte Carlo result) + mean
    # Left side should be: original * (1 + std * gauss(...))
    assert isinstance(vth_param.default.left, BinaryOp)
    assert vth_param.default.left.tp == BinaryOperator.MUL
    assert isinstance(vth_param.default.left.left, Float) and vth_param.default.left.left.val == 0.4  # Original value
    assert vth_param.default.right == Float(0.1)  # Expected: mean value added

    # Check mobility parameter: process variations apply Monte Carlo first, then add mean
    mobility_param = params[1]
    assert isinstance(mobility_param.default, BinaryOp)
    assert mobility_param.default.tp == BinaryOperator.ADD
    # Left side should be: original * (1 + std * gauss(...))
    assert isinstance(mobility_param.default.left, BinaryOp)
    assert mobility_param.default.left.tp == BinaryOperator.MUL
    assert isinstance(mobility_param.default.left.left, Float) and mobility_param.default.left.left.val == 300.0
    assert mobility_param.default.right == Float(-50.0)  # Expected: negative mean value added

    # Check junction_depth parameter: process variations apply Monte Carlo first, then add mean
    jd_param = params[2]
    assert isinstance(jd_param.default, BinaryOp)
    assert jd_param.default.tp == BinaryOperator.ADD
    # Left side should be: original * (1 + std * gauss(...))
    assert isinstance(jd_param.default.left, BinaryOp)
    assert jd_param.default.left.tp == BinaryOperator.MUL
    assert isinstance(jd_param.default.left.left, Float) and jd_param.default.left.left.val == 0.1e-6
    assert jd_param.default.right == Float(0.02e-6)  # Expected: mean value added


def test_apply_statistics_vary_no_match():
    """Test that variations are ignored if no corresponding parameter exists."""
    from netlist.write.spice import apply_statistics_variations
    from netlist.data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float

    # Create a program with parameters that don't match the statistics variations
    # Parameters: width, length
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("length"), default=Float(0.5), distr=None),
    ]
    
    # Statistics block with variations for parameters that don't exist
    # Expected: variations should be ignored, parameters remain unchanged
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
    
    # Apply variations (both process and mismatch blocks present, but no matching parameters)
    # Pass XYCE format to test Xyce-specific behavior
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Expected: parameters should remain unchanged since no matching variations exist
    assert params[0].default == original_width, "width parameter should remain unchanged"
    assert params[1].default == original_length, "length parameter should remain unchanged"


def test_apply_statistics_vary_spectre_no_xyce_transformations():
    """Test that Xyce-specific transformations are NOT applied when output format is Spectre.
    
    When writing to Spectre format, mismatch functions, enable_mismatch parameter, and lnorm
    function definitions should NOT be created, as these are Xyce-specific.
    """
    from netlist.write.spice import apply_statistics_variations
    from netlist.data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float, FunctionDef

    # Create a program with parameters and mismatch variations
    params = [
        ParamDecl(name=Ident("width"), default=Float(1.0), distr=None),
        ParamDecl(name=Ident("tox"), default=Float(2.0e-9), distr=None),
    ]
    
    # Statistics block with mismatch variations (which would create Xyce .FUNC definitions)
    stats = StatisticsBlock(
        process=None,
        mismatch=[
            Variation(name=Ident("width"), dist="gauss", std=Float(0.1), mean=None),
            Variation(name=Ident("tox"), dist="lnorm", std=Float(0.15), mean=None),
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Apply variations with SPECTRE format (not XYCE)
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
    assert enable_mismatch_param is None, "enable_mismatch parameter should NOT be added for Spectre format"

    # Verify mismatch functions were NOT created
    mismatch_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name.endswith("_mismatch")
    ]
    assert len(mismatch_funcs) == 0, "Mismatch functions should NOT be created for Spectre format"

    # Verify lnorm function definitions were NOT created
    lnorm_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name in ("lnorm", "alnorm")
    ]
    assert len(lnorm_funcs) == 0, "lnorm function definitions should NOT be created for Spectre format"

    # Verify parameters remain unchanged (mismatch variations are not applied for non-Xyce formats)
    assert params[0].default == Float(1.0), "width parameter should remain unchanged for Spectre format"
    assert params[1].default == Float(2.0e-9), "tox parameter should remain unchanged for Spectre format"


def test_param_mismatch_and_corner():
    """Test that parameters with both process corner and mismatch variations are handled correctly.
    
    This test verifies that parameters appearing in both process and mismatch blocks are handled correctly.
    The expected behavior:
    - Process variation: original * (1 + std * gauss(...)) + mean (corner analysis)
    - Mismatch variation: creates mismatch function and adds it
    - Combined: (original * (1 + std * gauss(...)) + mean) + mismatch_function()
    
    This represents a parameter that has both global process corner variation and per-instance mismatch.
    """
    from netlist.write.spice import apply_statistics_variations
    from netlist.data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float, BinaryOp, BinaryOperator, Ref, Ident, Call, FunctionDef, Return, Int

    # Create a program with a parameter that appears in both process and mismatch blocks
    params = [
        ParamDecl(name=Ident("vth"), default=Float(0.4), distr=None),
    ]
    
    # Statistics block with the same parameter in both process and mismatch
    stats = StatisticsBlock(
        process=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.05), mean=Float(0.1)),  # Process variation
        ],
        mismatch=[
            Variation(name=Ident("vth"), dist="gauss", std=Float(0.02), mean=None),  # Mismatch variation
        ]
    )
    program = Program(files=[SourceFile(path="test", contents=[ParamDecls(params=params), stats])])

    # Apply variations with XYCE format
    apply_statistics_variations(program, output_format=NetlistDialects.XYCE)

    # Expected: vth should have both process variation (Monte Carlo + mean) and mismatch function
    # Structure: ((original * (1 + std * gauss(...))) + mean) + vth_mismatch()
    vth_param = params[0]
    
    # The parameter should be updated to include both variations
    assert isinstance(vth_param.default, BinaryOp), "vth parameter should be updated with variations"
    
    # The outermost operation should be ADD (original + mismatch function)
    assert vth_param.default.tp == BinaryOperator.ADD, "Outermost operation should be ADD for combined variations"
    
    # The left side should be the process variation result: (original * (1 + std * gauss(...))) + mean
    process_result = vth_param.default.left
    assert isinstance(process_result, BinaryOp), "Process variation should be applied"
    assert process_result.tp == BinaryOperator.ADD, "Process variation should add mean"
    
    # The right side should be the mismatch function call
    mismatch_call = vth_param.default.right
    assert isinstance(mismatch_call, Call), "Mismatch should be a function call"
    assert mismatch_call.func.ident.name == "vth_mismatch", "Mismatch function should be vth_mismatch()"
    
    # Verify mismatch function was created
    mismatch_funcs = [
        entry for file in program.files
        for entry in file.contents
        if isinstance(entry, FunctionDef) and entry.name.name == "vth_mismatch"
    ]
    assert len(mismatch_funcs) == 1, "vth_mismatch function should be created"

