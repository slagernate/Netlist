
from netlist import (
    __version__,
    ParamDecl,
    Ident,
    Float,
    BinaryOp,
    BinaryOperator,
)

def test_version():
    assert __version__ == "0.1.0"


def test_param_values():
    # Basic test to ensure ParamDecl can be instantiated with complex values
    p = ParamDecl(Ident("a"), Float(5))
    assert p.name.name == "a"
    assert p.default.val == 5.0

    p = ParamDecl(
        Ident("b"),
        BinaryOp(
            BinaryOperator.ADD,
            Float(1e-3),
            BinaryOp(BinaryOperator.MUL, Float(2.0), Float(0.3)),
        ),
    )
    assert p.name.name == "b"
    assert isinstance(p.default, BinaryOp)

