import io

import pytest

from netlist.dialects.spectre import SpectreDialectParser
from netlist.write.xyce import XyceNetlister
from netlist.data import Program


def _parse_expr(expr: str):
    # Use the Spectre dialect parser so '*' is multiplication inside expressions.
    p = SpectreDialectParser.from_str(expr)
    return p.parse()


def _xyce_expr_string(expr: str) -> str:
    # Format only the expression using the Xyce netlister's formatter.
    # Provide a minimal Program so the netlister can be instantiated.
    prog = Program(files=[])
    n = XyceNetlister(src=prog, dest=io.StringIO())
    return n.format_expr(_parse_expr(expr))


def test_division_chain_is_left_associative():
    # Historically this was emitted as 26/(16.88/_conum) due to right-associative parsing.
    s = _xyce_expr_string("26/16.88/_conum")
    assert "26/(16.88/_conum)" not in s


def test_long_division_chain_is_left_associative():
    # Another representative chain from the RF MOS wrapper.
    s = _xyce_expr_string("lspace/2/nr/(wr*factor+_xw)")
    assert "lspace/(2/(nr/" not in s





