from __future__ import annotations

from io import StringIO
from pathlib import Path

from netlist.data import Ident, ModelDef, Program, SourceFile, SubcktDef
from netlist.write.xyce import XyceNetlister


def _netlist_xyce(program: Program) -> str:
    buf = StringIO()
    XyceNetlister(program, buf).netlist()
    return buf.getvalue()


def test_xyce_single_model_dot0_suffix_is_stripped_in_subckt() -> None:
    # Single model card inside a subckt: .model foo.0 must be emitted as .model foo
    # (Xyce model lookup expects the base name).
    subckt = SubcktDef(
        name=Ident("top"),
        ports=[],
        params=[],
        entries=[
            ModelDef(name=Ident("foo.0"), mtype=Ident("nmos"), args=[], params=[]),
        ],
    )
    program = Program(files=[SourceFile(path=Path("in.scs"), contents=[subckt])])
    out = _netlist_xyce(program)

    assert ".model foo nmos" in out
    assert ".model foo.0 nmos" not in out


def test_xyce_multiple_numeric_model_suffixes_are_not_stripped_in_subckt() -> None:
    # Multiple model cards inside a subckt: keep all numeric suffixes unchanged.
    # This preserves distinct binned models (foo.1/foo.2/foo.3/...) for later selection.
    subckt = SubcktDef(
        name=Ident("top"),
        ports=[],
        params=[],
        entries=[
            ModelDef(name=Ident("foo.1"), mtype=Ident("nmos"), args=[], params=[]),
            ModelDef(name=Ident("foo.2"), mtype=Ident("nmos"), args=[], params=[]),
            ModelDef(name=Ident("foo.3"), mtype=Ident("nmos"), args=[], params=[]),
        ],
    )
    program = Program(files=[SourceFile(path=Path("in.scs"), contents=[subckt])])
    out = _netlist_xyce(program)

    assert ".model foo.1 nmos" in out
    assert ".model foo.2 nmos" in out
    assert ".model foo.3 nmos" in out
    assert ".model foo nmos" not in out

