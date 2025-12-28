from netlist.parse import parse_str, ParseOptions
from netlist.data import NetlistDialects, SubcktDef, Instance, ParamDecl, ParamVal, BinaryOp


def _first_entry_of_type(entries, tp):
    for e in entries:
        if isinstance(e, tp):
            return e
    raise AssertionError(f"Expected entry of type {tp.__name__} not found")


def test_xyce_parse_subckt_params_colon() -> None:
    txt = """
.subckt mysub a b PARAMS: l=1 w={1+2}
.ends mysub
"""
    prog = parse_str(txt, options=ParseOptions(dialect=NetlistDialects.XYCE, recurse=False))
    sub = _first_entry_of_type(prog.files[0].contents, SubcktDef)

    assert sub.name.name == "mysub"
    assert [p.name for p in sub.ports] == ["a", "b"]

    # Subckt params should be captured as ParamDecl with defaults.
    assert any(isinstance(p, ParamDecl) and p.name.name == "l" for p in sub.params)
    assert any(isinstance(p, ParamDecl) and p.name.name == "w" for p in sub.params)

    w = [p for p in sub.params if p.name.name == "w"][0]
    assert isinstance(w.default, BinaryOp)


def test_xyce_parse_instance_params_colon() -> None:
    txt = """
.subckt child a b PARAMS: foo=1
.ends child

.subckt top a b
X1 a b child PARAMS: foo={1+2}
.ends top
"""
    prog = parse_str(txt, options=ParseOptions(dialect=NetlistDialects.XYCE, recurse=False))
    top = [e for e in prog.files[0].contents if isinstance(e, SubcktDef) and e.name.name == "top"][0]

    inst = _first_entry_of_type(top.entries, Instance)
    assert inst.name.name.lower().startswith("x")
    assert inst.module.ident.name == "child"
    assert [n.name for n in inst.conns] == ["a", "b"]

    assert any(isinstance(p, ParamVal) and p.name.name == "foo" for p in inst.params)
    foo = [p for p in inst.params if p.name.name == "foo"][0]
    assert isinstance(foo.val, BinaryOp)


