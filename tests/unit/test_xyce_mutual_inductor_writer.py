from io import StringIO

import re

from netlist import (
    Program,
    SourceFile,
    Ident,
    Ref,
    Float,
    Instance,
    ParamVal,
    netlist as write_netlist,
    NetlistDialects,
)
from netlist.write import WriteOptions


def test_xyce_writer_translates_spectre_mutual_inductor_to_k_element():
    """
    Spectre expresses mutual coupling as an element with module name 'mutual_inductor':
        k1 mutual_inductor coupling=0.9 ind1=ls1_1 ind2=ls1_2

    Xyce does NOT have a 'mutual_inductor' subckt/device. It requires the native K element:
        K1 ls1_1 ls1_2 0.9

    This test ensures the Xyce writer emits a K element and does NOT emit 'mutual_inductor'
    as a subcircuit instance.
    """
    inst = Instance(
        name=Ident("k1"),
        conns=[],
        module=Ref(ident=Ident("mutual_inductor")),
        params=[
            ParamVal(name=Ident("coupling"), val=Float(0.9)),
            ParamVal(name=Ident("ind1"), val=Ref(ident=Ident("ls1_1"))),
            ParamVal(name=Ident("ind2"), val=Ref(ident=Ident("ls1_2"))),
        ],
    )

    program = Program(files=[SourceFile(path="test.scs", contents=[inst])])
    dest = StringIO()
    write_netlist(src=program, dest=dest, options=WriteOptions(fmt=NetlistDialects.XYCE))
    out = dest.getvalue()

    # Must NOT emit the Spectre-style mutual_inductor subckt call.
    assert "mutual_inductor" not in out.lower()
    # Must not emit a subckt instance prefix for the mutual coupling element.
    assert re.search(r"(?im)^\s*Xk1\b", out) is None

    # Must emit a K element. We accept either "K1" or "K" + something that preserves the original.
    assert re.search(r"(?im)^\s*K1\s+ls1_1\s+ls1_2\s+0\.9\s*$", out) is not None


