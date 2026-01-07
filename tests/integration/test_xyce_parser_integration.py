from pathlib import Path

from netlist import ParseOptions, parse_files
from netlist.data import NetlistDialects, ParamDecls, Primitive, SubcktDef


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "spectre_to_xyce"


def test_parse_xyce_emitted_parameters_out_cir() -> None:
    prog = parse_files(
        [EXAMPLES_DIR / "parameters" / "out.cir"],
        options=ParseOptions(dialect=NetlistDialects.XYCE, recurse=False),
    )
    # Should parse .param statements into ParamDecls.
    assert any(isinstance(e, ParamDecls) for e in prog.files[0].contents)


def test_parse_xyce_emitted_bsources_out_cir() -> None:
    prog = parse_files(
        [EXAMPLES_DIR / "bsources" / "out.cir"],
        options=ParseOptions(dialect=NetlistDialects.XYCE, recurse=False),
    )
    # Should parse Bsource_* primitive instances.
    assert any(isinstance(e, Primitive) for e in prog.files[0].contents)


def test_parse_xyce_emitted_subckts_out_cir() -> None:
    prog = parse_files(
        [EXAMPLES_DIR / "subckts" / "out.cir"],
        options=ParseOptions(dialect=NetlistDialects.XYCE, recurse=False),
    )
    # Should parse multiple subckt definitions and tolerate ';' comments.
    subckts = [e for e in prog.files[0].contents if isinstance(e, SubcktDef)]
    assert len(subckts) >= 2



