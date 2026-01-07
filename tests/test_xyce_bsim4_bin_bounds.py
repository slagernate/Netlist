from __future__ import annotations

from pathlib import Path

from netlist import NetlistDialects, ParseOptions, WriteOptions, netlist, parse_files


def test_xyce_bsim4_bin_bounds_constant_fold(tmp_path: Path) -> None:
    """
    Xyce cannot accept expressions/parameter refs in BSIM4 bin bounds
    (lmin/lmax/wmin/wmax). We must constant-fold these during netlisting.
    """
    src = tmp_path / "in.scs"
    src.write_text(
        "\n".join(
            [
                'simulator lang=spectre',
                '',
                'parameters dxlp_33=1e-08',
                'parameters enable_mismatch=0',
                # mimic mismatch-derived term name (we treat *misp* as zero for bounds)
                'parameters dxlmisp_33=1e-06',
                '',
                'model pch_33 bsim4 {',
                '  1: type=p',
                '  + lmin=1e-05-(dxlp_33+dxlmisp_33)',
                '  + lmax=9e-04',
                '  + wmin=1e-05',
                '  + wmax=9e-04',
                '}',
                '',
            ]
        )
        + "\n"
    )

    program = parse_files(src, options=ParseOptions(dialect=NetlistDialects.SPECTRE, recurse=False))

    out = tmp_path / "out.cir"
    with out.open("w") as f:
        netlist(
            src=program,
            dest=f,
            options=WriteOptions(fmt=NetlistDialects.XYCE, file_type="models"),
        )

    text = out.read_text()
    # The folded lmin should be numeric: 1e-5 - 1e-8 - 0 (dxlmisp treated as 0) = 9.99e-6.
    assert "lmin=" in text.lower()
    lmin_lines = [ln for ln in text.splitlines() if "lmin" in ln.lower()]
    assert lmin_lines, "expected an lmin line in output"
    lmin_joined = "\n".join(lmin_lines).lower()
    assert "dxlp_33" not in lmin_joined
    assert "dxlmisp_33" not in lmin_joined
    assert "{" not in lmin_joined



