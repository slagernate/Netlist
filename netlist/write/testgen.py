from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..data import Library, LibSectionDef, SubcktDef, UseLibSection


def _safe_dirname(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "subckt"


def _iter_sections(program) -> Iterable[LibSectionDef]:
    for f in getattr(program, "files", []):
        for e in getattr(f, "contents", []):
            if isinstance(e, LibSectionDef):
                yield e
            elif isinstance(e, Library):
                for sec in e.sections:
                    yield sec


def _collect_lib_section_names(program) -> Set[str]:
    return {sec.name.name for sec in _iter_sections(program)}


def _collect_subckts(program) -> Dict[str, SubcktDef]:
    """Collect all SubcktDef nodes anywhere in the AST (including Library sections)."""
    found: Dict[str, SubcktDef] = {}

    def visit_entry(e) -> None:
        if isinstance(e, SubcktDef):
            found.setdefault(e.name.name, e)
            for sub in e.entries:
                visit_entry(sub)
            return
        if isinstance(e, LibSectionDef):
            for sub in e.entries:
                visit_entry(sub)
            return
        if isinstance(e, Library):
            for sec in e.sections:
                for sub in sec.entries:
                    visit_entry(sub)
            return

    for f in getattr(program, "files", []):
        for e in getattr(f, "contents", []):
            visit_entry(e)

    return found


def _collect_subckt_defining_sections(program) -> Dict[str, str]:
    """Map subckt name -> library section name that defines it (first hit wins)."""
    subckt_to_section: Dict[str, str] = {}
    for sec in _iter_sections(program):
        sec_name = sec.name.name
        for entry in sec.entries:
            if isinstance(entry, SubcktDef):
                subckt_to_section.setdefault(entry.name.name, sec_name)
    return subckt_to_section


def _collect_section_dependencies(program) -> Dict[str, Set[str]]:
    """Map section name -> set of sections it references via UseLibSection."""
    deps: Dict[str, Set[str]] = {}
    for sec in _iter_sections(program):
        sec_name = sec.name.name
        sdeps = deps.setdefault(sec_name, set())
        for entry in sec.entries:
            if isinstance(entry, UseLibSection):
                sdeps.add(entry.section.name)
    return deps


def _pick_preferred_corner_section(*, defining_section: str, section_deps: Dict[str, Set[str]]) -> str:
    """Pick a 'reasonable default' corner wrapper section for a subckt.

    Preference order:
    - Any section starting with 'tt' that includes `defining_section`
    - Otherwise `defining_section` itself
    """
    defining_lower = defining_section.lower()
    tt_wrappers: List[str] = []
    for sec, deps in section_deps.items():
        if defining_section in deps or defining_lower in {d.lower() for d in deps}:
            if sec.lower().startswith("tt"):
                tt_wrappers.append(sec)

    if tt_wrappers:
        # Prefer the shortest "tt_*" wrapper (usually the intended one, e.g. tt_rfmos).
        return sorted(tt_wrappers, key=lambda s: (len(s), s.lower()))[0]

    return defining_section


@dataclass(frozen=True)
class XyceSubcktTestConfig:
    """Config for generating per-subckt Xyce sanity netlists."""

    # Base sections commonly needed for "typical" device families
    base_sections: Tuple[str, ...] = ("tt", "tt_res", "tt_mos_cap")
    # Always try to include this generic mismatch helper section if present
    include_stat_mis: bool = True
    # Include the flicker-noise statistical helper section if present (defines random_fn__process__)
    include_stat_noise: bool = True


def _write_one_subckt_test_netlist(
    *,
    dest_path: Path,
    subckt: SubcktDef,
    lib_filename: str,
    include_sections: List[str],
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    port_count = len(subckt.ports)
    nodes = [f"n{i}" for i in range(port_count)]

    lines: List[str] = []
    lines.append(f"* Sanity Check Netlist for subckt: {subckt.name.name}\n")
    lines.append("\n")
    lines.append("* Monte Carlo Control Parameters\n")
    lines.append(".param process_mc_factor=0\n")
    lines.append(".param enable_mismatch=0\n")
    lines.append(".param mismatch_factor=0\n")
    lines.append(".param corner_factor=0\n")
    lines.append("\n")
    lines.append("* Includes\n")
    for sec in include_sections:
        lines.append(f".lib '{lib_filename}' {sec}\n")
    lines.append("\n")
    lines.append("* Ground Reference\n")
    lines.append("R_gnd_ref 0 0 1M\n")
    lines.append("\n")
    lines.append("* Port pull-downs\n")
    for n in nodes:
        lines.append(f"R_pd_{n} {n} 0 1M\n")
    lines.append("\n")
    lines.append("* Device Instantiation\n")
    inst_name = f"X_{subckt.name.name}"
    if nodes:
        lines.append(f"{inst_name} {' '.join(nodes)} {subckt.name.name}\n")
    else:
        lines.append(f"{inst_name} {subckt.name.name}\n")
    lines.append("+ m=1\n")
    lines.append("\n")
    lines.append("* Simulation Commands\n")
    lines.append(".tran 1n 10n\n")
    lines.append(".print tran V(*)\n")
    lines.append(".end\n")

    dest_path.write_text("".join(lines))


def generate_xyce_subckt_tests(
    *,
    program,
    output_test_dir: Path,
    lib_filename: str,
    config: Optional[XyceSubcktTestConfig] = None,
) -> int:
    """Generate one `test.cir` per subckt under `output_test_dir/<subckt>/test.cir`."""
    cfg = config or XyceSubcktTestConfig()

    subckts = _collect_subckts(program)
    subckt_sections = _collect_subckt_defining_sections(program)
    section_deps = _collect_section_dependencies(program)
    all_sections = _collect_lib_section_names(program)

    created = 0
    used_dirnames: Set[str] = set()

    for subckt_name, subckt in sorted(subckts.items(), key=lambda kv: kv[0].lower()):
        dname = _safe_dirname(subckt_name)
        if dname in used_dirnames:
            suffix = 2
            while f"{dname}_{suffix}" in used_dirnames:
                suffix += 1
            dname = f"{dname}_{suffix}"
        used_dirnames.add(dname)

        defining_section = subckt_sections.get(subckt_name, "tt")
        preferred_section = _pick_preferred_corner_section(defining_section=defining_section, section_deps=section_deps)

        # Include strategy:
        # - always include common base sections (if present)
        # - include the "best" section for this subckt
        # - include mismatch helper sections if present (stat_mis and stat_mis_<suffix>)
        include_sections: List[str] = []
        for sec in cfg.base_sections:
            if sec in all_sections:
                include_sections.append(sec)

        if preferred_section in all_sections and preferred_section not in include_sections:
            include_sections.append(preferred_section)

        if cfg.include_stat_mis:
            extra: List[str] = []
            if preferred_section.lower().startswith("tt_"):
                extra.append(f"stat_mis_{preferred_section[3:]}")
            extra.append("stat_mis")

            for sec in extra:
                if sec in all_sections and sec not in include_sections:
                    include_sections.append(sec)

        if cfg.include_stat_noise and "stat_noise" in all_sections and "stat_noise" not in include_sections:
            include_sections.append("stat_noise")

        dest_path = output_test_dir / dname / "test.cir"
        _write_one_subckt_test_netlist(
            dest_path=dest_path,
            subckt=subckt,
            lib_filename=lib_filename,
            include_sections=include_sections,
        )
        created += 1

    return created


