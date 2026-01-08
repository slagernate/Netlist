"""
Xschem test-schematic generator.

Writes an xschem `.sch` that places every generated symbol, using the
`symbol_mapping.json` artifact produced by `BatchSymbolGenerator`.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def create_schematic_direct(symbol_dir: Path, output_sch: Path) -> Path:
    """Create a test schematic file directly."""

    def _parse_symbol_template_assignments(sym_path: Path) -> list[tuple[str, str]]:
        """Parse K-block template assignments from an Xschem .sym file.

        Returns a list of (key, value) in the order they appear in the template.
        """
        try:
            text = sym_path.read_text(errors="ignore")
        except Exception:
            return []

        m = re.search(r'template="([^"]*)"', text, flags=re.DOTALL)
        if not m:
            return []

        tmpl = m.group(1)
        out: list[tuple[str, str]] = []
        for line in tmpl.splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            out.append((k, v))
        return out

    mapping_file = symbol_dir / "symbol_mapping.json"
    mapping = json.loads(mapping_file.read_text())

    categories: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, info in mapping.items():
        prim_type = info.get("primitive_type", "unknown")
        ports = info.get("ports", [])
        port_count = len(ports) if ports else 0

        if prim_type == "mosfet":
            subckt_name = name.lower()
            is_pmos = subckt_name.startswith("p") or "pmos" in subckt_name or "pfet" in subckt_name
            mos_type = "pmos" if is_pmos else "nmos"
            if "mac" in subckt_name:
                category = f"{mos_type}_mac_{port_count}t"
            else:
                category = f"{mos_type}_{port_count}t"
        else:
            category = f"{prim_type}_{port_count}p"

        categories[category].append((name, info))

    category_order = [
        "nmos_4t",
        "pmos_4t",
        "nmos_5t",
        "pmos_5t",
        "nmos_6t",
        "pmos_6t",
        "nmos_mac_4t",
        "pmos_mac_4t",
        "capacitor_2p",
        "capacitor_3p",
        "resistor_2p",
        "resistor_3p",
        "diode_2p",
        "diode_3p",
        "inductor_2p",
        "inductor_3p",
        "inductor_4p",
        "unknown_0p",
        "unknown_1p",
        "unknown_2p",
        "unknown_3p",
        "unknown_4p",
        "unknown_5p",
        "unknown_6p",
    ]

    lines: list[str] = []
    lines.append("v {xschem version=3.0.0 file_version=1.2}")
    lines.append("G {}")
    lines.append("K {}")
    lines.append("V {}")
    lines.append("S {}")
    lines.append("E {}")

    x_spacing = 150
    x_start = 100
    y_start = 100
    current_x = x_start
    current_y = y_start
    row_height = 0

    inst_counters: dict[str, int] = defaultdict(int)

    def _inst_prefix(prim_type: str) -> str:
        if prim_type == "mosfet":
            return "M"
        if prim_type == "resistor":
            return "R"
        if prim_type == "capacitor":
            return "C"
        if prim_type == "diode":
            return "D"
        if prim_type == "inductor":
            return "L"
        return "X"

    def _emit_devices(devices: list[tuple[str, dict]], row_x_spacing: int) -> None:
        nonlocal current_x, current_y, row_height

        for name, info in devices:
            symbol_file = symbol_dir / f"{name}.sym"
            if not symbol_file.exists():
                continue

            prim_type = info.get("primitive_type", "unknown")
            prefix = _inst_prefix(prim_type)
            inst_counters[prefix] += 1
            inst_name = f"{prefix}{inst_counters[prefix]}"

            tmpl_kv = _parse_symbol_template_assignments(symbol_file)
            props = [("name", inst_name)]
            for k, v in tmpl_kv:
                if k.lower() == "name":
                    continue
                props.append((k, v))
            props_str = " ".join([f"{k}={v}" for k, v in props])

            # Symbol path resolution: xschem often resolves relative paths from its launch cwd.
            try:
                rel_path = symbol_file.relative_to(symbol_dir.parent)
            except Exception:
                rel_path = symbol_file.relative_to(output_sch.parent)

            lines.append(f"C {{{rel_path}}} {current_x} {current_y} 0 0 {{{props_str}}}")
            current_x += row_x_spacing
            row_height = max(row_height, 80)

    for category in category_order:
        devices = categories.get(category)
        if not devices:
            continue

        current_x = x_start
        current_y = current_y + row_height + 50
        row_height = 0
        row_x_spacing = 230 if category.startswith(("nmos_", "pmos_")) else x_spacing
        _emit_devices(devices, row_x_spacing=row_x_spacing)

    for category, devices in sorted(categories.items()):
        if category in category_order or not devices:
            continue

        current_x = x_start
        current_y = current_y + row_height + 50
        row_height = 0
        row_x_spacing = 230 if category.startswith(("nmos_", "pmos_")) else x_spacing
        _emit_devices(devices, row_x_spacing=row_x_spacing)

    output_sch.parent.mkdir(parents=True, exist_ok=True)
    output_sch.write_text("\n".join(lines) + "\n")
    return output_sch


def create_schematic_from_subckts(
    *,
    symbol_dir: Path,
    output_sch: Path,
    mapping: dict,
    subckts: dict,
) -> Path:
    """Create the placement schematic from in-memory `SubcktDef`s.

    This avoids re-reading `symbol_mapping.json` and avoids parsing `.sym` files
    to recover template-default properties.
    """
    categories: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, info in mapping.items():
        prim_type = info.get("primitive_type", "unknown")
        ports = info.get("ports", [])
        port_count = len(ports) if ports else 0

        if prim_type == "mosfet":
            subckt_name = name.lower()
            is_pmos = subckt_name.startswith("p") or "pmos" in subckt_name or "pfet" in subckt_name
            mos_type = "pmos" if is_pmos else "nmos"
            if "mac" in subckt_name:
                category = f"{mos_type}_mac_{port_count}t"
            else:
                category = f"{mos_type}_{port_count}t"
        else:
            category = f"{prim_type}_{port_count}p"

        categories[category].append((name, info))

    category_order = [
        "nmos_4t",
        "pmos_4t",
        "nmos_5t",
        "pmos_5t",
        "nmos_6t",
        "pmos_6t",
        "nmos_mac_4t",
        "pmos_mac_4t",
        "capacitor_2p",
        "capacitor_3p",
        "resistor_2p",
        "resistor_3p",
        "diode_2p",
        "diode_3p",
        "inductor_2p",
        "inductor_3p",
        "inductor_4p",
        "unknown_0p",
        "unknown_1p",
        "unknown_2p",
        "unknown_3p",
        "unknown_4p",
        "unknown_5p",
        "unknown_6p",
    ]

    lines: list[str] = []
    lines.append("v {xschem version=3.0.0 file_version=1.2}")
    lines.append("G {}")
    lines.append("K {}")
    lines.append("V {}")
    lines.append("S {}")
    lines.append("E {}")

    x_spacing = 150
    x_start = 100
    y_start = 100
    current_x = x_start
    current_y = y_start
    row_height = 0

    inst_counters: dict[str, int] = defaultdict(int)

    def _inst_prefix(prim_type: str) -> str:
        if prim_type == "mosfet":
            return "M"
        if prim_type == "resistor":
            return "R"
        if prim_type == "capacitor":
            return "C"
        if prim_type == "diode":
            return "D"
        if prim_type == "inductor":
            return "L"
        return "X"

    def _fmt_default(v) -> str:
        # ParamDecl.default is an Expr; in our strict mode it is MetricNum(str).
        if v is None:
            return ""
        return str(getattr(v, "val", v))

    def _is_constant_numeric_default_str(s: str) -> bool:
        ss = str(s).strip()
        if not ss:
            return False
        if any(ch in ss for ch in ["{", "}", "(", ")", "/", "*", "=", ","]):
            return False
        for i, ch in enumerate(ss):
            if ch in ["+", "-"]:
                if i == 0:
                    continue
                if ss[i - 1] in ["e", "E"]:
                    continue
                return False
        metric = r"(?:t|g|meg|k|m|u|n|p|f)?"
        if not re.match(rf"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?{metric}$", ss, flags=re.IGNORECASE):
            return False
        return True

    def _escape_prop_value(v: str) -> str:
        """Escape/quote a value for use inside an xschem instance `{...}` property block.

        Xschem uses braces to delimit the entire property string. If a value contains
        braces (common in SPICE/Xyce expressions like `{foo}`), unquoted nested braces
        can corrupt parsing and make attributes disappear.
        """
        if v is None:
            return ""
        s = str(v)
        if s == "":
            return ""
        # Quote anything with spaces or braces or quotes.
        if any(ch in s for ch in ['{', '}', ' ', '\t', '"', '\n']):
            # Escape backslashes/quotes/newlines first...
            s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            # ...then escape braces so they can't terminate the outer `{ ... }` property block.
            s = s.replace("{", "\\{").replace("}", "\\}")
            return f'"{s}"'
        return s

    def _emit_devices(devices: list[tuple[str, dict]], row_x_spacing: int) -> None:
        nonlocal current_x, current_y, row_height

        for name, info in devices:
            symbol_file = symbol_dir / f"{name}.sym"
            if not symbol_file.exists():
                continue

            prim_type = info.get("primitive_type", "unknown")
            prefix = _inst_prefix(prim_type)
            inst_counters[prefix] += 1
            inst_name = f"{prefix}{inst_counters[prefix]}"

            # Build instance properties from in-memory subckt defaults
            props = [("name", inst_name)]
            subckt = subckts.get(name)
            if subckt is not None:
                # Keep placement schematic simulator-agnostic:
                # include ALL header params with constant numeric defaults.
                for p in getattr(subckt, "params", []):
                    dv = _fmt_default(p.default)
                    if _is_constant_numeric_default_str(dv):
                        props.append((p.name.name, dv))
                props.append(("model", name))
                props.append(("spiceprefix", "X"))

            # Put each attribute on its own line to avoid Xschem UI truncation and
            # to make long brace-expressions readable. Values that contain braces
            # are quoted by `_escape_prop_value()` to avoid nested `{}` issues.
            props_str = "\n".join([f"{k}={_escape_prop_value(v)}" for k, v in props if v != ""])

            try:
                rel_path = symbol_file.relative_to(symbol_dir.parent)
            except Exception:
                rel_path = symbol_file.relative_to(output_sch.parent)

            lines.append(f"C {{{rel_path}}} {current_x} {current_y} 0 0 {{{props_str}}}")
            current_x += row_x_spacing
            row_height = max(row_height, 80)

    for category in category_order:
        devices = categories.get(category)
        if not devices:
            continue

        current_x = x_start
        current_y = current_y + row_height + 50
        row_height = 0
        row_x_spacing = 230 if category.startswith(("nmos_", "pmos_")) else x_spacing
        _emit_devices(devices, row_x_spacing=row_x_spacing)

    for category, devices in sorted(categories.items()):
        if category in category_order or not devices:
            continue

        current_x = x_start
        current_y = current_y + row_height + 50
        row_height = 0
        row_x_spacing = 230 if category.startswith(("nmos_", "pmos_")) else x_spacing
        _emit_devices(devices, row_x_spacing=row_x_spacing)

    output_sch.parent.mkdir(parents=True, exist_ok=True)
    output_sch.write_text("\n".join(lines) + "\n")
    return output_sch


