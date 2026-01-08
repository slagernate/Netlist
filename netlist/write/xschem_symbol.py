"""
Xschem Symbol Generator

Generates xschem .sym symbol files for primitive device types.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
from enum import Enum

# Optional yaml support (for primitive overrides config)
try:
    import yaml  # type: ignore

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class PrimitiveType(Enum):
    """Enumerated primitive device types (Xschem symbol/template classification)."""

    MOSFET = "mosfet"
    BJT = "bjt"
    DIODE = "diode"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    UNKNOWN = "unknown"


class PrimitiveDetector:
    """Detects primitive device types from SubcktDef structures (Xschem-focused)."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_overrides: Dict[str, PrimitiveType] = {}
        if config_file:
            self._load_config(config_file)

    def _load_config(self, config_file: str) -> None:
        p = Path(config_file)
        if not p.exists():
            return
        try:
            with open(p, "r") as f:
                if p.suffix.lower() in [".yaml", ".yml"]:
                    if HAS_YAML:
                        config = yaml.safe_load(f)
                    else:
                        return
                else:
                    config = json.load(f)
            if isinstance(config, dict):
                for subckt_name, prim_type_str in config.items():
                    try:
                        self.config_overrides[subckt_name.lower()] = PrimitiveType(prim_type_str.lower())
                    except ValueError:
                        pass
        except Exception:
            pass

    def detect(self, subckt: "SubcktDef") -> PrimitiveType:
        subckt_name_lower = subckt.name.name.lower()
        if subckt_name_lower in self.config_overrides:
            return self.config_overrides[subckt_name_lower]

        port_count = len(subckt.ports)
        param_names = {p.name.name.lower() for p in subckt.params}

        name_has_mos = any(k in subckt_name_lower for k in ["mos", "fet", "transistor", "nch", "pch"])
        name_has_bjt = any(k in subckt_name_lower for k in ["npn", "pnp", "bjt", "q"])
        name_has_res = any(k in subckt_name_lower for k in ["resistor", "res"])
        name_has_diode = any(k in subckt_name_lower for k in ["diode", "dio", "sbd"])
        name_has_cap = any(k in subckt_name_lower for k in ["capacitor", "cap"])
        name_has_ind = any(k in subckt_name_lower for k in ["inductor", "ind"])

        has_l_w = {"l", "w"} <= param_names
        has_lr_wr = {"lr", "wr"} <= param_names
        name_is_nch_pch = ("nch" in subckt_name_lower) or ("pch" in subckt_name_lower)
        if port_count in [4, 5, 6] and (has_l_w or has_lr_wr or name_is_nch_pch) and name_has_mos:
            return PrimitiveType.MOSFET

        if port_count == 4 and name_has_bjt:
            return PrimitiveType.BJT

        name_has_dio = ("dio" in subckt_name_lower) or ("sbd" in subckt_name_lower)
        is_r_device = subckt_name_lower.startswith("r")
        name_has_nod = (not is_r_device) and ("nod" in subckt_name_lower)
        name_has_pod = (not is_r_device) and ("pod" in subckt_name_lower)
        has_diode_params = any(p in param_names for p in ["area", "perim", "pj", "aw", "al"])
        port_names_lower = [p.name.lower() for p in subckt.ports]
        is_sbd = "sbd" in subckt_name_lower
        has_gnode_like = any(
            p in ["gnode", "gnd", "gnd!", "sub", "substrate", "bulk", "body", "b"] for p in port_names_lower
        )
        if (port_count in [2, 3]) and (name_has_diode or name_has_dio or name_has_nod or name_has_pod or has_diode_params) and (
            port_count == 2 or has_gnode_like or is_sbd
        ):
            return PrimitiveType.DIODE

        if subckt_name_lower.startswith("r") and port_count in [2, 3]:
            return PrimitiveType.RESISTOR

        if port_count == 2 and (name_has_res or ("r" in param_names)):
            return PrimitiveType.RESISTOR

        if port_count in [2, 3] and (name_has_cap or ("c" in param_names)):
            return PrimitiveType.CAPACITOR

        name_has_spiral = "spiral" in subckt_name_lower
        if port_count in [2, 3, 4] and (name_has_ind or name_has_spiral) and not has_l_w:
            return PrimitiveType.INDUCTOR

        return PrimitiveType.UNKNOWN

from ..data import SubcktDef, ParamDecl, Ident, MetricNum


class SymbolMatcher:
    """Matches subcircuits to generic Xschem symbol templates using heuristics.

    Xschem-specific by design: it assumes templates are `.sym` files organized by
    primitive type in `xschem_symbol_templates/<primitive>/...`.
    """

    def __init__(self, template_dir: Path):
        self.template_dir = Path(template_dir)
        self._load_available_templates()

    def _load_available_templates(self) -> None:
        self.templates: Dict[PrimitiveType, List[Path]] = {
            PrimitiveType.MOSFET: [],
            PrimitiveType.RESISTOR: [],
            PrimitiveType.CAPACITOR: [],
            PrimitiveType.DIODE: [],
            PrimitiveType.BJT: [],
            PrimitiveType.INDUCTOR: [],
        }

        mosfet_dir = self.template_dir / "mosfet"
        if mosfet_dir.exists():
            self.templates[PrimitiveType.MOSFET].extend(mosfet_dir.glob("*.sym"))

        resistor_dir = self.template_dir / "resistor"
        if resistor_dir.exists():
            self.templates[PrimitiveType.RESISTOR].extend(resistor_dir.glob("*.sym"))

        for dev_type in [
            PrimitiveType.CAPACITOR,
            PrimitiveType.DIODE,
            PrimitiveType.BJT,
            PrimitiveType.INDUCTOR,
        ]:
            dev_dir = self.template_dir / dev_type.value
            if dev_dir.exists():
                self.templates[dev_type].extend(dev_dir.glob("*.sym"))

    def match(self, subckt: SubcktDef, prim_type: PrimitiveType) -> Tuple[Optional[Path], float, List[str]]:
        """Return (template_path, confidence, reasons)."""
        if prim_type == PrimitiveType.UNKNOWN:
            return None, 0.0, ["primitive_type_unknown"]

        available_templates = self.templates.get(prim_type, [])
        if not available_templates:
            return None, 0.0, [f"no_templates_available_for_{prim_type.value}"]

        best_template: Optional[Path] = None
        best_score = 0.0
        best_reasons: List[str] = []

        subckt_name_lower = subckt.name.name.lower()
        port_names = {p.name.lower() for p in subckt.ports}
        param_names = {p.name.name.lower() for p in subckt.params}

        for template_path in available_templates:
            score = 0.0
            reasons: List[str] = []
            template_name = template_path.stem.lower()

            if prim_type == PrimitiveType.MOSFET:
                is_pmos = subckt_name_lower.startswith("p") or "pmos" in subckt_name_lower or "pfet" in subckt_name_lower
                is_nmos = subckt_name_lower.startswith("n") or "nmos" in subckt_name_lower or "nfet" in subckt_name_lower

                if "pmos" in template_name or "pfet" in template_name:
                    if is_pmos:
                        score += 0.4
                        reasons.append("name_contains_pmos")
                    elif is_nmos:
                        score -= 0.2
                elif "nmos" in template_name or "nfet" in template_name:
                    if is_nmos:
                        score += 0.4
                        reasons.append("name_contains_nmos")
                    elif is_pmos:
                        score -= 0.2

                if "4pin" in template_name:
                    if len(subckt.ports) == 4:
                        score += 0.3
                        reasons.append("pins_match_4pin")
                        has_dgsb = port_names >= {"d", "g", "s", "b"}
                        has_n1234 = port_names >= {"n1", "n2", "n3", "n4"}
                        has_digisibi = port_names >= {"di", "gi", "si", "bi"}
                        if has_dgsb or has_n1234 or has_digisibi:
                            score += 0.2
                            if has_dgsb:
                                reasons.append("pins_match_dgsb")
                            elif has_n1234:
                                reasons.append("pins_match_n1n2n3n4_as_dgsb")
                            else:
                                reasons.append("pins_match_di_gi_si_bi_as_dgsb")
                    elif len(subckt.ports) in [5, 6]:
                        score += 0.1
                        reasons.append("pins_match_variant")

                if "lr" in param_names and "wr" in param_names:
                    score += 0.1
                    reasons.append("has_lr_wr_params")
                elif "l" in param_names and "w" in param_names:
                    score += 0.1
                    reasons.append("has_L_W_params")

            elif prim_type == PrimitiveType.CAPACITOR:
                if "capacitor" in template_name or "cap" in template_name:
                    if "cap" in subckt_name_lower or "capacitor" in subckt_name_lower:
                        score += 0.3
                        reasons.append("name_contains_cap")

                if "2pin" in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.1
                        reasons.append("pins_match_3pin")
                elif "3pin" in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.1
                        reasons.append("pins_match_2pin")

                if "c" in param_names or "w" in param_names or "l" in param_names:
                    score += 0.2
                    reasons.append("has_cap_params")

            elif prim_type == PrimitiveType.RESISTOR:
                if "resistor" in template_name or "res" in template_name:
                    if "res" in subckt_name_lower or "resistor" in subckt_name_lower:
                        score += 0.2
                        reasons.append("name_contains_res")
                    else:
                        score += 0.1
                        reasons.append("resistor_type_detected")

                if "2pin" in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.2
                        reasons.append("pins_match_3pin")
                elif "3pin" in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.2
                        reasons.append("pins_match_2pin")

                if "r" in param_names:
                    score += 0.2
                    reasons.append("has_r_param")
                elif "l" in param_names or "w" in param_names:
                    score += 0.15
                    reasons.append("has_L_W_params")

            elif prim_type == PrimitiveType.INDUCTOR:
                if "2pin" in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.3
                        reasons.append("pins_match_2pin_can_add_3rd")
                    elif len(subckt.ports) == 4:
                        score += 0.25
                        reasons.append("pins_match_2pin_can_add_extras")
                elif "3pin" in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.2
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 4:
                        score += 0.35
                        reasons.append("pins_match_3pin_can_add_4th")

                if "inductor" in template_name or "ind" in template_name:
                    if "ind" in subckt_name_lower or "spiral" in subckt_name_lower:
                        score += 0.2
                        reasons.append("name_contains_ind")

            else:
                if len(subckt.ports) == 2:
                    score += 0.3
                    reasons.append("pins_match_2pin")
                elif len(subckt.ports) == 4:
                    score += 0.3
                    reasons.append("pins_match_4pin")

            if score > best_score:
                best_score = score
                best_template = template_path
                best_reasons = reasons

        confidence = min(1.0, max(0.0, best_score))
        threshold = 0.2 if prim_type in [PrimitiveType.RESISTOR, PrimitiveType.INDUCTOR] else 0.3

        if best_template and confidence > threshold:
            return best_template, confidence, best_reasons
        if best_reasons:
            return None, confidence, best_reasons + ["below_threshold"]
        return None, confidence, ["no_match_found", f"available_templates: {[t.name for t in available_templates]}"]


class XschemSymbolGenerator:
    """Generates xschem .sym symbol files for primitive device types"""
    
    # Pin positions for different primitive types
    # Format: (x, y) coordinates, matching Sky130 style (compact, -30 to 30 range)
    PIN_POSITIONS = {
        PrimitiveType.MOSFET: {
            'd': (20, -20),   # Drain (right, top) - matches Sky130 nfet
            'g': (-20, 0),    # Gate (left, center) - matches Sky130
            's': (20, 20),    # Source (right, bottom) - matches Sky130 nfet
            'b': (0, 0),      # Bulk/Substrate (center) - will be positioned separately
            # Extra pins (for 5t, 6t variants) - placed off to the side
            'ng': (30, -20),  # N-gate (far right, top)
            'pg': (-30, -20), # P-gate (far left, top)
        },
        PrimitiveType.BJT: {
            'c': (-20, 0),    # Collector (left)
            'b': (0, -20),    # Base (top)
            'e': (20, 0),     # Emitter (right)
            's': (0, 20),     # Substrate (bottom)
        },
        PrimitiveType.DIODE: {
            'p': (-30, 0),    # Anode (left)
            'n': (30, 0),     # Cathode (right)
        },
        PrimitiveType.RESISTOR: {
            'p': (0, -30),    # Positive terminal (top) - matches Sky130
            'n': (0, 30),     # Negative terminal (bottom) - matches Sky130
        },
        PrimitiveType.CAPACITOR: {
            'p': (-20, 0),    # Positive terminal (left)
            'n': (20, 0),     # Negative terminal (right)
        },
        PrimitiveType.INDUCTOR: {
            'p': (-20, 0),    # Positive terminal (left)
            'n': (20, 0),     # Negative terminal (right)
        },
    }
    
    # Spice primitive prefixes
    SPICE_PREFIX = {
        PrimitiveType.MOSFET: 'M',
        PrimitiveType.BJT: 'Q',
        PrimitiveType.DIODE: 'D',
        PrimitiveType.RESISTOR: 'R',
        PrimitiveType.CAPACITOR: 'C',
        PrimitiveType.INDUCTOR: 'L',
    }
    
    def __init__(self, output_dir: Path, template_dir: Optional[Path] = None):
        """Initialize the symbol generator.
        
        Args:
            output_dir: Directory where .sym files will be written
            template_dir: Optional directory containing generic symbol templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(template_dir) if template_dir else None
        self._generated_symbols: Dict[PrimitiveType, str] = {}

    @staticmethod
    def _is_constant_numeric_default_str(s: str) -> bool:
        """Return True if s looks like a plain numeric literal (optionally with metric suffix).

        This is used to keep generated symbols simulator-agnostic: we only pre-populate
        numeric constants and let derived/expr defaults come from the simulator model library.
        """
        ss = str(s).strip()
        if not ss:
            return False
        # Reject anything that is obviously an expression / reference.
        if any(ch in ss for ch in ["{", "}", "(", ")", "/", "*", "=", ","]):
            return False
        # Allow a leading sign and exponent sign; reject other operator usage.
        # If there's a + or - not at the beginning and not immediately after e/E, treat as expression.
        for i, ch in enumerate(ss):
            if ch in ["+", "-"]:
                if i == 0:
                    continue
                if ss[i - 1] in ["e", "E"]:
                    continue
                return False
        metric = r"(?:t|g|meg|k|m|u|n|p|f)?"
        pat = re.compile(rf"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?{metric}$", re.IGNORECASE)
        return bool(pat.match(ss))

    @staticmethod
    def _primary_param_names_for_prim(prim_type: PrimitiveType) -> List[str]:
        """Small set of user-facing params to propagate into symbols/netlists.

        These are intended to be simulator-agnostic, and avoid exporting hundreds of
        model-internal tuning coefficients even if they appear as header PARAMS.
        """
        if prim_type == PrimitiveType.MOSFET:
            # RF MOS subckts often use (wr, lr, nr); 3.3V macro MOS often uses (w, l, nf, multi).
            # Keep sigma as a common constant toggle and allow multi-finger/replication knobs.
            return ["wr", "lr", "nr", "w", "l", "nf", "multi", "sigma", "m", "mf"]
        if prim_type == PrimitiveType.RESISTOR:
            return ["wr", "lr", "w", "l", "r", "nf", "multi", "m", "mf"]
        if prim_type == PrimitiveType.CAPACITOR:
            return ["w", "l", "wr", "lr", "c", "nf", "multi", "m", "mf"]
        if prim_type == PrimitiveType.INDUCTOR:
            return ["w", "rad", "nr", "l", "nf", "multi", "m", "mf"]
        if prim_type == PrimitiveType.DIODE:
            return ["area", "perim", "pj", "aw", "al", "nf", "multi", "m", "mf"]
        if prim_type == PrimitiveType.BJT:
            return ["area", "nf", "multi", "m", "mf"]
        return []
    
    def generate_symbol(self, subckt: SubcktDef, prim_type: PrimitiveType, 
                       template_path: Optional[Path] = None) -> Path:
        """Generate a .sym file for a primitive subcircuit.
        Uses IR subcircuit name to preserve fidelity to the IR.
        
        Args:
            subckt: The SubcktDef to generate symbol for
            prim_type: The detected primitive type
            template_path: Optional path to a template symbol file to adapt
            
        Returns:
            Path to the generated .sym file
        """
        # Use IR subcircuit name (not primitive type name)
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        sym_file = self.output_dir / f"{subckt_name}.sym"
        
        # If we've already generated this symbol, reuse it
        if subckt_name in self._generated_symbols:
            return sym_file
        
        # Generate the symbol file content
        if template_path and template_path.exists():
            content = self._load_and_adapt_template(template_path, subckt, prim_type)
        else:
            content = self._generate_symbol_content(subckt, prim_type)
        # Note: source context is added in `_generate_symbol_content()` as K-block attributes.
        
        # Write to file
        with open(sym_file, 'w') as f:
            f.write(content)
        
        self._generated_symbols[subckt_name] = subckt_name
        return sym_file
    
    def _load_and_adapt_template(self, template_path: Path, subckt: SubcktDef, 
                                 prim_type: PrimitiveType) -> str:
        """Load a template symbol file and adapt it to the subcircuit.
        
        Args:
            template_path: Path to the template .sym file
            subckt: The SubcktDef to adapt the template for
            prim_type: The detected primitive type
            
        Returns:
            Adapted symbol file content as string
        """
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Extract the K block from template
        import re
        k_block_match = re.search(r'K \{([^}]+)\}', template_content, re.DOTALL)
        if not k_block_match:
            # Fall back to generating from scratch
            return self._generate_symbol_content(subckt, prim_type)
        
        # Extract drawing elements (L, P, A, B, T lines) from template
        drawing_lines = []
        template_pin_blocks = []
        template_text_blocks = []
        
        for line in template_content.split('\n'):
            line = line.strip()
            if line.startswith('L ') or line.startswith('P ') or line.startswith('A ') or line.startswith('R '):
                drawing_lines.append(line)
            elif line.startswith('B '):
                template_pin_blocks.append(line)
            elif line.startswith('T '):
                template_text_blocks.append(line)
        
        # Generate K block, pins, and text using our logic
        # but use template's drawing elements
        content = self._generate_symbol_content(subckt, prim_type)
        
        # Replace drawing elements with template's drawing
        # BUT: For MOSFETs, use our generated drawing (which correctly handles PMOS vs NMOS)
        # For other types, use template drawing
        use_template_drawing = (prim_type != PrimitiveType.MOSFET)
        
        # Find where drawing elements start (after E block)
        lines = content.split('\n')
        new_lines = []
        in_drawing = False
        drawing_inserted = False
        
        for line in lines:
            if line.strip() == 'E {}':
                new_lines.append(line)
                in_drawing = True
                continue
            elif in_drawing and not drawing_inserted:
                if use_template_drawing:
                    # Insert template drawing elements here
                    for draw_line in drawing_lines:
                        new_lines.append(draw_line)
                # Skip our generated drawing elements until we hit pins or text
                if line.startswith('B ') or line.startswith('T '):
                    new_lines.append(line)
                elif not use_template_drawing:
                    # For MOSFETs, keep our generated drawing
                    if line.startswith('L ') or line.startswith('P ') or line.startswith('A ') or line.startswith('R '):
                        new_lines.append(line)
                drawing_inserted = True
            elif line.startswith('B ') or line.startswith('T '):
                # Keep our generated pins and text
                new_lines.append(line)
            elif not in_drawing:
                # Keep everything before drawing (header, K, V, S, E)
                # But make sure model=@model is replaced with actual model name in template string
                subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
                # Replace model=@model in template attribute (can span multiple lines)
                if 'model=@model' in line:
                    line = line.replace('model=@model', f'model={subckt_name}')
                new_lines.append(line)
            elif use_template_drawing and (line.startswith('L ') or line.startswith('P ') or line.startswith('A ') or line.startswith('R ')):
                # Skip our generated drawing if using template drawing
                continue
            elif not use_template_drawing and (line.startswith('L ') or line.startswith('P ') or line.startswith('A ') or line.startswith('R ')):
                # Keep our generated drawing for MOSFETs
                new_lines.append(line)
        
        # Final pass: replace any remaining @model in the entire content
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        result = '\n'.join(new_lines) + '\n'
        # Replace model=@model in template strings (handle multi-line)
        # Also handle cases where @model appears without = (e.g., in format strings)
        # BUT: Don't replace @model in text display strings (T blocks) - those should stay as @model
        # Only replace in format, lvs_format, drc, and template attributes
        import re
        # Replace @model in format strings (can span multiple lines with \n+)
        # Match format="...@model..." or format="...@model\n+..."
        result = re.sub(r'(format="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.MULTILINE)
        # Also handle multi-line format strings
        result = re.sub(r'(format="[^"]*)\n\+[^"]*@model([^"]*")', r'\1\n+' + subckt_name + r'\2', result, flags=re.MULTILINE)
        # Replace in lvs_format strings (can also span multiple lines)
        result = re.sub(r'(lvs_format="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.MULTILINE)
        result = re.sub(r'(lvs_format="[^"]*)\n\+[^"]*@model([^"]*")', r'\1\n+' + subckt_name + r'\2', result, flags=re.MULTILINE)
        # Replace in drc strings
        result = re.sub(r'(drc="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.MULTILINE)
        # Replace model=@model in template (can span multiple lines)
        result = result.replace('model=@model', f'model={subckt_name}')
        # Final catch-all: replace any remaining @model in format/lvs_format/drc (but NOT in T blocks)
        # This handles cases where the regex didn't match
        lines = result.split('\n')
        new_lines = []
        in_format_attr = False
        for line in lines:
            if 'format="' in line or 'lvs_format="' in line or 'drc="' in line:
                in_format_attr = True
                line = line.replace('@model', subckt_name)
            elif in_format_attr and line.strip().startswith('+'):
                # Continuation line of format string
                line = line.replace('@model', subckt_name)
            elif in_format_attr and '"' in line:
                # End of format attribute
                in_format_attr = False
            new_lines.append(line)
        result = '\n'.join(new_lines)
        
        return result
    
    def _generate_symbol_content(self, subckt: SubcktDef, prim_type: PrimitiveType) -> str:
        """Generate the content of a .sym file.
        
        Args:
            subckt: The SubcktDef
            prim_type: The primitive type
            
        Returns:
            String content of the .sym file
        """
        lines = []
        
        # Header - use IR subcircuit name
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        lines.append("v {xschem version=3.0.0 file_version=1.2}")
        # Use IR name in the header so generated symbols are uniquely named and traceable.
        lines.append(f"G {subckt_name}")
        
        # Get pin positions for this primitive type
        pin_positions = self.PIN_POSITIONS.get(prim_type, {})
        
        # Build pin list for template (standard pins first, then extra)
        standard_order = ['d', 'g', 's', 'b'] if prim_type == PrimitiveType.MOSFET else []
        pin_list_parts = []
        port_dict = {p.name.lower(): p for p in subckt.ports}
        
        # Add standard pins in order
        for std_name in standard_order:
            if std_name in port_dict:
                pin_list_parts.append(port_dict[std_name].name)
        
        # Add extra pins (maintain original order)
        for port in subckt.ports:
            if port.name.lower() not in standard_order:
                pin_list_parts.append(port.name)
        
        pin_list = " ".join(pin_list_parts) if pin_list_parts else " ".join([p.name for p in subckt.ports])
        
        # Get subcircuit name early (needed for format strings)
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        
        # Generate template
        template = self._generate_template(subckt, prim_type)
        
        # Build format string.
        # Sort params: wr and lr first (for MOSFETs), then others.
        sorted_params = self._sort_params_for_mosfet(subckt.params, prim_type)
        # Dialect-agnostic policy:
        # Include ALL header params that have constant numeric defaults (no expressions).
        constant_params = [
            p
            for p in sorted_params
            if self._is_constant_numeric_default_str(getattr(p.default, "val", p.default))
        ]
        
        # Generated symbols instantiate IR subcircuits, so use X prefix directly (tests expect X@name).
        # Keep `format` single-line (native xschem symbols do this); multi-line quoted strings here
        # can cause Xschem to skip parsing the rest of the K-block (leading to "fresh device has 0 params").
        format_str = f"X@name @pinlist {subckt_name}"
        fmt_params = " ".join([f"{p.name.name}=@{p.name.name}" for p in constant_params])
        if fmt_params:
            format_str += " " + fmt_params
        
        # Build template string - use the _generate_template method
        template_str = self._generate_template(subckt, prim_type)
        
        # Build K block (key attributes) - this is critical for Xschem!
        # NOTE: Keep K-block type as the generic primitive category (tests expect e.g. "mosfet"),
        # but use a separate MOSFET flavor ("nmos"/"pmos") for pin swapping + drawing.
        symbol_type = prim_type.value
        mosfet_flavor = None
        if prim_type == PrimitiveType.MOSFET:
            subckt_name_lower = subckt_name.lower()
            mosfet_flavor = "pmos" if (
                subckt_name_lower.startswith("p")
                or "pmos" in subckt_name_lower
                or "pfet" in subckt_name_lower
            ) else "nmos"
        
        # Build lvs_format (LVS-oriented / minimal).
        # Keep it small (typical libraries only include key geometry),
        # while `format=` keeps the full parameter list for netlisting.
        def _lvs_params(pt: PrimitiveType, params: List[ParamDecl]) -> List[ParamDecl]:
            idx = {p.name.name.lower(): p for p in params}
            if pt == PrimitiveType.MOSFET:
                want = ["wr", "lr", "nr", "w", "l"]
            elif pt == PrimitiveType.RESISTOR:
                want = ["wr", "lr", "mf", "r"]
            elif pt == PrimitiveType.CAPACITOR:
                want = ["w", "l", "mf", "c"]
            elif pt == PrimitiveType.INDUCTOR:
                want = ["w", "rad", "nr", "l"]
            elif pt == PrimitiveType.DIODE:
                want = ["area", "perim", "pj", "aw", "al"]
            else:
                want = []
            chosen = [idx[k] for k in want if k in idx]
            # Enforce constant-only policy.
            return [
                p
                for p in chosen
                if self._is_constant_numeric_default_str(getattr(p.default, "val", p.default))
            ]

        lvs_format_str = f"X@name @pinlist {subckt_name}"
        lvs_params = " ".join([f"{p.name.name}=@{p.name.name}" for p in _lvs_params(prim_type, constant_params)])
        if lvs_params:
            lvs_format_str += " " + lvs_params
        
        # Build drc attribute (for MOSFETs, match Sky130 format)
        drc_str = ""
        if prim_type == PrimitiveType.MOSFET and constant_params:
            # For MOSFETs, use fet_drc with name, symname, model, and key params
            # Use wr and lr from sorted params (they should be first)
            param1 = constant_params[0].name.name if constant_params else "wr"
            param2 = constant_params[1].name.name if len(constant_params) > 1 else "lr"
            # Escape braces properly: need \\{ for { in output, and @param needs proper escaping
            # Format: fet_drc {@name\} {@symname\} {@model\} {@lr\} {@wr\}
            drc_str = 'fet_drc \\{{@name\\\\\\}} \\{{@symname\\\\\\}} \\{{@model\\\\\\}} \\{{@{}\\\\\\}} \\{{@{}\\\\\\}}'.format(
                param1, param2
            )
        
        # Ensure format_str and lvs_format_str don't have @model (safety check)
        # subckt_name is already defined above
        format_str = format_str.replace('@model', subckt_name)
        lvs_format_str = lvs_format_str.replace('@model', subckt_name)
        if drc_str:
            drc_str = drc_str.replace('@model', subckt_name)
        
        k_block_parts = [
            "type={}".format(symbol_type),
            "spice_primitive=true",
            f"model={subckt_name}",
            'lvs_format="{}"'.format(lvs_format_str),
            'format="{}"'.format(format_str),
            'template="{}"'.format(template_str)
        ]
        # Add source context (and legacy comment) as escaped, quoted K-block attributes.
        # IMPORTANT: do NOT post-process the K block with a naive regex: many PARAMS defaults
        # contain `}` which would corrupt the block.
        ctx = str(getattr(subckt, "_subckt_context", "") or "").strip()
        if not ctx:
            ctx = f"{getattr(subckt, '_comments_above', '')}\n{getattr(subckt, '_subckt_line', '')}".strip()
        if ctx:
            # Escape for K-block quoted strings. Braces matter to xschem parsing even when
            # nested in larger `{ ... }` attribute blocks, so escape them too.
            ctx_escaped = (
                ctx.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("\n", "\\n")
            )
            k_block_parts.append(f'comment="{ctx_escaped}"')
            k_block_parts.append(f'source_context="{ctx_escaped}"')
        if drc_str:
            k_block_parts.append('drc="{}"'.format(drc_str))
        k_block = "K {{{}}}".format("\n".join(k_block_parts))
        # Final safety check: replace any remaining @model in the K block
        k_block = k_block.replace('@model', subckt_name)
        lines.append(k_block)
        
        # Add V, S, E blocks (empty but required)
        lines.append("V {}")
        lines.append("S {}")
        lines.append("E {}")
        
        # Generate pins using B blocks (not P lines) - this is the proper format
        # First, map standard pins (d, g, s, b)
        standard_pins = {}
        extra_pins = []
        
        # For resistors, handle ports separately (they're not in standard_pins)
        if prim_type == PrimitiveType.RESISTOR:
            pin_number = 1
            for port_idx, port in enumerate(subckt.ports):
                port_name = port.name.lower()
                # Map by order: first to top (y=27.5-32.5), second to bottom (y=-32.5 to -27.5)
                if port_idx == 0:
                    x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                elif port_idx == 1:
                    x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                else:
                    continue
                # B block for resistor pin
                lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                    x1, y1, x2, y2, port.name
                ))
                pin_number += 1
        elif prim_type == PrimitiveType.CAPACITOR:
            # Capacitor pins: 2-pin (top/bottom) or 3-pin (top/bottom/gnode)
            pin_number = 1
            port_dict = {p.name.lower(): p for p in subckt.ports}
            port_count = len(subckt.ports)
            
            if port_count == 2:
                # 2-pin capacitor: top and bottom
                for port_idx, port in enumerate(subckt.ports):
                    if port_idx == 0:
                        # Top pin
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    elif port_idx == 1:
                        # Bottom pin
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
            elif port_count == 3:
                # 3-pin capacitor: top, bottom, and gnode (on right side)
                for port_idx, port in enumerate(subckt.ports):
                    port_name = port.name.lower()
                    if port_name in ['top', 'gate', 'c0'] or port_idx == 0:
                        # Top pin
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    elif port_name in ['bottom', 'bulk', 'c1'] or port_idx == 1:
                        # Bottom pin
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    else:
                        # Third pin (gnode) on right side
                        x1, y1, x2, y2 = 30, -2.5, 35, 2.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
        elif prim_type == PrimitiveType.DIODE:
            # Diode pins: top (d1/cathode) and bottom (d0/anode) - matching Sky130 diode.sym
            # Note: Sky130 uses d0 at bottom (y=27.5-32.5) and d1 at top (y=-32.5 to -27.5)
            pin_number = 1
            port_dict = {p.name.lower(): p for p in subckt.ports}
            port_count = len(subckt.ports)
            
            if port_count == 2:
                for port_idx, port in enumerate(subckt.ports):
                    port_name = port.name.lower()
                    if port_name in ['cathode', 'c', 'd1'] or port_idx == 1:
                        # Cathode/top pin (top of symbol, y=-32.5 to -27.5)
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # Anode/bottom pin (bottom of symbol, y=27.5-32.5)
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
            elif port_count == 3:
                # 3-pin diode: anode, cathode, plus a gnode/sub terminal.
                # Keep anode/cathode on the vertical pins; put gnode/sub on the right side.
                gnode_idx = None
                for idx, p in enumerate(subckt.ports):
                    pname = p.name.lower()
                    if pname in ['gnode', 'gnd', 'sub', 'substrate', 'bulk', 'body', 'b']:
                        gnode_idx = idx
                        break
                # If we can't identify, assume last pin is gnode
                if gnode_idx is None:
                    gnode_idx = 2
                for port_idx, port in enumerate(subckt.ports):
                    port_name = port.name.lower()
                    if port_idx == gnode_idx:
                        # gnode/sub pin on right
                        x1, y1, x2, y2 = 30, -2.5, 35, 2.5
                    elif port_name in ['cathode', 'c', 'd1'] or port_idx == 1:
                        # Cathode/top pin
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # Anode/bottom pin
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
        elif prim_type == PrimitiveType.INDUCTOR:
            # Inductor pins: match Xschem's native `ind.sym` (top/bottom terminals)
            pin_number = 1
            port_dict = {p.name.lower(): p for p in subckt.ports}
            port_count = len(subckt.ports)
            
            if port_count == 2:
                # 2-pin inductor: top and bottom
                for port_idx, port in enumerate(subckt.ports):
                    if port_idx == 0:
                        # Top pin
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # Bottom pin
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
            elif port_count == 3:
                # 3-pin inductor: top, bottom, and gnode/tap (right side)
                # Identify gnode/tap by name; otherwise assume 3rd port is gnode/tap.
                port_names = [p.name.lower() for p in subckt.ports]
                gnode_idx = None
                for idx, pname in enumerate(port_names):
                    if pname in ['gnode', 'tap']:
                        gnode_idx = idx
                        break
                # If no gnode found by name, assume third port is gnode
                if gnode_idx is None:
                    gnode_idx = 2
                
                for port_idx, port in enumerate(subckt.ports):
                    port_name = port.name.lower()
                    if port_idx == gnode_idx:
                        # Third pin (gnode/tap) on right
                        x1, y1, x2, y2 = 27.5, -2.5, 32.5, 2.5
                    elif port_idx == 0:
                        # Top pin
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # Bottom pin
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
            elif port_count == 4:
                # 4-pin spiral inductors often have: top, bottom, gnode, tap
                # Keep top/bottom on the main pins; place gnode on right and tap on left.
                port_names = [p.name.lower() for p in subckt.ports]
                gnode_idx = None
                tap_idx = None
                for idx, pname in enumerate(port_names):
                    if pname == 'gnode':
                        gnode_idx = idx
                    elif pname == 'tap':
                        tap_idx = idx
                # Fallbacks: assume last two pins are extras if we can't identify by name
                if gnode_idx is None:
                    gnode_idx = 2
                if tap_idx is None:
                    tap_idx = 3 if gnode_idx != 3 else 2

                for port_idx, port in enumerate(subckt.ports):
                    if port_idx == gnode_idx:
                        # gnode on right
                        x1, y1, x2, y2 = 27.5, -2.5, 32.5, 2.5
                    elif port_idx == tap_idx:
                        # tap on left
                        x1, y1, x2, y2 = -32.5, -2.5, -27.5, 2.5
                    elif port_idx == 0:
                        # top
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # bottom
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                        x1, y1, x2, y2, port.name
                    ))
        else:
            # For MOSFETs and other types, use standard pin mapping
            # (including common aliases like n1..n4 and di/gi/si/bi for MOSFETs)
            for idx, port in enumerate(subckt.ports):
                port_name = port.name.lower()
                std = self._map_port_to_pin(port_name, prim_type, idx)
                if prim_type == PrimitiveType.MOSFET and std in ['d', 'g', 's', 'b']:
                    standard_pins[std] = port
                elif port_name in ['d', 'g', 's', 'b'] and prim_type != PrimitiveType.MOSFET:
                    standard_pins[port_name] = port
                else:
                    # Extra pin (ng, pg, etc.) - will be placed off to the side
                    extra_pins.append((port_name, port))
            
            # Place standard pins first using B blocks
            # B format: B width x1 y1 x2 y2 {name=... dir=...}
            # For MOSFET, match Sky130 pin positions exactly
            pin_number = 1
            # Determine if PMOS to swap S/D positions
            is_pmos = (prim_type == PrimitiveType.MOSFET and mosfet_flavor == "pmos")
            for port_name in ['d', 'g', 's', 'b']:
                if port_name in standard_pins:
                    # Use Sky130-style pin positions
                    if prim_type == PrimitiveType.MOSFET:
                        if port_name == 'd':
                            if is_pmos:
                                # PMOS: Drain is at bottom (S and D are swapped per Sky130 pfet_01v8.sym)
                                x1, y1, x2, y2 = 17.5, 27.5, 22.5, 32.5
                            else:
                                # NMOS: Drain is at top
                                x1, y1, x2, y2 = 17.5, -32.5, 22.5, -27.5
                            pin_name = "D"  # Uppercase to match Sky130
                        elif port_name == 'g':
                            # Gate: left side, center (matches Sky130 for both NMOS and PMOS)
                            x1, y1, x2, y2 = -22.5, -2.5, -17.5, 2.5
                            pin_name = "G"  # Uppercase to match Sky130
                        elif port_name == 's':
                            if is_pmos:
                                # PMOS: Source is at top (S and D are swapped per Sky130 pfet_01v8.sym)
                                x1, y1, x2, y2 = 17.5, -32.5, 22.5, -27.5
                            else:
                                # NMOS: Source is at bottom
                                x1, y1, x2, y2 = 17.5, 27.5, 22.5, 32.5
                            pin_name = "S"  # Uppercase to match Sky130
                        else:  # bulk
                            # Bulk: matches Sky130 position (center right, small)
                            x1, y1, x2, y2 = 19.921875, -0.078125, 20.078125, 0.078125
                            pin_name = "B"  # Uppercase to match Sky130
                elif prim_type == PrimitiveType.RESISTOR:
                    # Resistor pins: must align with resistor drawing endpoints
                    # In Xschem, y increases downward, so:
                    # - y=-30 is at the top, y=30 is at the bottom
                    # - Resistor drawing: top at y=30 (line goes 0 20 to 0 30), bottom at y=-30 (line goes 0 -30 to 0 -20)
                    # - Pins should be at: top pin at y=27.5-32.5 (connects to y=30), bottom pin at y=-32.5 to -27.5 (connects to y=-30)
                    # - Match Sky130: M (top) at y=27.5-32.5, P (bottom) at y=-32.5 to -27.5
                    # - Ports can be n1/n2, p/n, or other names - map by port order
                    port_idx = [p[0] for p in resistor_port_order].index(port_name) if resistor_port_order else -1
                    if port_idx == 0:
                        # First port goes to top (y=27.5-32.5)
                        x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                    elif port_idx == 1:
                        # Second port goes to bottom (y=-32.5 to -27.5)
                        x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                    else:
                        # Fallback: try to match by name
                        if port_name == 'p' or port_name == 'm' or port_name == 'n1':
                            x1, y1, x2, y2 = -2.5, 27.5, 2.5, 32.5
                        elif port_name == 'n' or port_name == 'n2':
                            x1, y1, x2, y2 = -2.5, -32.5, 2.5, -27.5
                        else:
                            continue
                else:
                    # For other types, use pin_positions
                    if port_name in pin_positions:
                        x, y = pin_positions[port_name]
                        x1, y1, x2, y2 = x-2.5, y-2.5, x+2.5, y+2.5
                    else:
                        continue
                
                # B block for pin: B width x1 y1 x2 y2 {name=... dir=...}
                # For MOSFET: d/s are inout, g is in, b is in (matching Sky130)
                if port_name == 'b':
                    dir_type = "in"  # Bulk is input in Sky130
                elif port_name in ['d', 's']:
                    dir_type = "inout"
                else:  # gate
                    dir_type = "in"
                # Use uppercase pin name for MOSFET to match Sky130 format
                if prim_type == PrimitiveType.MOSFET:
                    pin_display_name = pin_name
                else:
                    pin_display_name = port_dict[port_name].name
                # Match Sky130 format: no pinnumber attribute
                lines.append("B 5 {} {} {} {} {{name={} dir={}}}".format(
                    x1, y1, x2, y2, pin_display_name, dir_type
                ))
                pin_number += 1
        
        # Place extra pins on the LEFT side (for 5T/6T FETs)
        # Position them vertically spaced, avoiding overlap with the gate/name text area.
        if extra_pins:
            extra_pin_x = -35  # Left side, far from gate
            # Move extra terminals higher so they don't sit on top of the gate area.
            extra_pin_y_start = -35  # Start higher (more negative y is up)
            extra_pin_spacing = 15
            for idx, (port_name, port) in enumerate(extra_pins):
                # Place on left side, vertically spaced
                y = extra_pin_y_start + (idx * extra_pin_spacing)
                x1, y1, x2, y2 = extra_pin_x - 2.5, y - 2.5, extra_pin_x + 2.5, y + 2.5
                # B block for extra pin
                lines.append("B 5 {} {} {} {} {{name={} dir=inout}}".format(
                    x1, y1, x2, y2, port.name
                ))
                # Add text label for extra pin (on left side, easy to see)
                # Position label to the left of the pin, avoiding overlap
                label_x = extra_pin_x - 8
                label_y = y
                lines.append("T {{{}}} {} {} 0 0 0.15 0.15 {{layer=7}}".format(
                    port.name, label_x, label_y
                ))
                pin_number += 1
        
        # Add drawing elements based on primitive type
        # Use L format: L width x1 y1 x2 y2 {}
        # Pass mosfet_flavor for MOSFETs to get correct NMOS/PMOS drawing
        mosfet_type = mosfet_flavor if prim_type == PrimitiveType.MOSFET else None
        drawing_lines = self._get_drawing_elements(prim_type, mosfet_type)
        for line_cmd in drawing_lines:
            if line_cmd.startswith("L "):
                # Convert to proper L format: L width x1 y1 x2 y2 {}
                parts = line_cmd.split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = parts[1], parts[2], parts[3], parts[4]
                    lines.append("L 4 {} {} {} {} {{}}".format(x1, y1, x2, y2))
            elif line_cmd.startswith("P "):
                # Polygon format: P width x1 y1 x2 y2 x3 y3 x4 y4 {fill=true}
                # Keep as-is (already in correct format)
                lines.append(line_cmd)
            elif line_cmd.startswith("A "):
                # Arc format: A x y rx ry start end {}
                parts = line_cmd.split()
                if len(parts) >= 7:
                    x, y, rx, ry, start, end = parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
                    lines.append("A {} {} {} {} {} {} {{}}".format(x, y, rx, ry, start, end))
            elif line_cmd.startswith("R "):
                # Rectangle format: R x1 y1 x2 y2
                parts = line_cmd.split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = parts[1], parts[2], parts[3], parts[4]
                    lines.append("R {} {} {} {}".format(x1, y1, x2, y2))
        
        # Add text labels (device name and model) - matching Sky130 style
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        if prim_type == PrimitiveType.MOSFET:
            # MOSFET labels matching Sky130 nfet_01v8.sym (single braces, not double)
            # Position model text to avoid overlap with extra pin labels on left side
            # If we have extra pins, move model text further right
            model_x = 30 if not extra_pins else 40
            lines.append("T {@name} 5 -30 0 1 0.2 0.2 {}")
            # IMPORTANT: Do not use `.format()` on strings containing Xschem `{...}` attributes
            # like `{@model}`; Python will treat it as a format field and raise KeyError.
            # Place model name at top-right of symbol area.
            lines.append(f"T {{@model}} {model_x} -30 2 1 0.2 0.2 {{}}")
            # Pin labels: for PMOS, S/D are swapped relative to NMOS
            if mosfet_flavor == "pmos":
                lines.append("T {S} 22.5 -17.5 2 1 0.15 0.15 {layer=7}")
                lines.append("T {D} 22.5 17.5 0 0 0.15 0.15 {layer=7}")
            else:
                lines.append("T {S} 22.5 17.5 0 0 0.15 0.15 {layer=7}")
                lines.append("T {D} 22.5 -17.5 2 1 0.15 0.15 {layer=7}")
            lines.append("T {G} -10 -10 0 1 0.15 0.15 {layer=7}")
            # Bulk pin label (if present)
            port_dict = {p.name.lower(): p for p in subckt.ports}
            if 'b' in port_dict:
                lines.append("T {B} 20 -10 0 0 0.15 0.15 {layer=7}")
            # Parameter display: show key sizing + fingers/rows if available
            has_wr = any(p.name.name.lower() == 'wr' for p in subckt.params)
            has_lr = any(p.name.name.lower() == 'lr' for p in subckt.params)
            has_nr = any(p.name.name.lower() == 'nr' for p in subckt.params)
            if has_wr and has_lr:
                # RF MOSFET: show wr/lr
                # Keep display simple and robust across Xschem expression handling.
                # (Using arithmetic like `*1e6` caused some setups to render only `/`.)
                lines.append("T {@wr / @lr} 31.25 10 0 0 0.2 0.2 { layer=13}")
            else:
                # Standard MOSFET: show W/L if available
                has_w = any(p.name.name.lower() == 'w' for p in subckt.params)
                has_l = any(p.name.name.lower() == 'l' for p in subckt.params)
                if has_w and has_l:
                    # Use lowercase @w/@l since many subckts (e.g. *mac) use lowercase params.
                    lines.append("T {@w / @l} 31.25 10 0 0 0.2 0.2 { layer=13}")
                else:
                    # Fallback for symbols that use uppercase W/L
                    has_W = any(p.name.name == 'W' for p in subckt.params)
                    has_L = any(p.name.name == 'L' for p in subckt.params)
                    if has_W and has_L:
                        lines.append("T {@W / @L} 31.25 10 0 0 0.2 0.2 { layer=13}")

            # Show nr (number of rows / fingers) if present
            if has_nr:
                # Push down to avoid overlapping the W/L label.
                lines.append("T {nr=@nr} 31.25 22.5 0 0 0.2 0.2 { layer=13}")
            # Also show nf if present (many bsim devices use nf as fingers)
            has_nf = any(p.name.name.lower() == 'nf' for p in subckt.params)
            if has_nf:
                lines.append("T {nf=@nf} 31.25 32.5 0 0 0.2 0.2 { layer=13}")
        elif prim_type == PrimitiveType.RESISTOR:
            # Resistor labels matching Sky130
            lines.append("T {@name} 15 -27.5 0 0 0.2 0.2 {}")
            # Place model name at top-right of symbol area.
            lines.append("T {@model} 35 -27.5 0 0 0.2 0.2 {}")
            # Pin labels: show actual port names (can be long) near pins
            if len(subckt.ports) >= 1:
                lines.append(f"T {{{subckt.ports[0].name}}} -15 -27.5 0 1 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 2:
                lines.append(f"T {{{subckt.ports[1].name}}} -15 27.5 0 0 0.15 0.15 {{layer=7}}")
            # Parameter display: show resistance value if available
            has_r = any(p.name.name.lower() == 'r' for p in subckt.params)
            has_m = any(p.name.name.lower() == 'm' for p in subckt.params)
            if has_r:
                if has_m:
                    # Show R value with multiplier
                    lines.append("T {@m * @r} 15 13.75 0 0 0.2 0.2 {layer=13}")
                else:
                    # Show R value
                    lines.append("T {@r} 15 13.75 0 0 0.2 0.2 {layer=13}")
            # Resistance calculation display (if we have L and W parameters)
            has_l = any(p.name.name.lower() == 'l' for p in subckt.params)
            has_w = any(p.name.name.lower() == 'w' for p in subckt.params)
            if has_l and has_w:
                # Show L/W ratio
                lines.append("T {@l / @w} 15 0 0 0 0.2 0.2 {layer=13}")
        elif prim_type == PrimitiveType.CAPACITOR:
            # Show instance name + model on symbol (like other primitives)
            lines.append("T {@name} 15 -27.5 0 0 0.2 0.2 {}")
            lines.append("T {@model} 35 -27.5 0 0 0.2 0.2 {}")
            # Capacitor: show actual port names near pins (2-pin or 3-pin)
            if len(subckt.ports) >= 1:
                lines.append(f"T {{{subckt.ports[0].name}}} -15 -27.5 0 1 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 2:
                lines.append(f"T {{{subckt.ports[1].name}}} -15 27.5 0 0 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 3:
                # 3rd pin on right side
                lines.append(f"T {{{subckt.ports[2].name}}} 35 0 0 0 0.15 0.15 {{layer=7}}")
            # Parameter display: show W/L if present (or wr/lr for rf caps)
            has_wr = any(p.name.name.lower() == 'wr' for p in subckt.params)
            has_lr = any(p.name.name.lower() == 'lr' for p in subckt.params)
            if has_wr and has_lr:
                lines.append("T {@wr / @lr} 15 13.75 0 0 0.2 0.2 {layer=13}")
            else:
                has_w = any(p.name.name.lower() == 'w' for p in subckt.params)
                has_l = any(p.name.name.lower() == 'l' for p in subckt.params)
                if has_w and has_l:
                    lines.append("T {@w / @l} 15 13.75 0 0 0.2 0.2 {layer=13}")
        elif prim_type == PrimitiveType.DIODE:
            # Show instance name + model on symbol
            lines.append("T {@name} 15 -27.5 0 0 0.2 0.2 {}")
            lines.append("T {@model} 35 -27.5 0 0 0.2 0.2 {}")
            # Diode: show actual port names near pins (2-pin or 3-pin)
            if len(subckt.ports) >= 1:
                # Top pin label (cathode-ish)
                lines.append(f"T {{{subckt.ports[0].name}}} -15 -27.5 0 1 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 2:
                # Bottom pin label (anode-ish)
                lines.append(f"T {{{subckt.ports[1].name}}} -15 27.5 0 0 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 3:
                # 3rd pin on right side
                lines.append(f"T {{{subckt.ports[2].name}}} 35 0 0 0 0.15 0.15 {{layer=7}}")
        elif prim_type == PrimitiveType.INDUCTOR:
            # Show instance name + model on symbol
            lines.append("T {@name} 15 -27.5 0 0 0.2 0.2 {}")
            lines.append("T {@model} 35 -27.5 0 0 0.2 0.2 {}")
            # Inductor: show actual port names near pins (2/3/4 pin)
            if len(subckt.ports) >= 1:
                lines.append(f"T {{{subckt.ports[0].name}}} -10 -27.5 0 1 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 2:
                lines.append(f"T {{{subckt.ports[1].name}}} -10 27.5 0 0 0.15 0.15 {{layer=7}}")
            # For 3/4-pin inductors, extra pins are on right/left
            if len(subckt.ports) >= 3:
                lines.append(f"T {{{subckt.ports[2].name}}} 35 0 0 0 0.15 0.15 {{layer=7}}")
            if len(subckt.ports) >= 4:
                lines.append(f"T {{{subckt.ports[3].name}}} -40 0 0 1 0.15 0.15 {{layer=7}}")
            # Parameter display: show key geometry (w/rad/nr if present)
            has_w = any(p.name.name.lower() == 'w' for p in subckt.params)
            has_rad = any(p.name.name.lower() == 'rad' for p in subckt.params)
            has_nr = any(p.name.name.lower() == 'nr' for p in subckt.params)
            if has_w and has_rad:
                lines.append("T {w=@w rad=@rad} 15 13.75 0 0 0.2 0.2 {layer=13}")
            elif has_w:
                lines.append("T {w=@w} 15 13.75 0 0 0.2 0.2 {layer=13}")
            if has_nr:
                lines.append("T {nr=@nr} 15 20 0 0 0.2 0.2 {layer=13}")
        else:
            # Default labels
            lines.append("T {@name} 5 -30 0 1 0.2 0.2 {}")
            lines.append("T {@model} 30 -17.5 2 1 0.2 0.2 {}")
        
        result = "\n".join(lines) + "\n"
        # Final safety check: replace any remaining @model in format/lvs_format/drc/template attributes
        # (but NOT in text display strings T blocks)
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        import re
        # Replace @model in K block attributes only (handle multi-line format strings)
        # Match format="...@model..." or format="...\n+...@model..."
        result = re.sub(r'(format="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.DOTALL)
        result = re.sub(r'(lvs_format="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.DOTALL)
        result = re.sub(r'(drc="[^"]*)@model([^"]*")', r'\1' + subckt_name + r'\2', result, flags=re.DOTALL)
        result = re.sub(r'(model=)@model', r'\1' + subckt_name, result)
        # Also handle continuation lines (format strings can span multiple lines with \n+)
        lines_list = result.split('\n')
        new_lines = []
        in_format_attr = False
        for line in lines_list:
            if 'format="' in line or 'lvs_format="' in line or 'drc="' in line:
                in_format_attr = True
                line = line.replace('@model', subckt_name)
            elif in_format_attr and (line.strip().startswith('+') or '"' not in line):
                # Continuation line of format string (starts with + or doesn't have closing quote)
                line = line.replace('@model', subckt_name)
                if '"' in line:
                    in_format_attr = False
            elif in_format_attr and '"' in line:
                # End of format attribute
                in_format_attr = False
                line = line.replace('@model', subckt_name)
            elif 'model=@model' in line:
                line = line.replace('model=@model', f'model={subckt_name}')
            new_lines.append(line)
        result = '\n'.join(new_lines)
        return result
    
    def _map_port_to_pin(self, port_name: str, prim_type: PrimitiveType, pin_idx: int) -> str:
        """Map a port name to a standard pin name.
        
        Args:
            port_name: The port name from the subcircuit
            prim_type: The primitive type
            pin_idx: The index of the port
            
        Returns:
            Standard pin name
        """
        port_lower = port_name.lower()
        
        # Try to match common port names
        if prim_type == PrimitiveType.MOSFET:
            if port_lower in ['d', 'drain']:
                return 'd'
            elif port_lower in ['g', 'gate']:
                return 'g'
            elif port_lower in ['s', 'source']:
                return 's'
            elif port_lower in ['b', 'bulk', 'substrate', 'sub']:
                return 'b'
            # Some PDK subckts (e.g. *mac) use n1..n4 for D/G/S/B
            elif port_lower in ['n1', 'di']:
                return 'd'
            elif port_lower in ['n2', 'gi']:
                return 'g'
            elif port_lower in ['n3', 'si']:
                return 's'
            elif port_lower in ['n4', 'bi']:
                return 'b'
        elif prim_type == PrimitiveType.BJT:
            if port_lower in ['c', 'collector']:
                return 'c'
            elif port_lower in ['b', 'base']:
                return 'b'
            elif port_lower in ['e', 'emitter']:
                return 'e'
            elif port_lower in ['s', 'substrate', 'sub']:
                return 's'
        elif prim_type in [PrimitiveType.DIODE, PrimitiveType.RESISTOR, 
                          PrimitiveType.CAPACITOR, PrimitiveType.INDUCTOR]:
            if port_lower in ['p', 'plus', 'pos', 'a', 'anode']:
                return 'p'
            elif port_lower in ['n', 'minus', 'neg', 'c', 'cathode']:
                return 'n'
        
        # Fallback: use index-based mapping
        if prim_type == PrimitiveType.MOSFET:
            pins = ['d', 'g', 's', 'b']
        elif prim_type == PrimitiveType.BJT:
            pins = ['c', 'b', 'e', 's']
        else:
            pins = ['p', 'n']
        
        if pin_idx < len(pins):
            return pins[pin_idx]
        return port_name
    
    def _sort_params_for_mosfet(self, params, prim_type: PrimitiveType):
        """Sort parameters so wr and lr come first for MOSFETs.
        
        Args:
            params: List of ParamDecl objects
            prim_type: The primitive type
            
        Returns:
            Sorted list of ParamDecl objects
        """
        if prim_type != PrimitiveType.MOSFET:
            return params
        
        # For MOSFETs, prioritize wr and lr first
        priority_params = ['wr', 'lr']
        priority_list = []
        other_list = []
        
        for param in params:
            param_name = param.name.name.lower()
            if param_name in priority_params:
                priority_list.append(param)
            else:
                other_list.append(param)
        
        # Sort priority list: wr first, then lr
        priority_list.sort(key=lambda p: priority_params.index(p.name.name.lower()) if p.name.name.lower() in priority_params else 999)
        
        return priority_list + other_list
    
    def _generate_template(self, subckt: SubcktDef, prim_type: PrimitiveType) -> str:
        """Generate the template string for spice netlist generation.
        Uses Sky130-style format with actual default values.
        
        Args:
            subckt: The SubcktDef
            prim_type: The primitive type
            
        Returns:
            Template string in Sky130 format
        """
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        
        # Sort params: wr and lr first (for MOSFETs)
        sorted_params = self._sort_params_for_mosfet(subckt.params, prim_type)
        
        # Build template lines - Sky130 format shows actual default values
        # Use default name like Sky130 (M1 for MOSFETs, R1 for resistors, etc.)
        # Xschem will use this as default but allow override
        if prim_type == PrimitiveType.MOSFET:
            default_name = "M1"
        elif prim_type == PrimitiveType.RESISTOR:
            default_name = "R1"
        elif prim_type == PrimitiveType.CAPACITOR:
            default_name = "C1"
        elif prim_type == PrimitiveType.DIODE:
            default_name = "D1"
        elif prim_type == PrimitiveType.INDUCTOR:
            default_name = "L1"
        elif prim_type == PrimitiveType.BJT:
            default_name = "Q1"
        else:
            default_name = "X1"
        template_lines = [f'name={default_name}']
        
        # Add parameters with their default values:
        # - Only a small user-facing allowlist per primitive type
        # - Only constant numeric defaults (no expressions/references)
        # NOTE: We intentionally do NOT apply the per-primitive allowlist here.
        # Policy: include ALL header params with constant numeric defaults.
        for param in sorted_params:
            default_raw = getattr(param.default, "val", param.default)
            if not self._is_constant_numeric_default_str(default_raw):
                continue
            param_name = param.name.name
            default_val = self._format_param_default(param)
            template_lines.append("{}={}".format(param_name, default_val))
        
        # Use actual subcircuit name as model (the subcircuit name IS the model name)
        template_lines.append(f"model={subckt_name}")
        template_lines.append("spiceprefix=X")
        
        # Join with newlines (Sky130 format)
        return "\n".join(template_lines)
    
    def _get_drawing_elements(self, prim_type: PrimitiveType, mosfet_type: str = None) -> List[str]:
        """Get drawing elements for a primitive type.
        
        Args:
            prim_type: The primitive type
            mosfet_type: For MOSFETs, either "nmos" or "pmos" to get correct drawing style
            
        Returns:
            List of drawing command strings (L, A, R, etc.)
        """
        # Standard Xschem symbol drawings - matching Sky130 exactly
        if prim_type == PrimitiveType.MOSFET:
            if mosfet_type == "pmos":
                # PMOS-style drawing - EXACT copy from pfet_01v8.sym
                # Key differences: gate on right, circle at gate, arrow pointing left, S/D swapped
                return [
                    # Channel (vertical line, right side)
                    "L 7.5 -22.5 7.5 22.5",
                    # Drain connection (right, top) - note: D is at bottom for PMOS
                    "L 20 -30 20 -17.5",
                    # Source connection (right, bottom) - note: S is at top for PMOS
                    "L 20 17.5 20 30",
                    # Channel body (vertical line inside)
                    "L 2.5 -15 2.5 15",
                    # Source connection to channel (top)
                    "L 7.5 17.5 20 17.5",
                    # Drain connection to channel (bottom)
                    "L 12.5 -17.5 20 -17.5",
                    # Gate (horizontal line, right side) - PMOS has gate on right
                    "L -20 0 -7.5 0",
                    # Circle at gate (PMOS indicator)
                    "A 4 -2.5 0 5 180 360 {}",
                    # Arrow polygon (pointing left, for PMOS)
                    "P 4 4 12.5 -20 7.5 -17.5 12.5 -15 12.5 -20 {fill=true}",
                    # Bulk connection indicator polygon
                    "P 5 4 15 -2.5 20 0 15 2.5 15 -2.5 {fill=true}",
                ]
            else:
                # NMOS-style drawing - EXACT copy from nfet_01v8.sym
                return [
                    # Channel (vertical line, right side)
                    "L 7.5 -22.5 7.5 22.5",
                    # Gate (horizontal line, left side)
                    "L -20 0 2.5 0",
                    # Drain connection (right, top)
                    "L 20 -30 20 -17.5",
                    # Source connection (right, bottom)
                    "L 20 17.5 20 30",
                    # Channel body (vertical line inside)
                    "L 2.5 -15 2.5 15",
                    # Drain connection to channel
                    "L 7.5 -17.5 20 -17.5",
                    # Source connection to channel
                    "L 7.5 17.5 15 17.5",
                    # Arrow polygon (pointing right, for NMOS) - NO extra line, just polygon
                    "P 4 4 15 15 20 17.5 15 20 15 15 {fill=true}",
                    # Bulk connection indicator polygon (matching nfet_01v8.sym)
                    "P 5 4 20 -2.5 15 0 20 2.5 20 -2.5 {fill=true}",
                ]
        elif prim_type == PrimitiveType.BJT:
            # NPN-style drawing
            return [
                # Collector (left terminal)
                "L -200 -100 -300 -100",
                # Base (top terminal)
                "L 0 -200 0 -300",
                # Emitter (right terminal)
                "L 200 100 300 100",
                # Substrate (bottom terminal)
                "L 0 200 0 300",
                # Base line
                "L -200 -100 200 100",
                # Emitter arrow (pointing right)
                "L 100 50 200 100",
                "L 100 150 200 100",
            ]
        elif prim_type == PrimitiveType.RESISTOR:
            # Compact zigzag resistor matching Sky130 style
            return [
                # Top terminal
                "L 0 20 0 30",
                # Top connection
                "L 0 20 7.5 17.5",
                # Zigzag body (compact, matching Sky130)
                "L -7.5 12.5 7.5 17.5",
                "L -7.5 12.5 7.5 7.5",
                "L -7.5 2.5 7.5 7.5",
                "L -7.5 2.5 7.5 -2.5",
                "L -7.5 -7.5 7.5 -2.5",
                "L -7.5 -7.5 7.5 -12.5",
                "L -7.5 -17.5 7.5 -12.5",
                # Bottom connection
                "L -7.5 -17.5 0 -20",
                # Bottom terminal
                "L 0 -30 0 -20",
            ]
        elif prim_type == PrimitiveType.CAPACITOR:
            return [
                # Left terminal
                "L -200 0 -300 0",
                # Right terminal
                "L 200 0 300 0",
                # Top plate
                "L -50 -50 -50 50",
                # Bottom plate
                "L 50 -50 50 50",
            ]
        elif prim_type == PrimitiveType.DIODE:
            # Diode drawing matching Sky130 diode.sym
            return [
                # Top terminal (vertical line, top)
                "L 0 -30 0 -5",
                # Bottom terminal (vertical line, bottom)
                "L 0 5 0 30",
                # Horizontal line (separator)
                "L -10 -5 10 -5",
                # Diode triangle (filled polygon pointing right)
                "P 4 4 0 -5 -10 5 10 5 0 -5 {fill=true}",
            ]
        elif prim_type == PrimitiveType.INDUCTOR:
            # Match Xschem's native `devices/ind.sym` style (top/bottom terminals, 3 arcs)
            return [
                # Top / bottom leads
                "L 0 22.5 0 30",
                "L 0 -30 0 -22.5",
                # Small marker near top (as in ind.sym)
                "L 7.5 -26.25 7.5 -21.25",
                "L 5 -23.75 10 -23.75",
                # Coil arcs
                "A 4 0 15 7.5 90 180",
                "A 4 0 0 7.5 90 180",
                "A 4 0 -15 7.5 90 180",
            ]
        else:
            # Default: simple rectangle
            return ["R -100 -100 100 100"]
    
    def _format_param_default(self, param: ParamDecl) -> str:
        """Format a parameter default value for the symbol file.
        
        Args:
            param: The parameter declaration
            
        Returns:
            Formatted default value string
        """
        if param.default is None:
            return ""
        
        from ..data import Int, Float, MetricNum, Ref
        if isinstance(param.default, (Int, Float, MetricNum)):
            s = str(param.default.val) if hasattr(param.default, 'val') else str(param.default)
        elif isinstance(param.default, Ref):
            s = param.default.ident.name if hasattr(param.default, 'ident') else str(param.default)
        else:
            s = str(param.default)

        # Xschem uses `{...}` as structural syntax in .sym/.sch files. Many Xyce defaults
        # use `{expr}`. Escape braces so Xschem doesn't truncate the K-block/template,
        # but the resulting parsed value still contains literal braces.
        # (Tcl-style: `\{` -> `{` after parsing)
        s = s.replace("{", "\\{").replace("}", "\\}")
        return s


#
# Batch symbol generation (Xschem-specific)
#

"""
Parameter policy for symbol generation:

- **Strict** (current): only parameters explicitly declared on the `.SUBCKT ... PARAMS:` header
  are exposed on the generated symbol. This matches "params that can be passed at the subckt level"
  under the common SPICE convention.

If a future workflow wants to expose additional parameters referenced inside the subckt body,
it should do so in a separate, opt-in layer/tool.
"""


def extract_all_top_level_subckts(xyce_file: Path) -> List[str]:
    """Extract all top-level subcircuit names from an Xyce file."""
    subckt_names: List[str] = []
    lines = Path(xyce_file).read_text(errors="ignore").splitlines(True)

    referenced_subckts = set()
    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith(".SUBCKT"):
            parts = line_stripped.split()
            if len(parts) >= 2:
                subckt_names.append(parts[1])
        elif line_stripped.startswith("X") or line_stripped.startswith("x"):
            parts = line_stripped.split()
            if len(parts) >= 2:
                for part in parts[1:]:
                    if part and not part.startswith("{") and "=" not in part:
                        referenced_subckts.add(part)
                        break

    top_level = [name for name in subckt_names if name not in referenced_subckts]
    return top_level or subckt_names


def extract_subckt_from_xyce(xyce_file: Path, subckt_name: str) -> SubcktDef:
    """Extract a subcircuit definition from Xyce file, for Xschem symbol generation."""
    lines = Path(xyce_file).read_text(errors="ignore").splitlines(True)

    # Find the .SUBCKT line and extract comments above it + surrounding context (±5 lines)
    # Also capture continuation lines (leading '+') as part of the same header.
    subckt_line = None
    subckt_line_idx = None
    comments_above: List[str] = []
    context_lines: List[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith(f".SUBCKT {subckt_name}"):
            header_parts = [line.strip()]
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("+"):
                header_parts.append(lines[j].strip().lstrip("+").strip())
                j += 1
            subckt_line = " ".join(header_parts).strip()
            subckt_line_idx = i

            start_idx = max(0, i - 20)
            for jj in range(start_idx, i):
                comment_line = lines[jj].rstrip()
                if comment_line.strip() and not comment_line.strip().startswith(".SUBCKT"):
                    comments_above.append(comment_line)

            ctx_start = max(0, i - 5)
            ctx_end = min(len(lines), i + 6)
            context_lines = [ln.rstrip("\n") for ln in lines[ctx_start:ctx_end]]
            break

    if not subckt_line:
        raise ValueError(f"Subcircuit '{subckt_name}' not found in {xyce_file}")

    parts = subckt_line.split()
    if parts[0] != ".SUBCKT":
        raise ValueError(f"Invalid subcircuit line: {subckt_line}")

    name = parts[1]

    ports: List[str] = []
    for part in parts[2:]:
        if part == "PARAMS:":
            break
        ports.append(part)

    # STRICT PARAM POLICY: only include header params
    header_params: List[Tuple[str, str]] = []
    if "PARAMS:" in subckt_line:
        params_start = subckt_line.find("PARAMS:") + len("PARAMS:")
        params_str = subckt_line[params_start:].strip()
        for param_str in params_str.split():
            if "=" not in param_str:
                continue
            param_name, param_val = param_str.split("=", 1)
            header_params.append((param_name.strip(), param_val.strip()))

    param_decls: List[ParamDecl] = []
    for param_name, default_str in header_params:
        param_decls.append(ParamDecl(name=Ident(param_name), default=MetricNum(default_str)))

    subckt_def = SubcktDef(
        name=Ident(name),
        ports=[Ident(p) for p in ports],
        params=param_decls,
        entries=[],
    )

    # Attach context metadata for symbol-generation (comment/source_context)
    if subckt_line_idx is not None and subckt_line_idx >= 0:
        subckt_def._comments_above = "\n".join(comments_above)
        subckt_def._subckt_line = subckt_line
        subckt_def._subckt_context = "\n".join(context_lines).strip()
    else:
        subckt_def._comments_above = ""
        subckt_def._subckt_line = subckt_line
        subckt_def._subckt_context = "\n".join(context_lines).strip() if context_lines else ""

    return subckt_def


class BatchSymbolGenerator:
    """Generate Xschem symbols for all top-level subcircuits from an Xyce file."""

    def __init__(
        self,
        xyce_file: Path,
        output_dir: Path,
        template_dir: Optional[Path] = None,
        primitive_config_file: Optional[Path] = None,
    ):
        self.xyce_file = Path(xyce_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PrimitiveDetector(config_file=str(primitive_config_file) if primitive_config_file else None)
        template_base = template_dir or (Path(__file__).parent / "xschem_symbol_templates")
        self.matcher = SymbolMatcher(template_base)
        self.generator = XschemSymbolGenerator(self.output_dir, template_dir=template_base)

        self.mapping: Dict[str, Dict] = {}
        # Store extracted subckts and types for downstream tasks (e.g. schematic placement)
        self.subckts: Dict[str, SubcktDef] = {}
        self.prim_types: Dict[str, PrimitiveType] = {}

    def generate_all(self) -> Dict[str, Dict]:
        # Single-pass parse of the xyce file: read it once, then reuse `lines` throughout.
        lines = self.xyce_file.read_text(errors="ignore").splitlines(True)

        def _extract_all_top_level_subckts_from_lines(lines_: List[str]) -> List[str]:
            subckt_names_: List[str] = []
            referenced_subckts_ = set()
            for line in lines_:
                line_stripped = line.strip()
                if line_stripped.startswith(".SUBCKT"):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        subckt_names_.append(parts[1])
                elif line_stripped.startswith("X") or line_stripped.startswith("x"):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        for part in parts[1:]:
                            if part and not part.startswith("{") and "=" not in part:
                                referenced_subckts_.add(part)
                                break
            top_level_ = [n for n in subckt_names_ if n not in referenced_subckts_]
            return top_level_ or subckt_names_

        def _extract_subckt_from_lines(lines_: List[str], subckt_name_: str) -> SubcktDef:
            # Find .SUBCKT header and capture continuation '+'
            subckt_line_ = None
            subckt_line_idx_ = None
            comments_above_: List[str] = []
            context_lines_: List[str] = []
            for i, line in enumerate(lines_):
                if line.strip().startswith(f".SUBCKT {subckt_name_}"):
                    header_parts = [line.strip()]
                    j = i + 1
                    while j < len(lines_) and lines_[j].lstrip().startswith("+"):
                        header_parts.append(lines_[j].strip().lstrip("+").strip())
                        j += 1
                    subckt_line_ = " ".join(header_parts).strip()
                    subckt_line_idx_ = i

                    start_idx = max(0, i - 20)
                    for jj in range(start_idx, i):
                        comment_line = lines_[jj].rstrip()
                        if comment_line.strip() and not comment_line.strip().startswith(".SUBCKT"):
                            comments_above_.append(comment_line)

                    ctx_start = max(0, i - 5)
                    ctx_end = min(len(lines_), i + 6)
                    context_lines_ = [ln.rstrip("\n") for ln in lines_[ctx_start:ctx_end]]
                    break

            if not subckt_line_:
                raise ValueError(f"Subcircuit '{subckt_name_}' not found in {self.xyce_file}")

            parts = subckt_line_.split()
            if parts[0] != ".SUBCKT":
                raise ValueError(f"Invalid subcircuit line: {subckt_line_}")

            name = parts[1]
            ports: List[str] = []
            for part in parts[2:]:
                if part == "PARAMS:":
                    break
                ports.append(part)

            header_params: List[Tuple[str, str]] = []
            if "PARAMS:" in subckt_line_:
                params_start = subckt_line_.find("PARAMS:") + len("PARAMS:")
                params_str = subckt_line_[params_start:].strip()
                for param_str in params_str.split():
                    if "=" not in param_str:
                        continue
                    param_name, param_val = param_str.split("=", 1)
                    header_params.append((param_name.strip(), param_val.strip()))

            param_decls: List[ParamDecl] = []
            for param_name, default_str in header_params:
                param_decls.append(ParamDecl(name=Ident(param_name), default=MetricNum(default_str)))

            subckt_def = SubcktDef(
                name=Ident(name),
                ports=[Ident(p) for p in ports],
                params=param_decls,
                entries=[],
            )

            # attach context metadata
            if subckt_line_idx_ is not None and subckt_line_idx_ >= 0:
                subckt_def._comments_above = "\n".join(comments_above_)
                subckt_def._subckt_line = subckt_line_
                subckt_def._subckt_context = "\n".join(context_lines_).strip()
            else:
                subckt_def._comments_above = ""
                subckt_def._subckt_line = subckt_line_
                subckt_def._subckt_context = "\n".join(context_lines_).strip() if context_lines_ else ""

            return subckt_def

        print(f"Extracting top-level subcircuits from {self.xyce_file}...")
        subckt_names = _extract_all_top_level_subckts_from_lines(lines)
        print(f"Found {len(subckt_names)} top-level subcircuits")

        primitives = []
        for subckt_name in subckt_names:
            try:
                subckt = _extract_subckt_from_lines(lines, subckt_name)
                prim_type = self.detector.detect(subckt)
                primitives.append((subckt_name, subckt, prim_type))
                print(f"  {subckt_name}: {prim_type.value}")
            except Exception as e:
                print(f"  Warning: Failed to extract {subckt_name}: {e}")
                continue

        print(f"\nGenerating symbols for {len(primitives)} primitives...")
        for subckt_name, subckt, prim_type in primitives:
            try:
                self.subckts[subckt_name] = subckt
                self.prim_types[subckt_name] = prim_type
                template_path, confidence, match_reasons = self.matcher.match(subckt, prim_type)

                if template_path:
                    _ = self.generator.generate_symbol(subckt, prim_type, template_path=template_path)
                    template_name = template_path.name
                    template_rel_path = template_path.relative_to(Path(__file__).parent)
                    print(f"  ✓ {subckt_name}: {template_name} (confidence: {confidence:.2f})")
                else:
                    _ = self.generator.generate_symbol(subckt, prim_type)
                    template_name = None
                    template_rel_path = None
                    print(f"  ⚠ {subckt_name}: NO TEMPLATE MATCH (confidence: {confidence:.2f}, reasons: {match_reasons})")
                    print(f"     Generated from scratch. Ports: {[p.name for p in subckt.ports]}, Params: {[p.name.name for p in subckt.params]}")

                port_names = [str(p.name) for p in subckt.ports]
                self.mapping[subckt_name] = {
                    "template": template_name,
                    "template_path": str(template_rel_path) if template_rel_path else None,
                    "primitive_type": prim_type.value,
                    "confidence": confidence,
                    "match_reasons": match_reasons,
                    "ports": port_names,
                }

            except Exception as e:
                print(f"  ✗ {subckt_name}: ERROR - {e}")
                import traceback

                traceback.print_exc()
                port_names = []
                try:
                    port_names = [str(p.name) for p in subckt.ports]
                except Exception:
                    pass
                self.mapping[subckt_name] = {
                    "template": None,
                    "template_path": None,
                    "primitive_type": prim_type.value,
                    "confidence": 0.0,
                    "match_reasons": [f"error: {str(e)}"],
                    "ports": port_names if port_names else None,
                }

        return self.mapping

    def save_mapping(self, mapping_file: Path) -> None:
        mapping_file = Path(mapping_file)
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        mapping_file.write_text(json.dumps(self.mapping, indent=2) + "\n")
        print(f"\nMapping saved to {mapping_file}")

    def write_test_schematic(self, output_sch: Path) -> Path:
        """Generate a top-level schematic placing all generated symbols, using in-memory data."""
        from .xschem_schematic import create_schematic_from_subckts

        symbol_dir = self.output_dir
        return create_schematic_from_subckts(
            symbol_dir=symbol_dir,
            output_sch=Path(output_sch),
            mapping=self.mapping,
            subckts=self.subckts,
        )

