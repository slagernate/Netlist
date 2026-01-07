"""
Batch Symbol Generator

Generates Xschem symbols for all primitive subcircuits from an Xyce file,
matching them to generic templates and producing a mapping artifact.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..data import SubcktDef, Ident, ParamDecl, Float
from .primitive_detector import PrimitiveDetector, PrimitiveType
from .symbol_matcher import SymbolMatcher
from .xschem_symbol import XschemSymbolGenerator

import re

_SPICE_SUFFIX_MULT = {
    'f': 1e-15,
    'p': 1e-12,
    'n': 1e-9,
    'u': 1e-6,
    'm': 1e-3,
    'k': 1e3,
    'meg': 1e6,
    'g': 1e9,
    't': 1e12,
}

def _parse_spice_number(s: str) -> Optional[float]:
    """Parse a simple SPICE numeric literal (supports exponent + suffix like u/k/meg)."""
    s = (s or "").strip()
    if not s:
        return None
    if s.startswith('{') or s.endswith('}'):
        return None

    s_lower = s.lower()
    suffix = None
    mantissa = s_lower
    if s_lower.endswith('meg'):
        suffix = 'meg'
        mantissa = s_lower[:-3]
    else:
        last = s_lower[-1]
        if last.isalpha():
            suffix = last
            mantissa = s_lower[:-1]

    if not re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', mantissa):
        return None

    try:
        val = float(mantissa)
    except ValueError:
        return None

    if suffix is None:
        return val
    mult = _SPICE_SUFFIX_MULT.get(suffix)
    if mult is None:
        return None
    return val * mult

# Try to import extract_subckt_params, but handle if it's not available
try:
    import sys
    from pathlib import Path
    # Add project root to path to find extract_subckt_params
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from extract_subckt_params import get_recommended_symbol_params
except ImportError:
    # Fall back to simple extraction if module not available
    def get_recommended_symbol_params(xyce_file, subckt_name, essential_params=None):
        return {}


def extract_all_top_level_subckts(xyce_file: Path) -> List[str]:
    """Extract all top-level subcircuit names from an Xyce file.
    
    Args:
        xyce_file: Path to the Xyce file
        
    Returns:
        List of subcircuit names (top-level only)
    """
    subckt_names = []
    with open(xyce_file, 'r') as f:
        lines = f.readlines()
    
    # Track which subcircuits are referenced (not top-level)
    referenced_subckts = set()
    current_subckt = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Find .SUBCKT declarations
        if line_stripped.startswith('.SUBCKT'):
            parts = line_stripped.split()
            if len(parts) >= 2:
                subckt_name = parts[1]
                subckt_names.append(subckt_name)
                current_subckt = subckt_name
        
        # Find .ENDS to mark end of subcircuit
        elif line_stripped.startswith('.ENDS'):
            current_subckt = None
        
        # Find subcircuit instantiations (X lines)
        elif line_stripped.startswith('X') or line_stripped.startswith('x'):
            parts = line_stripped.split()
            if len(parts) >= 2:
                # The subcircuit name is typically the last token before parameters
                # or after all node names
                for part in parts[1:]:
                    if part and not part.startswith('{') and '=' not in part:
                        referenced_subckts.add(part)
                        break
    
    # Top-level subcircuits are those that are declared but not referenced
    top_level = [name for name in subckt_names if name not in referenced_subckts]
    
    # If we couldn't determine, return all (better to generate too many than miss some)
    if not top_level:
        return subckt_names
    
    return top_level


def extract_subckt_from_xyce(xyce_file: Path, subckt_name: str) -> SubcktDef:
    """Extract a subcircuit definition from Xyce file.
    
    Args:
        xyce_file: Path to the Xyce file
        subckt_name: Name of the subcircuit to extract
        
    Returns:
        SubcktDef object
    """
    with open(xyce_file, 'r') as f:
        lines = f.readlines()
    
    # Find the .SUBCKT line and extract comments above it + surrounding context (±10 lines)
    subckt_line = None
    subckt_line_idx = None
    comments_above = []
    context_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith(f'.SUBCKT {subckt_name}'):
            subckt_line = line.strip()
            subckt_line_idx = i
            # Extract up to 20 lines above (comments)
            start_idx = max(0, i - 20)
            for j in range(start_idx, i):
                comment_line = lines[j].rstrip()
                if comment_line.strip() and not comment_line.strip().startswith('.SUBCKT'):
                    comments_above.append(comment_line)
            # Extract ±5 lines surrounding the .SUBCKT line (user request)
            ctx_start = max(0, i - 5)
            ctx_end = min(len(lines), i + 6)
            context_lines = [ln.rstrip("\n") for ln in lines[ctx_start:ctx_end]]
            break
    
    if not subckt_line:
        raise ValueError(f"Subcircuit '{subckt_name}' not found in {xyce_file}")
    
    # Ensure comments_above is defined even if no comments found
    if subckt_line_idx is None:
        comments_above = []
        subckt_line_idx = -1
    
    # Parse the line: .SUBCKT name ports PARAMS: params...
    parts = subckt_line.split()
    if parts[0] != '.SUBCKT':
        raise ValueError(f"Invalid subcircuit line: {subckt_line}")
    
    # Extract name
    name = parts[1]
    
    # Extract ports (everything until PARAMS:)
    ports = []
    for i, part in enumerate(parts[2:], start=2):
        if part == 'PARAMS:':
            break
        ports.append(part)
    
    # Use systematic parameter extraction
    # Determine essential params based on subcircuit name
    essential_params = []
    subckt_name_lower = subckt_name.lower()
    if 'rf' in subckt_name_lower or 'mos' in subckt_name_lower:
        # RF MOSFETs use lr, wr, nr
        essential_params = ['lr', 'wr', 'nr']
    elif 'nch' in subckt_name_lower or 'pch' in subckt_name_lower:
        # Common MOSFET naming in this PDK (ensure L/W are extracted so primitive detection works)
        # Also include nr where present (some devices use nr instead of / in addition to nf).
        essential_params = ['l', 'w', 'nf', 'nr', 'mult', 'm']
    elif 'res' in subckt_name_lower or subckt_name_lower.startswith('r'):
        # Resistors might use l, w, or r
        essential_params = ['r', 'l', 'w']
    elif 'spiral' in subckt_name_lower or 'ind' in subckt_name_lower:
        # Spiral inductors: expose key geometry knobs
        essential_params = ['rad', 'w', 'nr', 'spacing', 'gdis', 'lay', 'factor']
    
    try:
        recommended_params = get_recommended_symbol_params(
            xyce_file, subckt_name, essential_params=essential_params
        )
    except Exception as e:
        # Fall back to extracting from header only
        recommended_params = {}
        if 'PARAMS:' in subckt_line:
            params_start = subckt_line.find('PARAMS:') + len('PARAMS:')
            params_str = subckt_line[params_start:].strip()
            for param_str in params_str.split():
                if '=' in param_str:
                    param_name, param_val = param_str.split('=', 1)
                    param_val_clean = param_val.strip('{}')
                    recommended_params[param_name] = param_val_clean
    
    # Convert to ParamDecl objects
    param_decls = []
    for param_name, default_str in sorted(recommended_params.items()):
        # Prefer real numeric defaults from the PARAMS line, including exponent and metric suffixes.
        parsed = _parse_spice_number(default_str)
        if parsed is not None:
            default_val = Float(float(parsed))
        else:
            # Complex expressions / parameter references: use safe defaults
            if param_name in ['sca', 'scb', 'scc', 'nrd', 'nrs']:
                default_val = Float(0.0)
            elif param_name in ['sigma', 'rbflag']:
                default_val = Float(1.0)
            elif param_name == 'mismatchflag':
                default_val = Float(0.0)
            elif param_name == 'lr':
                default_val = Float(6e-08)
            elif param_name == 'wr':
                default_val = Float(1e-06)
            elif param_name == 'nr':
                default_val = Float(4.0)
            else:
                default_val = Float(0.0)
        
        param_decls.append(ParamDecl(name=Ident(param_name), default=default_val))
    
    # Store comments as metadata (we'll add this to the symbol later)
    subckt_def = SubcktDef(
        name=Ident(name),
        ports=[Ident(p) for p in ports],
        params=param_decls,
        entries=[]
    )
    # Store comments in a custom attribute (we'll access this in symbol generation)
    if subckt_line_idx is not None and subckt_line_idx >= 0:
        subckt_def._comments_above = "\n".join(comments_above)
        subckt_def._subckt_line = subckt_line
        subckt_def._subckt_context = "\n".join(context_lines).strip()
    else:
        subckt_def._comments_above = ""
        subckt_def._subckt_line = subckt_line if subckt_line else ""
        subckt_def._subckt_context = "\n".join(context_lines).strip() if context_lines else ""
    
    return subckt_def


class BatchSymbolGenerator:
    """Generates symbols for all primitive subcircuits in an Xyce file."""
    
    def __init__(self, xyce_file: Path, output_dir: Path, 
                 template_dir: Optional[Path] = None,
                 primitive_config_file: Optional[Path] = None):
        """Initialize the batch symbol generator.
        
        Args:
            xyce_file: Path to the Xyce file
            output_dir: Directory where symbols will be written
            template_dir: Directory containing generic symbol templates
            primitive_config_file: Optional config file for primitive detection
        """
        self.xyce_file = Path(xyce_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = PrimitiveDetector(
            config_file=str(primitive_config_file) if primitive_config_file else None
        )
        
        template_base = template_dir or (Path(__file__).parent / "xschem_symbol_templates")
        self.matcher = SymbolMatcher(template_base)
        self.generator = XschemSymbolGenerator(self.output_dir, template_dir=template_base)
        
        self.mapping: Dict[str, Dict] = {}
    
    def generate_all(self) -> Dict[str, Dict]:
        """Generate symbols for all primitive subcircuits.
        
        Returns:
            Mapping dictionary: {subckt_name: {template, template_path, primitive_type, confidence, match_reasons}}
        """
        print(f"Extracting top-level subcircuits from {self.xyce_file}...")
        subckt_names = extract_all_top_level_subckts(self.xyce_file)
        print(f"Found {len(subckt_names)} top-level subcircuits")
        
        primitives = []
        for subckt_name in subckt_names:
            try:
                subckt = extract_subckt_from_xyce(self.xyce_file, subckt_name)
                prim_type = self.detector.detect(subckt)
                
                # Include ALL top-level subcircuits in symbol generation, even if unknown.
                # This provides a complete library + helps surface missing template coverage.
                primitives.append((subckt_name, subckt, prim_type))
                print(f"  {subckt_name}: {prim_type.value}")
            except Exception as e:
                print(f"  Warning: Failed to extract {subckt_name}: {e}")
                continue
        
        print(f"\nGenerating symbols for {len(primitives)} primitives...")
        
        for subckt_name, subckt, prim_type in primitives:
            try:
                # Match to template
                template_path, confidence, match_reasons = self.matcher.match(subckt, prim_type)
                
                # Generate symbol
                if template_path:
                    sym_file = self.generator.generate_symbol(subckt, prim_type, template_path=template_path)
                    template_name = template_path.name
                    template_rel_path = template_path.relative_to(Path(__file__).parent)
                    print(f"  ✓ {subckt_name}: {template_name} (confidence: {confidence:.2f})")
                else:
                    # No template match - report as error/warning
                    sym_file = self.generator.generate_symbol(subckt, prim_type)
                    template_name = None
                    template_rel_path = None
                    print(f"  ⚠ {subckt_name}: NO TEMPLATE MATCH (confidence: {confidence:.2f}, reasons: {match_reasons})")
                    print(f"     Generated from scratch. Ports: {[p.name for p in subckt.ports]}, Params: {[p.name.name for p in subckt.params]}")
                
                # Extract port names from subcircuit
                # Ports are Ident objects where port.name is the string name
                port_names = []
                for port in subckt.ports:
                    if hasattr(port, 'name'):
                        # port.name is already the string name (not another Ident)
                        port_names.append(str(port.name))
                    else:
                        port_names.append(str(port))
                
                # Record mapping
                self.mapping[subckt_name] = {
                    "template": template_name,
                    "template_path": str(template_rel_path) if template_rel_path else None,
                    "primitive_type": prim_type.value,
                    "confidence": confidence,
                    "match_reasons": match_reasons,
                    "ports": port_names
                }
                
            except Exception as e:
                print(f"  ✗ {subckt_name}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                # Extract port names even on error
                port_names = []
                try:
                    for port in subckt.ports:
                        if hasattr(port, 'name'):
                            port_names.append(str(port.name))
                        else:
                            port_names.append(str(port))
                except:
                    pass
                
                self.mapping[subckt_name] = {
                    "template": None,
                    "template_path": None,
                    "primitive_type": prim_type.value,
                    "confidence": 0.0,
                    "match_reasons": [f"error: {str(e)}"],
                    "ports": port_names if port_names else None
                }
        
        return self.mapping
    
    def save_mapping(self, mapping_file: Path) -> None:
        """Save the mapping to a JSON file.
        
        Args:
            mapping_file: Path to the JSON file to write
        """
        with open(mapping_file, 'w') as f:
            json.dump(self.mapping, f, indent=2)
        print(f"\nMapping saved to {mapping_file}")

