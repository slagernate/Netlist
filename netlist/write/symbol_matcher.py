"""
Symbol Matcher

Matches subcircuits to generic symbol templates using heuristics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..data import SubcktDef
from .primitive_detector import PrimitiveType


class SymbolMatcher:
    """Matches subcircuits to generic symbol templates using heuristics."""
    
    def __init__(self, template_dir: Path):
        """Initialize the symbol matcher.
        
        Args:
            template_dir: Directory containing generic symbol templates
        """
        self.template_dir = template_dir
        self._load_available_templates()
    
    def _load_available_templates(self) -> None:
        """Load available templates from the template directory."""
        self.templates: Dict[PrimitiveType, List[Path]] = {
            PrimitiveType.MOSFET: [],
            PrimitiveType.RESISTOR: [],
            PrimitiveType.CAPACITOR: [],
            PrimitiveType.DIODE: [],
            PrimitiveType.BJT: [],
            PrimitiveType.INDUCTOR: [],
        }
        
        # Load MOSFET templates
        mosfet_dir = self.template_dir / "mosfet"
        if mosfet_dir.exists():
            for template_file in mosfet_dir.glob("*.sym"):
                self.templates[PrimitiveType.MOSFET].append(template_file)
        
        # Load resistor templates
        resistor_dir = self.template_dir / "resistor"
        if resistor_dir.exists():
            for template_file in resistor_dir.glob("*.sym"):
                self.templates[PrimitiveType.RESISTOR].append(template_file)
        
        # Load other device types similarly
        for dev_type in [PrimitiveType.CAPACITOR, PrimitiveType.DIODE, 
                         PrimitiveType.BJT, PrimitiveType.INDUCTOR]:
            dev_dir = self.template_dir / dev_type.value
            if dev_dir.exists():
                for template_file in dev_dir.glob("*.sym"):
                    self.templates[dev_type].append(template_file)
    
    def match(self, subckt: SubcktDef, prim_type: PrimitiveType) -> Tuple[Optional[Path], float, List[str]]:
        """Match a subcircuit to a template.
        
        Args:
            subckt: The SubcktDef to match
            prim_type: The detected primitive type
            
        Returns:
            Tuple of (template_path, confidence_score, match_reasons)
            Returns (None, 0.0, []) if no match found
        """
        if prim_type == PrimitiveType.UNKNOWN:
            return None, 0.0, ["primitive_type_unknown"]
        
        available_templates = self.templates.get(prim_type, [])
        if not available_templates:
            return None, 0.0, [f"no_templates_available_for_{prim_type.value}"]
        
        # For now, use simple heuristics to select best template
        # More sophisticated matching can be added later
        best_template = None
        best_score = 0.0
        best_reasons: List[str] = []
        
        subckt_name_lower = subckt.name.name.lower()
        port_names = {p.name.lower() for p in subckt.ports}
        param_names = {p.name.name.lower() for p in subckt.params}
        
        for template_path in available_templates:
            score = 0.0
            reasons: List[str] = []
            
            template_name = template_path.stem.lower()
            
            # Name-based matching
            if prim_type == PrimitiveType.MOSFET:
                # Check for NMOS vs PMOS
                is_pmos = (subckt_name_lower.startswith('p') or 
                          'pmos' in subckt_name_lower or 
                          'pfet' in subckt_name_lower)
                is_nmos = (subckt_name_lower.startswith('n') or 
                          'nmos' in subckt_name_lower or 
                          'nfet' in subckt_name_lower)
                
                if 'pmos' in template_name or 'pfet' in template_name:
                    if is_pmos:
                        score += 0.4
                        reasons.append("name_contains_pmos")
                    elif is_nmos:
                        score -= 0.2  # Penalty for wrong type
                elif 'nmos' in template_name or 'nfet' in template_name:
                    if is_nmos:
                        score += 0.4
                        reasons.append("name_contains_nmos")
                    elif is_pmos:
                        score -= 0.2  # Penalty for wrong type
                
                # Pin matching
                if '4pin' in template_name:
                    if len(subckt.ports) == 4:
                        score += 0.3
                        reasons.append("pins_match_4pin")
                        # Check for standard MOSFET pins
                        has_dgsb = (port_names >= {'d', 'g', 's', 'b'})
                        # Treat n1/n2/n3/n4 (or di/gi/si/bi) as D/G/S/B for *mac subckts.
                        has_n1234 = port_names >= {'n1', 'n2', 'n3', 'n4'}
                        has_digisibi = port_names >= {'di', 'gi', 'si', 'bi'}
                        if has_dgsb or has_n1234 or has_digisibi:
                            score += 0.2
                            if has_dgsb:
                                reasons.append("pins_match_dgsb")
                            elif has_n1234:
                                reasons.append("pins_match_n1n2n3n4_as_dgsb")
                            else:
                                reasons.append("pins_match_di_gi_si_bi_as_dgsb")
                    elif len(subckt.ports) in [5, 6]:
                        score += 0.1  # Partial match for variants
                        reasons.append("pins_match_variant")
                
                # Parameter matching
                if 'lr' in param_names and 'wr' in param_names:
                    # RF MOSFET
                    score += 0.1
                    reasons.append("has_lr_wr_params")
                elif 'l' in param_names and 'w' in param_names:
                    # Standard MOSFET
                    score += 0.1
                    reasons.append("has_L_W_params")
            
            elif prim_type == PrimitiveType.CAPACITOR:
                if 'capacitor' in template_name or 'cap' in template_name:
                    if 'cap' in subckt_name_lower or 'capacitor' in subckt_name_lower:
                        score += 0.3
                        reasons.append("name_contains_cap")
                
                # Pin matching - 2 or 3 terminals for capacitors
                if '2pin' in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.1  # Partial match
                        reasons.append("pins_match_3pin")
                elif '3pin' in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.1  # Partial match
                        reasons.append("pins_match_2pin")
                
                # Parameter matching
                if 'c' in param_names or 'w' in param_names or 'l' in param_names:
                    score += 0.2
                    reasons.append("has_cap_params")
            
            elif prim_type == PrimitiveType.RESISTOR:
                # For resistors, be more lenient - if primitive type is already RESISTOR,
                # we know it's a resistor even if name doesn't contain "res"
                if 'resistor' in template_name or 'res' in template_name:
                    # Name-based matching (bonus if name contains res, but not required)
                    if 'res' in subckt_name_lower or 'resistor' in subckt_name_lower:
                        score += 0.2
                        reasons.append("name_contains_res")
                    else:
                        # Still give some score if it's a resistor template and we detected RESISTOR type
                        score += 0.1
                        reasons.append("resistor_type_detected")
                
                # Pin matching - 2 or 3 terminals for resistors
                if '2pin' in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.2  # Partial match for 3-terminal resistors
                        reasons.append("pins_match_3pin")
                elif '3pin' in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.2  # Partial match
                        reasons.append("pins_match_2pin")
                
                # Parameter matching - resistors can have 'r', 'l', 'w', or other params
                if 'r' in param_names:
                    score += 0.2
                    reasons.append("has_r_param")
                elif 'l' in param_names or 'w' in param_names:
                    # Many resistors use L/W instead of R
                    score += 0.15
                    reasons.append("has_L_W_params")
            
            elif prim_type == PrimitiveType.INDUCTOR:
                # For inductors, allow 2-pin template to match 3-pin devices (just add 3rd pin)
                if '2pin' in template_name:
                    if len(subckt.ports) == 2:
                        score += 0.4
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 3:
                        score += 0.3  # Can use 2-pin template and add 3rd pin
                        reasons.append("pins_match_2pin_can_add_3rd")
                    elif len(subckt.ports) == 4:
                        score += 0.25  # Can use 2-pin template and add extra pins
                        reasons.append("pins_match_2pin_can_add_extras")
                elif '3pin' in template_name:
                    if len(subckt.ports) == 3:
                        score += 0.4
                        reasons.append("pins_match_3pin")
                    elif len(subckt.ports) == 2:
                        score += 0.2  # Partial match
                        reasons.append("pins_match_2pin")
                    elif len(subckt.ports) == 4:
                        score += 0.35  # Prefer 3-pin for 4-pin spiral w/ tap
                        reasons.append("pins_match_3pin_can_add_4th")
                
                # Name-based matching
                if 'inductor' in template_name or 'ind' in template_name:
                    if 'ind' in subckt_name_lower or 'spiral' in subckt_name_lower:
                        score += 0.2
                        reasons.append("name_contains_ind")
            
            # Generic matching for other types
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
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, max(0.0, best_score))
        
        # Lower threshold for resistors and inductors since they're simpler devices
        threshold = 0.2 if prim_type in [PrimitiveType.RESISTOR, PrimitiveType.INDUCTOR] else 0.3
        
        if best_template and confidence > threshold:
            return best_template, confidence, best_reasons
        else:
            # Return None with reasons for debugging
            if not available_templates:
                return None, 0.0, [f"no_templates_available_for_{prim_type.value}"]
            elif best_reasons:
                return None, confidence, best_reasons + ["below_threshold"]
            else:
                return None, confidence, ["no_match_found", f"available_templates: {[t.name for t in available_templates]}"]

