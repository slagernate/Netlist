"""
Primitive Detection System

Detects primitive device types (MOSFET, BJT, Diode, Resistor, Capacitor, Inductor)
from SubcktDef structures using heuristics and config file overrides.
"""

from enum import Enum
from typing import Optional, Dict, List, Set
from pathlib import Path
import json

# Optional yaml support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from ..data import SubcktDef, ParamDecl, Ident, Ref


class PrimitiveType(Enum):
    """Enumerated primitive device types"""
    MOSFET = "mosfet"
    BJT = "bjt"
    DIODE = "diode"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    UNKNOWN = "unknown"


class PrimitiveDetector:
    """Detects primitive device types from SubcktDef structures"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the primitive detector.
        
        Args:
            config_file: Optional path to JSON/YAML config file with template overrides
        """
        self.config_overrides: Dict[str, PrimitiveType] = {}
        if config_file:
            self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> None:
        """Load configuration file with template overrides.
        
        Args:
            config_file: Path to JSON or YAML config file
        """
        path = Path(config_file)
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    if HAS_YAML:
                        config = yaml.safe_load(f)
                    else:
                        # YAML not available, skip YAML files
                        return
                else:
                    config = json.load(f)
            
            # Config format: {"subckt_name": "mosfet", ...}
            if isinstance(config, dict):
                for subckt_name, prim_type_str in config.items():
                    try:
                        self.config_overrides[subckt_name.lower()] = PrimitiveType(prim_type_str.lower())
                    except ValueError:
                        # Invalid primitive type, skip
                        pass
        except Exception:
            # If config loading fails, just continue without overrides
            pass
    
    def detect(self, subckt: SubcktDef) -> PrimitiveType:
        """Detect the primitive type of a SubcktDef.
        
        Args:
            subckt: The SubcktDef to analyze
            
        Returns:
            PrimitiveType enum value
        """
        # Check config overrides first
        subckt_name_lower = subckt.name.name.lower()
        if subckt_name_lower in self.config_overrides:
            return self.config_overrides[subckt_name_lower]
        
        # Analyze subcircuit structure
        port_count = len(subckt.ports)
        param_names = {p.name.name.lower() for p in subckt.params}
        
        # Check name patterns
        name_has_mos = any(keyword in subckt_name_lower for keyword in ['mos', 'fet', 'transistor', 'nch', 'pch'])
        name_has_bjt = any(keyword in subckt_name_lower for keyword in ['npn', 'pnp', 'bjt', 'q'])
        # Avoid single-letter heuristics like 'r' which match too broadly (e.g. "circuit").
        name_has_res = any(keyword in subckt_name_lower for keyword in ['resistor', 'res'])
        name_has_diode = any(keyword in subckt_name_lower for keyword in ['diode', 'dio', 'sbd'])
        # Avoid single-letter heuristics like 'c' which match too broadly (e.g. "circuit").
        name_has_cap = any(keyword in subckt_name_lower for keyword in ['capacitor', 'cap'])
        name_has_ind = any(keyword in subckt_name_lower for keyword in ['inductor', 'ind'])
        # Don't use 'l' alone as it matches too many things (MOSFET length, etc.)
        
        # Check for MOSFET (4, 5, or 6 ports, has l/w or lr/wr params, name suggests MOS)
        has_l_w = {'l', 'w'} <= param_names
        has_lr_wr = {'lr', 'wr'} <= param_names
        # Some subckts (nch*/pch*) may not have their L/W extracted in our reduced param set yet.
        # For these, allow name+port-count to classify as MOSFET even if L/W are missing.
        name_is_nch_pch = ('nch' in subckt_name_lower) or ('pch' in subckt_name_lower)
        is_mosfet = (
            port_count in [4, 5, 6]  # 4, 5, or 6 ports (d, g, s, b, [ng], [pg])
            and (has_l_w or has_lr_wr or name_is_nch_pch)  # Has either sizing params OR is clearly nch/pch
            and name_has_mos  # Name indicates MOS
        )
        
        if is_mosfet:
            return PrimitiveType.MOSFET
        
        # Check for BJT (4 ports, name suggests BJT)
        is_bjt = (
            port_count == 4  # Exactly 4 ports (c, b, e, s)
            and name_has_bjt  # Name indicates BJT
        )
        
        if is_bjt:
            return PrimitiveType.BJT
        
        # Check for Diode (usually 2 ports, but allow 3-port variants with a gnode/sub terminal)
        #
        # IMPORTANT: Names like rnod*/rpod* in this PDK are resistors (user confirmed),
        # even though they contain "nod"/"pod". So we only treat "nod"/"pod" as diode
        # hints when the device is NOT an R* device.
        name_has_dio = ('dio' in subckt_name_lower) or ('sbd' in subckt_name_lower)
        is_r_device = subckt_name_lower.startswith('r')
        name_has_nod = (not is_r_device) and ('nod' in subckt_name_lower)
        name_has_pod = (not is_r_device) and ('pod' in subckt_name_lower)
        has_diode_params = any(p in param_names for p in ['area', 'perim', 'pj', 'aw', 'al'])
        port_names_lower = [p.name.lower() for p in subckt.ports]
        is_sbd = 'sbd' in subckt_name_lower
        has_gnode_like = any(
            p in ['gnode', 'gnd', 'gnd!', 'sub', 'substrate', 'bulk', 'body', 'b']
            for p in port_names_lower
        )
        is_diode = (
            (port_count in [2, 3])
            and (name_has_diode or name_has_dio or name_has_nod or name_has_pod or has_diode_params)
            # For 3-pin diodes, require a gnode/sub-like terminal unless this is an SBD device
            # which uses numeric terminals (1 2 3) but is still a diode network.
            and (port_count == 2 or has_gnode_like or is_sbd)
        )
        
        if is_diode:
            return PrimitiveType.DIODE

        # Strong resistor heuristic for this PDK: many resistors are named r* and may not have an explicit 'r' param.
        # Treat 2- or 3-terminal r* devices as resistors.
        if subckt_name_lower.startswith('r') and port_count in [2, 3]:
            return PrimitiveType.RESISTOR
        
        # Check for Resistor (2 ports, has 'r' param or name suggests resistor)
        has_r_param = 'r' in param_names
        is_resistor = (
            port_count == 2  # Exactly 2 ports
            and (name_has_res or has_r_param)
        )
        
        if is_resistor:
            return PrimitiveType.RESISTOR
        
        # Check for Capacitor (2 or 3 ports, has 'c' param or name suggests capacitor)
        # Note: MOSFET capacitors (moscap) can have 2 or 3 ports
        has_c_param = 'c' in param_names
        is_capacitor = (
            port_count in [2, 3]  # 2 or 3 ports (regular cap or MOSFET cap)
            and (name_has_cap or has_c_param)
        )
        
        if is_capacitor:
            return PrimitiveType.CAPACITOR
        
        # Check for Inductor (2, 3, or 4 ports, name suggests inductor like "spiral")
        # Note: 'l' is also used for MOSFET length, so we need to be careful
        has_l_param = 'l' in param_names
        name_has_spiral = 'spiral' in subckt_name_lower
        is_inductor = (
            port_count in [2, 3, 4]  # 2, 3, or 4 ports (spiral inductors can have extra terminals)
            and (name_has_ind or name_has_spiral)  # Name must suggest inductor
            and not has_l_w  # Not a MOSFET (doesn't have both l and w)
        )
        
        if is_inductor:
            return PrimitiveType.INDUCTOR
        
        return PrimitiveType.UNKNOWN

