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
        name_has_mos = any(keyword in subckt_name_lower for keyword in ['mos', 'fet', 'transistor'])
        name_has_bjt = any(keyword in subckt_name_lower for keyword in ['npn', 'pnp', 'bjt', 'q'])
        name_has_res = any(keyword in subckt_name_lower for keyword in ['resistor', 'res', 'r'])
        name_has_diode = any(keyword in subckt_name_lower for keyword in ['diode', 'd'])
        name_has_cap = any(keyword in subckt_name_lower for keyword in ['capacitor', 'cap', 'c'])
        name_has_ind = any(keyword in subckt_name_lower for keyword in ['inductor', 'ind', 'l'])
        
        # Check for MOSFET (4 ports, has l and w params, name suggests MOS)
        has_l_w = {'l', 'w'} <= param_names
        is_mosfet = (
            port_count == 4  # Exactly 4 ports (d, g, s, b)
            and has_l_w  # Has both 'l' and 'w' params
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
        
        # Check for Resistor (2 ports, has 'r' param or name suggests resistor)
        has_r_param = 'r' in param_names
        is_resistor = (
            port_count == 2  # Exactly 2 ports
            and (name_has_res or has_r_param)
        )
        
        if is_resistor:
            return PrimitiveType.RESISTOR
        
        # Check for Diode (2 ports, has diode params or name suggests diode)
        has_diode_params = any(p in param_names for p in ['area', 'perim', 'pj'])
        is_diode = (
            port_count == 2  # Exactly 2 ports
            and (name_has_diode or has_diode_params)
        )
        
        if is_diode:
            return PrimitiveType.DIODE
        
        # Check for Capacitor (2 ports, has 'c' param or name suggests capacitor)
        has_c_param = 'c' in param_names
        is_capacitor = (
            port_count == 2  # Exactly 2 ports
            and (name_has_cap or has_c_param)
        )
        
        if is_capacitor:
            return PrimitiveType.CAPACITOR
        
        # Check for Inductor (2 ports, has 'l' param or name suggests inductor)
        # Note: 'l' is also used for MOSFET length, so we need to be careful
        has_l_param = 'l' in param_names
        is_inductor = (
            port_count == 2  # Exactly 2 ports
            and name_has_ind  # Name must suggest inductor (to avoid confusion with MOSFET)
            and not has_l_w  # Not a MOSFET (doesn't have both l and w)
        )
        
        if is_inductor:
            return PrimitiveType.INDUCTOR
        
        return PrimitiveType.UNKNOWN

