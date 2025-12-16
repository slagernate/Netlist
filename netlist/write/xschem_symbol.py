"""
Xschem Symbol Generator

Generates xschem .sym symbol files for primitive device types.
"""

from pathlib import Path
from typing import Dict, List, Optional
from .primitive_detector import PrimitiveType
from ..data import SubcktDef, ParamDecl, Ident


class XschemSymbolGenerator:
    """Generates xschem .sym symbol files for primitive device types"""
    
    # Pin positions for different primitive types
    # Format: (x, y) coordinates, where (0, 0) is center
    PIN_POSITIONS = {
        PrimitiveType.MOSFET: {
            'd': (-200, 0),   # Drain (left)
            'g': (0, -200),   # Gate (top)
            's': (200, 0),    # Source (right)
            'b': (0, 200),    # Bulk/Substrate (bottom)
        },
        PrimitiveType.BJT: {
            'c': (-200, 0),   # Collector (left)
            'b': (0, -200),   # Base (top)
            'e': (200, 0),    # Emitter (right)
            's': (0, 200),    # Substrate (bottom)
        },
        PrimitiveType.DIODE: {
            'p': (-200, 0),   # Anode (left)
            'n': (200, 0),    # Cathode (right)
        },
        PrimitiveType.RESISTOR: {
            'p': (-200, 0),   # Positive terminal (left)
            'n': (200, 0),    # Negative terminal (right)
        },
        PrimitiveType.CAPACITOR: {
            'p': (-200, 0),   # Positive terminal (left)
            'n': (200, 0),    # Negative terminal (right)
        },
        PrimitiveType.INDUCTOR: {
            'p': (-200, 0),   # Positive terminal (left)
            'n': (200, 0),    # Negative terminal (right)
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
    
    def __init__(self, output_dir: Path):
        """Initialize the symbol generator.
        
        Args:
            output_dir: Directory where .sym files will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._generated_symbols: Dict[PrimitiveType, str] = {}
    
    def generate_symbol(self, subckt: SubcktDef, prim_type: PrimitiveType) -> Path:
        """Generate a .sym file for a primitive subcircuit.
        Uses IR subcircuit name to preserve fidelity to the IR.
        
        Args:
            subckt: The SubcktDef to generate symbol for
            prim_type: The detected primitive type
            
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
        content = self._generate_symbol_content(subckt, prim_type)
        
        # Write to file
        with open(sym_file, 'w') as f:
            f.write(content)
        
        self._generated_symbols[subckt_name] = subckt_name
        return sym_file
    
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
        lines.append("G {} 0 0 0 0".format(subckt_name))
        
        # Get pin positions for this primitive type
        pin_positions = self.PIN_POSITIONS.get(prim_type, {})
        
        # Generate pins based on subcircuit ports
        pin_idx = 0
        for port in subckt.ports:
            port_name = port.name.lower()
            # Map port to standard pin name if possible
            pin_name = self._map_port_to_pin(port_name, prim_type, pin_idx)
            if pin_name in pin_positions:
                x, y = pin_positions[pin_name]
                # Xschem pin format: P x y x y 0 0 0 0
                # P x1 y1 x2 y2 orientation length label_offset label_angle
                lines.append("P {} {} {} {} 0 0 0 0".format(x, y, x, y))
                pin_idx += 1
        
        # Add attributes
        lines.append("T {} 0 0 0 0 0 0 0 0".format(prim_type.value))
        
        # Set spice_primitive attribute
        lines.append("T {}=true 0 0 0 0 0 0 0 0".format("spice_primitive"))
        
        # Set type attribute
        lines.append("T type={} 0 0 0 0 0 0 0 0".format(prim_type.value))
        
        # Set format attribute (not used for subcircuit instances, but kept for compatibility)
        lines.append("T format=\"@name @pinlist @symname @model\" 0 0 0 0 0 0 0 0")
        
        # Set template attribute for spice netlist generation
        # Uses IR subcircuit name in @model variable
        template = self._generate_template(subckt, prim_type)
        lines.append("T template=\"{}\" 0 0 0 0 0 0 0 0".format(template))
        
        # Set model attribute to IR subcircuit name
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        lines.append("T model={} 0 0 0 0 0 0 0 0".format(subckt_name))
        
        # Add parameter attributes
        for param in subckt.params:
            param_name = param.name.name
            default_val = self._format_param_default(param)
            lines.append("T {}={} 0 0 0 0 0 0 0 0".format(param_name, default_val))
        
        # Add drawing elements (simple rectangle for now)
        # Rectangle: R x1 y1 x2 y2
        lines.append("R -100 -100 100 100")
        
        return "\n".join(lines) + "\n"
    
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
    
    def _generate_template(self, subckt: SubcktDef, prim_type: PrimitiveType) -> str:
        """Generate the template string for spice netlist generation.
        Uses IR subcircuit name and X prefix format for subcircuit instances.
        
        Args:
            subckt: The SubcktDef
            prim_type: The primitive type
            
        Returns:
            Template string
        """
        # Use X prefix for subcircuit instances (not true primitives)
        # The @model variable will reference the IR subcircuit name
        subckt_name = subckt.name.name if hasattr(subckt.name, 'name') else str(subckt.name)
        
        # Build pin list
        pin_list = " ".join([port.name for port in subckt.ports])
        
        # Build parameter list
        param_list = []
        for param in subckt.params:
            param_name = param.name.name
            param_list.append("{}={{{}}}".format(param_name, param_name))
        
        # Use X prefix format: X@name @pinlist @model param1={param1} param2={param2}
        if param_list:
            template = "X@name {} @model {}".format(pin_list, " ".join(param_list))
        else:
            template = "X@name {} @model".format(pin_list)
        
        return template
    
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
            return str(param.default.val) if hasattr(param.default, 'val') else str(param.default)
        elif isinstance(param.default, Ref):
            return param.default.ident.name if hasattr(param.default, 'ident') else str(param.default)
        else:
            return str(param.default)

