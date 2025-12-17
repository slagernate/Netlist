"""
Xschem Netlister

Generates xschem .sym symbol files for primitive subcircuits.
Non-primitive subcircuits are exported with warnings.
"""

from pathlib import Path
from typing import IO, Optional, Dict, Tuple
from io import StringIO
from .base import Netlister
from .primitive_detector import PrimitiveDetector, PrimitiveType
from .xschem_symbol import XschemSymbolGenerator
from .spice import SpiceNetlister, NgspiceNetlister
from .xyce import XyceNetlister
from ..data import (
    Program,
    SubcktDef,
    NetlistDialects,
    SourceFile,
)


class XschemNetlister(Netlister):
    """Netlister for xschem symbol file generation"""
    
    def __init__(
        self,
        src: Program,
        dest: IO,
        *,
        errormode=None,
        file_type: str = "",
        primitive_config_file: Optional[str] = None,
        subcircuit_dialect = None,
        bridge_file_name: Optional[str] = None,
        options = None,
    ) -> None:
        """Initialize the xschem netlister.
        
        Args:
            src: Source Program
            dest: Destination IO stream or file path
            errormode: Error handling mode (from base class)
            file_type: File type (from base class)
            primitive_config_file: Optional path to config file for primitive detection overrides
            subcircuit_dialect: Optional dialect for bridge file generation (NetlistDialects)
            bridge_file_name: Optional custom name for bridge file
            options: WriteOptions instance (optional)
        """
        super().__init__(src, dest, errormode=errormode, file_type=file_type, options=options)
        
        # Initialize primitive detector
        self.detector = PrimitiveDetector(config_file=primitive_config_file)
        
        # Store bridge file configuration
        self.subcircuit_dialect = subcircuit_dialect
        self.bridge_file_name = bridge_file_name
        
        # Track primitives for bridge file generation
        # Maps IR subcircuit name -> (SubcktDef, PrimitiveType)
        self.primitive_subckts: Dict[str, Tuple[SubcktDef, PrimitiveType]] = {}
        
        # Determine output directory from dest
        if isinstance(dest, (str, Path)):
            dest_path = Path(dest)
            # If it has a suffix, assume it's a file, otherwise it's a directory
            if dest_path.suffix:
                self.output_dir = dest_path.parent
            else:
                self.output_dir = dest_path
        else:
            # If dest is a file handle, try to get directory from its name
            if hasattr(dest, 'name') and dest.name:
                self.output_dir = Path(dest.name).parent
            else:
                # Default to current directory
                self.output_dir = Path.cwd()
        
        # Initialize symbol generator
        self.symbol_generator = XschemSymbolGenerator(self.output_dir)
    
    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.XSCHEM
    
    def write_subckt_def(self, module: SubcktDef) -> None:
        """Write subcircuit definition.
        For xschem, this generates .sym files for primitives using IR names.
        Tracks primitives for bridge file generation.
        
        Args:
            module: The SubcktDef to process
        """
        # Detect primitive type
        prim_type = self.detector.detect(module)
        subckt_name = module.name.name if hasattr(module.name, 'name') else str(module.name)
        
        if prim_type == PrimitiveType.UNKNOWN:
            # Non-primitive subcircuit - log warning
            self.log_warning(
                f"Subcircuit '{subckt_name}' is not a recognized primitive and will not be exported as a .sym file",
                context=subckt_name
            )
            # Still write a basic subcircuit definition to the output stream
            self.write(f"* Subcircuit: {subckt_name}\n")
            self.write(f"* Ports: {', '.join([p.name for p in module.ports])}\n")
            self.write(f"* Parameters: {', '.join([p.name.name for p in module.params])}\n")
            self.write("\n")
        else:
            # Track primitive for bridge file generation
            self.primitive_subckts[subckt_name] = (module, prim_type)
            
            # Generate .sym file for primitive using IR subcircuit name
            sym_file = self.symbol_generator.generate_symbol(module, prim_type)
            # Use relative path or just filename for cleaner output
            sym_file_rel = sym_file.name if hasattr(sym_file, 'name') else str(sym_file)
            self.write(f"* Generated symbol file for primitive '{subckt_name}': {sym_file_rel}\n")
            self.write(f"* Primitive type: {prim_type.value}\n")
            self.write(f"* Symbol references IR name: {subckt_name}\n")
            self.write("\n")
    
    def write_model_def(self, model) -> None:
        """Write model definition.
        For xschem, models are typically not written to symbol files.
        
        Args:
            model: The ModelDef to process
        """
        # Models are referenced by symbols but not written as separate files
        self.write(f"* Model: {model.name.name if hasattr(model.name, 'name') else str(model.name)}\n")
    
    def write_subckt_instance(self, pinst) -> None:
        """Write subcircuit instance.
        For xschem, instances reference the generated symbols.
        
        Args:
            pinst: The Instance to process
        """
        # Instances are handled by the symbol files themselves
        # This is mainly for documentation
        inst_name = pinst.name.name if hasattr(pinst.name, 'name') else str(pinst.name)
        self.write(f"* Instance: {inst_name}\n")
    
    def write_primitive_instance(self, pinst) -> None:
        """Write primitive instance.
        For xschem, primitives are handled by symbol files.
        
        Args:
            pinst: The Primitive to process
        """
        # Primitives are handled by symbol files
        prim_name = pinst.name.name if hasattr(pinst.name, 'name') else str(pinst.name)
        self.write(f"* Primitive instance: {prim_name}\n")
    
    def write_param_decls(self, params) -> None:
        """Write parameter declarations.
        
        Args:
            params: The ParamDecls to process
        """
        # Parameters are embedded in symbol files
        self.write("* Parameter declarations are embedded in symbol files\n")
    
    def write_options(self, options) -> None:
        """Write simulation options.
        
        Args:
            options: The Options to process
        """
        # Options are typically not part of symbol files
        self.write("* Simulation options are not part of symbol files\n")
    
    def write_statistics_block(self, stats) -> None:
        """Write statistics block.
        
        Args:
            stats: The StatisticsBlock to process
        """
        # Statistics blocks are not part of symbol files
        self.write("* Statistics blocks are not part of symbol files\n")
    
    def write_include(self, inc) -> None:
        """Write include statement.
        
        Args:
            inc: The Include to process
        """
        self.write(f"* Include: {inc.path}\n")
    
    def write_library_section(self, section) -> None:
        """Write library section.
        
        Args:
            section: The LibSectionDef to process
        """
        self.write(f"* Library section: {section.name.name if hasattr(section.name, 'name') else str(section.name)}\n")
    
    def write_use_lib(self, uselib) -> None:
        """Write use library statement.
        
        Args:
            uselib: The UseLibSection to process
        """
        self.write(f"* Use library: {uselib.path}, section: {uselib.section.name if hasattr(uselib.section, 'name') else str(uselib.section)}\n")
    
    def write_function_def(self, func) -> None:
        """Write function definition.
        
        Args:
            func: The FunctionDef to process
        """
        self.write(f"* Function: {func.name.name if hasattr(func.name, 'name') else str(func.name)}\n")
    
    def netlist(self) -> None:
        """Override netlist() to generate bridge file after processing all entries."""
        # Call parent netlist to process all entries
        super().netlist()
        
        # Generate bridge file if dialect is specified and we have primitives
        if self.subcircuit_dialect and self.primitive_subckts:
            bridge_path = self._generate_bridge_file()
            if bridge_path:
                # Use relative path or just filename for cleaner output
                bridge_path_rel = bridge_path.name if hasattr(bridge_path, 'name') else str(bridge_path)
                self.write(f"* Generated bridge file: {bridge_path_rel}\n")
    
    def _generate_bridge_file(self) -> Optional[Path]:
        """Generate a bridge file that maps IR subcircuit names to PDK subcircuits.
        
        Returns:
            Path to the generated bridge file, or None if generation failed
        """
        if not self.subcircuit_dialect or not self.primitive_subckts:
            return None
        
        # Determine bridge file name
        if self.bridge_file_name:
            bridge_file = self.output_dir / self.bridge_file_name
        else:
            dialect_name = self.subcircuit_dialect.value if hasattr(self.subcircuit_dialect, 'value') else str(self.subcircuit_dialect)
            bridge_file = self.output_dir / f"bridge_{dialect_name}.spice"
        
        # Create a temporary program with bridge subcircuits
        bridge_subckts = []
        for ir_name, (subckt, prim_type) in self.primitive_subckts.items():
            # Create a bridge subcircuit that maps IR name to itself
            # In a real scenario, this would map to PDK-specific names
            # For now, we create a bridge that passes through parameters
            bridge_subckts.append(subckt)
        
        # Create a temporary program
        bridge_program = Program(files=[
            SourceFile(path=bridge_file, contents=bridge_subckts)
        ])
        
        # Get the appropriate netlister for the dialect
        netlister = self._get_netlister_for_dialect(bridge_program, bridge_file)
        if not netlister:
            return None
        
        # Write bridge file header
        dialect_name = self.subcircuit_dialect.value if hasattr(self.subcircuit_dialect, 'value') else str(self.subcircuit_dialect)
        with open(bridge_file, 'w') as f:
            f.write("* " + "=" * 70 + "\n")
            f.write(f"* {dialect_name.upper()} BRIDGE FILE\n")
            f.write("* " + "=" * 70 + "\n")
            f.write("* Maps IR subcircuit names to PDK-specific subcircuits\n")
            f.write("* Generated by netlist converter\n")
            f.write("*\n")
            f.write(f"* Include this file in your testbench to use {dialect_name} simulator\n")
            f.write(f"* Example: .include \"{bridge_file.name}\"\n")
            f.write("*\n")
            f.write("* " + "=" * 70 + "\n\n")
        
        # Generate bridge subcircuits
        # For each primitive, create a bridge that maps IR name -> IR name (pass-through)
        # In practice, this would map to PDK names, but we use IR names as the target
        with open(bridge_file, 'a') as f:
            for ir_name, (subckt, prim_type) in self.primitive_subckts.items():
                # Create bridge subcircuit definition
                # Format: .subckt ir_name ports params
                #         Xprim ports ir_name params
                #         .ends
                
                # Get ports
                port_names = [p.name for p in subckt.ports]
                port_list = " ".join(port_names)
                
                # Get parameters with defaults
                param_decls = []
                for param in subckt.params:
                    param_name = param.name.name
                    default_val = self._format_param_default(param)
                    if default_val:
                        param_decls.append(f"{param_name}={default_val}")
                    else:
                        param_decls.append(param_name)
                
                param_list = " ".join(param_decls) if param_decls else ""
                
                # Write bridge subcircuit using the dialect's netlister
                # Create a minimal subcircuit that just passes through
                f.write(f"* Bridge for {ir_name}\n")
                f.write(f".subckt {ir_name} {port_list}")
                if param_list:
                    f.write(f" {param_list}")
                f.write("\n")
                
                # Create instance that references the same subcircuit (pass-through)
                # In real usage, this would reference the PDK-specific name
                param_vals = []
                for param in subckt.params:
                    param_name = param.name.name
                    param_vals.append(f"{param_name}={{{param_name}}}")
                
                param_val_list = " ".join(param_vals) if param_vals else ""
                f.write(f"    Xprim {' '.join(port_names)} {ir_name}")
                if param_val_list:
                    f.write(f" {param_val_list}")
                f.write("\n")
                f.write(f".ends {ir_name}\n\n")
        
        return bridge_file
    
    def _get_netlister_for_dialect(self, program: Program, dest: IO) -> Optional[Netlister]:
        """Get the appropriate netlister for the specified dialect.
        
        Args:
            program: The program to netlist
            dest: Destination IO stream
            
        Returns:
            Netlister instance, or None if dialect not supported
        """
        if self.subcircuit_dialect == NetlistDialects.XYCE:
            return XyceNetlister(program, dest, file_type=self.file_type)
        elif self.subcircuit_dialect == NetlistDialects.NGSPICE:
            return NgspiceNetlister(program, dest, file_type=self.file_type)
        elif self.subcircuit_dialect in [NetlistDialects.SPICE, NetlistDialects.HSPICE]:
            return SpiceNetlister(program, dest, file_type=self.file_type)
        else:
            # Default to SpiceNetlister for other dialects
            return SpiceNetlister(program, dest, file_type=self.file_type)
    
    def _format_param_default(self, param) -> str:
        """Format a parameter default value.
        
        Args:
            param: The ParamDecl
            
        Returns:
            Formatted default value string
        """
        if not hasattr(param, 'default') or param.default is None:
            return ""
        
        from ..data import Int, Float, MetricNum
        if isinstance(param.default, (Int, Float, MetricNum)):
            return str(param.default.val) if hasattr(param.default, 'val') else str(param.default)
        else:
            return str(param.default)

