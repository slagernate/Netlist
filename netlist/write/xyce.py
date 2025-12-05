"""
Xyce Dialect Netlister
"""

import re
import sys
from enum import Enum
from typing import IO, List, Dict, Optional, Tuple, Union
from warnings import warn

from ..data import (
    Program,
    Ident,
    Ref,
    ParamDecl,
    ParamDecls,
    SubcktDef,
    LibSectionDef,
    FunctionDef,
    Return,
    ParamVal,
    Instance,
    Primitive,
    Options,
    ModelDef,
    ModelFamily,
    ModelVariant,
    Expr,
    Int,
    Float,
    MetricNum,
    QuotedString,
    StatisticsBlock,
    NetlistDialects,
    Call,
    Comment,
)
from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, apply_statistics_variations, debug_find_all_param_refs, replace_param_refs_in_program, expr_references_param, count_param_refs_in_entry


class XyceNetlister(SpiceNetlister):
    """Xyce-Format Netlister"""

    def __init__(self, src: Program, dest: IO, *, errormode: ErrorMode = ErrorMode.RAISE, file_type: str = "", includes: List[Tuple[str, str]] = None, model_file: str = None, model_level_mapping: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> None:
        super().__init__(src, dest, errormode=errormode, file_type=file_type)
        self.includes = includes or []
        self.model_file = model_file
        self._last_entry_was_instance = False  # Track if last entry written was an instance
        # Track reserved parameter name mappings (reserved_name -> safe_name)
        self._reserved_param_map: Dict[str, str] = {}
        # Xyce reserved variable names (case-insensitive)
        self._reserved_names = {'vt', 'temp', 'time', 'freq', 'omega', 'pi', 'e'}
        # Collect all parameter names from the AST to only rename actual parameters
        self._param_names = self._collect_param_names(src)
        # Process model_level_mapping: convert lists to dicts for efficient lookup
        self._model_level_mapping: Dict[str, Dict[int, int]] = {}
        if model_level_mapping:
            for key, mappings in model_level_mapping.items():
                self._model_level_mapping[key.lower()] = {in_level: out_level for in_level, out_level in mappings}

    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.XYCE

    def _collect_param_names(self, program: Optional[Program]) -> set:
        """Collect all parameter names from the program AST."""
        param_names = set()
        
        if program is None:
            return param_names
        
        def collect_from_entry(entry):
            if isinstance(entry, ParamDecl):
                param_names.add(entry.name.name.lower())
            elif isinstance(entry, ParamDecls):
                for param in entry.params:
                    param_names.add(param.name.name.lower())
            elif isinstance(entry, SubcktDef):
                # Subcircuit formal parameters
                for param in entry.params:
                    param_names.add(param.name.name.lower())
                # Parameters declared inside subcircuit body
                for sub_entry in entry.entries:
                    collect_from_entry(sub_entry)
            elif isinstance(entry, LibSectionDef):
                for sub_entry in entry.entries:
                    collect_from_entry(sub_entry)
        
        for file in program.files:
            for entry in file.contents:
                collect_from_entry(entry)
        
        return param_names
    
    def format_ident(self, ident_or_ref: Union[Ident, Ref]) -> str:
        """Format an identifier or reference, renaming reserved Xyce parameter names."""
        # Get the base name
        if isinstance(ident_or_ref, Ref):
            name = ident_or_ref.ident.name
        else:
            name = ident_or_ref.name
        
        # Only rename if this is both:
        # 1. A reserved name (case-insensitive)
        # 2. Actually a parameter name (not a built-in variable)
        name_lower = name.lower()
        if name_lower in self._reserved_names and name_lower in self._param_names:
            # Check if we've already created a mapping for this name
            if name_lower not in self._reserved_param_map:
                # Create a safe name by appending _param
                safe_name = f"{name}_param"
                self._reserved_param_map[name_lower] = safe_name
                self.log_warning(f"Renaming reserved parameter '{name}' to '{safe_name}' for Xyce compatibility")
            return self._reserved_param_map[name_lower]
        
        return name

    def netlist(self) -> None:
        """Override netlist() to apply Xyce-specific statistics variations before writing.

        This applies statistics variations (process and mismatch) to the Program,
        creating Xyce-specific artifacts like .FUNC definitions for mismatch parameters
        and enable_mismatch parameter. This only happens when writing to Xyce format.
        """
        # Check if we are generating a test netlist
        if self.file_type == "test":
            return self.write_test_netlist()

        # Apply statistics variations with XYCE format before writing
        # This modifies the AST directly, replacing StatisticsBlocks with generated content
        apply_statistics_variations(self.src, output_format=NetlistDialects.XYCE)

        # Call parent implementation to do the actual writing
        super().netlist()

    def write_test_netlist(self) -> None:
        """Write a sanity check netlist."""
        # Header
        self.write("* Sanity Check Netlist generated by XyceNetlister\n\n")
        
        # Monte Carlo Control Parameters
        self.write("* Monte Carlo Control Parameters\n")
        self.write(".param process_mc_factor=0\n")
        self.write(".param enable_mismatch=0\n")
        self.write(".param mismatch_factor=0\n")
        self.write(".param corner_factor=0\n\n")
        
        # Includes
        self.write("* Includes\n")
        for inc_file, section in self.includes:
            if section:
                self.write(f".lib '{inc_file}' {section}\n")
            else:
                self.write(f".include '{inc_file}'\n")
        
        if self.model_file:
             self.write(f".include '{self.model_file}'\n")
        self.write("\n")
        
        # Ground Reference
        self.write("* Ground Reference\n")
        self.write("R_gnd_ref 0 0 1M\n\n")
        
        self.write("* Device Instantiations\n\n")
        
        inst_count = 0
        node_count = 0  # Simple counter for net names
        
        # Iterate definitions - only instantiate subcircuits, not models
        definitions = []
        for file in self.src.files:
            for entry in file.contents:
                if isinstance(entry, SubcktDef):
                    definitions.append(entry)
                    
        for defn in definitions:
            name = defn.name.name
            inst_name = f"{name}_{inst_count}"
            inst_count += 1
            
            # Get ports from subcircuit definition
            ports = [p.name for p in defn.ports]
            prefix = "X"
            
            # Instantiate resistors to ground for each port with simple net names
            nodes = []
            for port in ports:
                node = f"n{node_count}"
                nodes.append(node)
                self.write(f"R_pd_{node} {node} 0 1M\n")
                node_count += 1
            
            # Instantiate subcircuit with only m=1 parameter (skip all other defaults)
            full_inst_name = f"{prefix}_{inst_name}"
            self.write(f"{full_inst_name} {' '.join(nodes)} {name}\n")
            
            # Collect parameters to write - only m=1 for test files
            inst_params = []
            
            # Only add m=1 parameter, skip all other default parameters
            inst_params.append("m=1")
            
            # Write parameters
            if inst_params:
                self.write(f"+ {' '.join(inst_params)}\n")
            
            self.write("\n")
            
        self.write("* Simulation Commands\n")
        self.write(".tran 1n 10n\n")
        self.write(".print tran V(*)\n")
        self.write(".end\n")

    def _validate_content(self) -> None:
        """Validate that the program content matches the expected file_type."""

        for source_file in self.src.files:
            for entry in source_file.contents:
                if self.file_type == "library":
                    # Library files should ONLY contain LibSectionDef and Comments (no subcircuits, no loose parameters)
                    if not isinstance(entry, (LibSectionDef, Comment)):
                        raise ValueError(f"Library file contains non-library content: {type(entry).__name__}. "
                                       "Library files should only contain library sections and comments.")
                # For models files, allow anything (SubcktDef, ParamDecls, FunctionDef, FlatStatement, etc.)
                # No validation needed for models files


    def write_library_section(self, section: LibSectionDef) -> None:
        """Write a Library Section definition."""
        # Write library section normally (no process variations here)
        self.write(f".lib {self.format_ident(section.name)}\n")
        for entry in section.entries:
            self.write_entry(entry)
        self.write(f".endl {self.format_ident(section.name)}\n\n")

    def write_module_params(self, params: List[ParamDecl]) -> None:
        """Write the parameter declarations for Module `module`.
        Parameter declaration format:
        .SUBCKT <name> <ports>
        + PARAMS: name1=val1 name2=val2 name3=val3 \n
        """
        self.write("+ PARAMS: ")  # <= Xyce-specific
        for param in params:
            self.write_param_decl(param)
            self.write(" ")
        self.write("\n")

    def _get_model_type(self, ref: Ref) -> Optional[str]:
        """Resolve a reference to a model and return its type (mtype)."""
        if not isinstance(ref, Ref):
            return None
        
        # Try to find the model in the program
        model_name = ref.ident.name
        
        # Check resolved first
        if ref.resolved:
             if hasattr(ref.resolved, 'mtype'):
                 return ref.resolved.mtype.name.lower()
             if hasattr(ref.resolved, 'variants') and ref.resolved.variants:
                 return ref.resolved.variants[0].mtype.name.lower()
        
        # Search program
        from ..data import ModelDef, ModelFamily, ModelVariant, LibSectionDef, SubcktDef
        
        def check_entry(entry):
            if isinstance(entry, ModelDef) and entry.name.name.lower() == model_name.lower():
                return entry.mtype.name.lower()
            if isinstance(entry, ModelFamily) and entry.name.name.lower() == model_name.lower():
                return entry.mtype.name.lower()
            if isinstance(entry, ModelVariant):
                 if f"{entry.model.name}.{entry.variant.name}".lower() == model_name.lower() or entry.model.name.lower() == model_name.lower():
                     return entry.mtype.name.lower()
            return None

        for file in self.src.files:
            for entry in file.contents:
                mtype = check_entry(entry)
                if mtype: return mtype
                
                if isinstance(entry, LibSectionDef):
                    for e in entry.entries:
                        mtype = check_entry(e)
                        if mtype: return mtype
                
                if isinstance(entry, SubcktDef):
                    for e in entry.entries:
                        mtype = check_entry(e)
                        if mtype: return mtype
        return None

    def _get_module_definition(self, ref: Ref) -> Optional[Union[SubcktDef, ModelDef, ModelFamily, ModelVariant]]:
        """Get the subcircuit or model definition from a Ref."""
        if not isinstance(ref, Ref):
            return None
        
        # Check resolved first
        if ref.resolved:
            from ..data import SubcktDef, ModelDef, ModelFamily, ModelVariant
            if isinstance(ref.resolved, (SubcktDef, ModelDef, ModelFamily, ModelVariant)):
                return ref.resolved
        
        # Search program
        from ..data import ModelDef, ModelFamily, ModelVariant, LibSectionDef, SubcktDef
        module_name = ref.ident.name
        
        def find_entry(entry):
            if isinstance(entry, (SubcktDef, ModelDef, ModelFamily)) and entry.name.name.lower() == module_name.lower():
                return entry
            if isinstance(entry, ModelVariant):
                if f"{entry.model.name}.{entry.variant.name}".lower() == module_name.lower() or entry.model.name.lower() == module_name.lower():
                    return entry
            return None

        for file in self.src.files:
            for entry in file.contents:
                result = find_entry(entry)
                if result: return result
                
                if isinstance(entry, LibSectionDef):
                    for e in entry.entries:
                        result = find_entry(e)
                        if result: return result
                
                if isinstance(entry, SubcktDef):
                    for e in entry.entries:
                        result = find_entry(e)
                        if result: return result
        return None

    def _is_default_param_value(self, pval: ParamVal, module_def: Optional[Union[SubcktDef, ModelDef, ModelFamily, ModelVariant]]) -> bool:
        """Check if a parameter value equals its default value in the module definition."""
        if module_def is None:
            return False
        
        from ..data import ModelFamily, ModelVariant
        
        # Get parameter list from module definition
        params = []
        if isinstance(module_def, SubcktDef):
            params = module_def.params
        elif isinstance(module_def, ModelDef):
            params = module_def.params
        elif isinstance(module_def, ModelFamily):
            # For ModelFamily, check the first variant (or all variants if needed)
            if module_def.variants:
                params = module_def.variants[0].params
        elif isinstance(module_def, ModelVariant):
            params = module_def.params
        
        # Find the default value for this parameter
        param_name = pval.name.name
        default_param = None
        for param in params:
            if param.name.name == param_name:
                default_param = param
                break
        
        if default_param is None or default_param.default is None:
            # No default defined, so this is not a default value
            return False
        
        # Compare the instance value with the default
        # For simple literals, compare directly
        from ..data import Int, Float, MetricNum
        if isinstance(pval.val, (Int, Float, MetricNum)) and isinstance(default_param.default, (Int, Float, MetricNum)):
            # Compare numeric values
            pval_num = float(pval.val.val) if hasattr(pval.val, 'val') else float(pval.val)
            default_num = float(default_param.default.val) if hasattr(default_param.default, 'val') else float(default_param.default)
            return abs(pval_num - default_num) < 1e-10
        
        # For expressions, compare formatted strings (simple approach)
        # This is not perfect but should work for most cases
        pval_str = self.format_expr(pval.val)
        default_str = self.format_expr(default_param.default)
        return pval_str == default_str

    def write_instance_params(self, pvals: List[ParamVal], is_mos_like: bool = False, module_ref: Optional[Ref] = None) -> None:
        """Write the parameter-values for Instance `pinst`.

        Parameter-values format:
        ```
        XNAME
        + <ports>
        + <subckt-name>
        + PARAMS: name1=val1 name2=val2 name3=val3  (for subcircuits)
        + name1=val1 name2=val2 name3=val3  (for models, no PARAMS:)
        
        Args:
            pvals: List of parameter values to write
            is_mos_like: Whether this is a model instance (MOS, BJT, etc) - suppresses PARAMS:
            module_ref: Optional Ref to the module/model being instantiated (for BSIM4 detection)
        """
        self.write("+ ")

        if not pvals:  # Write a quick comment for no parameters
            return self.write_comment("No parameters")

        # Filter deltox from instance parameters if this instance references a BSIM4 model
        # (dtox is only valid in model definitions, not instance parameters)
        params_to_write = pvals
        if module_ref:
            is_bsim4 = self._is_bsim4_model_ref(module_ref)
            if is_bsim4:
                # Filter deltox AND dtox from instance parameters
                # Xyce does not support them on instance, they must be on the model.
                # We handle moving them to the model in write_subckt_def, so here we just ensure they are gone.
                params_to_write = [pval for pval in pvals if pval.name.name not in ("deltox", "dtox")]

        # Filter out default parameters (except m=1 which should always be written)
        # Only do this for test file type
        if self.file_type == "test" and module_ref:
            module_def = self._get_module_definition(module_ref)
            if module_def:
                filtered_params = []
                for pval in params_to_write:
                    param_name = pval.name.name
                    # Always include m=1
                    if param_name == "m":
                        from ..data import Int, Float
                        # Check if it's m=1
                        is_m_one = False
                        if isinstance(pval.val, (Int, Float)):
                            val = float(pval.val.val) if hasattr(pval.val, 'val') else float(pval.val)
                            is_m_one = abs(val - 1.0) < 1e-10
                        elif isinstance(pval.val, Ref) and pval.val.ident.name == "m":
                            # m={m} reference - always include
                            is_m_one = True
                        
                        if is_m_one:
                            # Always include m=1
                            filtered_params.append(pval)
                        else:
                            # For m != 1, skip if it matches default
                            if not self._is_default_param_value(pval, module_def):
                                filtered_params.append(pval)
                    else:
                        # For other parameters, skip if they match defaults
                        if not self._is_default_param_value(pval, module_def):
                            filtered_params.append(pval)
                params_to_write = filtered_params

        # Models (primitives) instances should NOT have PARAMS: keyword
        if not is_mos_like:
            self.write("PARAMS: ")  # <= Xyce-specific for subcircuits
        # And write them
        for pval in params_to_write:
            self.write_param_val(pval)
            self.write(" ")

        self.write("\n")

    def write_subckt_instance(self, pinst: Instance) -> None:
        """Write sub-circuit-instance `pinst`. Override to handle model instances without PARAMS:."""
        
        # Check if the module being instantiated is a Model (not a Subckt)
        mtype = None
        if isinstance(pinst.module, Ref):
            mtype = self._get_model_type(pinst.module)
            
            # Fallback: Check for resistor/capacitor/inductor by name if model definition not found
            if mtype is None:
                mod_name = pinst.module.ident.name.lower()
                if mod_name in ('resistor', 'res'):
                    mtype = 'r'
                elif mod_name in ('capacitor', 'cap'):
                    mtype = 'c'
                elif mod_name in ('inductor', 'ind'):
                    mtype = 'l'
        
        is_model = mtype is not None
        
        prefix = "X"
        if is_model:
            # Determine prefix based on model type
            if "mos" in mtype or mtype == "bsim4": prefix = "M"
            elif "res" in mtype or mtype == "r": prefix = "R"
            elif "cap" in mtype or mtype == "c": prefix = "C"
            elif "ind" in mtype or mtype == "l": prefix = "L"
            elif "dio" in mtype or mtype == "d": prefix = "D"
            elif "pnp" in mtype or "npn" in mtype or mtype == "q": prefix = "Q"
            # Add more types as needed
        
        # Fallback heuristic for MOS if type not found but looks like MOS
        if not is_model:
             # Detect if the instance looks like a MOS device (even if parsed as subcircuit instance)
            is_mos_like_heuristic = (
                len(pinst.conns) == 4  # Exactly 4 ports (d, g, s, b)
                and isinstance(pinst.module, Ref)  # References a model
                and {'l', 'w'} <= {p.name.name for p in pinst.params}  # Has both 'l' and 'w' params
                and any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet', 'pmos', 'nmos'])  # Instance name indicates MOS
            )
            if is_mos_like_heuristic:
                prefix = "M"
                is_model = True # Treat as model for params formatting

        # Check if model is resistor, capacitor, or inductor and handle accordingly
        skip_model_name = False
        if isinstance(pinst.module, Ref):
            mod_name = pinst.module.ident.name.lower()
            inst_name_base = self.format_ident(pinst.name)
            
            # For resistor, capacitor, and inductor, Xyce doesn't need the model name
            # (it's inferred from the prefix)
            if mod_name in ("resistor", "res", "r"):
                if not inst_name_base.upper().startswith("R"):
                    prefix = "R"
                    self.log_warning(f"Instance {pinst.name.name} uses resistor model but lacks R prefix. Prepending R_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "R"  # Ensure prefix is set
                    # If name starts with lowercase 'r', we still want to ensure it's uppercase R
                    if inst_name_base.startswith("r") and not inst_name_base.startswith("R"):
                        inst_name_base = "R" + inst_name_base[1:]
                skip_model_name = True
            elif mod_name in ("capacitor", "cap", "c"):
                if not inst_name_base.upper().startswith("C"):
                    prefix = "C"
                    self.log_warning(f"Instance {pinst.name.name} uses capacitor model but lacks C prefix. Prepending C_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "C"  # Ensure prefix is set
                    # If name starts with lowercase 'c', we still want to ensure it's uppercase C
                    if inst_name_base.startswith("c") and not inst_name_base.startswith("C"):
                        inst_name_base = "C" + inst_name_base[1:]
                skip_model_name = True
            elif mod_name in ("inductor", "ind", "l"):
                if not inst_name_base.upper().startswith("L"):
                    prefix = "L"
                    self.log_warning(f"Instance {pinst.name.name} uses inductor model but lacks L prefix. Prepending L_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "L"  # Ensure prefix is set
                    # If name starts with lowercase 'l', we still want to ensure it's uppercase L
                    if inst_name_base.startswith("l") and not inst_name_base.startswith("L"):
                        inst_name_base = "L" + inst_name_base[1:]
                skip_model_name = True

        inst_name = self.format_ident(pinst.name)
        if prefix and not inst_name.upper().startswith(prefix):
            # If we're skipping model name (resistor/capacitor/inductor), use underscore separator
            if skip_model_name:
                inst_name = f"{prefix}_{inst_name}"
            else:
                inst_name = f"{prefix}{inst_name}"

        # Write the instance name
        self.write(inst_name + " \n")

        # Write its port-connections
        self.write_instance_conns(pinst)

        # Write the sub-circuit/model name (skip for resistor/capacitor/inductor)
        if not skip_model_name:
            self.write("+ " + self.format_ident(pinst.module.ident) + " \n")

        # Check if instance parameter expressions reference 'm' and add m={m} if needed
        has_m_in_params = any(p.name.name == "m" for p in pinst.params)
        references_m = False
        
        if is_model:
            # Check if any parameter value references 'm'
            for pval in pinst.params:
                if isinstance(pval.val, Expr) and expr_references_param(pval.val, "m"):
                    references_m = True
                    break
        
        # Add m={m} if referenced but not present, and parent subcircuit has m parameter
        if references_m and not has_m_in_params:
            # Check if parent subcircuit has m parameter
            parent_has_m = False
            if not hasattr(self, '_current_subckt'):
                self._current_subckt = None
            if self._current_subckt:
                parent_has_m = any(p.name.name == "m" for p in self._current_subckt.params)
            
            if parent_has_m:
                # Add m={m} to instance params
                m_param_val = ParamVal(name=Ident("m"), val=Ref(ident=Ident("m")))
                # Insert m parameter at the beginning (typical MOS device ordering)
                pinst.params.insert(0, m_param_val)
                self.log_warning(f"Added m={{m}} parameter to instance {pinst.name.name} (referenced in expressions)", f"Instance: {pinst.name.name}")

        # Write its parameter values (pass is_model to skip PARAMS: for models, and module_ref for BSIM4 detection)
        module_ref = pinst.module if isinstance(pinst.module, Ref) else None
        self.write_instance_params(pinst.params, is_mos_like=is_model, module_ref=module_ref)

        # Add a blank line after each instance for spacing between instances
        # If next entry is a model, write_model_def will handle spacing (but won't add extra)
        self.write("\n")
        # Mark that last entry was an instance
        self._last_entry_was_instance = True

    def write_primitive_instance(self, pinst: Instance) -> None:
        """Write primitive-instance `pinst` of `rmodule`."""
        
        # Identify model reference
        module_ref = None
        if pinst.args and isinstance(pinst.args[-1], Ref):
            module_ref = pinst.args[-1]
            
        # Determine model type
        mtype = self._get_model_type(module_ref) if module_ref else None
        
        is_model = True # Primitives are generally models
        
        prefix = ""
        if mtype:
            if "mos" in mtype or mtype == "bsim4": prefix = "M"
            elif "res" in mtype or mtype == "r": prefix = "R"
            elif "cap" in mtype or mtype == "c": prefix = "C"
            elif "ind" in mtype or mtype == "l": prefix = "L"
            elif "dio" in mtype or mtype == "d": prefix = "D"
            elif "pnp" in mtype or "npn" in mtype or mtype == "q": prefix = "Q"
        
        # Fallback heuristic
        if not prefix:
            is_mos_like = (
                len(pinst.args) == 5  # Exactly 4 ports + 1 model
                and isinstance(pinst.args[-1], Ref)  # Last arg is the model reference
                and {'l', 'w'} <= {p.name.name for p in pinst.kwargs}  # Has both 'l' and 'w' params
                and any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet', 'pmos', 'nmos'])  # Instance name indicates MOS
            )
            if is_mos_like:
                prefix = "M"

        # Get base instance name for prefix checking
        inst_name_base = self.format_ident(pinst.name)
        
        # Check if model is resistor, capacitor, or inductor and handle accordingly
        model_name = None
        skip_model_name = False
        if pinst.args and isinstance(pinst.args[-1], Ref):
            model_name = pinst.args[-1].ident.name.lower()
            
            # For resistor, capacitor, and inductor, Xyce doesn't need the model name
            # (it's inferred from the prefix)
            if model_name in ("resistor", "res", "r"):
                if not inst_name_base.upper().startswith("R"):
                    prefix = "R"
                    self.log_warning(f"Instance {pinst.name.name} uses resistor model but lacks R prefix. Prepending R_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "R"  # Ensure prefix is set even if name already has it
                skip_model_name = True
            elif model_name in ("capacitor", "cap", "c"):
                if not inst_name_base.upper().startswith("C"):
                    prefix = "C"
                    self.log_warning(f"Instance {pinst.name.name} uses capacitor model but lacks C prefix. Prepending C_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "C"  # Ensure prefix is set even if name already has it
                skip_model_name = True
            elif model_name in ("inductor", "ind", "l"):
                if not inst_name_base.upper().startswith("L"):
                    prefix = "L"
                    self.log_warning(f"Instance {pinst.name.name} uses inductor model but lacks L prefix. Prepending L_", f"Instance: {pinst.name.name}")
                else:
                    prefix = "L"  # Ensure prefix is set even if name already has it
                skip_model_name = True

        inst_name = inst_name_base
        
        if prefix and not inst_name.upper().startswith(prefix):
            # If we're skipping model name (resistor/capacitor/inductor), use underscore separator
            if skip_model_name:
                inst_name = f"{prefix}_{inst_name}"
            else:
                inst_name = f"{prefix}{inst_name}"
        self.write(inst_name + " \n")

        # Write ports (excluding last (model)) on a separate continuation line
        self.write("+ ")
        for arg in pinst.args[:-1]:  # Ports only (exclude the model)
            if isinstance(arg, Ident):
                self.write(self.format_ident(arg) + " ")
            elif isinstance(arg, Ref):
                # Ports are identifiers, not expressions - use format_ident to avoid braces
                self.write(self.format_ident(arg.ident) + " ")
            elif isinstance(arg, (Int, Float, MetricNum)):
                self.write(self.format_number(arg) + " ")
            else:
                # For other expression types, still format as expression (but this shouldn't happen for ports)
                self.write(self.format_expr(arg) + " ")
        self.write("\n")
        # Write the model (last arg) on another continuation line
        # Skip model name for resistor, capacitor, and inductor (Xyce infers from prefix)
        if not skip_model_name and pinst.args:
            self.write("+ " + self.format_ident(pinst.args[-1]) + " \n")

        # Filter deltox if referencing BSIM4 model
        # (dtox is only valid in model definitions, not instance parameters)
        kwargs_to_write = pinst.kwargs
        if module_ref and self._is_bsim4_model_ref(module_ref):
            # Filter deltox AND dtox from instance parameters
            kwargs_to_write = [kw for kw in pinst.kwargs if kw.name.name not in ("deltox", "dtox")]
        
        self.write("+ ")
        for kwarg in kwargs_to_write:
            self.write_param_val(kwarg)
            self.write(" ")

        # Add a blank line after each instance for spacing between instances
        # If next entry is a model, write_model_def will handle spacing (but won't add extra)
        self.write("\n")
        # Mark that last entry was an instance
        self._last_entry_was_instance = True

    def write_comment(self, comment: str) -> None:
        """Xyce comments *kinda* support the Spice-typical `*` charater,
        but *only* as the first character in a line.
        Any mid-line-starting comments must use `;` instead.
        So, just use it all the time."""
        self.write(f"; {comment}\n")

    def write_entry(self, entry) -> None:
        """Override write_entry to handle FunctionDef for Xyce and reset instance flag."""
        # Handle FunctionDef specially
        if isinstance(entry, FunctionDef):
            self._last_entry_was_instance = False
            return self.write_function_def(entry)
        
        # For ModelDef, let it check the flag first, then it will reset it
        if isinstance(entry, ModelDef):
            return self.write_model_def(entry)
        
        # For Instance and Primitive, let their write methods set the flag
        # For other entries, reset the flag before calling parent
        if not isinstance(entry, (Instance, Primitive)):
            self._last_entry_was_instance = False
        
        # Call parent implementation for other entries
        # This will call write_subckt_instance or write_primitive_instance which set the flag
        return super().write_entry(entry)

    def expression_delimiters(self) -> Tuple[str, str]:
        """Return the starting and closing delimiters for expressions."""
        return ("{", "}")

    def format_expr(self, expr: Expr) -> str:
        """Format an expression for Xyce.
        
        Overrides base implementation to ensure Ref objects (parameters) 
        are wrapped in curly braces, e.g. {m} instead of just m.
        """
        # Base cases that don't need braces
        if isinstance(expr, (Int, Float, MetricNum)):
            return self.format_number(expr)
        if isinstance(expr, QuotedString):
            return f"'{expr.val}'"

        # Everything else (Ref, BinaryOp, Call, etc.) gets wrapped in braces
        start, end = self.expression_delimiters()
        inner = self._format_expr_inner(expr)
        return f"{start}{inner}{end}"

    def write_options(self, options: Options) -> None:
        """Write Options `options`

        Xyce differs from many Spice-class simulators
        in categorizing options, and requiring that users known their category.
        Example categories include settings for the netlist parser,
        device models, and many, many solver configurations.

        The `netlist` AST also does not include this information,
        so valid options per-category are defined here.
        """

        if options.name is not None:
            msg = f"Warning invalid `name`d Options"
            self.log_warning(msg)
            self.write_comment(msg)
            self.write("\n")

        class Category(Enum):
            """Xyce-specific categories"""

            DEVICE = "device"
            PARSER = "parser"

        # Mapping from option-names to categories
        categories = {
            "scale": Category.PARSER,
            # FIXME: everything else
        }

        # Group the options by category
        by_category = {}
        for option in options.vals:
            name = self.format_ident(option.name)
            if name not in categories:
                msg = f"Unknown Xyce option `{name}`, cannot find `.options` category"
                self.handle_error(option, msg)
                continue
            category = categories[name].value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(option)

        for (category_name, category_list) in by_category.items():
            # Get to the actual option-writing
            self.write(f".options {category_name} \n")
            for option in category_list:
                self.write("+ ")
                # FIXME: add the cases included in `OptionVal` but not `ParamVal`, notably quoted string-paths
                self.write_param_val(option)
                self.write(" \n")
            self.write("\n")

    def write_subckt_def(self, module: SubcktDef) -> None:
        """Write the `SUBCKT` definition for `Module` `module` in Xyce format."""
        
        # Pre-process: Move deltox/dtox from instances to local BSIM4 models
        # Xyce requires dtox to be on the model card, not the instance.
        
        # 1. Find local BSIM4 models
        local_models = {}
        for entry in module.entries:
            if isinstance(entry, ModelDef):
                if entry.mtype.name.lower() == "bsim4":
                    local_models[entry.name.name] = entry
            elif isinstance(entry, ModelFamily):
                if entry.mtype.name.lower() == "bsim4":
                    local_models[entry.name.name] = entry
                    
        # 2. Find instances using these models and extract deltox
        for entry in module.entries:
            if isinstance(entry, Instance) and isinstance(entry.module, Ref):
                model_name = entry.module.ident.name
                if model_name in local_models:
                    # Found instance of local BSIM4 model
                    dtox_val = None
                    params_to_keep = []
                    # Check params for deltox or dtox
                    for pval in entry.params:
                        if pval.name.name in ("deltox", "dtox"):
                            dtox_val = pval.val
                        else:
                            params_to_keep.append(pval)
                    
                    if dtox_val is not None:
                        # Remove from instance (modify in place)
                        entry.params = params_to_keep
                        
                        # Add to model (modify in place)
                        model = local_models[model_name]
                        
                        # Helper to add dtox to a param list
                        def add_dtox(params_list, val):
                            # Remove existing dtox/deltox
                            new_params = [p for p in params_list if p.name.name not in ("deltox", "dtox")]
                            # Add new dtox
                            new_params.append(ParamDecl(name=Ident("dtox"), default=val, distr=None))
                            return new_params

                        if isinstance(model, ModelDef):
                            model.params = add_dtox(model.params, dtox_val)
                        elif isinstance(model, ModelFamily):
                            for variant in model.variants:
                                variant.params = add_dtox(variant.params, dtox_val)

        # Reset instance flag at start of subcircuit
        self._last_entry_was_instance = False
        # Store current subcircuit context for use in write_subckt_instance
        self._current_subckt = module

        # Create the module name
        module_name = self.format_ident(module.name)

        # Check if any entries reference 'm' parameter and ensure it exists in subcircuit params
        has_m_param = any(p.name.name == "m" for p in module.params)
        needs_m_param = False
        
        # First check the subcircuit's own parameter declarations for 'm' references
        for param in module.params:
            if param.default is not None and isinstance(param.default, Expr):
                if expr_references_param(param.default, "m"):
                    needs_m_param = True
                    break
        
        # Check all entries for 'm' references
        for entry in module.entries:
            if isinstance(entry, Instance):
                # Check if any instance parameter value references 'm'
                for pval in entry.params:
                    if isinstance(pval.val, Expr):
                        if expr_references_param(pval.val, "m"):
                            needs_m_param = True
                            break
                if needs_m_param:
                    break
            elif isinstance(entry, ParamDecls):
                # Check if any parameter declaration references 'm' in its default value
                for param in entry.params:
                    if param.default is not None and isinstance(param.default, Expr):
                        if expr_references_param(param.default, "m"):
                            needs_m_param = True
                            break
                if needs_m_param:
                    break
            elif isinstance(entry, ModelDef):
                # Check if any model parameter references 'm'
                for param in entry.params:
                    if param.default is not None and isinstance(param.default, Expr):
                        if expr_references_param(param.default, "m"):
                            needs_m_param = True
                            break
                if needs_m_param:
                    break
        
        # Add m=1 parameter if needed but not present
        if needs_m_param and not has_m_param:
            m_param = ParamDecl(name=Ident("m"), default=Float(1.0), distr=None)
            module.params.insert(0, m_param)

        # Start the sub-circuit definition header with inline ports
        self.write(f".SUBCKT {module_name}")
        if module.ports:
            for port in module.ports:
                self.write(f" {self.format_ident(port)}")
        self.write("\n")

        # Add parameters on a continuation line
        if module.params:
            self.write_module_params(module.params)
        else:
            self.write("+ ")
            self.write_comment("No parameters")

        # End the header with a blank line
        self.write("\n")

        # Write internal content
        for entry in module.entries:
            self.write_entry(entry)
        
        # Reset instance flag after subcircuit ends
        self._last_entry_was_instance = False

        # Close up the sub-circuit
        self.write(".ENDS\n\n")
        # Clear current subcircuit context
        self._current_subckt = None

    def write_function_def(self, func: FunctionDef) -> None:
        """Write a Xyce .FUNC definition for FunctionDef `func`.

        Format: .FUNC func_name(arg1, arg2, ...) { expression }
        """
        func_name = self.format_ident(func.name)

        # Write function signature
        self.write(f".FUNC {func_name}(")

        # Write arguments
        if func.args:
            arg_names = [self.format_ident(arg.name) for arg in func.args]
            self.write(",".join(arg_names))

        self.write(") { ")

        # Write function body - should be a single Return statement
        if func.stmts and len(func.stmts) == 1 and isinstance(func.stmts[0], Return):
            expr_str = self.format_expr(func.stmts[0].val)
            self.write(expr_str)
        else:
            # Fallback: handle multiple statements (shouldn't happen for our use case)
            self.handle_error(func, "FunctionDef with multiple statements not supported")

        self.write(" }\n")

    def _resolve_ref_to_definition(self, ref: Ref) -> Optional[Union[ModelDef, ModelFamily, SubcktDef]]:
        """Resolve a reference to its definition (Model, Subckt, etc.)."""
        from ..data import ModelDef, ModelFamily, SubcktDef, LibSectionDef

        # Check if already resolved
        if ref.resolved is not None:
            if isinstance(ref.resolved, (ModelDef, ModelFamily, SubcktDef)):
                return ref.resolved

        target_name = ref.ident.name
        found = None

        def check_entry(entry) -> bool:
            nonlocal found
            if isinstance(entry, (ModelDef, ModelFamily, SubcktDef)):
                if entry.name.name == target_name:
                    found = entry
                    return True
            return False

        # Search through all files in the program
        for file in self.src.files:
            for entry in file.contents:
                if check_entry(entry):
                    return found
                if isinstance(entry, LibSectionDef):
                    for sub in entry.entries:
                        if check_entry(sub):
                            return found
                elif isinstance(entry, SubcktDef):
                    # Subckts can contain models
                    if check_entry(entry):
                        return found
                    for sub in entry.entries:
                        if check_entry(sub):
                            return found

        return None

    def _get_param_default_from_definition(self, definition: Union[ModelDef, ModelFamily, SubcktDef], param_name: str) -> Optional[Expr]:
        """Get the default value for a parameter from a definition."""
        from ..data import ModelDef, ModelFamily, SubcktDef
        
        # DEBUG: Uncomment to enable debug output
        # def_type = type(definition).__name__
        # def_name = definition.name.name if hasattr(definition, 'name') else 'unknown'
        # print(f"[DEBUG] _get_param_default_from_definition: {def_name} (type: {def_type}), param={param_name}")
        
        params = []
        if isinstance(definition, (ModelDef, SubcktDef)):
            params = definition.params
        elif isinstance(definition, ModelFamily):
            # Check all variants, not just the first
            for variant in definition.variants:
                for p in variant.params:
                    if p.name.name == param_name:
                        return p.default
            return None
        
        for p in params:
            if p.name.name == param_name:
                # DEBUG: Uncomment to enable debug output
                # print(f"[DEBUG]   Found param {param_name}: default={p.default}")
                # if p.default is not None:
                #     is_numeric = self._is_number(p.default)
                #     print(f"[DEBUG]     Default is numeric: {is_numeric}")
                return p.default
        
        # DEBUG: Uncomment to enable debug output
        # print(f"[DEBUG]   Param {param_name} not found in definition")
        return None

    def _is_number(self, val: Expr) -> bool:
        """Check if an expression is a numeric literal."""
        from ..data import Int, Float, MetricNum
        return isinstance(val, (Int, Float, MetricNum))

    def _get_scale_factor(self) -> float:
        """Get the scale factor from Options in the program.
        
        Returns the scale factor as a float (e.g., 1e-6 for '1.0u').
        Defaults to 1e-6 (microns) if not found.
        """
        from ..data import Options, MetricNum, Float, Int

        # Metric suffix to multiplier mapping
        suffix_map = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
            'M': 1e-3, 'MIL': 2.54e-5, 'U': 1e-6, 'u': 1e-6,
            'N': 1e-9, 'n': 1e-9, 'P': 1e-12, 'p': 1e-12,
            'F': 1e-15, 'f': 1e-15, 'A': 1e-18, 'a': 1e-18
        }

        # Search through all files for Options entries
        for file in self.src.files:
            for entry in file.contents:
                if isinstance(entry, Options):
                    for option in entry.vals:
                        if option.name.name.lower() == "scale":
                            scale_val = option.val
                            if isinstance(scale_val, MetricNum):
                                # Parse metric suffix (e.g., "1.0u" -> 1e-6)
                                val_str = scale_val.val
                                match = re.match(r'([\d.]+)([a-zA-Z]+)?', val_str)
                                if match:
                                    num = float(match.group(1))
                                    suffix = match.group(2) or ''
                                    multiplier = suffix_map.get(suffix.upper(), 1.0)
                                    return num * multiplier
                                return float(val_str) if val_str else 1e-6
                            elif isinstance(scale_val, (Float, Int)):
                                return float(scale_val.val) if hasattr(scale_val, 'val') else float(scale_val)

        # Default to microns (1e-6) if scale not found
        return 1e-6

    def _find_param_default_from_usage(self, param_name: str, subckt: SubcktDef) -> Optional[Expr]:
        """Try to find a default value for a parameter by looking at how it's used in the subcircuit."""
        from ..data import Instance, Ref
        
        # DEBUG: Uncomment to enable debug output
        # print(f"[DEBUG] _find_param_default_from_usage: subckt={subckt.name.name}, param={param_name}")
        
        for entry in subckt.entries:
            if isinstance(entry, Instance):
                # Check if this instance uses the parameter
                target_param_name = None
                if entry.params:
                    for pval in entry.params:
                        # Check if value is a reference to our param_name
                        if isinstance(pval.val, Ref) and pval.val.ident.name == param_name:
                             target_param_name = pval.name.name
                             break
                
                if target_param_name:
                    # Found usage. Now resolve model.
                    model = self._resolve_ref_to_definition(entry.module)
                    if model:
                        # Look for default in model
                        default = self._get_param_default_from_definition(model, target_param_name)
                        if default and self._is_number(default):
                            return default
        # DEBUG: Uncomment to enable debug output
        # print(f"[DEBUG]   No default recovered for param {param_name}")
        return None

    def write_param_decl(self, param: ParamDecl) -> str:
        """Format a parameter declaration, with special handling for param functions in Xyce."""
        if param.distr is not None:
            msg = f"Unsupported `distr` for parameter {param.name} will be ignored"
            self.log_warning(msg, f"Parameter: {param.name.name}")
            self.write("\n+ ")
            self.write_comment(msg)
            self.write("\n+ ")

        param_name = self.format_ident(param.name)

        # Check if this is a param function (name contains parentheses)
        if '(' in param_name and param_name.endswith(')'):
            # This is a param function like lnorm(mu,sigma) or mm_z1__mismatch__(dummy_param)
            if param.default is None:
                msg = f"Required (non-default) param function {param.name} is not supported."
                self.log_warning(msg, f"Parameter: {param.name.name}")
                default = str(sys.float_info.max)
            else:
                # For param functions, use = for lnorm, ' quotes for mismatch
                default = param.default
                if isinstance(default, str):
                    # Already formatted
                    pass
                else:
                    default = self.format_expr(default)
                # Remove outer braces/quotes if present
                if (default.startswith('{') and default.endswith('}')) or (default.startswith("'") and default.endswith("'")):
                    default = default[1:-1]

                # Use single quotes for all param functions
                self.write(f"{param_name}='{default}'")
                return

        # Normal parameter declaration
        if param.default is None:
            # Try to recover default from usage in current subcircuit
            recovered_default = None
            if getattr(self, '_current_subckt', None):
                recovered_default = self._find_param_default_from_usage(param.name.name, self._current_subckt)

            if recovered_default:
                default = self.format_expr(recovered_default)
            else:
                # Safety net: Use common defaults for geometric parameters
                # Adjust based on scale factor (defaults assume microns, i.e., scale=1e-6)
                scale_factor = self._get_scale_factor()
                # Default values in microns: l=1u, w=1u, perim=4u, area=1u^2
                # Convert to the actual scale units
                micron_to_scale = 1e-6 / scale_factor

                common_defaults = {
                    'l': Float(1.0 * micron_to_scale),  # length
                    'w': Float(1.0 * micron_to_scale),  # width
                    'perim': Float(4.0 * micron_to_scale),  # perimeter (typical for square)
                    'area': Float(1.0 * micron_to_scale * micron_to_scale),  # area (scale^2)
                }

                if param.name.name.lower() in common_defaults:
                    default = self.format_expr(common_defaults[param.name.name.lower()])
                else:
                    msg = f"Required (non-default) parameter {param.name} is not supported by {self.__class__.__name__}. "
                    msg += f"Setting to maximum floating-point value {sys.float_info.max}, which almost certainly will not work if instantiated."
                    self.log_warning(msg, f"Parameter: {param.name.name}")
                    default = str(sys.float_info.max)
        else:
            default = self.format_expr(param.default)

        self.write(f"{param_name}={default}")
        
        # Write inline comment if present
        if param.comment:
            self.write(f" ; {param.comment}")

    def _is_bsim4_model_ref(self, ref: Ref) -> bool:
        """Check if a Ref points to a BSIM4 model definition.
        
        First checks if ref.resolved is already set (more direct).
        Otherwise searches through the program recursively, including library sections.
        
        Handles:
        - Direct ModelDef matches (exact name)
        - ModelFamily base name matches (instance references base, model has variants)
        - ModelVariant matches (standalone variants)
        - Models nested in LibSectionDef entries
        
        Args:
            ref: A Ref object pointing to a model name
            
        Returns:
            True if the reference points to a BSIM4 model, False otherwise
        """
        if not isinstance(ref, Ref):
            return False
        
        model_name = ref.ident.name
        
        # First, check if ref.resolved is already set (more direct and reliable)
        if ref.resolved is not None:
            from ..data import Model, ModelDef, ModelFamily
            if isinstance(ref.resolved, Model):
                if isinstance(ref.resolved, ModelDef):
                    return ref.resolved.mtype.name.lower() == "bsim4"
                elif isinstance(ref.resolved, ModelFamily):
                    return any(v.mtype.name.lower() == "bsim4" for v in ref.resolved.variants)
            return False
        
        # If not resolved, search through the program
        model_name = ref.ident.name  # e.g., "plowvt_model"
        
        # Import model types for use in nested function
        from ..data import ModelDef, ModelFamily, ModelVariant, LibSectionDef, SubcktDef
        
        def check_entry(entry) -> bool:
            """Helper to check if an entry is a BSIM4 model matching model_name."""
            if isinstance(entry, ModelDef):
                if entry.name.name == model_name:
                    return entry.mtype.name.lower() == "bsim4"
            elif isinstance(entry, ModelFamily):
                if entry.name.name == model_name:
                    return any(v.mtype.name.lower() == "bsim4" for v in entry.variants)
            elif isinstance(entry, ModelVariant):
                variant_name = f"{entry.model.name}.{entry.variant.name}"
                if variant_name == model_name or entry.model.name == model_name:
                    return entry.mtype.name.lower() == "bsim4"
            return False
        
        # Search through all files in the program
        for file in self.src.files:
            for entry in file.contents:
                if check_entry(entry):
                    return True
                # Also search inside library sections recursively
                elif isinstance(entry, LibSectionDef):
                    for lib_entry in entry.entries:
                        if check_entry(lib_entry):
                            return True
                # Also search inside subcircuit definitions recursively
                elif isinstance(entry, SubcktDef):
                    for subckt_entry in entry.entries:
                        if check_entry(subckt_entry):
                            return True
        
        return False

    def _clamp_param_value(self, expr: Expr, min_val: Optional[float], max_val: Optional[float], 
                          min_exclusive: bool = False, max_exclusive: bool = False) -> Expr:
        """Clamp a parameter value expression to valid range.
        
        Args:
            expr: The expression to clamp (Int, Float, or MetricNum)
            min_val: Minimum value (None for unbounded)
            max_val: Maximum value (None for unbounded)
            min_exclusive: If True, min is exclusive (use min + epsilon)
            max_exclusive: If True, max is exclusive (use max - epsilon)
            
        Returns:
            Clamped Expr (preserves original type), or original if not a numeric literal
        """
        from ..data import Int, Float, MetricNum
        
        # Only clamp numeric literals, not expressions
        if not isinstance(expr, (Int, Float, MetricNum)):
            return expr
        
        # Extract numeric value
        if isinstance(expr, Int):
            val = float(expr.val)
        elif isinstance(expr, Float):
            val = expr.val
        elif isinstance(expr, MetricNum):
            # Parse metric suffix
            val_str = expr.val
            match = re.match(r'([\d.]+)([a-zA-Z]+)?', val_str)
            if match:
                num = float(match.group(1))
                suffix = match.group(2) or ''
                suffix_map = {
                    'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
                    'M': 1e-3, 'MIL': 2.54e-5, 'U': 1e-6, 'u': 1e-6,
                    'N': 1e-9, 'n': 1e-9, 'P': 1e-12, 'p': 1e-12,
                    'F': 1e-15, 'f': 1e-15, 'A': 1e-18, 'a': 1e-18
                }
                multiplier = suffix_map.get(suffix.upper(), 1.0)
                val = num * multiplier
            else:
                val = float(val_str) if val_str else 0.0
        else:
            return expr
        
        # Clamp to range
        # Small epsilon for exclusive bounds (consistent with expression wrapping)
        EPSILON = 1e-12
        clamped_val = val
        
        if min_val is not None:
            if min_exclusive:
                if val <= min_val:
                    clamped_val = min_val + EPSILON
            else:
                if val < min_val:
                    clamped_val = min_val
        
        if max_val is not None:
            if max_exclusive:
                if val >= max_val:
                    clamped_val = max_val - EPSILON
            else:
                if val > max_val:
                    clamped_val = max_val
        
        # Return new Expr with clamped value
        # If value changed (even slightly), use Float to preserve precision
        # This is especially important for exclusive bounds where we add epsilon
        if abs(clamped_val - val) >= 1e-12:
            # Value changed, use Float to preserve precision
            return Float(clamped_val)
        elif isinstance(expr, Int):
            return Int(int(clamped_val))
        elif isinstance(expr, Float):
            return Float(clamped_val)
        else:  # MetricNum - convert to Float since we can't preserve suffix easily
            return Float(clamped_val)

    def _map_bjt_level1_to_mextram_params(self, params: List[ParamDecl], model_name: str = "") -> List[Tuple[ParamDecl, Optional[str], bool]]:
        """Map BJT level 1 (Gummel-Poon) parameters to MEXTRAM (level 504) parameters.
        
        Args:
            params: List of ParamDecl objects with level 1 parameter names
            model_name: Name of the model being converted (for warning context)
            
        Returns:
            List of (ParamDecl, comment, commented_out) tuples
        """
        # Track warnings for this conversion
        commented_params = []
        mapped_params_list = []
        clamped_params = []
        # Mapping table: (level1_param_name, mextram_param_name_or_action, optional_range_tuple)
        # Actions can be:
        # - 'PARAM': Direct mapping to new name
        # - None: Drop parameter (comment out)
        # - (None, 'Warning'): Drop with custom warning/comment
        # - ('PARAM', 'Warning'): Map with warning/comment
        # - 'SPECIAL_...': Special handling required
        # Range tuple format: (min_val, max_val, min_exclusive=False, max_exclusive=False)
        # Use None for unbounded ends
        
        level1_to_mextram_mapping = [
            # === Core transport parameters ===
            ('IS',   'IS'),     # saturation current – direct
            ('BF',   'BF', (0.0001, None)),  # forward beta – direct, range: [0.0001, +inf)
            ('BR',   'IBR'),    # reverse beta → MEXTRAM uses IBR
            ('NF',   'SPECIAL_NF'),  # forward emission coeff -> PE + MLF
            ('NR',   'PC'),     # reverse emission coeff -> PC
            ('VAF',  'VEF', (0.01, None)),  # forward Early voltage -> VEF, range: [0.01, +inf)
            ('VAR',  'VER', (0.01, None)),  # reverse Early voltage -> VER, range: [0.01, +inf)
            ('IKF',  'SPECIAL_IK'),  # forward knee -> IK (conflict with IKR)
            ('IKR',  'SPECIAL_IK'),  # reverse knee -> IK (conflict with IKF)
            ('ISE',  'IBF'),    # B-E leakage sat current -> IBF
            ('ISC',  'ICSS'),   # B-C leakage sat current -> ICSS
            ('NE',   'MLF'),    # B-E leakage emission coeff -> MLF
            ('NC',   'PS', (0.01, 0.99, True, True)),  # B-C leakage emission coeff -> PS, range: ]0.01, 0.99[ (conflict with NS)
            
            # === Resistances ===
            ('RB',   'RBV'),    # parasitic base resistance -> RBV (variable part)
            ('RBM',  'RBC', (0.001, None)),  # minimum intrinsic base resistance -> RBC, range: [0.001, +inf)
            ('RBI',  'SPECIAL_RBI'), # distribute to RBV + RBC
            ('IRB',  (None, "Unsupported; no direct IRB equivalent")),
            ('RE',   'RE'),     # emitter resistance
            ('RC',   'RCC'),    # collector resistance -> RCC
            
            # === Capacitances & junction parameters ===
            ('CJE',  'CJE'),    # B-E capacitance
            ('VJE',  'VDE'),    # B-E potential -> VDE
            ('MJE',  'PE'),     # B-E grading -> PE
            ('CJC',  'CJC'),    # B-C capacitance
            ('VJC',  'VDC'),    # B-C potential -> VDC
            ('MJC',  'PC'),     # B-C grading -> PC
            ('XCJC', 'XEXT'),   # fraction of CJC external
            ('FC',   'MC'),     # forward-bias coeff -> MC
            ('CJS',  'CJS'),    # substrate capacitance
            ('VJS',  'VDS'),    # substrate potential -> VDS
            ('MJS',  'PS', (0.01, 0.99, True, True)),  # substrate grading -> PS, range: ]0.01, 0.99[
            ('ISS',  'ISS'),    # substrate saturation current
            ('NS',   'PS', (0.01, 0.99, True, True)),  # substrate emission coeff -> PS, range: ]0.01, 0.99[ (conflict with NC)
            
            # === Transit times ===
            ('TF',   'TAUB', (0.0, None, True, False)),  # forward transit time -> TAUB, range: ]0.0, +inf)
            ('TR',   'TAUR'),   # reverse transit time -> TAUR
            ('ITF',  (None, "Unsupported; IK handles high-injection knee")),
            ('VTF',  (None, "Unsupported; no VTF threshold in MEXTRAM")),
            ('XTF',  (None, "Unsupported; no exponential TF bias in MEXTRAM")),
            ('PTF',  (None, "Unsupported; no quadratic TF scaling in MEXTRAM")),
            
            # === Temperature coefficients ===
            ('XTI',  (None, "XIS not supported in Xyce MEXTRAM")),
            ('XTB',  (None, "TEF not supported in Xyce MEXTRAM")),
            ('TREF', 'TREF'),   # reference temp
            
            # === Noise ===
            ('KF',   'KF'),     # flicker noise coeff
            ('AF',   'AF'),     # flicker noise exponent
            
            # === Unsupported / Spectre-only ===
            ('EG',   (None, "Bandgap voltage - MEXTRAM uses fixed 1.11 eV Si")),
            ('GAP1', (None, "Spectre-only parameter")),
            ('GAP2', (None, "Spectre-only parameter")),
            ('DCAP', (None, "Spectre overlap cap model - no equivalent")),
            ('SUBS', (None, "Substrate connection parameter - not in MEXTRAM")),
            ('IBC',  (None, "Base-collector leakage current - not directly supported")),
            ('NKF',  (None, "High-injection parameter - not in MEXTRAM")),
            
            # === Temperature params (unsupported) ===
            ('CTC',  None), ('CTE',  None), ('CTS',  None),
            ('TLEV', None), ('TLEVC', None),
            ('TVJC', None), ('TVJE', None), ('TVJS', None),
            
            # === Spectre temp coefficients (order 1 & 2) ===
            ('TIS1', None), ('TISE1', None), ('TISC1', None), ('TNF1', None), ('TNR1', None),
            ('TNE1', None), ('TNC1', None), ('TBF1', None), ('TBR1', None), ('TISS1', None),
            ('TVAF1', None), ('TVAR1', None), ('TIKF1', None), ('TIKR1', None), ('TNS1', None),
            ('TRB1', None), ('TRC1', None), ('TRE1', None), ('TIRB1', None), ('TRM1', None),
            ('TMJC1', None), ('TMJE1', None), ('TMJS1', None), ('TTF1', None), ('TITF1', None), ('TTR1', None),
            ('TIS2', None), ('TISE2', None), ('TISC2', None), ('TNF2', None), ('TNR2', None),
            ('TNE2', None), ('TNC2', None), ('TBF2', None), ('TBR2', None), ('TISS2', None),
            ('TVAF2', None), ('TVAR2', None), ('TIKF2', None), ('TIKR2', None), ('TNS2', None),
            ('TRB2', None), ('TRC2', None), ('TRE2', None), ('TIRB2', None), ('TRM2', None),
            ('TMJC2', None), ('TMJE2', None), ('TMJS2', None), ('TTF2', None), ('TITF2', None), ('TTR2', None),
        ]
        
        # Convert to dictionary for efficient lookup
        # Format: (level1_name, action, optional_range_tuple)
        param_map = {}
        range_map = {}  # Store range info separately
        for mapping_entry in level1_to_mextram_mapping:
            level1_name = mapping_entry[0]
            action = mapping_entry[1]
            param_map[level1_name.upper()] = action
            # Check if there's a range tuple (3rd element)
            if len(mapping_entry) >= 3:
                range_map[level1_name.upper()] = mapping_entry[2]
            
        # Pre-scan params to handle conflicts/merges (IK, NF, RBI)
        params_dict = {p.name.name.upper(): p for p in params}
        
        # Special handling values
        ikf_val = params_dict.get('IKF')
        ikr_val = params_dict.get('IKR')
        nf_val = params_dict.get('NF')
        rbi_val = params_dict.get('RBI')
        ns_val = params_dict.get('NS')
        nc_val = params_dict.get('NC')
        
        mapped_params = [] # List of (ParamDecl, comment, commented_out)
        processed_names = set()
        
        # Helper to add param if not already added
        def add_param(name, default, distr=None, comment=None, commented_out=False):
            if name.upper() not in processed_names:
                mapped_params.append((
                    ParamDecl(name=Ident(name), default=default, distr=distr),
                    comment,
                    commented_out
                ))
                processed_names.add(name.upper())
        
        # 1. Handle SPECIAL_IK (IKF/IKR merge)
        if ikf_val or ikr_val:
            # If both present, use average (simple heuristic) or prefer IKF
            val_to_use = None
            src_name = ""
            if ikf_val and ikr_val:
                val_to_use = ikf_val.default
                src_name = "IKF/IKR"
            elif ikf_val:
                val_to_use = ikf_val.default
                src_name = "IKF"
            elif ikr_val:
                val_to_use = ikr_val.default
                src_name = "IKR"
            
            if val_to_use:
                add_param('IK', val_to_use, comment=f"{src_name} -> IK (lvl 1 -> lvl 504)")
                
        # 2. Handle SPECIAL_NF (NF -> PE + MLF)
        if nf_val:
            # Check value if possible
            nf_num = None
            if isinstance(nf_val.default, (Int, Float)):
                nf_num = float(nf_val.default.val) if hasattr(nf_val.default, 'val') else float(nf_val.default)
            elif isinstance(nf_val.default, MetricNum):
                nf_num = float(nf_val.default.val)
                
            # Always map to MLF
            add_param('MLF', nf_val.default, comment="NF -> MLF (lvl 1 -> lvl 504)")
            
            # Map to PE only if < 0.5 (approximate grading)
            if nf_num is not None and nf_num < 0.5:
                add_param('PE', nf_val.default, comment="NF -> PE (lvl 1 -> lvl 504, NF < 0.5)")
                
        # 3. Handle SPECIAL_RBI (RBI -> RBV + RBC)
        if rbi_val:
            # Set RBV = RBI, RBC = 0 (or leave RBC if defined elsewhere)
            add_param('RBV', rbi_val.default, comment="RBI -> RBV (lvl 1 -> lvl 504)")
            
        # 4. Handle NS vs NC for PS (conflict)
        if ns_val or nc_val:
            # Prefer NS for PS if both present
            val_to_use = ns_val.default if ns_val else nc_val.default
            src_name = "NS" if ns_val else "NC"
            # Apply range clamping/wrapping for PS: ]0.01, 0.99[
            if self._is_number(val_to_use):
                clamped_val = self._clamp_param_value(val_to_use, 0.01, 0.99, min_exclusive=True, max_exclusive=True)
            else:
                # Expression - wrap in max()/min() to enforce bounds
                EPSILON = 1e-12
                min_expr = Float(0.01 + EPSILON)  # Exclusive min
                max_expr = Float(0.99 - EPSILON)  # Exclusive max
                # Wrap: min(max(expr, min_val), max_val)
                max_call = Call(func=Ref(ident=Ident("max")), args=[val_to_use, min_expr])
                clamped_val = Call(func=Ref(ident=Ident("min")), args=[max_call, max_expr])
            add_param('PS', clamped_val, comment=f"{src_name} -> PS (lvl 1 -> lvl 504)")

        # Process all parameters
        for param in params:
            name_upper = param.name.name.upper()
            
            # Skip if special handled and consumed
            if name_upper in ('IKF', 'IKR', 'NF', 'RBI', 'NS', 'NC'):
                # These are now effectively "commented out" in the new scheme by not being added directly?
                # No, we want to keep them in the list but commented out if they weren't mapped?
                # Actually, the SPECIAL handlers added the new parameters. 
                # We should perhaps add the original ones as commented out?
                # For now, let's assume consumed means replaced.
                continue
                
            if name_upper in param_map:
                action = param_map[name_upper]
                
                if action is None:
                    # Drop (comment out)
                    commented_params.append(param.name.name)
                    mapped_params.append((param, "Dropped: No MEXTRAM equivalent", True))
                    continue
                    
                if isinstance(action, tuple):
                    new_name, warning_msg = action
                    if new_name is None:
                        # Comment out with custom message
                        commented_params.append(f"{param.name.name} ({warning_msg})")
                        mapped_params.append((param, f"Dropped: {warning_msg}", True))
                        continue
                    else:
                        # Map with warning/comment
                        # Check for range clamping
                        default_val = param.default
                        if name_upper in range_map:
                            range_info = range_map[name_upper]
                            if len(range_info) >= 2:
                                min_val = range_info[0]
                                max_val = range_info[1]
                                min_exclusive = range_info[2] if len(range_info) > 2 else False
                                max_exclusive = range_info[3] if len(range_info) > 3 else False
                                
                                # Clamp numeric literals or wrap expressions
                                if default_val is None:
                                    clamped_default = min_val + (1e-12 if min_exclusive else 0)
                                    default_val = Float(clamped_default)
                                elif self._is_number(default_val):
                                    clamped = self._clamp_param_value(default_val, min_val, max_val, min_exclusive, max_exclusive)
                                    if self._is_number(clamped):
                                        if isinstance(default_val, (Int, Float)) and isinstance(clamped, (Int, Float)):
                                            old_val = default_val.val if hasattr(default_val, 'val') else default_val
                                            new_val = clamped.val if hasattr(clamped, 'val') else clamped
                                            if abs(old_val - new_val) >= 1e-12:
                                                clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val})")
                                                default_val = clamped
                                        else:
                                            default_val = clamped
                                else:
                                    # Expression - wrap in max()/min() to enforce bounds
                                    EPSILON = 1e-12
                                    if min_val is not None:
                                        min_expr = Float(min_val + (EPSILON if min_exclusive else 0))
                                        default_val = Call(func=Ref(ident=Ident("max")), args=[default_val, min_expr])
                                    if max_val is not None:
                                        max_expr = Float(max_val - (EPSILON if max_exclusive else 0))
                                        default_val = Call(func=Ref(ident=Ident("min")), args=[default_val, max_expr])
                        add_param(new_name, default_val, param.distr, comment=f"{param.name.name} -> {new_name} (lvl 1 -> lvl 504). {warning_msg}")
                        
                elif isinstance(action, str):
                    if action.startswith('SPECIAL_'):
                        continue # Already handled
                    new_name = action
                    
                    # Apply range clamping if range info exists
                    default_val = param.default
                    if name_upper in range_map:
                        range_info = range_map[name_upper]
                        if len(range_info) >= 2:
                            min_val = range_info[0]
                            max_val = range_info[1]
                            min_exclusive = range_info[2] if len(range_info) > 2 else False
                            max_exclusive = range_info[3] if len(range_info) > 3 else False
                            
                            # If default is None, provide a clamped default value
                            if default_val is None:
                                # Use minimum value as default (with epsilon for exclusive)
                                clamped_default = min_val + (1e-12 if min_exclusive else 0)
                                default_val = Float(clamped_default)
                            else:
                                clamped = self._clamp_param_value(default_val, min_val, max_val, min_exclusive, max_exclusive)
                                # Only use clamped value if it's a numeric literal (clamping worked)
                                if self._is_number(clamped):
                                    # Check if value actually changed (use >= to catch epsilon changes)
                                    if isinstance(default_val, (Int, Float)) and isinstance(clamped, (Int, Float)):
                                        old_val = default_val.val if hasattr(default_val, 'val') else default_val
                                        new_val = clamped.val if hasattr(clamped, 'val') else clamped
                                        if abs(old_val - new_val) >= 1e-12:
                                            clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val})")
                                            default_val = clamped
                                    else:
                                        default_val = clamped
                                elif self._is_number(default_val):
                                    # If clamping didn't produce a number but original was, keep original
                                    pass
                                else:
                                    # Neither is a number, but we have a range constraint
                                    # Wrap expression in max()/min() to ensure it stays within bounds
                                    if not self._is_number(default_val):
                                        EPSILON = 1e-12  # Consistent with _clamp_param_value
                                        if min_val is not None:
                                            # Create max(expr, min_val) to ensure minimum value
                                            min_expr = Float(min_val + (EPSILON if min_exclusive else 0))
                                            default_val = Call(func=Ref(ident=Ident("max")), args=[default_val, min_expr])
                                        if max_val is not None:
                                            # Create min(expr, max_val) to ensure maximum value
                                            max_expr = Float(max_val - (EPSILON if max_exclusive else 0))
                                            default_val = Call(func=Ref(ident=Ident("min")), args=[default_val, max_expr])
                                    else:
                                        default_val = clamped
                    
                    # Suppress comment if rename is only capitalization change
                    comment = f"{param.name.name} -> {new_name} (lvl 1 -> lvl 504)"
                    if param.name.name.lower() == new_name.lower():
                        comment = None
                    else:
                        mapped_params_list.append(f"{param.name.name} -> {new_name}")
                        
                    add_param(new_name, default_val, param.distr, comment=comment)
                else:
                    # Fallback
                    commented_params.append(f"{param.name.name} (Unknown mapping action)")
                    mapped_params.append((param, "Dropped: Unknown mapping action", True))
            else:
                # Not in mapping, keep as-is
                add_param(param.name.name, param.default, param.distr)
        
        # Log warnings about the conversion
        context = f"Model: {model_name}" if model_name else None
        if commented_params:
            self.log_warning(
                f"BJT Level 1 -> MEXTRAM conversion: {len(commented_params)} parameter(s) commented out (not compatible with Xyce MEXTRAM): {', '.join(commented_params)}",
                context
            )
        if mapped_params_list:
            self.log_warning(
                f"BJT Level 1 -> MEXTRAM conversion: {len(mapped_params_list)} parameter(s) renamed: {', '.join(mapped_params_list)}",
                context
            )
        if clamped_params:
            self.log_warning(
                f"BJT Level 1 -> MEXTRAM conversion: {len(clamped_params)} parameter value(s) clamped to valid range: {', '.join(clamped_params)}",
                context
            )
        
        return mapped_params

    def _get_param_name(self, item: Union[ParamDecl, Tuple[ParamDecl, Optional[str], bool]]) -> str:
        """Helper to get param name safely whether it's a ParamDecl or (ParamDecl, ...) tuple."""
        if isinstance(item, tuple):
            return item[0].name.name
        return item.name.name
        
    def _get_param_default(self, item: Union[ParamDecl, Tuple[ParamDecl, Optional[str], bool]]) -> Optional[Expr]:
        """Helper to get param default value safely."""
        if isinstance(item, tuple):
            return item[0].default
        return item.default

    def write_model_def(self, model: ModelDef) -> None:
        """Write a model definition in Xyce format, handling BSIM4 conversions."""
        # Helper to get param name safely whether it's a ParamDecl or (ParamDecl, ...) tuple
        def get_param_name(item):
            if isinstance(item, tuple):
                return item[0].name.name
            return item.name.name
            
        # Helper to get param default value safely
        def get_param_default(item):
            if isinstance(item, tuple):
                return item[0].default
            return item.default

        # Add blank line before model if last entry was an instance
        if self._last_entry_was_instance:
            self.write("\n")
            self._last_entry_was_instance = False
        
        mname = self.format_ident(model.name)
        mtype = self.format_ident(model.mtype).lower()
        
        # Map model types to Xyce equivalents
        xyce_mtype_map = {
            "diode": "D",
            "npn": "NPN",
            "pnp": "PNP",
            "r": "R",
            "res": "R",
            "resistor": "R",
            "c": "C",
            "cap": "C",
            "capacitor": "C",
            "d": "D",
        }
        
        # Use local variable for params to avoid mutating the model object
        params_to_write = list(model.params)  # Create a copy
        
        # Check for model level mapping (specific model name or device type)
        level_mapping_applied = False
        output_level = None
        if self._model_level_mapping:
            model_name_lower = model.name.name.lower()
            mtype_lower = mtype.lower()
            
            # First check for specific model name mapping, then device type mapping
            mapping_dict = None
            if model_name_lower in self._model_level_mapping:
                mapping_dict = self._model_level_mapping[model_name_lower]
            elif mtype_lower in self._model_level_mapping:
                mapping_dict = self._model_level_mapping[mtype_lower]
            
            if mapping_dict:
                # Extract current level (default to 1 if not specified)
                current_level = 1
                has_level = False
                for p in params_to_write:
                    if p.name.name.lower() == "level":
                        has_level = True
                        if isinstance(p.default, (Int, Float)):
                            current_level = int(float(p.default.val) if hasattr(p.default, 'val') else float(p.default))
                        elif isinstance(p.default, MetricNum):
                            current_level = int(float(p.default.val))
                        break
                
                # Check if current level has a mapping
                if current_level in mapping_dict:
                    output_level = mapping_dict[current_level]
                    level_mapping_applied = True
                    
                    # Apply parameter mapping based on device type and level transition
                    if mtype_lower in ("npn", "pnp") and current_level == 1 and output_level == 504:
                        # Log summary warning about BJT conversion
                        self.log_warning(
                            f"Converting BJT model '{mname}' from Level 1 (Gummel-Poon) to Level 504 (MEXTRAM). "
                            "Many Level 1 parameters are not compatible with Xyce MEXTRAM and will be commented out. "
                            "See detailed warnings below.",
                            f"Model: {mname}"
                        )
                        # Map level 1 BJT parameters to MEXTRAM (level 504)
                        params_to_write = self._map_bjt_level1_to_mextram_params(params_to_write, model_name=mname)
                    
                    # Update or add level parameter
                    if has_level:
                        # Replace existing level parameter
                        for i, p in enumerate(params_to_write):
                            if self._get_param_name(p).lower() == "level":
                                new_level_param = ParamDecl(name=Ident("level"), default=Int(output_level), distr=None)
                                if isinstance(p, tuple):
                                    # Preserve comment if present, but ensure not commented out
                                    params_to_write[i] = (new_level_param, p[1], False)
                                else:
                                    params_to_write[i] = new_level_param
                                break
                    else:
                        # Add level parameter at the beginning
                        level_param = ParamDecl(name=Ident("level"), default=Int(output_level), distr=None)
                        params_to_write.insert(0, level_param)
        
        # Handle BSIM4: replace with pmos/nmos, add level, map deltox to dtox
        if mtype == "bsim4":
            # Determine pmos/nmos from type parameter
            type_param = next((p for p in model.params if p.name.name == "type"), None)
            if type_param and isinstance(type_param.default, Ref) and type_param.default.ident.name == "p":
                mtype = "pmos"
            elif type_param and isinstance(type_param.default, Ref) and type_param.default.ident.name == "n":
                mtype = "nmos"
            else:
                # Default to nmos if type parameter is not found or unclear
                mtype = "nmos"
            
            # Check if level parameter already exists
            has_level = any(self._get_param_name(p) == "level" for p in params_to_write)
            
            # Map deltox to dtox instead of filtering
            mapped_params = []
            for p in params_to_write:
                if self._get_param_name(p) == "deltox":
                    # Create new ParamDecl with name "dtox" and same value
                    p_default = self._get_param_default(p)
                    p_distr = p[0].distr if isinstance(p, tuple) else p.distr
                    mapped_params.append(ParamDecl(name=Ident("dtox"), default=p_default, distr=p_distr))
                else:
                    mapped_params.append(p)
            params_to_write = mapped_params
            
            # Add level=54 at the beginning if not already present
            if not has_level:
                level_param = ParamDecl(name=Ident("level"), default=Float(54.0), distr=None)
                params_to_write.insert(0, level_param)
        
        # Handle BJT models (NPN and PNP): add level=504 (Mextram) if not already present
        # Only apply default behavior if level mapping was not already applied
        bjt_uses_level_504 = False
        if mtype in ("npn", "pnp") and not level_mapping_applied:
            # Check if level parameter already exists
            has_level = any(self._get_param_name(p) == "level" for p in params_to_write)
            
            # Check current level if it exists
            current_level = 1  # Default
            if has_level:
                for p in params_to_write:
                    if self._get_param_name(p) == "level":
                        default_val = self._get_param_default(p)
                        if isinstance(default_val, (Int, Float)):
                            current_level = int(float(default_val.val) if hasattr(default_val, 'val') else float(default_val))
                        elif isinstance(default_val, MetricNum):
                            current_level = int(float(default_val.val))
                        break
            
            # Add level=504 at the beginning if not already present
            if not has_level:
                level_param = ParamDecl(name=Ident("level"), default=Int(504), distr=None)
                params_to_write.insert(0, level_param)
                bjt_uses_level_504 = True
                # If we're defaulting to level 504, check if we need to convert parameters
                if current_level == 1:
                    self.log_warning(
                        f"BJT model '{mname}' defaulting to Level 504 (MEXTRAM). "
                        "Many Level 1 parameters may not be compatible and will be commented out. "
                        "See detailed warnings below.",
                        f"Model: {mname}"
                    )
                    params_to_write = self._map_bjt_level1_to_mextram_params(params_to_write, model_name=mname)
            else:
                # If level exists, check its value
                for i, p in enumerate(params_to_write):
                    if self._get_param_name(p) == "level":
                        # Check if current level value is less than 504
                        current_level = None
                        default_val = self._get_param_default(p)
                        if isinstance(default_val, (Int, Float)):
                            current_level = float(default_val.val) if hasattr(default_val, 'val') else float(default_val)
                        elif isinstance(default_val, MetricNum):
                            current_level = float(default_val.val)
                        
                        if current_level is not None:
                            if current_level < 504.0:
                                # Update to level=504
                                params_to_write[i] = ParamDecl(name=Ident("level"), default=Int(504), distr=None)
                                bjt_uses_level_504 = True
                                # Convert parameters if upgrading from level 1
                                if current_level == 1:
                                    self.log_warning(
                                        f"BJT model '{mname}' upgraded from Level {int(current_level)} to Level 504 (MEXTRAM). "
                                        "Many Level 1 parameters are not compatible and will be commented out. "
                                        "See detailed warnings below.",
                                        f"Model: {mname}"
                                    )
                                    params_to_write = self._map_bjt_level1_to_mextram_params(params_to_write, model_name=mname)
                            elif abs(current_level - 504.0) < 0.1:  # Close to 504
                                # Already level 504, but ensure it's Int not Float
                                params_to_write[i] = ParamDecl(name=Ident("level"), default=Int(504), distr=None)
                                bjt_uses_level_504 = True
                        break
        elif mtype in ("npn", "pnp") and level_mapping_applied and output_level == 504:
            # Level mapping was applied and resulted in level 504
            bjt_uses_level_504 = True
        
        # Map to Xyce model type equivalent
        xyce_mtype = xyce_mtype_map.get(mtype, mtype)
        
        # Write model header
        if mtype.lower() == "bsim4":
            # Should not happen after conversion, but handle just in case
            self.writeln(f".model {mname}")
        else:
            self.writeln(f".model {mname} {xyce_mtype}")
        
        # Write arguments if any
        if model.args:
            self.write("+ ")
            for arg in model.args:
                self.write(self.format_expr(arg) + " ")
            self.write("\n")
        
        # Write parameters using the filtered list
        for item in params_to_write:
            # Handle new format (tuple) from _map_bjt_level1_to_mextram_params
            if isinstance(item, tuple) and len(item) == 3:
                param, comment, commented_out = item
            else:
                # Standard ParamDecl from other paths
                param = item
                comment = None
                commented_out = False
            
            if commented_out:
                self.write("; ") # Comment out line with semicolon (preferred in Xyce)
            else:
                self.write("+ ")
            
            # For BJT models with level=504, uppercase parameter names
            # Note: Parameters from _map_bjt_level1_to_mextram_params are already uppercase
            if bjt_uses_level_504 and param.name.name != "level":
                # Create a temporary param with uppercase name for writing
                param_upper = ParamDecl(
                    name=Ident(param.name.name.upper()),
                    default=param.default,
                    distr=param.distr
                )
                self.write_param_decl(param_upper)
            else:
                self.write_param_decl(param)
                
            if comment:
                self.write(f" ; {comment}")
                
            self.write("\n")
        
        self.write("\n")  # Ending blank-line
