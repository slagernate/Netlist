"""
Xyce Dialect Netlister
"""

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
)
from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, apply_statistics_variations, debug_find_all_param_refs, replace_param_refs_in_program, expr_references_param


class XyceNetlister(SpiceNetlister):
    """Xyce-Format Netlister"""

    def __init__(self, src: Program, dest: IO, *, errormode: ErrorMode = ErrorMode.RAISE, file_type: str = "") -> None:
        super().__init__(src, dest, errormode=errormode, file_type=file_type)
        self._last_entry_was_instance = False  # Track if last entry written was an instance
        # Track reserved parameter name mappings (reserved_name -> safe_name)
        self._reserved_param_map: Dict[str, str] = {}
        # Xyce reserved variable names (case-insensitive)
        self._reserved_names = {'vt', 'temp', 'time', 'freq', 'omega', 'pi', 'e'}
        # Collect all parameter names from the AST to only rename actual parameters
        self._param_names = self._collect_param_names(src)

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
                warn(f"Renaming reserved parameter '{name}' to '{safe_name}' for Xyce compatibility")
            return self._reserved_param_map[name_lower]
        
        return name

    def netlist(self) -> None:
        """Override netlist() to apply Xyce-specific statistics variations before writing.

        This applies statistics variations (process and mismatch) to the Program,
        creating Xyce-specific artifacts like .FUNC definitions for mismatch parameters
        and enable_mismatch parameter. This only happens when writing to Xyce format.
        """
        # Apply statistics variations with XYCE format before writing
        # This modifies the AST directly, replacing StatisticsBlocks with generated content
        apply_statistics_variations(self.src, output_format=NetlistDialects.XYCE)

        # Call parent implementation to do the actual writing
        super().netlist()

    def _validate_content(self) -> None:
        """Validate that the program content matches the expected file_type."""

        for source_file in self.src.files:
            for entry in source_file.contents:
                if self.file_type == "library":
                    # Library files should ONLY contain LibSectionDef (no subcircuits, no loose parameters)
                    if not isinstance(entry, LibSectionDef):
                        raise ValueError(f"Library file contains non-library content: {type(entry).__name__}. "
                                       "Library files should only contain library sections.")
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

    def write_instance_params(self, pvals: List[ParamVal], is_mos_like: bool = False, module_ref: Optional[Ref] = None) -> None:
        """Write the parameter-values for Instance `pinst`.

        Parameter-values format:
        ```
        XNAME
        + <ports>
        + <subckt-name>
        + PARAMS: name1=val1 name2=val2 name3=val3  (for subcircuits)
        + name1=val1 name2=val2 name3=val3  (for MOS primitives, no PARAMS:)
        
        Args:
            pvals: List of parameter values to write
            is_mos_like: Whether this is a MOS-like instance
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
                # Check if deltox is present before filtering
                has_deltox = any(pval.name.name == "deltox" for pval in pvals)
                if has_deltox:
                    warn(f"Filtering 'deltox' parameter from instance parameters for BSIM4 model '{module_ref.ident.name}'. "
                         f"Xyce does not support deltox/dtox in instance parameters, only in model definitions.")
                params_to_write = [pval for pval in pvals if pval.name.name != "deltox"]

        # MOS primitive instances should NOT have PARAMS: keyword
        if not is_mos_like:
            self.write("PARAMS: ")  # <= Xyce-specific for subcircuits
        # And write them
        for pval in params_to_write:
            self.write_param_val(pval)
            self.write(" ")

        self.write("\n")

    def write_subckt_instance(self, pinst: Instance) -> None:
        """Write sub-circuit-instance `pinst`. Override to handle MOS instances without PARAMS:."""
        
        # Detect if the instance looks like a MOS device (even if parsed as subcircuit instance)
        is_mos_like = (
            len(pinst.conns) == 4  # Exactly 4 ports (d, g, s, b)
            and isinstance(pinst.module, Ref)  # References a model
            and {'l', 'w'} <= {p.name.name for p in pinst.params}  # Has both 'l' and 'w' params
            and any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet', 'pmos', 'nmos'])  # Instance name indicates MOS
        )

        prefix = 'M' if is_mos_like else 'X'

        inst_name = self.format_ident(pinst.name)
        if prefix and not inst_name.upper().startswith(prefix):
            inst_name = f"{prefix}{inst_name}"

        # Write the instance name
        self.write(inst_name + " \n")

        # Write its port-connections
        self.write_instance_conns(pinst)

        # Write the sub-circuit name
        self.write("+ " + self.format_ident(pinst.module.ident) + " \n")

        # Check if instance parameter expressions reference 'm' and add m={m} if needed
        has_m_in_params = any(p.name.name == "m" for p in pinst.params)
        references_m = False
        
        if is_mos_like:
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
                warn(f"Added m={{m}} parameter to instance {pinst.name.name} (referenced in expressions)")

        # Write its parameter values (pass is_mos_like to skip PARAMS: for MOS, and module_ref for BSIM4 detection)
        module_ref = pinst.module if isinstance(pinst.module, Ref) else None
        self.write_instance_params(pinst.params, is_mos_like=is_mos_like, module_ref=module_ref)

        # Add a blank line after each instance for spacing between instances
        # If next entry is a model, write_model_def will handle spacing (but won't add extra)
        self.write("\n")
        # Mark that last entry was an instance
        self._last_entry_was_instance = True

    def write_primitive_instance(self, pinst: Instance) -> None:
        """Write primitive-instance `pinst` of `rmodule`."""
        is_mos_like = (
            len(pinst.args) == 5  # Exactly 4 ports + 1 model
            and isinstance(pinst.args[-1], Ref)  # Last arg is the model reference
            and {'l', 'w'} <= {p.name.name for p in pinst.kwargs}  # Has both 'l' and 'w' params
            and any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet', 'pmos', 'nmos'])  # Instance name indicates MOS
        )
        prefix = 'M' if is_mos_like else ''
        inst_name = self.format_ident(pinst.name)
        if prefix and not inst_name.upper().startswith(prefix):
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
        self.write("+ " + self.format_ident(pinst.args[-1]) + " \n")

        # Extract module_ref from args for BSIM4 detection
        module_ref = None
        if pinst.args and isinstance(pinst.args[-1], Ref):
            module_ref = pinst.args[-1]
        
        # Filter deltox if referencing BSIM4 model
        # (dtox is only valid in model definitions, not instance parameters)
        kwargs_to_write = pinst.kwargs
        if module_ref and self._is_bsim4_model_ref(module_ref):
            # Check if deltox is present before filtering
            has_deltox = any(kw.name.name == "deltox" for kw in pinst.kwargs)
            if has_deltox:
                warn(f"Filtering 'deltox' parameter from instance parameters for BSIM4 model '{module_ref.ident.name}'. "
                     f"Xyce does not support deltox/dtox in instance parameters, only in model definitions.")
            kwargs_to_write = [kw for kw in pinst.kwargs if kw.name.name != "deltox"]
        
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
            warn(msg)
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
        # Reset instance flag at start of subcircuit
        self._last_entry_was_instance = False
        # Store current subcircuit context for use in write_subckt_instance
        self._current_subckt = module

        # Create the module name
        module_name = self.format_ident(module.name)

        # Check if any entries reference 'm' parameter and ensure it exists in subcircuit params
        has_m_param = any(p.name.name == "m" for p in module.params)
        needs_m_param = False
        
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
        
        # Add m=1 parameter if needed but not present
        if needs_m_param and not has_m_param:
            m_param = ParamDecl(name=Ident("m"), default=Float(1.0), distr=None)
            module.params.append(m_param)

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

    def write_param_decl(self, param: ParamDecl) -> str:
        """Format a parameter declaration, with special handling for param functions in Xyce."""
        if param.distr is not None:
            msg = f"Unsupported `distr` for parameter {param.name} will be ignored"
            warn(msg)
            self.write("\n+ ")
            self.write_comment(msg)
            self.write("\n+ ")

        param_name = self.format_ident(param.name)

        # Check if this is a param function (name contains parentheses)
        if '(' in param_name and param_name.endswith(')'):
            # This is a param function like lnorm(mu,sigma) or mm_z1__mismatch__(dummy_param)
            if param.default is None:
                msg = f"Required (non-default) param function {param.name} is not supported."
                warn(msg)
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
            msg = f"Required (non-default) parameter {param.name} is not supported by {self.__class__.__name__}. "
            msg += "Setting to maximum floating-point value {sys.float_info.max}, which almost certainly will not work if instantiated."
            warn(msg)
            default = str(sys.float_info.max)
        else:
            default = self.format_expr(param.default)

        self.write(f"{param_name}={default}")

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

    def write_model_def(self, model: ModelDef) -> None:
        """Write a model definition in Xyce format, handling BSIM4 conversions."""
        # Add blank line before model if last entry was an instance
        if self._last_entry_was_instance:
            self.write("\n")
            self._last_entry_was_instance = False
        
        mname = self.format_ident(model.name)
        mtype = self.format_ident(model.mtype).lower()
        
        # Use local variable for params to avoid mutating the model object
        params_to_write = list(model.params)  # Create a copy
        
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
            has_level = any(p.name.name == "level" for p in params_to_write)
            
            # Map deltox to dtox instead of filtering
            mapped_params = []
            for p in params_to_write:
                if p.name.name == "deltox":
                    # Create new ParamDecl with name "dtox" and same value
                    mapped_params.append(ParamDecl(name=Ident("dtox"), default=p.default, distr=p.distr))
                else:
                    mapped_params.append(p)
            params_to_write = mapped_params
            
            # Add level=54 at the beginning if not already present
            if not has_level:
                level_param = ParamDecl(name=Ident("level"), default=Float(54.0), distr=None)
                params_to_write.insert(0, level_param)
        
        # Write model header
        if mtype.lower() == "bsim4":
            # Should not happen after conversion, but handle just in case
            self.writeln(f".model {mname}")
        else:
            self.writeln(f".model {mname} {mtype}")
        
        # Write arguments if any
        if model.args:
            self.write("+ ")
            for arg in model.args:
                self.write(self.format_expr(arg) + " ")
            self.write("\n")
        
        # Write parameters using the filtered list
        for param in params_to_write:
            self.write("+ ")
            self.write_param_decl(param)
            self.write("\n")
        
        self.write("\n")  # Ending blank-line
