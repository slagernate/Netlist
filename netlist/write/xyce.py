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
    Library,
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
    BinaryOp,
    BinaryOperator,
    UnaryOp,
    TernOp,
)
from .base import Netlister, ErrorMode
from .spice import SpiceNetlister, apply_statistics_variations, debug_find_all_param_refs, replace_param_refs_in_program, expr_references_param, count_param_refs_in_entry


class XyceNetlister(SpiceNetlister):
    """Xyce-Format Netlister"""

    def __init__(self, src: Program, dest: IO, *, errormode: ErrorMode = ErrorMode.RAISE, file_type: str = "", includes: List[Tuple[str, str]] = None, model_file: str = None, model_level_mapping: Optional[Dict[str, List[Tuple[int, int]]]] = None, options = None) -> None:
        super().__init__(src, dest, errormode=errormode, file_type=file_type, options=options)
        self.includes = includes or []
        self.model_file = model_file
        self._last_entry_was_instance = False  # Track if last entry written was an instance
        self._current_fet_subckt = None  # Track current FET subcircuit for parameter replacement
        self._current_subckt = None  # Track current subcircuit being written
        # Track reserved parameter name mappings (reserved_name -> safe_name)
        self._reserved_param_map: Dict[str, str] = {}
        # Xyce reserved variable names (case-insensitive)
        self._reserved_names = {'vt', 'temp', 'time', 'freq', 'omega', 'pi', 'e'}
        # Collect all parameter names from the AST to only rename actual parameters
        self._param_names = self._collect_param_names(src)
        # Collect user-defined function names (from FunctionDef nodes) so we don't
        # mis-classify them as unsupported built-ins during expression validation.
        self._defined_function_names = self._collect_function_names(src)
        # Expression-function compatibility (Spectre -> Xyce).
        # Xyce does not support all Spectre function names (e.g. round()) in .param expressions.
        # We apply a safe alias-map and fail-fast on unsupported built-in functions.
        self._xyce_func_alias_map: Dict[str, str] = {
            "round": "nint",
        }
        self._xyce_func_alias_counts: Dict[str, int] = {}
        # Xyce built-in function allowlist (lowercased).
        # Includes UG arithmetic functions + commonly used functions in our models (e.g. gauss).
        self._xyce_builtin_funcs = {
            # Arithmetic / misc
            "abs", "m", "min", "max", "ceil", "floor", "int", "nint", "sgn", "sign",
            "sqrt", "fmod", "if", "limit", "stp", "uramp", "pwr", "pow", "pwrs",
            "ddt", "ddx", "sdt",
            # Circuit accessors (B-source style)
            "v", "i",
            # Compat helper functions (defined by this netlister when needed)
            "lnorm", "alnorm",
            "mm_z1__mismatch__", "mm_z2__mismatch__",
            # Table / interpolation
            "table", "fasttable", "spline", "akima", "cubic", "wodicka", "bli", "tablefile",
            # Complex helpers
            "db", "img", "ph", "r", "re",
            # Exponential / log / trig
            "exp", "ln", "log", "log10",
            "sin", "sinh", "cos", "cosh", "tan", "tanh",
            "asin", "asinh", "acos", "acosh",
            "atan", "atanh", "atan2", "arctan",
            # Random distributions
            "gauss", "agauss", "unif", "aunif", "rand",
        }
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

    def _collect_function_names(self, program: Optional[Program]) -> set:
        """Collect all user-defined function names from the program AST.

        Includes:
        - `FunctionDef` nodes (true AST functions)
        - Xyce `.func`-style parameter functions represented as `ParamDecl` where the
          parameter name includes parentheses, e.g. `par1nrf__mismatch__(dummy_param)`.
        """
        fn_names = set()

        if program is None:
            return fn_names

        def collect_from_entry(entry):
            if isinstance(entry, FunctionDef):
                fn_names.add(entry.name.name.lower())
            elif isinstance(entry, ParamDecl):
                # Parameter-declared function (Xyce .func)
                name = entry.name.name
                if "(" in name:
                    fn_names.add(name.split("(", 1)[0].lower())
            elif isinstance(entry, ParamDecls):
                for p in entry.params:
                    collect_from_entry(p)
            elif isinstance(entry, SubcktDef):
                for sub_entry in entry.entries:
                    collect_from_entry(sub_entry)
            elif isinstance(entry, LibSectionDef):
                for sub_entry in entry.entries:
                    collect_from_entry(sub_entry)
            elif isinstance(entry, Library):
                for sec in entry.sections:
                    collect_from_entry(sec)

        for file in program.files:
            for entry in file.contents:
                collect_from_entry(entry)

        return fn_names
    
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
        if self.options is None or getattr(self.options, "apply_statistics", True):
            apply_statistics_variations(self.src, output_format=NetlistDialects.XYCE)

        # Call parent implementation to do the actual writing
        super().netlist()

        # Summarize any function-alias mappings applied during netlisting.
        if self._xyce_func_alias_counts:
            parts = []
            for src_name, dst_name in self._xyce_func_alias_map.items():
                n = self._xyce_func_alias_counts.get(src_name, 0)
                if n:
                    parts.append(f"{src_name}->{dst_name} ({n}x)")
            if parts:
                self.log_warning("Applied Xyce function aliases: " + ", ".join(parts))


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
        
        # Iterate definitions - only instantiate subcircuits that are "reachable"
        # from the sections included in this test netlist.
        #
        # This avoids instantiating subcircuits whose defining section isn't selected
        # (e.g. rf_mos subckts when only 'tt' is included), which would otherwise
        # cause "Subcircuit <X> has not been defined" errors in Xyce.
        from ..data import LibSectionDef, Library, UseLibSection

        # Build section graph (section -> referenced sections)
        sections: dict[str, LibSectionDef] = {}
        deps: dict[str, set[str]] = {}

        def index_entry(entry) -> None:
            if isinstance(entry, LibSectionDef):
                sec_name = entry.name.name
                sections[sec_name] = entry
                sdeps = deps.setdefault(sec_name, set())
                for e in entry.entries:
                    if isinstance(e, UseLibSection):
                        sdeps.add(e.section.name)
                for e in entry.entries:
                    index_entry(e)
                return
            if isinstance(entry, Library):
                for sec in entry.sections:
                    index_entry(sec)
                return
            if isinstance(entry, SubcktDef):
                for e in entry.entries:
                    index_entry(e)
                return

        for file in self.src.files:
            for entry in file.contents:
                index_entry(entry)

        # Compute closure of included sections
        start_sections = {sec for _, sec in self.includes if sec}
        allowed_sections: set[str] = set(start_sections)
        queue = list(start_sections)
        while queue:
            s = queue.pop()
            for d in deps.get(s, set()):
                if d not in allowed_sections:
                    allowed_sections.add(d)
                    queue.append(d)

        definitions: list[SubcktDef] = []

        def collect(entry, current_section: str | None = None) -> None:
            if isinstance(entry, LibSectionDef):
                sec_name = entry.name.name
                for e in entry.entries:
                    collect(e, current_section=sec_name)
                return
            if isinstance(entry, Library):
                for sec in entry.sections:
                    collect(sec, current_section=current_section)
                return
            if isinstance(entry, SubcktDef):
                # Include top-level subckts, and sectioned subckts that are reachable.
                if current_section is None or current_section in allowed_sections:
                    definitions.append(entry)
                return

        for file in self.src.files:
            for entry in file.contents:
                collect(entry, current_section=None)
                    
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

    def write_use_lib(self, uselib) -> None:
        """Write a sectioned library-usage (.lib <file> <section>).

        Many Spectre PDK decks self-reference their source file via:
          include "<thisfile>.scs" section=<name>
        which parses to `UseLibSection(path=<thisfile>.scs, section=<name>)`.

        When converting 1:1 into `<thisfile>.lib.cir`, those self-references must be
        rewritten to point at the converted output, otherwise Xyce will try (and fail)
        to include the original Spectre `.scs` file.
        """
        from pathlib import Path
        from ..data import UseLibSection

        if isinstance(uselib, UseLibSection):
            src0 = None
            if self.src.files:
                try:
                    src0 = Path(self.src.files[0].path)
                except Exception:
                    src0 = None

            p = Path(uselib.path)
            if src0 is not None and p.suffix == ".scs":
                # Rewrite self-references (match by basename, and also tolerate relative paths).
                if p.name == src0.name or p.stem == src0.stem:
                    p = Path(p.name).with_suffix(".lib.cir")

            # Use the same formatting as the base SpiceNetlister, but with our potentially rewritten path.
            self.write(f".lib {str(p)} {self.format_ident(uselib.section)} \n\n")
            return

        # Fallback to base implementation for unexpected types
        return super().write_use_lib(uselib)

    def _validate_content(self) -> None:
        """Validate that the program content matches the expected file_type."""

        for source_file in self.src.files:
            for entry in source_file.contents:
                if self.file_type == "library":
                    # Library files should contain LibSectionDef, Library objects, Comments, and BlankLines
                    # Library objects will be handled by write_library() which extracts sections
                    from ..data import BlankLine
                    if not isinstance(entry, (LibSectionDef, Library, Comment, BlankLine)):
                        raise ValueError(f"Library file contains non-library content: {type(entry).__name__}. "
                                       "Library files should only contain library sections, library wrappers, comments, and blank lines.")
                # For models files, allow anything (SubcktDef, ParamDecls, FunctionDef, FlatStatement, etc.)
                # No validation needed for models files


    def write_library_section(self, section: LibSectionDef) -> None:
        """Write a Library Section definition."""
        # Write library section normally (no process variations here)
        section_name = self.format_ident(section.name)
        self.write(f".lib {section_name}\n")
        pbar = None
        if self.options is not None and getattr(self.options, "show_progress", False):
            try:
                from tqdm import tqdm  # type: ignore
                base_desc = getattr(self.options, "progress_desc", None) or "xyce:library"
                pbar = tqdm(total=len(section.entries), desc=f"{base_desc}:{section_name}", unit="entry", mininterval=0.2, leave=False)
            except Exception:
                pbar = None
        for entry in section.entries:
            self.write_entry(entry)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        self.write(f".endl {self.format_ident(section.name)}\n\n")

    def write_library(self, library) -> None:
        """Write a Library object, logging a warning about the omitted library wrapper.
        
        Overrides base class implementation to log a warning when library wrappers are omitted.
        Xyce has no equivalent to Spectre's library wrapper construct, so we omit it and log a warning.
        
        Args:
            library: Library object containing sections to write
        """
        # Always log a warning when omitting library wrappers
        library_name = self.format_ident(library.name)
        self.log_warning(
            f"Spectre library wrapper '{library_name}' omitted (no Xyce equivalent)",
            context=f"Library contains {len(library.sections)} section(s)"
        )
        
        # Extract sections from the library and write each one
        for section in library.sections:
            self.write_library_section(section)

    def write_module_params(self, params: List[ParamDecl]) -> None:
        """Write the parameter declarations for Module `module`.
        Parameter declaration format:
        .SUBCKT <name> <ports>
        + PARAMS: name1=val1
        + name2=val2
        + name3=val3
        Each parameter on its own line to avoid comment issues.
        """
        if not params:
            self.write("+ ")
            self.write_comment("No parameters")
            self.write("\n")
            return
            
        # Write first parameter on PARAMS line
        self.write("+ PARAMS: ")
        self.write_param_decl(params[0])
        self.write("\n")
        
        # Write remaining parameters, each on its own continuation line
        for param in params[1:]:
            self.write("+ ")
            self.write_param_decl(param)
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
        
        # Search program (must recurse into library sections and subcircuits)
        from ..data import ModelDef, ModelFamily, ModelVariant, LibSectionDef, SubcktDef, Library
        
        def check_entry(entry):
            if isinstance(entry, ModelDef) and entry.name.name.lower() == model_name.lower():
                return entry.mtype.name.lower()
            if isinstance(entry, ModelFamily) and entry.name.name.lower() == model_name.lower():
                return entry.mtype.name.lower()
            if isinstance(entry, ModelVariant):
                 if f"{entry.model.name}.{entry.variant.name}".lower() == model_name.lower() or entry.model.name.lower() == model_name.lower():
                     return entry.mtype.name.lower()
            return None

        def visit(entry) -> Optional[str]:
            mtype = check_entry(entry)
            if mtype:
                return mtype

            if isinstance(entry, Library):
                for sec in entry.sections:
                    mtype = visit(sec)
                    if mtype:
                        return mtype
                return None

            if isinstance(entry, LibSectionDef):
                for e in entry.entries:
                    mtype = visit(e)
                    if mtype:
                        return mtype
                return None

            if isinstance(entry, SubcktDef):
                for e in entry.entries:
                    mtype = visit(e)
                    if mtype:
                        return mtype
                return None

            return None

        for file in self.src.files:
            for entry in file.contents:
                mtype = visit(entry)
                if mtype:
                    return mtype
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
        
        # Search program (must recurse into library sections and subcircuits)
        from ..data import ModelDef, ModelFamily, ModelVariant, LibSectionDef, SubcktDef, Library
        module_name = ref.ident.name
        
        def find_entry(entry):
            if isinstance(entry, (SubcktDef, ModelDef, ModelFamily)) and entry.name.name.lower() == module_name.lower():
                return entry
            if isinstance(entry, ModelVariant):
                if f"{entry.model.name}.{entry.variant.name}".lower() == module_name.lower() or entry.model.name.lower() == module_name.lower():
                    return entry
            return None

        def visit(entry):
            result = find_entry(entry)
            if result:
                return result

            if isinstance(entry, Library):
                for sec in entry.sections:
                    result = visit(sec)
                    if result:
                        return result
                return None

            if isinstance(entry, LibSectionDef):
                for e in entry.entries:
                    result = visit(e)
                    if result:
                        return result
                return None

            if isinstance(entry, SubcktDef):
                for e in entry.entries:
                    result = visit(e)
                    if result:
                        return result
                return None

            return None

        for file in self.src.files:
            for entry in file.contents:
                result = visit(entry)
                if result:
                    return result
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

    def write_param_decls(self, params: ParamDecls) -> None:
        """Write parameter declarations for Xyce.

        Xyce supports parameter *functions* via:
          .func name(args) {expr}

        Our statistics/mismatch flow represents these as ParamDecl where the formatted name
        includes parentheses, e.g. `par1nrf__mismatch__(dummy_param)`.
        """
        for p in params.params:
            param_name = self.format_ident(p.name)

            # Param "functions" must be emitted as .func in Xyce.
            if "(" in param_name and param_name.endswith(")"):
                if p.default is None:
                    default_str = "0"
                else:
                    if isinstance(p.default, str):
                        default_str = p.default
                    else:
                        default_str = self.format_expr(p.default)

                    # Strip outer braces/quotes if present so we can wrap in { }
                    if (default_str.startswith("{") and default_str.endswith("}")) or (
                        default_str.startswith("'") and default_str.endswith("'")
                    ):
                        default_str = default_str[1:-1]

                self.write(f".func {param_name} {{{default_str}}}\n")
                continue

            # Normal parameters
            self.write(".param ")
            self.write_param_decl(p)
            self.write("\n")

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

        # Filter unsupported parameters from instance parameters based on model type
        # Xyce does not support certain parameters on instance lines for specific model types
        params_to_write = pvals
        is_bsim4 = False
        is_mos = False
        is_resistor = False
        
        params_to_filter = []
        
        if module_ref:
            is_bsim4 = self._is_bsim4_model_ref(module_ref)
            is_mos = self._is_mos_model_ref(module_ref)
            is_resistor = self._is_resistor_model_ref(module_ref)
            
            if is_bsim4:
                # Filter deltox AND dtox from BSIM4 instance parameters
                # Xyce does not support them on instance, they must be on the model.
                # We handle moving them to the model in write_subckt_def, so here we just ensure they are gone.
                params_to_filter.extend(["deltox", "dtox"])
                # BSIM4 is a MOS model, so also filter MOS-specific parameters
                # MULU0, MULVSAT, DELK1 are not supported on instance lines in Xyce
                params_to_filter.extend(["mulu0", "mulvsat", "delk1"])
            
            if is_mos:
                # Filter unsupported parameters for nmos/pmos models
                # DELTOX, MULU0, MULVSAT, DELK1 are not supported on instance lines in Xyce
                params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
            
            if is_resistor:
                # Filter unsupported parameters for resistor models
                # TC1R, TC2R are not supported on instance lines in Xyce (use tc1, tc2 instead)
                params_to_filter.extend(["tc1r", "tc2r"])
        
        # Fallback: if model type detection failed (even with module_ref), check is_mos_like or parameter names
        # This handles cases where models are defined in the current subcircuit being written
        if not is_mos and not is_bsim4:
            # Check if is_mos_like flag is set (from instance name heuristic in write_subckt_instance)
            if is_mos_like:
                params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
            else:
                # Additional fallback: check parameter names to detect MOS device
                # (l, w, ad, as, pd, ps, nrd, nrs are typical MOS params)
                param_names = [p.name.name.lower() for p in pvals] if pvals else []
                has_mos_params = {'l', 'w'} <= set(param_names) and any(p in param_names for p in ['ad', 'as', 'pd', 'ps', 'nrd', 'nrs'])
                if has_mos_params:
                    params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
        
        if params_to_filter:
            params_to_write = [pval for pval in pvals if pval.name.name.lower() not in [p.lower() for p in params_to_filter]]

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

        # Spectre "bsource" (behavioral source) translation.
        #
        # In Spectre, constructs like:
        #   r1 (nx 5) bsource r=<expr>
        # appear frequently (e.g. ESD/RF diode models, RF MOS gate charge/ current).
        #
        # Our parser represents these as subckt instances with module name "bsource",
        # but Xyce does NOT have a built-in "bsource" subckt.
        #
        # Translate the common forms into native Xyce primitives.
        # All translated elements are named with the "Bsource_" prefix (starts with 'B'):
        #   Bsource_<originalName> ...
        #
        # - r=<expr> : voltage-controlled resistor -> B-source current I = v(p,n)/r
        # - i=<expr> : B-source current source (B element with I=<expr>)
        # - v=<expr> : B-source voltage source (B element with V=<expr>)
        # - q=<expr> : charge source -> current source I=ddt(q)
        if isinstance(pinst.module, Ref) and pinst.module.ident.name.lower() == "bsource":
            if len(pinst.conns) != 2:
                self.log_warning(
                    f"Unsupported bsource with {len(pinst.conns)} terminal(s); emitting as subckt instance",
                    f"Instance: {pinst.name.name}",
                )
            else:
                # Collect params by lowercase name.
                pmap = {pv.name.name.lower(): pv for pv in (pinst.params or [])}

                # Determine bsource "mode"
                mode = None
                if "r" in pmap:
                    mode = "r"
                elif "i" in pmap:
                    mode = "i"
                elif "v" in pmap:
                    mode = "v"
                elif "q" in pmap:
                    mode = "q"

                if mode is None:
                    self.log_warning(
                        "Unsupported bsource (no r/i/v/q parameter found); emitting as subckt instance",
                        f"Instance: {pinst.name.name}",
                    )
                else:
                    # Ensure a stable naming convention for all translated bsource primitives.
                    base_name = self.format_ident(pinst.name)
                    elem_name = base_name if base_name.lower().startswith("bsource_") else f"Bsource_{base_name}"

                    # Write element header and connections.
                    self.write(elem_name + " \n")
                    self.write_instance_conns(pinst)

                    if mode == "r":
                        # Voltage-controlled resistor modeled as current source:
                        #   I = V(p,n) / r
                        #
                        # This matches the common Spectre usage in our PDK where "bsource r=<expr>"
                        # represents a 2-terminal element with an effective resistance.
                        #
                        # Note: If additional bsource params exist, they are ignored for now.
                        from ..data import BinaryOp, BinaryOperator

                        rval = pmap["r"].val

                        # Build v(p,n) call from the two connection nodes.
                        def _conn_ident_name(c) -> Optional[str]:
                            if isinstance(c, Ident):
                                return c.name
                            if isinstance(c, Ref):
                                return c.ident.name
                            return None

                        n1 = _conn_ident_name(pinst.conns[0])
                        n2 = _conn_ident_name(pinst.conns[1])
                        if not n1 or not n2:
                            self.log_warning(
                                "bsource r= translation requires two named nodes; emitting as subckt instance",
                                f"Instance: {pinst.name.name}",
                            )
                        else:
                            vcall = Call(
                                func=Ref(ident=Ident("v")),
                                args=[Ref(ident=Ident(n1)), Ref(ident=Ident(n2))],
                            )
                            i_expr = BinaryOp(tp=BinaryOperator.DIV, left=vcall, right=rval)
                            self.write("+ I=" + self.format_expr(i_expr) + " \n")

                            extra = sorted(k for k in pmap.keys() if k not in {"r"})
                            if extra:
                                self.log_warning(
                                    "Ignoring unsupported bsource parameter(s) for r= translation: " + ", ".join(extra),
                                    f"Instance: {pinst.name.name}",
                                )
                        self.write("\n")
                        return

                    # Behavioral B-source
                    if mode == "i":
                        ival = pmap["i"].val
                        self.write("+ I=" + self.format_expr(ival) + " \n")
                        extra = sorted(k for k in pmap.keys() if k not in {"i"})
                        if extra:
                            self.log_warning(
                                "Ignoring unsupported bsource parameter(s) for i= translation: " + ", ".join(extra),
                                f"Instance: {pinst.name.name}",
                            )
                        self.write("\n")
                        return

                    if mode == "v":
                        vval = pmap["v"].val
                        self.write("+ V=" + self.format_expr(vval) + " \n")
                        extra = sorted(k for k in pmap.keys() if k not in {"v"})
                        if extra:
                            self.log_warning(
                                "Ignoring unsupported bsource parameter(s) for v= translation: " + ", ".join(extra),
                                f"Instance: {pinst.name.name}",
                            )
                        self.write("\n")
                        return

                    if mode == "q":
                        # Charge Q(V) -> current source I = ddt(Q)
                        qval = pmap["q"].val
                        i_expr = Call(func=Ref(ident=Ident("ddt")), args=[qval])
                        self.write("+ I=" + self.format_expr(i_expr) + " \n")
                        extra = sorted(k for k in pmap.keys() if k not in {"q"})
                        if extra:
                            self.log_warning(
                                "Ignoring unsupported bsource parameter(s) for q= translation: " + ", ".join(extra),
                                f"Instance: {pinst.name.name}",
                            )
                        self.write("\n")
                        return
        
        # Check if the module being instantiated is a Model (not a Subckt)
        mtype = None
        # CRITICAL FIX: Check if module_ref resolves to a SubcktDef (not ModelDef)
        # If it's a subcircuit, we must write it as a subcircuit instance (X prefix, PARAMS:)
        # not as a primitive/model instance.
        module_ref = pinst.module if isinstance(pinst.module, Ref) else None
        module_def = self._get_module_definition(module_ref) if module_ref else None
        from ..data import SubcktDef
        is_subcircuit = isinstance(module_def, SubcktDef)
        
        mtype = None
        if isinstance(pinst.module, Ref) and not is_subcircuit:
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
        
        is_model = mtype is not None and not is_subcircuit
        
        prefix = "X"
        if is_model:
            # Determine prefix based on model type
            if "mos" in mtype or mtype == "bsim4": prefix = "M"
            elif "res" in mtype or mtype == "r": prefix = "R"
            elif "cap" in mtype or mtype == "c": prefix = "C"
            elif "ind" in mtype or mtype == "l": prefix = "L"
            # Note: Diodes don't need prefix prepending - just write on one line
            elif "pnp" in mtype or "npn" in mtype or mtype == "q": prefix = "Q"
            # Add more types as needed
        
        # Fallback heuristic for MOS if type not found but looks like MOS
        # BUT: Don't apply if it's actually a subcircuit (keep X prefix)
        if not is_model and not is_subcircuit:
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
        
        # Check if this is a diode - diodes must be written on one line (no continuation lines)
        is_diode = (isinstance(pinst.module, Ref) and pinst.module.ident.name.lower() in ("diode", "d"))
        if isinstance(pinst.module, Ref):
            mod_name = pinst.module.ident.name.lower()
            is_diode = is_diode or ("dio" in mod_name and mod_name not in ("resistor", "res", "capacitor", "cap"))
        
        # Only prepend prefix for non-diodes
        if not is_diode and prefix and not inst_name.upper().startswith(prefix):
            # If we're skipping model name (resistor/capacitor/inductor), use underscore separator
            if skip_model_name:
                inst_name = f"{prefix}_{inst_name}"
            else:
                inst_name = f"{prefix}{inst_name}"
        
        if is_diode:
            # Write everything on one line for diodes
            self.write(inst_name + " ")
            # Write port connections (same logic as write_instance_conns but on same line)
            if pinst.conns:
                if isinstance(pinst.conns[0], tuple):
                    for conn in pinst.conns:
                        if isinstance(conn[0], Ident):
                            self.write(self.format_ident(conn[0]) + " ")
                        elif isinstance(conn[0], Ref):
                            self.write(self.format_ident(conn[0].ident) + " ")
                        else:
                            self.write(self.format_expr(conn[0]) + " ")
                else:
                    for conn in pinst.conns:
                        if isinstance(conn, Ident):
                            self.write(self.format_ident(conn) + " ")
                        elif isinstance(conn, Ref):
                            self.write(self.format_ident(conn.ident) + " ")
                        else:
                            self.write(self.format_expr(conn) + " ")
            # Write the sub-circuit/model name
            if not skip_model_name:
                self.write(self.format_ident(pinst.module.ident) + " ")
        else:
            # Original behavior: use continuation lines for non-diodes
            # Write the instance name
            self.write(inst_name + " \n")

            # Write its port-connections
            self.write_instance_conns(pinst)

            # Write the sub-circuit/model name (skip for resistor/capacitor/inductor)
            if not skip_model_name:
                module_name = self.format_ident(pinst.module.ident)
                module_ref = pinst.module if isinstance(pinst.module, Ref) else None

                # If this is a model reference, align instance model names with the
                # emitted `.model` card names.
                #
                # IMPORTANT: Xyce model cards in our output are emitted using the base
                # model name (e.g. `nch_rf`), not the variant-qualified name
                # (e.g. `nch_rf.1`). Use the base model name for instances so Xyce can
                # resolve the model card.
                if module_ref is not None and is_model:
                    from ..data import ModelVariant, ModelFamily, ModelDef

                    resolved = self._get_module_definition(module_ref)
                    if isinstance(resolved, ModelVariant):
                        module_name = resolved.model.name
                    elif isinstance(resolved, ModelFamily):
                        module_name = resolved.name.name
                    elif isinstance(resolved, ModelDef):
                        module_name = resolved.name.name

                self.write("+ " + module_name + " \n")

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
        # module_ref already set above
        if is_diode:
            # Write parameters on same line for diodes.
            #
            # IMPORTANT: Do NOT write AREA as a positional value. Xyce's parser is picky
            # about positional diode fields, especially when expressions are used.
            # Instead, write named parameters (area=...).
            #
            # Also drop Spectre perimeter-style instance params (`pj`/`perim`) which are
            # not supported as diode instance params in Xyce.
            kept: list[ParamVal] = []
            dropped_names: list[str] = []
            for pval in pinst.params:
                pname = pval.name.name.lower()
                if pname in {"perim", "pj"}:
                    dropped_names.append(pval.name.name)
                    continue
                kept.append(pval)

            if dropped_names:
                self.log_warning(
                    "Dropping unsupported diode instance parameter(s): " + ", ".join(sorted({n.upper() for n in dropped_names})),
                    f"Instance: {pinst.name.name}",
                )

            for pval in kept:
                self.write_param_val(pval)
                self.write(" ")
            self.write("\n")
        else:
            # For subcircuit instances, check if the referenced module is a MOS model
            # to apply parameter filtering (MULU0, MULVSAT, etc.)
            is_mos_like_for_params = is_model
            if not is_model and module_ref:
                # Check if the referenced module is a MOS model (even if it's a subcircuit instance)
                is_mos_ref = self._is_mos_model_ref(module_ref) if module_ref else False
                is_bsim4_ref = self._is_bsim4_model_ref(module_ref) if module_ref else False
                if is_mos_ref or is_bsim4_ref:
                    is_mos_like_for_params = True
                else:
                    # Fallback: check if instance name suggests MOS device
                    inst_name_lower = pinst.name.name.lower()
                    if any(keyword in inst_name_lower for keyword in ['mos', 'fet', 'pmos', 'nmos', 'mnmos', 'mpmos']):
                        is_mos_like_for_params = True
            self.write_instance_params(pinst.params, is_mos_like=is_mos_like_for_params, module_ref=module_ref)

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
        
        # CRITICAL FIX: Check if module_ref resolves to a SubcktDef (not ModelDef)
        # If it's a subcircuit, we must write it as a subcircuit instance (X prefix, PARAMS:)
        # not as a primitive/model instance.
        module_def = self._get_module_definition(module_ref) if module_ref else None
        from ..data import SubcktDef, Ident
        if isinstance(module_def, SubcktDef):
            # This is actually a subcircuit instance, not a primitive/model
            # Convert Primitive to Instance format and delegate to write_subckt_instance
            from ..data import Instance as InstanceType
            # Convert Primitive args/kwargs to Instance format
            # conns must be List[Ident] or List[Tuple[Ident, Ident]]
            conns = []
            for arg in pinst.args[:-1]:  # All args except last (which is the module ref)
                if isinstance(arg, Ident):
                    conns.append(arg)
                elif isinstance(arg, Ref):
                    conns.append(arg.ident)  # Extract Ident from Ref
                else:
                    # For other types (expressions), create an Ident wrapper if possible
                    # This shouldn't normally happen for ports, but handle gracefully
                    if hasattr(arg, 'ident'):
                        conns.append(arg.ident)
                    else:
                        # Fallback: try to create Ident from string representation
                        # This is a safety net, but ports should be Ident/Ref
                        self.log_warning(f"Unexpected port type {type(arg)} in Primitive {pinst.name.name}, treating as Ident", f"Instance: {pinst.name.name}")
                        conns.append(Ident(name=str(arg)))
            
            # Convert kwargs to ParamVal list (already ParamVal objects)
            params = pinst.kwargs
            
            # Create a synthetic Instance node
            fake_instance = InstanceType(
                name=pinst.name,
                module=module_ref,
                conns=conns,
                params=params
            )
            # Delegate to subcircuit instance writer
            return self.write_subckt_instance(fake_instance)
            
        # Determine model type
        mtype = self._get_model_type(module_ref) if module_ref else None
        
        is_model = True # Primitives are generally models
        
        prefix = ""
        if mtype:
            if "mos" in mtype or mtype == "bsim4": prefix = "M"
            elif "res" in mtype or mtype == "r": prefix = "R"
            elif "cap" in mtype or mtype == "c": prefix = "C"
            elif "ind" in mtype or mtype == "l": prefix = "L"
            # Note: Diodes don't need prefix prepending - just write on one line
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
        
        # Check if this is a diode - diodes must be written on one line (no continuation lines)
        # BUT: Only if it's actually a ModelDef, not a SubcktDef (already handled above)
        is_diode = (model_name and model_name in ("diode", "d"))
        if not is_diode and model_name:
            is_diode = ("dio" in model_name and model_name not in ("resistor", "res", "capacitor", "cap"))
        
        # Only prepend prefix for non-diodes
        if not is_diode and prefix and not inst_name.upper().startswith(prefix):
            # If we're skipping model name (resistor/capacitor/inductor), use underscore separator
            if skip_model_name:
                inst_name = f"{prefix}_{inst_name}"
            else:
                inst_name = f"{prefix}{inst_name}"
        
        if is_diode:
            # Write everything on one line for diodes
            self.write(inst_name + " ")
            # Write ports (excluding last (model))
            for arg in pinst.args[:-1]:  # Ports only (exclude the model)
                if isinstance(arg, Ident):
                    self.write(self.format_ident(arg) + " ")
                elif isinstance(arg, Ref):
                    self.write(self.format_ident(arg.ident) + " ")
                elif isinstance(arg, (Int, Float, MetricNum)):
                    self.write(self.format_number(arg) + " ")
                else:
                    self.write(self.format_expr(arg) + " ")
            # Write the model (last arg)
            if not skip_model_name and pinst.args:
                self.write(self.format_ident(pinst.args[-1]) + " ")
            # Write parameters on same line
            # For diodes, only area is supported as positional parameter (perim not supported in Level 3)
            area_val = None
            other_kwargs = []
            kwargs_to_write = pinst.kwargs
            if module_ref:
                is_bsim4 = self._is_bsim4_model_ref(module_ref)
                is_mos = self._is_mos_model_ref(module_ref)
                is_resistor = self._is_resistor_model_ref(module_ref)
                
                params_to_filter = []
                if is_bsim4:
                    params_to_filter.extend(["deltox", "dtox"])
                if is_mos:
                    params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
                if is_resistor:
                    params_to_filter.extend(["tc1r"])
                
                if params_to_filter:
                    kwargs_to_write = [kw for kw in pinst.kwargs if kw.name.name.lower() not in [p.lower() for p in params_to_filter]]
            for kwarg in kwargs_to_write:
                param_name = kwarg.name.name.lower()
                if param_name == "area":
                    area_val = kwarg.val
                # Note: perim is not supported for Level 3 diodes in Xyce, so we skip it
                elif param_name != "perim":
                    other_kwargs.append(kwarg)
            # Write area as positional parameter (just value, no name)
            if area_val is not None:
                self.write(self.format_expr(area_val) + " ")
            # Write other parameters as named (if any)
            for kwarg in other_kwargs:
                self.write_param_val(kwarg)
                self.write(" ")
            self.write("\n")
        else:
            # Original behavior: use continuation lines for non-diodes
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

            # Filter unsupported parameters based on model type
            # (certain parameters are only valid in model definitions, not instance parameters)
            kwargs_to_write = pinst.kwargs
            params_to_filter = []
            
            is_bsim4 = False
            is_mos = False
            is_resistor = False
            
            if module_ref:
                is_bsim4 = self._is_bsim4_model_ref(module_ref)
                is_mos = self._is_mos_model_ref(module_ref)
                is_resistor = self._is_resistor_model_ref(module_ref)
                
                if is_bsim4:
                    params_to_filter.extend(["deltox", "dtox"])
                if is_mos:
                    params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
                if is_resistor:
                    params_to_filter.extend(["tc1r", "tc2r"])
            
            # Fallback: if model type detection failed, check if instance looks like MOS
            # This handles cases where models are defined in the current subcircuit being written
            if not is_mos and not is_bsim4:
                inst_name_lower = pinst.name.name.lower()
                if any(keyword in inst_name_lower for keyword in ['mos', 'fet', 'pmos', 'nmos', 'mnmos', 'mpmos']):
                    params_to_filter.extend(["deltox", "mulu0", "mulvsat", "delk1"])
            
            if params_to_filter:
                # Remove duplicates while preserving order
                seen = set()
                unique_params_to_filter = []
                for p in params_to_filter:
                    p_lower = p.lower()
                    if p_lower not in seen:
                        seen.add(p_lower)
                        unique_params_to_filter.append(p)
                kwargs_to_write = [kw for kw in pinst.kwargs if kw.name.name.lower() not in [p.lower() for p in unique_params_to_filter]]
            
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
        So, just use it all the time.
        
        Preserves exact whitespace from comment text - if comment already has
        leading whitespace, don't add an extra space after semicolon."""
        # Empty comment-only line (e.g. `//`) => emit bare `;` line.
        if comment == "":
            self.write(";\n")
        # If comment starts with whitespace, preserve it exactly (don't add extra space)
        elif comment and comment[0].isspace():
            self.write(f";{comment}\n")
        else:
            # No leading whitespace, add standard space after semicolon
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
    
    def _format_expr_inner(self, expr: Expr) -> str:
        """Format the *inner* content of an expression, without delimiters.
        
        Overrides base implementation to replace ^ operator with ** for Xyce compatibility.
        In Xyce, ^ is boolean XOR, not exponentiation. Xyce uses ** for exponentiation.
        """
        if isinstance(expr, (Int, Float, MetricNum)):
            return self.format_number(expr)
        if isinstance(expr, Ref):
            return self.format_ident(expr)
        
        if isinstance(expr, BinaryOp):
            # Replace ^ operator with ** for Xyce compatibility
            # In Xyce, ^ is boolean XOR, not exponentiation. Use ** for exponentiation.
            op_value = expr.tp.value if hasattr(expr.tp, 'value') else str(expr.tp)
            if op_value == '^' or (hasattr(BinaryOperator, 'POW') and expr.tp == BinaryOperator.POW):
                left = self._format_expr_inner(expr.left)
                right = self._format_expr_inner(expr.right)
                # Wrap operands in parentheses for safety
                if isinstance(expr.left, BinaryOp):
                    left = f"({left})"
                if isinstance(expr.right, BinaryOp):
                    right = f"({right})"
                return f"{left}**{right}"
            
            # Handle other binary operators normally
            left = self._format_expr_inner(expr.left)
            right = self._format_expr_inner(expr.right)
            op = op_value
            
            # Check if we need parentheses for precedence
            if isinstance(expr.left, BinaryOp):
                left = f"({left})"
            if isinstance(expr.right, BinaryOp):
                right = f"({right})"
                
            return f"{left}{op}{right}"
            
        if isinstance(expr, UnaryOp):
            targ = self._format_expr_inner(expr.targ)
            op_value = expr.tp.value if hasattr(expr.tp, 'value') else str(expr.tp)
            return f"{op_value}({targ})"
            
        if isinstance(expr, TernOp):
            cond = self._format_expr_inner(expr.cond)
            if_true = self._format_expr_inner(expr.if_true)
            if_false = self._format_expr_inner(expr.if_false)
            return f"({cond} ? {if_true} : {if_false})"
            
        if isinstance(expr, Call):
            raw = expr.func.ident.name
            raw_l = raw.lower()
            mapped_l = raw_l

            # If this call resolves to a user-defined function, allow it without validation.
            resolved = getattr(expr.func, "resolved", None)
            if resolved is None and raw_l not in self._defined_function_names:
                mapped_l = self._xyce_func_alias_map.get(raw_l, raw_l)
                if raw_l in self._xyce_func_alias_map:
                    self._xyce_func_alias_counts[raw_l] = self._xyce_func_alias_counts.get(raw_l, 0) + 1
                if mapped_l not in self._xyce_builtin_funcs:
                    self.handle_error(
                        expr,
                        f"Unsupported function '{raw}' in Xyce output (after mapping -> '{mapped_l}')",
                    )
            func = mapped_l
            args = [self._format_expr_inner(arg) for arg in expr.args]
            return f"{func}({','.join(args)})"

        self.handle_error(expr, f"Unknown Expression Type {expr}")
        return ""

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
        # Also handle delvto by converting to dvth0 parameter and modifying vth0
        
        from ..data import ModelDef, ModelFamily, Instance, Ref, ParamVal, Ident, ParamDecl, Primitive, BinaryOp, BinaryOperator, Int, Float, MetricNum, Expr
        
        # 1. Find local BSIM4 models (check by mtype and level parameter)
        local_models = {}
        has_delvto = False
        
        for entry in module.entries:
            if isinstance(entry, ModelDef):
                is_bsim4 = entry.mtype.name.lower() == "bsim4"
                # Also check for level=14 or 54 in params (for nmos/pmos with level)
                if not is_bsim4:
                    for param in entry.params:
                        if param.name.name == "level":
                            if isinstance(param.default, (Int, Float)):
                                level_val = param.default.val if hasattr(param.default, 'val') else float(param.default)
                                if level_val in (14, 54):
                                    is_bsim4 = True
                                    break
                if is_bsim4:
                    local_models[entry.name.name] = entry
            elif isinstance(entry, ModelFamily):
                if entry.mtype.name.lower() == "bsim4":
                    local_models[entry.name.name] = entry
            
            # Check for delvto parameters
            if isinstance(entry, Instance):
                for pval in entry.params:
                    if pval.name.name == "delvto":
                        has_delvto = True
                        break
            elif isinstance(entry, Primitive):
                for kwarg in entry.kwargs:
                    if kwarg.name.name == "delvto":
                        has_delvto = True
                        break
        
        # Also check if entries reference BSIM4 models (global models)
        # We need to detect BSIM4 models even if they're not local to convert delvto
        has_global_bsim4 = False
        if not local_models:
            for entry in module.entries:
                model_ref = None
                if isinstance(entry, Instance) and isinstance(entry.module, Ref):
                    model_ref = entry.module
                elif isinstance(entry, Primitive) and len(entry.args) > 0 and isinstance(entry.args[-1], Ref):
                    model_ref = entry.args[-1]
                
                if model_ref:
                    if self._is_bsim4_model_ref(model_ref):
                        # Found BSIM4 model reference, but it's not local
                        # We'll handle delvto conversion but can't modify the model
                        has_global_bsim4 = True
                        break
        
        # If we found delvto but not BSIM4 model, assume BSIM4 (delvto is only used with BSIM4)
        has_bsim4_model = len(local_models) > 0 or has_global_bsim4 or has_delvto
                    
        # 2. Process delvto and deltox if subcircuit has BSIM4 model OR has delvto
        # (delvto is only used with BSIM4, so if we find it, we should convert it)
        needs_dvth0_param = False
        delvto_entries = []  # Track entries with delvto for processing
        
        # Process all entries to find delvto/deltox, regardless of model detection
        # This ensures we convert delvto even if model detection fails
        if has_bsim4_model or has_delvto:
            for entry in module.entries:
                dtox_val = None
                delvto_val = None
                params_to_keep = []
                kwargs_to_keep = []
                
                # Check Instance entries
                if isinstance(entry, Instance) and isinstance(entry.module, Ref):
                    model_name = entry.module.ident.name
                    is_bsim4_instance = model_name in local_models or self._is_bsim4_model_ref(entry.module)
                    
                    # Check for delvto/deltox in any instance (delvto indicates BSIM4)
                    for pval in entry.params:
                        if pval.name.name in ("deltox", "dtox"):
                            dtox_val = pval.val
                        elif pval.name.name == "delvto":
                            delvto_val = pval.val
                            needs_dvth0_param = True
                            delvto_entries.append((entry, delvto_val, "params"))
                        else:
                            params_to_keep.append(pval)
                    
                    # Update entry params (remove deltox/dtox/delvto) if found
                    if dtox_val is not None or delvto_val is not None:
                        entry.params = params_to_keep
                        
                        # Handle deltox -> dtox for local models only
                        # BUT: Only if dtox_val doesn't reference instance parameters (m, l, w)
                        # Model parameters can't reference instance parameters in Xyce
                        if dtox_val is not None and model_name in local_models:
                            # Check if dtox_val references instance parameters
                            # expr_references_param is imported from spice module at top level (line 43)
                            references_instance_params = False
                            if isinstance(dtox_val, Expr):
                                # Check for m, l, w references
                                for param_name in ['m', 'l', 'w']:
                                    if expr_references_param(dtox_val, param_name):
                                        references_instance_params = True
                                        break
                            
                            if not references_instance_params:
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
                            # else: dtox references instance parameters - can't move to model
                            # It will be filtered out by parameter filtering (deltox/dtox are filtered)
                
                # Check Primitive entries
                elif isinstance(entry, Primitive) and len(entry.args) > 0 and isinstance(entry.args[-1], Ref):
                    model_ref = entry.args[-1]
                    model_name = model_ref.ident.name
                    is_bsim4_primitive = model_name in local_models or self._is_bsim4_model_ref(model_ref)
                    
                    # Check for delvto/deltox in any primitive (delvto indicates BSIM4)
                    for kwarg in entry.kwargs:
                        if kwarg.name.name in ("deltox", "dtox"):
                            dtox_val = kwarg.val
                        elif kwarg.name.name == "delvto":
                            delvto_val = kwarg.val
                            needs_dvth0_param = True
                            delvto_entries.append((entry, delvto_val, "kwargs"))
                        else:
                            kwargs_to_keep.append(kwarg)
                    
                    # Update entry kwargs (remove deltox/dtox/delvto) if found
                    if dtox_val is not None or delvto_val is not None:
                        entry.kwargs = kwargs_to_keep
                        
                        # Handle deltox -> dtox for local models only
                        # BUT: Only if dtox_val doesn't reference instance parameters (m, l, w)
                        # Model parameters can't reference instance parameters in Xyce
                        if dtox_val is not None and model_name in local_models:
                            # Check if dtox_val references instance parameters
                            # expr_references_param is imported from spice module at top level (line 43)
                            references_instance_params = False
                            if isinstance(dtox_val, Expr):
                                # Check for m, l, w references
                                for param_name in ['m', 'l', 'w']:
                                    if expr_references_param(dtox_val, param_name):
                                        references_instance_params = True
                                        break
                            
                            if not references_instance_params:
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
                            # else: dtox references instance parameters - can't move to model
                            # It will be filtered out by parameter filtering (deltox/dtox are filtered)
        
        # 3. Add dvth0 parameter to subcircuit if needed, and modify model vth0
        if needs_dvth0_param:
            # Check if dvth0 already exists in subcircuit params
            has_dvth0 = any(p.name.name == "dvth0" for p in module.params)
            if not has_dvth0:
                # Add dvth0 parameter with default 0
                dvth0_param = ParamDecl(name=Ident("dvth0"), default=Float(0.0), distr=None)
                module.params.append(dvth0_param)
            
            # Modify model's vth0 parameter to include dvth0 (only for local models)
            for model_name, model in local_models.items():
                # Helper to modify vth0 to include dvth0
                def modify_vth0(params_list):
                    new_params = []
                    vth0_found = False
                    dvth0_ref = Ref(ident=Ident("dvth0"))
                    
                    for p in params_list:
                        if p.name.name == "vth0":
                            vth0_found = True
                            # Modify vth0 to be vth0 + dvth0
                            if isinstance(p.default, (Int, Float, MetricNum)):
                                # If vth0 is a literal, create expression: vth0 + dvth0
                                vth0_expr = BinaryOp(
                                    tp=BinaryOperator.ADD,
                                    left=p.default,
                                    right=dvth0_ref
                                )
                                new_params.append(ParamDecl(name=p.name, default=vth0_expr, distr=p.distr, comment=p.comment))
                            elif isinstance(p.default, Expr):
                                # If vth0 is already an expression, wrap it: (vth0_expr) + dvth0
                                vth0_expr = BinaryOp(
                                    tp=BinaryOperator.ADD,
                                    left=p.default,
                                    right=dvth0_ref
                                )
                                new_params.append(ParamDecl(name=p.name, default=vth0_expr, distr=p.distr, comment=p.comment))
                            else:
                                # Fallback: just add dvth0
                                vth0_expr = BinaryOp(
                                    tp=BinaryOperator.ADD,
                                    left=Float(0.0),
                                    right=dvth0_ref
                                )
                                new_params.append(ParamDecl(name=p.name, default=vth0_expr, distr=p.distr, comment=p.comment))
                        else:
                            new_params.append(p)
                    
                    # If vth0 not found, add it as dvth0 (so it's just dvth0)
                    if not vth0_found:
                        new_params.append(ParamDecl(name=Ident("vth0"), default=dvth0_ref, distr=None))
                    
                    return new_params
                
                if isinstance(model, ModelDef):
                    model.params = modify_vth0(model.params)
                elif isinstance(model, ModelFamily):
                    for variant in model.variants:
                        variant.params = modify_vth0(variant.params)
            
            # Update entries (instances or primitives) to use dvth0 instead of delvto
            # Note: For Xyce, we don't add dvth0 to instance parameters because:
            # 1. dvth0 is a subcircuit parameter (already added above)
            # 2. The model's vth0 expression references dvth0 from subcircuit scope
            # 3. Xyce doesn't support dvth0 as an instance parameter
            # The delvto value is already incorporated into the model's vth0 expression,
            # so we just remove delvto from instances (already done above) and don't add dvth0

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

        # Check if this is a BJT subcircuit by:
        # 1. Checking subcircuit name (npn_*, pnp_*)
        # 2. Checking for BJT models inside
        # 3. Checking for BJT-specific parameters
        is_bjt_subckt = False
        bjt_param_names = {'dkisnpn1x1', 'dkbfnpn1x1', 'dkispnp', 'dkbfpnp', 'var_is', 'var_bf', 'm'}
        
        # Check subcircuit name
        module_name_lower = module_name.lower()
        if module_name_lower.startswith('npn_') or module_name_lower.startswith('pnp_'):
            is_bjt_subckt = True
            # Debug: log BJT detection
            self.log_warning(f"BJT subcircuit detected: {module_name}", f"Subcircuit: {module_name}")
        
        # Check for BJT models inside subcircuit
        if not is_bjt_subckt:
            for entry in module.entries:
                if isinstance(entry, ModelDef):
                    mtype = entry.mtype.name.lower() if hasattr(entry.mtype, 'name') else str(entry.mtype).lower()
                    if mtype in ('npn', 'pnp', 'q'):
                        is_bjt_subckt = True
                        break
                elif isinstance(entry, ModelFamily):
                    for variant in entry.variants:
                        mtype = variant.mtype.name.lower() if hasattr(variant.mtype, 'name') else str(variant.mtype).lower()
                        if mtype in ('npn', 'pnp', 'q'):
                            is_bjt_subckt = True
                            break
                    if is_bjt_subckt:
                        break
        
        # Check for BJT-specific parameters in params list
        # But exclude 'm' from this check since 'm' is common to many subcircuits
        if not is_bjt_subckt and module.params:
            for param in module.params:
                param_name_lower = param.name.name.lower()
                if param_name_lower in bjt_param_names and param_name_lower != 'm':
                    is_bjt_subckt = True
                    break

        # Check if this is a FET/MOS subcircuit - detect by checking for FET-specific parameters
        # that are used in model definitions (swx_nrds, swx_vth)
        # These need to be accessible to models, but also settable from outside
        is_fet_subckt = False
        fet_model_param_names = {'swx_nrds', 'swx_vth'}  # Parameters used in model definitions
        
        # Check if subcircuit has FET model parameters in PARAMS
        if module.params:
            for param in module.params:
                param_name_lower = param.name.name.lower()
                if param_name_lower in fet_model_param_names:
                    is_fet_subckt = True
                    # FET parameters are kept in PARAMS only - models can access them directly
                    break
        
        # For BJT and FET subcircuits, separate parameters into:
        # 1. Instance-specific params (m, l, w, etc.) -> keep in PARAMS:
        # 2. Model-accessed params (dkisnpn1x1, swx_nrds, swx_vth) -> write as .param statements
        params_for_params_line = []
        params_for_param_statements = []
        
        # Debug: log state before parameter separation
        if is_fet_subckt:
            self.log_warning(
                f"DEBUG: is_fet_subckt={is_fet_subckt}, is_bjt_subckt={is_bjt_subckt}, has_params={bool(module.params)}, num_params={len(module.params) if module.params else 0}",
                f"Subcircuit: {module_name}"
            )
        
        if is_bjt_subckt and module.params:
            for param in module.params:
                param_name = param.name.name.lower()
                if param_name in bjt_param_names and param_name != 'm':
                    # BJT-specific parameter used in model -> write as .param statement
                    params_for_param_statements.append(param)
                else:
                    # Instance-specific or other parameter -> keep in PARAMS:
                    params_for_params_line.append(param)
        elif is_fet_subckt and module.params:
            # For FET subcircuits: keep all parameters in PARAMS only
            # Models can access PARAMS directly, no need for .param statements
            for param in module.params:
                # All parameters go to PARAMS line only
                params_for_params_line.append(param)
        else:
            # Not a BJT/FET subcircuit, or no params -> use normal behavior
            params_for_params_line = module.params

        # Start the sub-circuit definition header with inline ports
        self.write(f".SUBCKT {module_name}")
        if module.ports:
            for port in module.ports:
                self.write(f" {self.format_ident(port)}")
        self.write("\n")

        # Add parameters on a continuation line (only instance-specific params for BJT)
        if params_for_params_line:
            self.write_module_params(params_for_params_line)
        else:
            self.write("+ ")
            self.write_comment("No parameters")

        # End the header with a blank line
        self.write("\n")

        # For BJT and FET subcircuits, write .param statements for model-accessed parameters
        if (is_bjt_subckt or is_fet_subckt) and params_for_param_statements:
            if is_bjt_subckt:
                self.write_comment("BJT parameters moved from PARAMS: to .param statements for model access")
            else:
                self.write_comment("FET parameters: kept in PARAMS (for external setting) and also as .param (for model access)")
            self.write("\n")
            for param in params_for_param_statements:
                self.write(".param ")
                # For FET: write the expression directly (evaluates it), not a circular reference
                # For BJT: use the parameter's default value directly
                if is_fet_subckt and param.name.name.lower() in fet_model_param_names:
                    # Write .param with the expression from PARAMS default
                    # This evaluates the expression, making it accessible to models
                    # Note: If external user sets swx_nrds via PARAMS, that value is used for instance params
                    # but model will use the .param value (evaluated expression)
                    # For true inheritance, we'd need .param swx_nrds={swx_nrds} but that's circular in Xyce
                    param_name = self.format_ident(param.name)
                    self.write(f"{param_name}=")
                    if param.default:
                        # Use the expression from PARAMS default
                        expr_inner = self._format_expr_inner(param.default)
                        self.write(f"{{{expr_inner}}}\n")
                    else:
                        self.write("0\n")
                else:
                    # For BJT: use the parameter's default value directly
                    self.write_param_decl(param)
                    self.write("\n")
            self.write("\n")

        # Store FET subcircuit info for model parameter replacement
        if is_fet_subckt:
            self._current_fet_subckt = module
        else:
            self._current_fet_subckt = None
        
        # Write internal content
        for entry in module.entries:
            self.write_entry(entry)
        
        # Reset instance flag after subcircuit ends
        self._last_entry_was_instance = False

        # Close up the sub-circuit
        self.write(".ENDS\n\n")
        # Clear current subcircuit context
        self._current_subckt = None
        self._current_fet_subckt = None

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

    def _is_mos_model_ref(self, ref: Ref) -> bool:
        """Check if a Ref points to an nmos or pmos model definition.
        
        Args:
            ref: A Ref object pointing to a model name
            
        Returns:
            True if the reference points to an nmos or pmos model, False otherwise
        """
        if not isinstance(ref, Ref):
            return False
        
        mtype = self._get_model_type(ref)
        if mtype:
            return mtype in ("nmos", "pmos")
        
        # Fallback: heuristic check based on model name
        # If _get_model_type fails (e.g., model defined in current subcircuit being written),
        # check if model name suggests it's a MOS model
        model_name_lower = ref.ident.name.lower()
        if "nmos" in model_name_lower or "pmos" in model_name_lower or "mos" in model_name_lower:
            # Additional check: exclude obvious non-MOS models
            if "res" not in model_name_lower and "cap" not in model_name_lower and "ind" not in model_name_lower:
                return True
        
        return False

    def _is_resistor_model_ref(self, ref: Ref) -> bool:
        """Check if a Ref points to a resistor model definition.
        
        Args:
            ref: A Ref object pointing to a model name
            
        Returns:
            True if the reference points to a resistor model, False otherwise
        """
        if not isinstance(ref, Ref):
            return False
        
        mtype = self._get_model_type(ref)
        if mtype:
            return mtype in ("r", "res", "resistor")
        
        # Fallback: heuristic check based on model name
        model_name_lower = ref.ident.name.lower()
        if "res" in model_name_lower or "rpoly" in model_name_lower or "r_" in model_name_lower:
            return True
        
        return False

    def _clamp_param_value(self, expr: Expr, min_val: Optional[float], max_val: Optional[float], 
                          min_exclusive: bool = False, max_exclusive: bool = False, preserve_zero: bool = False) -> Expr:
        """Clamp a parameter value expression to valid range.
        
        Args:
            expr: The expression to clamp (Int, Float, or MetricNum)
            min_val: Minimum value (None for unbounded)
            max_val: Maximum value (None for unbounded)
            min_exclusive: If True, min is exclusive (use min + epsilon)
            max_exclusive: If True, max is exclusive (use max - epsilon)
            preserve_zero: If True, preserve value 0 (don't clamp) as 0 often has special meaning
            
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
        
        # Preserve zero if requested (0 often has special meaning in SPICE models)
        if preserve_zero and abs(val) < 1e-12:
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

    def _map_bjt_level1_to_level1_params(self, params: List[ParamDecl], model_name: str = "") -> List[Tuple[ParamDecl, Optional[str], bool]]:
        """Filter BJT Level 1 (Gummel-Poon) parameters to keep only those supported in Xyce Level 1.
        
        Args:
            params: List of ParamDecl objects with level 1 parameter names
            model_name: Name of the model being converted (for warning context)
            
        Returns:
            List of (ParamDecl, comment, commented_out) tuples
        """
        # Track warnings for this conversion
        dropped_params = []
        kept_params = []
        
        # List of parameters supported in Xyce Level 1 (Gummel-Poon)
        # These map directly from Spectre Level 1 to Xyce Level 1
        supported_level1_params = {
            # DC parameters
            'IS', 'BF', 'BR', 'NF', 'NR', 'VAF', 'VAR',
            'IKF', 'IKR', 'ISE', 'ISC', 'NE', 'NC',
            # Resistance parameters
            'RB', 'RBM', 'RE', 'RC', 'IRB',
            # Capacitance parameters
            'CJE', 'CJC', 'CJS', 'VJE', 'VJC', 'VJS',
            'MJE', 'MJC', 'MJS', 'FC', 'XCJC',
            # Transit time parameters
            'TF', 'TR',
            # Temperature parameters
            'TNOM', 'EG', 'XTI', 'XTB',  # Note: TREF is mapped to TNOM, not kept as-is
            # Noise parameters
            'AF', 'KF',
        }
        
        # Parameters to drop (not in Xyce Level 1)
        unsupported_params = {
            # Advanced TF parameters
            'ITF', 'VTF', 'XTF', 'PTF',
            # Substrate/advanced parameters
            'ISS', 'NKF', 'IBC', 'SUBS',
            # Spectre-only parameters
            'DCAP', 'GAP1', 'GAP2',
            # Temperature coefficients (Xyce Level 1 uses simpler temp model)
            'CTC', 'CTE', 'CTS', 'TLEV', 'TLEVC',
            'TVJC', 'TVJE', 'TVJS',
            # All T* temperature coefficient parameters
        }
        
        # Process each parameter
        filtered_params = []
        for param_item in params:
            # Handle both ParamDecl and tuple formats
            if isinstance(param_item, tuple):
                param = param_item[0]  # Extract ParamDecl from tuple
            else:
                param = param_item
            
            param_name_upper = param.name.name.upper()
            
            # Check if parameter is supported
            if param_name_upper in supported_level1_params:
                # Special handling for VAR=0 (infinite reverse Early voltage)
                # Set to large value (100000V) to avoid numerical issues in base charge calculation
                if param_name_upper == 'VAR':
                    # Check if VAR is 0 (either as number or in expression)
                    var_val = None
                    if self._is_number(param.default):
                        var_val = float(param.default.val) if hasattr(param.default, 'val') else float(param.default)
                    elif isinstance(param.default, (Int, Float)):
                        var_val = float(param.default.val) if hasattr(param.default, 'val') else float(param.default)
                    
                    if var_val is not None and abs(var_val) < 1e-12:  # VAR = 0
                        # Set VAR to large value (100000V) instead of 0
                        from ..data import Float
                        var_param = ParamDecl(
                            name=param.name,
                            default=Float(100000.0),
                            distr=param.distr
                        )
                        filtered_params.append((var_param, "var=0 means infinite/unused in SPICE, VAR set to 100000V for numerical stability", False))
                        kept_params.append(param.name.name)
                        continue
                
                # Keep parameter as-is (direct mapping, same name)
                filtered_params.append((param, None, False))
                kept_params.append(param.name.name)
            elif param_name_upper in unsupported_params:
                # Drop parameter
                dropped_params.append(param.name.name)
                # Don't add to filtered_params (effectively drops it)
            elif param_name_upper.startswith('T') and len(param_name_upper) > 1:
                # Check if it's a temperature coefficient parameter (TIS1, TBF1, etc.)
                # These are not in Xyce Level 1
                # But exclude TREF, TF, TR which are valid
                if param_name_upper not in ('TREF', 'TF', 'TR'):
                    dropped_params.append(param.name.name)
            elif param_name_upper == 'LEVEL':
                # Level parameter is handled separately by level mapping system, drop it here
                # (it will be added back with the correct output level)
                dropped_params.append(param.name.name)
            elif param_name_upper == 'NS':
                # NS (substrate emission coefficient) - not in Xyce Level 1, drop it
                dropped_params.append(param.name.name)
            elif param_name_upper == 'TREF':
                # TREF (reference temperature) - map to TNOM in Xyce
                from ..data import Ident
                renamed_param = ParamDecl(
                    name=Ident('TNOM'),
                    default=param.default,
                    distr=param.distr
                )
                filtered_params.append((renamed_param, "TREF -> TNOM (Xyce standard)", False))
                kept_params.append("TREF -> TNOM")
            else:
                # Unknown parameter - drop it with warning (don't keep unknown params)
                self.log_warning(
                    f"BJT Level 1 parameter '{param.name.name}' not recognized. Dropping it (not in Xyce Level 1 standard).",
                    f"Model: {model_name}"
                )
                dropped_params.append(param.name.name)
        
        # Log summary
        context = f"Model: {model_name}" if model_name else "BJT conversion"
        if dropped_params:
            self.log_warning(
                f"BJT Level 1 -> Level 1 conversion: {len(dropped_params)} parameter(s) dropped (not supported in Xyce Level 1): {', '.join(dropped_params)}",
                context
            )
        
        return filtered_params

    def _map_diode_level3_to_level1_params(self, params: List[ParamDecl], model_name: str = "") -> List[Tuple[ParamDecl, Optional[str], bool]]:
        """Map diode Level 3 parameters to Level 1 parameters.
        
        Filters out Level 3-specific parameters and keeps only Level 1 supported ones.
        
        Args:
            params: List of ParamDecl objects with level 3 parameter names
            model_name: Name of the model being converted (for warning context)
            
        Returns:
            List of (ParamDecl, comment, commented_out) tuples
        """
        # Track warnings for this conversion
        dropped_params = []
        kept_params = []
        
        # Level 1 diode parameters (based on SPICE3f5 standard)
        # Standard Level 1 params: IS, N, RS, CJO, VJ, M, TT, EG, XTI, KF, AF, FC, BV, IBV, TNOM
        # Note: AREA is an instance parameter, not a model parameter in Level 1
        supported_level1_params = {
            'IS', 'N', 'RS', 'CJO', 'VJ', 'M', 'TT', 'EG', 'XTI', 'KF', 'AF', 'FC', 'BV', 'IBV', 'TNOM'
        }
        
        # Level 3 specific parameters to drop (not supported in Level 1)
        unsupported_params = {
            # Level 3 geometry parameters
            'XW', 'W', 'L', 'DEFW', 'DELL', 'LM', 'LP', 'WM', 'WP', 'XM', 'XL', 'XP', 'XOI', 'XOM',
            # Level 3 temperature parameters
            'TLEVC', 'TLEV', 'GAP1', 'GAP2', 'TCV', 'TREF',
            # Level 3 advanced parameters
            'TTT1', 'TTT2', 'TM1', 'TM2', 'CTA', 'CTP', 'TPB', 'TPHP', 'TTPB', 'TTPHP',
            # Sidewall capacitance parameters (not in Level 1)
            'CJSW', 'MJSW', 'PHP', 'JSW',
            # Current parameters (IK/IKR are BJT params, not diode Level 1)
            'IK', 'IKR',
            # AREA is instance-level only in Level 1, not a model parameter
            'AREA',
            # TRS (temperature coefficient of RS) - drop if RS already exists, otherwise could map to RS
            # But to avoid conflicts, just drop TRS
            'TRS',
        }
        
        # Parameter name mappings (Level 3 -> Level 1)
        # These need to be renamed to Level 1 standard names
        # Keys are lowercase to match param_name_lower check
        param_name_mapping = {
            'cj': 'CJO',   # Zero-bias junction capacitance
            'pb': 'VJ',    # Junction potential (built-in potential)
            'mj': 'M',     # Grading coefficient
            'js': 'IS',    # Saturation current (if used as model param)
            'vb': 'BV',    # Breakdown voltage
            # Note: 'trs' -> 'RS' mapping removed - if both RS and TRS exist, keep RS and drop TRS
        }
        
        # Process each parameter
        filtered_params = []
        # Track which output parameter names we've already added (to avoid duplicates)
        seen_output_params = set()
        
        for param_item in params:
            # Handle both ParamDecl and tuple formats
            if isinstance(param_item, tuple):
                param = param_item[0]
            else:
                param = param_item
            
            param_name_upper = param.name.name.upper()
            param_name_lower = param.name.name.lower()
            
            # First check if parameter should be renamed (mapping takes priority)
            if param_name_lower in param_name_mapping:
                new_name = param_name_mapping[param_name_lower]
                if new_name is None:
                    # Drop this parameter
                    dropped_params.append(param.name.name)
                    continue
                # Check if we already have this output parameter name
                if new_name.upper() in seen_output_params:
                    # Skip duplicate - keep the first one, drop this one
                    self.log_warning(
                        f"Diode Level 3 parameter '{param.name.name}' maps to '{new_name}' which already exists. Dropping duplicate.",
                        f"Model: {model_name}"
                    )
                    dropped_params.append(param.name.name)
                    continue
                # Rename parameter
                from ..data import Ident
                renamed_param = ParamDecl(
                    name=Ident(new_name),
                    default=param.default,
                    distr=param.distr
                )
                filtered_params.append((renamed_param, f"{param.name.name} -> {new_name} (Level 3 -> Level 1)", False))
                kept_params.append(f"{param.name.name} -> {new_name}")
                seen_output_params.add(new_name.upper())
                continue
            
            # Check if parameter is in unsupported list (drop it)
            if param_name_upper in unsupported_params:
                dropped_params.append(param.name.name)
                continue
            
            # Check if parameter is supported in Level 1 (keep as-is)
            if param_name_upper in supported_level1_params:
                # Check for duplicates
                if param_name_upper in seen_output_params:
                    self.log_warning(
                        f"Diode Level 3 parameter '{param.name.name}' is duplicate. Dropping it.",
                        f"Model: {model_name}"
                    )
                    dropped_params.append(param.name.name)
                else:
                    filtered_params.append((param, None, False))
                    kept_params.append(param.name.name)
                    seen_output_params.add(param_name_upper)
            else:
                # Unknown parameter - drop it with warning (don't keep unknown params)
                self.log_warning(
                    f"Diode Level 3 parameter '{param.name.name}' not recognized. Dropping it (not in Level 1 standard).",
                    f"Model: {model_name}"
                )
                dropped_params.append(param.name.name)
        
        # Log summary
        context = f"Model: {model_name}" if model_name else "Diode conversion"
        if dropped_params:
            self.log_warning(
                f"Diode Level 3 -> Level 1 conversion: {len(dropped_params)} parameter(s) dropped (not supported in Xyce Level 1): {', '.join(dropped_params)}",
                context
            )
        
        return filtered_params

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
            ('VAR',  'VER', (0.0, None)),  # reverse Early voltage -> VER, range: [0.0, +inf) - var=0 means infinite/unused in SPICE, will be handled specially
            ('IKF',  'SPECIAL_IK'),  # forward knee -> IK (conflict with IKR)
            ('IKR',  'SPECIAL_IK'),  # reverse knee -> IK (conflict with IKF)
            ('ISE',  'IBF'),    # B-E leakage sat current -> IBF
            ('ISC',  'ICSS'),   # B-C leakage sat current -> ICSS
            ('NE',   'MLF'),    # B-E leakage emission coeff -> MLF
            ('NC',   'PS', (0.01, 0.99, True, True)),  # B-C leakage emission coeff -> PS, range: ]0.01, 0.99[ (conflict with NS)
            
            # === Resistances ===
            ('RB',   'RBV'),    # parasitic base resistance -> RBV (variable part)
            ('RBM',  'RBC', (0.001, None)),  # minimum intrinsic base resistance -> RBC, range: [0.001, +inf) - MEXTRAM requires RBC >= 0.001
            ('RBI',  'SPECIAL_RBI'), # distribute to RBV + RBC
            ('IRB',  (None, "Unsupported; no direct IRB equivalent")),
            ('RE',   'RE'),     # emitter resistance
            ('RC',   'RCC'),    # collector resistance -> RCC
            # NOTE: RCC should be calculated using Buried Layer (NBL) sheet resistance (~20Ω/□), not N-Well (~1000Ω/□)
            # If RCC is too high (e.g., >50Ω for a 1x1 device), it may indicate wrong sheet resistance was used
            
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
        preserve_zero_map = {}  # Track which params should preserve 0
        for mapping_entry in level1_to_mextram_mapping:
            level1_name = mapping_entry[0]
            action = mapping_entry[1]
            param_map[level1_name.upper()] = action
            # Check if there's a range tuple (3rd element)
            if len(mapping_entry) >= 3:
                range_info = mapping_entry[2]
                range_map[level1_name.upper()] = range_info
                # Check if preserve_zero flag is set (5th element in tuple)
                if len(range_info) >= 5:
                    preserve_zero_map[level1_name.upper()] = range_info[4]
            
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
                        # Special handling: VAR=0 means infinite/unused in SPICE, set VER to large value (100000V) for convergence
                        if name_upper == 'VAR' and self._is_number(param.default):
                            var_val = float(param.default.val) if hasattr(param.default, 'val') else float(param.default)
                            if abs(var_val) < 1e-12:  # VAR = 0
                                # Set VER to large value (100000V) instead of skipping - helps Xyce convergence
                                ver_val = Float(100000.0)
                                add_param('VER', ver_val, param.distr, comment="var=0 means infinite/unused in SPICE, VER set to 100000V for convergence")
                                continue
                        
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
                                    # Check if we should preserve zero for this parameter
                                    preserve_zero = preserve_zero_map.get(name_upper, False)
                                    # Normal clamping
                                    clamped = self._clamp_param_value(default_val, min_val, max_val, min_exclusive, max_exclusive, preserve_zero=preserve_zero)
                                    if self._is_number(clamped):
                                        if isinstance(default_val, (Int, Float)) and isinstance(clamped, (Int, Float)):
                                            old_val = default_val.val if hasattr(default_val, 'val') else default_val
                                            new_val = clamped.val if hasattr(clamped, 'val') else clamped
                                            # Special case: VER cannot be 0 in Xyce, use minimum 0.01 if clamped to 0
                                            if name_upper == 'VAR' and new_name == 'VER' and abs(new_val) < 1e-12:
                                                new_val = 0.01
                                                clamped = Float(new_val)
                                                clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val} - Xyce requires VER > 0)")
                                            elif abs(old_val - new_val) >= 1e-12:
                                                clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val})")
                                            default_val = clamped
                                    else:
                                        default_val = clamped
                                else:
                                    # Expression - DO NOT wrap in max()/min() for expressions
                                    # Xyce has trouble with max()/min() during DC analysis convergence
                                    # Range constraints are only enforced for numeric literals
                                    # Expressions are assumed to evaluate correctly (or will fail at runtime if invalid)
                                    # This matches the working simplified model which uses pre-evaluated numeric values
                                    pass  # Keep expression as-is without max()/min() wrapping
                        add_param(new_name, default_val, param.distr, comment=f"{param.name.name} -> {new_name} (lvl 1 -> lvl 504). {warning_msg}")
                        
                elif isinstance(action, str):
                    if action.startswith('SPECIAL_'):
                        continue # Already handled
                    new_name = action
                    
                    # Special handling: VAR=0 means infinite/unused in SPICE, set VER to large value (100000V) for convergence
                    if name_upper == 'VAR' and new_name == 'VER' and self._is_number(param.default):
                        var_val = float(param.default.val) if hasattr(param.default, 'val') else float(param.default)
                        if abs(var_val) < 1e-12:  # VAR = 0
                            # Set VER to large value (100000V) instead of skipping - helps Xyce convergence
                            ver_val = Float(100000.0)
                            add_param('VER', ver_val, param.distr, comment="var=0 means infinite/unused in SPICE, VER set to 100000V for convergence")
                            continue
                    
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
                            elif self._is_number(default_val):
                                # Check if we should preserve zero for this parameter
                                preserve_zero = preserve_zero_map.get(name_upper, False)
                                # Normal clamping
                                clamped = self._clamp_param_value(default_val, min_val, max_val, min_exclusive, max_exclusive, preserve_zero=preserve_zero)
                                # Only use clamped value if it's a numeric literal (clamping worked)
                                if self._is_number(clamped):
                                    # Check if value actually changed (use >= to catch epsilon changes)
                                    if isinstance(default_val, (Int, Float)) and isinstance(clamped, (Int, Float)):
                                        old_val = default_val.val if hasattr(default_val, 'val') else default_val
                                        new_val = clamped.val if hasattr(clamped, 'val') else clamped
                                        # Special case: VER cannot be 0 in Xyce, use minimum 0.01 if clamped to 0
                                        if name_upper == 'VAR' and new_name == 'VER' and abs(new_val) < 1e-12:
                                            new_val = 0.01
                                            clamped = Float(new_val)
                                            clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val} - Xyce requires VER > 0)")
                                        elif abs(old_val - new_val) >= 1e-12:
                                            clamped_params.append(f"{param.name.name} (clamped from {old_val} to {new_val})")
                                        default_val = clamped
                                else:
                                    default_val = clamped
                            else:
                                # Expression - DO NOT wrap in max()/min() for expressions
                                # Xyce has trouble with max()/min() during DC analysis convergence
                                # Range constraints are only enforced for numeric literals
                                # Expressions are assumed to evaluate correctly (or will fail at runtime if invalid)
                                # This matches the working simplified model which uses pre-evaluated numeric values
                                pass  # Keep expression as-is without max()/min() wrapping
                    
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

    def _filter_diode_level1_unsupported_model_params(
        self, params: List[ParamDecl], *, model_name: str
    ) -> List[ParamDecl]:
        """Drop diode model parameters which Xyce Level 1 does not recognize.

        This is intentionally conservative: we only drop parameters we *know* Xyce
        rejects (producing noisy warnings), and we log a single warning per model.
        """
        # These are the exact params we see Xyce warning about in practice.
        # Keep the list uppercase for easy case-insensitive matching.
        unsupported = {
            "ISW",
            "PJ",
            "MJ",
            "AREA",
            "TLEVC",
            "CTA",
            "CTP",
            "TLEV",
            "PTA",
            "PTP",
            "NZ",
            "IMAX",
            "MINR",
            "ALLOW_SCALING",
        }

        kept: List[ParamDecl] = []
        dropped: List[str] = []
        for p in params:
            name_upper = p.name.name.upper()
            if name_upper in unsupported:
                dropped.append(p.name.name)
                continue
            kept.append(p)

        if dropped:
            self.log_warning(
                "Dropping unsupported Xyce diode (Level 1) model parameters: "
                + ", ".join(sorted({d.upper() for d in dropped})),
                f"Model: {model_name}",
            )

        return kept

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
        
        # Check if we're in a subcircuit and if this model should have its suffix removed
        # Rule: If there's only one model, remove any numeric suffix (.0, .1, .2, etc.)
        #       If there are multiple models, remove suffix from the first variant (lowest number)
        # (Xyce doesn't match numeric variant suffixes when instances reference base name)
        from ..data import ModelDef, ModelFamily, Ident
        import re
        model_name = model.name.name
        if self._current_subckt is not None:
            # Check if model name ends with a numeric suffix (e.g., .0, .1, .2, .28)
            suffix_match = re.match(r'^(.+)\.(\d+)$', model_name)
            if suffix_match:
                model_base = suffix_match.group(1)
                suffix_num = int(suffix_match.group(2))
                
                # Find all models with the same base name and their suffix numbers
                # Include the current model in the count
                variant_suffixes = []
                for entry in self._current_subckt.entries:
                    if isinstance(entry, ModelDef):
                        entry_name = entry.name.name
                        entry_base = entry_name.split('.')[0]
                        if entry_base == model_base:
                            # Extract suffix number if present
                            entry_suffix_match = re.match(r'^.+\.(\d+)$', entry_name)
                            if entry_suffix_match:
                                entry_suffix_num = int(entry_suffix_match.group(1))
                                variant_suffixes.append(entry_suffix_num)
                            else:
                                # No suffix - this means suffix was already removed
                                # Don't count it as a variant
                                pass
                    elif isinstance(entry, ModelFamily):
                        # ModelFamily: check if base name matches
                        family_base = entry.name.name.split('.')[0]
                        if family_base == model_base:
                            # For ModelFamily, we need to check variants
                            # This is more complex - for now, just count them
                            variant_suffixes.extend([i+1 for i in range(len(entry.variants))])
                
                # Also include the current model's suffix in the list
                variant_suffixes.append(suffix_num)
                
                # Remove duplicates and sort to find the lowest suffix number
                variant_suffixes = sorted(set(variant_suffixes))
                
                # If there's only one model, remove its suffix
                # If there are multiple models, remove suffix from the first one (lowest number)
                if len(variant_suffixes) == 1:
                    # Only one model - remove suffix
                    model_name = model_base
                    model.name = Ident(name=model_name)
                elif len(variant_suffixes) > 1 and suffix_num == variant_suffixes[0]:
                    # Multiple models - remove suffix from the first one (lowest number)
                    model_name = model_base
                    model.name = Ident(name=model_name)
        
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
                    if mtype_lower in ("npn", "pnp") and current_level == 1 and output_level == 1:
                        # Keep Level 1, just filter unsupported parameters
                        params_to_write = self._map_bjt_level1_to_level1_params(params_to_write, model_name=mname)
                    elif mtype_lower in ("npn", "pnp") and current_level == 1 and output_level == 504:
                        # Log summary warning about BJT conversion
                        self.log_warning(
                            f"Converting BJT model '{mname}' from Level 1 (Gummel-Poon) to Level 504 (MEXTRAM). "
                            "Many Level 1 parameters are not compatible with Xyce MEXTRAM and will be commented out. "
                            "See detailed warnings below.",
                            f"Model: {mname}"
                        )
                        # Map level 1 BJT parameters to MEXTRAM (level 504)
                        params_to_write = self._map_bjt_level1_to_mextram_params(params_to_write, model_name=mname)
                    elif mtype_lower in ("diode", "d") and current_level == 3 and output_level == 1:
                        # Convert diode Level 3 to Level 1
                        self.log_warning(
                            f"Converting diode model '{mname}' from Level 3 to Level 1. "
                            "Level 3-specific parameters will be dropped. "
                            "Level 1 supports area but not perim as instance parameters.",
                            f"Model: {mname}"
                        )
                        params_to_write = self._map_diode_level3_to_level1_params(params_to_write, model_name=mname)
                    
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
            
            # Map deltox to dtox and drop TYPE parameter (Spectre-specific, not in Xyce)
            mapped_params = []
            for p in params_to_write:
                param_name = self._get_param_name(p)
                if param_name == "deltox":
                    # Create new ParamDecl with name "dtox" and same value
                    p_default = self._get_param_default(p)
                    p_distr = p[0].distr if isinstance(p, tuple) else p.distr
                    mapped_params.append(ParamDecl(name=Ident("dtox"), default=p_default, distr=p_distr))
                elif param_name.upper() == "TYPE":
                    # TYPE is Spectre-specific parameter, drop it (not in Xyce)
                    self.log_warning(
                        f"BSIM4 parameter 'TYPE' dropped (Spectre-specific, not in Xyce BSIM4)",
                        f"Model: {mname}"
                    )
                    # Don't add to mapped_params (effectively drops it)
                else:
                    mapped_params.append(p)
            params_to_write = mapped_params
            
            # Add level=54 at the beginning if not already present
            if not has_level:
                level_param = ParamDecl(name=Ident("level"), default=Float(54.0), distr=None)
                params_to_write.insert(0, level_param)

        # Drop MOS model parameters which Xyce does not recognize.
        # (These otherwise generate noisy warnings during parsing.)
        if mtype in ("nmos", "pmos"):
            dropped = []
            kept = []
            for p in params_to_write:
                pname = self._get_param_name(p).lower()
                if pname == "minr":
                    dropped.append(self._get_param_name(p))
                    continue
                kept.append(p)
            if dropped:
                self.log_warning(
                    "Dropping unsupported Xyce MOS model parameter(s): "
                    + ", ".join(sorted({d.upper() for d in dropped})),
                    f"Model: {mname}",
                )
            params_to_write = kept
        
        # Handle BJT models (NPN and PNP): keep level=1 (Gummel-Poon) if not already set
        # Only apply default behavior if level mapping was not already applied
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
            
            # Add level=1 at the beginning if not already present
            if not has_level:
                level_param = ParamDecl(name=Ident("level"), default=Int(1), distr=None)
                params_to_write.insert(0, level_param)
                # Filter parameters to keep only Level 1 supported ones
                params_to_write = self._map_bjt_level1_to_level1_params(params_to_write, model_name=mname)
            else:
                # If level exists, check its value
                for i, p in enumerate(params_to_write):
                    if self._get_param_name(p) == "level":
                        # Check current level value
                        current_level = None
                        default_val = self._get_param_default(p)
                        if isinstance(default_val, (Int, Float)):
                            current_level = float(default_val.val) if hasattr(default_val, 'val') else float(default_val)
                        elif isinstance(default_val, MetricNum):
                            current_level = float(default_val.val)
                        
                        if current_level is not None:
                            if current_level == 1.0:
                                # Already level 1, filter parameters
                                params_to_write = self._map_bjt_level1_to_level1_params(params_to_write, model_name=mname)
                            elif abs(current_level - 504.0) < 0.1:  # Level 504
                                # Keep level 504 if explicitly set (for backward compatibility)
                                # But this shouldn't happen in normal flow now
                                pass
                        break
        elif mtype in ("npn", "pnp") and level_mapping_applied:
            # Level mapping was applied - check what level it resulted in
            if output_level == 1:
                # Filter parameters if needed
                params_to_write = self._map_bjt_level1_to_level1_params(params_to_write, model_name=mname)
        
        # Map to Xyce model type equivalent
        xyce_mtype = xyce_mtype_map.get(mtype, mtype)

        # If this is a diode model targeting Xyce's diode model (D), drop the
        # Spectre/PDK extension parameters Xyce rejects to avoid warning spam.
        #
        # Apply only for Level 1 diodes (including those explicitly mapped to level=1).
        if str(xyce_mtype).upper() == "D":
            # Determine diode model level (default to 1 if not specified)
            diode_level = 1
            for p in params_to_write:
                if isinstance(p, tuple):
                    p_obj = p[0]
                else:
                    p_obj = p
                if p_obj.name.name.lower() == "level":
                    lvl = self._get_param_default(p_obj)
                    if isinstance(lvl, (Int, Float)):
                        diode_level = int(float(lvl.val) if hasattr(lvl, "val") else float(lvl))
                    elif isinstance(lvl, MetricNum):
                        diode_level = int(float(lvl.val))
                    break

            if diode_level == 1:
                # Only filter ParamDecl objects (diode path does not use the tuple form)
                decls = [p[0] if isinstance(p, tuple) else p for p in params_to_write]
                filtered = self._filter_diode_level1_unsupported_model_params(decls, model_name=mname)
                params_to_write = filtered
        
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
        
        # If we're in a FET subcircuit, models can now access swx_nrds/swx_vth via .param aliases
        # No need to replace references - the .param aliases reference PARAMS, so models can use the same names
        # The .param statements (e.g., .param swx_nrds={swx_nrds}) create aliases that models can access
        
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
            
            # For BJT models with level=504, uppercase parameter names (MEXTRAM convention)
            # For BJT models with level=1, keep lowercase parameter names (Gummel-Poon convention)
            # Note: Parameters from _map_bjt_level1_to_mextram_params are already uppercase
            # Note: Parameters from _map_bjt_level1_to_level1_params keep original case
            # Check if this is a Level 504 model by checking if level parameter exists and equals 504
            is_level_504 = False
            for p in params_to_write:
                if isinstance(p, tuple):
                    p_param = p[0]
                else:
                    p_param = p
                if p_param.name.name.lower() == "level":
                    level_val = self._get_param_default(p_param)
                    if isinstance(level_val, (Int, Float)):
                        level_num = int(float(level_val.val) if hasattr(level_val, 'val') else float(level_val))
                        if level_num == 504:
                            is_level_504 = True
                            break
            
            if is_level_504 and param.name.name != "level":
                # Create a temporary param with uppercase name for writing (MEXTRAM)
                param_upper = ParamDecl(
                    name=Ident(param.name.name.upper()),
                    default=param.default,
                    distr=param.distr
                )
                self.write_param_decl(param_upper)
            else:
                # Level 1 or other models - keep original case
                self.write_param_decl(param)
                
            if comment:
                self.write(f" ; {comment}")
                
            self.write("\n")
        
        self.write("\n")  # Ending blank-line
