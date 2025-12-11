""" 
# Spice Format Netlisting

"Spice-format" is a bit of a misnomer in netlist-world. 
Of the countless Spice-class simulators have been designed the past half-century, 
most have a similar general netlist format, including: 

* Simulation input comprises a file-full of: 
  * (a) Circuit elements, arranged in `vlsir.Module`s, and
  * (b) Simulator control "cards", such as analysis statements, global parameters, measurements, probes, and the like.
* Circuit-specification is aided by hierarchy, generally in the form of "sub-circuits", denoted `SUBCKT`. 
  * Sub-circuits can commonly be parameterized, and can use a limited set of "parameter programming" to maniupulate their own parameter-values into those of their child-instances. 
  * For example, an instance might be declared as: `xdiode p n area=`width*length``, where `width` and `length` are parameters or its parent.
* `Signal`s are all scalar nets, which are created "out of thin air" whenever referenced. 
* "Typing" performed by instance-name prefixes, e.g. instances named `r1` being interpreted as resistors. 
* Many other subtleties, such as the typical case-insensitivity of netlist content (e.g. `V1` and `v1` are the same net). 

However each simulator also differs in ways large and small. 
Common differences manifest in the areas of: 

* How sub-circuits parameters are declared, and correspondingly how instance-parameter values are set. 
  * Sandia Lab's *Xyce* differs in a prominent fashion, adding a `PARAMS:` keyword where declarations and values begin. 
* How arithmetic expressions are specified, and what functions and expressions are available.
  * Common methods include back-ticks (Hspice) and squiggly-brackets (NgSpice).
  * Notably the asterisk-character (`*`) is the comment-character in many of these formats, and must be wrapped in an expression to perform multiplication. 
* The types and locations of *comments* that are supported. 
  * Some include the fun behavior that comments beginning mid-line require *different* comment-characters from those starting at the beginning of a line.
* While not an HDL attribute, they often differ even more in how simulation control is specified, particularly in analysis and saving is specified. 

"Spice" netlisting therefore requires a small family of "Spice Dialects", 
heavily re-using a central `SpiceNetlister` class, but requiring simulator-specific implementation details. 

"""

# Std-Lib Imports
import sys
from enum import Enum
from warnings import warn
from typing import Tuple, Union, List, get_args, Dict, IO, Optional

# Local Imports
from ..data import (
    Program,
    SubcktDef,
    Instance,
    ParamDecl,
    Ident,
    Float,
    ParamVal,
    Expr,
    Ref,
    Primitive,
    Int,
    MetricNum,
    Options,
    StatisticsBlock,
    ParamDecls,
    ModelFamily,
    ModelVariant,
    ModelDef,
    LibSectionDef,
    Comment,
    Include,
    UseLibSection,
    Entry,
    Call,
    BinaryOp,
    BinaryOperator,
    UnaryOp,
    TernOp,
    QuotedString,
    Variation,
    NetlistDialects,
    FunctionDef,  # Added FunctionDef to imports
    Return,       # Added Return to imports (needed for FunctionDef logic)
)
from .base import Netlister, ErrorMode


class SpiceNetlister(Netlister):
    """
    # "Generic" Spice Netlister
    and base-class for Spice dialects.

    Performs nearly all data-model traversal,
    offloading syntax-specifics to dialect-specific sub-classes.

    Attempts to write only the "generic" subset of Spice-content,
    in the "most generic" methods as perceived by the authors.
    This may not work for *any* particular simulator; see the simulator-specific dialects below,
    and the module-level commentary above for more on why.
    """

    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.SPICE

    def write_subckt_def(self, module: SubcktDef) -> None:
        """Write the `SUBCKT` definition for `Module` `module`."""
        # Store current subcircuit context for use in write_subckt_instance
        self._current_subckt = module

        # Create the module name
        module_name = self.format_ident(module.name)

        # Create the sub-circuit definition header
        self.write(f".SUBCKT {module_name} \n")

        # Create its ports, if any are defined
        if module.ports:
            self.write_port_declarations(module)
        else:
            self.write("+ ")
            self.write_comment("No ports")

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
        
        # Create its parameters, if any are defined
        if module.params:
            self.write_module_params(module.params)
        else:
            self.write("+ ")
            self.write_comment("No parameters")

        # End the `subckt` header-content with a blank line
        self.write("\n")

        # Write its internal content/ entries
        for entry in module.entries:
            self.write_entry(entry)

        # And close up the sub-circuit
        self.write(".ENDS\n\n")
        # Clear current subcircuit context
        self._current_subckt = None

    def write_port_declarations(self, module: SubcktDef) -> None:
        """Write the port declarations for Module `module`."""
        self.write("+ ")
        for port in module.ports:
            self.write(self.format_ident(port) + " ")
        self.write("\n")

    def write_module_params(self, params: List[ParamDecl]) -> None:
        """Write the parameter declarations for Module `module`.
        Parameter declaration format: `name1=val1 name2=val2 name3=val3 \n`"""
        self.write("+ ")
        for param in params:
            self.write_param_decl(param)
            self.write(" ")
        self.write("\n")

    def write_subckt_instance(self, pinst: Instance) -> None:
        """Write sub-circuit-instance `pinst`."""
        from ..data import ParamVal, Ref, Ident
        # Detect if the instance looks like a MOS device (even if parsed as subcircuit instance)
        conn_count = len(pinst.conns)
        has_l_w = {'l', 'w'} <= {p.name.name for p in pinst.params}
        is_module_ref = isinstance(pinst.module, Ref)
        name_has_mos = any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet'])
        name_has_bjt = any(keyword in pinst.name.name.lower() for keyword in ['npn', 'pnp', 'bjt', 'q'])
        name_has_res = any(keyword in pinst.name.name.lower() for keyword in ['res', 'rc', 'r'])
        name_has_diode = any(keyword in pinst.name.name.lower() for keyword in ['diode', 'd', 'dn'])
        
        # Check if module name suggests resistor or diode
        module_name_res = False
        module_name_diode = False
        if is_module_ref:
            module_name = pinst.module.ident.name.lower()
            module_name_res = any(keyword in module_name for keyword in ['resistor', 'res', 'r'])
            module_name_diode = any(keyword in module_name for keyword in ['diode', 'd'])
        
        is_mos_like = (
            conn_count == 4  # Exactly 4 ports (d, g, s, b)
            and is_module_ref  # References a model
            and has_l_w  # Has both 'l' and 'w' params
            and name_has_mos  # Instance name indicates MOS
        )
        
        # Detect if the instance looks like a BJT device (even if parsed as subcircuit instance)
        is_bjt_like = (
            conn_count == 4  # Exactly 4 ports (c, b, e, s)
            and is_module_ref  # References a model
            and name_has_bjt  # Instance name indicates BJT
        )
        
        # Detect if the instance looks like a resistor (even if parsed as subcircuit instance)
        # Resistors typically have 2 ports and reference a resistor model
        has_r_param = 'r' in {p.name.name.lower() for p in pinst.params}
        is_res_like = (
            conn_count == 2  # Exactly 2 ports
            and (is_module_ref and module_name_res) or (not is_module_ref and (name_has_res or has_r_param))
        )
        
        # Detect if the instance looks like a diode (even if parsed as subcircuit instance)
        # Diodes typically have 2 ports and reference a diode model
        has_diode_params = any(p.name.name.lower() in ['area', 'perim', 'pj'] for p in pinst.params)
        is_diode_like = (
            conn_count == 2  # Exactly 2 ports
            and (is_module_ref and module_name_diode) or (not is_module_ref and (name_has_diode or has_diode_params))
        )

        prefix = 'M' if is_mos_like else ('Q' if is_bjt_like else ('R' if is_res_like else ('D' if is_diode_like else 'X')))

        inst_name = self.format_ident(pinst.name)
        if prefix and not inst_name.upper().startswith(prefix):
            inst_name = f"{prefix}{inst_name}"

        # Write the instance name
        self.write(inst_name + " \n")

        # Write its port-connections
        self.write_instance_conns(pinst)

        # For resistors and diodes detected as primitives, don't write the module name (they're primitives)
        # For other devices, write the sub-circuit/model name
        if not is_res_like and not is_diode_like:
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
        
        # Write its parameter values
        # For resistors, replace 'temp' with 'temper' in expressions (ngspice uses 'temper' as built-in)
        if is_res_like:
            # Create modified params with temp->temper replacement
            modified_params = []
            for pval in pinst.params:
                if isinstance(pval.val, Expr):
                    # Replace temp references with temper in expressions
                    modified_val = self._replace_temp_with_temper(pval.val)
                    modified_params.append(ParamVal(name=pval.name, val=modified_val))
                else:
                    modified_params.append(pval)
            self.write_instance_params(modified_params)
        else:
            self.write_instance_params(pinst.params)

        # Add a blank after each instance
        self.write("\n")

    def write_primitive_instance(self, pinst: Instance) -> None:
        """Write primitive-instance `pinst` of `rmodule`.
        Note spice's primitive instances often differn syntactically from sub-circuit instances,
        in that they can have positional (only) parameters."""

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

        # Write ports (exluding last (model)) on a separate continuation line
        # For ngspice, ensure node names are written without braces
        self.write("+ ")
        for arg in pinst.args[:-1]:  # Ports only (exclude the model)
            if isinstance(arg, Ident):
                conn_name = arg.name
            elif isinstance(arg, Ref):
                conn_name = arg.ident.name
            elif isinstance(arg, (Int, Float, MetricNum)):
                self.write(self.format_number(arg) + " ")
                continue
            else:
                # For other types, try to extract name, but fall back to format_expr
                # Then remove braces if present
                conn_name = self.format_expr(arg)
                if conn_name.startswith('{') and conn_name.endswith('}'):
                    conn_name = conn_name[1:-1]
            
            # Write node name without braces (SPICE standard)
            if isinstance(arg, (Ident, Ref)) or (not isinstance(arg, (Int, Float, MetricNum))):
                self.write(conn_name + " ")
        self.write("\n")
        # Write the model (last arg) on another continuation line
        self.write("+ " + self.format_ident(pinst.args[-1]) + " \n")

        self.write("+ ")
        for kwarg in pinst.kwargs:
            self.write_param_val(kwarg)
            self.write(" ")

        self.write("\n")

    def write_instance_conns(self, pinst: Instance) -> None:
        """Write the port-connections for Instance `pinst`"""

        # Write a quick comment for port-less modules
        if not len(pinst.conns):
            self.write("+ ")
            return self.write_comment("No ports")

        if isinstance(pinst.conns[0], tuple):
            # FIXME: connections by-name are not supported.
            raise RuntimeError(f"Unsupported by-name connections on {pinst}")

        self.write("+ ")
        # And write the Instance ports, in that order
        for pconn in pinst.conns:
            self.write(self.format_ident(pconn) + " ")
        self.write("\n")

    def write_instance_params(self, pvals: List[ParamVal]) -> None:
        """
        Format and write the parameter-values in dictionary `pvals`.

        Parameter-values format:
        ```
        XNAME
        + <ports>
        + <subckt-name>
        + name1=val1 name2=val2 name3=val3
        """

        self.write("+ ")

        if not pvals:  # Write a quick comment for no parameters
            return self.write_comment("No parameters")

        # And write them
        for pval in pvals:
            self.write_param_val(pval)
            self.write(" ")

        self.write("\n")

    @classmethod
    def format_bus_bit(cls, index: Union[int, str]) -> str:
        """Format-specific string-representation of a bus bit-index"""
        # Spectre netlisting uses an underscore prefix, e.g. `bus_0`
        return "_" + str(index)

    def write_param_decl(self, param: ParamDecl) -> str:
        """Format a parameter declaration"""

        if param.distr is not None:
            msg = f"Unsupported `distr` for parameter {param.name} will be ignored"
            warn(msg)
            self.write("\n+ ")
            self.write_comment(msg)
            self.write("\n+ ")

        if param.default is None:
            msg = f"Required (non-default) parameter {param.name} is not supported by {self.__class__.__name__}. "
            msg += "Setting to maximum floating-point value {sys.float_info.max}, which almost certainly will not work if instantiated."
            warn(msg)
            default = str(sys.float_info.max)
        else:
            default = self.format_expr(param.default)

        self.write(f"{self.format_ident(param.name)}={default}")
        
        # Write inline comment if present
        if param.comment:
            self.write(f" ; {param.comment}")

    def write_param_val(self, param: ParamVal) -> None:
        """Write a parameter value"""

        name = self.format_ident(param.name)
        val = self.format_expr(param.val)
        self.write(f"{name}={val}")
        
        # Write inline comment if present (only for ParamVal, not Option)
        if hasattr(param, 'comment') and param.comment:
            self.write(f" ; {param.comment}")

    def write_comment(self, comment: str) -> None:
        """While dialects vary, the *generic* Spice-comment begins with the asterisk."""
        self.write(f"* {comment}\n")

    def write_options(self, options: Options) -> None:
        """Write Options `options`"""
        if options.name is not None:
            msg = f"Warning invalid `name`d Options"
            warn(msg)
            self.write_comment(msg)
            self.write("\n")

        # Get to the actual option-writing
        self.write(".option \n")
        for option in options.vals:
            self.write("+ ")
            # FIXME: add the cases included in `OptionVal` but not `ParamVal`, notably quoted string-paths
            self.write_param_val(option)
            self.write(" \n")
        self.write("\n")

    def write_statistics_block(self, stats: StatisticsBlock) -> None:
        """Write statistics block (no-op for Spice formats).
        
        For Spice/Xyce formats, statistics variations are applied to parameters
        during write preparation, so StatisticsBlock entries are not written directly.
        For Spectre format, this should be overridden to write the statistics block.
        """
        # No-op: Statistics blocks are processed by apply_statistics_variations() before writing
        # (for Xyce) or should be written directly (for Spectre - but SpectreNetlister should override this)
        pass

    def write_param_decls(self, params: ParamDecls) -> None:
        """Write parameter declarations"""
        for p in params.params:
            self.write(".param ")
            self.write_param_decl(p)
            # Note: write_param_decl already writes inline comments if present
            self.write("\n")
        # Don't add extra newline - let BlankLine entries handle spacing
        # Note: Comments that appear within parameter blocks are handled
        # by the netlist() method in base.py, which writes entries in order
    
    def write_param_decls_with_inline_comments(self, params: ParamDecls, inline_comments) -> None:
        """Write parameter declarations with inline comments.
        
        If parameters already have inline comments stored in ParamDecl.comment,
        those are written by write_param_decl. This method handles any additional
        inline comments from the inline_comments list that weren't captured as ParamDecl.comment.
        """
        from ..data import Comment
        
        # Collect comments that are already stored inline in params
        param_comments = {p.comment for p in params.params if p.comment}
        
        # Write parameters - write_param_decl already writes inline comments from ParamDecl.comment
        for p in params.params:
            self.write(".param ")
            self.write_param_decl(p)
            self.write("\n")
        
        # Write any remaining inline_comments that don't match param comments
        # (these would be comments that weren't captured as inline comments during parsing)
        for comment in inline_comments:
            if isinstance(comment, Comment) and comment.text not in param_comments:
                self.write_comment(comment.text)

    def write_model_family(self, mfam: ModelFamily) -> None:
        """Write a model family"""
        # Just requires writing each variant.
        # They will be output with `modelname.variant` names, as most SPICE formats want.
        for variant in mfam.variants:
            self.write_model_variant(variant)

    def write_model_variant(self, mvar: ModelVariant) -> None:
        """Write a model variant"""

        # This just convertes to a `ModelDef` with a dot-separated name, and running `write_model_def`.
        model = ModelDef(
            name=Ident(f"{mvar.model.name}.{mvar.variant.name}"),
            mtype=mvar.mtype,
            args=mvar.args,
            params=mvar.params,
        )
        return self.write_model_def(model)

    def write_model_def(self, model: ModelDef) -> None:
        """Write a model definition"""

        mname = self.format_ident(model.name)
        mtype = self.format_ident(model.mtype)

        if mtype.lower() == "bsim4":
            self.writeln(f".model {mname}")
        else:
            self.writeln(f".model {mname} {mtype}")

        self.write("+ ")

        for arg in model.args:
            self.write(self.format_expr(arg) + " ")

        self.write("\n")
        for param in model.params:
            self.write("+ ")
            self.write_param_decl(param)
            self.write("\n")

        self.write("\n")  # Ending blank-line

    def write_library_section(self, section: LibSectionDef) -> None:
        """Write a Library Section definition"""
        self.write(f".lib {self.format_ident(section.name)}\n")
        for entry in section.entries:
            self.write_entry(entry)
        self.write(f".endl {self.format_ident(section.name)}\n\n")

    def write_include(self, inc: Include) -> None:
        """Write a file-Include"""
        # Format: `.include {path} `
        self.write(f".include {str(inc.path)}\n\n")

    def write_use_lib(self, uselib: UseLibSection) -> None:
        """Write a sectioned Library-usage"""
        # Format: `.lib {path} {section}`
        # Note quotes here are interpreted as part of `section`, and generally must be avoided.
        self.write(f".lib {str(uselib.path)} {self.format_ident(uselib.section)} \n\n")


class HspiceNetlister(SpiceNetlister):
    """
    # Hspice-Format Netlister

    Other than its `NetlistFormat` enumeration, `HspiceNetlister` is identical to the base `SpiceNetlister`.
    """

    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.HSPICE


class NgspiceNetlister(SpiceNetlister):
    """Ngspice-Format Netlister
    
    ngspice uses standard SPICE syntax with some specific features:
    - Expressions use { } delimiters (same as Xyce)
    - Comments use * (standard SPICE, not ; like Xyce)
    - No PARAMS: keyword for subcircuit parameters
    - Parameter functions use .param func_name(args) = {expression} syntax
    """

    def write_instance_conns(self, pinst: Instance) -> None:
        """Write the port-connections for Instance `pinst`.
        
        Override to ensure node names are written without braces (ngspice interprets braces as expressions).
        """
        from ..data import Ref, Ident, Expr
        
        # Write a quick comment for port-less modules
        if not len(pinst.conns):
            self.write("+ ")
            return self.write_comment("No ports")

        if isinstance(pinst.conns[0], tuple):
            # FIXME: connections by-name are not supported.
            raise RuntimeError(f"Unsupported by-name connections on {pinst}")

        self.write("+ ")
        # And write the Instance ports, in that order
        # For ngspice, ensure node names are written without braces
        # Node names should NEVER have braces - braces are for expressions/parameters only
        for pconn in pinst.conns:
            # Extract the identifier name directly without any expression formatting
            # Connections are always identifiers (node names), never expressions
            # Check Ident first (most common)
            if isinstance(pconn, Ident):
                conn_name = pconn.name
            # Check Ref second (Ref is a subclass of Expr, so check before Expr)
            elif isinstance(pconn, Ref):
                conn_name = pconn.ident.name
            # Check if it has a name attribute (but Ref and Ident are already handled above)
            elif hasattr(pconn, 'name'):
                try:
                    # Try to get name directly
                    name_attr = getattr(pconn, 'name')
                    # If it's a property/method, call it; otherwise use the value
                    if callable(name_attr):
                        conn_name = name_attr()
                    else:
                        conn_name = name_attr
                except:
                    conn_name = None
            # Check if it has an ident attribute (Ref-like)
            elif hasattr(pconn, 'ident'):
                try:
                    ident = getattr(pconn, 'ident')
                    if hasattr(ident, 'name'):
                        name_attr = getattr(ident, 'name')
                        if callable(name_attr):
                            conn_name = name_attr()
                        else:
                            conn_name = name_attr
                    else:
                        conn_name = None
                except:
                    conn_name = None
            else:
                conn_name = None
            
            # If we still don't have a name, use format_ident as last resort
            if conn_name is None:
                conn_name = self.format_ident(pconn)
            
            # Ensure conn_name is a string
            if not isinstance(conn_name, str):
                conn_name = str(conn_name)
            
            # CRITICAL: Remove any braces that might have been added
            # SPICE node names must NEVER have braces - braces are only for expressions
            # This is a safety check in case format_ident or something else added braces
            if conn_name.startswith('{') and conn_name.endswith('}'):
                conn_name = conn_name[1:-1]
            
            # Write node name without braces (SPICE standard)
            self.write(conn_name + " ")
        self.write("\n")

    def __init__(self, src: Program, dest: IO, *, errormode: ErrorMode = ErrorMode.RAISE, file_type: str = "", includes: List[Tuple[str, str]] = None, model_file: str = None, model_level_mapping: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> None:
        super().__init__(src, dest, errormode=errormode, file_type=file_type)
        self.includes = includes or []
        self.model_file = model_file
        # Process model_level_mapping: convert lists to dicts for efficient lookup
        self._model_level_mapping: Dict[str, Dict[int, int]] = {}
        if model_level_mapping:
            for key, mappings in model_level_mapping.items():
                self._model_level_mapping[key.lower()] = {in_level: out_level for in_level, out_level in mappings}

    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.NGSPICE

    def expression_delimiters(self) -> Tuple[str, str]:
        """Return the starting and closing delimiters for expressions.
        ngspice uses curly braces like Xyce."""
        return ("{", "}")

    def format_expr(self, expr: Expr) -> str:
        """Format an expression for ngspice.
        
        Overrides base implementation to ensure Ref objects (parameters) 
        are wrapped in curly braces, e.g. {Vdd} instead of just Vdd.
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

    def write_comment(self, comment: str) -> None:
        """ngspice uses standard SPICE comment syntax with asterisk."""
        self.write(f"* {comment}\n")

    def _format_expr_inner(self, expr: Expr) -> str:
        """Override to replace round() with nint() for ngspice compatibility."""
        if isinstance(expr, Call):
            func = self.format_ident(expr.func)
            # Replace round() with nint() for ngspice compatibility
            if func.lower() == "round":
                func = "nint"
            args = [self._format_expr_inner(arg) for arg in expr.args]
            return f"{func}({','.join(args)})"
        
        # For all other expression types, call parent implementation
        return super()._format_expr_inner(expr)

    def write_function_def(self, func: FunctionDef) -> None:
        """Write a ngspice .param function definition for FunctionDef `func`.

        Format: .param func_name(arg1, arg2, ...) = { expression }
        """
        func_name = self.format_ident(func.name)

        # Write function signature
        self.write(f".param {func_name}(")

        # Write arguments
        if func.args:
            arg_names = [self.format_ident(arg.name) for arg in func.args]
            self.write(",".join(arg_names))

        self.write(") = { ")

        # Write function body - should be a single Return statement
        if func.stmts and len(func.stmts) == 1 and isinstance(func.stmts[0], Return):
            expr_str = self.format_expr(func.stmts[0].val)
            # Remove outer braces if present (format_expr adds them, but we're already in { })
            if expr_str.startswith("{") and expr_str.endswith("}"):
                expr_str = expr_str[1:-1]
            self.write(expr_str)
        else:
            # Fallback: handle multiple statements (shouldn't happen for our use case)
            self.handle_error(func, "FunctionDef with multiple statements not supported")

        self.write(" }\n")

    def write_subckt_def(self, module: SubcktDef) -> None:
        """Write the `SUBCKT` definition for `Module` `module`.
        
        Overrides base class to put ports on the same line as .SUBCKT (standard SPICE format).
        """
        # Store current subcircuit context for use in write_subckt_instance
        self._current_subckt = module

        # Create the module name
        module_name = self.format_ident(module.name)

        # Create the sub-circuit definition header with ports on same line (standard SPICE)
        if module.ports:
            port_names = [self.format_ident(port) for port in module.ports]
            self.write(f".SUBCKT {module_name} {' '.join(port_names)}\n")
        else:
            self.write(f".SUBCKT {module_name}\n")
            self.write("+ ")
            self.write_comment("No ports")
            self.write("\n")

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
        
        # Create its parameters, if any are defined
        if module.params:
            self.write_module_params(module.params)
        else:
            self.write("+ ")
            self.write_comment("No parameters")
            self.write("\n")

        # End the `subckt` header-content with a blank line
        self.write("\n")

        # For ngspice: Check if any models inside this subcircuit reference subcircuit parameters
        # If so, add .param statements for those parameters to ensure they're in scope when models are evaluated
        if module.params:
            from ..data import ModelDef, ModelFamily, Ref
            
            # Collect parameter names from subcircuit
            subckt_param_names = {p.name.name for p in module.params}
            
            # Check all entries for models and instances that reference subcircuit parameters
            params_needed_as_param_statements = set()
            for entry in module.entries:
                if isinstance(entry, ModelDef):
                    # Check all model parameters for references to subcircuit parameters
                    for param in entry.params:
                        if param.default is not None and isinstance(param.default, Expr):
                            # Check if this expression references any subcircuit parameter
                            for subckt_param_name in subckt_param_names:
                                if expr_references_param(param.default, subckt_param_name):
                                    params_needed_as_param_statements.add(subckt_param_name)
                                    break
                elif isinstance(entry, ModelFamily):
                    # Check all variants
                    for variant in entry.variants:
                        for param in variant.params:
                            if param.default is not None and isinstance(param.default, Expr):
                                for subckt_param_name in subckt_param_names:
                                    if expr_references_param(param.default, subckt_param_name):
                                        params_needed_as_param_statements.add(subckt_param_name)
                                        break
                elif isinstance(entry, Instance):
                    # Check instance parameters for references to subcircuit parameters
                    for pval in entry.params:
                        # Check Ref first (it's a subclass of Expr, so order matters)
                        if isinstance(pval.val, Ref):
                            # Direct parameter reference (e.g., nrd={nrd})
                            if pval.val.ident.name in subckt_param_names:
                                params_needed_as_param_statements.add(pval.val.ident.name)
                        elif isinstance(pval.val, Expr):
                            # Complex expression that might reference subcircuit parameters
                            for subckt_param_name in subckt_param_names:
                                if expr_references_param(pval.val, subckt_param_name):
                                    params_needed_as_param_statements.add(subckt_param_name)
                                    break
            
            # Add .param statements for parameters that are referenced in models
            # This ensures they're available when ngspice evaluates the model parameters
            if params_needed_as_param_statements:
                for param_name in sorted(params_needed_as_param_statements):  # Sort for deterministic output
                    # Find the parameter definition
                    for param in module.params:
                        if param.name.name == param_name:
                            # Write as .param statement
                            self.write(".param ")
                            self.write_param_decl(param)
                            self.write("\n")
                            break

        # Write its internal content/ entries
        for entry in module.entries:
            self.write_entry(entry)

        # And close up the sub-circuit
        self.write(".ENDS\n\n")
        # Clear current subcircuit context
        self._current_subckt = None

    def write_module_params(self, params: List[ParamDecl]) -> None:
        """Write the parameter declarations for Module parameters.
        
        For ngspice, parameters are written on continuation lines without spaces around =.
        Format: + name1=val1 name2=val2 name3=val3
        """
        self.write("+ ")
        for param in params:
            self.write_param_decl(param)
            self.write(" ")
        self.write("\n")

    def write_param_decl(self, param: ParamDecl) -> str:
        """Format a parameter declaration for ngspice.
        
        For parameter functions (with parentheses), uses .param func(args) = {expr} syntax.
        For normal parameters, uses name={expr} syntax (no spaces around =, without .param prefix, as write_param_decls adds it).
        """
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
                # For param functions, format expression and wrap in { }
                default = param.default
                if isinstance(default, str):
                    # Already formatted
                    default_str = default
                else:
                    default_str = self.format_expr(default)
                    # Remove outer braces if present (format_expr adds them, but we're already in { })
                    if default_str.startswith("{") and default_str.endswith("}"):
                        default_str = default_str[1:-1]

                # Use .param func_name(args) = {expression} syntax
                # Note: write_param_decls adds ".param " prefix, so we just write the function definition
                self.write(f"{param_name} = {{ {default_str} }}")
                return

        # Normal parameter declaration
        # Note: write_param_decls adds ".param " prefix, so we just write name=value (no spaces around =)
        if param.default is None:
            msg = f"Required (non-default) parameter {param.name} is not supported by {self.__class__.__name__}. "
            msg += f"Setting to maximum floating-point value {sys.float_info.max}, which almost certainly will not work if instantiated."
            self.log_warning(msg, f"Parameter: {param.name.name}")
            default = str(sys.float_info.max)
        else:
            # Special handling for 'type' parameter in model definitions
            # In ngspice, type should be a string literal (type=n or type=p), not an expression (type={n})
            if param_name.lower() == "type" and isinstance(param.default, Ref):
                # If type is a simple reference like {n} or {p}, write as string literal
                ref_name = param.default.ident.name.lower()
                if ref_name in ('n', 'p'):
                    default = ref_name
                else:
                    default = self.format_expr(param.default)
            else:
                default = self.format_expr(param.default)

        self.write(f"{param_name}={default}")
        
        # Write inline comment if present
        # In ngspice, inline comments use semicolon (;) on the same line
        if param.comment:
            self.write(f" ; {param.comment}")

    def write_model_def(self, model: ModelDef) -> None:
        """Write a model definition in ngspice format, handling BJT level mapping."""
        from ..data import Int, Float, MetricNum, ParamDecl, Ident
        
        # Helper to get param name safely
        def get_param_name(item):
            if isinstance(item, tuple):
                return item[0].name.name
            return item.name.name

        mname = self.format_ident(model.name)
        mtype = self.format_ident(model.mtype).lower()
        
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
                    if get_param_name(p).lower() == "level":
                        has_level = True
                        if isinstance(p, tuple):
                            p_obj = p[0]
                        else:
                            p_obj = p
                        if isinstance(p_obj.default, (Int, Float)):
                            current_level = int(float(p_obj.default.val) if hasattr(p_obj.default, 'val') else float(p_obj.default))
                        elif isinstance(p_obj.default, MetricNum):
                            current_level = int(float(p_obj.default.val))
                        break
                
                # Check if current level has a mapping
                if current_level in mapping_dict:
                    output_level = mapping_dict[current_level]
                    level_mapping_applied = True
                    
                    # Apply parameter mapping based on device type and level transition
                    # ngspice uses level 6 for MEXTRAM (same as Xyce level 504)
                    if mtype_lower in ("npn", "pnp") and current_level == 1 and output_level == 6:
                        # Import the mapping function from XyceNetlister
                        from .xyce import XyceNetlister
                        # Create a temporary XyceNetlister instance to use its mapping function
                        # We'll use a dummy IO object
                        import io
                        temp_xyce = XyceNetlister(self.src, io.StringIO(), model_level_mapping={"npn": [(1, 504)], "pnp": [(1, 504)]})
                        # Extract just the ParamDecl objects (not tuples)
                        param_decls = [p[0] if isinstance(p, tuple) else p for p in params_to_write]
                        # Map level 1 BJT parameters to MEXTRAM (level 6 uses same params as level 504)
                        mapped_params = temp_xyce._map_bjt_level1_to_mextram_params(param_decls, model_name=mname)
                        # Convert back to list format (mapped_params returns tuples)
                        params_to_write = mapped_params
                    
                    # Update or add level parameter
                    if has_level:
                        # Replace existing level parameter
                        for i, p in enumerate(params_to_write):
                            if get_param_name(p).lower() == "level":
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
        
        # Write model header
        # For BSIM4 models in ngspice, extract type from parameters and put it on .model line
        # Format: .model name nmos level=14 or .model name pmos level=14
        model_type_on_line = mtype.lower()
        type_param_removed = False
        
        if mtype.lower() == "bsim4":
            # Look for type parameter in params_to_write
            for i, param_item in enumerate(params_to_write):
                param = param_item[0] if isinstance(param_item, tuple) else param_item
                if param.name.name.lower() == "type" and isinstance(param.default, Ref):
                    ref_name = param.default.ident.name.lower()
                    if ref_name in ('n', 'p'):
                        # Use nmos/pmos on the .model line
                        model_type_on_line = f"{ref_name}mos"
                        # Remove type parameter from the list
                        params_to_write.pop(i)
                        type_param_removed = True
                        break
                elif param.name.name.lower() == "type" and isinstance(param.default, (Int, Float, MetricNum)):
                    # Handle numeric type (shouldn't happen, but just in case)
                    type_val = str(param.default.val if hasattr(param.default, 'val') else param.default)
                    if type_val.lower() in ('n', 'p'):
                        model_type_on_line = f"{type_val.lower()}mos"
                        params_to_write.pop(i)
                        type_param_removed = True
                        break
            
            # Check for level parameter to add to .model line (BSIM4 uses level=14)
            level_str = " level=14"  # Default for BSIM4
            level_param_index = None
            for i, param_item in enumerate(params_to_write):
                param = param_item[0] if isinstance(param_item, tuple) else param_item
                if param.name.name.lower() == "level":
                    level_param_index = i
                    if isinstance(param.default, (Int, Float)):
                        level_val = int(float(param.default.val) if hasattr(param.default, 'val') else float(param.default))
                        level_str = f" level={level_val}"
                    elif isinstance(param.default, MetricNum):
                        level_val = int(float(param.default.val))
                        level_str = f" level={level_val}"
                    break
            
            # Remove level parameter from params_to_write since it's on the .model line
            if level_param_index is not None:
                params_to_write.pop(level_param_index)
            
            self.writeln(f".model {mname} {model_type_on_line}{level_str}")
        else:
            self.writeln(f".model {mname} {mtype}")
        
        self.write("+ ")
        
        for arg in model.args:
            self.write(self.format_expr(arg) + " ")
        
        self.write("\n")
        
        # Write parameters
        for param_item in params_to_write:
            self.write("+ ")
            if isinstance(param_item, tuple):
                # Handle (ParamDecl, comment, commented_out) tuple from mapping
                param, comment, commented_out = param_item
                if commented_out:
                    # Write as comment
                    self.write_comment(f"{get_param_name(param_item)}={self.format_expr(param.default)} - {comment}")
                else:
                    # Write parameter with optional comment
                    self.write_param_decl(param)
                    if comment:
                        self.write(f" * {comment}")
                    self.write("\n")
            else:
                # Normal parameter
                self.write_param_decl(param_item)
                self.write("\n")
        
        self.write("\n")  # Ending blank-line

    def netlist(self) -> None:
        """Override netlist() to apply ngspice-specific statistics variations before writing.

        This applies statistics variations (process and mismatch) to the Program,
        creating ngspice-specific artifacts like .param function definitions for mismatch parameters.
        This only happens when writing to ngspice format.
        """
        # Check if we are generating a test netlist
        if self.file_type == "test":
            return self.write_test_netlist()

        # Apply statistics variations with NGSPICE format before writing
        # This modifies the AST directly, replacing StatisticsBlocks with generated content
        apply_statistics_variations(self.src, output_format=NetlistDialects.NGSPICE)

        # For models files, add global tref and temp parameters if not already defined
        # (tref and temp are commonly used in resistor expressions)
        # Note: ngspice uses 'temper' as built-in, but expressions may use 'temp', so we define temp=temper
        if self.file_type == "models":
            from ..data import ParamDecls, ParamDecl, Float, Ident, Ref
            # Check if tref and temp are already defined as global parameters
            tref_defined = False
            temp_defined = False
            for file in self.src.files:
                for entry in file.contents:
                    if isinstance(entry, ParamDecls):
                        for param in entry.params:
                            if param.name.name.lower() == "tref":
                                tref_defined = True
                            if param.name.name.lower() == "temp":
                                temp_defined = True
                    elif isinstance(entry, ParamDecl):
                        if entry.name.name.lower() == "tref":
                            tref_defined = True
                        if entry.name.name.lower() == "temp":
                            temp_defined = True
                    if tref_defined and temp_defined:
                        break
                if tref_defined and temp_defined:
                    break
            
            # Add missing parameters at the beginning of the first file
            if self.src.files:
                first_file = self.src.files[0]
                params_to_add = []
                if not tref_defined:
                    tref_param = ParamDecl(name=Ident("tref"), default=Float(30.0), distr=None)
                    params_to_add.append(tref_param)
                if not temp_defined:
                    # Define temp as alias to ngspice's built-in 'temper' variable
                    # Note: In ngspice, 'temper' is the built-in temperature variable
                    temp_param = ParamDecl(name=Ident("temp"), default=Ref(ident=Ident("temper")), distr=None)
                    params_to_add.append(temp_param)
                
                if params_to_add:
                    # Insert at the beginning
                    first_file.contents.insert(0, ParamDecls(params=params_to_add))

        # Call parent implementation to do the actual writing
        super().netlist()

    def _replace_temp_with_temper(self, expr: Expr) -> Expr:
        """Replace 'temp' parameter references with 'temper' (ngspice's built-in temperature variable) in expressions.
        
        This recursively walks the expression tree and replaces Ref objects with name 'temp'
        with Ref objects with name 'temper'.
        """
        from ..data import Ref, Ident, BinaryOp, UnaryOp, TernOp, Call
        
        # Base case: if it's a Ref to 'temp', replace with 'temper'
        if isinstance(expr, Ref):
            if expr.ident.name.lower() == "temp":
                # Create new Ref to 'temper'
                return Ref(ident=Ident("temper"), resolved=expr.resolved)
            return expr
        
        # For literals, return as-is
        if isinstance(expr, (Int, Float, MetricNum, QuotedString)):
            return expr
        
        # For compound expressions, recurse
        if isinstance(expr, BinaryOp):
            return BinaryOp(
                tp=expr.tp,
                left=self._replace_temp_with_temper(expr.left),
                right=self._replace_temp_with_temper(expr.right)
            )
        
        if isinstance(expr, UnaryOp):
            return UnaryOp(
                tp=expr.tp,
                targ=self._replace_temp_with_temper(expr.targ)
            )
        
        if isinstance(expr, TernOp):
            return TernOp(
                cond=self._replace_temp_with_temper(expr.cond),
                if_true=self._replace_temp_with_temper(expr.if_true),
                if_false=self._replace_temp_with_temper(expr.if_false)
            )
        
        if isinstance(expr, Call):
            return Call(
                func=expr.func,
                args=[self._replace_temp_with_temper(arg) for arg in expr.args]
            )
        
        # For anything else, return as-is
        return expr

    def write_test_netlist(self) -> None:
        """Write a sanity check netlist."""
        # Header
        self.write("* Sanity Check Netlist generated by NgspiceNetlister\n\n")
        
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


class CdlNetlister(SpiceNetlister):
    """FIXME: CDL-Format Netlister"""

    def __init__(self, *_, **__):
        raise NotImplementedError

    @property
    def enum(self):
        """Get our entry in the `NetlistDialects` enumeration"""
        return NetlistDialects.CDL


# Utility functions for expression analysis

def find_matching_param(program: Program, param_name: str) -> Optional[ParamDecl]:
    """Find a parameter by name in the program.
    
    Searches in:
    1. Global ParamDecls (top-level)
    2. Library sections (LibSectionDef.entries containing ParamDecls)
    
    Returns the first matching parameter found.
    """
    # First search global ParamDecls
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                for param in entry.params:
                    if param.name.name == param_name:
                        return param
    
    # Then search library sections
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, LibSectionDef):
                for sub_entry in entry.entries:
                    if isinstance(sub_entry, ParamDecls):
                        for param in sub_entry.params:
                            if param.name.name == param_name:
                                return param
    
    return None


def expr_references_param(expr: Expr, param_name: str) -> bool:
    """Check if an expression references a parameter by name.
    
    Recursively searches through the expression tree for Ref nodes
    that match the parameter name. Case-insensitive matching.
    
    Args:
        expr: The expression to check
        param_name: The parameter name to search for
        
    Returns:
        True if the expression references the parameter, False otherwise
    """
    # Handle Ref nodes - check if name matches (case-insensitive)
    if isinstance(expr, Ref):
        return expr.ident.name.lower() == param_name.lower()
    
    # Handle Call nodes - recurse into arguments
    if isinstance(expr, Call):
        return any(expr_references_param(arg, param_name) for arg in expr.args)
    
    # Handle BinaryOp - recurse into left and right
    if isinstance(expr, BinaryOp):
        return (expr_references_param(expr.left, param_name) or 
                expr_references_param(expr.right, param_name))
    
    # Handle UnaryOp - recurse into target
    if isinstance(expr, UnaryOp):
        return expr_references_param(expr.targ, param_name)
    
    # Handle TernOp - recurse into all three parts
    if isinstance(expr, TernOp):
        return (expr_references_param(expr.cond, param_name) or
                expr_references_param(expr.if_true, param_name) or
                expr_references_param(expr.if_false, param_name))
    
    # Literals and other types - no parameter references
    return False


def count_param_refs_in_entry(entry: Entry, param_name: str) -> int:
    """Count how many times a parameter is referenced in an entry.
    
    Args:
        entry: The entry to check
        param_name: The parameter name to count
        
    Returns:
        Number of references found
    """
    count = 0
    
    if isinstance(entry, ParamDecls):
        for param in entry.params:
            if param.default is not None and expr_references_param(param.default, param_name):
                count += 1
    
    elif isinstance(entry, SubcktDef):
        for param in entry.params:
            if param.default is not None and expr_references_param(param.default, param_name):
                count += 1
        for sub_entry in entry.entries:
            count += count_param_refs_in_entry(sub_entry, param_name)
    
    elif isinstance(entry, ModelDef):
        for param in entry.params:
            if param.default is not None and expr_references_param(param.default, param_name):
                count += 1
    
    elif isinstance(entry, ModelVariant):
        for param in entry.params:
            if param.default is not None and expr_references_param(param.default, param_name):
                count += 1
    
    elif isinstance(entry, ModelFamily):
        for variant in entry.variants:
            for param in variant.params:
                if param.default is not None and expr_references_param(param.default, param_name):
                    count += 1
    
    elif isinstance(entry, FunctionDef):
        for stmt in entry.stmts:
            if isinstance(stmt, Return):
                if expr_references_param(stmt.val, param_name):
                    count += 1
    
    elif isinstance(entry, Instance):
        for param_val in entry.params:
            if expr_references_param(param_val.val, param_name):
                count += 1
    
    elif isinstance(entry, Primitive):
        for param_val in entry.kwargs:
            if expr_references_param(param_val.val, param_name):
                count += 1
    
    elif isinstance(entry, Options):
        for option in entry.vals:
            if not isinstance(option.val, QuotedString):
                if expr_references_param(option.val, param_name):
                    count += 1
    
    elif isinstance(entry, LibSectionDef):
        for sub_entry in entry.entries:
            count += count_param_refs_in_entry(sub_entry, param_name)
    
    return count


def debug_find_all_param_refs(program: Program, param_name: str) -> List[Tuple[str, Expr]]:
    """Find all parameter references in the program and return their locations.
    
    Args:
        program: The program to search
        param_name: The parameter name to find
        
    Returns:
        List of (location_description, expr) tuples where param_name is referenced.
        Location format: "file:entry_type:entry_name:field_path"
    """
    locations = []
    
    def find_in_entry(entry: Entry, file_path: str, entry_path: str) -> None:
        """Recursively find parameter references in an entry."""
        entry_type = type(entry).__name__
        
        if isinstance(entry, ParamDecls):
            for i, param in enumerate(entry.params):
                if param.default is not None and expr_references_param(param.default, param_name):
                    locations.append((f"{file_path}:ParamDecls:param[{i}].default", param.default))
        
        elif isinstance(entry, SubcktDef):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for i, param in enumerate(entry.params):
                if param.default is not None and expr_references_param(param.default, param_name):
                    locations.append((f"{file_path}:SubcktDef:{entry_name}:param[{i}].default", param.default))
            for j, sub_entry in enumerate(entry.entries):
                find_in_entry(sub_entry, file_path, f"{entry_path}.entries[{j}]")
        
        elif isinstance(entry, ModelDef):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for i, param in enumerate(entry.params):
                if param.default is not None and expr_references_param(param.default, param_name):
                    locations.append((f"{file_path}:ModelDef:{entry_name}:param[{i}].default", param.default))
        
        elif isinstance(entry, ModelVariant):
            variant_name = entry.variant.name if hasattr(entry.variant, 'name') else str(entry.variant)
            for i, param in enumerate(entry.params):
                if param.default is not None and expr_references_param(param.default, param_name):
                    locations.append((f"{file_path}:ModelVariant:{variant_name}:param[{i}].default", param.default))
        
        elif isinstance(entry, ModelFamily):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for v, variant in enumerate(entry.variants):
                for i, param in enumerate(variant.params):
                    if param.default is not None and expr_references_param(param.default, param_name):
                        locations.append((f"{file_path}:ModelFamily:{entry_name}:variant[{v}].param[{i}].default", param.default))
        
        elif isinstance(entry, FunctionDef):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for s, stmt in enumerate(entry.stmts):
                if isinstance(stmt, Return):
                    if expr_references_param(stmt.val, param_name):
                        locations.append((f"{file_path}:FunctionDef:{entry_name}:stmt[{s}].val", stmt.val))
        
        elif isinstance(entry, Instance):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for i, param_val in enumerate(entry.params):
                if expr_references_param(param_val.val, param_name):
                    locations.append((f"{file_path}:Instance:{entry_name}:params[{i}].val", param_val.val))
        
        elif isinstance(entry, Primitive):
            entry_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for i, param_val in enumerate(entry.kwargs):
                if expr_references_param(param_val.val, param_name):
                    locations.append((f"{file_path}:Primitive:{entry_name}:kwargs[{i}].val", param_val.val))
        
        elif isinstance(entry, Options):
            for i, option in enumerate(entry.vals):
                if not isinstance(option.val, QuotedString):
                    if expr_references_param(option.val, param_name):
                        locations.append((f"{file_path}:Options:vals[{i}].val", option.val))
        
        elif isinstance(entry, LibSectionDef):
            section_name = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            for j, sub_entry in enumerate(entry.entries):
                find_in_entry(sub_entry, file_path, f"{entry_path}.entries[{j}]")
    
    for file in program.files:
        file_path = str(file.path) if hasattr(file, 'path') else "unknown"
        for i, entry in enumerate(file.contents):
            find_in_entry(entry, file_path, f"contents[{i}]")
    
    return locations


def replace_param_ref_in_expr(expr: Expr, param_name: str, replacement: Expr) -> Expr:
    """Recursively replace Ref nodes matching param_name with replacement in an expression.
    
    Args:
        expr: The expression to search and replace in
        param_name: The parameter name to replace
        replacement: The expression to replace with (can be Call, Ref, or other Expr)
    """
    # Handle Ref nodes - this is the key replacement
    if isinstance(expr, Ref):
        if expr.ident.name == param_name:
            return replacement
        return expr
    
    # Handle Call nodes - recurse into arguments
    if isinstance(expr, Call):
        new_args = [replace_param_ref_in_expr(arg, param_name, replacement) for arg in expr.args]
        if new_args != expr.args:
            return Call(func=expr.func, args=new_args)
        return expr
    
    # Handle BinaryOp - recurse into left and right
    if isinstance(expr, BinaryOp):
        new_left = replace_param_ref_in_expr(expr.left, param_name, replacement)
        new_right = replace_param_ref_in_expr(expr.right, param_name, replacement)
        if new_left != expr.left or new_right != expr.right:
            return BinaryOp(tp=expr.tp, left=new_left, right=new_right)
        return expr
    
    # Handle UnaryOp - recurse into target
    if isinstance(expr, UnaryOp):
        new_targ = replace_param_ref_in_expr(expr.targ, param_name, replacement)
        if new_targ != expr.targ:
            return UnaryOp(tp=expr.tp, targ=new_targ)
        return expr
    
    # Handle TernOp - recurse into all three parts
    if isinstance(expr, TernOp):
        new_cond = replace_param_ref_in_expr(expr.cond, param_name, replacement)
        new_if_true = replace_param_ref_in_expr(expr.if_true, param_name, replacement)
        new_if_false = replace_param_ref_in_expr(expr.if_false, param_name, replacement)
        if (new_cond != expr.cond or new_if_true != expr.if_true or 
            new_if_false != expr.if_false):
            return TernOp(cond=new_cond, if_true=new_if_true, if_false=new_if_false)
        return expr
    
    # Literals and other types - no replacement needed
    return expr


def replace_param_refs_in_entry(entry: Entry, param_name: str, replacement: Expr) -> None:
    """Replace parameter references in a single Entry."""
    # ParamDecls - replace in each param's default
    if isinstance(entry, ParamDecls):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, replacement)
    
    # SubcktDef - replace in params and recurse into entries
    elif isinstance(entry, SubcktDef):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, replacement)
        for sub_entry in entry.entries:
            replace_param_refs_in_entry(sub_entry, param_name, replacement)
    
    # ModelDef - replace in params
    elif isinstance(entry, ModelDef):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, replacement)
            # Note: Parameters without defaults (None) are positional args and don't need replacement
    
    # ModelVariant - replace in params (explicit handling for completeness)
    elif isinstance(entry, ModelVariant):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, replacement)
    
    # ModelFamily - recurse into variants
    elif isinstance(entry, ModelFamily):
        for variant in entry.variants:
            for param in variant.params:
                if param.default is not None:
                    param.default = replace_param_ref_in_expr(param.default, param_name, replacement)
    
    # FunctionDef - replace in return statement
    elif isinstance(entry, FunctionDef):
        for stmt in entry.stmts:
            if isinstance(stmt, Return):
                stmt.val = replace_param_ref_in_expr(stmt.val, param_name, replacement)
    
    # Instance - replace in param values
    elif isinstance(entry, Instance):
        for param_val in entry.params:
            param_val.val = replace_param_ref_in_expr(param_val.val, param_name, replacement)
    
    # Primitive - replace in kwargs
    elif isinstance(entry, Primitive):
        for param_val in entry.kwargs:
            param_val.val = replace_param_ref_in_expr(param_val.val, param_name, replacement)
    
    # Options - replace in option values (if Expr, not QuotedString)
    elif isinstance(entry, Options):
        for option in entry.vals:
            # OptionVal = Union[QuotedString, Expr], so check if it's not QuotedString
            if not isinstance(option.val, QuotedString):
                option.val = replace_param_ref_in_expr(option.val, param_name, replacement)
    
    # LibSectionDef - recurse into entries
    elif isinstance(entry, LibSectionDef):
        for sub_entry in entry.entries:
            replace_param_refs_in_entry(sub_entry, param_name, replacement)
    
    # Catch-all for unhandled entry types
    else:
        # Entry types that don't contain parameter references: Include, AhdlInclude, UseLibSection,
        # DialectChange, Unknown, End, StatisticsBlock, Comment, BlankLine (already processed or no params)
        # Log a warning for any unexpected types
        entry_type_name = type(entry).__name__
        if entry_type_name not in ('Include', 'AhdlInclude', 'UseLibSection', 'DialectChange', 'Unknown', 'End', 'StatisticsBlock', 'Comment', 'BlankLine'):
            warn(f"replace_param_refs_in_entry: Unhandled entry type {entry_type_name} for param {param_name}")


def replace_param_refs_in_program(program: Program, param_name: str, replacement: Expr, debug: bool = False) -> None:
    """Replace all references to param_name throughout the entire program with replacement.
    
    Args:
        program: The program to process
        param_name: The parameter name to replace
        replacement: The expression to replace with
        debug: If True, log entry types processed and replacement counts
    """
    if debug:
        entry_type_counts = {}
        replacement_counts = {}
    
    for file in program.files:
        for entry in file.contents:
            entry_type = type(entry).__name__
            if debug:
                entry_type_counts[entry_type] = entry_type_counts.get(entry_type, 0) + 1
            
            # Count references before replacement
            if debug:
                before_count = count_param_refs_in_entry(entry, param_name)
                if before_count > 0:
                    replacement_counts[entry_type] = replacement_counts.get(entry_type, 0) + before_count
            
            replace_param_refs_in_entry(entry, param_name, replacement)
    
    if debug:
        warn(f"replace_param_refs_in_program({param_name}): Processed {sum(entry_type_counts.values())} entries")
        for entry_type, count in sorted(entry_type_counts.items()):
            replacements = replacement_counts.get(entry_type, 0)
            if replacements > 0:
                warn(f"  {entry_type}: {count} entries, {replacements} replacements")


def remove_param_declaration(program: Program, param_name: str) -> None:
    """Remove the parameter declaration from global ParamDecls and library sections."""
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                entry.params = [p for p in entry.params if p.name.name != param_name]
            elif isinstance(entry, LibSectionDef):
                # Also remove from library sections
                for sub_entry in entry.entries:
                    if isinstance(sub_entry, ParamDecls):
                        sub_entry.params = [p for p in sub_entry.params if p.name.name != param_name]


def find_alias_parameters(program: Program, target_param_name: str) -> List[ParamDecl]:
    """Find all parameters that are defined as simple references to target_param_name.
    
    An alias parameter is one whose default value is just a Ref to the target parameter.
    For example, if sw_tox_lv = sw_tox_lv_corner, then sw_tox_lv is an alias for sw_tox_lv_corner.
    
    Returns:
        List of ParamDecl objects that are aliases for the target parameter.
    """
    aliases = []
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                for param in entry.params:
                    # Check if this parameter is a simple reference to the target
                    if param.default is not None:
                        if isinstance(param.default, Ref):
                            if param.default.ident.name == target_param_name:
                                aliases.append(param)
            elif isinstance(entry, LibSectionDef):
                for sub_entry in entry.entries:
                    if isinstance(sub_entry, ParamDecls):
                        for param in sub_entry.params:
                            if param.default is not None:
                                if isinstance(param.default, Ref):
                                    if param.default.ident.name == target_param_name:
                                        aliases.append(param)
    return aliases


def collect_statistics_blocks(program: Program) -> List[StatisticsBlock]:
    """Collect all statistics blocks from the program."""
    stats_blocks = []
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, StatisticsBlock):
                stats_blocks.append(entry)
    return stats_blocks


def has_lognorm_distributions(stats_blocks: List[StatisticsBlock]) -> bool:
    """Check if any variations use lognormal distributions."""
    return any(
        var.dist and 'lnorm' in var.dist.lower()
        for stats in stats_blocks
        for section in (stats.process, stats.mismatch)
        if section
        for var in section
    )


def add_lnorm_functions(program: Program) -> None:
    """Add lnorm and alnorm .param declarations to the program.

    Creates: .param lnorm(mu,sigma)=exp(gauss(mu,sigma))
    These are added to the beginning of the file containing statistics blocks.
    """
    if not program.files:
        return

    # Find the file that contains statistics blocks (models.scs) to add params there
    stats_file = None
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, StatisticsBlock):
                stats_file = file
                break
        if stats_file:
            break

    # Fallback to first file if no stats file found
    if not stats_file:
        stats_file = program.files[0]

    # Create .param declarations for lnorm and alnorm
    # Create expression: exp(gauss(mu,sigma))
    gauss_call = Call(
        func=Ref(ident=Ident("gauss")),
        args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma"))]
    )
    exp_call = Call(
        func=Ref(ident=Ident("exp")),
        args=[gauss_call]
    )

    math_params = [
        ParamDecl(
            name=Ident("lnorm(mu,sigma)"),
            default=exp_call,
            distr=None
        ),
        ParamDecl(
            name=Ident("alnorm(mu,sigma)"),
            default=exp_call,
            distr=None
        )
    ]

    # Insert params at the beginning of the file
    stats_file.contents.insert(0, ParamDecls(params=math_params))


def add_lnorm_functions_ngspice(program: Program) -> None:
    """Add lnorm and alnorm .param function declarations for ngspice.

    Creates: .param lnorm(mu,sigma) = {exp(gauss(mu,sigma))}
    These are added to the beginning of the file containing statistics blocks.
    """
    if not program.files:
        return

    # Find the file that contains statistics blocks (models.scs) to add params there
    stats_file = None
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, StatisticsBlock):
                stats_file = file
                break
        if stats_file:
            break

    # Fallback to first file if no stats file found
    if not stats_file:
        stats_file = program.files[0]

    # Create .param function declarations for lnorm and alnorm
    # Create expression: exp(gauss(mu,sigma))
    gauss_call = Call(
        func=Ref(ident=Ident("gauss")),
        args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma"))]
    )
    exp_call = Call(
        func=Ref(ident=Ident("exp")),
        args=[gauss_call]
    )

    math_params = [
        ParamDecl(
            name=Ident("lnorm(mu,sigma)"),
            default=exp_call,
            distr=None
        ),
        ParamDecl(
            name=Ident("alnorm(mu,sigma)"),
            default=exp_call,
            distr=None
        )
    ]

    # Insert params at the beginning of the file
    stats_file.contents.insert(0, ParamDecls(params=math_params))


def create_monte_carlo_distribution_call(dist_type: str, std: Float) -> Optional[Call]:
    """Create a distribution function call (gauss or lnorm) for Monte Carlo variation.

    Returns a Call expression for the distribution function, or None if the distribution
    type is not supported. Warns if an unsupported distribution type is encountered.
    """
    if not dist_type:
        return None

    dist_type_lower = dist_type.lower()
    if 'gauss' in dist_type_lower:
        return Call(func=Ref(ident=Ident("gauss")), args=[Int(0), Float(1)])
    elif 'lnorm' in dist_type_lower:
        return Call(func=Ref(ident=Ident("lnorm")), args=[Int(0), Float(1)])

    # Warn about unsupported distribution type
    warn(f"Unsupported distribution type '{dist_type}' for Monte Carlo variation. Supported types: gauss, lnorm")
    return None


def apply_monte_carlo_variation(original_expr: Expr, var: Variation) -> Expr:
    """Apply Monte Carlo variation to an expression: original * (1 + std * dist(...))."""
    if not var.dist:
        return original_expr

    dist_call = create_monte_carlo_distribution_call(var.dist, var.std)
    if not dist_call:
        return original_expr

    variation_term = BinaryOp(tp=BinaryOperator.MUL, left=var.std, right=dist_call)
    one_plus_variation = BinaryOp(tp=BinaryOperator.ADD, left=Float(1), right=variation_term)
    return BinaryOp(tp=BinaryOperator.MUL, left=original_expr, right=one_plus_variation)


def create_relative_process_variation_expr(param_name: str, var: Variation) -> Expr:
    """Create a relative process variation expression that references the parameter itself.
    
    Returns: {param_name * (1 + std * dist(...)) + mean}
    This allows the variation to work regardless of which library section's value is used.
    """
    # Reference to the parameter itself
    param_ref = Ref(ident=Ident(param_name))
    
    # Apply Monte Carlo variation: param_name * (1 + std * dist(...))
    if var.dist:
        dist_call = create_monte_carlo_distribution_call(var.dist, var.std)
        if dist_call:
            variation_term = BinaryOp(tp=BinaryOperator.MUL, left=var.std, right=dist_call)
            one_plus_variation = BinaryOp(tp=BinaryOperator.ADD, left=Float(1), right=variation_term)
            varied_expr = BinaryOp(tp=BinaryOperator.MUL, left=param_ref, right=one_plus_variation)
        else:
            varied_expr = param_ref
    else:
        varied_expr = param_ref
    
    # Add mean value if present (for corner analysis)
    if var.mean:
        varied_expr = BinaryOp(tp=BinaryOperator.ADD, left=varied_expr, right=var.mean)
    
    return varied_expr


def verify_process_variation_params(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Verify that all parameters in process variations exist (in library sections or global).
    
    Raises RuntimeError if any parameters are missing.
    """
    missing_params = []
    for stats in stats_blocks:
        if stats.process:
            for var in stats.process:
                param_name = var.name.name
                matching_param = find_matching_param(program, param_name)
                if not matching_param:
                    missing_params.append(param_name)
    
    if missing_params:
        raise RuntimeError(
            f"Process variation parameters not found: {', '.join(missing_params)}. "
            f"These parameters must exist in at least one library section or as global parameters."
        )


def calculate_process_variation_expr(param_name: str, var: Variation) -> Expr:
    """Calculate the relative process variation expression for a parameter.
    
    Returns an expression that references the parameter itself: {param_name * (1 + std * dist(...)) + mean}
    This expression will be applied after library sections as a relative assignment.
    """
    return create_relative_process_variation_expr(param_name, var)


def is_param_in_process_variations(stats_blocks: List[StatisticsBlock], param_name: str) -> bool:
    """Check if a parameter name appears in any process variations."""
    for stats in stats_blocks:
        if stats.process:
            for var in stats.process:
                if var.name.name == param_name:
                    return True
    return False


def is_param_in_library_section(program: Program, param_name: str) -> bool:
    """Check if a parameter exists in a library section (not just global ParamDecls)."""
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, LibSectionDef):
                for sub_entry in entry.entries:
                    if isinstance(sub_entry, ParamDecls):
                        for param in sub_entry.params:
                            if param.name.name == param_name:
                                return True
    return False


def apply_all_process_variations_legacy(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Legacy behavior: apply variations directly to global parameters (for non-Xyce formats)."""
    for stats in stats_blocks:
        if stats.process:
            for var in stats.process:
                # Only apply to global parameters, not library section parameters
                if not is_param_in_library_section(program, var.name.name):
                    matching_param = find_matching_param(program, var.name.name)
                    if matching_param:
                        original_expr = matching_param.default or Int(0)
                        new_expr = apply_monte_carlo_variation(original_expr, var)
                        if var.mean:
                            new_expr = BinaryOp(tp=BinaryOperator.ADD, left=new_expr, right=var.mean)
                        matching_param.default = new_expr


def replace_statistics_blocks_with_generated_content(program: Program, stats_blocks: List[StatisticsBlock], output_format: Optional["NetlistDialects"] = None) -> None:
    """Replace StatisticsBlocks in the AST with generated functions and process variations for library parameters."""
    # Verify all parameters exist (error if any missing)
    verify_process_variation_params(program, stats_blocks)

    # Process each statistics block
    for stats in stats_blocks:
        # Collect generated content to replace the StatisticsBlock
        generated_content = []

        # Add process variations for ALL parameters (both global and library section)
        if stats.process:
            # Create process variation expressions
            process_variations = []
            for var in stats.process:
                param_name = var.name.name
                # Check if parameter is in library section or global
                if is_param_in_library_section(program, param_name):
                    # Library section parameter: use relative expression
                    varied_expr = create_relative_process_variation_expr(param_name, var)
                else:
                    # Global parameter: get original value and apply variation
                    matching_param = find_matching_param(program, param_name)
                    if matching_param:
                        original_expr = matching_param.default or Int(0)
                        varied_expr = apply_monte_carlo_variation(original_expr, var)
                        if var.mean:
                            varied_expr = BinaryOp(tp=BinaryOperator.ADD, left=varied_expr, right=var.mean)
                    else:
                        continue  # Skip if parameter not found
                process_variations.append((param_name, varied_expr))

            if process_variations:
                # Create ParamDecls blocks with blank lines between logical groups
                # Group parameters by category to create natural breaks
                from collections import defaultdict
                grouped_params = defaultdict(list)
                for param_name, varied_expr in process_variations:
                    process_param_name = f"{param_name}__process__"
                    param_decl = ParamDecl(name=Ident(process_param_name), default=varied_expr, distr=None)
                    # Group by parameter category (e.g., sw_tox, sw_vth0, sw_rpoly, sw_rm, sw_m, sw_rc, sw_cap, etc.)
                    # Use first two underscore-separated parts as the category
                    parts = param_name.split('_')
                    if len(parts) >= 2:
                        category = '_'.join(parts[:2])  # e.g., 'sw_tox', 'sw_vth0', 'sw_rpoly'
                    else:
                        category = parts[0]
                    grouped_params[category].append(param_decl)
                
                # Create ParamDecls blocks with BlankLine entries between groups
                # Sort groups by category name for consistent ordering
                sorted_groups = sorted(grouped_params.items())
                first_group = True
                for category, params in sorted_groups:
                    if not first_group:
                        # Add blank line between groups
                        from ..data import BlankLine
                        blank_line = BlankLine()
                        generated_content.append(blank_line)
                    generated_content.append(ParamDecls(params=params))
                    first_group = False
                
                # Replace all references to original parameters with __process__ version
                # Note: Process variations are applied BEFORE mismatch variations (see line 1496),
                # ensuring we always get __process____mismatch__ and never __mismatch____process__
                for param_name, _ in process_variations:
                    # Find alias parameters BEFORE doing replacement (they reference the original param name)
                    # For example, if sw_tox_lv = sw_tox_lv_corner, then sw_tox_lv is an alias
                    # When sw_tox_lv_corner is replaced with sw_tox_lv_corner__process__,
                    # we should also replace all uses of sw_tox_lv with sw_tox_lv_corner__process__
                    alias_params = find_alias_parameters(program, param_name)
                    alias_names = [alias.name.name for alias in alias_params]
                    
                    # Replace all references to the original parameter with the __process__ version
                    # This applies to both global and library section parameters
                    process_param_ref = Ref(ident=Ident(f"{param_name}__process__"))
                    replace_param_refs_in_program(program, param_name, process_param_ref)
                    
                    # Verify replacement was successful
                    remaining_refs = debug_find_all_param_refs(program, param_name)
                    if remaining_refs:
                        warn(f"WARNING: {len(remaining_refs)} references to {param_name} still exist after replacement")
                        # Log first few locations for debugging
                        for loc, expr in remaining_refs[:5]:
                            warn(f"  Unreplaced reference at: {loc}")
                        if len(remaining_refs) > 5:
                            warn(f"  ... and {len(remaining_refs) - 5} more locations")
                    
                    # Now handle alias parameters: replace all uses of aliases with the __process__ version
                    for alias_name in alias_names:
                        # Replace all uses of the alias with the __process__ version
                        replace_param_refs_in_program(program, alias_name, process_param_ref)
                        
                        # Verify alias replacement was successful
                        remaining_alias_refs = debug_find_all_param_refs(program, alias_name)
                        if remaining_alias_refs:
                            warn(f"WARNING: {len(remaining_alias_refs)} references to {alias_name} still exist after replacement")
                            for loc, expr in remaining_alias_refs[:5]:
                                warn(f"  Unreplaced alias reference at: {loc}")
                        
                        # Remove the alias parameter declaration
                        remove_param_declaration(program, alias_name)
                    
                    # Remove the original parameter declaration only for global parameters
                    # Library section parameters are kept (they're needed for the relative expressions)
                    if not is_param_in_library_section(program, param_name):
                        remove_param_declaration(program, param_name)

        # Replace the StatisticsBlock with generated content
        # First, collect any comments that appear before and after the StatisticsBlock
        # so they can be written in the correct positions
        for file in program.files:
            for i, entry in enumerate(file.contents):
                if isinstance(entry, StatisticsBlock) and entry is stats:
                    # Collect comments that appear before this StatisticsBlock
                    # (they should be written before the generated content)
                    before_comments = []
                    j = i - 1
                    while j >= 0 and isinstance(file.contents[j], Comment):
                        before_comments.insert(0, file.contents[j])
                        j -= 1
                    
                    # Collect comments that appear after this StatisticsBlock
                    # (they should be written after the generated content)
                    after_comments = []
                    k = i + 1
                    while k < len(file.contents) and isinstance(file.contents[k], Comment):
                        after_comments.append(file.contents[k])
                        k += 1
                    
                    # Remove the StatisticsBlock and comments from their original positions
                    # and replace with before_comments + generated_content + after_comments
                    start_idx = j + 1  # Start of comments (or StatisticsBlock if no comments)
                    end_idx = k  # End after StatisticsBlock and after-comments
                    file.contents[start_idx:end_idx] = before_comments + generated_content + after_comments
                    break

    # Now apply mismatch variations (creates functions and replaces references)
    # IMPORTANT: Process variations MUST be applied before mismatch variations.
    # This ensures we always get __process____mismatch__ suffixes and never __mismatch____process__.
    apply_all_mismatch_variations(program, stats_blocks, output_format=output_format)


def move_header_comments_to_top(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Move header comments (those that were before the first StatisticsBlock) to the top of the file.
    
    These comments should appear before all generated content (lnorm functions, mismatch params, etc.)
    """
    if not stats_blocks:
        return
    
    # Find the first StatisticsBlock to identify header comments
    first_stats = stats_blocks[0]
    
    for file in program.files:
        # Find where the first StatisticsBlock's comments ended up
        # They should be right before the process variation params
        header_comments = []
        header_start_idx = None
        header_end_idx = None
        
        # Look for a sequence of comments that match the header pattern
        # (comments before process variation params, which start with sw_*__process__)
        for i, entry in enumerate(file.contents):
            if isinstance(entry, Comment):
                # Check if this is part of a header comment block
                # Header comments typically appear before process variation params
                # Look ahead to see if we're before process params
                is_header = False
                for j in range(i + 1, min(i + 10, len(file.contents))):
                    next_entry = file.contents[j]
                    if isinstance(next_entry, ParamDecls):
                        # Check if this contains process variation params
                        if next_entry.params and any('__process__' in param.name.name for param in next_entry.params):
                            is_header = True
                            break
                    elif not isinstance(next_entry, Comment):
                        # Non-comment, non-param entry - not a header
                        break
                
                if is_header:
                    if header_start_idx is None:
                        header_start_idx = i
                    header_comments.append(entry)
                    header_end_idx = i + 1
                elif header_start_idx is not None:
                    # We've passed the header comments
                    break
        
        # If we found header comments, move them to the top
        if header_comments and header_start_idx is not None:
            # Remove comments from their current position
            del file.contents[header_start_idx:header_end_idx]
            
            # Find the insertion point (after any existing top comments, before generated params)
            insert_idx = 0
            for i, entry in enumerate(file.contents):
                if isinstance(entry, Comment):
                    # Keep existing top comments
                    insert_idx = i + 1
                elif isinstance(entry, ParamDecls):
                    # Insert before first ParamDecls (generated content)
                    insert_idx = i
                    break
                else:
                    # Insert before first non-comment, non-param entry
                    insert_idx = i
                    break
            
            # Insert header comments at the top
            file.contents[insert_idx:insert_idx] = header_comments
            break


def create_mismatch_function(var: Variation, idx: int, original_expr: Optional[Expr] = None, output_format: Optional["NetlistDialects"] = None) -> Optional[ParamDecl]:
    """Create a mismatch .param declaration with dummy parameter syntax as requested.

    For Xyce: Creates .FUNC mm_z1__mismatch__(dummy_param) {original_expr+enable_mismatch*gauss(0,mismatch_factor)}
    For ngspice: Creates .param mm_z1__mismatch__(dummy_param) = {original_expr+enable_mismatch*gauss(0,mismatch_factor)}
    This is placed near the top and called as mm_z1__mismatch__(0).
    
    Args:
        var: The variation definition
        idx: Index for uniqueness
        original_expr: The original parameter's default expression (if None, uses Int(0))
        output_format: The target output format (Xyce or ngspice)
    """
    if not var.dist:
        return None

    from .. import NetlistDialects
    is_ngspice = output_format == NetlistDialects.NGSPICE

    param_name = f"{var.name.name}__mismatch__(dummy_param)"
    dist_type_lower = var.dist.lower()

    # Use original expression if provided, otherwise use 0
    base_expr = original_expr if original_expr is not None else Int(0)
    enable_mismatch_ref = Ref(ident=Ident("enable_mismatch"))
    mismatch_factor_ref = Ref(ident=Ident("mismatch_factor"))

    if 'gauss' in dist_type_lower:
        dist_call = Call(
            func=Ref(ident=Ident("gauss")),
            args=[Int(0), mismatch_factor_ref]
        )
    elif 'lnorm' in dist_type_lower:
        dist_call = Call(
            func=Ref(ident=Ident("lnorm")),
            args=[Int(0), mismatch_factor_ref]
        )
    else:
        return None

    mismatch_term = BinaryOp(tp=BinaryOperator.MUL, left=enable_mismatch_ref, right=dist_call)
    param_expr = BinaryOp(tp=BinaryOperator.ADD, left=base_expr, right=mismatch_term)

    # For ngspice, the syntax is the same (parameter function), but it will be written
    # as .param instead of .FUNC by the NgspiceNetlister.write_param_decl() method
    return ParamDecl(
        name=Ident(param_name),
        default=param_expr,
        distr=None
    )


def apply_mismatch_variation(program: Program, var: Variation, idx: int, stats_blocks: List[StatisticsBlock], output_format: Optional["NetlistDialects"] = None) -> None:
    """Apply a single mismatch variation by creating a function and replacing all references.
    
    This effectively "deletes" the parameter by:
    1. Creating a mismatch function with the original value (or process variation if it exists)
    2. Replacing ALL references to the parameter with the function call
    3. Removing the parameter declaration
    
    If the parameter has a process variation, the mismatch function will reference the __process__ version.
    """
    param_name = var.name.name
    
    # Check if parameter has a process variation - if so, mismatch should reference the __process__ version
    has_process = is_param_in_process_variations(stats_blocks, param_name)
    
    if has_process:
        # Parameter has process variation - mismatch should reference the __process__ version
        process_param_name = f"{param_name}__process__"
        # Find the process variation parameter
        process_param = None
        for file in program.files:
            for entry in file.contents:
                if isinstance(entry, ParamDecls):
                    for param in entry.params:
                        if param.name.name == process_param_name:
                            process_param = param
                            break
                    if process_param:
                        break
                if process_param:
                    break
            if process_param:
                break
        
        if process_param:
            # Use the process variation expression as the base for mismatch
            original_expr = process_param.default
            # Create mismatch function name with both suffixes
            mismatch_param_name = f"{param_name}__process____mismatch__(dummy_param)"
        else:
            # Process param not found, fall back to original
            matching_param = find_matching_param(program, param_name)
            if not matching_param:
                warn(f"Process variation parameter '{process_param_name}' not found for mismatch variation '{param_name}'. Skipping.")
                return
            original_expr = matching_param.default or Int(0)
            mismatch_param_name = f"{param_name}__mismatch__(dummy_param)"
    else:
        # No process variation - use original parameter
        matching_param = find_matching_param(program, param_name)
        if not matching_param:
            warn(f"No matching parameter found for mismatch variation '{param_name}'. Skipping.")
            return
        original_expr = matching_param.default or Int(0)
        mismatch_param_name = f"{param_name}__mismatch__(dummy_param)"

    # Create mismatch function with the correct name
    mismatch_param = create_mismatch_function(var, idx, original_expr=original_expr, output_format=output_format)
    if not mismatch_param:
        return
    
    # Update the mismatch parameter name if it has process variation
    if has_process:
        mismatch_param.name = Ident(mismatch_param_name)

    # Add param as a separate entry at the top of the file
    # Find the file that contains statistics blocks
    stats_file = None
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, StatisticsBlock):
                stats_file = file
                break
        if stats_file:
            break

    # Fallback to first file if no stats file found
    if not stats_file:
        stats_file = program.files[0]

    # Insert param at the beginning of the file (after lnorm/alnorm if they exist)
    # Look for existing ParamDecls containers that contain param functions, or find insertion point
    found_paramdecls = None
    insert_idx = 0
    
    # First, look for existing ParamDecls with param functions (like lnorm/alnorm or other mismatch params)
    for i, entry in enumerate(stats_file.contents):
        if isinstance(entry, ParamDecls):
            # Check if this ParamDecls is for param functions (contains params with parentheses in name)
            if entry.params and any('(' in param.name.name for param in entry.params):
                found_paramdecls = entry
                # Continue to find where to insert if we need a new container
                insert_idx = i + 1
            else:
                # Regular ParamDecls, skip it
                insert_idx = i + 1
        else:
            # Found first non-ParamDecls entry, insert before it
            insert_idx = i
            break

    # If we found an existing ParamDecls container for param functions, add to it
    if found_paramdecls:
        found_paramdecls.params.append(mismatch_param)
    else:
        # Create a new ParamDecls container with this param
        stats_file.contents.insert(insert_idx, ParamDecls(params=[mismatch_param]))

    # Create param function call to use as replacement (with dummy argument)
    # Extract the base name without (dummy_param)
    base_name = mismatch_param.name.name.split('(')[0]
    param_call = Call(func=Ref(ident=Ident(base_name)), args=[Int(0)])  # Use 0 as dummy arg

    # Replace ALL references to this parameter throughout the program
    # If parameter has process variation, replace references to the __process__ version
    # Otherwise, replace references to the original parameter
    if has_process:
        # Replace references to the __process__ version
        process_param_name = f"{var.name.name}__process__"
        replace_param_refs_in_program(program, process_param_name, param_call)
        # Also remove the __process__ parameter declaration
        remove_param_declaration(program, process_param_name)
        # Also remove the original parameter declaration (it was updated to reference __process__)
        remove_param_declaration(program, var.name.name)
    else:
        # Replace references to the original parameter
        replace_param_refs_in_program(program, var.name.name, param_call)
        # Remove the parameter declaration (effectively "deleting" it)
        remove_param_declaration(program, var.name.name)


def apply_all_mismatch_variations(program: Program, stats_blocks: List[StatisticsBlock], output_format: Optional["NetlistDialects"] = None) -> None:
    """Apply all mismatch variations from statistics blocks (Xyce/ngspice-specific)."""

    mismatch_idx = 0
    for stats in stats_blocks:
        if stats.mismatch:
            for var in stats.mismatch:
                apply_mismatch_variation(program, var, mismatch_idx, stats_blocks, output_format=output_format)
                mismatch_idx += 1


def apply_statistics_variations(program: Program, output_format: Optional["NetlistDialects"] = None) -> None:
    """Apply statistics vary statements to corresponding parameters.

    Automatically detects process {} and mismatch {} blocks and applies variations accordingly.
    Both process and mismatch variations use Monte Carlo (implied).
    Process variations may also include mean values for corner analysis.

    For Xyce and ngspice formats, replaces StatisticsBlocks in the AST with generated functions and process variations.
    For other formats, applies variations directly to parameters (legacy behavior).
    """
    stats_blocks = collect_statistics_blocks(program)
    if not stats_blocks:
        return  # No statistics to apply

    from .. import NetlistDialects
    is_xyce = output_format == NetlistDialects.XYCE
    is_ngspice = output_format == NetlistDialects.NGSPICE

    if is_xyce or is_ngspice:
        # Add lnorm functions if needed (for both Xyce and ngspice)
        if has_lognorm_distributions(stats_blocks):
            if is_xyce:
                add_lnorm_functions(program)
            else:  # ngspice
                add_lnorm_functions_ngspice(program)

        # For Xyce/ngspice: replace StatisticsBlocks in AST with generated content
        replace_statistics_blocks_with_generated_content(program, stats_blocks, output_format=output_format)
        
        # Move header comments (those that were before the first StatisticsBlock) to the top
        move_header_comments_to_top(program, stats_blocks)
    else:
        # Legacy behavior: apply directly to parameters (for non-Xyce/ngspice formats)
        apply_all_process_variations_legacy(program, stats_blocks)
