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
from typing import Tuple, Union, List, get_args

# Local Imports
from ..data import *
from .base import Netlister
from typing import Optional


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
        """Get our entry in the `NetlistFormat` enumeration"""
        from . import NetlistFormat

        return NetlistFormat.SPICE

    def write_subckt_def(self, module: SubcktDef) -> None:
        """Write the `SUBCKT` definition for `Module` `module`."""

        # Create the module name
        module_name = self.format_ident(module.name)
        # # Check for double-definition
        # if module_name in self.module_names:
        #     raise RuntimeError(f"Module {module_name} doubly defined")
        # # Add to our visited lists
        # self.module_names.add(module_name)
        # self.pmodules[module.name] = module

        # Create the sub-circuit definition header
        self.write(f".SUBCKT {module_name} \n")

        # Create its ports, if any are defined
        if module.ports:
            self.write_port_declarations(module)
        else:
            self.write("+ ")
            self.write_comment("No ports")

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

    def write_port_declarations(self, module: SubcktDef) -> None:
        """Write the port declarations for Module `module`."""
        self.write("+ ")
        for port in module.ports:
            self.write(self.format_ident(port) + " ")
        self.write("\n")

    def write_module_params(self, module: SubcktDef) -> None:
        """Write the parameter declarations for Module `module`.
        Parameter declaration format: `name1=val1 name2=val2 name3=val3 \n`"""
        self.write("+ ")
        for name, pparam in module.parameters.items():
            self.write_param_decl(name, pparam)
        self.write("\n")

    def write_subckt_instance(self, pinst: Instance) -> None:
        """Write sub-circuit-instance `pinst`."""

        # Detect if the instance looks like a MOS device (even if parsed as subcircuit instance)
        is_mos_like = (
            len(pinst.conns) == 4  # Exactly 4 ports (d, g, s, b)
            and isinstance(pinst.module, Ref)  # References a model
            and {'l', 'w'} <= {p.name.name for p in pinst.params}  # Has both 'l' and 'w' params
            and any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet'])  # Instance name indicates MOS
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

        # Write its parameter values
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
        self.write("+ ")
        for arg in pinst.args[:-1]:  # Ports only (exclude the model)
            if isinstance(arg, Ident):
                self.write(self.format_ident(arg) + " ")
            elif isinstance(arg, (Int, Float, MetricNum)):
                self.write(self.format_number(arg) + " ")
            else:
                self.write(self.format_expr(arg) + " ")
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

    # def format_concat(self, pconc: vlsir.circuit.Concat) -> str:
    #     """ Format the Concatenation of several other Connections """
    #     out = ""
    #     for part in pconc.parts:
    #         out += self.format_connection(part) + " "
    #     return out

    # @classmethod
    # def format_port_decl(cls, pport: vlsir.circuit.Port) -> str:
    #     """ Get a netlist `Port` definition """
    #     return cls.format_signal_ref(pport.signal)

    # @classmethod
    # def format_port_ref(cls, pport: vlsir.circuit.Port) -> str:
    #     """ Get a netlist `Port` reference """
    #     return cls.format_signal_ref(pport.signal)

    # @classmethod
    # def format_signal_ref(cls, psig: vlsir.circuit.Signal) -> str:
    #     """ Get a netlist definition for Signal `psig` """
    #     if psig.width < 1:
    #         raise RuntimeError
    #     if psig.width == 1:  # width==1, i.e. a scalar signal
    #         return psig.name
    #     # Vector/ multi "bit" Signal. Creates several spice signals.
    #     return " ".join(
    #         [f"{psig.name}{cls.format_bus_bit(k)}" for k in reversed(range(psig.width))]
    #     )

    # @classmethod
    # def format_signal_slice(cls, pslice: vlsir.circuit.Slice) -> str:
    #     """ Get a netlist definition for Signal-Slice `pslice` """
    #     base = pslice.signal
    #     indices = list(reversed(range(pslice.bot, pslice.top + 1)))
    #     if not len(indices):
    #         raise RuntimeError(f"Attempting to netlist empty slice {pslice}")
    #     return " ".join([f"{base}{cls.format_bus_bit(k)}" for k in indices])

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

    def write_param_val(self, param: ParamVal) -> None:
        """Write a parameter value"""

        name = self.format_ident(param.name)
        val = self.format_expr(param.val)
        self.write(f"{name}={val}")

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
        self.write(".param \n")
        for p in params.params:
            self.write("+ ")
            self.write_param_decl(p)
            self.write("\n")
        self.write("\n")

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
        """Get our entry in the `NetlistFormat` enumeration"""
        from . import NetlistFormat

        return NetlistFormat.HSPICE


class XyceNetlister(SpiceNetlister):
    """Xyce-Format Netlister"""

    @property
    def enum(self):
        """Get our entry in the `NetlistFormat` enumeration"""
        from . import NetlistFormat

        return NetlistFormat.XYCE

    def netlist(self) -> None:
        """Override netlist() to apply Xyce-specific statistics variations before writing.
        
        This applies statistics variations (process and mismatch) to the Program,
        creating Xyce-specific artifacts like .FUNC definitions for mismatch parameters
        and enable_mismatch parameter. This only happens when writing to Xyce format.
        """
        # Apply statistics variations with XYCE format before writing
        apply_statistics_variations(self.src, output_format=NetlistDialects.XYCE)
        
        # Call parent implementation to do the actual writing
        super().netlist()

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

    def write_instance_params(self, pvals: List[ParamVal]) -> None:
        """Write the parameter-values for Instance `pinst`.

        Parameter-values format:
        ```
        XNAME
        + <ports>
        + <subckt-name>
        + PARAMS: name1=val1 name2=val2 name3=val3
        """
        self.write("+ ")

        if not pvals:  # Write a quick comment for no parameters
            return self.write_comment("No parameters")

        self.write("PARAMS: ")  # <= Xyce-specific
        # And write them
        for pval in pvals:
            self.write_param_val(pval)
            self.write(" ")

        self.write("\n")

    def write_comment(self, comment: str) -> None:
        """Xyce comments *kinda* support the Spice-typical `*` charater,
        but *only* as the first character in a line.
        Any mid-line-starting comments must use `;` instead.
        So, just use it all the time."""
        self.write(f"; {comment}\n")

    def write_entry(self, entry) -> None:
        """Override write_entry to handle FunctionDef for Xyce."""
        if isinstance(entry, FunctionDef):
            return self.write_function_def(entry)
        # Call parent implementation for other entries
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

        # Create the module name
        module_name = self.format_ident(module.name)

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

        # Close up the sub-circuit
        self.write(".ENDS\n\n")

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


class NgspiceNetlister(SpiceNetlister):
    """FIXME: Ngspice-Format Netlister"""

    def __init__(self, *_, **__):
        raise NotImplementedError

    @property
    def enum(self):
        """Get our entry in the `NetlistFormat` enumeration"""
        from . import NetlistFormat

        return NetlistFormat.NGSPICE


class CdlNetlister(SpiceNetlister):
    """FIXME: CDL-Format Netlister"""

    def __init__(self, *_, **__):
        raise NotImplementedError

    @property
    def enum(self):
        """Get our entry in the `NetlistFormat` enumeration"""
        from . import NetlistFormat

        return NetlistFormat.CDL


# Xyce-specific statistics variations handling

def find_matching_param(program: Program, param_name: str) -> Optional[ParamDecl]:
    """Find a parameter by name in the program."""
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                for param in entry.params:
                    if param.name.name == param_name:
                        return param
    return None


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
    """Add lnorm and alnorm function definitions to the program (Xyce .FUNC definitions)."""
    if not program.files:
        return
    
    math_funcs = [
        FunctionDef(
            name=Ident("lnorm"),
            rtype=ArgType.REAL,
            args=[
                TypedArg(tp=ArgType.REAL, name=Ident("mu")),
                TypedArg(tp=ArgType.REAL, name=Ident("sigma")),
                TypedArg(tp=ArgType.REAL, name=Ident("seed"))
            ],
            stmts=[Return(val=Call(
                func=Ref(ident=Ident("exp")),
                args=[Call(
                    func=Ref(ident=Ident("gauss")),
                    args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma")), Ref(ident=Ident("seed"))]
                )]
            ))]
        ),
        FunctionDef(
            name=Ident("alnorm"),
            rtype=ArgType.REAL,
            args=[
                TypedArg(tp=ArgType.REAL, name=Ident("mu")),
                TypedArg(tp=ArgType.REAL, name=Ident("sigma")),
                TypedArg(tp=ArgType.REAL, name=Ident("seed"))
            ],
            stmts=[Return(val=Call(
                func=Ref(ident=Ident("exp")),
                args=[Call(
                    func=Ref(ident=Ident("gauss")),
                    args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma")), Ref(ident=Ident("seed"))]
                )]
            ))]
        )
    ]
    program.files[0].contents.extend(math_funcs)


def create_monte_carlo_distribution_call(dist_type: str, std: Float, seed: int) -> Optional[Call]:
    """Create a distribution function call (gauss or lnorm) for Monte Carlo variation.
    
    Returns a Call expression for the distribution function, or None if the distribution
    type is not supported. Warns if an unsupported distribution type is encountered.
    """
    if not dist_type:
        return None
    
    dist_type_lower = dist_type.lower()
    if 'gauss' in dist_type_lower:
        return Call(func=Ref(ident=Ident("gauss")), args=[Int(0), Float(1), Int(1)])
    elif 'lnorm' in dist_type_lower:
        return Call(func=Ref(ident=Ident("lnorm")), args=[Int(0), Float(1), Int(1)])
    
    # Warn about unsupported distribution type
    warn(f"Unsupported distribution type '{dist_type}' for Monte Carlo variation. Supported types: gauss, lnorm")
    return None


def apply_monte_carlo_variation(original_expr: Expr, var: Variation) -> Expr:
    """Apply Monte Carlo variation to an expression: original * (1 + std * dist(...))."""
    if not var.dist:
        return original_expr
    
    dist_call = create_monte_carlo_distribution_call(var.dist, var.std, 1)
    if not dist_call:
        return original_expr
    
    variation_term = BinaryOp(tp=BinaryOperator.MUL, left=var.std, right=dist_call)
    one_plus_variation = BinaryOp(tp=BinaryOperator.ADD, left=Float(1), right=variation_term)
    return BinaryOp(tp=BinaryOperator.MUL, left=original_expr, right=one_plus_variation)


def apply_process_variation(program: Program, var: Variation) -> None:
    """Apply a single process variation to its corresponding parameter."""
    matching_param = find_matching_param(program, var.name.name)
    if not matching_param:
        return
    
    original_expr = matching_param.default or Int(0)
    new_expr = apply_monte_carlo_variation(original_expr, var)
    
    # Add mean value if present (for corner analysis)
    if var.mean:
        new_expr = BinaryOp(tp=BinaryOperator.ADD, left=new_expr, right=var.mean)
    
    matching_param.default = new_expr


def apply_all_process_variations(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Apply all process variations from statistics blocks."""
    for stats in stats_blocks:
        if stats.process:
            for var in stats.process:
                apply_process_variation(program, var)


def add_enable_mismatch_parameter(program: Program) -> None:
    """Add enable_mismatch parameter to the program if it doesn't exist."""
    if find_matching_param(program, "enable_mismatch"):
        return  # Already exists
    
    if not program.files:
        return
    
    enable_mismatch_decl = ParamDecl(name=Ident("enable_mismatch"), default=Float(1.0), distr=None)
    
    # Try to find existing ParamDecls to add to
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                entry.params.append(enable_mismatch_decl)
                return
    
    # If no ParamDecls found, create a new one
    program.files[0].contents.insert(0, ParamDecls(params=[enable_mismatch_decl]))


def create_mismatch_function(var: Variation, idx: int) -> Optional[FunctionDef]:
    """Create a mismatch function definition for a variation."""
    if not var.dist:
        return None
    
    func_name = f"{var.name.name}_mismatch"
    dist_type_lower = var.dist.lower()
    
    if 'gauss' in dist_type_lower:
        dist_func_name = "gauss"
    elif 'lnorm' in dist_type_lower:
        dist_func_name = "lnorm"
    else:
        return None
    
    dist_call = Call(
        func=Ref(ident=Ident(dist_func_name)),
        args=[Int(0), var.std, Int(idx + 1)]  # seed = idx+1 for uniqueness
    )
    
    # Multiply by enable_mismatch to allow toggling mismatch on/off
    enable_ref = Ref(ident=Ident("enable_mismatch"))
    mismatch_value = BinaryOp(tp=BinaryOperator.MUL, left=enable_ref, right=dist_call)
    
    return FunctionDef(
        name=Ident(func_name),
        rtype=ArgType.REAL,
        args=[],  # No arguments - called as param_name_mismatch()
        stmts=[Return(val=mismatch_value)]
    )


def apply_mismatch_variation(program: Program, var: Variation, idx: int) -> None:
    """Apply a single mismatch variation by creating a function and updating the parameter."""
    matching_param = find_matching_param(program, var.name.name)
    if not matching_param:
        return
    
    mismatch_func = create_mismatch_function(var, idx)
    if not mismatch_func:
        return
    
    func_name = mismatch_func.name.name
    
    # Add function to the first file
    if program.files:
        program.files[0].contents.append(mismatch_func)
    
    # Update parameter to reference the function call
    original_expr = matching_param.default or Int(0)
    func_call = Call(func=Ref(ident=Ident(func_name)), args=[])
    new_expr = BinaryOp(tp=BinaryOperator.ADD, left=original_expr, right=func_call)
    matching_param.default = new_expr


def apply_all_mismatch_variations(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Apply all mismatch variations from statistics blocks (Xyce-specific)."""
    add_enable_mismatch_parameter(program)
    
    mismatch_idx = 0
    for stats in stats_blocks:
        if stats.mismatch:
            for var in stats.mismatch:
                apply_mismatch_variation(program, var, mismatch_idx)
                mismatch_idx += 1


def apply_statistics_variations(program: Program, output_format: Optional[NetlistDialects] = None) -> None:
    """Apply statistics vary statements to corresponding parameters.
    
    Automatically detects process {} and mismatch {} blocks and applies variations accordingly.
    Both process and mismatch variations use Monte Carlo (implied).
    Process variations may also include mean values for corner analysis.
    
    Xyce-specific transformations (mismatch functions, enable_mismatch parameter, lnorm function definitions)
    are only applied when output_format is NetlistDialects.XYCE. This allows the same program to be
    written to different formats (e.g., Spectre) without Xyce-specific artifacts.
    """
    stats_blocks = collect_statistics_blocks(program)
    if not stats_blocks:
        return  # No statistics to apply
    
    is_xyce = output_format == NetlistDialects.XYCE
    
    # Add lnorm functions if needed (Xyce-specific)
    if has_lognorm_distributions(stats_blocks) and is_xyce:
        add_lnorm_functions(program)
    
    # Apply process variations (format-agnostic)
    apply_all_process_variations(program, stats_blocks)
    
    # Apply mismatch variations (Xyce-specific)
    if is_xyce:
        apply_all_mismatch_variations(program, stats_blocks)
