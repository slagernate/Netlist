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
from typing import Tuple, Union, List, get_args, Dict, IO

# Local Imports
from ..data import *
from .base import Netlister, ErrorMode
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
        # Store current subcircuit context for use in write_subckt_instance
        self._current_subckt = module

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
        conn_count = len(pinst.conns)
        has_l_w = {'l', 'w'} <= {p.name.name for p in pinst.params}
        is_module_ref = isinstance(pinst.module, Ref)
        name_has_mos = any(keyword in pinst.name.name.lower() for keyword in ['mos', 'fet'])
        
        is_mos_like = (
            conn_count == 4  # Exactly 4 ports (d, g, s, b)
            and is_module_ref  # References a model
            and has_l_w  # Has both 'l' and 'w' params
            and name_has_mos  # Instance name indicates MOS
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
        
        # Write its parameter values
        module_ref = pinst.module if isinstance(pinst.module, Ref) else None
        self.write_instance_params(pinst.params, is_mos_like=is_mos_like, module_ref=module_ref)

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
        for p in params.params:
            self.write(".param ")
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
        # This modifies the AST directly, replacing StatisticsBlocks with generated content
        apply_statistics_variations(self.src, output_format=NetlistDialects.XYCE)

        # Call parent implementation to do the actual writing
        super().netlist()

    def _validate_content(self) -> None:
        """Validate that the program content matches the expected file_type."""
        from ..data import LibSectionDef

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
        params_to_write = pvals
        if module_ref:
            is_bsim4 = self._is_bsim4_model_ref(module_ref)
            if is_bsim4:
                # Check if deltox is present before filtering
                has_deltox = any(pval.name.name == "deltox" for pval in pvals)
                if has_deltox:
                    warn(f"Filtering 'deltox' parameter from instance parameters for BSIM4 model '{module_ref.ident.name}'. "
                         f"Xyce does not support deltox for BSIM4 models.")
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

        # Add a blank after each instance
        self.write("\n")

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
            elif isinstance(arg, (Int, Float, MetricNum)):
                self.write(self.format_number(arg) + " ")
            else:
                self.write(self.format_expr(arg) + " ")
        self.write("\n")
        # Write the model (last arg) on another continuation line
        self.write("+ " + self.format_ident(pinst.args[-1]) + " \n")

        # Extract module_ref from args for BSIM4 detection
        module_ref = None
        if pinst.args and isinstance(pinst.args[-1], Ref):
            module_ref = pinst.args[-1]
        
        # Filter deltox if referencing BSIM4 model
        kwargs_to_write = pinst.kwargs
        if module_ref and self._is_bsim4_model_ref(module_ref):
            # Check if deltox is present before filtering
            has_deltox = any(kw.name.name == "deltox" for kw in pinst.kwargs)
            if has_deltox:
                warn(f"Filtering 'deltox' parameter from instance parameters for BSIM4 model '{module_ref.ident.name}'. "
                     f"Xyce does not support deltox for BSIM4 models.")
            kwargs_to_write = [kw for kw in pinst.kwargs if kw.name.name != "deltox"]
        
        self.write("+ ")
        for kwarg in kwargs_to_write:
            self.write_param_val(kwarg)
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
        mname = self.format_ident(model.name)
        mtype = self.format_ident(model.mtype).lower()
        
        # Use local variable for params to avoid mutating the model object
        params_to_write = list(model.params)  # Create a copy
        
        # Handle BSIM4: replace with pmos/nmos, add level, omit deltox
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
            
            # Filter out deltox parameter
            params_to_write = [p for p in params_to_write if p.name.name != "deltox"]
            
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
    that match the parameter name.
    
    Args:
        expr: The expression to check
        param_name: The parameter name to search for
        
    Returns:
        True if the expression references the parameter, False otherwise
    """
    # Handle Ref nodes - check if name matches
    if isinstance(expr, Ref):
        return expr.ident.name == param_name
    
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


def replace_param_ref_in_expr(expr: Expr, param_name: str, func_call: Call) -> Expr:
    """Recursively replace Ref nodes matching param_name with func_call in an expression."""
    # Handle Ref nodes - this is the key replacement
    if isinstance(expr, Ref):
        if expr.ident.name == param_name:
            return func_call
        return expr
    
    # Handle Call nodes - recurse into arguments
    if isinstance(expr, Call):
        new_args = [replace_param_ref_in_expr(arg, param_name, func_call) for arg in expr.args]
        if new_args != expr.args:
            return Call(func=expr.func, args=new_args)
        return expr
    
    # Handle BinaryOp - recurse into left and right
    if isinstance(expr, BinaryOp):
        new_left = replace_param_ref_in_expr(expr.left, param_name, func_call)
        new_right = replace_param_ref_in_expr(expr.right, param_name, func_call)
        if new_left != expr.left or new_right != expr.right:
            return BinaryOp(tp=expr.tp, left=new_left, right=new_right)
        return expr
    
    # Handle UnaryOp - recurse into target
    if isinstance(expr, UnaryOp):
        new_targ = replace_param_ref_in_expr(expr.targ, param_name, func_call)
        if new_targ != expr.targ:
            return UnaryOp(tp=expr.tp, targ=new_targ)
        return expr
    
    # Handle TernOp - recurse into all three parts
    if isinstance(expr, TernOp):
        new_cond = replace_param_ref_in_expr(expr.cond, param_name, func_call)
        new_if_true = replace_param_ref_in_expr(expr.if_true, param_name, func_call)
        new_if_false = replace_param_ref_in_expr(expr.if_false, param_name, func_call)
        if (new_cond != expr.cond or new_if_true != expr.if_true or 
            new_if_false != expr.if_false):
            return TernOp(cond=new_cond, if_true=new_if_true, if_false=new_if_false)
        return expr
    
    # Literals and other types - no replacement needed
    return expr


def replace_param_refs_in_entry(entry: Entry, param_name: str, func_call: Call) -> None:
    """Replace parameter references in a single Entry."""
    # ParamDecls - replace in each param's default
    if isinstance(entry, ParamDecls):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, func_call)
    
    # SubcktDef - replace in params and recurse into entries
    elif isinstance(entry, SubcktDef):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, func_call)
        for sub_entry in entry.entries:
            replace_param_refs_in_entry(sub_entry, param_name, func_call)
    
    # ModelDef - replace in params
    elif isinstance(entry, ModelDef):
        for param in entry.params:
            if param.default is not None:
                param.default = replace_param_ref_in_expr(param.default, param_name, func_call)
    
    # ModelFamily - recurse into variants
    elif isinstance(entry, ModelFamily):
        for variant in entry.variants:
            for param in variant.params:
                if param.default is not None:
                    param.default = replace_param_ref_in_expr(param.default, param_name, func_call)
    
    # FunctionDef - replace in return statement
    elif isinstance(entry, FunctionDef):
        for stmt in entry.stmts:
            if isinstance(stmt, Return):
                stmt.val = replace_param_ref_in_expr(stmt.val, param_name, func_call)
    
    # Instance - replace in param values
    elif isinstance(entry, Instance):
        for param_val in entry.params:
            param_val.val = replace_param_ref_in_expr(param_val.val, param_name, func_call)
    
    # Primitive - replace in kwargs
    elif isinstance(entry, Primitive):
        for param_val in entry.kwargs:
            param_val.val = replace_param_ref_in_expr(param_val.val, param_name, func_call)
    
    # Options - replace in option values (if Expr, not QuotedString)
    elif isinstance(entry, Options):
        for option in entry.vals:
            # OptionVal = Union[QuotedString, Expr], so check if it's not QuotedString
            if not isinstance(option.val, QuotedString):
                option.val = replace_param_ref_in_expr(option.val, param_name, func_call)
    
    # LibSectionDef - recurse into entries
    elif isinstance(entry, LibSectionDef):
        for sub_entry in entry.entries:
            replace_param_refs_in_entry(sub_entry, param_name, func_call)


def replace_param_refs_in_program(program: Program, param_name: str, func_call: Call) -> None:
    """Replace all references to param_name throughout the entire program with func_call."""
    for file in program.files:
        for entry in file.contents:
            replace_param_refs_in_entry(entry, param_name, func_call)


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


def replace_statistics_blocks_with_generated_content(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Replace StatisticsBlocks in the AST with generated functions and process variations for library parameters."""
    # First apply process variations to all parameters (library and global)
    # This maintains backward compatibility for global parameters
    apply_all_process_variations_legacy(program, stats_blocks)

    # Verify all parameters exist (error if any missing)
    verify_process_variation_params(program, stats_blocks)

    # Process each statistics block
    for stats in stats_blocks:
        # Collect generated content to replace the StatisticsBlock
        generated_content = []

        # Add process variations for library parameters only
        if stats.process:
            # Create process variation expressions
            process_variations = []
            for var in stats.process:
                param_name = var.name.name
                if is_param_in_library_section(program, param_name):
                    varied_expr = create_relative_process_variation_expr(param_name, var)
                    process_variations.append((param_name, varied_expr))

            if process_variations:
                # Create a ParamDecls with the process variations
                param_decls = []
                for param_name, varied_expr in process_variations:
                    param_decls.append(ParamDecl(name=Ident(param_name), default=varied_expr, distr=None))
                generated_content.append(ParamDecls(params=param_decls))

        # Replace the StatisticsBlock with generated content
        for file in program.files:
            for i, entry in enumerate(file.contents):
                if isinstance(entry, StatisticsBlock) and entry is stats:
                    file.contents[i:i+1] = generated_content
                    break

    # Now apply mismatch variations (creates functions and replaces references)
    apply_all_mismatch_variations(program, stats_blocks)


def create_mismatch_function(var: Variation, idx: int) -> Optional[ParamDecl]:
    """Create a mismatch .param declaration with dummy parameter syntax as requested.

    Creates: mm_z1__mismatch__(dummy_param) with expression 0+enable_mismatch*gauss(0,mismatch_factor)
    This is placed near the top and called as mm_z1__mismatch__(0).
    """
    if not var.dist:
        return None

    param_name = f"{var.name.name}__mismatch__(dummy_param)"
    dist_type_lower = var.dist.lower()

    # Create expression: 0 + enable_mismatch * gauss(0,mismatch_factor)
    zero = Int(0)
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
    param_expr = BinaryOp(tp=BinaryOperator.ADD, left=zero, right=mismatch_term)

    return ParamDecl(
        name=Ident(param_name),
        default=param_expr,
        distr=None
    )


def apply_mismatch_variation(program: Program, var: Variation, idx: int, stats_blocks: List[StatisticsBlock]) -> None:
    """Apply a single mismatch variation by creating a function and replacing all references.
    
    This effectively "deletes" the parameter by:
    1. Creating a mismatch function with the original value
    2. Replacing ALL references to the parameter with the function call
    3. Removing the parameter declaration
    """
    matching_param = find_matching_param(program, var.name.name)
    if not matching_param:
        # Check if parameter is in process variations - if so, silently skip (it's process-only)
        if is_param_in_process_variations(stats_blocks, var.name.name):
            return  # Silently skip - parameter is process-only, not a real error
        # Otherwise, warn (real error - parameter doesn't exist)
        warn(f"No matching parameter found for mismatch variation '{var.name.name}'. Skipping.")
        return

    mismatch_param = create_mismatch_function(var, idx)
    if not mismatch_param:
        return

    param_name = mismatch_param.name.name

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
    base_name = param_name.split('(')[0]
    param_call = Call(func=Ref(ident=Ident(base_name)), args=[Int(0)])  # Use 0 as dummy arg

    # Replace ALL references to this parameter throughout the program
    replace_param_refs_in_program(program, var.name.name, param_call)

    # Remove the parameter declaration (effectively "deleting" it)
    remove_param_declaration(program, var.name.name)


def apply_all_mismatch_variations(program: Program, stats_blocks: List[StatisticsBlock]) -> None:
    """Apply all mismatch variations from statistics blocks (Xyce-specific)."""

    mismatch_idx = 0
    for stats in stats_blocks:
        if stats.mismatch:
            for var in stats.mismatch:
                apply_mismatch_variation(program, var, mismatch_idx, stats_blocks)
                mismatch_idx += 1


def apply_statistics_variations(program: Program, output_format: Optional[NetlistDialects] = None) -> None:
    """Apply statistics vary statements to corresponding parameters.

    Automatically detects process {} and mismatch {} blocks and applies variations accordingly.
    Both process and mismatch variations use Monte Carlo (implied).
    Process variations may also include mean values for corner analysis.

    For Xyce format, replaces StatisticsBlocks in the AST with generated functions and process variations.
    For other formats, applies variations directly to parameters (legacy behavior).
    """
    stats_blocks = collect_statistics_blocks(program)
    if not stats_blocks:
        return  # No statistics to apply

    is_xyce = output_format == NetlistDialects.XYCE

    if is_xyce:
        # Add lnorm functions if needed (Xyce-specific)
        if has_lognorm_distributions(stats_blocks):
            add_lnorm_functions(program)

        # For Xyce: replace StatisticsBlocks in AST with generated content
        replace_statistics_blocks_with_generated_content(program, stats_blocks)
    else:
        # Legacy behavior: apply directly to parameters (for non-Xyce formats)
        apply_all_process_variations_legacy(program, stats_blocks)

