""" 

# Netlist Data Model 

All elements in the netlist-internal "IR", 
primarily in the form of dataclasses. 

"""

# Std-Lib Imports
from enum import Enum
from pathlib import Path
from dataclasses import field
from typing import Optional, Union, List, Tuple, Generic, TypeVar

# PyPi Imports
from pydantic.dataclasses import dataclass
from pydantic import BaseModel


class NetlistParseError(Exception):
    """Netlist Parse Error"""

    @staticmethod
    def throw(*args, **kwargs):
        """Exception-raising debug wrapper. Breakpoint to catch `NetlistParseError`s."""
        raise NetlistParseError(*args, **kwargs)


class NetlistDialects(Enum):
    """Enumerated, Supported Netlist Dialects"""

    SPECTRE = "spectre"
    SPECTRE_SPICE = "spectre_spice"
    SPICE = "spice"
    HSPICE = "hspice"
    NGSPICE = "ngspice"
    XYCE = "xyce"
    CDL = "cdl"

    @staticmethod
    def get(spec: "NetlistFormatSpec") -> "NetlistDialects":
        """Get the format specified by `spec`, in either enum or string terms.
        Only does real work in the case when `spec` is a string, otherwise returns it unchanged."""
        if isinstance(spec, (NetlistDialects, str)):
            return NetlistDialects(spec)
        raise TypeError


# Type-alias for specifying format, either in enum or string terms
NetlistFormatSpec = Union[NetlistDialects, str]


def to_json(arg) -> str:
    """Dump any `pydantic.dataclass` or simple combination thereof to JSON string."""
    import json
    from pydantic.json import pydantic_encoder
    from .ast_to_cst import Scope
    
    # Handle circular references in Scope objects
    seen_scopes = set()
    
    def custom_encoder(obj):
        # Handle Scope objects with circular parent references
        if isinstance(obj, Scope):
            obj_id = id(obj)
            if obj_id in seen_scopes:
                # Return a reference marker instead of the full object to break circular reference
                return {
                    "$ref": "circular_scope_reference",
                    "name": getattr(obj, 'name', None),
                    "sid": str(getattr(obj, 'sid', '')) if hasattr(obj, 'sid') else None
                }
            seen_scopes.add(obj_id)
            
            # Create a dict representation, replacing parent with just a reference
            result = {}
            for key, value in obj.__dict__.items():
                if key == 'parent' and value is not None:
                    # Replace parent with just name/sid reference
                    result[key] = {
                        "$ref": "parent_scope",
                        "name": getattr(value, 'name', None),
                        "sid": str(getattr(value, 'sid', '')) if hasattr(value, 'sid') else None
                    }
                else:
                    result[key] = value
            return result
        
        try:
            return pydantic_encoder(obj)
        except (TypeError, ValueError, RecursionError):
            # Fallback for objects that pydantic can't encode
            if hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            return str(obj)
    
    return json.dumps(arg, indent=2, default=custom_encoder)


def write_json(obj: Union["Program", "SourceFile", "Entry", "Scope"], path: Union[str, Path]) -> None:
    """Write AST/CST object to JSON file.
    
    Args:
        obj: Program, SourceFile, Entry, or Scope to serialize
        path: Output file path (str or Path)
    """
    from pathlib import Path
    from .ast_to_cst import Scope
    from .data import Program
    
    path = Path(path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # For Scope objects, simplify the output
    if isinstance(obj, Scope):
        json_str = _scope_to_json_simplified(obj)
    elif isinstance(obj, Program):
        json_str = _program_to_json_simplified(obj)
    else:
        json_str = to_json(obj)
    
    with open(path, "w") as f:
        f.write(json_str)


def _scope_to_json_simplified(scope: "Scope") -> str:
    """Convert Scope to clean, simplified JSON showing the syntax tree structure."""
    import json
    from .ast_to_cst import NoRedefDict
    from .data import Ident, ParamVal, Expr
    
    seen_scopes = set()
    
    def simplify_value(val, depth=0):
        """Recursively simplify values, keeping only essential structure."""
        if isinstance(val, NoRedefDict):
            # Unwrap NoRedefDict - just return the data dict
            return simplify_value(val.data, depth)
        elif isinstance(val, dict):
            # Filter out empty dicts and simplify values
            result = {}
            for k, v in val.items():
                if k in ('__pydantic_initialised__', 'source_info'):
                    continue
                simplified = simplify_value(v, depth + 1)
                # Skip empty dicts and lists
                if simplified not in ({}, [], None):
                    result[k] = simplified
            return result if result else None
        elif isinstance(val, list):
            # Simplify list items
            simplified = [simplify_value(item, depth + 1) for item in val]
            filtered = [item for item in simplified if item not in ({}, [], None)]
            return filtered if filtered else None
        elif hasattr(val, '__class__') and val.__class__.__name__ == 'Scope':
            # Handle Scope circular references
            obj_id = id(val)
            if obj_id in seen_scopes:
                return {"$ref": "scope", "name": getattr(val, 'name', None)}
            seen_scopes.add(obj_id)
            
            # Recursively simplify scope - only show key structure
            result = {}
            if getattr(val, 'name', None):
                result["name"] = val.name
            result["type"] = str(getattr(val, 'stype', '')).replace('ScopeType.', '')
            
            # Add children if any
            if hasattr(val, 'children') and val.children:
                result["children"] = {k: simplify_value(v, depth + 1) for k, v in val.children.items()}
            
            # Add key collections (instances, models, etc.)
            for attr in ['subckt_instances', 'primitive_instances', 'models', 'params', 'subckts', 'functions']:
                if hasattr(val, attr):
                    items = simplify_value(getattr(val, attr), depth + 1)
                    if items:
                        result[attr] = items
            
            return result
        elif isinstance(val, Ident):
            # Simplify Ident - just show the name
            return val.name if hasattr(val, 'name') else str(val)
        elif isinstance(val, ParamVal):
            # Simplify ParamVal - show just the value
            if hasattr(val, 'val'):
                return simplify_value(val.val, depth + 1)
            return str(val)
        elif hasattr(val, '__class__') and hasattr(val, 'val'):
            # For value objects (MetricNum, Int, Float, etc.), extract the value
            cls_name = val.__class__.__name__
            if cls_name in ('MetricNum', 'Int', 'Float', 'String'):
                return val.val if hasattr(val, 'val') else str(val)
        elif hasattr(val, 'name') and hasattr(val, '__class__'):
            # For named objects (like ParamDecl, Instance, etc.), show name and key fields
            result = {"name": getattr(val, 'name', None)}
            if hasattr(val, 'name') and isinstance(val.name, Ident):
                result["name"] = val.name.name if hasattr(val.name, 'name') else str(val.name)
            
            # Add key fields based on type
            cls_name = val.__class__.__name__
            if cls_name == 'Instance':
                if hasattr(val, 'module'):
                    mod = val.module
                    if hasattr(mod, 'ident') and hasattr(mod.ident, 'name'):
                        result["module"] = mod.ident.name
                if hasattr(val, 'conns'):
                    result["conns"] = [c.name if hasattr(c, 'name') else str(c) for c in val.conns]
                if hasattr(val, 'params'):
                    params = simplify_value(val.params, depth + 1)
                    if params:
                        result["params"] = params
            elif cls_name == 'Primitive':
                if hasattr(val, 'ptype'):
                    result["type"] = str(val.ptype).replace('PrimitiveType.', '')
                if hasattr(val, 'conns'):
                    result["conns"] = [c.name if hasattr(c, 'name') else str(c) for c in val.conns]
                if hasattr(val, 'params'):
                    params = simplify_value(val.params, depth + 1)
                    if params:
                        result["params"] = params
            elif cls_name == 'ModelDef':
                if hasattr(val, 'mtype') and hasattr(val.mtype, 'name'):
                    result["mtype"] = val.mtype.name
                if hasattr(val, 'params'):
                    params = simplify_value(val.params, depth + 1)
                    if params:
                        result["params"] = params
            elif cls_name == 'ParamDecl':
                if hasattr(val, 'default'):
                    result["default"] = simplify_value(val.default, depth + 1)
            
            return result
        elif hasattr(val, '__dict__'):
            # For other objects, try to extract key info
            d = val.__dict__
            if 'name' in d:
                return simplify_value(d['name'], depth + 1)
            # For expressions, just show a simplified representation
            if hasattr(val, '__class__'):
                cls_name = val.__class__.__name__
                if 'Expr' in cls_name or 'Call' in cls_name:
                    return f"<{cls_name}>"  # Don't expand expressions deeply
            return str(val)
        else:
            return val
    
    simplified = simplify_value(scope)
    return json.dumps(simplified, indent=2, default=str)


def _program_to_json_simplified(program: "Program") -> str:
    """Convert Program to clean, simplified JSON showing the AST structure used by the writer."""
    import json
    from .data import SubcktDef, Instance, Primitive, ModelDef, ParamDecls
    
    def simplify_entry(entry, depth=0):
        """Simplify an AST entry to show its structure."""
        cls_name = entry.__class__.__name__
        result = {"type": cls_name}
        
        if isinstance(entry, SubcktDef):
            result["name"] = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            if hasattr(entry, 'ports'):
                result["ports"] = [p.name if hasattr(p, 'name') else str(p) for p in entry.ports]
            if hasattr(entry, 'entries'):
                result["entries"] = [simplify_entry(e, depth + 1) for e in entry.entries]
        elif isinstance(entry, Instance):
            result["name"] = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            if hasattr(entry, 'module') and hasattr(entry.module, 'ident'):
                result["module"] = entry.module.ident.name if hasattr(entry.module.ident, 'name') else str(entry.module.ident)
            if hasattr(entry, 'conns'):
                result["conns"] = [c.name if hasattr(c, 'name') else str(c) for c in entry.conns]
        elif isinstance(entry, Primitive):
            result["name"] = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            # Primitives store connections in args, params in kwargs
            if hasattr(entry, 'args'):
                result["args"] = [simplify_entry(a, depth + 1) if hasattr(a, '__class__') else str(a) for a in entry.args]
            if hasattr(entry, 'kwargs'):
                params = [{"name": p.name.name if hasattr(p.name, 'name') else str(p.name), 
                          "val": str(p.val)[:50]} for p in entry.kwargs] if hasattr(entry, 'kwargs') else []
                if params:
                    result["params"] = params
        elif isinstance(entry, ModelDef):
            result["name"] = entry.name.name if hasattr(entry.name, 'name') else str(entry.name)
            if hasattr(entry, 'mtype') and hasattr(entry.mtype, 'name'):
                result["mtype"] = entry.mtype.name
        elif isinstance(entry, ParamDecls):
            if hasattr(entry, 'params'):
                result["params"] = [p.name.name if hasattr(p.name, 'name') else str(p.name) for p in entry.params]
        
        return result
    
    result = {
        "files": [
            {
                "path": str(f.path),
                "entries": [simplify_entry(e) for e in f.contents]
            }
            for f in program.files
        ]
    }
    
    return json.dumps(result, indent=2, default=str)


@dataclass
class SourceInfo:
    """Parser Source Information"""

    line: int  # Source-File Line Number
    dialect: "NetlistDialects"  # Netlist Dialect


# Keep a list of datatypes defined here,
# primarily so that we can update their forward-references at the end of this module.
datatypes = [SourceInfo]


def datatype(cls: type) -> type:
    """Register a class as a datatype."""

    # Add an `Optional[SourceInfo]` field to the class, with a default value of `None`.
    # Creates the `__annotations__` field if it does not already exist.
    anno = getattr(cls, "__annotations__", {})
    anno["source_info"] = Optional[SourceInfo]
    cls.__annotations__ = anno
    cls.source_info = None

    # Convert it to a `pydantic.dataclasses.dataclass`
    cls = dataclass(cls)

    # And add it to the list of datatypes
    datatypes.append(cls)
    return cls


@datatype
class Ident:
    """Identifier"""

    name: str


class RefType(Enum):
    """External Reference Types Enumeration
    Store on each `ExternalRef` to note which types would be valid in context."""

    SUBCKT = "SUBCKT"  # Sub-circuit definition
    MODEL = "MODEL"  # Model definition
    PARAM = "PARAM"  # Parameter reference
    FUNCTION = "FUNCTION"  # Function definition
    FILEPATH = "FILEPATH"  # File-system path, e.g. in `Include`s not parsed

    def __repr__(self):
        return f"RefType.{self.name}"


@datatype
class ExternalRef:
    """# External Reference
    Name-based reference to something outside the netlist-program."""

    # Referred-to identifier
    ident: Ident
    # List of types which this can validly refer to
    types: List[RefType] = field(default_factory=list)


# Generic type of "the thing that a `Ref` refers to"
Referent = TypeVar("Referent")


class Ref(BaseModel, Generic[Referent]):
    """# Reference to another Netlist object
    Intially an identifier, then in later stages resolved to a generic `Referent`."""

    # Referred-to identifier
    ident: Ident
    # Resolved referent, or `None` if unresolved.
    resolved: Optional[Union[Referent, ExternalRef]] = None


@datatype
class HierPath:
    """Hierarchical Path Identifier"""

    path: List[Ident]  # FIXME: Ref?


@datatype
class ParamDecl:
    """Parameter Declaration
    Includes Optional Distribution Information"""

    name: Ident
    default: Optional["Expr"]
    distr: Optional[str] = None


@datatype
class ParamDecls:
    """Parameter Declarations,
    as via the `param` keywords."""

    params: List[ParamDecl]


@datatype
class ParamVal:
    """Parameter Value-Set"""

    name: Ident
    val: "Expr"


@datatype
class QuotedString:
    """Quoted String Value"""

    txt: str


# Simulation Options can take on the values of expressions, e.g. parameter combinations,
# and those of quoted strings, often for file-path.
OptionVal = Union[QuotedString, "Expr"]


@datatype
class Option:
    """Simulation Option"""

    name: Ident  # Option Name
    val: OptionVal  # Option Value


@datatype
class Options:
    """Simulation Options"""

    name: Optional[Ident]  # Option Name. FIXME: could this be removed
    vals: List[Option]  # List of [`Option`]s


@datatype
class StartSubckt:
    """Start of a Subckt / Module Definition"""

    name: Ident  # Module/ Subcircuit Name
    ports: List[Ident]  # Port List
    params: List[ParamDecl]  # Parameter Declarations


@datatype
class EndSubckt:
    """End of a Subckt / Module Definition"""

    name: Optional[Ident]


@datatype
class SubcktDef:
    """Sub-Circuit / Module Definition"""

    name: Ident  # Module/ Subcircuit Name
    ports: List[Ident]  # Port List
    params: List[ParamDecl]  # Parameter Declarations
    entries: List["Entry"]


@datatype
class ModelDef:
    """Model Definition"""

    name: Ident  # Model Name
    mtype: Ident  # Model Type
    args: List[Ident]  # Positional Arguments
    params: List[ParamDecl]  # Parameter Declarations & Defaults


@datatype
class ModelVariant:
    """Model Variant within a `ModelFamily`"""

    model: Ident  # Model Family Name
    variant: Ident  # Variant Name
    mtype: Ident  # Model Type
    args: List[Ident]  # Positional Arguments
    params: List[ParamDecl]  # Parameter Declarations & Defaults


@datatype
class ModelFamily:
    """Model Family
    A set of related, identically named models, generally separated by limiting parameters such as {lmin, lmax} or {wmin, wmax}."""

    name: Ident  # Model Family Name
    mtype: Ident  # Model Type
    variants: List[ModelVariant]  # Variants


# The primary `Model` type-union includes both single-variant and multi-variant versions
Model = Union[ModelDef, ModelFamily]


@datatype
class Instance:
    """Subckt / Module Instance"""

    name: Ident  # Instance Name
    module: Ref[Union[SubcktDef, Model]]  # Module/ Subcircuit Reference

    # Connections, either by-position or by-name
    conns: Union[List[Ident], List[Tuple[Ident, Ident]]]
    params: List[ParamVal]  # Parameter Values


@datatype
class Primitive:
    """
    Primitive Instance

    Note at parsing-time, before models are sorted out,
    it is not always clear what is a port, model name, and parameter value.
    Primitives instead store positional and keyword arguments `args` and `kwargs`.
    """

    name: Ident
    args: List["Expr"]
    kwargs: List[ParamVal]


@datatype
class Include:
    """Include (a File) Statement"""

    path: Path


@datatype
class AhdlInclude:
    """Analog HDL Include (a File) Statement"""

    path: Path


@datatype
class StartLib:
    """Start of a `Library`"""

    name: Ident


@datatype
class EndLib:
    """End of a `Library`"""

    name: Optional[Ident]


@datatype
class StartLibSection:
    """Start of a `LibrarySection`"""

    name: Ident


@datatype
class EndLibSection:
    """End of a `LibrarySection`"""

    name: Ident


@datatype
class LibSectionDef:
    """Library Section
    A named section of a library, commonly incorporated with a `UseLibSection` or similar."""

    name: Ident  # Section Name
    entries: List["Entry"]  # Entry List


@datatype
class StartProtectedSection:
    """Start of a `ProtectedSection`"""

    ...  # Empty


@datatype
class EndProtectedSection:
    """End of a `ProtectedSection`"""

    ...  # Empty


@datatype
class ProtectedSection:
    """Protected Section"""

    entries: List["Entry"]  # Entry List


@datatype
class Library:
    """Library, as Generated by the Spice `.lib` Definition Card
    Includes a list of named `LibSectionDef`s which can be included by their string-name,
    as common for "corner" inclusions e.g. `.inc "mylib.sp" "tt"`"""

    name: Ident  # Library Name
    sections: List[LibSectionDef]  # Library Sections


@datatype
class UseLibSection:
    """Use a Library"""

    path: Path  # Library File Path
    section: Ident  # Section Name


@datatype
class End:
    """Empty class represents `.end` Statements"""

    ...  # Empty


@datatype
class Variation:
    """Single-Parameter Variation Declaration"""

    name: Ident  # Parameter Name
    dist: str  # Distribution Name/Type
    std: "Expr"  # Standard Deviation
    mean: Optional["Expr"] = None  # Mean value (optional, defaults based on dist)


@datatype
class StatisticsBlock:
    """Statistical Descriptions"""

    process: Optional[List[Variation]]
    mismatch: Optional[List[Variation]]


@datatype
class Unknown:
    """Unknown Netlist Statement. Stored as an un-parsed string."""

    txt: str


@datatype
class DialectChange:
    """Netlist Dialect Changes, e.g. `simulator lang=xyz`"""

    dialect: str


# Union of "flat" statements lacking (substantial) hierarchy
FlatStatement = Union[
    Instance,
    Primitive,
    ParamDecls,
    ModelDef,
    ModelVariant,
    ModelFamily,
    DialectChange,
    "FunctionDef",
    Unknown,
    Options,
    Include,
    AhdlInclude,
    UseLibSection,
    StatisticsBlock,
]

# Statements which indicate the beginning and end of hierarchical elements,
# and ultimately disappear into the structured AST
DelimStatement = Union[
    StartLib,
    EndLib,
    StartLibSection,
    EndLibSection,
    End,
]

# Statements
# The union of types which can appear in first-pass parsing netlist
Statement = Union[FlatStatement, DelimStatement]

# Entries - the union of types which serve as "high-level" AST nodes,
# i.e. those which can be the direct children of a `SourceFile`.
Entry = Union[FlatStatement, SubcktDef, Library, LibSectionDef, End]


@datatype
class SourceFile:
    path: Path  # Source File Path
    contents: List[Entry]  # Statements and their associated SourceInfo


@datatype
class Program:
    """
    # Multi-File "Netlist Program"
    The name of this type is a bit misleading, but borrowed from more typical compiler-parsers.
    Spice-culture generally lacks a term for "the totality of a simulator invocation input",
    or even "a pile of source-files to be used together".
    So, `Program` it is.
    """

    files: List[SourceFile]  # List of Source-File Contents


@datatype
class Int:
    """Integer Number"""

    val: int


@datatype
class Float:
    """Floating Point Number"""

    val: float


@datatype
class MetricNum:
    """Number with Metric Suffix"""

    val: str  # No conversion, just stored as string for now


class ArgType(Enum):
    """Enumerated Supported Types for Function Arguments and Return Values"""

    REAL = "REAL"
    UNKNOWN = "UNKNOWN"

    def __repr__(self):
        return f"ArgType.{self.name}"


@datatype
class TypedArg:
    """Typed Function Argument"""

    tp: ArgType  # Argument Type
    name: Ident  # Argument Name


@datatype
class Return:
    """Function Return Node"""

    val: "Expr"


# Types which can be used inside a function definition.
# Will of course grow, in time.
FuncStatement = Union[Return]


@datatype
class FunctionDef:
    """Function Definition"""

    name: Ident  # Function Name
    rtype: ArgType  # Return Type
    args: List[TypedArg]  # Argument List
    stmts: List[FuncStatement]  # Function Body/ Statements


@datatype
class Call:
    """
    Function Call Node

    All valid parameter-generating function calls return a single value,
    usable in a mathematical expression (`Expr`) context.
    All arguments are provided by position and stored in a List.
    All arguments must also be resolvable as mathematical expressions.

    Examples:
    `sqrt(2)` => Call(func=Ident("sqrt"), args=([Int(2)]),)
    """

    func: Ref[FunctionDef]  # Function Name
    args: List["Expr"]  # Arguments List


class UnaryOperator(Enum):
    """Enumerated, Supported Unary Operators
    Values generally equal their string-format equivalents."""

    PLUS = "+"
    NEG = "-"

    def __repr__(self):
        return f"UnaryOperator.{self.name}"


@datatype
class UnaryOp:
    """Unary Operation"""

    tp: UnaryOperator  # Operator Type
    targ: "Expr"  # Target Expression


class BinaryOperator(Enum):
    """Enumerated, Supported Binary Operators
    Values generally equal their string-format equivalents."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"  # Note there is some divergence between caret and double-star here.
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="

    def __repr__(self):
        return f"BinaryOperator.{self.name}"


@datatype
class BinaryOp:
    """Binary Operation"""

    tp: BinaryOperator  # Enumerated Operator Type
    left: "Expr"  # Left Operand Expression
    right: "Expr"  # Right Operand Expression


@datatype
class TernOp:
    """Ternary Operation"""

    cond: "Expr"  # Condition Expression
    if_true: "Expr"  # Value if `cond` is True
    if_false: "Expr"  # Value if `cond` is False


# Expression Union
# Everything which can be used as a mathematical expression,
# and ultimately resolves to a scalar value at runtime.
Expr = Union[UnaryOp, BinaryOp, TernOp, Int, Float, MetricNum, Ref, Call]


# Update all the forward type-references
for tp in datatypes:
    # Pydantic v1: pydantic dataclasses need to update forward refs via __pydantic_model__
    if hasattr(tp, '__pydantic_model__'):
        tp.__pydantic_model__.update_forward_refs()
    # BaseModel classes use update_forward_refs() directly
    elif issubclass(tp, BaseModel):
        tp.update_forward_refs()

# Also update Ref which is a BaseModel but not in datatypes
Ref.update_forward_refs()

# And solely export the defined datatypes
# (at least with star-imports, which are hard to avoid using with all these types)
__all__ = [tp.__name__ for tp in datatypes] + [
    "NetlistDialects",
    "NetlistParseError",
    "BinaryOperator",
    "UnaryOperator",
    "Expr",
    "Entry",
    "Ref",
    "RefType",
    "ExternalRef",
    "OptionVal",
    "ArgType",
    "Model",
    "Statement",
    "to_json",
    "write_json",
]
