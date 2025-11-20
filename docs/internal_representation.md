# Internal Representation of SPICE Dialects

This document describes the internal representation (IR) data structures created by this repository when parsing SPICE format netlists. The IR serves as a dialect-agnostic intermediate format that can represent netlists from various SPICE dialects (SPICE, SPECTRE, HSPICE, NGSPICE, etc.) in a unified structure.

## Table of Contents

1. [Top-Level Structure](#top-level-structure)
2. [Core Data Types](#core-data-types)
3. [Entry Types](#entry-types)
4. [Expression System](#expression-system)
5. [Reference System](#reference-system)
6. [Parameter System](#parameter-system)
7. [Model System](#model-system)
8. [Instance System](#instance-system)
9. [Subcircuit System](#subcircuit-system)
10. [Library System](#library-system)
11. [Source Information](#source-information)
12. [Dialect Handling](#dialect-handling)
13. [Examples](#examples)

## Top-Level Structure

The internal representation follows a hierarchical structure:

```
Program
└── List[SourceFile]
    └── List[Entry]
```

### Program

The `Program` class (defined in `netlist/data.py`) is the top-level container representing a complete netlist program. A "program" in this context refers to the totality of input files for a simulator invocation, which may span multiple source files.

```python
@dataclass
class Program:
    files: List[SourceFile]  # List of Source-File Contents
```

### SourceFile

A `SourceFile` represents a single parsed netlist file, containing its file path and a list of entries.

```python
@dataclass
class SourceFile:
    path: Path  # Source File Path
    contents: List[Entry]  # Statements and their associated SourceInfo
```

### Entry

`Entry` is a union type representing all high-level AST nodes that can appear as direct children of a `SourceFile`. The `Entry` union includes:

```python
Entry = Union[
    FlatStatement,      # Most statement types
    SubcktDef,          # Subcircuit definitions
    Library,            # Library definitions
    LibSectionDef,      # Library section definitions
    End                 # End statements
]
```

## Core Data Types

### Identifiers

The `Ident` class represents identifiers (names) used throughout the netlist:

```python
@dataclass
class Ident:
    name: str
```

Identifiers are used for:
- Node/net names
- Instance names
- Model names
- Parameter names
- Subcircuit names
- Function names

### SourceInfo

Every datatype in the IR (marked with the `@datatype` decorator) includes optional source information for tracking the origin of parsed elements:

```python
@dataclass
class SourceInfo:
    line: int                    # Source-File Line Number
    dialect: NetlistDialects     # Netlist Dialect
```

The `@datatype` decorator automatically adds a `source_info: Optional[SourceInfo]` field to each datatype class.

## Entry Types

### FlatStatement

`FlatStatement` is a union of statement types that don't have substantial hierarchy:

```python
FlatStatement = Union[
    Instance,           # Subcircuit/module instances
    Primitive,          # Primitive element instances
    ParamDecls,         # Parameter declarations
    ModelDef,           # Model definitions
    ModelVariant,       # Model variants
    ModelFamily,        # Model families
    DialectChange,      # Dialect change statements
    FunctionDef,        # Function definitions
    Unknown,            # Unparsed/unknown statements
    Options,            # Simulation options
    Include,            # File includes
    AhdlInclude,        # Analog HDL includes
    UseLibSection,      # Library section usage
    StatisticsBlock,    # Statistical descriptions
]
```

### DelimStatement

`DelimStatement` types indicate the beginning and end of hierarchical elements:

```python
DelimStatement = Union[
    StartLib,           # Start of a library
    EndLib,             # End of a library
    StartLibSection,    # Start of a library section
    EndLibSection,       # End of a library section
    End,                 # End statements
]
```

These delimiter statements are used during parsing to build hierarchical structures and are ultimately converted into structured AST nodes.

## Expression System

The `Expr` type represents mathematical expressions that can be evaluated to scalar values. The expression system supports:

### Expression Types

```python
Expr = Union[
    UnaryOp,      # Unary operations (+x, -x)
    BinaryOp,     # Binary operations (x+y, x*y, etc.)
    TernOp,       # Ternary operations (condition ? if_true : if_false)
    Int,          # Integer literals
    Float,        # Floating-point literals
    MetricNum,    # Numbers with metric suffixes (e.g., "1.5K", "2.3MEG")
    Ref,          # References to parameters or other values
    Call          # Function calls
]
```

### Numeric Literals

```python
@dataclass
class Int:
    val: int

@dataclass
class Float:
    val: float

@dataclass
class MetricNum:
    val: str  # Stored as string, e.g., "1.5K", "2.3MEG"
```

### Unary Operations

```python
class UnaryOperator(Enum):
    PLUS = "+"
    NEG = "-"

@dataclass
class UnaryOp:
    tp: UnaryOperator    # Operator type
    targ: Expr           # Target expression
```

### Binary Operations

```python
class BinaryOperator(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"    # Power operator (also **)
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="

@dataclass
class BinaryOp:
    tp: BinaryOperator   # Operator type
    left: Expr           # Left operand
    right: Expr          # Right operand
```

### Ternary Operations

```python
@dataclass
class TernOp:
    cond: Expr       # Condition expression
    if_true: Expr    # Value if condition is True
    if_false: Expr   # Value if condition is False
```

### Function Calls

```python
@dataclass
class Call:
    func: Ref[FunctionDef]  # Function reference
    args: List[Expr]        # Argument expressions
```

## Reference System

### Ref

The `Ref` class represents references to other netlist objects. Initially, a `Ref` contains only an identifier; it may later be resolved to point to the actual referent object.

```python
class Ref(BaseModel, Generic[Referent]):
    ident: Ident                                    # Referred-to identifier
    resolved: Optional[Union[Referent, ExternalRef]] = None  # Resolved referent
```

`Ref` is generic over the type of object it refers to. For example:
- `Ref[SubcktDef]` - reference to a subcircuit definition
- `Ref[Model]` - reference to a model
- `Ref[FunctionDef]` - reference to a function definition

### ExternalRef

An `ExternalRef` represents a reference to something defined outside the current netlist program:

```python
@dataclass
class ExternalRef:
    ident: Ident                    # Referred-to identifier
    types: List[RefType] = []       # Valid referent types
```

### RefType

`RefType` enumerates the types of external references:

```python
class RefType(Enum):
    SUBCKT = "SUBCKT"      # Sub-circuit definition
    MODEL = "MODEL"         # Model definition
    PARAM = "PARAM"         # Parameter reference
    FUNCTION = "FUNCTION"   # Function definition
    FILEPATH = "FILEPATH"   # File-system path
```

## Parameter System

### ParamDecl

A `ParamDecl` represents a parameter declaration with an optional default value and optional distribution information:

```python
@dataclass
class ParamDecl:
    name: Ident                    # Parameter name
    default: Optional[Expr]        # Default value expression
    distr: Optional[str] = None    # Distribution information
```

### ParamVal

A `ParamVal` represents a parameter value assignment:

```python
@dataclass
class ParamVal:
    name: Ident    # Parameter name
    val: Expr      # Parameter value expression
```

### ParamDecls

`ParamDecls` groups multiple parameter declarations together:

```python
@dataclass
class ParamDecls:
    params: List[ParamDecl]
```

## Model System

The model system supports both single models and model families (binned models).

### ModelDef

A `ModelDef` represents a single device model definition:

```python
@dataclass
class ModelDef:
    name: Ident                    # Model name
    mtype: Ident                   # Model type (e.g., "nmos", "pmos")
    args: List[Ident]              # Positional arguments
    params: List[ParamDecl]        # Parameter declarations & defaults
```

### ModelVariant

A `ModelVariant` represents a single variant within a model family:

```python
@dataclass
class ModelVariant:
    model: Ident                   # Model family name
    variant: Ident                 # Variant name
    mtype: Ident                   # Model type
    args: List[Ident]              # Positional arguments
    params: List[ParamDecl]        # Parameter declarations & defaults
```

### ModelFamily

A `ModelFamily` groups related model variants, typically separated by limiting parameters such as `{lmin, lmax}` or `{wmin, wmax}`:

```python
@dataclass
class ModelFamily:
    name: Ident                    # Model family name
    mtype: Ident                   # Model type
    variants: List[ModelVariant]   # List of variants
```

### Model Union

The `Model` type is a union of single and multi-variant models:

```python
Model = Union[ModelDef, ModelFamily]
```

## Instance System

The instance system distinguishes between subcircuit instances and primitive element instances.

### Instance

An `Instance` represents an instance of a subcircuit or module:

```python
@dataclass
class Instance:
    name: Ident                                    # Instance name
    module: Ref[Union[SubcktDef, Model]]          # Module/subcircuit reference
    conns: Union[List[Ident], List[Tuple[Ident, Ident]]]  # Connections (by position or by name)
    params: List[ParamVal]                        # Parameter values
```

Connections can be:
- **By position**: `List[Ident]` - ordered list of net names
- **By name**: `List[Tuple[Ident, Ident]]` - list of (port_name, net_name) pairs

### Primitive

A `Primitive` represents an instance of a primitive circuit element (resistor, capacitor, transistor, etc.):

```python
@dataclass
class Primitive:
    name: Ident           # Instance name
    args: List[Expr]      # Positional arguments (ports and values)
    kwargs: List[ParamVal] # Keyword arguments (parameters)
```

**Note**: At parsing time, it's not always clear which arguments are ports, model names, or parameter values. Primitives store all positional arguments as `Expr` and sort them out later during resolution.

## Subcircuit System

### SubcktDef

A `SubcktDef` represents a complete subcircuit definition:

```python
@dataclass
class SubcktDef:
    name: Ident           # Module/subcircuit name
    ports: List[Ident]   # Port list
    params: List[ParamDecl]  # Parameter declarations
    entries: List[Entry] # Subcircuit contents
```

The `entries` field contains all statements within the subcircuit, including:
- Instances of other subcircuits
- Primitive instances
- Nested subcircuit definitions
- Parameter declarations
- Model definitions

### StartSubckt / EndSubckt

During parsing, subcircuits are initially represented as delimiter statements:

```python
@dataclass
class StartSubckt:
    name: Ident
    ports: List[Ident]
    params: List[ParamDecl]

@dataclass
class EndSubckt:
    name: Optional[Ident]
```

These are converted to `SubcktDef` during the hierarchicalization phase.

## Library System

Libraries provide a mechanism for organizing netlist content into named sections, commonly used for process corners (e.g., "tt", "ff", "ss").

### Library

A `Library` represents a complete library definition:

```python
@dataclass
class Library:
    name: Ident                    # Library name
    sections: List[LibSectionDef]  # Library sections
```

### LibSectionDef

A `LibSectionDef` represents a named section within a library:

```python
@dataclass
class LibSectionDef:
    name: Ident           # Section name
    entries: List[Entry]  # Section contents
```

### UseLibSection

A `UseLibSection` represents the inclusion of a specific section from a library file:

```python
@dataclass
class UseLibSection:
    path: Path    # Library file path
    section: Ident  # Section name
```

### Library Delimiters

During parsing, libraries use delimiter statements:

```python
@dataclass
class StartLib:
    name: Ident

@dataclass
class EndLib:
    name: Optional[Ident]

@dataclass
class StartLibSection:
    name: Ident

@dataclass
class EndLibSection:
    name: Ident
```

## Function System

### FunctionDef

A `FunctionDef` represents a user-defined function:

```python
@dataclass
class FunctionDef:
    name: Ident                    # Function name
    rtype: ArgType                 # Return type
    args: List[TypedArg]           # Argument list
    stmts: List[FuncStatement]     # Function body/statements
```

### TypedArg

A `TypedArg` represents a typed function argument:

```python
@dataclass
class TypedArg:
    tp: ArgType    # Argument type
    name: Ident    # Argument name
```

### ArgType

`ArgType` enumerates supported function argument and return types:

```python
class ArgType(Enum):
    REAL = "REAL"
    UNKNOWN = "UNKNOWN"
```

### FuncStatement

Currently, the only supported function statement type is `Return`:

```python
FuncStatement = Union[Return]

@dataclass
class Return:
    val: Expr  # Return value expression
```

## Statistics System

### StatisticsBlock

A `StatisticsBlock` represents statistical descriptions for process and mismatch variations:

```python
@dataclass
class StatisticsBlock:
    process: Optional[List[Variation]]   # Process variations
    mismatch: Optional[List[Variation]]  # Mismatch variations
```

### Variation

A `Variation` represents a single-parameter variation declaration:

```python
@dataclass
class Variation:
    name: Ident           # Parameter name
    dist: str             # Distribution name/type (e.g., "gauss", "lnorm")
    std: Expr             # Standard deviation
    mean: Optional[Expr] = None  # Mean value (optional)
```

## Options System

### Option

An `Option` represents a single simulation option:

```python
@dataclass
class Option:
    name: Ident        # Option name
    val: OptionVal     # Option value
```

### OptionVal

Option values can be either expressions or quoted strings:

```python
OptionVal = Union[QuotedString, Expr]

@dataclass
class QuotedString:
    txt: str
```

### Options

`Options` groups multiple options together:

```python
@dataclass
class Options:
    name: Optional[Ident]  # Option name (may be None)
    vals: List[Option]     # List of options
```

## Source Information

### SourceInfo

Every datatype marked with `@datatype` includes optional source information:

```python
@dataclass
class SourceInfo:
    line: int                    # Source file line number
    dialect: NetlistDialects     # Netlist dialect at parse time
```

The `source_info` field is automatically added by the `@datatype` decorator and defaults to `None`. It's populated during parsing to track where each element originated.

### NetlistDialects

Supported netlist dialects are enumerated:

```python
class NetlistDialects(Enum):
    SPECTRE = "spectre"
    SPECTRE_SPICE = "spectre_spice"
    SPICE = "spice"
    HSPICE = "hspice"
    NGSPICE = "ngspice"
    XYCE = "xyce"
    CDL = "cdl"
```

## Dialect Handling

### Dialect-Agnostic IR

The internal representation is designed to be dialect-agnostic. Different SPICE dialects (SPICE, SPECTRE, HSPICE, etc.) are parsed into the same IR structures, allowing:

- Conversion between dialects
- Dialect-agnostic analysis and manipulation
- Unified tooling across different netlist formats

### DialectChange

Some dialects (notably Spectre-Spice) support runtime dialect switching:

```python
@dataclass
class DialectChange:
    dialect: str  # Target dialect name (e.g., "spectre", "spice")
```

When a `simulator lang=xyz` statement is encountered, a `DialectChange` is created and the parser switches to the specified dialect.

### Dialect-Specific Parsing

While the IR is dialect-agnostic, parsing is handled by dialect-specific parsers:

- `SpiceDialectParser` - Base SPICE dialect parser
- `NgSpiceDialectParser` - NGSPICE-specific parser
- `HspiceDialectParser` - HSPICE-specific parser
- `SpectreDialectParser` - Spectre language parser
- `SpectreSpiceDialectParser` - Spectre-Spice dialect parser

Each parser produces the same IR structures, but handles dialect-specific syntax variations during parsing.

## Examples

### Example 1: Simple Parameter Declaration

**SPICE Input:**
```spice
.param vdd=1.8 vss=0.0
```

**IR Representation:**
```python
ParamDecls(
    params=[
        ParamDecl(name=Ident("vdd"), default=Float(1.8), distr=None),
        ParamDecl(name=Ident("vss"), default=Float(0.0), distr=None)
    ],
    source_info=SourceInfo(line=1, dialect=NetlistDialects.SPICE)
)
```

### Example 2: Model Definition

**SPICE Input:**
```spice
.model nmos1 nmos (vth0=0.5 tox=2e-9)
```

**IR Representation:**
```python
ModelDef(
    name=Ident("nmos1"),
    mtype=Ident("nmos"),
    args=[],
    params=[
        ParamDecl(name=Ident("vth0"), default=Float(0.5), distr=None),
        ParamDecl(name=Ident("tox"), default=Float(2e-9), distr=None)
    ],
    source_info=SourceInfo(line=1, dialect=NetlistDialects.SPICE)
)
```

### Example 3: Subcircuit Definition

**SPICE Input:**
```spice
.subckt inverter in out vdd vss
m1 out in vss vss nmos1 w=1u l=0.1u
m2 out in vdd vdd pmos1 w=2u l=0.1u
.ends inverter
```

**IR Representation:**
```python
SubcktDef(
    name=Ident("inverter"),
    ports=[
        Ident("in"),
        Ident("out"),
        Ident("vdd"),
        Ident("vss")
    ],
    params=[],
    entries=[
        Primitive(
            name=Ident("m1"),
            args=[
                Ident("out"),
                Ident("in"),
                Ident("vss"),
                Ident("vss"),
                Ref(ident=Ident("nmos1"))
            ],
            kwargs=[
                ParamVal(name=Ident("w"), val=MetricNum("1u")),
                ParamVal(name=Ident("l"), val=MetricNum("0.1u"))
            ],
            source_info=SourceInfo(line=2, dialect=NetlistDialects.SPICE)
        ),
        Primitive(
            name=Ident("m2"),
            args=[
                Ident("out"),
                Ident("in"),
                Ident("vdd"),
                Ident("vdd"),
                Ref(ident=Ident("pmos1"))
            ],
            kwargs=[
                ParamVal(name=Ident("w"), val=MetricNum("2u")),
                ParamVal(name=Ident("l"), val=MetricNum("0.1u"))
            ],
            source_info=SourceInfo(line=3, dialect=NetlistDialects.SPICE)
        )
    ],
    source_info=SourceInfo(line=1, dialect=NetlistDialects.SPICE)
)
```

### Example 4: Expression with Function Call

**SPICE Input:**
```spice
.param width=1u length='width * 2'
.param area='width * length'
.param sqrt_area='sqrt(area)'
```

**IR Representation:**
```python
ParamDecls(
    params=[
        ParamDecl(
            name=Ident("width"),
            default=MetricNum("1u"),
            distr=None
        ),
        ParamDecl(
            name=Ident("length"),
            default=BinaryOp(
                tp=BinaryOperator.MUL,
                left=Ref(ident=Ident("width")),
                right=Int(2)
            ),
            distr=None
        ),
        ParamDecl(
            name=Ident("area"),
            default=BinaryOp(
                tp=BinaryOperator.MUL,
                left=Ref(ident=Ident("width")),
                right=Ref(ident=Ident("length"))
            ),
            distr=None
        ),
        ParamDecl(
            name=Ident("sqrt_area"),
            default=Call(
                func=Ref(ident=Ident("sqrt")),
                args=[Ref(ident=Ident("area"))]
            ),
            distr=None
        )
    ]
)
```

### Example 5: Model Family (Spectre Format)

**Spectre Input:**
```spectre
model nmos1 nmos {
    1: vth0=0.5 tox=2e-9
    2: vth0=0.6 tox=2e-9
}
```

**IR Representation:**
```python
ModelFamily(
    name=Ident("nmos1"),
    mtype=Ident("nmos"),
    variants=[
        ModelVariant(
            model=Ident("nmos1"),
            variant=Ident("1"),
            mtype=Ident("nmos"),
            args=[],
            params=[
                ParamDecl(name=Ident("vth0"), default=Float(0.5), distr=None),
                ParamDecl(name=Ident("tox"), default=Float(2e-9), distr=None)
            ]
        ),
        ModelVariant(
            model=Ident("nmos1"),
            variant=Ident("2"),
            mtype=Ident("nmos"),
            args=[],
            params=[
                ParamDecl(name=Ident("vth0"), default=Float(0.6), distr=None),
                ParamDecl(name=Ident("tox"), default=Float(2e-9), distr=None)
            ]
        )
    ]
)
```

### Example 6: Complete Program Structure

**SPICE Input:**
```spice
* Main circuit file
.param vdd=1.8
.include "models.cir"
.subckt mycircuit in out
x1 in out inverter
.ends
```

**IR Representation:**
```python
Program(
    files=[
        SourceFile(
            path=Path("main.cir"),
            contents=[
                ParamDecls(
                    params=[
                        ParamDecl(name=Ident("vdd"), default=Float(1.8), distr=None)
                    ]
                ),
                Include(path=Path("models.cir")),
                SubcktDef(
                    name=Ident("mycircuit"),
                    ports=[Ident("in"), Ident("out")],
                    params=[],
                    entries=[
                        Instance(
                            name=Ident("x1"),
                            module=Ref(ident=Ident("inverter")),
                            conns=[Ident("in"), Ident("out")],
                            params=[]
                        )
                    ]
                )
            ]
        )
    ]
)
```

## JSON Serialization

The IR can be serialized to JSON using the `to_json` function:

```python
from netlist.data import to_json

program = parse("circuit.cir")
json_str = to_json(program)
```

This produces a JSON representation of the entire IR structure, useful for:
- Debugging
- External tool integration
- Storage and transmission
- Analysis and transformation

## Notes and Limitations

### FIXME Items

Several FIXME comments in the codebase indicate areas for future improvement:

1. **HierPath**: The `path` field in `HierPath` is noted as potentially using `Ref` instead of `Ident`
2. **Options.name**: The `name` field in `Options` may be removable
3. **Unknown statements**: When parsing errors occur in `ErrorMode.STORE` mode, the line content is not included in `Unknown` statements
4. **Expression delimiters**: Some dialects use different expression delimiters (single quotes, curly braces, etc.)

### Unsupported Constructs

The IR focuses on netlist constructs used for device models and technology libraries. Common unsupported constructs include:
- Simulation analyses (`.ac`, `.dc`, `.tran`, etc.)
- Measurements and probes
- Control statements beyond basic options
- Some advanced dialect-specific features

### Resolution

Many references in the IR are initially unresolved (the `resolved` field in `Ref` is `None`). Resolution typically occurs in later compilation or analysis phases, not during initial parsing.

## Related Documentation

- **Parsing**: See `netlist/parse.py` for parsing logic
- **Dialect Parsers**: See `netlist/dialects/` for dialect-specific parsing
- **Writing**: See `netlist/write/` for IR-to-netlist conversion
- **AST to CST**: See `netlist/ast_to_cst.py` for scope-based resolution

