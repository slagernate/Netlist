"""
# Netlist "Compilation" 

Parse and convert a netlist-program into flattened, simulatable form. 
Occurs in three primary steps:

1. Parsing produces an AST `Program`
2. Concretizing converts this to a nested set of definition `Scope`s 
3. TODO: Elaboration flattens this into a single list of de-parameterized `Primitive` instances
"""
import os
from .data import Program, SourceFile, StatisticsBlock, Variation, ParamDecls, ParamDecl, Float, BinaryOp, BinaryOperator, Ref, Ident, Call, Int
from typing import Union, Sequence, Optional

# Local Imports
from .data import Program
from .ast_to_cst import Scope, ast_to_cst
from .parse import ParseOptions, parse_files, parse_str


def compile(src: Union[str, os.PathLike, Sequence[os.PathLike]], *, options: Optional[ParseOptions] = None) -> Scope:
    ast_program: Program = parse_files(src, options=options) if isinstance(src, list) else parse_str(src, options)

    # Post-parsing: Apply statistics variations
    apply_statistics_variations(ast_program, enable_monte_carlo=True, enable_process_corners=True)

    return ast_to_cst(ast_program)

def apply_statistics_variations(program: Program, enable_monte_carlo: bool = True, enable_process_corners: bool = True):
    """Apply statistics variations to matching parameters."""
    
    # Collect all statistics blocks
    stats_blocks = []
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, StatisticsBlock):
                stats_blocks.append(entry)
    
    if not stats_blocks:
        return  # No statistics to apply
    
    # Check for lognormal distributions to add math functions
    has_lognorm = any(var.dist and 'lnorm' in var.dist.lower() for stats in stats_blocks for section in (stats.process, stats.mismatch) if section for var in section)
    if has_lognorm:
        # Add math functions directly to the first file (e.g., models.scs)
        if program.files:
            math_funcs = [
                FunctionDef(
                    name=Ident("lnorm"),
                    rtype=ArgType.REAL,
                    args=[TypedArg(tp=ArgType.REAL, name=Ident("mu")), TypedArg(tp=ArgType.REAL, name=Ident("sigma")), TypedArg(tp=ArgType.REAL, name=Ident("seed"))],
                    stmts=[Return(val=Call(func=Ref(ident=Ident("exp")), args=[Call(func=Ref(ident=Ident("gauss")), args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma")), Ref(ident=Ident("seed"))])]))]
                ),
                FunctionDef(
                    name=Ident("alnorm"),
                    rtype=ArgType.REAL,
                    args=[TypedArg(tp=ArgType.REAL, name=Ident("mu")), TypedArg(tp=ArgType.REAL, name=Ident("sigma")), TypedArg(tp=ArgType.REAL, name=Ident("seed"))],
                    stmts=[Return(val=Call(func=Ref(ident=Ident("exp")), args=[Call(func=Ref(ident=Ident("agauss")), args=[Ref(ident=Ident("mu")), Ref(ident=Ident("sigma")), Ref(ident=Ident("seed"))])]))]
                )
            ]
            # Insert into the first file's contents (after any existing entries)
            program.files[0].contents.extend(math_funcs)
    
    # For each statistics block, apply variations
    for stats in stats_blocks:
        for section in (stats.process, stats.mismatch):
            if section:
                for var in section:
                    # Find matching parameter in the program
                    matching_param = find_matching_param(program, var.name.name)
                    if matching_param:
                        # Apply variations
                        original_expr = matching_param.default or Int(0)
                        new_expr = original_expr

                        # Monte Carlo: Multiply by variation (assuming Gaussian)
                        if enable_monte_carlo and var.dist and 'gauss' in var.dist.lower():
                            mc_var = BinaryOp(tp=BinaryOperator.ADD, left=original_expr,
                                              right=BinaryOp(tp=BinaryOperator.MUL, left=Float(var.std), right=Call(func=Ref(ident=Ident("gauss")), args=[Int(0), Float(1), Int(1)])))
                            new_expr = BinaryOp(tp=BinaryOperator.MUL, left=new_expr, right=mc_var)

                        # Process Corners: Add variation (or multiply, depending on design)
                        if enable_process_corners and var.mean:
                            pc_var = BinaryOp(tp=BinaryOperator.ADD, left=new_expr, right=var.mean)
                            new_expr = pc_var  # Or multiply if needed

                        # Update the parameter
                        matching_param.default = new_expr

def find_matching_param(program: Program, param_name: str) -> Optional[ParamDecl]:
    """Find a parameter by name in the program."""
    for file in program.files:
        for entry in file.contents:
            if isinstance(entry, ParamDecls):
                for param in entry.params:
                    if param.name.name == param_name:
                        return param
    return None
