"""
Base Netlister Class and Error Modes
"""

import sys
from enum import Enum
from typing import IO, Union, Tuple, List, Optional
from warnings import warn
from datetime import datetime
from ..data import (
    Program,
    Expr,
    Ident,
    MetricNum,
    Int,
    Float,
    BinaryOp,
    UnaryOp,
    Call,
    Ref,
    TernOp,
    QuotedString,
    ModelDef,
    ModelFamily,
    ModelVariant,
    FunctionDef,
    Comment,
)

class ErrorMode(Enum):
    """Error Handling Modes"""

    RAISE = "raise"  # Raise an exception on error
    WARN = "warn"  # Print a warning and continue
    IGNORE = "ignore"  # Ignore errors and continue
    STORE = "store"  # Store errors in a list


class Netlister:
    """
    # Abstract Base `Netlister` Class

    The `Netlister` class is the abstract base for all netlisters.
    It provides the core interface for writing netlists,
    including methods for writing generic tokens, identifiers, numbers, and expressions.
    """

    def __init__(
        self,
        src: Program,
        dest: IO,
        *,
        errormode: ErrorMode = ErrorMode.RAISE,
        file_type: str = "",
    ) -> None:
        self.src = src
        self.dest = dest
        self.errormode = errormode
        self.indent = "  "
        self.file_type = file_type  # "models" or "library" (or empty for default)
        self.errors = []
        self._warnings = []  # List of (message, context) tuples for log file

    @property
    def enum(self):
        """Get our entry in the `NetlistFormat` enumeration"""
        raise NotImplementedError

    def netlist(self) -> None:
        """Netlist the `Program` to the destination stream"""
        # Validate content based on file_type
        if self.file_type:
            self._validate_content()
            
        for file in self.src.files:
            entries = file.contents
            i = 0
            while i < len(entries):
                entry = entries[i]
                
                # Special handling for ParamDecls: collect all "after" comments that follow
                # and write them inline with the parameters
                from ..data import ParamDecls, Comment
                if isinstance(entry, ParamDecls):
                    # Collect consecutive "after" comments that follow this ParamDecls
                    # Skip comments that are already stored inline in ParamDecl.comment to avoid duplicates
                    inline_comments = []
                    j = i + 1
                    while j < len(entries):
                        next_entry = entries[j]
                        if isinstance(next_entry, Comment) and next_entry.position == "after":
                            # Skip if this comment text matches any inline comment already stored in params
                            comment_text = next_entry.text
                            is_duplicate = any(param.comment == comment_text for param in entry.params)
                            
                            if not is_duplicate:
                                inline_comments.append(next_entry)
                            j += 1
                        else:
                            break
                    
                    # Write ParamDecls with inline comments
                    if inline_comments:
                        self.write_param_decls_with_inline_comments(entry, inline_comments)
                        i = j  # Skip ParamDecls and all inline comments
                        continue
                    else:
                        # No additional inline comments, but params might have comments stored
                        # Write normally (write_param_decls will handle inline comments from ParamDecl.comment)
                        self.write_entry(entry)
                        i += 1
                        continue
                
                # Skip comments that match inline comments already stored in ParamDecl.comment fields
                # This prevents duplicate comments from appearing both inline and as standalone entries
                if isinstance(entry, Comment):
                    comment_text = entry.text
                    # Check all ParamDecls entries to see if any param has this as an inline comment
                    is_duplicate = any(
                        param.comment == comment_text
                        for check_entry in entries
                        if isinstance(check_entry, ParamDecls)
                        for param in check_entry.params
                    )
                    
                    if is_duplicate:
                        # Skip this duplicate comment (it's already written inline with its parameter)
                        i += 1
                        continue
                
                # Check if next entry is a comment that should be inline
                if i + 1 < len(entries):
                    next_entry = entries[i + 1]
                    if isinstance(next_entry, Comment) and next_entry.position == "after":
                        # Check if it's on the same line (inline comment)
                        if (hasattr(entry, 'source_info') and hasattr(next_entry, 'source_info') and
                            entry.source_info and next_entry.source_info and
                            entry.source_info.line == next_entry.source_info.line):
                            # Write entry with inline comment
                            self.write_entry_with_inline_comment(entry, next_entry)
                            i += 2  # Skip both entry and comment
                            continue
                
                # Normal entry writing
                self.write_entry(entry)
                i += 1
        self.dest.flush()
    
    def write_entry_with_inline_comment(self, entry, comment: "Comment") -> None:
        """Write an entry with an inline comment. Default implementation writes entry then comment.
        Subclasses can override to append comment inline."""
        self.write_entry(entry)
        # For now, write comment on separate line - subclasses can override for true inline
        self.write_comment(comment.text)
    
    def write_param_decls_with_inline_comments(self, params, inline_comments) -> None:
        """Write parameter declarations with inline comments.
        Default implementation just writes params normally and ignores inline comments.
        Subclasses should override to handle inline comments properly."""
        self.write_entry(params)
        # Write any remaining comments as separate entries
        from ..data import Comment
        for comment in inline_comments:
            if isinstance(comment, Comment):
                self.write_comment(comment.text)

    def _validate_content(self) -> None:
        """Validate that the program content matches the expected file_type.
        Can be overridden by subclasses to implement specific validation logic.
        """
        pass

    def write_entry(self, entry) -> None:
        """Write a general `Entry`. Dispatches to type-specific methods."""
        from ..data import (
            SubcktDef,
            ModelDef,
            Instance,
            Primitive,
            ParamDecls,
            Options,
            StatisticsBlock,
            Include,
            LibSectionDef,
            UseLibSection,
            ModelFamily,
            ModelVariant,
            FunctionDef,
            StartProtectedSection,
            EndProtectedSection,
            Comment,
            BlankLine,
        )

        if isinstance(entry, BlankLine):
            return self.write_blank_line()
        if isinstance(entry, Comment):
            return self.write_comment_entry(entry)
        if isinstance(entry, SubcktDef):
            return self.write_subckt_def(entry)
        if isinstance(entry, ModelDef):
            return self.write_model_def(entry)
        if isinstance(entry, Instance):
            return self.write_subckt_instance(entry)
        if isinstance(entry, Primitive):
            return self.write_primitive_instance(entry)
        if isinstance(entry, ParamDecls):
            return self.write_param_decls(entry)
        if isinstance(entry, Options):
            return self.write_options(entry)
        if isinstance(entry, StatisticsBlock):
            return self.write_statistics_block(entry)
        if isinstance(entry, Include):
            return self.write_include(entry)
        if isinstance(entry, LibSectionDef):
            return self.write_library_section(entry)
        if isinstance(entry, UseLibSection):
            return self.write_use_lib(entry)
        if isinstance(entry, ModelFamily):
            return self.write_model_family(entry)
        if isinstance(entry, ModelVariant):
            return self.write_model_variant(entry)
        if isinstance(entry, FunctionDef):
            return self.write_function_def(entry)
        if isinstance(entry, (StartProtectedSection, EndProtectedSection)):
            # Skip protected section markers in output
            return

        self.handle_error(entry, f"Unknown Entry Type {entry}")
    
    def write_blank_line(self) -> None:
        """Write a blank line to preserve formatting."""
        self.write("\n")
    
    def write_comment_entry(self, comment: "Comment") -> None:
        """Write a standalone comment entry"""
        if comment.position in ("standalone", "before", "after"):
            self.write_comment(comment.text)
    
    def write_comment(self, comment: str) -> None:
        """Write a comment. Default implementation uses * for line-starting comments.
        Subclasses should override to use appropriate comment syntax."""
        self.write(f"* {comment}\n")

    def write(self, s: str) -> None:
        """Write string `s` to the destination stream"""
        self.dest.write(s)

    def writeln(self, s: str) -> None:
        """Write string `s` plus a newline to the destination stream"""
        self.write(s + "\n")

    def handle_error(self, obj: object, msg: str) -> None:
        """Handle an error, based on the `ErrorMode`"""
        if self.errormode == ErrorMode.RAISE:
            raise RuntimeError(f"{msg} in {obj}")
        if self.errormode == ErrorMode.WARN:
            print(f"Warning: {msg} in {obj}")
        if self.errormode == ErrorMode.STORE:
            self.errors.append((obj, msg))
    
    def log_warning(self, message: str, context: Optional[str] = None) -> None:
        """Log a warning message for inclusion in the translation log file.
        
        Args:
            message: The warning message
            context: Optional context information (e.g., model name, parameter name)
        """
        self._warnings.append((message, context))
        # Also emit standard warning for backward compatibility
        if context:
            warn(f"{message} (Context: {context})")
        else:
            warn(message)
    
    def get_warnings(self) -> List[Tuple[str, Optional[str]]]:
        """Get all collected warnings.
        
        Returns:
            List of (message, context) tuples
        """
        return self._warnings.copy()

    def format_ident(self, ident: Union[Ident, Ref]) -> str:
        """Format an identifier"""
        if isinstance(ident, Ref):
            return ident.ident.name
        return ident.name

    def format_number(self, num: Union[Int, Float, MetricNum]) -> str:
        """Format a number"""
        if isinstance(num, MetricNum):
            return str(num.val)
        return str(num.val)

    def expression_delimiters(self) -> Tuple[str, str]:
        """Return the starting and closing delimiters for expressions.
        By default, SPICE-class languages use single quotes.
        Sub-classes may override this, e.g. Xyce uses curly braces."""
        return ("'", "'")

    def format_expr(self, expr: Expr) -> str:
        """Format an expression"""
        
        # Base cases: Literals, Refs, QuotedStrings
        if isinstance(expr, (Int, Float, MetricNum)):
            return self.format_number(expr)
        if isinstance(expr, Ref):
            return self.format_ident(expr)
        if isinstance(expr, QuotedString):
            return f"'{expr.val}'"

        # Compound expressions requiring recursion
        start, end = self.expression_delimiters()
        inner = self._format_expr_inner(expr)
        return f"{start}{inner}{end}"

    def _format_expr_inner(self, expr: Expr) -> str:
        """Format the *inner* content of an expression, without delimiters.
        This is the recursive part of `format_expr`."""
        
        if isinstance(expr, (Int, Float, MetricNum)):
            return self.format_number(expr)
        if isinstance(expr, Ref):
            return self.format_ident(expr)
        
        if isinstance(expr, BinaryOp):
            left = self._format_expr_inner(expr.left)
            right = self._format_expr_inner(expr.right)
            op = expr.tp.value
            
            # Check if we need parentheses for precedence
            # This is a simple heuristic: if the sub-expressions are also binary ops, wrap them.
            # A more sophisticated approach would check operator precedence.
            if isinstance(expr.left, BinaryOp):
                left = f"({left})"
            if isinstance(expr.right, BinaryOp):
                right = f"({right})"
                
            return f"{left}{op}{right}"
            
        if isinstance(expr, UnaryOp):
            targ = self._format_expr_inner(expr.targ)
            return f"{expr.tp.value}({targ})"
            
        if isinstance(expr, TernOp):
            cond = self._format_expr_inner(expr.cond)
            if_true = self._format_expr_inner(expr.if_true)
            if_false = self._format_expr_inner(expr.if_false)
            return f"({cond} ? {if_true} : {if_false})"
            
        if isinstance(expr, Call):
            func = self.format_ident(expr.func)
            args = [self._format_expr_inner(arg) for arg in expr.args]
            return f"{func}({','.join(args)})"

        self.handle_error(expr, f"Unknown Expression Type {expr}")
        return ""

    def write_subckt_def(self, module) -> None:
        raise NotImplementedError

    def write_model_def(self, model) -> None:
        raise NotImplementedError
    
    def write_model_family(self, mfam: ModelFamily) -> None:
        raise NotImplementedError
    
    def write_model_variant(self, mvar: ModelVariant) -> None:
        raise NotImplementedError

    def write_subckt_instance(self, pinst) -> None:
        raise NotImplementedError

    def write_primitive_instance(self, pinst) -> None:
        raise NotImplementedError

    def write_param_decls(self, params) -> None:
        raise NotImplementedError

    def write_options(self, options) -> None:
        raise NotImplementedError

    def write_statistics_block(self, stats) -> None:
        raise NotImplementedError

    def write_include(self, inc) -> None:
        raise NotImplementedError

    def write_library_section(self, section) -> None:
        raise NotImplementedError

    def write_use_lib(self, uselib) -> None:
        raise NotImplementedError
    
    def write_function_def(self, func: FunctionDef) -> None:
        raise NotImplementedError
