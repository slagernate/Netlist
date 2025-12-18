""" 
# Netlist Lexing
"""

# Std-Lib Imports
import re
from typing import Iterable, Optional, List, Tuple

# PyPi Imports
from pydantic.dataclasses import dataclass


# Numeric-value suffixes
suffixes = dict(
    T=1.0e12,
    G=1.0e9,
    MEG=1.0e6,
    X=1.0e6,
    K=1.0e3,
    M=1.0e-3,
    MIL=2.54e-5,  # (1/1000 inch)
    U=1.0e-6,
    N=1.0e-9,
    P=1.0e-12,
    F=1.0e-15,
    A=1.0e-18,
)
# Include both upper and lower-case versions.
# (Python `re` complains about inserting this as an inline flag.)
suffix_pattern = "|".join(list(suffixes.keys()) + [k.lower() for k in suffixes.keys()])

# Pattern for a string identifier
# An initial alpha character, followed by any number of chars, numbers, and underscores.
# Note some Spice "identifiers" - particularly nets - are often numerically-valued.
# "Node zero" is a prominent example.
# Those are not covered here.
ident_pattern = r"[A-Za-z_][A-Za-z0-9_]*"

# Master mapping of tokens <=> patterns
_patterns1 = dict(
    DUBSLASH=r"\/\/",
    DUBSTAR=r"\*\*",
    LPAREN=r"\(",
    RPAREN=r"\)",
    LBRACKET=r"\{",
    RBRACKET=r"\}",
    NEWLINE=r"\n",
    WHITE=r"\s",
    SLASH=r"\/",
    BACKSLASH=r"\\",
    CARET=r"\^",
    STAR=r"\*",
    TICK=r"\'",
    COMMA=r"\,",
    SEMICOLON=r"\;",
    COLON=r"\:",
    GE=r"\>\=",
    LE=r"\<\=",
    DUBEQUALS=r"\=\=",  # Equality comparison operator
    GT=r"\>",
    LT=r"\<",
    EQUALS=r"\=",
    DOLLAR=r"\$",
    QUESTION=r"\?",
    DUBQUOTE=r"\"",  # Double-quote. Surrounds file-paths, and in some cases, expressions.
    MODEL_VARIANT=rf"{ident_pattern}\.(\d|{ident_pattern})+",  # nmos.0, mymodel.global, etc
    METRIC_NUM=rf"(\d+(\.\d+)?|\.\d+)({suffix_pattern})",  # 1M or 1.0f or .1k
    FLOAT=r"(\d+[eE][+-]?\d+|(\d+\.\d*|\.\d+)([eE][+-]?\d+)?)",  # 1e3 or 1.0 or .1 (optional e-3)
    INT=r"\d+",
    PLUS=r"\+",
    MINUS=r"\-",
    DOT=r"\.",
)
_keywords = dict(
    ENDLIBRARY=r"endlibrary",
    ENDSECTION=r"endsection",
    SECTION=r"section",
    AHDL=r"ahdl_include",
    INCLUDE=r"(include|INCLUDE)",
    INC=r"(inc|INC)",
    INLINE=r"inline",
    SUBCKT=r"(subckt|SUBCKT)",
    ENDS=r"(ends|ENDS)",
    LIBRARY=r"library",
    LIB=r"(lib|LIB)",
    ENDL=r"(endl|ENDL)",
    MODEL=r"(model|MODEL)",
    STATS=r"statistics",
    SIMULATOR=r"simulator",
    LANG=r"lang",
    PARAMETERS=r"(parameters|params)",
    PARAM=r"(param|PARAM)",
    OPTIONS=r"(options|OPTIONS)",
    OPTION=r"(option|OPTION)",
    REAL=r"real",
    RETURN=r"return",
    DEV_GAUSS=r"dev\/gauss",  # Perhaps there are more "dev/{x}" to be added; gauss is the known one for now.
    # Note the order of these `protect` tokens is important, as `prot` is a subset of all of them! It must come last and lowest priority.
    UNPROTECT=r"(unprotect|UNPROTECT)",
    UNPROT=r"(unprot|UNPROT)",
    PROTECT=r"(protect|PROTECT)",
    PROT=r"(prot|PROT)",
)
_patterns2 = dict(
    IDENT=ident_pattern,
    ERROR=r"[\s\S]",
)
# Given each token its name as a key in the overall regex
tokens = {key: rf"(?P<{key}>{val})" for key, val in _patterns1.items()}
for key, val in _keywords.items():
    # Insert \b word-boundaries around keywords
    tokens[key] = rf"(?P<{key}>\b{val}\b)"
for key, val in _patterns2.items():
    # Add the lower-priority patterns last
    tokens[key] = rf"(?P<{key}>{val})"
# Build our overall regex pattern, a union of all
pat = re.compile("|".join(tokens.values()))
# Create an enum-ish class of these token-types
Tokens = type("Tokens", (object,), {k: k for k in tokens.keys()})


@dataclass
class Token:
    """Lexer Token
    Includes type-annotation (as a string), and the token's text value."""

    tp: str  # Type Annotation. A value from `Tokens`, which should be enumerated, some day.
    val: str  # Text Content Value


class Lexer:
    """# Netlist Lexer"""

    def __init__(self, lines: Iterable[str]):
        self.parser = None
        self.lines = lines
        self.line = next(self.lines, None)
        self.line_num = 1
        self.lexed_nonwhite_on_this_line = False
        self.toks = iter(pat.scanner(self.line).match, None)
        self.recent_lines = []
        # Track comments found on the current line
        self.current_line_comments: List[Tuple[str, str]] = []  # (comment_text, comment_type)
        # Persistent queue of comments that haven't been retrieved yet
        self.comment_queue: List[Tuple[str, str, int]] = []  # (comment_text, comment_type, line_num)
        # Track line numbers which were *full-line* comments (comment began before any non-whitespace token).
        # Used to prevent these lines from also being treated as blank lines downstream.
        self.full_line_comment_lines: set[int] = set()
        # Track if current line ends with backslash (for continuation)
        self.current_line_ends_with_backslash = False

    def nxt(self) -> Optional[Token]:
        """Get our next Token, pulling a new line if necessary."""
        m = next(self.toks, None)
        if m is None:  # Grab a new line
            self.line = next(self.lines, None)
            if self.line is None:  # End of input
                return None
            self.line_num += 1

            # buffer last 5 lines for debugging
            self.recent_lines.append(self.line.rstrip('\n'))
            if len(self.recent_lines) > 5:
                self.recent_lines.pop(0)

            # Check if current line ends with backslash (for continuation)
            # Strip trailing whitespace and newline, then check last char
            line_stripped = self.line.rstrip('\n\r').rstrip()
            self.current_line_ends_with_backslash = line_stripped.endswith('\\')

            self.toks = iter(pat.scanner(self.line).match, None)
            m = next(self.toks, None)
            if m is None:
                return None
        return Token(m.lastgroup, m.group())

    def eat_idle(self, token) -> Optional[Token]:
        """Consume whitespace and comments, returning the next (potential) action-token.
        Does not handle line-continuations."""
        while token and token.tp == Tokens.WHITE:
            token = self.nxt()
        if token and self.parser.is_comment(token):
            # Determine comment type
            if token.tp == Tokens.DUBSLASH:
                comment_type = "//"
            elif token.tp == Tokens.DOLLAR:
                comment_type = "$"
            elif token.tp == Tokens.STAR:
                comment_type = "*"
            else:
                comment_type = "*"  # Default fallback
            
            # Collect comment text - start with the comment marker
            comment_text = token.val
            token = self.nxt()
            # Collect all tokens until newline
            while token and token.tp != Tokens.NEWLINE:
                comment_text += token.val
                token = self.nxt()
            
            # Store the comment, preserving whitespace (only remove comment markers)
            if comment_text:
                # Remove only the comment markers, preserve all whitespace
                cleaned_text = comment_text
                if comment_type == "//":
                    cleaned_text = comment_text[2:] if len(comment_text) >= 2 else ""  # Preserve whitespace after //
                elif comment_type == "$":
                    cleaned_text = comment_text[1:] if len(comment_text) >= 1 else ""  # Preserve whitespace after $
                elif comment_type == "*":
                    cleaned_text = comment_text[1:] if len(comment_text) >= 1 else ""  # Preserve whitespace after *
                
                # Remove only trailing newline, preserve all other whitespace
                cleaned_text = cleaned_text.rstrip('\n\r')
                
                if cleaned_text or cleaned_text == "":  # Store even empty comments to preserve structure
                    # If we haven't lexed any non-whitespace token on this line yet, this was a full-line comment.
                    # Mark it so the parser won't also emit a BlankLine for the same physical line.
                    if not self.lexed_nonwhite_on_this_line:
                        self.full_line_comment_lines.add(self.line_num)
                    self.current_line_comments.append((cleaned_text, comment_type))
                    # Also add to persistent queue
                    self.comment_queue.append((cleaned_text, comment_type, self.line_num))
        return token
    
    def get_line_comments(self) -> List[Tuple[str, str]]:
        """Get comments found on the current line and clear the list.
        Returns list of (comment_text, comment_type) tuples."""
        comments = self.current_line_comments.copy()
        self.current_line_comments.clear()
        return comments
    
    def get_queued_comments(self, up_to_line: Optional[int] = None) -> List[Tuple[str, str, int]]:
        """Get comments from the queue, optionally up to a specific line number.
        Comments are removed from the queue after retrieval."""
        if up_to_line is None:
            # Return all queued comments
            comments = self.comment_queue.copy()
            self.comment_queue.clear()
            return comments
        else:
            # Return comments up to and including the specified line
            comments = [c for c in self.comment_queue if c[2] <= up_to_line]
            self.comment_queue = [c for c in self.comment_queue if c[2] > up_to_line]
            return comments

    def lex(self):
        """Create an iterator over pattern-matches"""
        token = self.nxt()
        while token:  # Iterate over token-matches

            # Skip whitespace & comments
            token = self.eat_idle(token)

            # Handle backslash line continuation
            # If we see a BACKSLASH and the current line ends with backslash, skip it and the following NEWLINE
            if token and token.tp == Tokens.BACKSLASH and self.current_line_ends_with_backslash:
                # Skip the BACKSLASH token
                token = self.nxt()
                # Skip whitespace
                while token and token.tp == Tokens.WHITE:
                    token = self.nxt()
                # If next is NEWLINE, skip it (continuation)
                if token and token.tp == Tokens.NEWLINE:
                    self.lexed_nonwhite_on_this_line = False
                    self.current_line_comments.clear()
                    # Continue to next line without yielding NEWLINE
                    token = self.eat_idle(self.nxt())
                    # Skip any additional newlines/whitespace
                    while token and token.tp in (
                        Tokens.NEWLINE,
                        Tokens.WHITE,
                    ):
                        token = self.eat_idle(self.nxt())
                    continue
                # If not NEWLINE, fall through to yield the BACKSLASH (shouldn't happen normally)

            # Handle continuation-lines
            if token and token.tp == Tokens.NEWLINE:
                self.lexed_nonwhite_on_this_line = False
                # Comments have been captured to current_line_comments and comment_queue
                # Clear current_line_comments for next line, but keep in queue
                self.current_line_comments.clear()

                # Pull the next token (potentially skipping whitespace/comments), but
                # DO NOT collapse multiple NEWLINEs into one. We want the parser to
                # see each physical newline so it can preserve blank lines and
                # comment-only lines accurately.
                token = self.eat_idle(self.nxt())

                # Skip whitespace on the *next* line to detect plus-continuations,
                # but do not skip NEWLINE tokens (those represent real blank lines).
                while token and token.tp == Tokens.WHITE:
                    token = self.eat_idle(self.nxt())

                if token and token.tp == Tokens.PLUS:
                    # Cancelled newline; skip the PLUS and continue on the same statement.
                    self.lexed_nonwhite_on_this_line = True
                    token = self.nxt()
                    continue

                # Non-cancelled newline; yield this NEWLINE and continue parsing.
                yield Token(Tokens.NEWLINE, "\n")
                continue

            self.lexed_nonwhite_on_this_line = True
            yield token
            token = self.nxt()
