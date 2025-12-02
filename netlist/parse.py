"""
# Netlist Parsing 
"""

import os, copy, codecs
from enum import Enum
from pathlib import Path
from warnings import warn
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional, Sequence, Union, Set
from pydantic.dataclasses import dataclass

# Local Imports
from .dialects import DialectParser
from .data import *


class ErrorMode(Enum):
    """Enumerated Error-Response Strategies"""

    RAISE = 0  # Raise any generated exceptions
    STORE = 1  # Store error-generating content in `Unknown` elements


@dataclass
class ParseOptions:
    """Parse Options"""

    dialect: Optional[NetlistDialects] = None  # Initial netlist dialect
    errormode: ErrorMode = ErrorMode.RAISE  # Error-handling mode
    recurse: bool = True  # Whether to recurse into dependent & included files


def parse(
    src: Union[str, os.PathLike, Sequence[os.PathLike]],
    *,
    options: Optional[ParseOptions] = None,
) -> Program:
    """
    Primary netist-parsing entry point.
    Parse a potentially multi-file netlist-`Program` starting at `src`.
    Optional argument `options` sets all behavior laid out by the `ParseOptions` class.
    """
    if isinstance(src, str):
        return parse_str(src, options=options)
    return parse_files(src, options=options)


def parse_files(
    src: Union[os.PathLike, Sequence[os.PathLike]],
    *,
    options: Optional[ParseOptions] = None,
) -> Program:
    """Parse netlist content from file or files `src`."""

    if options is None:  # If not provided, create the default `ParseOptions`.
        options = ParseOptions()

    # Cover the cases of a single file
    if not isinstance(src, list):
        src = [src]

    # If an initial dialect has not been provided, infer it from the first file
    if options.dialect is None:
        options.dialect = default_dialect(src[0])

    # Create the parser, and parse each file
    p = Parser(src, options)
    p.parse()

    for e in p.entries():
        if isinstance(e, Unknown):
            warn(f"Unknown Netlist Entry {e}")

    # Return the parsed `Program`
    return p.program


def parse_str(src: str, *, options: Optional[ParseOptions] = None) -> Program:
    """Parse netlist content from a string"""

    if options is None:
        options = ParseOptions()
    if options.dialect is None:
        # Set "spectre-spice" as the default dialect
        options.dialect = NetlistDialects.SPECTRE_SPICE

    # Maybe a hack, maybe not, since so much of netlist-world revolves around files.
    # We write `src` to a temporary file, and parse that.

    f = NamedTemporaryFile()
    f.write(bytes(src, "utf-8"))
    f.seek(0)

    return parse_files(src=[f.name], options=options)


def default_dialect(path: os.PathLike) -> NetlistDialects:
    """
    Infer a default dialect from a file name, particularly its suffix.

    For files of suffix `scs` this is straightforwardly set to SPECTRE.
    All other suffixes are less clear without knowing more context.
    They are set to the most flexible SPECTRE_SPICE,
    which generally uses SPICE-style syntax, but also includes dialect-changes.
    """

    p = Path(path).absolute()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix == ".scs":
        return NetlistDialects.SPECTRE
    return NetlistDialects.SPECTRE_SPICE


class Parser:
    """Multi-File "Netlist Program" Parser"""

    def __init__(
        self,
        src: List[Path],
        options: ParseOptions,
    ):
        self.options = options
        self.pending: List[Path] = [Path(s).absolute() for s in src]
        # Initialize the dialects for each file
        self.pending_dialects = {s: options.dialect for s in self.pending}
        self.done: Set[Path] = set()
        self.program = Program([])
        self.dialect = options.dialect
        self.file_parser = None

    def parse(self) -> None:
        """Parse all specified input, including (potentially recursive) dependencies."""

        while self.pending:
            sourcefile = self.parse_one()
            self.program.files.append(sourcefile)

    def parse_one(self) -> SourceFile:
        """Parse a single file, returning its resultant `SourceFile` tree.
        Note the result-value *is not* stored in `self.program`."""

        path = self.pending.pop()

        # Check for validity of source file
        p = Path(path).absolute()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(p)

        # Source found; start parsing, to a list of Statements
        self.dialect = self.pending_dialects[p]
        self.file_parser = FileParser(self.dialect, self.options)
        path, stmts = self.file_parser.parse(p)
        self.done.add(p)

        # Filter out a few things
        # Before filtering DialectChange, preserve any comments that were associated with them
        # by ensuring they're already in the stmts list as separate entries
        # (they should already be there from the parsing logic, but we'll verify)
        filtered_stmts = []
        for e in stmts:
            if not isinstance(e, DialectChange):
                filtered_stmts.append(e)
        stmts = filtered_stmts

        # Recursively descend into files it depends upon
        if self.options.recurse:
            file_stmts = [e for e in stmts if isinstance(e, (Include, UseLibSection))]
            self.recurse(path, file_stmts)

        # Form into a scope-hierarchical tree, and add it to our result-`Program`
        sourcefile = hierarchicalize(path, stmts)
        return sourcefile

    def recurse(self, path: Path, contents: List[Statement]):
        """Parse included-files in SourceFile `f`"""
        for s in contents:
            if isinstance(s, (Include, UseLibSection)):
                # Differentiate absolute vs relative paths, relative to active source-file
                incp = s.path if s.path.is_absolute() else path.parent / s.path
                incp = incp.resolve()
                if not incp.exists() or not incp.is_file():
                    raise FileNotFoundError(incp)
                if incp in self.done:
                    continue

                # Sort out the initial dialect for the new file.
                # Spectre-family dialects expect included files to start in the dialect indicated by their file-extension.
                # All other dialects (that we know of) include files in their own dialect.
                if self.dialect in (
                    NetlistDialects.SPECTRE,
                    NetlistDialects.SPECTRE_SPICE,
                ):
                    # Update our dialect based on the file-extension we load
                    dialect_enum = default_dialect(incp)
                    # self.file_parser = FileParser(dialect_enum)
                else:  # Keep our existing dialect
                    dialect_enum = self.dialect
                    # self.file_parser = FileParser(dialect_enum)

                # Add the file and its dialect to the pending-queue
                self.pending.append(incp)
                self.pending_dialects[incp] = dialect_enum

                # # And kick off parsing of the included file
                # self.parse(incp)

    def entries(self):
        """Iterator of all parsed Entries"""
        for f in self.program.files:
            for e in f.contents:
                yield e


class FileParser:
    """Single-File Parser
    Produces a flat list of Statement-Entries"""

    def __init__(
        self,
        dialect: NetlistDialects,
        options: ParseOptions,
    ):
        self.dialect = dialect
        self.dialect_parser = None
        self.options = options
        self.line_num = 1

    def notify(self, reason: DialectChange):
        """Notification from dialect-parser that something's up.
        E.g. for dialect changes via `simulator lang` statements."""

        s = reason.dialect.lower().strip()
        if s == "spectre":
            dialect = NetlistDialects.SPECTRE
        elif s == "spice":
            dialect = NetlistDialects.SPECTRE_SPICE
        else:
            self.fail(f"Invalid `simulator lang` {s}")

        dcls = DialectParser.from_enum(dialect)
        self.dialect_parser = dcls.from_parser(self.dialect_parser)

    def _parse_file(self) -> List[Statement]:
        """Parse the (open) file-pointer at `self.fp`"""

        # Create our iterator over file-lines, and core dialect-parser
        lines = iter(self.fp.readline, None)
        dcls = DialectParser.from_enum(self.dialect)
        self.dialect_parser = dcls.from_lines(lines=lines, parent=self)
        self.dialect_parser.start()

        stmts = []
        s = True
        pending_comments = []  # Track comments that need to be associated
        
        while s:  # Main loop over statements
            si = SourceInfo(
                line=self.dialect_parser.line_num,
                dialect=self.dialect_parser.enum,
            )
            
            # Check for blank lines before this statement
            # The parser tracks blank lines consumed by eat_blanks()
            if hasattr(self.dialect_parser, '_blank_lines_before_statement') and self.dialect_parser._blank_lines_before_statement:
                blank_count = self.dialect_parser._blank_lines_before_statement.pop(0)
                if blank_count > 0:
                    # Add BlankLine entries for each blank line
                    # BlankLine is already imported via "from .data import *"
                    for i in range(blank_count):
                        blank_line = BlankLine()
                        blank_line.source_info = SourceInfo(line=si.line - blank_count + i, dialect=si.dialect)
                        stmts.append(blank_line)

            try:  # Catch errors in primary parsing routines
                s = self.dialect_parser.parse_statement()
            except NetlistParseError as e:
                # Error Handling. Can either include `Unknown` statements, or re-raise.
                if self.options.errormode == ErrorMode.RAISE:
                    raise e
                # Otherwise, we include `Unknown` statements
                warn(e)  # Issue a warning
                s = Unknown("")  # FIXME: include the line-content in Unknowns
                self.dialect_parser.eat_rest_of_statement()

            if s is not None:  # None-value indicates EOF
                s.source_info = si
                
                # Check for pending comments from parser
                if hasattr(self.dialect_parser, '_pending_comments') and self.dialect_parser._pending_comments:
                    # Find comments for this statement
                    for i, (stmt, before_comments, after_comments) in enumerate(self.dialect_parser._pending_comments):
                        if stmt is s:
                            # Add before comments as separate entries (they'll be written before the statement)
                            for j, comment in enumerate(before_comments):
                                comment.source_info = SourceInfo(line=si.line - len(before_comments) + j, dialect=si.dialect)
                                stmts.append(comment)
                            # Store after comments to add after the statement
                            if after_comments:
                                pending_comments.append((s, after_comments))
                            # Remove from pending
                            self.dialect_parser._pending_comments.pop(i)
                            break
                
                stmts.append(s)
                
                # Add any after comments for this statement
                for stmt, after_comments in pending_comments[:]:
                    if stmt is s:  # Comments for current statement
                        for comment in after_comments:
                            comment.source_info = si
                            stmts.append(comment)
                        pending_comments.remove((stmt, after_comments))

        # Add any remaining standalone comments from the comment queue
        # But first, collect all comments that were already added as "before" or "after" comments
        # to avoid duplicates. Use text content as the key since line numbers might differ
        # due to how comments are associated with statements
        already_added_comment_texts = set()
        for stmt in stmts:
            if isinstance(stmt, Comment):
                # Use just the text as the key to catch duplicates even if line numbers differ
                already_added_comment_texts.add(stmt.text)
        
        remaining_comments = self.dialect_parser.lex.get_queued_comments()
        for comment_text, comment_type, line_num in remaining_comments:
            # Skip if this comment text was already added (regardless of line number)
            if comment_text not in already_added_comment_texts:
                comment = Comment(text=comment_text, position="standalone")
                comment.source_info = SourceInfo(line=line_num, dialect=self.dialect_parser.enum)
                stmts.append(comment)
                already_added_comment_texts.add(comment_text)  # Mark as added

        # Deduplicate comments in stmts list (remove consecutive duplicate comment sequences)
        # This handles cases where the same comments were collected as "before" for multiple statements
        # We look for sequences of consecutive comments and remove duplicate sequences
        deduplicated_stmts = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if isinstance(stmt, Comment):
                # Check if this starts a sequence of comments that we've seen before
                # Look ahead to find the end of this comment sequence
                comment_seq = []
                j = i
                while j < len(stmts) and isinstance(stmts[j], Comment):
                    comment_seq.append(stmts[j].text)
                    j += 1
                
                # Check if we've seen this exact sequence before (as consecutive comments)
                if len(comment_seq) > 0:
                    # Check previous entries for the same sequence
                    found_duplicate = False
                    if len(deduplicated_stmts) >= len(comment_seq):
                        # Check if the last len(comment_seq) entries are comments matching this sequence
                        prev_seq = [s.text for s in deduplicated_stmts[-len(comment_seq):] 
                                   if isinstance(s, Comment)]
                        if len(prev_seq) == len(comment_seq) and prev_seq == comment_seq:
                            found_duplicate = True
                    
                    if found_duplicate:
                        # Skip this duplicate sequence
                        i = j
                        continue
                
                # Add this comment (or sequence)
                deduplicated_stmts.append(stmt)
                i += 1
            else:
                # Always add non-comment statements
                deduplicated_stmts.append(stmt)
                i += 1

        return deduplicated_stmts

    def parse(self, path: Path) -> Tuple[Path, List[Statement]]:
        """Parse the netlist `SourceFile` at `path`."""
        with codecs.open(path, "r", encoding="utf-8", errors="replace") as f:
            self.path = path
            self.fp = f
            stmts = self._parse_file()
        return (path, stmts)


def hierarchicalize(p: Path, stmts: List[Statement]) -> SourceFile:
    """Convert a flat list of Statements into a hierarchical tree"""
    nodes = HierarchyCollector(stmts).collect_source_file()
    return SourceFile(p, nodes)


class HierarchyCollector:
    """
    Hierarchy-collector to create multi-statement tree structure, e.g.:
    * Libraries and Sections
    * (Potentially nested) sub-circuit definitions.
    """

    def __init__(self, stmts: List[Statement]):
        self.stmts = iter(stmts)

    def nxt(self) -> Optional[Statement]:
        """Get our next `Statement`, or `None` if exhausted"""
        return next(self.stmts, None)

    def collect_source_file(self) -> List[Entry]:
        """Primary Entry Point.
        Transform a list of flat file-entries to a hierarchical tree of file-nodes.
        While the return-type is `List[Entry]`, these lists are designed as the content of a `SourceFile`."""

        nodes = []
        params = []  # Collect parameter-declarations into one

        while True:
            stmt = self.nxt()
            if stmt is None:
                break  # library-end on file-end

            if isinstance(stmt, StartSubckt):
                s = self.collect_subckt(start=stmt)
            elif isinstance(stmt, StartLib):
                s = self.collect_lib(start=stmt)
            elif isinstance(stmt, StartLibSection):
                s = self.collect_lib_section(start=stmt)

            # Append any parameter-declarations to our running list
            elif isinstance(stmt, ParamDecl):
                params.append(stmt)
                continue
            elif isinstance(stmt, ParamDecls):
                params.extend(stmt.params)
                continue

            else:  # Everything else, copy along
                s = stmt
            nodes.append(s)

        if params:
            # If we found any parameters, stick them at the beginning of the nodes-list.
            # This tends to be the most interprable for eventual netlist formats.
            nodes = [ParamDecls(params)] + nodes

        return nodes

    def collect_subckt(self, start: StartSubckt) -> SubcktDef:
        """Collect a sub-circuit definition"""

        nodes = []
        params = copy.copy(start.params)

        while True:
            stmt = self.nxt()
            if stmt is None:
                break  # library-end on file-end

            if isinstance(stmt, EndSubckt):
                break  # done with this module

            # Inline Subckt Parameter Promotion Logic
            # For Spectre `inline subckt`, parameters defined in the body (via `parameters` keyword)
            # are formal parameters, even if they have defaults.
            if start.inline and isinstance(stmt, (ParamDecl, ParamDecls)):
                if isinstance(stmt, ParamDecl):
                    params.append(stmt)
                else:
                    params.extend(stmt.params)
                continue

            # Parameter statements in subckt scope:
            # - Formal parameters (from "parameters:" statement, no defaults) should be promoted to subcircuit params
            # - Local/global parameters (from ".param" or ".PARAM", with defaults) should remain as entries
            if isinstance(stmt, ParamDecl):
                # Single parameter declaration - check if it's a formal param (no default)
                if stmt.default is None:
                    params.append(stmt)  # Formal parameter - promote to subcircuit param
                else:
                    nodes.append(stmt)  # Local parameter - keep as entry
                continue
            elif isinstance(stmt, ParamDecls):
                # Check if all params are formal (no defaults) - if so, promote them
                formal_params = [p for p in stmt.params if p.default is None]
                local_params = [p for p in stmt.params if p.default is not None]
                if formal_params:
                    params.extend(formal_params)  # Promote formal params
                if local_params:
                    # Keep local params as ParamDecls entry
                    nodes.append(ParamDecls(params=local_params))
                continue

            if isinstance(stmt, StartSubckt):
                # Collect nested sub-circuit definitions
                s = self.collect_subckt(start=stmt)
            else:  # Anything else, copy along
                s = stmt
            nodes.append(s)

        return SubcktDef(
            name=start.name, ports=start.ports, params=params, entries=nodes
        )

    def collect_lib_section(self, start: StartLibSection) -> LibSectionDef:
        """Collect a library section"""

        nodes = []
        params = []  # Collect parameter-declarations into one

        while True:
            stmt = self.nxt()
            if stmt is None:
                break  # library-end on file-end

            if isinstance(stmt, EndLibSection):
                break  # done with this section
            elif isinstance(stmt, StartLibSection):
                self.fail()  # these don't nest; something's wrong
            elif isinstance(stmt, StartSubckt):
                s = self.collect_subckt(start=stmt)

            # Append any parameter-declarations to our running list
            elif isinstance(stmt, ParamDecl):
                params.append(stmt)
                continue
            elif isinstance(stmt, ParamDecls):
                params.extend(stmt.params)
                continue

            else:  # Anything else, copy along
                s = stmt
            nodes.append(s)

        if params:
            # If we found any parameters, stick them at the beginning of the nodes-list.
            # This tends to be the most interprable for eventual netlist formats.
            nodes = [ParamDecls(params)] + nodes

        return LibSectionDef(name=start.name, entries=nodes)

    def collect_lib(self, start: StartLib) -> Library:
        """Collect a library definition"""

        # FIXME: is this really a thing? Or are we always collecting a `LibSectionDef` at a time?

        sections = []
        while True:
            stmt = self.nxt()
            if stmt is None:
                break  # library-end on file-end

            if isinstance(stmt, EndLib):
                break  # done with this library
            elif isinstance(stmt, EndLibSection):
                # something went wrong; lib section ends without beginning
                msg = "Invalid `EndLibSection` without `StartLibSection`"
                self.fail(msg)
            elif isinstance(stmt, StartLibSection):
                s = self.collect_lib_section(start=stmt)
                sections.append(s)
            else:
                msg = f"Invalid statement in library: {stmt}"
                self.fail(msg)  # invalid type

        return Library(name=start.name, sections=sections)

    def fail(self, msg: str):
        NetlistParseError.throw(msg)
