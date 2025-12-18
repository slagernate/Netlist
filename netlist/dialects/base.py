""" 
# Netlist Parser Base Class
"""

# Std-Lib Imports
from enum import Enum
from typing import Iterable, Any, Optional, List, Union, Tuple

# Local Imports
from ..data import *
from ..lex import Lexer, Token, Tokens


class ParserState(Enum):
    """States of the parser, as they need be understood by the lexer."""

    PROGRAM = 0  # Typical program content
    EXPR = 1  # High-priority expressions


class DialectParser:
    """Netlist Dialect-Parsing Base-Class"""

    enum = None

    def __init__(self, lex: Lexer, parent: Optional["FileParser"] = None):
        self.parent = parent
        # Initialize our state
        self.state = ParserState.PROGRAM
        self.tokens = None
        self.prev = None
        self.cur = None
        self.nxt = None
        self.nxt0 = None
        self.rewinding = False
        # Initialize our lexer and its token-generator
        self.lex = lex
        self.lex.parser = self

    @property
    def line_num(self):
        return self.lex.line_num

    @classmethod
    def from_parser(cls, p: "DialectParser") -> "DialectParser":
        """Create from another DialectParser,
        as during a `simulator lang` DialectChange.
        Includes copying its private internal state."""
        rv = cls(lex=p.lex, parent=p.parent)
        rv.state = p.state
        rv.tokens = p.tokens
        rv.cur = p.cur
        rv.nxt = p.nxt
        rv.nxt0 = p.nxt0
        rv.rewinding = p.rewinding
        return rv

    @classmethod
    def from_lines(cls, lines: Iterable[str], **kwargs) -> "DialectParser":
        """Create from a line iterator"""
        return cls(lex=Lexer(lines), **kwargs)

    @classmethod
    def from_str(cls, txt: str, **kwargs) -> "DialectParser":
        """Create from a multi-line input string"""
        ls = [line + "\n" for line in txt.split("\n")[:-1]]
        ls += [txt.split("\n")[-1]]
        return cls.from_lines(lines=iter(ls), **kwargs)

    @classmethod
    def from_enum(cls, dialect: Optional["NetlistDialects"] = None):
        """Return a Dialect sub-class based on the `NetlistDialects` enum.
        Returns the default class if argument `dialect` is not provided or `None`."""
        from .spice import SpiceDialectParser, NgSpiceDialectParser
        from .spectre import SpectreDialectParser, SpectreSpiceDialectParser

        if dialect is None:
            return SpectreDialectParser
        if dialect == NetlistDialects.SPECTRE:
            return SpectreDialectParser
        if dialect == NetlistDialects.SPECTRE_SPICE:
            return SpectreSpiceDialectParser
        if dialect == NetlistDialects.NGSPICE:
            return NgSpiceDialectParser
        if dialect == NetlistDialects.XYCE:
            return NgSpiceDialectParser
        raise ValueError

    def eat_blanks(self):
        """Pass over blank-lines, generally created by full-line comments.
        Returns the number of blank lines consumed."""
        blank_count = 0
        while self.nxt and self.nxt.tp == Tokens.NEWLINE:
            blank_count += 1
            self.advance()
        return blank_count

    def eat_rest_of_statement(self):
        """Ignore Tokens from `self.cur` to the end of the current statement.
        Largely used for error purposes."""
        while self.nxt and not self.match(Tokens.NEWLINE):
            self.advance()

    def start(self) -> None:
        # Initialize our token generator
        self.tokens = self.lex.lex()
        # And queue up our lookahead tokens
        self.advance()

    def advance(self) -> None:
        self.prev = self.cur
        self.cur = self.nxt
        if self.rewinding:
            self.rewinding = False
            self.nxt = self.nxt0
        else:
            self.nxt = next(self.tokens, None)
        self.nxt0 = None

    def rewind(self):
        """Rewind by one Token. Error if already in rewinding state."""
        if self.rewinding:
            self.fail()
        self.rewinding = True
        self.nxt0 = self.nxt
        self.nxt = self.cur
        self.cur = self.prev
        self.prev = None

    def peek(self) -> Optional[Token]:
        """Peek at the next Token"""
        return self.nxt

    def match(self, tp: str) -> bool:
        """Boolean indication of whether our next token matches `tp`"""
        if self.nxt and tp == self.nxt.tp:
            self.advance()
            return True
        return False

    def match_any(self, *tp: List[str]) -> Optional[Tokens]:
        """Boolean indication of whether our next token matche *any* provided `tp`.
        Returns the matching `Tokens` (type) if successful, or `None` otherwise."""
        if not self.nxt:
            return None
        for t in tp:
            if self.nxt.tp == t:
                self.advance()
                return t
        return None

    def expect(self, *tp: List[str]) -> None:
        """Assertion that our next token matches `tp`.
        Note this advances if successful, effectively discarding `self.cur`."""
        if not self.match_any(*tp):
            self.fail(f"Invalid token: {self.nxt}, expecting one of {tp}")

    def expect_any(self, *tp: List[str]) -> Tokens:
        """Assertion that our next token matches one of `tp`, and return the one it was."""
        rv = self.match_any(*tp)
        if rv is None:
            self.fail(f"Invalid token: {self.nxt}, expecting one of {tp}")
        return rv

    def parse(self, f=None) -> Any:
        """Perform parsing. Succeeds if top-level is parsable by function `f`.
        Defaults to parsing `Expr`."""
        self.start()
        func = f if f else self.parse_expr
        self.root = func()
        if self.nxt is not None:  # Check whether there's more stuff
            self.fail()
        return self.root

    def parse_subckt_start(self) -> StartSubckt:
        """module_name ( port1 port2 port2 ) p1=param1 p2=param2 ..."""

        # Boolean indication of the `inline`-ness
        _inline = self.match(Tokens.INLINE)
        self.expect(Tokens.SUBCKT)

        # Grab the module/subckt name
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)

        # Parse the parens-optional port-list
        if self.match(Tokens.LPAREN):  # Parens case
            term = lambda s: not s.nxt or s.nxt.tp in (Tokens.RPAREN, Tokens.NEWLINE)
            ports = self.parse_node_list(term)
            self.expect(Tokens.RPAREN)

        else:  # No-parens case
            ports = self.parse_node_list(_endargs_startkwargs)
            # If we landed on a key-value param key, rewind it
            if self.nxt and self.nxt.tp == Tokens.EQUALS:
                self.rewind()
                ports.pop()

        # Parse parameters
        params = self.parse_param_declarations()

        # And create & return our instance
        return StartSubckt(name=name, ports=ports, params=params, inline=_inline)

    def parse_ident(self) -> Ident:
        """Parse an Identifier"""
        self.expect(Tokens.IDENT)
        return Ident(self.cur.val)

    def parse_node_ident(self) -> Ident:
        """Parse a Node-Identifier - either a var-name-style Ident or an Int."""
        self.expect(Tokens.IDENT, Tokens.INT)
        return Ident(str(self.cur.val))

    def parse_list(self, parse_item, term, *, MAXN=10_000) -> List[Any]:
        """Parse a whitespace-separated list of entries possible by function `parse_item`.
        Terminated in the condition `term(self)`."""
        rv = []
        for i in range(MAXN, -1, -1):
            if term(self):
                break
            rv.append(parse_item())
        if i <= 0:  # Check whether the time-out triggered
            self.fail()
        return rv

    def parse_node_list(self, term, *, MAXN=10_000) -> List[Ident]:
        """Parse a Node-Identifier (Ident or Int) list,
        terminated in the condition `term(self)`."""
        return self.parse_list(self.parse_node_ident, term=term, MAXN=MAXN)

    def parse_ident_list(self, term, *, MAXN=10_000) -> List[Ident]:
        """Parse list of Identifiers"""
        return self.parse_list(self.parse_ident, term=term, MAXN=MAXN)

    def parse_expr_list(self, term, *, MAXN=10_000) -> List[Expr]:
        """Parse list of Expressions"""
        return self.parse_list(self.parse_expr, term=term, MAXN=MAXN)

    def parse_primitive(self) -> Primitive:
        """Parse a Spice-format primitive instance"""
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        args = []

        # For Primitives it is not necessarily clear at parse-time which arguments
        # are ports vs parameters. All are parsed as `Expr` and sorted out later.
        if self.match(Tokens.LPAREN):

            # Parse ports in parentheses
            # In this case, we actually do know what's a port vs param.
            # But both are still stored in `args`, for consistency with the alternate case.
            term = lambda s: not s.nxt or s.nxt.tp in (Tokens.RPAREN, Tokens.NEWLINE)
            args = self.parse_expr_list(term)
            self.expect(Tokens.RPAREN)

            # Now parse the positional parameters
            args = self.parse_expr_list(_endargs_startkwargs)

        else:  # No-parens case
            args = self.parse_expr_list(_endargs_startkwargs)
            # If we landed on a key-value param key, rewind it
            if self.nxt and self.nxt.tp == Tokens.EQUALS:
                self.rewind()
                args.pop()

        # Parse parameters
        params = self.parse_param_values()
        # And create & return our instance
        return Primitive(name=name, args=args, kwargs=params)

    def parse_instance(self) -> Instance:
        """iname (? port1 port2 port2 )? mname p1=param1 p2=param2 ..."""
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        conns = []

        # Parse the parens-optional port-list
        if self.match(Tokens.LPAREN):  # Parens case
            term = lambda s: not s.nxt or s.nxt.tp in (Tokens.RPAREN, Tokens.NEWLINE)
            conns = self.parse_node_list(term)
            self.expect(Tokens.RPAREN)

            # Grab the module name
            self.expect(Tokens.IDENT)
            module = Ref(ident=Ident(self.cur.val))

        else:  # No-parens case
            conns = self.parse_node_list(_endargs_startkwargs)
            # If we landed on a key-value param key, rewind it
            if self.nxt and self.nxt.tp == Tokens.EQUALS:
                self.rewind()
                if not len(conns):  # Something went wrong!
                    self.fail()
                conns.pop()
            # Grab the module name, at this point errantly in the `conns` list
            module = Ref(ident=conns.pop())

        # Parse parameters
        params = self.parse_instance_param_values()
        # And create & return our instance
        return Instance(name=name, module=module, conns=conns, params=params)

    def parse_instance_param_values(self) -> List[ParamVal]:
        # Base class parses instance-params as any other params
        return self.parse_param_values()

    def parse_param_val(self) -> ParamVal:
        """Parse a parameter value (name = expression) and capture any inline comments."""
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.EQUALS)
        
        # Store the current line number and comment queue state before parsing expression
        # This ensures we only capture comments that are truly on the same line
        param_line = self.lex.line_num
        comments_before = list(self.lex.comment_queue) if hasattr(self.lex, 'comment_queue') else []
        
        e = self.parse_expr()
        
        # After parsing expression, check comment queue for new comments on THIS line only
        comments_after = list(self.lex.comment_queue) if hasattr(self.lex, 'comment_queue') else []
        new_comments = [c for c in comments_after if c not in comments_before and c[2] == param_line]
        
        # Capture inline comment for this ParamVal
        inline_comment = None
        comment_to_remove = None
        
        for comment_text, comment_type, line_num in new_comments:
            if comment_type == "//":
                inline_comment = comment_text
                comment_to_remove = (comment_text, comment_type, line_num)
                break  # Take the first inline comment on this line
        
        # Also check current_line_comments (in case it hasn't been cleared yet)
        if inline_comment is None and hasattr(self.lex, 'current_line_comments'):
            for comment_text, comment_type in self.lex.current_line_comments:
                if comment_type == "//":
                    inline_comment = comment_text
                    # Remove from current_line_comments
                    self.lex.current_line_comments = [(t, ct) for t, ct in self.lex.current_line_comments 
                                                     if not (t == comment_text and ct == comment_type)]
                    break  # Take the first inline comment
        
        # If still no comment, check if the next token is a comment
        if inline_comment is None:
            # Skip whitespace to see if there's a comment coming
            while self.nxt and self.nxt.tp == Tokens.WHITE:
                self.advance()
            
            # Check if the next token is a comment (// style)
            if self.nxt and self.is_comment(self.nxt) and self.nxt.tp == Tokens.DUBSLASH:
                # There's an inline comment - manually process it
                comment_token = self.nxt
                # Call eat_idle to process the comment - this will collect it
                next_token = self.lex.eat_idle(comment_token)
                # The comment should now be in current_line_comments
                if hasattr(self.lex, 'current_line_comments'):
                    for comment_text, comment_type in self.lex.current_line_comments:
                        if comment_type == "//":
                            inline_comment = comment_text
                            # Remove from current_line_comments
                            self.lex.current_line_comments = [(t, ct) for t, ct in self.lex.current_line_comments 
                                                             if not (t == comment_text and ct == comment_type)]
                            break  # Take the first inline comment
                # Update nxt to the token after the comment
                self.nxt = next_token
        
        # Remove the comment from comment_queue if we captured it
        if comment_to_remove and hasattr(self.lex, 'comment_queue'):
            self.lex.comment_queue = [(t, ct, ln) for t, ct, ln in self.lex.comment_queue 
                                     if not (t == comment_to_remove[0] and ct == comment_to_remove[1] and ln == comment_to_remove[2])]
        
        # Note: We do NOT store inline comments in _param_inline_comments here because:
        # 1. For instance parameters (ParamVal), we store it directly in ParamVal.comment
        # 2. For parameter declarations (ParamDecl), parse_param_declaration() will handle it
        # This prevents comments from instance parameters from being incorrectly associated with model parameters
        
        return ParamVal(name, e, comment=inline_comment)

    def parse_param_declaration(self):
        val = self.parse_param_val()
        # FIXME: Skipping this auxiliary stuff for now
        if self.match(Tokens.DEV_GAUSS):
            self.expect(Tokens.EQUALS)
            _e = self.parse_expr()
        
        # Check for inline comment (// style) on the same line
        # Use the comment from the ParamVal (which was captured in parse_param_val)
        inline_comment = val.comment if hasattr(val, 'comment') else None
        
        # If no comment was found in ParamVal, check comment_queue for comments on the current line
        if inline_comment is None:
            current_line = self.lex.line_num
            if hasattr(self.lex, 'comment_queue'):
                # Look for comments on the current line
                for comment_text, comment_type, line_num in self.lex.comment_queue:
                    if comment_type == "//" and line_num == current_line:
                        inline_comment = comment_text
                        # Remove this comment from the queue since we're using it
                        self.lex.comment_queue = [(t, ct, ln) for t, ct, ln in self.lex.comment_queue 
                                                 if not (t == comment_text and ct == comment_type and ln == line_num)]
                        break  # Take the first inline comment on this line
        
        # Also check current_line_comments (in case it hasn't been cleared yet)
        if inline_comment is None and hasattr(self.lex, 'current_line_comments'):
            for comment_text, comment_type in self.lex.current_line_comments:
                if comment_type == "//":
                    inline_comment = comment_text
                    break  # Take the first inline comment
        
        # If still no comment, check if the next token is a comment
        if inline_comment is None:
            # Skip whitespace to see if there's a comment coming
            while self.nxt and self.nxt.tp == Tokens.WHITE:
                self.advance()
            
            # Check if the next token is a comment (// style)
            if self.nxt and self.is_comment(self.nxt) and self.nxt.tp == Tokens.DUBSLASH:
                # There's an inline comment - manually process it
                comment_token = self.nxt
                # Call eat_idle to process the comment - this will collect it
                next_token = self.lex.eat_idle(comment_token)
                # The comment should now be in current_line_comments
                if hasattr(self.lex, 'current_line_comments'):
                    for comment_text, comment_type in self.lex.current_line_comments:
                        if comment_type == "//":
                            inline_comment = comment_text
                            break  # Take the first inline comment
                # Update nxt to the token after the comment
                self.nxt = next_token
        
        return ParamDecl(val.name, val.val, comment=inline_comment)

    def parse_param_declarations(self) -> List[ParamDecl]:
        """Parse a set of parameter declarations"""
        term = lambda s: s.nxt is None or s.match(Tokens.NEWLINE)
        MAXN = 100_000
        # Believe it or not, we do find real netlists with more than `parse_list`'s default (10k) parameters
        # defined in a single `.param` statement. Set MAXN to 100k to be safe.
        return self.parse_list(self.parse_param_declaration, term=term, MAXN=MAXN)

    def parse_param_values(self) -> List[ParamVal]:
        """( ident = expr )*"""
        term = lambda s: s.nxt is None or s.match(Tokens.NEWLINE)
        return self.parse_list(self.parse_param_val, term=term)

    def parse_option_values(self) -> List[OptionVal]:
        """Parse a list of `OptionVal`s, which can be expressions or strings."""
        term = lambda s: s.nxt is None or s.match(Tokens.NEWLINE)
        return self.parse_list(self.parse_option, term=term)

    def parse_option(self) -> Option:
        """Parse an `Option` name: `OptionVal` pair"""
        name = self.parse_ident()
        self.expect(Tokens.EQUALS)
        if self.peek().tp in (Tokens.TICK, Tokens.DUBQUOTE):
            txt = self.parse_quote_string()
            val = QuotedString(txt)
        else:
            val = self.parse_expr()
        return Option(name, val)

    def parse_end_sub(self):
        self.expect(Tokens.ENDS)
        if self.match(Tokens.IDENT):
            name = Ident(self.cur.val)
        else:
            name = None
        self.expect(Tokens.NEWLINE)
        return EndSubckt(name)

    def parse_expr0(self) -> Expr:
        """expr0b ( (<|>|<=|>=|==) expr0b )?"""
        e = self.parse_expr0b()
        if self.match_any(Tokens.GT, Tokens.LT, Tokens.GE, Tokens.LE, Tokens.DUBEQUALS):
            tp = self.parse_binary_operator(self.cur.tp)
            right = self.parse_expr0b()
            return BinaryOp(tp=tp, left=e, right=right)
        return e

    def parse_expr0b(self) -> Expr:
        """expr1 ( (+|-) expr0 )?"""
        e = self.parse_expr1()
        if self.match_any(Tokens.PLUS, Tokens.MINUS):
            tp = self.parse_binary_operator(self.cur.tp)
            right = self.parse_expr0b()
            return BinaryOp(tp=tp, left=e, right=right)
        return e

    def parse_expr1(self) -> Expr:
        """expr2 ( (*|/) expr1 )?"""
        e = self.parse_expr2()
        if self.match_any(Tokens.STAR, Tokens.SLASH):
            tp = self.parse_binary_operator(self.cur.tp)
            right = self.parse_expr1()
            return BinaryOp(tp=tp, left=e, right=right)
        return e

    def parse_expr2(self) -> Expr:
        """expr3 ( (**|^) expr2 )?"""
        e = self.parse_expr2b()
        if self.match_any(Tokens.DUBSTAR, Tokens.CARET):
            tp = self.parse_binary_operator(self.cur.tp)
            return BinaryOp(tp=tp, left=e, right=self.parse_expr2())
        return e

    def parse_expr2b(self) -> Expr:
        """expr3 ( ? expr0 : expr0 )?"""
        e = self.parse_expr3()
        if self.match(Tokens.QUESTION):
            if_true = self.parse_expr0()
            self.expect(Tokens.COLON)
            if_false = self.parse_expr0()
            return TernOp(e, if_true, if_false)
        return e

    def parse_expr3(self) -> Expr:
        """( expr ) or term"""
        if self.match(Tokens.LPAREN):
            e = self.parse_expr()
            self.expect(Tokens.RPAREN)
            return e
        return self.parse_term()

    def parse_term(self) -> Union[Int, Float, Ref]:
        """Parse a terminal value, or raise an Exception
        ( number | ident | unary(term) | call )"""
        if self.match(Tokens.METRIC_NUM):
            return MetricNum(self.cur.val)
        if self.match(Tokens.FLOAT):
            return Float(float(self.cur.val))
        if self.match(Tokens.INT):
            return Int(int(self.cur.val))
        if self.match_any(Tokens.PLUS, Tokens.MINUS):
            tp = self.parse_unary_operator(self.cur.tp)
            return UnaryOp(tp=tp, targ=self.parse_term())
        if self.match(Tokens.IDENT):
            ref = Ref(ident=Ident(self.cur.val))
            if self.match(Tokens.LPAREN):  # Function-call syntax
                # Parse arguments
                args = []
                MAX_ARGS = 100  # Set a "time-out" so that we don't get stuck here.
                for i in range(MAX_ARGS, -1, -1):
                    if self.match(Tokens.RPAREN):
                        break
                    a = self.parse_expr0()  # Grab an argument-expression
                    args.append(a)
                    if self.match(Tokens.RPAREN):
                        break
                    self.expect(Tokens.COMMA)
                if i <= 0:  # Check the time-out
                    self.fail()
                return Call(func=ref, args=args)
            return ref
        self.fail(f"Unexpected token while parsing expression-term: {self.cur}")

    @staticmethod
    def parse_unary_operator(tp: Tokens) -> UnaryOperator:
        """Parse `tp` to a unary operator"""
        the_map = {
            Tokens.PLUS: UnaryOperator.PLUS,
            Tokens.MINUS: UnaryOperator.NEG,
        }
        if tp in the_map:
            return the_map[tp]
        raise ValueError(f"Invalid Token {tp} when expecting binary operator")

    @staticmethod
    def parse_binary_operator(tp: Tokens) -> BinaryOperator:
        """Parse `tp` to a binary operator"""
        the_map = {
            Tokens.PLUS: BinaryOperator.ADD,
            Tokens.MINUS: BinaryOperator.SUB,
            Tokens.STAR: BinaryOperator.MUL,
            Tokens.SLASH: BinaryOperator.DIV,
            Tokens.DUBSTAR: BinaryOperator.POW,
            Tokens.CARET: BinaryOperator.POW,
            Tokens.GT: BinaryOperator.GT,
            Tokens.LT: BinaryOperator.LT,
            Tokens.GE: BinaryOperator.GE,
            Tokens.LE: BinaryOperator.LE,
            Tokens.DUBEQUALS: BinaryOperator.EQ,
        }
        if tp in the_map:
            return the_map[tp]
        raise ValueError(f"Invalid Token {tp} when expecting binary operator")

    def is_comment(self, tok: Token) -> bool:
        """Boolean indication of whether `tok` begins a Comment"""
        return tok.tp in (
            Tokens.DUBSLASH,
            Tokens.DOLLAR,
        ) or (self.are_stars_comments_now() and tok.tp in (Tokens.DUBSTAR, Tokens.STAR))

    def collect_before_comments(self) -> List["Statement"]:
        """Collect any line-starting comments before current statement.
        Checks the lexer for comments that were captured on previous lines."""
        comments: List[Statement] = []
        # Get comments from queue up to current line
        queued_comments = self.lex.get_queued_comments(up_to_line=self.lex.line_num - 1)
        for comment_text, comment_type, line_num in queued_comments:
            # Line-starting comments can be "*" or "//" type
            if comment_type in ("*", "//"):
                # Preserve even empty comment-only lines (e.g. `//`) as comments, not blank lines.
                # This allows mapping `//` -> `;` in Xyce output.
                comments.append(Comment(text=comment_text or "", position="before"))
        # Also check current line comments
        line_comments = self.lex.get_line_comments()
        for comment_text, comment_type in line_comments:
            if comment_type in ("*", "//"):
                comments.append(Comment(text=comment_text or "", position="before"))
        return comments

    def collect_inline_comment(self) -> Optional["Comment"]:
        """Check for inline comment (//) and return if found.
        This should be called after parsing a token but before advancing."""
        # Check current token for inline comment
        if self.cur and self.is_comment(self.cur):
            # Get comments from lexer
            line_comments = self.lex.get_line_comments()
            for comment_text, comment_type in line_comments:
                if comment_type == "//":
                    return Comment(text=comment_text, position="inline")
        return None

    def collect_after_comments(self) -> List["Statement"]:
        """Collect comments after current statement.
        This should be called after a statement is complete."""
        comments: List[Statement] = []
        # Check lexer for any remaining comments on the current line
        line_comments = self.lex.get_line_comments()
        for comment_text, comment_type in line_comments:
            # Inline comments (//) are typically "after" the statement content
            if comment_type == "//":
                comments.append(Comment(text=comment_text or "", position="after"))
        return comments

    def parse_quote_string(self) -> str:
        """Parse a quoted string, ignoring internal token-types,
        solely appending them to a return-value string.
        FIXME: check for newlines, EOF, etc."""

        # Get the opening quote, which may be single or double
        tp = self.expect_any(Tokens.DUBQUOTE, Tokens.TICK)
        rv = ""
        # And accumulate until we hit a matching closing quote
        while not self.match(tp):
            rv += self.peek().val
            self.advance()
        return rv

    def fail(self, *args, **kwargs) -> None:
        """Failure Debug Helper.
        Primarily designed to capture state, and potentially break-points, when things go wrong."""
        print(f"DEBUG: Parse error at line {self.lex.line_num}")
        print(f"DEBUG: Current line: {repr(self.lex.line)}")
        print(f"DEBUG: Current token: {self.cur}")
        print(f"DEBUG: Recent lines (last 5):")
        for i, line in enumerate(self.lex.recent_lines[-5:], start=1):
            print(f" {i}: {repr(line)}")
        print("DEBUG: End debug")
        print(self)
        NetlistParseError.throw(*args, **kwargs)

    def is_expression_starter(self, tp: Tokens) -> Optional[Tuple[Tokens, Tokens]]:
        """Indicates whether token-type `tp` is a valid expression-*starting* Token-type.
        If not, returns None.
        If so, returns a tuple of the start token-type and its paired expression-ending token-type."""

        pairs = {  # FIXME: specialize this by dialect
            Tokens.TICK: Tokens.TICK,
            Tokens.DUBQUOTE: Tokens.DUBQUOTE,
            Tokens.LBRACKET: Tokens.RBRACKET,
        }
        if tp not in pairs:
            return None
        return tp, pairs[tp]

    def parse_expr(self) -> Expr:
        """Parse an Expression
        expr0 | 'expr0' | {expr0}"""
        # Note: moves into our `EXPR` state require a `peek`/`expect` combo,
        # otherwise we can mis-understand multiplication vs comment.
        from .base import ParserState

        # Check for "expression-mode starters", and update state if we find one.
        pair = self.is_expression_starter(self.peek().tp)

        if pair is None:
            return self.parse_expr0()

        # Got an "expression-mode starter" token - update state, and then parse.
        self.state = ParserState.EXPR  # Note: this comes first
        self.expect(pair[0])
        e = self.parse_expr0()
        self.state = ParserState.PROGRAM  # Note: this comes first
        self.expect(pair[1])
        return e

    def parse_protect(self) -> StartProtectedSection:
        self.expect_any(Tokens.PROT, Tokens.PROTECT)
        self.expect(Tokens.NEWLINE)
        return StartProtectedSection()

    def parse_unprotect(self) -> EndProtectedSection:
        self.expect_any(Tokens.UNPROT, Tokens.UNPROTECT)
        self.expect(Tokens.NEWLINE)
        return EndProtectedSection()

    """ Abstract Methods """

    def are_stars_comments_now(self) -> bool:
        """Boolean indication of whether Tokens.STAR and DUBSTAR should
        currently be lexed as a comment."""
        raise NotImplementedError

    def parse_statement(self) -> Optional[Statement]:
        """Statement Parser
        Dispatches to type-specific parsers based on prioritized set of matching rules.
        Returns `None` at end."""
        raise NotImplementedError

    def parse_model(self) -> Expr:
        """Parse a Model Declaration"""
        raise NotImplementedError


def _endargs_startkwargs(s):
    """A fairly intractible test of where argument-lists end and key-valued keyword args being.
    e.g.
    a b c d=1 e=2 ... => d
    a b c \n  => \n
    a b c EOF => EOF
    a b c (d=1 e=2) ... => (
    """
    return (
        s.nxt is None
        or s.nxt.tp == Tokens.NEWLINE
        or s.nxt.tp == Tokens.EQUALS
        or s.nxt.tp == Tokens.LPAREN
    )
