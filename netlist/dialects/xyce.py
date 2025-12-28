"""
# Xyce-Dialect Parsing (Discovery-grade)

This parser is intentionally minimal:
- It is only required to parse the Xyce we emit from our own Xyce netlister.
- It focuses on subckt/model/param/include/instances for discovery & classification.
- It does NOT aim to semantically parse analysis/print/measure/control constructs.
"""

from ..data import (
    NetlistDialects,
    StartSubckt,
    Ident,
    Unknown,
    FunctionDef,
    Return,
    TypedArg,
    ArgType,
)
from ..lex import Tokens
from .spice import SpiceDialectParser
from .base import _endargs_startkwargs


class XyceDialectParser(SpiceDialectParser):
    """Xyce-like SPICE dialect.

    Key differences vs the base Spice parser that matter for our emitted decks:
    - `PARAMS:` keyword can appear in `.SUBCKT` headers and `X...` subckt instances.
    - Curly braces `{ ... }` are used for expression grouping (same precedence as parentheses).
    """

    enum = NetlistDialects.XYCE

    def parse_statement(self):
        """Xyce statement parser (discovery-grade).

        - Parses core structural constructs we need for discovery.
        - For unsupported dot-statements, stores them as `Unknown` instead of failing.
        - Adds support for `.func ... {expr}` (common in our generated Xyce model decks).
        """
        self.eat_blanks()
        pk = self.peek()

        if pk is None:
            return None

        if self.match(Tokens.DOT):
            pk = self.peek()
            if pk is None:
                return None

            # `.func name(args) {expr}` (Xyce)
            if pk.tp == Tokens.IDENT and str(pk.val).lower() == "func":
                return self.parse_xyce_func()

            rules = self.get_rules()
            if pk.tp in rules:
                type_parser = rules[pk.tp]
                return type_parser(self)

            # Any other dot-statement: keep as Unknown and continue.
            txt = "."
            while self.nxt and self.nxt.tp != Tokens.NEWLINE:
                self.advance()
                txt += str(self.cur.val)
            self.expect(Tokens.NEWLINE)
            return Unknown(txt=txt)

        # `.parameters` / `parameters` keyword
        if pk.tp == Tokens.PARAMETERS:
            return self.parse_param_statement()

        # Instances / primitives
        if pk.tp == Tokens.IDENT:
            if str(pk.val).lower().startswith("x"):
                return self.parse_instance()
            return self.parse_primitive()

        return self.fail(f"Unexpected token to begin statement: {pk}")

    def parse_xyce_func(self) -> FunctionDef:
        """.func fname(arg1,arg2,...) {expr}"""
        # Consume IDENT 'func'
        self.expect(Tokens.IDENT)
        if str(self.cur.val).lower() != "func":
            self.fail(f"Expected .func, got .{self.cur.val}")

        name = self.parse_ident()
        self.expect(Tokens.LPAREN)

        # Parse arguments (commas optional in practice; tolerate whitespace-separated too)
        args = []
        MAX_ARGS = 500
        for i in range(MAX_ARGS, -1, -1):
            if self.match(Tokens.RPAREN):
                break
            a = self.parse_ident()
            args.append(TypedArg(tp=ArgType.UNKNOWN, name=a))
            if self.match(Tokens.RPAREN):
                break
            # Optional comma separator
            self.match(Tokens.COMMA)
        if i <= 0:
            self.fail(f"Unable to parse argument list for xyce .func {name.name}")

        # Function body is typically in braces: {expr}
        if self.match(Tokens.LBRACKET):
            body = self.parse_expr()
            self.expect(Tokens.RBRACKET)
        else:
            # Fallback: allow a single expression token sequence (rare in our decks).
            body = self.parse_expr()

        self.expect(Tokens.NEWLINE)
        return FunctionDef(name=name, rtype=ArgType.UNKNOWN, args=args, stmts=[Return(body)])

    def is_comment(self, tok) -> bool:
        """Xyce comment handling.

        Our emitted Xyce decks use leading ';' comments.
        Treat ';' as a line comment starter in PROGRAM state.
        """
        if tok.tp == Tokens.SEMICOLON and self.are_stars_comments_now():
            return True
        return super().is_comment(tok)

    def parse_subckt_start(self) -> StartSubckt:
        """.subckt <name> <ports...> [PARAMS:] <p1=... p2=...>

        Xyce uses `PARAMS:` (often uppercase) to introduce subckt parameters.
        Our lexer tokenizes `PARAMS:` as `Tokens.PARAMS_COLON`.
        """
        _inline = self.match(Tokens.INLINE)
        self.expect(Tokens.SUBCKT)

        # Name
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)

        # Ports (paren-less in our emitted decks)
        # Stop before `PARAMS:` or before `k=v` style params (EQUALS lookahead).
        term = lambda s: (
            s.nxt is None
            or s.nxt.tp == Tokens.NEWLINE
            or s.nxt.tp == Tokens.PARAMS_COLON
            or _endargs_startkwargs(s)
        )
        ports = self.parse_node_list(term)

        # Optional PARAMS:
        if self.match(Tokens.PARAMS_COLON):
            pass

        # Params until end-of-line
        params = self.parse_param_declarations()
        return StartSubckt(name=name, ports=ports, params=params, inline=_inline)

    def parse_instance(self):
        """Parse Xyce-style subckt instances.

        Xyce allows an explicit `PARAMS:` marker before instance parameters:
          X1 a b mysub PARAMS: foo=1

        This overrides the base implementation so that:
        - connection parsing stops if it encounters `PARAMS:`
        - `PARAMS:` is optional (we still accept bare `k=v` pairs)
        """
        from ..data import Instance, Ref

        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)

        # Parse positional node connections until we hit module name.
        # Terminate on PARAMS: too, because it can appear right after module name
        # and must not be consumed as a node identifier.
        term = lambda s: (
            s.nxt is None
            or s.nxt.tp in (Tokens.NEWLINE, Tokens.PARAMS_COLON)
            or s.nxt.tp == Tokens.EQUALS
            or s.nxt.tp == Tokens.LPAREN
        )
        conns = self.parse_node_list(term)

        # If we landed on a key-value param key, rewind it (same as base behavior).
        if self.nxt and self.nxt.tp == Tokens.EQUALS:
            self.rewind()
            if not len(conns):
                self.fail()
            conns.pop()

        # Module name is the last positional token we parsed into conns.
        if not conns:
            self.fail("Xyce instance missing module reference")
        module = Ref(ident=conns.pop())

        # Instance params (optionally preceded by PARAMS:)
        params = self.parse_instance_param_values()
        return Instance(name=name, module=module, conns=conns, params=params)

    def parse_instance_param_values(self):
        """Support optional `PARAMS:` marker before instance param values."""
        if self.match(Tokens.PARAMS_COLON):
            # Now parse k=v pairs
            return self.parse_param_values()
        return super().parse_instance_param_values()

    def parse_expr3(self):
        """( expr ) | { expr } | term"""
        if self.match(Tokens.LPAREN):
            e = self.parse_expr()
            self.expect(Tokens.RPAREN)
            return e
        if self.match(Tokens.LBRACKET):
            e = self.parse_expr()
            self.expect(Tokens.RBRACKET)
            return e
        return self.parse_term()


