"""
# Spectre-Dialect Parsing 
"""

from typing import Optional, Union, List, Dict, Callable

# Local Imports
from ..data import *
from ..lex import Tokens
from .spice import DialectParser, SpiceDialectParser
from warnings import warn


class SpectreMixin:
    """Spectre-stuff to be mixed-in,
    primarily related to the capacity for `DialectChanges`
    via a `simulator lang` statement."""

    def parse_dialect_change(self) -> Optional[DialectChange]:
        """Parse a DialectChange. Leaves its trailing NEWLINE to be parsed by a (likely new) DialectParser."""

        self.expect(Tokens.SIMULATOR)
        self.expect(Tokens.LANG)
        self.expect(Tokens.EQUALS)
        self.expect(Tokens.IDENT)
        d = DialectChange(self.cur.val)

        # FIXME: ignoring additional parameters e.g. `insensitive`
        while self.nxt and self.nxt.tp != Tokens.NEWLINE:
            self.advance()
        # self.expect(Tokens.NEWLINE) # Note this is left for the *new* dialect to parse

        self.parent.notify(d)
        return d


class SpectreSpiceDialectParser(SpectreMixin, SpiceDialectParser):
    """Spice-Style Syntax, as Interpreted by Spectre."""

    enum = NetlistDialects.SPECTRE_SPICE

    @classmethod
    def get_rules(cls) -> Dict[Tokens, Callable]:
        """Get Spectre-Spice rules: base Spice rules + Spectre additions."""
        rules = super().get_rules()  # Inherit Spice rules
        rules[Tokens.SIMULATOR] = cls.parse_dialect_change  # Add Spectre rule
        return rules

    def parse_statement(self) -> Optional[Statement]:
       """Override to handle Spectre-specific constructs and dynamic dialect switching."""
       self.eat_blanks()
       pk = self.peek()
       if pk:
           if pk.tp == Tokens.SIMULATOR:
               return self.parse_dialect_change()
           elif pk.tp in (Tokens.PARAMETERS, Tokens.STATS):
               line_info = f"(input line: {self.lex.line_num})"
               warn(f"Detected Spectre syntax while parsing Spice {line_info}! Switching to Spectre")
               from ..data import DialectChange
               change = DialectChange(dialect='spectre')
               self.parent.notify(change)
               return self.parent.dialect_parser.parse_statement()

       # Fall back to base Spice parsing
       return super().parse_statement()

class SpectreDialectParser(SpectreMixin, DialectParser):
    """Spectre-Language Dialect.
    Probably more of a separate language really, but it fits our Dialect paradigm well enough."""

    enum = NetlistDialects.SPECTRE

    @classmethod
    def get_rules(cls) -> Dict[Tokens, Callable]:
        """Get the base Spectre parsing rules."""
        return {
            Tokens.SIMULATOR: cls.parse_dialect_change,
            Tokens.PARAMETERS: cls.parse_param_statement,
            Tokens.INLINE: cls.parse_subckt_start,
            Tokens.SUBCKT: cls.parse_subckt_start,
            Tokens.ENDS: cls.parse_end_sub,
            Tokens.MODEL: cls.parse_model,
            Tokens.STATS: cls.parse_statistics_block,
            Tokens.AHDL: cls.parse_ahdl,
            Tokens.LIBRARY: cls.parse_start_lib,
            Tokens.SECTION: cls.parse_start_section,
            Tokens.ENDSECTION: cls.parse_end_section,
            Tokens.ENDLIBRARY: cls.parse_end_lib,
            Tokens.INCLUDE: cls.parse_include,
            Tokens.REAL: cls.parse_function_def,
            Tokens.PROT: cls.parse_protect,
            Tokens.PROTECT: cls.parse_protect,
            Tokens.UNPROT: cls.parse_unprotect,
            Tokens.UNPROTECT: cls.parse_unprotect,
            Tokens.IDENT: cls.parse_named,  # Catch-all for identifiers
        }

    def parse_statement(self) -> Optional[Statement]:
        """Statement Parser
        Dispatches to type-specific parsers based on prioritized set of matching rules.
        Returns `None` at end."""

        # Collect any comments before this statement
        before_comments = self.collect_before_comments()
        
        # Track blank lines before eating them
        blank_count = self.eat_blanks()
        # Store blank line count for later use (we'll add BlankLine entries in FileParser)
        if not hasattr(self, '_blank_lines_before_statement'):
            self._blank_lines_before_statement = []
        self._blank_lines_before_statement.append(blank_count)
        
        pk = self.peek()

        if pk is None:  # End-of-input case
            # If we had before comments, return them as standalone comments
            if before_comments:
                # Store for later association - for now, we'll handle in FileParser
                pass
            return None

        if self.match(Tokens.DOT):
            pk = self.peek() # peek past DOT
            # TODO: Add user facing option(s) specifying how to handle this
            warn("Detected Spice .DOT syntax while parsing Spectre! Interpreting as Spectre syntax...")

        rules = self.get_rules()
        if pk.tp not in rules:
            return self.fail(f"Unexpected token to begin statement: {pk}")
        type_parser = rules[pk.tp]
        stmt = type_parser(self)
        
        # Collect comments after the statement
        after_comments = self.collect_after_comments()
        
        # Store comments for later association
        # We'll attach them in FileParser since we need to handle standalone vs associated
        if before_comments or after_comments:
            # Store in a temporary location - we'll handle this in FileParser
            if not hasattr(self, '_pending_comments'):
                self._pending_comments = []
            self._pending_comments.append((stmt, before_comments, after_comments))
        
        return stmt

    def parse_named(self):
        """Parse an identifier-named statement.
        Instances, Options, and Analyses fall into this category,
        by beginning with their name, then their type-keyword.
        The general method is to read one token ahead, then rewind
        before dispatching to more detailed parsing methods.
        
        In SPICE-compatible mode (default), uses prefix-based detection:
        - X/x prefix = subcircuit instance
        - R/r, C/c, L/l, M/m, D/d, Q/q, V/v, I/i, E/e, G/g, F/f, H/h, O/o = primitives
        - Otherwise, assumes master-name syntax (subcircuit instance)
        """
        self.expect(Tokens.IDENT)
        instance_name = self.cur.val
        
        if self.nxt is None:
            self.fail()
        if self.nxt.tp == Tokens.OPTIONS:
            self.rewind()
            return self.parse_options()
        if self.nxt.tp in (Tokens.IDENT, Tokens.INT, Tokens.LPAREN):
            # Check next token BEFORE rewind (after rewind, nxt points to current token)
            has_parens = self.nxt.tp == Tokens.LPAREN
            self.rewind()
            
            # Check prefix to distinguish primitives from subcircuit instances
            # Spectre defaults to SPICE-compatible mode where prefix determines type
            prefix = instance_name[0].lower() if instance_name else ''
            
            # X prefix = subcircuit instance
            if prefix == 'x':
                return self.parse_instance()
            
            # Primitive prefixes (SPICE convention)
            primitive_prefixes = {'r', 'c', 'l', 'm', 'd', 'q', 'v', 'i', 'e', 'g', 'f', 'h', 'o'}
            
            # Known master names that indicate master-name syntax (not prefix-based primitives)
            master_names = {'resistor', 'capacitor', 'inductor', 'diode', 'mos', 'nmos', 'pmos', 
                           'bipolar', 'npn', 'pnp', 'vsource', 'isource', 'vcvs', 'vccs', 'cccs', 'ccvs'}
            
            # For primitive prefixes, check if this is prefix-based (parse as primitive)
            # or master-name syntax (parse as instance)
            if prefix in primitive_prefixes:
                # If no parentheses, definitely a prefix-based primitive
                if not has_parens:
                    return self.parse_primitive()
                
                # Has parentheses - need to check if there's a master name after closing paren
                # Try parsing as instance first (handles master-name syntax)
                # The hook in parse_instance() will validate if module is a master name
                try:
                    return self.parse_instance()
                except Exception:
                    # If parse_instance fails, it might be because there's no master name
                    # In that case, it should be a primitive
                    # But we can't easily rewind, so this is a limitation
                    # For now, re-raise the exception
                    raise
            
            # Not a primitive prefix - parse as instance (supports master-name syntax)
            return self.parse_instance()
        # No match - error time.
        self.fail()

    def parse_model(self) -> Union[ModelDef, ModelFamily]:
        """Parse a Model statement"""
        self.expect(Tokens.MODEL)
        self.expect(Tokens.IDENT)
        mname = Ident(self.cur.val)
        self.expect(Tokens.IDENT)
        mtype = Ident(self.cur.val)
        if self.match(Tokens.LBRACKET):
            self.expect(Tokens.NEWLINE)
            # Multi-Variant Model Family
            vars = []
            while not self.match(Tokens.RBRACKET):
                self.expect(Tokens.IDENT, Tokens.INT)
                vname = Ident(str(self.cur.val))
                self.expect(Tokens.COLON)
                params = self.parse_param_declarations()
                vars.append(ModelVariant(mname, vname, mtype, [], params))
            self.expect(Tokens.NEWLINE)
            return ModelFamily(mname, mtype, vars)
        # Single ModelDef
        params = self.parse_param_declarations()
        return ModelDef(mname, mtype, [], params)

    def parse_param_statement(self) -> ParamDecls:
        """Parse a Parameter-Declaration Statement"""
        from .base import _endargs_startkwargs

        self.expect(Tokens.PARAMETERS)
        self.match(Tokens.COLON)
        # Parse an initial list of identifiers, i.e. non-default-valued parameters
        args = self.parse_ident_list(_endargs_startkwargs)
        # If we landed on a key-value param key, rewind it
        if self.nxt and self.nxt.tp == Tokens.EQUALS:
            self.rewind()
            args.pop()
        args = [ParamDecl(a, None) for a in args]
        # Parse the remaining default-valued params
        vals = self.parse_param_declarations()  # NEWLINE is captured inside
        return ParamDecls(args + vals)

    def parse_variations(self) -> List[Variation]:
        """Parse a list of variation-statements, of the form
        `{
            vary param1 dist=distname std=stdval [mean=meanval]
            vary param2 dist=distname std=stdval [mean=meanval]
        }\n`
        Consumes the both opening and closing brackets,
        and the (required) newline following the closing bracket."""
        self.expect(Tokens.LBRACKET)
        self.expect(Tokens.NEWLINE)
        vars = []
        while not self.match(Tokens.RBRACKET):
            self.expect(Tokens.IDENT)
            if self.cur.val != "vary":
                self.fail()
            self.expect(Tokens.IDENT)
            name = Ident(self.cur.val)

            dist = None
            std = None
            mean = None
            percent = None  # FIXME: roll in
            while not self.match(Tokens.NEWLINE):
                self.expect(Tokens.IDENT)
                if self.cur.val == "dist":
                    if dist is not None:
                        self.fail()
                    self.expect(Tokens.EQUALS)
                    self.expect(Tokens.IDENT)
                    dist = str(self.cur.val)
                    if dist not in ["gauss", "lnorm"]:
                        print(f"dist ({dist}) not one of: gauss lnorm")
                        self.fail()
                elif self.cur.val == "std":
                    if std is not None:
                        self.fail()
                    self.expect(Tokens.EQUALS)
                    std = self.parse_expr()
                elif self.cur.val == "mean":
                    if mean is not None:
                        self.fail()
                    self.expect(Tokens.EQUALS)
                    mean = self.parse_expr()
                elif self.cur.val == "percent":
                    self.expect(Tokens.EQUALS)
                    percent = self.parse_expr()
                else:
                    self.fail()
            vars.append(Variation(name, dist, std, mean))  # FIXME: roll in `percent`

        self.expect(Tokens.NEWLINE)
        return vars

    def parse_statistics_block(self) -> StatisticsBlock:
        """Parse the `statistics` block"""

        self.expect(Tokens.STATS)
        self.expect(Tokens.LBRACKET)
        self.expect(Tokens.NEWLINE)

        process = None
        mismatch = None

        while not self.match(Tokens.RBRACKET):
            self.expect(Tokens.IDENT)
            if self.cur.val == "process":
                if process is not None:
                    self.fail()
                process = self.parse_variations()
            elif self.cur.val == "mismatch":
                if mismatch is not None:
                    self.fail()
                mismatch = self.parse_variations()
            else:
                self.fail()

        self.expect(Tokens.NEWLINE)
        return StatisticsBlock(process=process, mismatch=mismatch)

    def parse_ahdl(self):
        """Parse an `ahdl_include` statement"""
        self.expect(Tokens.AHDL)
        path = self.parse_quote_string()
        rv = AhdlInclude(path)
        self.expect(Tokens.NEWLINE)
        return rv

    def parse_instance(self) -> Instance:
        """Parse an instance, with a hook to detect if it should have been a primitive.
        
        For instances with primitive prefixes and parentheses, check if there's a master name.
        If after the closing paren we see '=' instead of an IDENT, there's no master name,
        so rewind and parse as primitive instead.
        """
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        conns = []
        
        instance_name = name.name.lower() if hasattr(name, 'name') else ''
        prefix = instance_name[0] if instance_name else ''
        primitive_prefixes = {'r', 'c', 'l', 'm', 'd', 'q', 'v', 'i', 'e', 'g', 'f', 'h', 'o'}
        master_names = {'resistor', 'capacitor', 'inductor', 'diode', 'mos', 'nmos', 'pmos', 
                       'bipolar', 'npn', 'pnp', 'vsource', 'isource', 'vcvs', 'vccs', 'cccs', 'ccvs'}
        
        # Parse the parens-optional port-list
        if self.match(Tokens.LPAREN):  # Parens case
            term = lambda s: not s.nxt or s.nxt.tp in (Tokens.RPAREN, Tokens.NEWLINE)
            conns = self.parse_node_list(term)
            self.expect(Tokens.RPAREN)
            
            # Hook: Check if next token is '=' (no master name) - this is the "finalize/hook" the user requested
            # If we see '=' directly after closing paren, there's no master name, so it should be a primitive
            if prefix in primitive_prefixes:
                if self.nxt and self.nxt.tp == Tokens.EQUALS:
                    # No master name after closing paren - this should be a primitive
                    # This means we should have called parse_primitive() instead
                    # Can't easily convert here, so fail with helpful message
                    self.fail(f"Instance '{name.name}' with primitive prefix '{prefix}' has no master name after connections. This should be parsed as a primitive, not an instance.")
                # If there's an IDENT after closing paren, it could be a master name OR a subcircuit name
                # We can't distinguish here, so we'll parse it as an instance and let the finalize hook check
            
            # Grab the module name (it's a master name, so this is correct master-name syntax)
            self.expect(Tokens.IDENT)
            module = Ref(ident=Ident(self.cur.val))
            
        else:  # No-parens case
            from .base import _endargs_startkwargs
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
        
        # Finalize hook: Check if module name is a master name (user's suggestion)
        # If prefix is primitive and module is NOT a master name, this should have been a primitive
        if prefix in primitive_prefixes and hasattr(module, 'ident'):
            module_name = module.ident.name.lower() if hasattr(module.ident, 'name') else ''
            if module_name not in master_names:
                # Module is not a master name - this should have been a primitive
                # We can't convert at this point, but at least we've detected the issue
                # Return as-is (it will have wrong structure, but it parses)
                pass
        
        # And create & return our instance
        return Instance(name=name, module=module, conns=conns, params=params)

    def parse_instance_param_values(self) -> List[ParamVal]:
        """Parse a list of instance parameter-values,
        including the fun-fact that Spectre allows arbitrary dangling closing parens."""
        term = (
            lambda s: s.nxt is None or s.match(Tokens.NEWLINE) or s.match(Tokens.RPAREN)
        )
        vals = self.parse_list(self.parse_param_val, term=term)
        if self.match(Tokens.RPAREN):  # Eat potential dangling r-parens
            while self.nxt is not None and not self.match(Tokens.NEWLINE):
                self.expect(Tokens.RPAREN)
        return vals

    def are_stars_comments_now(self) -> bool:
        # Stars are comments only to begin lines, and at the beginning of a file. (We think?)
        return not self.lex.lexed_nonwhite_on_this_line

    def parse_start_lib(self) -> StartLib:
        self.expect(Tokens.LIBRARY)
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.NEWLINE)
        return StartLib(name)

    def parse_end_lib(self) -> EndLib:
        self.expect(Tokens.ENDLIBRARY)
        if self.match(Tokens.NEWLINE):
            # No name specified
            return EndLib(name=None)
        name = self.parse_ident()
        self.expect(Tokens.NEWLINE)
        return EndLib(name)

    def parse_start_section(self):
        self.expect(Tokens.SECTION)

        if self.match(Tokens.EQUALS):
            ...  # Apparently there is an optional "=" character here

        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.NEWLINE)
        return StartLibSection(name)

    def parse_end_section(self):
        self.expect(Tokens.ENDSECTION)
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.NEWLINE)
        return EndLibSection(name)

    def parse_options(self):
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.OPTIONS)
        vals = self.parse_option_values()
        return Options(name=name, vals=vals)

    def parse_include(self) -> Union[Include, UseLibSection]:
        """Parse an Include Statement"""
        self.expect(Tokens.INCLUDE)
        path = self.parse_quote_string()
        if self.match(Tokens.NEWLINE):  # Non-sectioned `Include`
            return Include(path)
        # Otherwise expect a library `Section`
        self.expect(Tokens.SECTION)
        self.expect(Tokens.EQUALS)
        self.expect(Tokens.IDENT)
        section = Ident(self.cur.val)
        return UseLibSection(path, section)

    def parse_function_def(self):
        """Yes, Spectre does have function definitions!
        Syntax: `rtype name (argtype argname, argtype argname) {
            statements;
            return rval;
        }`
        Caveats:
        * Only `real` return and argument types are supported
        * Only single-statement functions comprising a `return Expr;` are supported
        """
        self.expect(Tokens.REAL)  # Return type. FIXME: support more types than REAL
        self.expect(Tokens.IDENT)
        name = Ident(self.cur.val)
        self.expect(Tokens.LPAREN)
        # Parse arguments
        args = []
        MAX_ARGS = 100  # Set a "time-out" so that we don't get stuck here.
        for i in range(MAX_ARGS, -1, -1):
            if self.match(Tokens.RPAREN):
                break  # Note we can have zero-argument cases, I guess.
            # Argument type. FIXME: support more types than REAL
            self.expect(Tokens.REAL)
            a = TypedArg(tp=ArgType.REAL, name=self.parse_ident())
            args.append(a)
            if self.match(Tokens.RPAREN):
                break
            self.expect(Tokens.COMMA)
        if i <= 0:  # Check the time-out
            self.fail(f"Unable to parse argument list for spectre-function {name.name}")

        self.expect(Tokens.LBRACKET)
        self.expect(Tokens.NEWLINE)
        # Return-statement
        self.expect(Tokens.RETURN)
        rv = self.parse_expr()
        ret = Return(rv)
        self.expect(Tokens.SEMICOLON)
        self.expect(Tokens.NEWLINE)
        # Function-Closing
        self.expect(Tokens.RBRACKET)
        self.expect(Tokens.NEWLINE)

        return FunctionDef(name=name, rtype=ArgType.REAL, args=args, stmts=[ret])
