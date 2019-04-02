import ast as python_ast
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Type,
    TypeVar,
    cast,
)

from lark import (
    Lark,
    Token,
)
from lark.indenter import (
    Indenter,
)
from lark.tree import (
    Tree,
)

from vyper_parser import (
    ast,
)
from vyper_parser.types import (
    LarkNode,
)


class PythonIndenter(Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = ('LPAR', 'LSQB', 'LBRACE')
    CLOSE_PAREN_types = ('RPAR', 'RSQB', 'RBRACE')
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4


parser = Lark.open(
    'python3.lark',
    parser='lalr',
    rel_to=__file__,
    postlex=PythonIndenter(),
    start='file_input',
    propagate_positions=True,
)


def parse_python(source_code: str) -> Tree:
    return parser.parse(source_code + '\n')


def get_node_type(node: LarkNode) -> str:
    """
    Returns the node type (name of matching grammar rule) for the given lark
    node.
    """
    if isinstance(node, Tree):
        return node.data
    elif isinstance(node, Token):
        return node.type
    else:
        raise Exception('Unrecognized node type')


def get_pos_kwargs(node: LarkNode) -> Dict[str, int]:
    """
    Returns a dictionary containing positional kwargs based on the parsing
    position of ``node``.
    """
    return {
        'lineno': node.line,
        'col_offset': node.column - 1,
    }


def get_str_ast(tree: Tree) -> ast.VyperAST:
    """
    Converts a lark "atom" rule match tree into a vyper AST representation.
    """
    s_repr = '(\n'

    for ch in tree.children:
        assert isinstance(ch, Tree)
        assert ch.data == 'string'

        str_token = ch.children[0]
        s_repr += str(str_token) + '\n'

    s_repr += ')'

    module_node = python_ast.parse(s_repr)
    str_node = module_node.body[0].value
    vyper_node = ast.VyperAST.from_python_ast(str_node)

    # Reset parsing position to the position of the parsed lark node
    ast.translate_parsing_pos(vyper_node, -vyper_node.lineno, -vyper_node.col_offset)
    ast.translate_parsing_pos(vyper_node, tree.line, tree.column)

    return vyper_node


TAst = TypeVar('TAst', Type[ast.unaryop], Type[ast.operator])


def make_op_token_visitor(
    doc: str,
    rule_name: str,
    op_mapping: Dict[str, TAst],
) -> Callable[[Any, Tree], TAst]:
    def op_token_visitor(self: 'CSTVisitor', tree: Tree) -> TAst:
        token = str(tree.children[0])
        try:
            return op_mapping[token]
        except KeyError:
            raise Exception(f'Invalid {rule_name}: {token}')

    op_token_visitor.__doc__ = doc

    return op_token_visitor


TSeq = TypeVar('TSeq', list, tuple)


class CSTVisitor(Generic[TSeq]):
    seq_class: Type[TSeq]

    def __init__(self, seq_class: Type[TSeq]):
        self.seq_class = seq_class

    def visit(self, node: LarkNode) -> ast.VyperAST:
        typ = get_node_type(node)
        visitor = getattr(self, f'visit_{typ}')

        return visitor(node)

    def _visit_children(self, tree: Tree) -> TSeq:
        return self.seq_class(self.visit(ch) for ch in tree.children)

    def visit_file_input(self, tree: Tree) -> ast.Module:
        """
        file_input: (_NEWLINE | stmt)*

        Analogous to:
        PyAST_FromNodeObject
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L789-L817)
        """
        return ast.Module(self._visit_children(tree))

    def visit_expr_stmt(self, tree: Tree) -> ast.stmt:
        """
        expr_stmt: testlist_star_expr (annassign | augassign (yield_expr|testlist)
                 | ("=" (yield_expr|testlist_star_expr))*)
        annassign: ":" test ["=" test]
        testlist_star_expr: (test|star_expr) ("," (test|star_expr))* [","]
        !augassign: ("+=" | "-=" | "*=" | "@=" | "/=" | "%=" | "&=" | "|="
                  | "^=" | "<<=" | ">>=" | "**=" | "//=")
        ?test: ... here starts the operator precedence dance

        Analogous to:
        ast_for_expr_stmt
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2902)

        Parses statements like:
        * ``True``
        * ``True, False``
        * ``x: int = 5``
        * ``x += 1``
        * ``x, y = 1, 2``
        * ``x = y = 1``
        """
        if len(tree.children) == 1:
            # If only one child, pass through to testlist_star_expr match
            return ast.Expr(
                cast(ast.expr, self.visit(tree.children[0])),
                **get_pos_kwargs(tree),
            )
        else:
            # TODO: This return statement is just a placeholder to satisfy
            # mypy.  Need to implement the rest of this CST transformer.
            return ast.Expr(
                cast(ast.expr, self.visit(tree.children[0])),
                **get_pos_kwargs(tree),
            )

    AUGASSIGN_OPS = {
        '+=': ast.Add,
        '-=': ast.Sub,
        '*=': ast.Mult,
        '@=': ast.MatMult,
        '/=': ast.Div,
        '%=': ast.Mod,
        '&=': ast.BitAnd,
        '|=': ast.BitOr,
        '^=': ast.BitXor,
        '<<=': ast.LShift,
        '>>=': ast.RShift,
        '**=': ast.Pow,
        '//=': ast.FloorDiv,
    }
    visit_augassign = make_op_token_visitor(
        """
        !augassign: ("+=" | "-=" | "*=" | "@=" | "/=" | "%=" | "&=" | "|=" | "^="
            | "<<=" | ">>=" | "**=" | "//=")

        Analogous to:
        ast_for_augassign
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L1116)
        """,
        'augassign',
        AUGASSIGN_OPS,
    )

    def visit_testlist_star_expr(self, tree: Tree) -> ast.VyperAST:
        """
        testlist_star_expr: (test|star_expr) ("," (test|star_expr))* [","]
        ?test: ... here starts the operator precedence dance
        star_expr: "*" expr
        expr: ... half of the operator precedence dance starts here

        Analogous to:
        ast_for_testlist
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2887-L2898)

        Parses statements like:
        * ``True``
        * ``True, False``
        * ``1, 2``
        """
        if len(tree.children) == 1:
            # If only one child, pass through to test or star_expr match
            return self.visit(tree.children[0])
        else:
            # Otherwise, this match is a comma-delimited tuple
            return ast.Tuple(
                self._visit_children(tree),
                ast.Load,
                **get_pos_kwargs(tree),
            )

    def visit_test(self, tree: Tree) -> ast.IfExp:
        """
        ?test: or_test ["if" or_test "else" test] | lambdef

        Analogous to:
        ast_for_ifexpr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L1750)

        Since this grammar rule has the '?' prefix, it will be filtered out of
        the parse tree whenever it is *not* matching a ternary expression i.e.
        when it has only one child.  Therefore, this rule visitor only handles
        the ternary case.
        """
        body, test, orelse = tree.children

        return ast.IfExp(
            cast(ast.expr, self.visit(test)),
            cast(ast.expr, self.visit(body)),
            cast(ast.expr, self.visit(orelse)),
            **get_pos_kwargs(tree),
        )

    def visit_lambdef(self, tree: Tree) -> ast.Lambda:
        """
        lambdef: "lambda" [varargslist] ":" test
        lambdef_nocond: "lambda" [varargslist] ":" test_nocond

        Analogous to:
        ast_for_lambdef
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L1722)
        """
        if len(tree.children) == 1:
            # Lambda with no args
            args, body = None, tree.children[0]
        else:
            args, body = tree.children

        return ast.Lambda(
            cast(ast.arguments, self.visit(args)),
            cast(ast.expr, self.visit(body)),
            **get_pos_kwargs(tree),
        )

    visit_lambdef_nocond = visit_lambdef

    def visit_or_test(self, tree: Tree) -> ast.BoolOp:
        """
        ?or_test: and_test ("or" and_test)*

        Analogous to:
        ast_for_expr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2580-L2599)
        """
        return ast.BoolOp(
            ast.Or,
            self._visit_children(tree),
            **get_pos_kwargs(tree),
        )

    def visit_and_test(self, tree: Tree) -> ast.BoolOp:
        """
        ?and_test: not_test ("and" not_test)*

        Analogous to:
        ast_for_expr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2580-L2599)
        """
        return ast.BoolOp(
            ast.And,
            self._visit_children(tree),
            **get_pos_kwargs(tree),
        )

    def visit_not(self, tree: Tree) -> ast.UnaryOp:
        """
        ?not_test: "not" not_test -> not
                 | comparison

        Analogous to:
        ast_for_expr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2600-L2612)
        """
        operand = tree.children[0]

        return ast.UnaryOp(
            ast.Not,
            cast(ast.expr, self.visit(operand)),
            **get_pos_kwargs(tree),
        )

    def visit_comparison(self, tree: Tree) -> ast.Compare:
        """
        ?comparison: expr (comp_op expr)*

        Analogous to:
        ast_for_expr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2613-L2653)
        """
        left = tree.children[0]
        ops = []
        comparators = []

        for i in range(1, len(tree.children), 2):
            ops.append(tree.children[i])
            comparators.append(tree.children[i + 1])

        return ast.Compare(
            cast(ast.expr, self.visit(left)),
            self.seq_class(self.visit(op) for op in ops),
            self.seq_class(self.visit(cmp) for cmp in comparators),
            **get_pos_kwargs(tree),
        )

    def visit_star_expr(self, tree: Tree) -> ast.Starred:
        """
        star_expr: "*" expr

        Analogous to:
        ast_for_starred
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2528)
        """
        value = tree.children[0]

        return ast.Starred(
            cast(ast.expr, self.visit(value)),
            ast.Load,
            **get_pos_kwargs(tree),
        )

    def _visit_bin_op(self, tree: Tree) -> ast.BinOp:
        """
        ?expr: xor_expr ("|" xor_expr)*
        ?xor_expr: and_expr ("^" and_expr)*
        ?and_expr: shift_expr ("&" shift_expr)*
        ?shift_expr: arith_expr (shift_op arith_expr)*

        Analogous to:
        ast_for_binop
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2310)
        """
        left, op, right = tree.children[:2]

        result = ast.BinOp(
            cast(ast.expr, self.visit(left)),
            cast(Type[ast.operator], self.visit(op)),
            cast(ast.expr, self.visit(right)),
            **get_pos_kwargs(tree),
        )

        num_ops = (len(tree.children) - 1) // 2
        for i in range(1, num_ops):
            next_op = tree.children[i * 2 + 1]
            next_expr = tree.children[i * 2 + 2]

            tmp_result = ast.BinOp(
                result,
                cast(Type[ast.operator], self.visit(next_op)),
                cast(ast.expr, self.visit(next_expr)),
                **get_pos_kwargs(next_op),
            )

            result = tmp_result

        return result

    visit_expr = _visit_bin_op
    visit_xor_expr = _visit_bin_op
    visit_and_expr = _visit_bin_op
    visit_shift_expr = _visit_bin_op
    visit_arith_expr = _visit_bin_op
    visit_term = _visit_bin_op

    def visit_factor(self, tree: Tree) -> ast.UnaryOp:
        """
        ?factor: factor_op factor | power
        !factor_op: "+"|"-"|"~"

        Analogous to:
        ast_for_factor
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2438)
        """
        op, operand = tree.children

        return ast.UnaryOp(
            cast(Type[ast.unaryop], self.visit(op)),
            cast(ast.expr, self.visit(operand)),
            **get_pos_kwargs(tree),
        )

    FACTOR_OPS = {
        '+': ast.UAdd,
        '-': ast.USub,
        '~': ast.Invert,
    }
    ADD_OPS = {
        '+': ast.Add,
        '-': ast.Sub,
    }
    SHIFT_OPS = {
        '<<': ast.LShift,
        '>>': ast.RShift,
    }
    MUL_OPS = {
        '*': ast.Mult,
        '@': ast.MatMult,
        '/': ast.Div,
        '%': ast.Mod,
        '//': ast.FloorDiv,
    }
    OP_TOKEN_DOCS = """
    !factor_op: "+"|"-"|"~"
    !add_op: "+"|"-"
    !shift_op: "<<"|">>"
    !mul_op: "*"|"@"|"/"|"%"|"//"

    Analogous to:
    ast_for_factor
    (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2438)
    --and--
    get_operator
    (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L898)
    """
    visit_factor_op = make_op_token_visitor(OP_TOKEN_DOCS, 'factor_op', FACTOR_OPS)
    visit_add_op = make_op_token_visitor(OP_TOKEN_DOCS, 'add_op', ADD_OPS)
    visit_shift_op = make_op_token_visitor(OP_TOKEN_DOCS, 'shift_op', SHIFT_OPS)
    visit_mul_op = make_op_token_visitor(OP_TOKEN_DOCS, 'mul_op', MUL_OPS)

    COMP_OPS = {
        ('<',): ast.Lt,
        ('>',): ast.Gt,
        ('==',): ast.Eq,
        ('<=',): ast.LtE,
        ('>=',): ast.GtE,
        ('!=',): ast.NotEq,
        ('in',): ast.In,
        ('is',): ast.Is,
        ('not', 'in'): ast.NotIn,
        ('is', 'not'): ast.IsNot,
    }

    def visit_comp_op(self, tree: Tree) -> Type[ast.cmpop]:
        """
        !comp_op: "<"|">"|"=="|">="|"<="|"<>"|"!="|"in"|"not" "in"|"is"|"is" "not"

        Analogous to:
        ast_for_comp_op
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L1156)
        """
        num_children = len(tree.children)
        tokens = tuple(str(token) for token in tree.children)

        if num_children not in (1, 2):
            raise Exception(f'Invalid comp_op: {tokens} has {num_children} children')
        try:
            return self.COMP_OPS[tokens]
        except KeyError:
            raise Exception(f'Invalid comp_op: {tokens}')

    def visit_power(self, tree: Tree) -> ast.BinOp:
        """
        ?power: await_expr ["**" factor]

        Analogous to:
        ast_for_power
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2507)
        """
        left, right = tree.children

        return ast.BinOp(
            cast(ast.expr, self.visit(left)),
            ast.Pow,
            cast(ast.expr, self.visit(right)),
            **get_pos_kwargs(tree),
        )

    def visit_await_expr(self, tree: Tree) -> ast.VyperAST:
        """
        await_expr: AWAIT? atom_expr
        AWAIT: "await"

        Analogous to:
        ast_for_atom_expr
        (https://github.com/python/cpython/blob/v3.6.8/Python/ast.c#L2463)
        """
        if len(tree.children) == 1:
            # Is just an atom_expr with no "await" keyword
            atom_expr = tree.children[0]
            return self.visit(atom_expr)

        # Otherwise, has "await" keyword
        await_str, value = tree.children
        assert str(await_str) == 'await'

        return ast.Await(
            cast(ast.expr, self.visit(value)),
            **get_pos_kwargs(tree),
        )

    def visit_ellipsis(self, tree: Tree) -> ast.Ellipsis:
        return ast.Ellipsis(**get_pos_kwargs(tree))

    def visit_const_none(self, tree: Tree) -> ast.NameConstant:
        return ast.NameConstant(None, **get_pos_kwargs(tree))

    def visit_const_true(self, tree: Tree) -> ast.NameConstant:
        return ast.NameConstant(True, **get_pos_kwargs(tree))

    def visit_const_false(self, tree: Tree) -> ast.NameConstant:
        return ast.NameConstant(False, **get_pos_kwargs(tree))
