import ast as python_ast
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Union,
    Type,
    TypeVar,
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

from . import ast

LarkNode = Union[Tree, Token]


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


def parse_python(source_code):
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


def assert_node_type(node: LarkNode, typ: str) -> None:
    """
    Asserts that the given node was matched by a certain grammar rule.
    """
    assert get_node_type(node) == typ


def get_pretty_lark_repr(node: LarkNode, indent: int = 0) -> str:
    """
    Returns a pretty-printed string representation of a lark node and all of
    its children.
    """
    nam = node.__class__.__name__
    typ = get_node_type(node)

    indent_repr = ''
    for i in range(indent):
        indent_repr += ' |' if i % 2 else ' :'

    s = indent_repr + f'{nam}({typ})'

    if isinstance(node, Tree) and len(node.children) > 0:
        s += '\n'
        ch_indent = indent + 1

        for ch in node.children:
            s += get_pretty_lark_repr(ch, ch_indent)
    elif isinstance(node, Token):
        s += f' = {node.value}\n'
    else:
        s += '\n'

    return s


def get_pretty_python_ast_repr(val: Any,
                               indent_repr: str = '',
                               alt_indent_repr: str = '',
                               indent_sep: str = ': ') -> str:
    """
    Returns a pretty-printed string representation of a python AST.
    """
    if isinstance(val, (list, tuple)) and len(val) > 0:
        s = f'{indent_repr}['
        for i in val:
            s += '\n' + get_pretty_python_ast_repr(
                i,
                alt_indent_repr + indent_sep,
                alt_indent_repr + indent_sep,
                indent_sep,
            ) + ','
        s += f'\n{alt_indent_repr}]'
    elif isinstance(val, python_ast.AST):
        node = val
        if hasattr(node, 'lineno'):
            pos_repr = f'<line {node.lineno}, col {node.col_offset}>'
        else:
            pos_repr = ''

        fields = node._fields
        if len(fields) > 0:
            s = f'{indent_repr}{node.__class__.__name__}(  {pos_repr}'

            max_ch_len = max(len(f) for f in fields)

            for i, f in enumerate(fields):
                ch = getattr(node, f)
                ch_indent_repr = indent_repr + indent_sep + f.ljust(max_ch_len) + '='
                ch_alt_indent_repr = alt_indent_repr + indent_sep + ' ' * max_ch_len + ' '

                s += '\n' + get_pretty_python_ast_repr(
                    ch,
                    ch_indent_repr,
                    ch_alt_indent_repr,
                    indent_sep,
                ) + ','

            s += f'\n{alt_indent_repr})'
        else:
            s = f'{indent_repr}{node.__class__.__name__}  {pos_repr}'
    else:
        s = f'{indent_repr}{repr(val)}'

    return s


def pretty_print_lark(node: LarkNode) -> None:
    """
    Pretty prints the given lark node to stdout.
    """
    print(get_pretty_lark_repr(node))


def parse_and_print(source_code: str) -> None:
    """
    Parses the given source code and pretty prints the parse tree.
    """
    pretty_print_lark(parse_python(source_code))


def get_num_stmts(tree: Tree) -> int:
    """
    Returns the number of statements contained within an instance of a
    supported set of node types.
    """
    typ = get_node_type(tree)
    num_children = len(tree.children)

    if typ == 'file_input':
        n = 0
        for ch in tree.children:
            if get_node_type(ch) == 'stmt':
                n += get_num_stmts(ch)
        return n

    if typ == 'stmt':
        return get_num_stmts(tree.children[0])

    if typ == 'compound_stmt':
        return 1

    if typ == 'simple_stmt':
        return num_children

    if typ == 'suite':
        n = 0
        for ch in tree.children:
            n += get_num_stmts(ch)
        return n

    raise Exception(f'Non-statement found: {typ} {num_children}')


TAst = TypeVar('TAst', bound=ast.VyperAST)


def make_op_token_visitor(
    doc: str,
    rule_name: str,
    op_mapping: Dict[str, TAst],
) -> Callable[[Any, Tree], TAst]:
    def op_token_visitor(self, tree: Tree) -> TAst:
        token = str(tree.children[0])
        try:
            return op_mapping[token]
        except KeyError:
            raise Exception(f'Invalid {rule_name}: {token}')

    op_token_visitor.__doc__ = doc

    return op_token_visitor


TSeq = TypeVar('TSeq', Type[list], Type[tuple])


class CSTVisitor(Generic[TSeq]):
    def __init__(self, seq_class: TSeq = tuple):
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
            return ast.Expr(
                self.visit(tree.children[0]),
                **get_pos_kwargs(tree),
            )

    def visit_testlist_star_expr(self, tree: Tree) -> TSeq:
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
            return self.visit(tree.children[0])
        else:
            return ast.Tuple(
                self._visit_children(tree),
                ast.Load,
                **get_pos_kwargs(tree),
            )

    def visit_test(self, tree: Tree) -> ast.Expr:
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
            self.visit(test),
            self.visit(body),
            self.visit(orelse),
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
            self.visit(args),
            self.visit(body),
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
            self.visit(operand),
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
            self.visit(left),
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
            self.visit(value),
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
            self.visit(left),
            self.visit(op),
            self.visit(right),
            **get_pos_kwargs(tree),
        )

        num_ops = (len(tree.children) - 1) // 2
        for i in range(1, num_ops):
            next_op = tree.children[i * 2 + 1]
            next_expr = tree.children[i * 2 + 2]

            tmp_result = ast.BinOp(
                result,
                self.visit(next_op),
                self.visit(next_expr),
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
            self.visit(op),
            self.visit(operand),
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

    def visit_comp_op(self, tree: Tree) -> ast.cmpop:
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
            self.visit(left),
            ast.Pow,
            self.visit(right),
            **get_pos_kwargs(tree),
        )

    def visit_await_expr(self, tree: Tree) -> ast.Await:
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
            self.visit(value),
            **get_pos_kwargs(tree),
        )

    def visit_ellipsis(self, tree: Tree) -> ast.Expr:
        return ast.Ellipsis(**get_pos_kwargs(tree))

    def visit_const_none(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(None, **get_pos_kwargs(tree))

    def visit_const_true(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(True, **get_pos_kwargs(tree))

    def visit_const_false(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(False, **get_pos_kwargs(tree))
