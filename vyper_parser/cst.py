from typing import (
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


def pretty_print_lark(node: LarkNode):
    """
    Pretty prints the given lark node to stdout.
    """
    print(get_pretty_lark_repr(node))


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

    def visit_not(self, tree: Tree) -> ast.BoolOp:
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

    def visit_ellipsis(self, tree: Tree) -> ast.Expr:
        return ast.Ellipsis(**get_pos_kwargs(tree))

    def visit_const_none(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(None, **get_pos_kwargs(tree))

    def visit_const_true(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(True, **get_pos_kwargs(tree))

    def visit_const_false(self, tree: Tree) -> ast.Expr:
        return ast.NameConstant(False, **get_pos_kwargs(tree))
