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

    def _visit_first_child(self, tree: Tree) -> ast.VyperAST:
        return self.visit(tree.children[0])

    def _visit_children(self, tree: Tree) -> TSeq:
        return self.seq_class(self.visit(ch) for ch in tree.children)

    def _visit_first_child_or_children(self, tree: Tree) -> Union[ast.VyperAST, TSeq]:
        if len(tree.children) == 1:
            return self._visit_first_child(tree)
        else:
            return self._visit_children(tree)

    def visit_file_input(self, tree: Tree) -> ast.Module:
        return ast.Module(self._visit_children(tree))

    # def visit_simple_stmt(self, tree: Tree) -> TSeq:
    #     return self._visit_children(tree)

    # def visit_expr_stmt(self, tree: Tree) -> TSeq:
    #     return self._visit_children(tree)

    # def visit_compound_stmt(self, tree: Tree) -> ast.stmt:
    #     return self.visit_first_child(tree)

    # def visit_testlist(self, tree: Tree) -> Union[ast.expr, TSeq]:
    #     return self._visit_first_child_or_children(tree)

    # def visit_testlist_star_expr(self, tree: Tree) -> Union[ast.expr, TSeq]:
    #     return self._visit_first_child_or_children(tree)

    # def visit_test(self, tree: Tree) -> ast.BinOp:
    #     return ast.IfExp(
    #         self.visit(tree.children[1]),
    #         self.visit(tree.children[0]),
    #         self.visit(tree.children[2]),
    #     )

    # def visit_expr(self, tree: Tree) -> ast.BinOp:
    #     return ast.BinOp(ast.BitOr, self._visit_children(tree))

    # def visit_xor_expr(self, tree: Tree) -> ast.BinOp:
    #     return ast.BinOp(ast.BitXor, self._visit_children(tree))

    # def visit_and_expr(self, tree: Tree) -> ast.BinOp:
    #     return ast.BinOp(ast.BitAnd, self._visit_children(tree))

    def visit_ellipsis(self, tree: Tree) -> ast.Expr:
        pos = get_pos_kwargs(tree)
        return ast.Expr(ast.Ellipsis(**pos), **pos)

    def visit_const_none(self, tree: Tree) -> ast.Expr:
        pos = get_pos_kwargs(tree)
        return ast.Expr(ast.NameConstant(None, **pos), **pos)

    def visit_const_true(self, tree: Tree) -> ast.Expr:
        pos = get_pos_kwargs(tree)
        return ast.Expr(ast.NameConstant(True, **pos), **pos)

    def visit_const_false(self, tree: Tree) -> ast.Expr:
        pos = get_pos_kwargs(tree)
        return ast.Expr(ast.NameConstant(False, **pos), **pos)
