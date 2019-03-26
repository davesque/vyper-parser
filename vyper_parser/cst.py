from typing import (
    Any,
    Union,
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


def require_node_type(node: LarkNode, typ: str) -> None:
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


class NodeVisitor:
    def visit(self, node: LarkNode) -> Any:
        if isinstance(node, Tree):
            name = node.data
        elif isinstance(node, Token):
            name = node.type
        else:
            raise Exception('Unrecognized node type')

        visitor = getattr(self, f'visit_{name}')

        return visitor(node)


class CSTVisitor(NodeVisitor):
    def visit_file_input(self, tree: Tree) -> ast.Module:
        body = [self.visit(ch) for ch in tree.children]

        # All nodes in module body must be statements
        assert all(isinstance(ch, ast.stmt) for ch in body)

        return ast.Module(body)
