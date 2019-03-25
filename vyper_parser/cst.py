import os
from typing import (
    Any,
    Union,
)

from lark import (
    Lark,
    Token,
)
from lark.indenter import Indenter
from lark.tree import Tree

from . import (
    ast,
)


LarkNode = Union[Tree, Token]


class PythonIndenter(Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = ('LPAR', 'LSQB', 'LBRACE')
    CLOSE_PAREN_types = ('RPAR', 'RSQB', 'RBRACE')
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4


parser = Lark.open(
    os.path.join(os.path.dirname(__file__), 'python3.lark'),
    parser='lalr',
    rel_to=__file__,
    postlex=PythonIndenter(),
    start='file_input',
)


def parse_python(source_code):
    return parser.parse(source_code + '\n')


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
