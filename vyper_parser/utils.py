import ast as python_ast
from typing import (
    Any,
)

from lark import (
    Token,
)
from lark.tree import (
    Tree,
)

from vyper_parser import (
    ast as vyper_ast,
)
from vyper_parser.types import (
    LarkNode,
)


def get_pretty_lark_repr(node: LarkNode,
                         indent_lvl: int = 0,
                         indent_repr: str = '') -> str:
    """
    Returns a pretty-printed string representation of a lark node and all of
    its children.
    """
    from vyper_parser.cst import get_node_type

    nam = node.__class__.__name__
    typ = get_node_type(node)

    s = f'{indent_repr}{nam}({typ})'

    if isinstance(node, Tree) and len(node.children) > 0:
        s += '\n'
        ch_indent_lvl = indent_lvl + 1
        ch_indent_repr = indent_repr + (' |' if ch_indent_lvl % 2 else ' :')

        for ch in node.children:
            s += get_pretty_lark_repr(
                ch,
                ch_indent_lvl,
                ch_indent_repr,
            )
    elif isinstance(node, Token):
        s += f' = {repr(node.value)}\n'
    else:
        s += '\n'

    return s


def pretty_print_lark(node: LarkNode) -> None:
    """
    Pretty prints the given lark node to stdout.
    """
    print(get_pretty_lark_repr(node))


def parse_and_print_lark(source_code: str) -> None:
    """
    Parses the given source code and pretty prints the parse tree.
    """
    from vyper_parser.cst import parse_python

    pretty_print_lark(parse_python(source_code))


def get_pretty_ast_repr(val: Any, indent_sep: str = ': ') -> str:
    """
    Returns a pretty-printed string representation of an AST.
    """
    if isinstance(val, (list, tuple)) and len(val) > 0:
        s = '['
        for i in val:
            i_repr = get_pretty_ast_repr(i, indent_sep)
            i_repr_lines = i_repr.splitlines()

            s += f'\n{indent_sep}' + f'\n{indent_sep}'.join(i_repr_lines) + ','
        s += '\n]'
    elif isinstance(val, (python_ast.AST, vyper_ast.VyperAST)):
        if hasattr(val, 'lineno'):
            pos_repr = f'  <line {val.lineno}, col {val.col_offset}>'  # type: ignore
        else:
            pos_repr = ''

        if isinstance(val, python_ast.AST):
            fields = val._fields
        elif isinstance(val, vyper_ast.VyperAST):
            fields = val.__slots__
        else:
            raise Exception('Unknown node type')

        if len(fields) > 0:
            s = f'{val.__class__.__name__}({pos_repr}'
            for field_name in fields:
                field_val = getattr(val, field_name)
                f_repr = get_pretty_ast_repr(field_val, indent_sep)
                f_repr_lines = f_repr.splitlines()

                field_name_eq = f'{field_name}='
                first_line_prefix = f'\n{indent_sep}{field_name_eq}'
                other_line_prefix = f'\n{indent_sep}' + ' ' * len(field_name_eq)

                s += first_line_prefix + other_line_prefix.join(f_repr_lines) + ','
            s += f'\n)'
        else:
            s = f'{val.__class__.__name__}{pos_repr}'
    else:
        s = repr(val)

    return s


def pretty_print_ast(node: python_ast.AST) -> None:
    """
    Pretty prints the given AST node to stdout.
    """
    print(get_pretty_ast_repr(node))


def parse_and_print_python(source_code: str) -> None:
    """
    Parses the given python source code and pretty prints its AST.
    """
    pretty_print_ast(python_ast.parse(source_code))
