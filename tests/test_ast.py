import ast as python_ast

from vyper_parser.ast import (
    VyperAST,
)

from .generation import (
    get_lib_path,
    parametrize_python_fixtures,
    read_file,
)
from .utils import (
    assert_trees_equal,
)


@parametrize_python_fixtures(
    'fixture_path',
    get_lib_path(),
)
def test_python_lib_is_convertible_to_vyper_ast(fixture_path):
    source_code = read_file(fixture_path)

    python_node = python_ast.parse(source_code)
    vyper_node = VyperAST.from_python_ast(python_node)

    assert_trees_equal(python_node, vyper_node)
