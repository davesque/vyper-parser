import ast as python_ast
from pathlib import (
    Path,
)

from vyper_parser.cst import (
    CSTVisitor,
    parse_python,
)

from .generation import (
    parametrize_python_fixtures,
    read_file,
)
from .utils import (
    assert_trees_equal,
)

FIXTURES_PATH = Path(__file__).parent / 'fixtures'


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURES_PATH,
)
def test_fixtures_are_parseable(fixture_path):
    parse_python(read_file(fixture_path))


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURES_PATH,
)
def test_fixture_trees_are_equivalent(fixture_path):
    source_code = read_file(fixture_path)

    cst = parse_python(source_code)
    vyper_node = CSTVisitor(tuple).visit(cst)

    python_node = python_ast.parse(source_code)

    assert_trees_equal(python_node, vyper_node)
