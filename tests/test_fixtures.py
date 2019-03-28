import ast as python_ast
import collections.abc
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

FIXTURES_PATH = Path(__file__).parent / 'fixtures'


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURES_PATH,
)
def test_fixtures_are_parseable(fixture_path):
    parse_python(read_file(fixture_path))


def assert_trees_equal(python_val, vyper_val):
    """
    Asserts that a python and vyper ast contain equivalent information.
    """
    # Assert sequences are equal
    if isinstance(python_val, collections.abc.Sequence):
        for x, y in zip(python_val, vyper_val):
            assert_trees_equal(x, y)

    # Assert analogous node classes are equal
    elif isinstance(python_val, python_ast.AST):
        # Class names are the same
        python_typ = python_val.__class__.__name__
        vyper_typ = vyper_val.__class__.__name__
        assert python_typ == vyper_typ

        # Parsing positions are the same
        # if isinstance(vyper_val, vyper_ast.PosAttributes):
        #     assert python_val.lineno == vyper_val.lineno
        #     assert python_val.col_offset == vyper_val.col_offset
        # else:
        #     assert python_val._attributes == ()

        # Node fields are the same
        for field in python_val._fields:
            python_field = getattr(python_val, field)
            vyper_field = getattr(vyper_val, field)

            assert_trees_equal(python_field, vyper_field)

    # Assert normal values are equal
    else:
        assert python_val == vyper_val


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURES_PATH,
)
def test_fixture_trees_are_equivalent(fixture_path):
    source_code = read_file(fixture_path)

    cst = parse_python(source_code)
    vyper_node = CSTVisitor().visit(cst)

    python_node = python_ast.parse(source_code)

    assert_trees_equal(python_node, vyper_node)
