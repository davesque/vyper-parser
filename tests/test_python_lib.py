from vyper_parser.cst import (
    parse_python,
)

from .generation import (
    get_lib_path,
    parametrize_python_fixtures,
)


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


@parametrize_python_fixtures(
    'fixture_path',
    get_lib_path(),
)
def test_python_lib_is_parseable(fixture_path):
    parse_python(read_file(fixture_path))
