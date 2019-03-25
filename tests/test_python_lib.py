from vyper_parser.cst import (
    parse_python,
)

from .generation import (
    generate_fixture_tests,
)


def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def pytest_generate_tests(metafunc):
    generate_fixture_tests(metafunc=metafunc)


def test_python_lib_is_parseable(fixture_path):
    parse_python(read_file(fixture_path))
