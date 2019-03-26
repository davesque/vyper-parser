from pathlib import (
    Path,
)

from vyper_parser.cst import (
    parse_python,
)

from .generation import (
    parametrize_python_fixtures,
    read_file,
)


@parametrize_python_fixtures(
    'fixture_path',
    Path(__file__).parent / 'fixtures',
)
def test_fixtures(fixture_path):
    parse_python(read_file(fixture_path))
