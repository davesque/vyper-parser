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

FIXTURES_PATH = Path(__file__).parent / 'fixtures'


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURES_PATH,
)
def test_fixtures_are_parseable(fixture_path):
    parse_python(read_file(fixture_path))
