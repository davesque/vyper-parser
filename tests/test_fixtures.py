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

FIXTURE_PATH = Path(__file__).parent / 'fixtures'


@parametrize_python_fixtures(
    'fixture_path',
    FIXTURE_PATH,
)
def test_fixtures(fixture_path):
    parse_python(read_file(fixture_path))
