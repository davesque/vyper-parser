import os
from pathlib import (
    Path,
)
import sys

from vyper_parser.cst import (
    parse_python,
)

from .generation import (
    parametrize_python_fixtures,
    read_file,
)


def get_lib_path() -> Path:
    if os.name == 'nt':
        return Path(sys.prefix) / 'Lib'
    else:
        major = sys.version_info.major
        minor = sys.version_info.minor
        version_str = f'{major}.{minor}'

        return Path([x for x in sys.path if x.endswith(version_str)][0])


@parametrize_python_fixtures(
    'fixture_path',
    get_lib_path(),
)
def test_python_lib_is_parseable(fixture_path):
    parse_python(read_file(fixture_path))
