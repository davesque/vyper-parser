import os
from pathlib import (
    Path,
)
import sys

from vyper_parser.cst import (
    parse_python,
)


def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def get_lib_path():
    if os.name == 'nt':
        # Special case for windows
        if 'PyPy' in sys.version:
            return Path(sys.prefix) / 'lib-python' / sys.winver
        else:
            return Path(sys.prefix) / 'Lib'

    # General case
    major = sys.version_info.major
    minor = sys.version_info.minor
    version_str = f'{major}.{minor}'

    return Path([x for x in sys.path if x.endswith(version_str)][0])


def find_fixture_files():
    return tuple(get_lib_path().glob('*.py'))


def test_python_lib():
    files = find_fixture_files()

    for f in files:
        print(f)
        parse_python(read_file(f))

    num_files = len(files)

    print(f'test_python_lib ({num_files} files)')
