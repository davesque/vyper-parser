import os
from pathlib import (
    Path,
)
import sys
from typing import (
    Any,
)


def get_id(fixture_path: Path) -> str:
    return str(fixture_path.resolve())


def get_lib_path():
    if os.name == 'nt':
        return Path(sys.prefix) / 'Lib'
    else:
        major = sys.version_info.major
        minor = sys.version_info.minor
        version_str = f'{major}.{minor}'

        return Path([x for x in sys.path if x.endswith(version_str)][0])


def find_fixture_files():
    return tuple(get_lib_path().glob('*.py'))


def generate_fixture_tests(metafunc: Any) -> None:
    if 'fixture_path' in metafunc.fixturenames:
        all_fixture_paths = sorted(find_fixture_files())

        if len(all_fixture_paths) == 0:
            raise Exception('No fixtures found')

        metafunc.parametrize('fixture_path', all_fixture_paths, ids=get_id)
