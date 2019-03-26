import os
from pathlib import (
    Path,
)
import sys
from typing import (
    Any,
)

import pytest


def get_lib_path() -> Path:
    if os.name == 'nt':
        return Path(sys.prefix) / 'Lib'
    else:
        major = sys.version_info.major
        minor = sys.version_info.minor
        version_str = f'{major}.{minor}'

        return Path([x for x in sys.path if x.endswith(version_str)][0])


def get_id(fixture_path: Path) -> str:
    return str(fixture_path.resolve())


def parametrize_python_fixtures(arg: str, path: Path) -> Any:
    return pytest.mark.parametrize(
        arg,
        sorted(path.glob('*.py')),
        ids=get_id,
    )
