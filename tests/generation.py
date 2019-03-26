from pathlib import (
    Path,
)
from typing import (
    Any,
)

import pytest


def read_file(path: Path) -> str:
    with open(path, 'r') as f:
        return f.read()


def get_id(path: Path) -> str:
    return str(path.resolve())


def parametrize_python_fixtures(arg: str, path: Path) -> Any:
    return pytest.mark.parametrize(
        arg,
        sorted(path.glob('*.py')),
        ids=get_id,
    )
