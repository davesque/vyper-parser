from pathlib import (
    Path,
)
from typing import (
    Any,
)

import pytest


def get_id(fixture_path: Path) -> str:
    return str(fixture_path.resolve())


def parametrize_python_fixtures(arg: str, path: Path) -> Any:
    return pytest.mark.parametrize(
        arg,
        sorted(path.glob('*.py')),
        ids=get_id,
    )
