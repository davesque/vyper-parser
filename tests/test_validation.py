import pytest

from vyper_parser.exceptions import (
    ValidationError,
)
from vyper_parser.validation import (
    FORBIDDEN_NAMES,
    validate_name,
)


@pytest.mark.parametrize(
    'name',
    FORBIDDEN_NAMES,
)
def test_validate_name_raises_validation_error(name):
    with pytest.raises(ValidationError):
        validate_name(name)


def test_validate_name_succeeds():
    valid_name = 'foo'

    assert valid_name not in FORBIDDEN_NAMES
    validate_name(valid_name)
