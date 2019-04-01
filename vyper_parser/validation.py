from vyper_parser.exceptions import (
    ValidationError,
)

FORBIDDEN_NAMES = {
    'None',
    'True',
    'False',
    '__debug__',
    'async',
    'await',
}


def validate_name(name: str, full_checks: bool = True) -> None:
    if name in FORBIDDEN_NAMES:
        name_reprs = ', '.join(repr(n) for n in FORBIDDEN_NAMES)
        raise ValidationError(
            f'Name {repr(name)} is forbidden.  Forbidden names are: {name_reprs}',
        )
