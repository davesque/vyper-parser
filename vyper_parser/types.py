from typing import (
    Union,
)

from lark import (
    Token,
)
from lark.tree import (
    Tree,
)

LarkNode = Union[Tree, Token]
