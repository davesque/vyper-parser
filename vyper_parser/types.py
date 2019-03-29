from typing import (
    Dict,
    Type,
    Union,
)

from lark import (
    Token,
)
from lark.tree import (
    Tree,
)

# cst types
LarkNode = Union[Tree, Token]

# utils types
SubclassesDict = Dict[str, Type]
