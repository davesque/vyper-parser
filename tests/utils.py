import ast as python_ast

from vyper_parser import (
    ast as vyper_ast,
)


def assert_trees_equal(python_val, vyper_val, check_pos=True):
    """
    Asserts that a python and vyper ast contain equivalent information.
    """
    # Assert sequences are equal
    if isinstance(python_val, (list, tuple)):
        for x, y in zip(python_val, vyper_val):
            assert_trees_equal(x, y, check_pos)

    # Assert analogous node classes are equal
    elif isinstance(python_val, python_ast.AST):
        # Class names are the same
        python_typ = python_val.__class__.__name__
        vyper_typ = vyper_val.__class__.__name__
        assert python_typ == vyper_typ

        # Parsing positions are the same
        if isinstance(vyper_val, vyper_ast.PosAttributes):
            if check_pos:
                assert python_val.lineno == vyper_val.lineno
                assert python_val.col_offset == vyper_val.col_offset
        else:
            assert python_val._attributes == ()

        # Node fields are the same
        for field in python_val._fields:
            python_field = getattr(python_val, field)
            vyper_field = getattr(vyper_val, field)

            assert_trees_equal(python_field, vyper_field, check_pos)

    # Assert normal values are equal
    else:
        assert python_val == vyper_val
