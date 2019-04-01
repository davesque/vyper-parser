class ValidationError(Exception):
    """
    Raised when elements of a source file do not conform to extra constraints
    that are not represented in the grammar.  For example, the short program
    "None = 1" is parseable but not valid.
    """
    pass
