"""Custom exceptions raised by the diversity package.

Exceptions
----------
DiversityError
    Base class for all custom diversity exceptions.
DiversityWarning
    Base class for all custom diversity warnings.
InvalidArgumentError
    Raised when invalid argument is passed to a function.
LogicError
    Raised when code logic is flawed.
ArgumentWarning
    Used for warnings of problematic argument choices.
"""


class DiversityError(Exception):
    pass


class DiversityWarning(Warning):
    pass


class InvalidArgumentError(DiversityError):
    pass


class LogicError(DiversityError):
    pass


class ArgumentWarning(DiversityWarning):
    pass
