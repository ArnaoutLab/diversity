"""Miscellaneous helper module for the Metacommunity package.

Classes
-------
UniqueRowsCorrespondence
    Corresponds items in non-unique sequence to a uniqued ordered
    sequence of those items.

Functions
---------
power_mean
    Calculates weighted power mean.
cached_property_depends_on
    Decorator for property caching return value dependent on arguments.
register
    Registers an unregistered item, if needed and returns its registered
    value.

Exceptions
----------
MetacommunityError
    Base class for all custom Chubacabra exceptions.
InvalidArgumentError
    Raised when invalid argument is passed to a function.
"""
from numpy import isclose, prod, amin, sum as numpy_sum, multiply, inf, power


class MetacommunityError(Exception):
    """Base class for all custom Metacommunity exceptions."""

    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""

    pass


def power_mean(order, weights, items):
    """Calculates a weighted power mean.

    Parameters
    ----------
    weights: numpy.ndarray
        The weights corresponding to items.
    items: numpy.ndarray
        The elements for which the weighted power mean is computed.

    Returns
    -------
    The power mean of items with exponent order, weighted by weights.
    When order is close to 1 or less than -100, analytical formulas
    for the limits at 1 and -infinity are used respectively.
    """
    mask = weights != 0
    if isclose(order, 0):
        return prod(power(items, weights, where=mask), axis=0, where=mask)
    elif order < -100:
        return amin(items, axis=0, where=mask, initial=inf)
    items_power = power(items, order, where=mask)
    items_product = multiply(items_power, weights, where=mask)
    items_sum = numpy_sum(items_product, axis=0, where=mask)
    return power(items_sum, 1 / order)
