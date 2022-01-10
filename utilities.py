"""Miscellaneous helper module for the metacommunity package.

Functions
---------
power_mean
    Calculates weighted power means.
unique_correspondence
    Describes correspondence between a sequence of items and the set of
    their unique elements.

Exceptions
----------
MetacommunityError
    Base class for all custom metacommunity exceptions.
InvalidArgumentError
    Raised when invalid argument is passed to a function.
"""
from numpy import (
    array,
    isclose,
    prod,
    amin,
    amax,
    sum as numpy_sum,
    multiply,
    inf,
    power,
    unique,
)


class MetacommunityError(Exception):
    """Base class for all custom metacommunity exceptions."""

    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""

    pass


def power_mean(order, weights, items):
    """Calculates weighted power means.

    Parameters
    ----------
    order: numeric
        Exponent used for the power mean.
    weights: numpy.ndarray
        The weights corresponding to items.
    items: numpy.ndarray
        The elements for which the weighted power mean is computed. Must
        have same shape as weights.

    Returns
    -------
    A numpy.ndarray of the power means of items along axis 0 using order
    as exponent, weighing by weights. The array shape is the same as
    that of weights, or items except with the 0-axis removed. In the
    case of 1-d weights and items, the result has shape (1,). When order
    is close to 0 (absolute value less than 1e-8), less than -100, or
    greater than 100 analytical formulas for the limits at 0, -infinity,
    or infinity are used respectively.
    """
    mask = abs(weights) > 1e-8
    if isclose(order, 0):
        return prod(power(items, weights, where=mask), axis=0, where=mask)
    elif order < -100:
        return amin(items, axis=0, where=mask, initial=inf)
    elif order > 100:
        return amax(items, axis=0, where=mask, initial=-inf)
    items_power = power(items, order, where=mask)
    items_product = multiply(items_power, weights, where=mask)
    items_sum = numpy_sum(items_product, axis=0, where=mask)
    return power(items_sum, 1 / order)


def unique_correspondence(items, ordered_unique_items=None):
    """Returns uniqued items and a mapping from items to uniqued items.

    Parameters
    ----------
    items: numpy.ndarray
        Array of items to unique and/or obtain mapping for.
    ordered_unique_items: Iterable
        Unique items in desired order. If None, ordering will be
        established according to numpy.unique.

    Returns
    -------
    A tuple with coordinates:
    0 - numpy.ndarray
        The ordered unique items.
    1 - numpy.ndarray
        The position in the unique items iterable for each item in
        items.
    """
    if ordered_unique_items is None:
        ordered_unique_items_, item_positions = unique(items, return_inverse=True)
    else:
        ordered_unique_items_ = array(ordered_unique_items)
        item_to_position = {item: pos for pos, item in enumerate(ordered_unique_items_)}
        if len(item_to_position) != len(ordered_unique_items):
            raise InvalidArgumentError(f"Expected ordered_unique_items to be uniqued.")
        item_positions = array([item_to_position[item] for item in items])
    return (ordered_unique_items_, item_positions)
