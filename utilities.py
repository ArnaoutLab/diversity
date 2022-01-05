"""Miscellaneous helper module for the Metacommunity package.

Functions
---------
unique_mapping
    Corresponds items in non-unique sequence to a uniqued ordered
    sequence of those items.
"""
from dataclasses import dataclass
from functools import cached_property

from numpy import (array, empty, unique, isclose, prod, broadcast_to,
                   amin, sum as numpy_sum, multiply, inf, power, int64)


class MetacommunityError(Exception):
    """Base class for all custom Metacommunity exceptions."""
    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""
    pass


@dataclass
class UniqueRowsCorrespondence:
    """Corresponds data array rows to order of a uniqued key column.

    Attributes    
    ----------
    data: numpy.ndarray
        The data for which to establish a correspondence.
    key_column_pos: int
        Index of column in data attribute for the keys according to
        which the data rows are uniqued.
    """

    data: array
    key_column_pos: int = 0

    @cached_property
    def unique_row_index(self):
        """Extracts index of rows corresponding to uniqued column.

        Returns
        -------
        A 1-d numpy.ndarray of indices which are the positions of the unique
        items in the key column.
        """
        _, index = unique(self.data[:, self.key_column_pos], return_index=True)
        return index

    @cached_property
    def unique_keys(self):
        """Obtains uniqued values in key column.

        Returns
        -------
        A 1-d numpy.ndarray of unique keys in key column.
        """
        return self.data[:, self.key_column_pos][self.unique_row_index]

    @cached_property
    def key_to_unique_pos(self):
        """Maps values in key column to positions in uniqued order.

        Returns
        -------
        A dict with values of key column as keys and their position in
        their uniqued ordering as values.
        """
        return dict((key, pos) for pos, key in enumerate(self.unique_keys))

    @cached_property
    def row_to_unique_pos(self):
        """Maps row positions to positions in uniqued order.

        Returns
        -------
        A 1-d numpy.array of the same length as object's data attribute
        containing the positions in uniqued ordering of corresponding
        rows in object's data atribute.
        """
        positions = empty(self.data.shape[0], dtype=int64)
        for key in self.data[:, self.key_column_pos]:
            positions[key] = self.key_to_unique_pos[key]
        return positions


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
        items = broadcast_to(items, weights.shape)
        return amin(items, axis=0, where=mask, initial=inf)
    items_power = power(items, order, where=mask)
    items_product = multiply(items_power, weights, where=mask)
    items_sum = numpy_sum(items_product, axis=0, where=mask)
    return power(items_sum, 1 / order)


def register(item, registry):
    """Returns value for item in registry, creating one if necessary.

    Registry is meant to be kept the same for a collection of items and
    should initially be empty.

    Parameters
    ----------
    item
        Object to query against registry.
    registry: dict
        Maps items to their registered value.

    Returns
    -------
    The value of item in registry. If item is not a key of registry,
    then the current size of registry becomes its key in an attempt to
    maintain a registry of unique integers assigned to different items.
    """
    if item not in registry:
        num = len(registry)
        registry[item] = num
    return registry[item]
