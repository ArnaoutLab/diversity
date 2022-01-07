"""Miscellaneous helper module for the Metacommunity package.

Functions
---------
unique_mapping
    Corresponds items in non-unique sequence to a uniqued ordered
    sequence of those items.
"""
from numpy import isclose, prod, zeros, amin, unique, arange, empty_like, sum as numpy_sum, multiply, inf, power, float64
from copy import deepcopy
from collections.abc import Mapping


class MetacommunityError(Exception):
    """Base class for all custom Metacommunity exceptions."""
    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""
    pass


def pivot_table(data, columns_index, indices_index, values_index):
    rows, row_indices = unique(data[:, indices_index], return_inverse=True)
    cols, col_indices = unique(data[:, columns_index], return_inverse=True)
    table = zeros((len(rows), len(cols)), dtype=float64)
    table[row_indices, col_indices] = data[:, values_index]
    return table, cols


# FIXME Currently not used
def reorder_rows(data, new_order):
    _, header_indices = unique(new_order, return_index=True)
    idx = empty_like(header_indices)
    idx[header_indices] = arange(len(header_indices))
    return data[idx, :]


class FrozenDict(Mapping):
    """An immutable dict-like class."""

    def __init__(self, mapping):
        """Initializes object with same keys and values as mapping.

        Parameters
        ----------
        mapping: dict
            The key-value pairs to initialize the object with.
        """
        self.__mapping = deepcopy(mapping)

    def __getitem__(self, key):
        return self.__mapping[key]

    def __iter__(self):
        return iter(self.__mapping)

    def __len__(self):
        return len(self.__mapping)

    def __repr__(self):
        return repr(self.__mapping)


def factorize(arr):
    """
    FIXME
    """
    names, _ = unique(arr, return_inverse=True)
    name_map = dict(zip(names, range(len(names))))
    return [name_map[i] for i in arr]


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
