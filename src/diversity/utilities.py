"""Miscellaneous helper module for the metacommunity package.

Functions
---------
get_file_delimiter
    Determines delimiter in datafile from file extension.
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
ArgumentWarning
    Used for warnings of problematic argument choices.
"""
from pathlib import Path
from warnings import warn

from numpy import (
    amax,
    amin,
    array,
    inf,
    isclose,
    multiply,
    power,
    prod,
    sum as numpy_sum,
    unique
)

from diversity.log import LOGGER


class MetacommunityError(Exception):
    """Base class for all custom metacommunity exceptions."""

    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""

    pass


class ArgumentWarning(Warning):
    """Used for warnings related to problematic argument choices."""

    pass


def get_file_delimiter(filepath):
    """Determines delimiter of file from filepath ending.

    Parameters
    ----------
    filepath: str
        The filepath whose delimiter is to be determine.

    Returns
    -------
    "," if the file has the csv extension and "\t" otherwise.

    Notes
    -----
    Emits a warning if filepath has neither .csv nor .tsv ending.
    """
    suffix = Path(filepath).suffix
    if suffix == ".csv":
        return ","
    elif suffix == ".tsv":
        return "\t"
    else:
        warn(
            f"File extension for {filepath} not recognized. Assuming"
            " tab-delimited file.",
            category=ArgumentWarning,
        )
        return "\t"


def __validate_power_mean_args(weights, items, atol, weight_is_nonzero):
    """Validates arguments for power_mean.

    Parameters
    ----------
    weights, items, atol
        Same as for power_mean.
    weight_is_nonzero
        Boolean array of same shape as weights, indicating those whose
        absolute value meets or exceeds atol.

    Raises
    ------
    InvalidArgumentError when:
    - weights has more than 2 axes.
    - weights and items do not have the same shape
    - any column in weights contains 0s only when weights is 2d and when
      weights is all 0.
    """
    if len(weights.shape) > 2:
        raise InvalidArgumentError(
            f"Invalid weights shape for power_mean: {weights.shape}."
        )
    if weights.shape != items.shape:
        raise InvalidArgumentError(
            f"Shape of weights ({weights.shape}) must be the same as"
            f" shape fo items ({items.shape})."
        )
    all_0_column = None
    if len(weights.shape) == 1:
        all_0_column = [not (weight_is_nonzero).any()]
    elif len(weights.shape) == 2:
        all_0_column = [
            not (weight_is_nonzero[:, col]).any() for col in range(weights.shape[1])
        ]
    if any(all_0_column):
        raise InvalidArgumentError(
            "power_mean expects at least one zero weight. Weights are"
            " considered 0, if absolute value does not meet or exceed"
            f" configurable minimum threshold: {atol:.2e}."
        )


def power_mean(order, weights, items, atol=1e-9):
    """Calculates weighted power means.

    Parameters
    ----------
    order: numeric
        Exponent used for the power mean.
    weights: numpy.ndarray
        The weights corresponding to items. Must be 1-d or 2-d. If 2-d,
        each column must contain at least one wheight whose absolute
        value meets or exceeds atol.
    items: numpy.ndarray
        The elements for which the weighted power mean is computed. Must
        have same shape as weights.
    atol: float
        Threshold below which weights are considered to be 0.

    Returns
    -------
    A numpy.ndarray of the power means of items along axis 0 using order
    as exponent, weighing by weights. The array shape is the same as
    that of weights, or items except with the 0-axis removed. In the
    case of 1-d weights and items, the result has shape (1,). When order
    is close to 0 (absolute value less than atol), less than -100, or
    greater than 100 analytical formulas for the limits at 0, -infinity,
    or infinity are used respectively. An exception is raised if all
    weights or a column of weights (in the 2-d case) are all too close
    to 0.
    """
    weight_is_nonzero = abs(weights) >= atol
    __validate_power_mean_args(weights, items, atol, weight_is_nonzero)
    if isclose(order, 0):
        return prod(
            power(items, weights, where=weight_is_nonzero),
            axis=0,
            where=weight_is_nonzero,
        )
    elif order < -100:
        return amin(items, axis=0, where=weight_is_nonzero, initial=inf)
    elif order > 100:
        return amax(items, axis=0, where=weight_is_nonzero, initial=-inf)
    else:
        items_power = power(items, order, where=weight_is_nonzero)
        items_product = multiply(items_power, weights, where=weight_is_nonzero)
        items_sum = numpy_sum(items_product, axis=0, where=weight_is_nonzero)
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
    LOGGER.debug(
        "unique_correspondence(%s, ordered_unique_items=%s"
        % (items, ordered_unique_items)
    )
    if ordered_unique_items is None:
        ordered_unique_items_, item_positions = unique(items, return_inverse=True)
    else:
        ordered_unique_items_ = array(ordered_unique_items)
        item_to_position = {
            str(item): pos for pos, item in enumerate(ordered_unique_items_)
        }
        if len(item_to_position) != len(ordered_unique_items):
            raise InvalidArgumentError("Expected ordered_unique_items to be uniqued.")
        item_positions = array([item_to_position[str(item)] for item in items])
    return (ordered_unique_items_, item_positions)
