"""Miscellaneous helper module for the metacommunity package.

Functions
---------
get_file_delimiter
    Determines delimiter in datafile from file extension.
power_mean
    Calculates weighted power means.
"""
from pathlib import Path
from warnings import warn

from numpy import (
    amax,
    amin,
    float64,
    inf,
    isclose,
    multiply,
    power,
    prod,
    sum as numpy_sum,
    zeros,
)

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.log import LOGGER


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
    LOGGER.debug(
        "power_mean(order=%s, weights=%s, items=%s, atol=%s)",
        order,
        weights,
        items,
        atol,
    )
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
        result = zeros(shape=items.shape, dtype=float64)
        power(items, order, where=weight_is_nonzero, out=result)
        multiply(result, weights, where=weight_is_nonzero, out=result)
        items_sum = numpy_sum(result, axis=0, where=weight_is_nonzero)
        return power(items_sum, 1 / order)
