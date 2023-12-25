"""Miscellaneous helper module for the metacommunity package.

Functions
---------
get_file_delimiter
    Determines delimiter in datafile from file extension.
power_mean
    Calculates weighted power means.
"""

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
    all,
    any,
    ndarray,
)

from greylock.exceptions import InvalidArgumentError


def __validate_power_mean_args(
    weights: ndarray, items: ndarray, atol: float, weight_is_nonzero: ndarray
) -> None:
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
            f"'weights' shape must have 1 or 2 dimensions, but 'weights' had shape {weights.shape}."
        )
    if weights.shape != items.shape:
        raise InvalidArgumentError(
            f"Shape of 'weights' ({weights.shape}) must be the same as"
            f" shape of 'items' ({items.shape})."
        )
    all_0_column = all(~weight_is_nonzero, axis=0)
    if any(all_0_column):
        raise InvalidArgumentError(
            "Argument 'weights' must have at least one nonzero weight in each column. A weight is"
            " considered 0 if its absolute value is greater than or equal to"
            f" configurable minimum threshold: {atol:.2e}."
        )


def power_mean(
    order: float, weights: ndarray, items: ndarray, atol: float = 1e-9
) -> ndarray:
    """Calculates weighted power means.

    Parameters
    ----------
    order:
        Exponent used for the power mean.
    weights:
        The weights corresponding to items. Must be 1-d or 2-d. If 2-d,
        each column must contain at least one weight whose absolute
        value is greater than or equal to atol.
    items:
        The elements for which the weighted power mean is computed. Must
        have same shape as weights.
    atol:
        Threshold below which weights are considered to be 0.

    Returns
    -------
    A numpy.ndarray of the power means of items along axis 0 using order
    as exponent, weighted by weights. The array shape is the same as
    that of weights, or items except with the 0-axis removed. In the
    case of 1-d weights and items, the result has shape (1,). When order
    is close to 0 (absolute value less than atol), less than -100, or
    greater than 100 analytical formulas for the limits at 0, -infinity,
    or infinity are used respectively. An exception is raised if all weights
    in a column are close to 0.
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
        result = zeros(shape=items.shape, dtype=float64)
        power(items, order, where=weight_is_nonzero, out=result)
        multiply(result, weights, where=weight_is_nonzero, out=result)
        items_sum = numpy_sum(result, axis=0, where=weight_is_nonzero)
        return power(items_sum, 1 / order)
