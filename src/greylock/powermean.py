"""Miscellaneous helper module for the metacommunity package.

Functions
---------
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
    allclose,
)

from greylock.exceptions import InvalidArgumentError


def __validate_power_mean_args(
        weights: ndarray, items: ndarray, atol: float, weight_is_nonzero: ndarray, backend = None
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
    if backend is None:
        all_0_column = any(all(~weight_is_nonzero, axis=0))
    else:
        all_0_column = backend.any_all_false_columns(weight_is_nonzero)
    if all_0_column:
        raise InvalidArgumentError(
            "Argument 'weights' must have at least one nonzero weight in each column. A weight is"
            " considered 0 if its absolute value is greater than or equal to"
            f" configurable minimum threshold: {atol:.2e}."
        )

def power_mean(
        order: float, weights: ndarray, items: ndarray, atol: float = 1e-9, backend = None
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
    if backend is None:
        weight_is_nonzero = abs(weights) >= atol
    else:
        weight_is_nonzero = backend.find_nonzero_entries(weights, atol)
    __validate_power_mean_args(weights, items, atol, weight_is_nonzero, backend=backend)
    if isclose(order, 0):
        if backend is None:
            power_result = power(items, weights, where=weight_is_nonzero)
            return prod(
                power_result,
                axis=0,
                where=weight_is_nonzero,
            )
        else:
            power_result = backend.pow(items, weights)

            if False:
                # EXTRANEOUS FOR TROUBLESHOOTING WILL REMOVE
                print(items.dtype)
                print(weights.dtype)
                items = items.to('cpu').numpy()
                weights = weights.to('cpu').numpy()
                print(items.dtype)
                print(weights.dtype)
                weight_is_nonzero = abs(weights) >= atol
                power_result_numpy = power(items, weights, where=weight_is_nonzero)
                diff = power_result.to('cpu').numpy() - power_result_numpy
                weirdness_mask = weight_is_nonzero & (abs(diff) > 0.0000001)
                print(items[weirdness_mask])
                print(items[weight_is_nonzero].mean())
                print(weights[weirdness_mask])
                print(weights[weight_is_nonzero].mean())

            # This shouldn't be neccessary:
            #power_result[backend.logical_not(weight_is_nonzero)] = 1.0
            return backend.prod0(power_result)
    elif order < -100:
        if backend is None:
            return amin(items, axis=0, where=weight_is_nonzero, initial=inf)
        else:
            return backend.find_amin(items, where=weight_is_nonzero, axis=0)
    elif order > 100:
        if backend is None:
            return amax(items, axis=0, where=weight_is_nonzero, initial=-inf)
        else:
            return backend.find_amax(items, where=weight_is_nonzero, axis=0)
    else:
        if abs(order) > 25:
            pass #breakpoint()
        if backend is None:
            result = zeros(shape=items.shape, dtype=float64)
            power(items, order, where=weight_is_nonzero, out=result)
            multiply(result, weights, where=weight_is_nonzero, out=result)
            items_sum = numpy_sum(result, axis=0, where=weight_is_nonzero)
            return power(items_sum, 1 / order)
        else:
            return backend.powermean(items, weights, order, weight_is_nonzero)
            
