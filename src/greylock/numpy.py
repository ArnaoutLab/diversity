from numpy import (
    divide,
    abs,
    broadcast_to,
    array,
    any,
    all,
    amin,
    amax,
    inf,
    multiply,
    power,
    prod,
    sum as numpy_sum,
    zeros,
    float64,
)
import warnings


def get_community_ratio(numerator, denominator):
    return divide(
        numerator,
        denominator,
        out=zeros(denominator.shape),
        where=denominator != 0,
    )


def to_numpy(a):
    return array(a)


def any_all_false_columns(m):
    return any(all(~m, axis=0))


def find_nonzero_entries(weights, atol):
    return abs(weights) >= atol


def zero_order_powermean(items, weights, weight_is_nonzero):
    with warnings.catch_warnings(category=RuntimeWarning) as w:
        warnings.simplefilter("ignore")
        power_result = power(items, weights, where=weight_is_nonzero)
    return prod(
        power_result,
        axis=0,
        where=weight_is_nonzero,
    )


def find_amin(items, where, axis=0):
    return amin(items, axis=0, where=where, initial=inf)


def find_amax(items, where, axis=0):
    return amax(items, axis=0, where=where, initial=-inf)


def powermean(items, weights, order, weight_is_nonzero):
    result = zeros(shape=items.shape, dtype=float64)
    with warnings.catch_warnings(category=RuntimeWarning) as w:
        warnings.simplefilter("ignore")
        power(items, order, where=weight_is_nonzero, out=result)
    multiply(result, weights, where=weight_is_nonzero, out=result)
    items_sum = numpy_sum(result, axis=0, where=weight_is_nonzero)
    with warnings.catch_warnings(category=RuntimeWarning) as w:
        warnings.simplefilter("ignore")
        return power(items_sum, 1 / order)
