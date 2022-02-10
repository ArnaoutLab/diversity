"""Miscellaneous helper module for the metacommunity package.

Functions
---------
get_file_delimiter
    Determines delimiter in datafile from file extension.
partition_range
    Splits range into evenly sized consecutive subranges.
pivot_table
    Converts long to wide formatted data.
power_mean
    Calculates weighted power means.
subset_by_column
    Returns portion of data frame with specific column values in subset.
unique_correspondence
    Describes correspondence between a sequence of items and the set of
    their unique elements.
"""
from pathlib import Path
from warnings import warn

from numpy import (
    amax,
    amin,
    array,
    dtype,
    inf,
    isclose,
    multiply,
    power,
    prod,
    sum as numpy_sum,
    unique,
    zeros,
    unique,
)

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import extract_data_if_shared


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


def partition_range(range_, num_chunks):
    """Splits range_ into evenly sized consecutive subranges.

    Parameters
    ----------
    range_: range
        The overall range to split into chunks.
    num_chunks: int
        The number of subranges to split range_ into. Must be positive.

    Returns
    -------
    A list of range objects. The ranges are evenly sized. When range_
    doesn't divide evenly by num_chunks, the ranges near the end of the
    returned list are 1 larger than ranges near the beginning.

    Raises
    ------
    InvalidArgumentError when num_chunks is not positive.
    """
    if num_chunks <= 0:
        raise InvalidArgumentError(
            f"Range can not be split into a non-positive number of chunks."
        )
    small_chunk_size, num_big_chunks = divmod(len(range_), num_chunks)
    num_small_chunks = num_chunks - num_big_chunks
    small_chunks_start = range_.start
    small_chunks = [
        range(
            small_chunks_start + (i * small_chunk_size),
            small_chunks_start + ((i + 1) * small_chunk_size),
        )
        for i in range(num_small_chunks)
    ]
    big_chunks_start = small_chunks_start + (num_small_chunks * small_chunk_size)
    big_chunk_size = small_chunk_size + 1
    big_chunks = [
        range(
            big_chunks_start + (i * big_chunk_size),
            big_chunks_start + ((i + 1) * big_chunk_size),
        )
        for i in range(num_big_chunks)
    ]
    return [*small_chunks, *big_chunks]


def pivot_table(
    data_frame,
    pivot_column,
    index_column,
    value_columns,
    pivot_ordering=None,
    index_ordering=None,
    out=None,
):
    """Converts long to wide formatted data.

    Assumes that combinations of pivot_column and index_colum value
    pairs appear at most once in table. If a combination appears more
    than once, then no guarantees are made whose values are chosen to be
    in the pivotted table.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data to pivot.
    pivot_column: str
        Name of column whose unique values will become new headers.
    index_column: str
        Name of column whose unique values will become the new index.
    value_columns: list[str]
        Names of columns whose data for each pivot_column-index_column
        value pair is included in the wide formatted table. Data should
        be numeric.
    pivot_ordering: Iterable
        Unique values of pivot column in desired order to determine
        ordering of resulting columns. If None, values are sorted.
    index_ordering: Iterable
        Unique values of index column in desired order to determine
        ordering of resulting index. If None, values are sorted.
    out: numpy.ndarray or diversity.shared.SharedArrayView
        Array of target shape in which to store result.

    Returns
    -------
    A numpy.ndarray where rows correspond to unique index_column values,
    columns to pairs of value_columns and unique pivot_column values,
    and each element is the corresponding value for that
    index-pivot-value column triple. If a pair of unique index and pivot
    column values does not appear in data_frame, its value is set to 0.0
    in the pivotted table.

    Notes
    -----
    With unique values ['foo', 'bar'] in pivot_column and value_columns
    ['a', 'b', 'c'], the resulting column ordering becomes:
        ['bar-a', 'bar-b', 'bar-c', 'foo-a', 'foo-b', 'foo-c']
    """
    LOGGER.debug(
        "pivot_table(data_frame=%s, pivot_column=%s, index_column=%s,"
        " value_columns=%s, pivot_ordering=%s, index_ordering=%s, out=%s)",
        data_frame,
        pivot_column,
        index_column,
        value_columns,
        pivot_ordering,
        index_ordering,
        out,
    )
    index_ordering_, index_positions = unique_correspondence(
        items=data_frame[index_column].to_numpy(),
        ordered_unique_items=index_ordering,
    )
    pivot_ordering_, pivot_positions = unique_correspondence(
        items=data_frame[pivot_column].to_numpy(),
        ordered_unique_items=pivot_ordering,
    )
    if out is None:
        out_ = zeros(
            (len(index_ordering_), len(pivot_ordering_) * len(value_columns)),
            dtype=dtype("f8"),
        )
    else:
        out_ = extract_data_if_shared(out)
        out_[:] = zeros(
            (len(index_ordering_), len(pivot_ordering_) * len(value_columns)),
            dtype=dtype("f8"),
        )
    for i, j, values in zip(
        index_positions,
        len(value_columns) * pivot_positions,
        data_frame[value_columns].to_numpy(),
    ):
        out_[i, j : j + len(value_columns)] = values
    return out_


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
        result = zeros(shape=items.shape, dtype=items.dtype)
        power(items, order, where=weight_is_nonzero, out=result)
        multiply(result, weights, where=weight_is_nonzero, out=result)
        items_sum = numpy_sum(result, axis=0, where=weight_is_nonzero)
        return power(items_sum, 1 / order)


def subset_by_column(data_frame, column, subset):
    """Returns portion of data_frame with column values in subset.

    If subset is None, entire data_frame is returned.
    """
    if subset is None:
        return data_frame
    return data_frame[data_frame[column].isin(subset)]


def unique_correspondence(items, ordered_unique_items=None):
    """Returns uniqued items and a mapping from items to uniqued items.

    Parameters
    ----------
    items: numpy.ndarray
        Array of items to unique and/or obtain mapping for.
    ordered_unique_items: Iterable
        Unique items in desired order. If None, unique items are sorted.

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
        "unique_correspondence(%s, ordered_unique_items=%s)",
        items,
        ordered_unique_items,
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
