"""Tests for diversity.utilities."""

from numpy import allclose, array, inf
from pytest import mark, raises

from greylock.exceptions import InvalidArgumentError
from greylock.utilities import power_mean

weights_3by3 = array(
    [
        [1.0e-07, 1.0e-07, 1.0e-07],
        [2.0e-08, 2.0e-08, 2.0e-08],
        [1.0e-01, 1.0e-01, 1.0e-01],
    ]
)
weights_6by3 = array(
    [
        [1.0e-09, 1.0e-09, 1.0e-09],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [1.0e-07, 1.0e-07, 1.0e-04],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [2.0e-08, 2.0e-08, 4.0e-08],
        [1.0e-01, 1.0e-01, 2.0e-01],
    ]
)
weights_all_zero_6by3 = array(
    [
        [1.0e-09, 1.0e-09, 1.0e-09],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [1.0e-10, 1.0e-10, 1.0e-10],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [2.0e-11, 2.0e-11, 2.0e-11],
        [0.5e-08, 0.5e-08, 0.5e-08],
    ]
)
weights_some_zero_6by3 = array(
    [
        [1.1e-09, 1.1e-09, 1.1e-09],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [1.0e-07, 1.0e-07, 1.0e-07],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [2.0e-08, 2.0e-08, 4.0e-08],
        [1.0e-06, 1.0e-06, 2.0e-06],
    ]
)
weights_shape_mismatch = array(
    [
        [1.1e-09, 1.1e-09, 1.1e-09],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [1.0e-07, 1.0e-07, 1.0e-07],
        [0.0e-00, 0.0e-00, 0.0e-00],
        [2.0e-08, 2.0e-08, 4.0e-08],
        [1.0e-06, 1.0e-06, 2.0e-06],
        [2.0e-05, 1.2e-02, 4.5e-07],
    ]
)
equal_weights_and_items = array(
    [
        [[1.1e-09, 2.2e-08], [3.3e-07, 4.4e-06]],
        [[5.5e-06, 6.6e-05], [7.7e-04, 8.8e-03]],
    ]
)
items_3by3 = array(
    [
        [0.30, 0.69, 0.53],
        [0.74, 0.14, 0.53],
        [0.71, 0.13, 0.86],
    ]
)
items_6by3_1 = array(
    [
        [0.99, 0.79, 0.90],
        [0.56, 0.28, 0.99],
        [0.30, 0.69, 0.53],
        [0.89, 0.09, 0.14],
        [0.74, 0.14, 0.53],
        [0.71, 0.13, 0.86],
    ]
)
items_6by3_2 = array(
    [
        [0.28, 0.10, 0.51],
        [0.56, 0.28, 0.99],
        [0.30, 0.69, 0.53],
        [0.28, 0.09, 0.14],
        [0.74, 0.14, 0.53],
        [0.71, 0.13, 0.86],
    ]
)
items_6by4 = array(
    [
        [0.52, 0.73, 0.85, 1.23],
        [0.56, 0.28, 0.99, 4.56],
        [0.30, 0.69, 0.53, 7.89],
        [0.65, 0.65, 0.65, 0.12],
        [0.74, 0.14, 0.53, 3.45],
        [0.71, 0.13, 0.86, 6.78],
    ]
)


@mark.parametrize(
    "order, weights, items, atol, expected",
    [
        (  # No zero weights; nonzero order; -100 <= order <= 100; 2-d data.
            3,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.32955284, 0.06034367, 0.39917668]),
        ),
        (  # No zero weights; nonzero order; -100 <= order <= 100; 2-d data.
            1,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.07100004, 0.01300007, 0.08600006]),
        ),
        (  # No zero weights; nonzero order; -100 <= order <= 100; 2-d data.
            -100,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.35246927, 0.13302809, 0.62156143]),
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 2-d data.
            2,
            weights_6by3,
            items_6by4[:, :3],
            1e-8,
            array([0.22452176, 0.04111019, 0.3846402231]),
        ),
        (  # No zero weights; zero order; 2-d data.
            0,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.96633071, 0.8154443, 0.9850308]),
        ),
        (  # Some zero weights; zero order; 2-d data.
            0,
            weights_6by3,
            items_6by4[:, :3],
            1e-8,
            array([0.96633071, 0.8154443, 0.9702242087]),
        ),
        (  # No zero weights; order < -100; 2-d data.
            -101,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.30, 0.13, 0.53]),
        ),
        (  # Some zero weights; order == -inf; 2-d data.
            -inf,
            weights_some_zero_6by3,
            items_6by3_2,
            1e-8,
            array([0.30, 0.13, 0.53]),
        ),
        (  # No zero weights; order > 100; 2-d data.
            101,
            weights_3by3,
            items_3by3,
            1e-8,
            array([0.74, 0.69, 0.86]),
        ),
        (  # Some zero weights; order == inf; 2-d data.
            inf,
            weights_6by3,
            items_6by3_1,
            1e-8,
            array([0.74, 0.69, 0.86]),
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 1-d data
            2,
            weights_6by3[:, 0],
            items_6by3_1[:, 0],
            1e-8,
            array([0.22452176]),
        ),
        (  # Some zero weights; zero order; 1-d data.
            0,
            weights_6by3[:, 0],
            items_6by3_1[:, 0],
            1e-8,
            array([0.96633071]),
        ),
        (  # Some zero weights; order == -inf; 1-d data.
            -inf,
            weights_6by3[:, 0],
            items_6by3_1[:, 0],
            1e-8,
            array([0.30]),
        ),
        (  # Some zero weights; order == inf; 1-d data.
            inf,
            weights_6by3[:, 0],
            items_6by3_1[:, 0],
            1e-8,
            array([0.74]),
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; atol == 1e-9.
            2,
            weights_some_zero_6by3,
            items_6by4[:, :3],
            1e-9,
            array([0.0007241198, 0.0002559066, 0.001232607298]),
        ),
        (  # Some zero weights; zero order; 2-d data; atol == 1e-2.
            0,
            array(
                [
                    [1.1e-02, 1.1e-02, 1.1e-02],
                    [0.0e-00, 0.0e-00, 1.0e-3],
                    [1.0e-01, 1.0e-01, 1.0e-01],
                    [0.0e-00, 0.0e-00, 0.0e-00],
                    [4.0e-02, 4.0e-01, 4.0e-01],
                    [2.0e-01, 2.0e-02, 2.0e-02],
                ]
            ),
            items_6by4[:, :3],
            1e-2,
            array([0.8120992339, 0.4198668065, 0.7245218911]),
        ),
        (  # Some zero weights; order == -inf; 2-d data; atol == 1e-9.
            -inf,
            weights_some_zero_6by3,
            items_6by3_2,
            1e-9,
            array([0.28, 0.10, 0.51]),
        ),
        (  # Some zero weights; order == inf; 2-d data; atol == 1e-9.
            inf,
            weights_some_zero_6by3,
            items_6by3_1,
            1e-9,
            array([0.99, 0.79, 0.90]),
        ),
    ],
)
def test_power_mean(order, weights, items, atol, expected):
    actual_result = power_mean(
        order=order,
        weights=weights,
        items=items,
        atol=atol,
    )
    assert allclose(actual_result, expected)


@mark.parametrize(
    "order, weights, items",
    [
        (  # Zero weight column; nonzero order; -100 <= order <= 100; 2-d data.
            2,
            weights_all_zero_6by3,
            items_6by4[:, :3],
        ),
        (  # All zero weights; zero order; 2-d data.
            0,
            weights_all_zero_6by3,
            items_6by4[:, :3],
        ),
        (  # Zero weight columns; order < -100; 2-d data.
            -382,
            weights_all_zero_6by3,
            items_6by4[:, :3],
        ),
        (  # All zero weights; order > 100; 2-d data.
            364,
            weights_all_zero_6by3,
            items_6by4[:, :3],
        ),
        (  # Zero weights; nonzero order; -100 <= order <= 100; 1-d data",
            2,
            weights_all_zero_6by3[:, 0],
            items_6by3_1[:, 0],
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; mismatching number of rows.",
            2,
            weights_shape_mismatch,
            items_6by4[:, :3],
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; mismatching number of columns.",
            2,
            weights_some_zero_6by3,
            items_6by4,
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 1/2-d data; mismatching dimensions.",
            2,
            weights_some_zero_6by3[:, 0],
            items_6by4[:, :3],
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; 1-d data; mismatching number of rows.",
            2,
            weights_shape_mismatch[:, 0],
            items_6by3_1[:, 0],
        ),
        (  # Some zero weights; nonzero order; -100 <= order <= 100; >2-d.",
            2,
            equal_weights_and_items,
            equal_weights_and_items,
        ),
    ],
)
def test_power_mean_invalid_args(order, weights, items):
    with raises(InvalidArgumentError):
        power_mean(
            order=order,
            weights=weights,
            items=items,
            atol=1e-8,
        )
