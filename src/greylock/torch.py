from typing import Iterable, Union
from greylock.similarity import Similarity
from torch import (
    tensor,
    zeros_like,
    empty,
    div,
    abs,
    any,
    all,
    logical_not,
    pow,
    prod,
    mul,
    sum,
    float64,
    broadcast_to,
    amin,
    amax,
)
from numpy import ndarray
from greylock.abundance import AbundanceForDiversity
import numpy as np
from greylock.log import timing

class SimilarityFromTensor(Similarity):
    def __init__(
        self,
        similarity: tensor,
        similarities_out: Union[ndarray, None] = None,
    ):
        super().__init__(similarities_out)
        self.similarity = similarity

    def is_expensive(self):
        return False

    def weighted_abundances(
        self,
        relative_abundance,
    ) -> ndarray:
        if self.similarities_out is not None:
            self.similarities_out[:, :] = self.similarity.numpy()
        return self.similarity @ relative_abundance


class AbundanceFromTensor(AbundanceForDiversity):
    def __init__(
        self,
        counts: tensor,
        subcommunity_names: Union[None, Iterable[Union[str, int]]] = None,
    ) -> None:
        self.num_subcommunities = counts.shape[1]
        if subcommunity_names is None:
            self.subcommunities_names = [i for i in range(self.num_subcommunities)]
        else:
            self.subcommunities_names = subcommunity_names
        self.min_count = min(1 / counts.sum().item(), 1e-9)

        self.subcommunity_abundance = self.make_subcommunity_abundance(counts=counts)
        self.normalized_subcommunity_abundance = (
            self.make_normalized_subcommunity_abundance()
        )
        self.metacommunity_abundance = self.make_metacommunity_abundance()


def get_community_ratio(numerator, denominator):
    result = div(numerator, denominator)
    result[denominator == 0] = 0
    return result


def find_nonzero_entries(t, atol):
    return abs(t) >= atol


def any_all_false_columns(t):
    has_true_in_col = any(t, dim=0)
    has_all_false_col = any(logical_not(has_true_in_col))
    return has_all_false_col


"""
def power(items, weights):
    return pow(items, weights)
"""


def prod0(t):
    return prod(t, dim=0)


def to_numpy(t):
    return t.to("cpu").numpy()


def find_amin(items, where, axis=0):
    items[~where] = np.inf
    return amin(items, axis)


def find_amax(items, where, axis=0):
    items[~where] = -np.inf
    return amax(items, axis)


def zero_order_powermean(items, weights, weight_is_nonzero):
    power_result = pow(items, weights)

    # This shouldn't be neccessary:
    power_result[logical_not(weight_is_nonzero)] = 1.0
    return prod0(power_result)


def powermean(items, weights, order, weight_is_nonzero):
    result = empty(size=items.size(), dtype=float64, device=items.device)
    pow(items, order, out=result)
    result[~weight_is_nonzero] = 0
    mul(result, weights, out=result)
    items_sum = sum(result, dim=0)
    return pow(items_sum, 1 / order)
