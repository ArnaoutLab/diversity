from typing import Iterable, Union
from greylock.similarity import Similarity
from torch import tensor, zeros_like, div, abs, any, all, logical_not, pow, prod, mul, sum, float64, broadcast_to, amin, amax
from numpy import ndarray
from greylock.abundance import AbundanceForDiversity
import numpy as np

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
        print("Z dtype:")
        print(self.similarity.dtype)
        print("counts dtype:")
        print(relative_abundance.dtype)
        return self.similarity @ relative_abundance

class AbundanceFromTensor(AbundanceForDiversity):
    def __init__(
        self, counts: tensor, subcommunity_names: Iterable[Union[str, int]]
    ) -> None:
        self.subcommunities_names = subcommunity_names
        self.num_subcommunities = counts.shape[1]
        self.min_count = min(1 / counts.sum().item(), 1e-9)

        self.subcommunity_abundance = self.make_subcommunity_abundance(counts=counts)
        self.normalized_subcommunity_abundance = (
            self.make_normalized_subcommunity_abundance()
        )
        self.metacommunity_abundance = self.make_metacommunity_abundance()

def get_community_ratio(numerator, denominator):
    community_ratio = zeros_like(denominator)
    mask = denominator != 0
    if isinstance(numerator, int):
        num = numerator
    elif numerator.shape == denominator.shape:
        num = numerator[mask]
    else:
        num = numerator
    div_results = div(
        num,
        denominator)
    community_ratio[mask] = div_results[mask]
    return community_ratio

def find_nonzero_entries(t, atol):
    return abs(t) >= atol

def any_all_false_columns(t):
    has_true_in_col = any(t, dim=0)
    has_all_false_col = any(logical_not(has_true_in_col))
    return has_all_false_col

def power(items, weights):
    return pow(items, weights)

def prod0(t):
    return prod(t, dim=0)

def powermean(items, weights, order, weight_is_nonzero):
    result = zeros_like(items, dtype=float64)
    pow(items, order, out=result)
    mul(result, weights, out=result)
    items_sum = sum(result, dim=0)
    return pow(items_sum, 1 / order)

def to_numpy(t):
    return t.to('cpu').numpy()

def find_amin(items, where, axis=0):
    items[~where] = np.inf
    return amin(items, axis)

def find_amax(items, where, axis=0):
    items[~where] = -np.inf
    return amax(items, axis)

    
