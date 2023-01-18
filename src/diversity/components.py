from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union

from numpy import ndarray

from diversity.abundance import Abundance
from diversity.similarity import Similarity


class Components(ABC):
    """Dispatches diversity components based on specified measure"""

    @abstractmethod
    def get_numerator(measure: str):
        pass

    @abstractmethod
    def get_denominator(measure: str):
        pass


class FrequencySensitiveComponents(Components):
    """Dispatches frequency-sensitive diversity components based on specified measure"""

    def __init__(self, abundance: Abundance) -> None:
        self.abundance = abundance

    def get_numerator(self, measure: str) -> Union[int, ndarray]:
        if measure in {"alpha", "gamma", "normalized_alpha"}:
            return 1
        elif measure in {"beta", "rho", "normalized_beta", "normalized_rho"}:
            return self.abundance.metacommunity_abundance

    def get_denominator(self, measure: str) -> ndarray:
        if measure in {"alpha", "beta", "rho"}:
            return self.abundance.subcommunity_abundance
        elif measure in {"normalized_alpha", "normalized_beta", "normalized_rho"}:
            return self.abundance.normalized_subcommunity_abundance
        elif measure == "gamma":
            return self.abundance.metacommunity_abundance


class SimilaritySensitiveComponents(Components):
    """Dispatches similarity-sensitive diversity components based on specified measure"""

    def __init__(self, abundance: Abundance, similarity: Similarity) -> None:
        self.abundance = abundance
        self.similarity = similarity

    @cached_property
    def metacommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def get_numerator(self, measure: str) -> Union[int, ndarray]:
        if measure in {"alpha", "gamma", "normalized_alpha"}:
            return 1
        elif measure in {"beta", "rho", "normalized_beta", "normalized_rho"}:
            return self.metacommunity_similarity

    def get_denominator(self, measure: str) -> ndarray:
        if measure in {"alpha", "beta", "rho"}:
            return self.subcommunity_similarity
        elif measure in {"normalized_alpha", "normalized_beta", "normalized_rho"}:
            return self.normalized_subcommunity_similarity
        elif measure == "gamma":
            return self.metacommunity_similarity


def make_components(
    abundance: Abundance,
    similarity: Similarity,
) -> Components:
    if similarity is None:
        return FrequencySensitiveComponents(abundance=abundance)
    else:
        return SimilaritySensitiveComponents(abundance=abundance, similarity=similarity)
