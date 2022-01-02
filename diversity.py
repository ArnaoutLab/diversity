"""Module for calculating metacommunity and subcommunity diversity measures.

Classes
-------

Abundance
    Represents the relative abundances or the metacommunity and its subcommunities


Similarity
    Represents the similarities between species weighted by their relative abundances

Metacommunity
    Represents a metacommunity made up of subcommunities and computes metacommunity
    subcommunity diversity measures.
"""
from csv import reader, writer
from pathlib import Path
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable
from numpy import (amin, dot, array, empty, unique, isclose,
                   prod, zeros, sum as numpy_sum, broadcast_to, power,
                   multiply, divide, float64, inf)

########################################################################


@dataclass
class Abundance:

    counts: array

    @cached_property
    def p(self):
        """
        Calculates the relative abundance of each species in the metacommunity
        """
        return self.P.sum(axis=1, keepdims=True)

    @cached_property
    def P(self):
        """Calculates the relative abundance of each species in each subcommunity.

        Returns
        -------
        A 2D numpy.ndarray where rows are unique species and columns are subcommunities
        and each element is a species count in a subcommunity
        """
        rows, row_pos = unique(self.counts[:, 0], return_inverse=True)
        cols, col_pos = unique(self.counts[:, 2], return_inverse=True)
        metacommunity_counts = zeros(
            (len(rows), len(cols)), dtype=float64)
        metacommunity_counts[row_pos, col_pos] = self.counts[:, 1]
        total_abundance = metacommunity_counts.sum()
        return metacommunity_counts / total_abundance

    @cached_property
    def w(self):
        """
        Calculates each subcommunity's relative abundance as a proportion of the 
        total metacommunity's relative abundance
        """
        return self.P.sum(axis=0)

    @cached_property
    def normalized_P(self):
        """
        Calculates normalized subcommunity relative abundances
        """
        return self.P / self.w


@dataclass
class Similarity:

    abundance: Abundance
    z_filepath: str
    similarity_fn: Callable = None
    features: array = None

    @ cached_property
    def Zp(self):
        return self.calculate_zp(self.abundance.p)

    @ cached_property
    def ZP(self):
        return self.calculate_zp(self.abundance.P)

    @ cached_property
    def normalized_ZP(self):
        return self.calculate_zp(self.abundance.normalized_P)

    def write_similarity_matrix(self):
        n_species = self.features.shape[0]
        z_i = empty(n_species, dtype=float64)
        with open(self.z_filepath, 'w') as f:
            csv_writer = writer(f)
            for species_i in self.features:
                for j, species_j in enumerate(self.features):
                    z_i[j] = self.similarity_fn(species_i, species_j)
                csv_writer.writerow(z_i)

    def zp_from_file(self, P):
        ZP = empty(P.shape, dtype=float64)
        with open(self.z_filepath, 'r') as f:
            for i, z_i in enumerate(reader(f)):
                z_i = array(z_i, dtype=float64)
                ZP[i, :] = dot(z_i, P)
        return ZP

    def calculate_zp(self, P):
        if not self.z_filepath.is_file():
            self.write_similarity_matrix()
        return self.zp_from_file(P)


@dataclass
class Metacommunity:
    """Class for metacommunities and calculation their diversity.

    Attributes
    ----------
    counts: numpy.ndarray
    _viewpoint: float
    z_filepath: str
    similarity_fn: Callable
    features: np.ndarray
    abundance: Abundance
    similarity: Similarity
    """

    counts: array
    _viewpoint: float
    z_filepath: str
    similarity_fn: Callable = None
    features: array = None
    abundance: Abundance = field(init=False)
    similarity: Similarity = field(init=False)

    def __post_init__(self):
        self.z_filepath = Path(self.z_filepath)
        self.abundance = Abundance(self.counts)
        self.similarity = Similarity(
            self.abundance, self.z_filepath, self.similarity_fn, self.features)

    # FIXME validate new viewpoint
    def set_viewpoint(self, viewpoint):
        self._viewpoint = viewpoint

    @property
    def alpha(self):
        return self.subcommunity_measure(1, self.similarity.ZP)

    @property
    def rho(self):
        return self.subcommunity_measure(self.similarity.Zp, self.similarity.ZP)

    @property
    def beta(self):
        return 1 / self.rho

    @property
    def gamma(self):
        return self.subcommunity_measure(1, self.similarity.Zp)

    @property
    def normalized_alpha(self):
        return self.subcommunity_measure(1, self.similarity.normalized_ZP)

    @property
    def normalized_rho(self):
        return self.subcommunity_measure(self.similarity.Zp, self.similarity.normalized_ZP)

    @property
    def normalized_beta(self):
        return 1 / self.normalized_rho

    @property
    def A(self):
        return self.metacommunity_measure(self.alpha)

    @property
    def R(self):
        return self.metacommunity_measure(self.rho)

    @property
    def B(self):
        return self.metacommunity_measure(self.beta)

    @property
    def G(self):
        return self.metacommunity_measure(self.gamma)

    @property
    def normalized_A(self):
        return self.metacommunity_measure(self.normalized_alpha)

    @property
    def normalized_R(self):
        return self.metacommunity_measure(self.normalized_rho)

    @property
    def normalized_B(self):
        return self.metacommunity_measure(self.normalized_beta)

    def subcommunity_measure(self, numerator, denominator):
        similarities = divide(numerator, denominator, out=zeros(
            denominator.shape), where=denominator != 0)
        return self.power_mean(self.abundance.normalized_P, similarities)

    def metacommunity_measure(self, subcommunity_measure):
        return self.power_mean(self.abundance.w, subcommunity_measure)

    def power_mean(self, weights, items):
        """Calculates a weighted power mean.

        Parameters
        ----------
        weights: numpy.ndarray
            The weights corresponding to items.
        items: numpy.ndarray
            The elements for which the weighted power mean is computed.

        Returns
        -------
        The power mean of items with exponent order, weighted by weights.
        When order is close to 1 or less than -100, analytical formulas
        for the limits at 1 and -infinity are used respectively.
        """
        order = 1 - self._viewpoint
        mask = weights != 0
        if isclose(order, 0):
            return prod(power(items, weights, where=mask), axis=0, where=mask)
        elif order < -100:
            items = broadcast_to(items, weights.shape)
            return amin(items, axis=0, where=mask, initial=inf)
        items_power = power(items, order, where=mask)
        items_product = multiply(items_power, weights, where=mask)
        items_sum = numpy_sum(items_product, axis=0, where=mask)
        return power(items_sum, 1 / order)

    # FIXME implement me!
    def format_results(self):
        pass
