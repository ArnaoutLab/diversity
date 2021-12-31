"""Module for calculating diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity and computes diversity measures.

Functions
---------
power_mean
    Calculates weighted power means.
safe_divide
    Divides 2 numpy.arrays avoiding 0-division.
sequence_similarity
    Calculates a similarity measure between two sequences of characters.
"""
from pathlib import Path
from numpy import (
    amax,
    divide,
    prod,
    sum as numpy_sum,
    zeros,
    )
import pandas as pd
import csv
from Levenshtein import distance

########################################################################
class Metacommunity:
    """Class for metacommunities and calculation their diversity.

    Attributes
    ----------
    counts: numpy.ndarray
    features: numpy.ndarray
    q: numpy.ndarray
    z_filepath: pathlib.Path
    similarity_fn: callable
    """
    # FIXME need to check if features are passed, and if not, need to enforce z_filepath to reference similarity matrix
    # FIXME Rename q -> suggestions: large_species_bias, viewpoint, inverse_order, order
    def __init__(self, counts, q, z_filepath, features=None):
        # Input
        self.counts = counts
        self.features = features
        self.q = np.array(q)
        self.z_filepath = Path(z_filepath)
        self.similarity_fn = sequence_similarity # FIXME no custom similarity function?
        # Diversity components
        self.P = self.relative_abundances()
        self.p = self.P.sum(axis=1).reshape((-1, 1))
        self.w = self.P.sum(axis=0)
        self.P_bar = self.P / self.w
        self.Zp = self.calculate_zp(self.p)
        self.ZP = self.calculate_zp(self.P)
        self.ZP_bar = self.calculate_zp(self.P_bar)
        # Subcommunity diversity measures
        self.raw_alpha = self.calculate_raw_alpha()
        self.raw_rho = self.calculate_raw_rho()
        self.raw_beta = 1 / self.raw_rho
        self.gamma = self.calculate_gamma()
        self.normalized_alpha = self.calculate_normalized_alpha()
        self.normalized_rho = self.calculate_normalized_rho()
        self.normalized_beta = 1 / self.normalized_rho
        # Metacommunity diversity measures
        self.A = self.metacommunity_measure(self.raw_alpha)
        self.R = self.metacommunity_measure(self.raw_rho)
        self.B = self.metacommunity_measure(self.raw_beta)
        self.G = self.metacommunity_measure(self.gamma)
        self.normalized_B = self.metacommunity_measure(self.normalized_beta)
        self.normalized_A = self.metacommunity_measure(self.normalized_alpha)
        self.normalized_R = self.metacommunity_measure(self.normalized_rho)

    def relative_abundances(self):
        rows, row_pos = np.unique(self.counts[:, 0], return_inverse=True)
        cols, col_pos = np.unique(self.counts[:, 2], return_inverse=True)
        metacommunity_counts = np.zeros(
            (len(rows), len(cols)), dtype=np.float64)
        metacommunity_counts[row_pos, col_pos] = self.counts[:, 1]
        total_abundance = metacommunity_counts.sum()
        return metacommunity_counts / total_abundance

    def write_similarity_matrix(self):
        n_species = self.features.shape[0]
        z_i = np.empty(n_species, dtype=np.float64)
        with open(self.z_filepath, 'w') as f:
            writer = csv.writer(f)
            for species_i in self.features:
                for j, species_j in enumerate(self.features):
                    z_i[j] = self.similarity_fn(species_i, species_j)
                writer.writerow(z_i)

    def zp_from_file(self, P):
        ZP = np.empty(P.shape, dtype=np.float64)
        with open(self.z_filepath, 'r') as f:
            for i, z_i in enumerate(csv.reader(f)):
                z_i = np.array(z_i, dtype=np.float64)
                ZP[i, :] = np.dot(z_i, P)
        return ZP

    def calculate_zp(self, P):
        if not Path(self.z_filepath).is_file():
            self.write_similarity_matrix()
        return self.zp_from_file(P)

    def subcommunity_measure(self, numerator, denominator):
        order = 1 - self.q
        x = safe_divide(numerator, denominator)
        measures = []
        for p, x in zip(self.P_bar.T, x.T):
            indices = np.where(p != 0)
            p = p[indices]
            x = x[indices]
            measures.append(power_means(order, p, x))
        return np.array(measures)

    def metacommunity_measure(self, subcommunity_measure):
        orders = 1 - self.q
        return [power_mean(order, self.w, measure) for order, measure in zip(orders, subcommunity_measure.T)]

    def calculate_raw_alpha(self):
        return self.subcommunity_measure(1, self.ZP)

    def calculate_normalized_alpha(self):
        return self.subcommunity_measure(1, self.ZP_bar)

    def calculate_raw_rho(self):
        return self.subcommunity_measure(self.Zp, self.ZP)

    def calculate_normalized_rho(self):
        return self.subcommunity_measure(self.Zp, self.ZP_bar)

    def calculate_gamma(self):
        return self.subcommunity_measure(1, self.Zp)

########################################################################
def sequence_similarity(a, b):
    """Calculates a Levenshtein distance derived similarity measure.

    Parameters
    ----------
    a: Iterable
        Sequence of characters comprising the first argument of the
        similarity function.
    b: Iterable
        Sequence of characters comprising the second argument of the
        similarity function.

    Result
    ------
    1 - (Levenshtein distance between concatenations of a and b divided
    by length of the longer of the two sequences).
    """
    a, b, = ''.join(a), ''.join(b)
    max_length = amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)

########################################################################
def power_mean(order, weights, items):
    """Calculates a weighted power mean.

    Parameters
    ----------
    order: numeric
        Exponent for the power mean.
    weights: numpy.ndarray
        The weights corresponding to items.
    items: numpy.ndarray
        The elements for which the weighted power mean is computed.

    Returns
    -------
    The power mean of items with exponent order, weighted by weights.
    When order is less than -100, it is treated as the special case
    where order is -infinity.
    """
    if order == 0:
        return prod(items ** weights)
    elif order < -100 or order == -np.inf:
        return amax(items)
    return numpy_sum((items ** order) * weights, axis=0) ** (1 / order)

########################################################################
def power_means(orders, weights, x):
    """Calculates power means for multiple exponents.

    Parameters
    ----------
    orders: iterable
        The exponents for which to calculate power means.
    weights, items
        See Chubacabra.diversity.power_mean.

    Returns
    -------
    list of return values of Chubacabra.diversity.power_mean for the
    different exponents.
    """
    return [power_mean(order, weights, x) for order in orders]

########################################################################
def safe_divide(numerator, denominator):
    """Divides two numpy.ndarray instances, avoiding 0-divisions.

    Parameters
    ----------
    numerator: numpy.ndarray
        Dividend array.
    denominator: numpy.ndarray
        Divisor array.

    Returns
    -------
    numpy.ndarray of element-wise quotients of numerator elements
    divided by denominator elements where denominator elements are
    non-zero and 0s where denominator elements are zero.
    """
    out = zeros(denominator.shape)
    return divide(numerator, denominator, out=out, where=denominator != 0)
