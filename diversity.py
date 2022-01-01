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
from csv import reader, writer
from pathlib import Path
from numpy import (amin, dot, array, empty, unique, where,
                   prod, zeros, sum as numpy_sum, divide, float64, inf)

########################################################################


class Abundance:

    def __init__(self, counts):
        self.P = self.subcommunities_relative_abundances(counts)
        self.p = self.metacommunity_relative_abundances()
        self.w = self.subcommunty_proportions()
        self.P_bar = self.normalized_subcommunities_relative_abundances()

    def subcommunities_relative_abundances(self, counts):
        rows, row_pos = unique(counts[:, 0], return_inverse=True)
        cols, col_pos = unique(counts[:, 2], return_inverse=True)
        metacommunity_counts = zeros(
            (len(rows), len(cols)), dtype=float64)
        metacommunity_counts[row_pos, col_pos] = counts[:, 1]
        total_abundance = metacommunity_counts.sum()
        return metacommunity_counts / total_abundance

    def metacommunity_relative_abundances(self):
        return self.P.sum(axis=1, keepdims=True)

    def subcommunty_proportions(self):
        return self.P.sum(axis=0)

    def normalized_subcommunities_relative_abundances(self):
        return self.P / self.w


class Similarity:

    def __init__(self, abundances, z_filepath, similarity_fn=None, features=None):
        self.abundances = abundances
        self.z_filepath = Path(z_filepath)
        self.similarity_fn = similarity_fn
        self.features = features
        self.Zp = self.calculate_zp(self.abundances.p)
        self.ZP = self.calculate_zp(self.abundances.P)
        self.ZP_bar = self.calculate_zp(self.abundances.P_bar)

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

    def __init__(self, counts, q, z_filepath, similarity_fn=None, features=None):
        # Input
        self.counts = counts
        self.q = array(q)
        # Diversity components
        self.abundance = Abundance(self.counts)
        self.similarity = Similarity(
            self.abundance, z_filepath, similarity_fn, features)
        # Subcommunity diversity measures
        self.alpha = self.calculate_alpha()
        self.rho = self.calculate_rho()
        self.beta = self.calculate_beta()
        self.gamma = self.calculate_gamma()
        self.normalized_alpha = self.calculate_normalized_alpha()
        self.normalized_rho = self.calculate_normalized_rho()
        self.normalized_beta = self.calculate_normalized_beta()
        # Metacommunity diversity measures
        self.A = self.calculate_A()
        self.R = self.calculate_R()
        self.B = self.calculate_B()
        self.G = self.calculate_G()
        self.normalized_A = self.calculate_normalized_A()
        self.normalized_R = self.calculate_normalized_R()
        self.normalized_B = self.calculate_normalized_B()

    def calculate_alpha(self):
        return self.subcommunity_measure(1, self.similarity.ZP)

    def calculate_rho(self):
        return self.subcommunity_measure(self.similarity.Zp, self.similarity.ZP)

    def calculate_beta(self):
        return 1 / self.rho

    def calculate_gamma(self):
        return self.subcommunity_measure(1, self.similarity.Zp)

    def calculate_normalized_alpha(self):
        return self.subcommunity_measure(1, self.similarity.ZP_bar)

    def calculate_normalized_rho(self):
        return self.subcommunity_measure(self.similarity.Zp, self.similarity.ZP_bar)

    def calculate_normalized_beta(self):
        return 1 / self.normalized_rho

    def calculate_A(self):
        return self.metacommunity_measure(self.alpha)

    def calculate_R(self):
        return self.metacommunity_measure(self.rho)

    def calculate_B(self):
        return self.metacommunity_measure(self.beta)

    def calculate_G(self):
        return self.metacommunity_measure(self.gamma)

    def calculate_normalized_A(self):
        return self.metacommunity_measure(self.normalized_alpha)

    def calculate_normalized_R(self):
        return self.metacommunity_measure(self.normalized_rho)

    def calculate_normalized_B(self):
        return self.metacommunity_measure(self.normalized_beta)

    def subcommunity_measure(self, numerator, denominator):
        order = 1 - self.q
        x = safe_divide(numerator, denominator)
        measures = []
        for p, x in zip(self.abundance.P_bar.T, x.T):
            indices = where(p != 0)
            p = p[indices]
            x = x[indices]
            measures.append(power_means(order, p, x))
        return array(measures)

    def metacommunity_measure(self, subcommunity_measures):
        orders = 1 - self.q
        return [power_mean(order, self.abundance.w, measure) for order, measure in zip(orders, subcommunity_measures.T)]

    # FIXME implement me!
    def format_results(self):
        pass


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
    elif order < -100 or order == -inf:
        return amin(items)
    return numpy_sum((items ** order) * weights, axis=0) ** (1 / order)

########################################################################


def power_means(orders, weights, x):
    """Calculates power means for multiple exponents.

    Parameters
    ----------
    orders: iterable
        The exponents for which to calculate power means.
    weights, items
        See metacommunity.diversity.power_mean.

    Returns
    -------
    list of return values of metacommunity.diversity.power_mean for the
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
