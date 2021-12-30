from pathlib import Path
import numpy as np
import csv
from Levenshtein import distance


class Metacommunity:

    # FIXME need to check if features are passed, and if not, need to enforce z_filepath to reference similarity matrix
    def __init__(self, counts, q, z_filepath, features=None):
        # Input
        self.counts = counts
        self.features = features
        self.q = np.array(q)
        self.z_filepath = Path(z_filepath)
        self.similarity_fn = sequence_similarity
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


def sequence_similarity(a, b):
    a, b, = ''.join(a), ''.join(b)
    max_length = np.amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)


def power_means(orders, weights, x):
    return [power_mean(order, weights, x) for order in orders]


def power_mean(order, weights, x):
    if order == 0:
        return np.prod(x ** weights)
    elif order < -100:
        return np.amax(x)
    return np.sum((x ** order) * weights, axis=0) ** (1 / order)


def safe_divide(numerator, denominator):
    out = np.zeros(denominator.shape)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)
