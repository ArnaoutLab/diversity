from pathlib import Path
import numpy as np
import pandas as pd
import csv
from Levenshtein import distance


class Metacommunity:

    def __init__(self, df, q, z_filepath):
        # Input
        self.counts = df.iloc[:, :3]
        self.features = df.iloc[:, 3:].to_numpy()
        self.q = np.array(q)
        self.z_filepath = Path(z_filepath)
        self.similarity_fn = sequence_similarity
        # Diversity components
        self.P = self.relative_abundances().to_numpy()
        self.p = self.P.sum(axis=1).reshape((-1, 1))
        self.w = self.P.sum(axis=0)
        self.P_bar = self.P / self.w
        self.Zp = self.calculate_zp(self.p)
        self.ZP = self.calculate_zp(self.P)
        self.ZP_bar = self.calculate_zp(self.P_bar)

    def relative_abundances(self):
        metacommunity_counts = pd.pivot_table(self.counts, values='count', index='species',
                                              columns='subcommunity', aggfunc='first', fill_value=0.0)
        total_abundance = metacommunity_counts.to_numpy().sum()
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

    def measure(self, numerator, denominator):
        order = 1 - self.q
        x = safe_divide(numerator, denominator)
        measures = []
        for p, x in zip(self.P_bar.T, x.T):
            indices = np.where(p != 0)
            p = p[indices]
            x = x[indices]
            measures.append(power_mean(order, p, x))
        return measures

    def raw_alpha(self):
        return self.measure(1, self.ZP)

    def normalized_alpha(self):
        return self.measure(1, self.ZP_bar)

    def raw_rho(self):
        return self.measure(self.Zp, self.ZP)

    def normalized_rho(self):
        return self.measure(self.Zp, self.ZP_bar)

    def gamma(self):
        return self.measure(1, self.Zp)


def sequence_similarity(a, b):
    a, b, = ''.join(a), ''.join(b)
    max_length = np.amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)


def power_mean(orders, weights, x):
    means = []
    for order in orders:
        if order == 0:
            means.append(np.prod(x ** weights))
        elif order < -100 or order == -np.inf:
            means.append(np.amax(x))
        else:
            means.append(np.sum((x ** order) * weights, axis=0) ** (1 / order))
    return means


def safe_divide(numerator, denominator):
    out = np.zeros(denominator.shape)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)
