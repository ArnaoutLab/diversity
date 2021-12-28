from pathlib import Path
import numpy as np
import pandas as pd
import csv
from Levenshtein import distance


# FIXME Levenshtein probably shouldn't be a dependency. Instead, we should define our similarity function outside of morty
def sequence_similarity(a, b):
    a, b, = ''.join(a), ''.join(b)
    max_length = np.amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)


def relative_abundances(df):
    metacommunity_counts = pd.pivot_table(df, values='count', index='species',
                                          columns='subcommunity', aggfunc='first', fill_value=0.0)
    total_abundance = metacommunity_counts.to_numpy().sum()
    return metacommunity_counts / total_abundance


def write_similarity_matrix(Z_filepath, features, similarity_fn):
    n_species = features.shape[0]
    z_i = np.empty(n_species, dtype=np.float64)
    with open(Z_filepath, 'w') as f:
        writer = csv.writer(f)
        for i, species_i in enumerate(features):
            for j, species_j in enumerate(features):
                z_i[j] = similarity_fn(species_i, species_j)
            writer.writerow(z_i)


def ZP_from_file(Z_filepath, P):
    ZP = np.empty(P.shape, dtype=np.float64)
    with open(Z_filepath, 'r') as f:
        for i, z_i in enumerate(csv.reader(f)):
            z_i = np.array(z_i, dtype=np.float64)
            ZP[i, :] = np.dot(z_i, P)
    return ZP


def calculate_ZP(Z_filepath, P, features, similarity_fn):
    if not Path(Z_filepath).is_file():
        write_similarity_matrix(Z_filepath, features, similarity_fn)
    return ZP_from_file(Z_filepath, P)


# FIXME the cases for q = 1 and q = infinity may be specific to the diversity measure, in which case each measure function would need its own set of conditionals
def power_mean(order, weights, x):
    indices = np.where(weights != 0)
    weights = weights[indices]
    x = x[indices]
    if order == 0:
        return np.prod(x ** weights)
    elif order < -100 or order == -np.inf:
        return np.amax(x)
    return np.sum((x ** order) * weights, axis=0) ** (1 / order)


# FIXME eventually remove sequence_similarity() as default similarity function
def raw_alpha(df, q, z_filepath, similarity_fn=sequence_similarity):
    features = df.iloc[:, 3:].to_numpy()
    order = 1 - q
    P = relative_abundances(df.iloc[:, :3]).to_numpy()
    w = P.sum(axis=0)
    P_bar = P / w
    ZP = calculate_ZP(z_filepath, P, features, similarity_fn)
    inverse_ZP = np.divide(1, ZP, out=ZP, where=ZP != 0)
    return [power_mean(order, p, zp) for p, zp in zip(P_bar.T, inverse_ZP.T)]


def normalized_alpha(df, q, z_filepath, similarity_fn=sequence_similarity):
    features = df.iloc[:, 3:].to_numpy()
    order = 1 - q
    P = relative_abundances(df.iloc[:, :3]).to_numpy()
    w = P.sum(axis=0)
    P_bar = P / w
    ZP_bar = calculate_ZP(z_filepath, P_bar, features, similarity_fn)
    inverse_ZP_bar = np.divide(1, ZP_bar, out=ZP_bar, where=ZP_bar != 0)
    return [power_mean(order, p, zp) for p, zp in zip(P_bar.T, inverse_ZP_bar.T)]


def raw_rho(df, q, z_filepath, similarity_fn=sequence_similarity):
    features = df.iloc[:, 3:].to_numpy()
    order = 1 - q
    P = relative_abundances(df.iloc[:, :3]).to_numpy()
    p = P.sum(axis=1).reshape((-1, 1))
    w = P.sum(axis=0)
    P_bar = P / w
    Zp = calculate_ZP(z_filepath, p, features, similarity_fn)
    ZP = calculate_ZP(z_filepath, P, features, similarity_fn)
    Zp_over_ZP = np.divide(Zp, ZP, out=ZP, where=ZP != 0)
    return [power_mean(order, p, zp) for p, zp in zip(P_bar.T, Zp_over_ZP.T)]


def normalized_rho(df, q, z_filepath, similarity_fn=sequence_similarity):
    features = df.iloc[:, 3:].to_numpy()
    order = 1 - q
    P = relative_abundances(df.iloc[:, :3]).to_numpy()
    p = P.sum(axis=1).reshape((-1, 1))
    w = P.sum(axis=0)
    P_bar = P / w
    Zp = calculate_ZP(z_filepath, p, features, similarity_fn)
    ZP_bar = calculate_ZP(z_filepath, P_bar, features, similarity_fn)
    Zp_over_ZP_bar = np.divide(Zp, ZP_bar, out=ZP_bar, where=ZP_bar != 0)
    return [power_mean(order, p, zp) for p, zp in zip(P_bar.T, Zp_over_ZP_bar.T)]


def gamma(df, q, z_filepath, similarity_fn=sequence_similarity):
    features = df.iloc[:, 3:].to_numpy()
    order = 1 - q
    P = relative_abundances(df.iloc[:, :3]).to_numpy()
    p = P.sum(axis=1).reshape((-1, 1))
    w = P.sum(axis=0)
    P_bar = P / w
    Zp = calculate_ZP(z_filepath, p, features, similarity_fn)
    inverse_Zp = 1 / Zp
    return [power_mean(order, p, zp) for p, zp in zip(P_bar.T, inverse_Zp.T)]
