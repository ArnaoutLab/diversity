import numpy as np
import csv
from Levenshtein import distance
import warnings


def sequence_similarity(a, b):
    a, b, = ''.join(a), ''.join(b)
    max_length = np.amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)


def zp_from_similarity_fn(p, features, n_species, similarity_fn):
    z_i = np.empty(n_species, dtype=np.float64)
    zp = np.empty(n_species, dtype=np.float64)
    for i, species_i in enumerate(features):
        for j, species_j in enumerate(features):
            z_i[j] = similarity_fn(species_i, species_j)
        zp[i] = np.dot(z_i, p)
    return zp


def zp_from_file(p, filepath):
    with open(filepath, 'r') as f:
        zp = [np.dot(np.array(z_i, dtype=np.float64), p)
              for z_i in csv.reader(f)]
        return np.array(zp)


def calculate_zp(p, features, n_species, filepath, similarity_fn):
    if filepath:
        return zp_from_file(p, filepath)
    else:
        return zp_from_similarity_fn(p, features, n_species, similarity_fn)


def power_mean(order, weights, x):
    if order == 0:
        return np.prod(x ** weights)
    elif order == -np.inf:
        return np.amax(x)
    elif order < -100:
        warnings.warn(
            "q > 100.0 defaults to the analytical formula for q = inf")
        return np.amax(x)
    return np.dot(weights, x ** order) ** (1 / order)


def alpha(features, counts, q, similarity_fn=sequence_similarity, filepath=None):
    n_species = len(features)
    weights = np.array(counts) / n_species
    zp = calculate_zp(weights, features, n_species, filepath, similarity_fn)
    order = 1 - q
    x = 1 / zp
    qDs = power_mean(order, weights, x)
    return qDs
