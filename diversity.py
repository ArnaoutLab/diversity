import numpy as np
import csv
from Levenshtein import distance


def sequence_similarity(a, b):
    a, b, = ''.join(a), ''.join(b)
    max_length = np.amax([len(a), len(b)])
    return 1 - (distance(a, b) / max_length)


def calculate_zp(features, n_species, p, similarity_fn):
    z_i = np.empty(n_species, dtype=np.float64)
    zp = np.empty(n_species, dtype=np.float64)
    for i, species_i in enumerate(features):
        for j, species_j in enumerate(features):
            z_i[j] = similarity_fn(species_i, species_j)
        zp[i] = np.dot(z_i, p)
    return zp


def zp_from_file(filepath, p):
    with open(filepath, 'r') as f:
        zp = [np.dot(np.array(z_i, dtype=np.float64), p)
              for z_i in csv.reader(f)]
        return np.array(zp)


def calculate_qDs(zp, p, q):
    if q == 1:
        return 1 / np.prod(zp ** p)
    elif q == np.inf:
        return 1 / np.amax(zp)
    return np.dot(p, zp ** (q - 1)) ** (1 / (1 - q))


def alpha_diversity(features, counts, q, similarity_fn=sequence_similarity, filepath=None):
    n_species = len(features)
    p = np.array(counts) / n_species
    if filepath:
        zp = zp_from_file(filepath, p)
    else:
        zp = calculate_zp(features, n_species, p, similarity_fn)
    qDs = calculate_qDs(zp, p, q)
    return qDs
