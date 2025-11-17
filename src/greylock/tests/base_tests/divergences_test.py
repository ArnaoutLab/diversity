from greylock.divergences import exp_relative_entropy
from greylock.similarity import SimilarityFromFunction, SimilarityFromSymmetricFunction,\
SimilarityFromFile, SimilarityFromDataFrame
import numpy as np
import pandas as pd

def test_exp_relative_entropy_no_similarity():
    counts_1 = np.array([[9/25], [12/25], [4/25]])
    counts_2 = np.array([[1/3], [1/3], [1/3]])

    results_default_viewpoint = exp_relative_entropy(counts_2, counts_1)
    results_viewpoint_2 = exp_relative_entropy(counts_2, counts_1, viewpoint=2)

    assert np.allclose(results_default_viewpoint[0], 1.1023618416445828, atol=1e-8)
    assert np.allclose(results_default_viewpoint[0], results_default_viewpoint[1].iloc[0,0], rtol=1e-5)
    assert results_viewpoint_2[0] > results_default_viewpoint[0]


def test_exp_relative_entropy_with_similarity_from_array():
    labels = ["owl", "eagle", "flamingo", "swan", "duck", "chicken", "turkey", "dodo", "dove"]
    no_species = len(labels)
    S = np.identity(n=no_species)


    S[0][1:9] = (0.91, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88) # owl
    S[1][2:9] = (      0.88, 0.89, 0.88, 0.88, 0.88, 0.89, 0.88) # eagle
    S[2][3:9] = (            0.90, 0.89, 0.88, 0.88, 0.88, 0.89) # flamingo
    S[3][4:9] = (                  0.92, 0.90, 0.89, 0.88, 0.88) # swan
    S[4][5:9] = (                        0.91, 0.89, 0.88, 0.88) # duck
    S[5][6:9] = (                              0.92, 0.88, 0.88) # chicken
    S[6][7:9] = (                                    0.89, 0.88) # turkey
    S[7][8:9] = (                                          0.88) # dodo
                                                                    # dove

    S = np.maximum(S, S.transpose() )
    counts_1 = pd.DataFrame({"Community": [1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=labels)
    counts_2 = pd.DataFrame({"Community": [1, 2, 1, 1, 1, 1, 1, 2, 1]}, index=labels)
    result_default_viewpoint = exp_relative_entropy(counts_1, counts_2, similarity=S, viewpoint=1)
    assert np.allclose(result_default_viewpoint[0], 1.0004668803029282)


def test_exp_relative_entropy_with_similarity_from_function():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    def similarity_function(species_i, species_j):
        return np.exp(-np.linalg.norm(species_i - species_j))

    counts_1 = pd.DataFrame({'community_1': [1,1,0], 'community_2': [1,0,1]})
    counts_2 = pd.DataFrame({'community_1': [2,1,0], 'community_2': [2,0,1]})

    results = exp_relative_entropy(counts_2, counts_1, \
    similarity=SimilarityFromFunction(similarity_function, X=X))

    assert np.allclose(results[0], 1.0655322169685402, atol=1e-8)