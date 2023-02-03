"""Module for calculating weighted subcommunity and metacommunity 
similarities.

Classes
-------
Similarity
    Abstract base class for relative abundance-weighted species
    similarities.
SimilarityFromDataFrame
    Implements Similarity by storing similarities in a pandas DataFrame.
SimilarityFromArray
    Implements Similarity by storing similarities in a numpy ndarray or 
    memmap.
SimilarityFromFile
    Implements Similarity by reading similarities from a csv or tsv 
    file.
SimilarityFromFunction:
    Implements Similarity by calculating pairwise similarities with a 
    callable function.

Functions
---------
make_similarity
    Chooses and creates instances of concrete Similarity 
    implementations.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union
from numpy import ndarray, empty, concatenate, float64
from pandas import DataFrame, read_csv
from scipy.sparse import spmatrix
from ray import remote, get, put


class Similarity(ABC):
    """Interface for classes computing weighted similarities."""

    def __init__(self, similarity: Union[DataFrame, ndarray, str, Callable]) -> None:
        """
        Parameters
        ----------
        abundance:
            Contains the relative abundances for the metacommunity and
            its subcomunities.
        similarity:
            A pairwise similarity matrix of shape (n_species, n_species)
            where each value is the similarity between a pair of
            species. Species must be in the same order as in the counts
            argument of the Metacommunity class.
        """
        self.similarity = similarity

    @abstractmethod
    def weighted_similarities(
        self, relative_abundances: Union[ndarray, spmatrix]
    ) -> ndarray:
        """Calculates weighted sums of similarities to each species.

        Returns
        -------
        A 2-d numpy.ndarray of shape (n_species, n_communities), where
        rows correspond to unique species, columns correspond to
        (meta-/sub-) communities and each element is a sum of
        similarities of all species to the row's species weighted by
        their relative abundances in the respective communities.
        """
        pass


class SimilarityFromDataFrame(Similarity):
    """Implements Similarity using similarities stored in pandas
    dataframe."""

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity.to_numpy() @ relative_abundance


class SimilarityFromArray(Similarity):
    """Implements Similarity using similarities stored in a numpy
    ndarray."""

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity @ relative_abundance


class SimilarityFromFile(Similarity):
    """Implements Similarity by using similarities stored in file.

    Similarity matrix rows are read from the file one chunk at a time.
    The size of chunks can be specified in numbers of rows to control
    memory load.
    """

    def __init__(self, similarity: str, chunk_size: int = 100) -> None:
        """
        Parameters
        ----------
        similarity:
            Path to a file containing a pairwise similarity matrix of
            shape (n_species, n_species). The file should have a header
            that denotes the unique species names.
        chunk_size:
            Number of rows to read from similarity matrix at a time.
        """
        super().__init__(similarity=similarity)
        self.chunk_size = chunk_size

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        weighted_similarities = empty(relative_abundance.shape, dtype=float64)
        with read_csv(
            self.similarity,
            chunksize=self.chunk_size,
            sep=None,
            engine="python",
            dtype=float64,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                weighted_similarities[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ relative_abundance
                )
                i += self.chunk_size
        return weighted_similarities


@remote
def weighted_similarity_chunk(
    similarity: Callable,
    X: ndarray,
    relative_abundance: ndarray,
    chunk_size: int,
    chunk_index: int,
) -> ndarray:
    chunk = X[chunk_index : chunk_index + chunk_size]
    similarities_chunk = empty(shape=(chunk.shape[0], X.shape[0]))
    for i, row_i in enumerate(chunk):
        for j, row_j in enumerate(X):
            similarities_chunk[i, j] = similarity(row_i, row_j)
    return similarities_chunk @ relative_abundance


class SimilarityFromFunction(Similarity):
    """Implements Similarity by calculating similarities with a callable
    function."""

    def __init__(
        self,
        similarity: Callable,
        X: ndarray,
        chunk_size: int = 100,
    ) -> None:
        """
        similarity:
            A Callable that calculates similarity between a pair of
            species. Must take two rows from X and return a numeric
            similarity value.
        X:
            An array where each row contains the feature values for a
            given species.
        chunk_size:
            Determines how many rows of the similarity matrix each will
            be processes at a time. In general, choosing a larger
            chunk_size will make the calculation faster, but will also
            require more memory.
        """
        super().__init__(similarity=similarity)
        self.X = X
        self.chunk_size = chunk_size

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        X_ref = put(self.X)
        abundance_ref = put(relative_abundance)
        futures = []
        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.similarity,
                X=X_ref,
                relative_abundance=abundance_ref,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
            )
            futures.append(chunk_future)
        weighted_similarity_chunks = get(futures)
        return concatenate(weighted_similarity_chunks)


def make_similarity(
    similarity: Union[DataFrame, ndarray, str, Callable],
    X: ndarray = None,
    chunk_size: int = 100,
) -> Similarity:
    """Initializes a concrete subclass of Similarity.

    Parameters
    ----------
    similarity:
        If pandas.DataFrame, see
        diversity.similarity.SimilarityFromDataFrame. If numpy.ndarray,
        see diversity.similarity.SimilarityFromArray. If str, see
        diversity.similarity.SimilarityFromFile. If Callable, see
        diversity.similarity.SimilarityFromFunction
    X:
        A 2-d array where each row is a species
    chunk_size:
        See diversity.similarity.SimilarityFromFile. Only relevant
        if a str is passed as argument for similarity.

    Returns
    -------
    An instance of a concrete subclass of Similarity.
    """
    if similarity is None:
        return None
    elif isinstance(similarity, DataFrame):
        return SimilarityFromDataFrame(similarity=similarity)
    elif isinstance(similarity, ndarray):
        return SimilarityFromArray(similarity=similarity)
    elif isinstance(similarity, str):
        return SimilarityFromFile(similarity=similarity, chunk_size=chunk_size)
    elif isinstance(similarity, Callable):
        return SimilarityFromFunction(similarity=similarity, X=X, chunk_size=chunk_size)
    else:
        raise NotImplementedError(
            f"Type {type(similarity)} is not supported for argument "
            "'similarity'. Valid types include pandas.DataFrame, "
            "numpy.ndarray, numpy.memmap, str, or typing.Callable"
        )
