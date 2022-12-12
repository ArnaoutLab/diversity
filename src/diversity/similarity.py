"""Module for calculating weighted sub- and metacommunity similarities.

Classes
-------
Similarity
    Abstract base class for relative abundance-weighted species
    similarities.
SimilarityFromFile
    Implements Similarity by reading similarities from a file.
SimilarityFromMemory
    Implements Similarity by storing similarities in memory.

Functions
---------
make_similarity
    Chooses and creates instance of concrete Similarity implementation.
"""
from abc import ABC, abstractmethod
from typing import Callable
from types import FunctionType
from numpy import dtype, ndarray, memmap, empty, concatenate
from pandas import DataFrame, read_csv
from ray import remote, get, put
from diversity.log import LOGGER
from diversity.utilities import (
    get_file_delimiter,
)


class Similarity(ABC):
    """Interface for classes computing weighted similarities."""

    @abstractmethod
    def weighted_similarities(self, relative_abundances: ndarray) -> ndarray:
        """Calculates weighted sums of similarities to each species.

        Parameters
        ----------
        relative_abundances: numpy.ndarray
            Array of shape (n_species, n_communities), where rows
            correspond to unique species, columns correspond to
            (meta-/sub-) communities and each element is the relative
            abundance of a species in a (meta-/sub-)community.

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
    """Implements Similarity using similarities stored in pandas dataframe"""

    def __init__(self, similarity: DataFrame):
        """
        similarity:
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        """
        self.similarity: DataFrame = similarity

    def weighted_similarities(self, relative_abundances: ndarray) -> ndarray:
        return self.similarity.to_numpy() @ relative_abundances


class SimilarityFromArray(Similarity):
    """Implements Similarity using similarities stored in a numpy ndarray"""

    def __init__(self, similarity: ndarray) -> None:
        """
        similarity:
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        """
        self.similarity: ndarray = similarity

    def weighted_similarities(self, relative_abundances: ndarray) -> ndarray:
        return self.similarity @ relative_abundances


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
            Path to similarities file containing a square matrix of
            similarities between species, together with a header
            containing the unique species names in the matrix's row and
            column ordering.
        chunk_size:
            Number of rows to read from similarity matrix at a time.
        """
        LOGGER.debug(
            "SimilarityFromFile(similarity=%s chunk_size=%s",
            similarity,
            chunk_size,
        )
        self.similarity: str = similarity
        self.chunk_size: int = chunk_size
        self.__delimiter = get_file_delimiter(self.similarity)

    def weighted_similarities(self, relative_abundances: ndarray) -> ndarray:
        weighted_similarities = empty(relative_abundances.shape, dtype=dtype("f8"))
        with read_csv(
            self.similarity,
            delimiter=self.__delimiter,
            chunksize=self.chunk_size,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                weighted_similarities[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ relative_abundances
                )
                i += self.chunk_size
        return weighted_similarities


class SimilarityFromFunction:
    """Implements Similarity by calculating similarities with a callable function"""

    def __init__(self, similarity: Callable, X: ndarray, chunk_size: int = 100) -> None:
        self.similarity: Callable = similarity
        self.X: ndarray = X
        self.chunk_size: int = chunk_size

    @remote
    def weighted_similarity_chunk(
        self, X: ndarray, relative_abundances: ndarray, i: int
    ) -> ndarray:
        chunk = X[i : i + self.chunk_size]
        similarities_chunk = empty(shape=(chunk.shape[0], X.shape[0]))
        for i, row_i in enumerate(chunk):
            for j, row_j in enumerate(X):
                similarities_chunk[i, j] = self.similarity(row_i, row_j)
        return similarities_chunk @ relative_abundances

    def weighted_similarities(self, relative_abundances: ndarray) -> ndarray:
        X_ref = put(self.X)
        abundance_ref = put(relative_abundances)
        futures = [
            self.weighted_similarity_chunk.remote(X_ref, abundance_ref, i)
            for i in range(0, self.X.shape[0], self.chunk_size)
        ]
        weighted_similarity_chunks = get(futures)
        return concatenate(weighted_similarity_chunks)


def make_similarity(
    similarity: DataFrame | ndarray | str | Callable | None,
    X: ndarray = None,
    chunk_size: int = 100,
) -> Similarity:
    """Initializes a concrete subclass of Similarity.

    Parameters
    ----------
    similarity:
        If pandas.DataFrame, see diversity.similarity.SimilarityFromDataFrame.
        If numpy.ndarray, see diversity.similarity.SimilarityFromArray.
        If str, see diversity.similarity.SimilarityFromFile.
        If Callable, see diversity.similarity.SimilarityFromFunction
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
    kwargs = {"similarity": similarity}
    similarity_strategies = {
        DataFrame: (SimilarityFromDataFrame, kwargs),
        ndarray: (SimilarityFromArray, kwargs),
        memmap: (SimilarityFromArray, kwargs),
        str: (SimilarityFromFile, kwargs | {"chunk_size": chunk_size}),
        FunctionType: (
            SimilarityFromFunction,
            kwargs | {"X": X, "chunk_size": chunk_size},
        ),
    }
    similarity_class, kwargs = similarity_strategies[type(similarity)]
    return similarity_class(**kwargs)


# def make_similarity(
#     similarity: DataFrame | ndarray | str, X: ndarray = None, chunk_size: int = 100
# ) -> Similarity:
#     """Initializes a concrete subclass of Similarity.

#     Parameters
#     ----------
#     similarity:
#         If pandas.DataFrame, see diversity.similarity.SimilarityFromDataFrame.
#         If numpy.ndarray, see diversity.similarity.SimilarityFromArray.
#         If str, see diversity.similarity.SimilarityFromFile.
#         If Callable, see diversity.similarity.SimilarityFromFunction
#     X:
#         A 2-d array where each row is a species
#     chunk_size:
#         See diversity.similarity.SimilarityFromFile. Only relevant
#         if a str is passed as argument for similarity.

#     Returns
#     -------
#     An instance of a concrete subclass of Similarity.
#     """
#     LOGGER.debug(
#         "make_similarity(similarity=%s, X=%s, chunk_size=%s)",
#         similarity,
#         X,
#         chunk_size,
#     )
#     similarity_type = type(similarity)
#     if similarity is None:
#         return None
#     elif similarity_type is DataFrame:
#         return SimilarityFromDataFrame()
#     elif similarity_type is ndarray or similarity_type is memmap:
#         return SimilarityFromArray()
#     elif similarity_type is str:
#         return SimilarityFromFile()
#     elif callable(similarity):
#         return SimilarityFromFunction()
#     else:
#         raise NotImplementedError(
#             f"Type {type(similarity)} is not supported for argument 'similarity'"
#         )
#     similarity_class = SIMILARITY_STRATEGIES[type(similarity)]
#     kwargs_dict = {"similarity": similarity}
#     kwargs_factory = {
#         SimilarityFromDataFrame: kwargs_dict,
#         SimilarityFromArray: kwargs_dict,
#         SimilarityFromFile: kwargs_dict.update({"chunk_size": chunk_size}),
#         SimilarityFromFunction: kwargs_dict.update({"X": X}),
#     }
#     kwargs = kwargs_factory[similarity_class]
#     return similarity_class(**kwargs)
