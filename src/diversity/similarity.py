"""Module for calculating weighted sub- and metacommunity similarities.

Classes
-------
ISimilarity
    Abstract base class for relative abundance-weighted species
    similarities.
SimilarityFromFile
    Implements Similarity by reading similarities from a file.
SimilarityFromMemory
    Implements Similarity by storing similarities in memory.

Functions
---------
make_similarity
    Chooses and creates instance of concrete ISimilarity implementation.
"""
from abc import ABC, abstractmethod
from numpy import dtype, empty, memmap, ndarray
from pandas import DataFrame, read_csv
from diversity.log import LOGGER
from diversity.utilities import (
    get_file_delimiter,
)


def make_similarity(similarity, chunk_size=100):
    """Initializes a concrete subclass of ISimilarity.

    Parameters
    ----------
    similarity: pandas.DataFrame, str, or Callable
        If pandas.DataFrame, see diversity.similarity.SimilarityFromMemory.
        If str, see diversity.similarity.SimilarityFromFile.
        If callable, see diversity.similarity.SimilarityFromFunction.
    chunk_size: int
        See diversity.similarity.SimilarityFromFile. Only relevant
        if a str is passed as argument for similarity.

    Returns
    -------
    An instance of a concrete subclass of ISimilarity.
    """
    LOGGER.debug(
        "make_similarity(similarity=%s, chunk_size=%s)",
        similarity,
        chunk_size,
    )
    strategies = {
        DataFrame: (SimilarityFromDataFrame, {"similarity": similarity}),
        ndarray: (SimilarityFromArray, {"similarity": similarity}),
        memmap: (SimilarityFromArray, {"similarity": similarity}),
        str: (
            SimilarityFromFile,
            {"similarity": similarity, "chunk_size": chunk_size},
        ),
    }
    strategy_choice = strategies[type(similarity)]
    similarity_class, kwargs = strategy_choice
    return similarity_class(**kwargs)


class ISimilarity(ABC):
    """Interface for classes computing weighted similarities."""

    @abstractmethod
    def calculate_weighted_similarities(self, relative_abundances):
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


class SimilarityFromFile(ISimilarity):
    """Implements ISimilarity by using similarities stored in file.

    Similarity matrix rows are read from the file one chunk at a time.
    The size of chunks can be specified in numbers of rows to control
    memory load.
    """

    def __init__(self, similarity, chunk_size=100):
        """Initializes object.

        Parameters
        ----------
        similarity: str
            Path to similarities file containing a square matrix of
            similarities between species, together with a header
            containing the unique species names in the matrix's row and
            column ordering.
        chunk_size: int
            Number of rows to read from similarity matrix at a time.
        """
        LOGGER.debug(
            "SimilarityFromFile(similarity=%s chunk_size=%s",
            similarity,
            chunk_size,
        )
        self.similarity = similarity
        self.chunk_size = chunk_size
        self.__delimiter = get_file_delimiter(self.similarity)

    def calculate_weighted_similarities(self, relative_abundances):
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


class SimilarityFromDataFrame(ISimilarity):
    """Implements Similarity using similarities stored in pandas dataframe"""

    def __init__(self, similarity):
        """Initializes object.

        similarity: pandas.DataFrame, numpy.ndarray, or numpy.memmap
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        """
        self.similarity = similarity

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity.to_numpy() @ relative_abundances


class SimilarityFromArray(ISimilarity):
    """Implements Similarity using similarities stored in a numpy array"""

    def __init__(self, similarity):
        """Initializes object.

        similarity: pandas.DataFrame, numpy.ndarray, or numpy.memmap
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        """
        self.similarity = similarity

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity @ relative_abundances
