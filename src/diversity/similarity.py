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
from collections.abc import Sequence
from functools import cached_property

from numpy import dtype, empty, memmap, ndarray
from pandas import DataFrame, read_csv

from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.utilities import (
    get_file_delimiter,
)


def make_similarity(
    similarity,
    species_order=None,
    chunk_size=None,
):
    """Initializes a concrete subclass of ISimilarity.

    Parameters
    ----------
    similarity: pandas.DataFrame, str, or Callable
        If pandas.DataFrame, see diversity.similarity.SimilarityFromMemory.
        If str, see diversity.similarity.SimilarityFromFile.
        If callable, see diversity.similarity.SimilarityFromFunction.
    species_order: a collection of str objects used to determine
        the species ordering
    chunk_size: int
        See diversity.similarity.SimilarityFromFile. Only relevant
        if a str is passed as argument for similarity.

    Returns
    -------
    An instance of a concrete subclass of ISimilarity.
    """
    LOGGER.debug(
        "make_similarity(similarity=%s, species_order=%s, chunk_size=%s)",
        similarity,
        species_order,
        chunk_size,
    )
    strategies = {
        DataFrame: (SimilarityFromDataFrame, {"similarity": similarity}),
        ndarray: (
            SimilarityFromArray,
            {"similarity": similarity, "species_order": species_order},
        ),
        memmap: (
            SimilarityFromMemmap,
            {"similarity": similarity, "species_order": species_order},
        ),
        str: (
            SimilarityFromFile,
            {
                "similarity": similarity,
                "chunk_size": chunk_size,
            },
        ),
    }
    strategy_choice = strategies[type(similarity)]
    similarity_class, kwargs = strategy_choice
    similarity = similarity_class(**kwargs)
    return similarity


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

    @property
    @abstractmethod
    def species_ordering(self):
        """The ordering of species used by the object.

        Returns
        -------
        A 1-d numpy.ndarray of species names in the ordering used for
        the return value of the object's .calculate_weighted_similarities
        method.
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
        self.__species_ordering = self.__get_species_ordering()

    def __get_species_ordering(self):
        """The species ordering used in similarity matrix file.

        Parameters
        ----------
        species_subset: collection of str objects, which supports membership test
            Set of species to include. If None, all species are
            included.

        Returns
        -------
        A tuple consisting of
        0 - numpy.ndarray (1d)
            Uniqued species ordered according to the similarity matrix
            file header.
        1 - numpy.ndarray (1d)
            Column numbers (0-based) of columns corresponding to members
            of species, or None if species is None.
        2 - numpy.ndarray (1d)
            Row numbers (0-based, header counts as row 0) of rows
            corresponding to non-members of species, or None if species
            is None.
        """
        with read_csv(
            self.similarity, delimiter=self.__delimiter, chunksize=self.chunk_size
        ) as similarity_matrix_header:
            return next(similarity_matrix_header).columns.astype(str)

    @cached_property
    def species_ordering(self):
        return self.__species_ordering

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


class SimilarityFromMemory(ISimilarity):
    """Implements Similarity using similarities stored in memory."""

    def __init__(self, similarity):
        """Initializes object.

        similarity: pandas.DataFrame, numpy.ndarray, or numpy.memmap
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        """
        self.similarity = similarity

    @property
    def species_ordering(self):
        pass

    def get_species_ordering(self, species_order):
        if (len(species_order) == self.similarity.shape[0]) and isinstance(
            species_order, Sequence
        ):
            return species_order
        raise InvalidArgumentError(
            """species_order must inherit from class Sequence and have length equal 
            to the dimensions of the similarity matrix"""
        )

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity.to_numpy() @ relative_abundances


class SimilarityFromDataFrame(SimilarityFromMemory):
    def __init__(self, similarity):
        super().__init__(similarity)
        self.__species_ordering = self.similarity.columns.astype(str)

    @property
    def species_ordering(self):
        return self.__species_ordering


class SimilarityFromArray(SimilarityFromMemory):
    """
    species_subset: Sequence, or collection of str objects, which supports membership test
        The species to include. Only similarities from columns and
        rows corresponding to these species are used. If
        numpy.ndarray, or numpy.memmap is used as the matrix, then
        species_subset must be a Sequence of the same length as
        rows/columns in the similarity matrix.
    """

    def __init__(self, similarity, species_order):
        super().__init__(similarity)
        self.__species_ordering = self.get_species_ordering(species_order)

    @property
    def species_ordering(self):
        return self.__species_ordering

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity @ relative_abundances


class SimilarityFromMemmap(SimilarityFromMemory):
    """
    species_subset: Sequence, or collection of str objects, which supports membership test
        The species to include. Only similarities from columns and
        rows corresponding to these species are used. If
        numpy.ndarray, or numpy.memmap is used as the matrix, then
        species_subset must be a Sequence of the same length as
        rows/columns in the similarity matrix.
    """

    def __init__(self, similarity, species_order):
        super().__init__(similarity)
        self.__species_ordering = self.get_species_ordering(species_order)

    @property
    def species_ordering(self):
        return self.__species_ordering

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity @ relative_abundances
