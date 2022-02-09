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
from functools import cached_property
from multiprocessing import cpu_count, Pool

from numpy import dtype, empty, flatnonzero, where
from pandas import DataFrame, read_csv

from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import LoadSharedArray
from diversity.utilities import (
    get_file_delimiter,
    partition_range,
)


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


def make_similarity(similarity_matrix, species_subset, chunk_size):
    similarity_type = type(similarity_matrix)
    if similarity_type not in {DataFrame, str}:
        raise InvalidArgumentError(
            "similarity_matrix must be a str or a pandas.DataFrame, but"
            f"was: {similarity_type}."
        )
    similarity_classes = {DataFrame: SimilarityFromMemory, str: SimilarityFromFile}
    similarity_arguments = {
        DataFrame: (similarity_matrix, species_subset),
        str: (similarity_matrix, species_subset, chunk_size),
    }
    similarity_class = similarity_classes[similarity_type]
    initializer_arguments = similarity_arguments[similarity_type]
    return similarity_class(*initializer_arguments)


class SimilarityFromFile(ISimilarity):
    """Implements ISimilarity by using similarities stored in file.

    Similarity matrix rows are read from the file one chunk at a time.
    The size of chunks can be specified in numbers of rows to control
    memory load.
    """

    def __init__(self, similarity_matrix, species_subset, chunk_size=1):
        """Initializes object.

        Parameters
        ----------
        similarity_matrix: str
            Path to similarities file containing a square matrix of
            similarities between species, together with a header
            containing the unique species names in the matrix's row and
            column ordering.
        species_subset: collection of str objects supporting membership test
            The species to include. Only similarities from columns and
            rows corresponding to these species are used.
        chunk_size: int
            Number of rows to read from similarity matrix at a time.
        """
        LOGGER.debug(
            "SimilarityFromFile(similarity_matrix=%s, species_subset=%s, chunk_size=%s",
            similarity_matrix,
            species_subset,
            chunk_size,
        )
        self.similarity_matrix = similarity_matrix
        self.__delimiter = get_file_delimiter(self.similarity_matrix)
        self.chunk_size = chunk_size
        (
            self.__species_ordering,
            self.__usecols,
            self.__skiprows,
        ) = self.__get_species_ordering(species_subset)

    def __get_species_ordering(self, species_subset):
        """The species ordering used in similarity matrix file.

        Parameters
        ----------
        species_subset: collection of str objects supporting membership test
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
        LOGGER.debug(
            "SimilarityFromFile.__get_species_ordering(%s, %s)" % (self, species_subset)
        )
        with read_csv(
            self.similarity_matrix, delimiter=self.__delimiter, chunksize=1
        ) as similarity_matrix_chunks:
            species = next(similarity_matrix_chunks).columns.astype(str)
        species_subset_indices = species.isin(species_subset)
        species_ordering = species[species_subset_indices]
        usecols = flatnonzero(species_subset_indices)
        skiprows = flatnonzero(~species_subset_indices) + 1
        return species_ordering, usecols, skiprows

    @cached_property
    def species_ordering(self):
        return self.__species_ordering

    def calculate_weighted_similarities(self, relative_abundances):
        weighted_similarities = empty(relative_abundances.shape, dtype=dtype("f8"))
        with read_csv(
            self.similarity_matrix,
            delimiter=self.__delimiter,
            chunksize=self.chunk_size,
            usecols=self.__usecols,
            skiprows=self.__skiprows,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                weighted_similarities[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ relative_abundances
                )
                i += self.chunk_size
        return weighted_similarities


class SimilarityFromFunction(ISimilarity):
    """Implements ISimilarity using a similarity function."""

    class ApplySimilarityFunction:
        """Applies similarity function to a chunk of data."""

        def __init__(self, func, valid_features_index):
            LOGGER.debug(
                "ApplySimilarityFunction(func=%s, valid_features_index=%s)",
                func,
                valid_features_index,
            )
            self.func = func
            self.__valid_features_index = valid_features_index

        def __call__(
            self,
            row_start,
            row_stop,
            weighted_similarities_spec,
            features_spec,
            relative_abundance_spec,
        ):
            """Computes pairwise weighted similarities for a data chunk.

            Parameters
            ----------
            row_start, row_stop: int
                Right-exclusive start and end indices of the rows in the
                weighted similarities array to populate.
            weighted_similarities_spec: diversity.shared.SharedArraySpec
                The specification for the memory block into which the
                weighted similarities are inserted.
            features_spec: diversity.shared.SharedArraySpec
                The specification of the memory block in which the
                features of species are listed.
            relative_abundance_spec: diversity.shared.SharedArraySpec
                The specification for the memory block containing the
                relative abundances of species (per community).
            """
            LOGGER.debug(
                "ApplySimilarityFunction.__call__(row_start=%s, row_stop=%s,"
                " weighted_similarities_spec=%s, features_spec=%s,"
                " relative_abundance_spec=%s)",
                row_start,
                row_stop,
                weighted_similarities_spec,
                features_spec,
                relative_abundance_spec,
            )
            with LoadSharedArray(
                weighted_similarities_spec
            ) as weighted_similarities, LoadSharedArray(
                relative_abundance_spec
            ) as relative_abundances, LoadSharedArray(
                features_spec
            ) as features:
                similarities_row_i = empty(
                    shape=(len(self.__valid_features_index),), dtype=dtype("f8")
                )
                for i, feature_i in enumerate(
                    self.__valid_features_index[row_start:row_stop]
                ):
                    for j, feature_j in enumerate(self.__valid_features_index):
                        similarities_row_i[j] = self.func(
                            features.data[feature_i], features.data[feature_j]
                        )
                    weighted_similarities.data[i + row_start] = (
                        similarities_row_i @ relative_abundances.data
                    )

    def __init__(
        self,
        similarity_function,
        features_spec,
        species_ordering,
        species_subset,
        shared_array_manager,
        num_processors=None,
    ):
        """Initializes object.

        Parameters
        ----------
        similarity_function: Callable
            Callable to determine similarity between species. Must take
            two items from the features argument and return a numeric
            similarity value. Must be pickleable.
        features: numpy.ndarray
            A numpy.ndarray where each item along axis 0 comprises the
            features of a species that are passed to
            similarity_function. The order of features is determined
            by species_ordering.
        species_ordering: pandas.Index
            The unique species in order corresponding to features. Row
            and column ordering in similarity matrix calculations is
            determined by this argument.
        species_subset: collection of str objects supporting membership test
            The species to include. Only similarities from columns and
            rows corresponding to these species are used.
        shared_array_manager: diversity.shared.SharedArrayManager
            An active manager for creating shared arrays.
        num_processors: int
            Number of processors to use.
        """
        LOGGER.debug(
            "SimilarityFromFunction(similarity_function=%s,"
            " features_spec=%s, species_ordering=%s, species_subset=%s"
            " shared_array_manager=%s, num_processors=%s)",
            similarity_function,
            features_spec,
            species_ordering,
            species_subset,
            shared_array_manager,
            num_processors,
        )
        valid_features_index = species_ordering.astype(str).isin(species_subset)
        self.similarity_function = self.ApplySimilarityFunction(
            similarity_function, where(valid_features_index)[0]
        )
        self.__features_spec = features_spec
        self.__species_ordering = species_ordering[valid_features_index]
        self.__shared_array_manager = shared_array_manager
        self.__num_processors = self.__get_num_processors(num_processors)

    def __get_num_processors(self, num_requested):
        return min(
            n_processors
            for n_processors in [num_requested, cpu_count(), len(self.species_ordering)]
            if n_processors is not None
        )

    @property
    def species_ordering(self):
        return self.__species_ordering

    def calculate_weighted_similarities(self, relative_abundances):
        """Same as diversity.metacommunity.ISimilarity.calculate_weighted_similarities, except using shared arrays."""
        LOGGER.debug(
            "calculate_weighted_similarities(relative_abundances=%s)",
            relative_abundances,
        )
        weighted_similarities = self.__shared_array_manager.empty(
            shape=relative_abundances.spec.shape, data_type=dtype("f8")
        )
        row_chunks = partition_range(
            range(len(self.species_ordering)), self.__num_processors
        )
        args_list = [
            (
                chunk.start,
                chunk.stop,
                weighted_similarities.spec,
                self.__features_spec,
                relative_abundances.spec,
            )
            for chunk in row_chunks
        ]
        with Pool(self.__num_processors) as pool:
            pool.starmap(self.similarity_function, args_list)
        return weighted_similarities


class SimilarityFromMemory(ISimilarity):
    """Implements Similarity using similarities stored in memory."""

    def __init__(self, similarity_matrix, species_subset):
        """Initializes object.

        similarity_matrix: pandas.DataFrame
            Similarities between species. Columns and index must be
            species names corresponding to the values in their rows and
            columns.
        species: Set
            Set of species to include. If None, all species are
            included.
        """
        self.__species_ordering = self.__get_species_ordering(
            similarity_matrix, species_subset
        )
        self.similarity_matrix = self.__reindex_similarity_matrix(similarity_matrix)

    @property
    def species_ordering(self):
        return self.__species_ordering

    def __get_species_ordering(self, similarity_matrix, species_subset):
        species = similarity_matrix.columns.astype(str)
        species_subset_indices = species.isin(species_subset)
        return species[species_subset_indices]

    def __reindex_similarity_matrix(self, similarity_matrix):
        return similarity_matrix.reindex(
            index=self.species_ordering, columns=self.species_ordering, copy=False
        )

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity_matrix.to_numpy() @ relative_abundances
