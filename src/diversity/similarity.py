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

from numpy import dtype, empty, flatnonzero
from pandas import DataFrame, read_csv

from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import (
    extract_data_if_shared,
    LoadSharedArray,
    SharedArrayManager,
    SharedArrayView,
)
from diversity.utilities import (
    get_file_delimiter,
    partition_range,
)


def make_similarity(
    similarity_method,
    species_subset,
    chunk_size=1,
    features_filepath=None,
    species_column=None,
    shared_array_manager=None,
    num_processors=None,
):
    """Initializes a concrete subclass of ISimilarity.

    Parameters
    ----------
    similarity_method: pandas.DataFrame, str, or Callable
        If pandas.DataFrame, see diversity.similarity.SimilarityFromMemory.
        If str, see diversity.similarity.SimilarityFromFile.
        If callable, see diversity.similarity.SimilarityFromFunction.
    species_subset: collection of str objects, which supports membership test
        The species to include. Only similarities between pairs of
        species from this collection are used.
    chunk_size: int
        See diversity.similarity.SimilarityFromFunction. Only relevant
        if a str is passed as argument for similarity_method.
    features_filepath, species_column: str
        See diversity.similarity.SimilarityFromFunction.read_shared_features.
        Only relevant if a callable is passed as argument for
        similarity_method.
    shared_array_manager, num_processors:
        See diversity.similarity.SimilarityFromFunction. Only relevant
        if a callable is passed as argument for similarity_method.

    Returns
    -------
    An instance of a concrete subclass of ISimilarity.

    Notes
    -----
    Valid parameter combinations are:
    (species_subset is required in all cases)
    - similarity_method: numpy.ndarray
      chunk_size: Any
      all others default
    - similarity_method: str
      chunk_size: int
      all others default
    - similarity_method: Callable
      chunk_size: Any
      features_filepath, species_column:
        see diversity.similarity.SimilarityFromFunction.read_shared_features
      shared_array_manager, num_processors:
        see diversity.similarity.SimilarityFromFunction.
    """
    LOGGER.debug(
        "make_similarity(similarity_method=%s, species_subset=%s,"
        " chunk_size=%s, features_filepath=%s, species_column=%s,"
        " shared_array_manager=%s, num_processors=%s)",
        similarity_method,
        species_subset,
        chunk_size,
        features_filepath,
        species_column,
        shared_array_manager,
        num_processors,
    )
    if isinstance(similarity_method, DataFrame) and all(
        map(
            lambda arg: arg is None,
            (features_filepath, species_column, shared_array_manager, num_processors),
        )
    ):
        similarity = SimilarityFromMemory(
            similarity_matrix=similarity_method, species_subset=species_subset
        )
    elif (
        isinstance(similarity_method, str)
        and isinstance(chunk_size, int)
        and all(
            map(
                lambda arg: arg is None,
                (
                    features_filepath,
                    species_column,
                    shared_array_manager,
                    num_processors,
                ),
            )
        )
    ):
        similarity = SimilarityFromFile(
            similarity_matrix=similarity_method,
            species_subset=species_subset,
            chunk_size=chunk_size,
        )
    elif (
        callable(similarity_method)
        and isinstance(features_filepath, str)
        and isinstance(species_column, str)
        and isinstance(shared_array_manager, SharedArrayManager)
        and (num_processors is None or isinstance(num_processors, int))
    ):
        features, species_ordering = SimilarityFromFunction.read_shared_features(
            filepath=features_filepath,
            species_column=species_column,
            species_subset=species_subset,
            shared_array_manager=shared_array_manager,
        )
        similarity = SimilarityFromFunction(
            similarity_function=similarity_method,
            features=features,
            species_ordering=species_ordering,
            shared_array_manager=shared_array_manager,
            num_processors=num_processors,
        )
    else:
        raise InvalidArgumentError(
            "Invalid argument types for make_similarity;"
            " similarity_method: %s, species_subset: %s,"
            " chunk_size: %s, features_filepath: %s, species_column: %s,"
            " shared_array_manager: %s, num_processors: %s",
            type(similarity_method),
            type(species_subset),
            type(chunk_size),
            type(features_filepath),
            type(species_column),
            type(shared_array_manager),
            type(num_processors),
        )
    return similarity


class ISimilarity(ABC):
    """Interface for classes computing weighted similarities."""

    @abstractmethod
    def calculate_weighted_similarities(self, relative_abundances, out=None):
        """Calculates weighted sums of similarities to each species.

        Parameters
        ----------
        relative_abundances: numpy.ndarray or diversity.shared.SharedArrayView
            Array of shape (n_species, n_communities), where rows
            correspond to unique species, columns correspond to
            (meta-/sub-) communities and each element is the relative
            abundance of a species in a (meta-/sub-)community.
        out: numpy.ndarray or diversity.shared.SharedArrayView
            Array of same shape as relative_abundances in which the
            result is stored, if desired.

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

    def __init__(self, similarity_matrix, species_subset, chunk_size=1):
        """Initializes object.

        Parameters
        ----------
        similarity_matrix: str
            Path to similarities file containing a square matrix of
            similarities between species, together with a header
            containing the unique species names in the matrix's row and
            column ordering.
        species_subset: collection of str objects, which supports membership test
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
        return (species_ordering, usecols, skiprows)

    @cached_property
    def species_ordering(self):
        return self.__species_ordering

    def calculate_weighted_similarities(self, relative_abundances, out=None):
        if out is None:
            out = empty(
                extract_data_if_shared(relative_abundances).shape, dtype=dtype("f8")
            )
        out_, relative_abundances_ = extract_data_if_shared(out, relative_abundances)
        with read_csv(
            self.similarity_matrix,
            delimiter=self.__delimiter,
            chunksize=self.chunk_size,
            usecols=self.__usecols,
            skiprows=self.__skiprows,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                out_[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ relative_abundances_
                )
                i += self.chunk_size
        return extract_data_if_shared(out)


class SimilarityFromFunction(ISimilarity):
    """Implements ISimilarity using a similarity function."""

    @staticmethod
    def read_shared_features(
        filepath, species_column, species_subset, shared_array_manager
    ):
        """Reads species features into a shared array.

        Parameters
        ----------
        filepath: str
            Path to .csv or .tsv file where one column contains species
            names and all other columns contain the features of the
            corresponding species.
        species_column: str
            Column header for column in features file which contains the
            species names.
        species_subset: collection of str objects, which supports membership test
            The species to include. Only similarities from columns and
            rows corresponding to these species are used.
        shared_array_manager: diversity.shared.SharedArrayManager
            An active manager for creating shared arrays.

        Returns
        -------
        A tuple consisting of:
        0: diversity.shared.SharedArrayView
            Contains one row of feature values per species.
        1: pandas.Index
            Contains the species corresponding in order to the rows in
            the shared features array.
        """
        LOGGER.debug(
            "read_features(filepath=%s, species_column=%s, species_subset=%s, shared_array_manager=%s)",
            filepath,
            species_column,
            species_subset,
            shared_array_manager,
        )
        delimiter = get_file_delimiter(filepath)
        features_df = read_csv(
            filepath,
            sep=delimiter,
            index_col=species_column,
            skiprows=lambda species: (
                species not in species_subset and species != species_column
            ),
        )
        features = shared_array_manager.from_array(features_df.to_numpy())
        species_ordering = features_df.index
        return (features, species_ordering)

    class ApplySimilarityFunction:
        """Applies similarity function to a chunk of data."""

        def __init__(self, func):
            LOGGER.debug("ApplySimilarityFunction(func=%s)", func)
            self.func = func

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
                    shape=(features.data.shape[0],), dtype=dtype("f8")
                )
                for i, feature_i in enumerate(features.data[row_start:row_stop]):
                    for j, feature_j in enumerate(features.data):
                        similarities_row_i[j] = self.func(feature_i, feature_j)
                    weighted_similarities.data[i + row_start] = (
                        similarities_row_i @ relative_abundances.data
                    )

    def __init__(
        self,
        similarity_function,
        features,
        species_ordering,
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
        features: diversity.shared.SharedArrayView
            Shared array containing one row of feature values per
            species.
        species_ordering: pandas.Index
            The species corresponding in order to the rows in the data
            stored in the shared features array.
        shared_array_manager: diversity.shared.SharedArrayManager
            An active manager for creating shared arrays.
        num_processors: int
            Number of processors to use.

        Notes
        -----
        Use class's static read_shared_features method to ensure that
        features_spec and species_ordering are read correctly.
        """
        LOGGER.debug(
            "SimilarityFromFunction(similarity_function=%s,"
            " features=%s, species_ordering=%s,"
            " shared_array_manager=%s, num_processors=%s)",
            similarity_function,
            features,
            species_ordering,
            shared_array_manager,
            num_processors,
        )
        if features.data.shape[0] != len(species_ordering):
            raise InvalidArgumentError(
                "Features and species ordering must be of the same"
                " length (features, species_ordering): ",
                features,
                species_ordering,
            )
        self.__features = features
        self.__species_ordering = species_ordering
        self.__similarity_function = self.ApplySimilarityFunction(
            func=similarity_function
        )
        self.__shared_array_manager = shared_array_manager
        self.__num_processors = self.__get_num_processors(num_processors)

    def __get_num_processors(self, num_requested):
        LOGGER.debug("__get_num_processors(num_requested=%s)", num_requested)
        return min(
            n_processors
            for n_processors in [num_requested, cpu_count(), len(self.species_ordering)]
            if n_processors is not None
        )

    @property
    def species_ordering(self):
        LOGGER.debug("species_ordering()")
        return self.__species_ordering

    def __make_shared(self, relative_abundances, out):
        """Creates shared arrays from the arguments if needed.

        out will be an empty shared array, and relative_abundances will
        be populated with the data already in the array.
        """
        LOGGER.debug(
            "__make_shared(relative_abundances=%s, out=%s)", relative_abundances, out
        )
        if not isinstance(relative_abundances, SharedArrayView):
            relative_abundances_ = self.__shared_array_manager.from_array(
                relative_abundances
            )
        else:
            relative_abundances_ = relative_abundances
        if out is None or not isinstance(out, SharedArrayView):
            out_ = self.__shared_array_manager.empty(
                shape=relative_abundances_.spec.shape, data_type=dtype("f8")
            )
        else:
            out_ = out
        return (relative_abundances_, out_)

    def calculate_weighted_similarities(self, relative_abundances, out=None):
        """Same as diversity.metacommunity.ISimilarity.calculate_weighted_similarities, except using shared arrays."""
        LOGGER.debug(
            "calculate_weighted_similarities(relative_abundances=%s, out=%s)",
            relative_abundances,
            out,
        )
        relative_abundances_, out_ = self.__make_shared(relative_abundances, out)

        row_chunks = partition_range(
            range(len(self.species_ordering)), self.__num_processors
        )
        args_list = [
            (
                chunk.start,
                chunk.stop,
                out_.spec,
                self.__features.spec,
                relative_abundances_.spec,
            )
            for chunk in row_chunks
        ]
        with Pool(self.__num_processors) as pool:
            pool.starmap(self.__similarity_function, args_list)
        if out is not None and not isinstance(out, SharedArrayView):
            out[:] = out_.data
            return out
        else:
            return out_.data


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

    def calculate_weighted_similarities(self, relative_abundances, out=None):
        if out is None:
            out = empty(
                extract_data_if_shared(relative_abundances).shape, dtype=dtype("f8")
            )
        out_, relative_abundances_ = extract_data_if_shared(out, relative_abundances)
        out_[:] = self.similarity_matrix.to_numpy() @ relative_abundances_
        return out_
