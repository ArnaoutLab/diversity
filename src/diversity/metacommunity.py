"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Abundance
    Species abundances in metacommunity.
Similarity
    Abstract base class for relative abundance-weighted species
    similarities.
SimilarityFromFile
    Implements Similarity by reading similarities from a file.
SimilarityFromFunction
    Implements Similarity by calculating similarities with a function.
SimilarityFromMemory
    Implements Similarity by storing similarities in memory.
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity subcommunity diversity measures.

Functions
---------
make_similarity
    Returns correct diversity.metacommunity.Similarity object fitting
    parameter specification.
make_metacommunity
    Builds diversity.metacommunity.Metacommunity object according to
    parameter specification.
"""
from abc import ABC, abstractmethod
from csv import reader
from functools import cached_property
from pathlib import Path

from pandas import DataFrame
from numpy import dot, array, empty, zeros, broadcast_to, divide, float64

from diversity.utilities import (
    InvalidArgumentError,
    get_file_delimiter,
    power_mean,
    unique_correspondence,
)


class Abundance:
    """Relative abundances of species in a metacommunity.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities that the
    species appears in.
    """

    def __init__(
        self,
        counts,
        species_order=None,
        subcommunity_order=None,
        subcommunity_column=0,
        species_column=1,
        count_column=2,
    ):
        """Initializes object.

        Determines species and subcommunity orderings if needed.

        Parameters
        ----------
        counts: numpy.ndarray
            A 2-d numpy.ndarray with subcommunity identifiers, species
            identifiers and number of appearances of the row's species
            in the row's subcommunity. The column ordering is determined
            by the subcommunity_column, species_column, and
            counts_column parameters. Each combination of species and
            subcommunity must appear no more than once.
        species_order: numpy.ndarray
            Ordered unique species identifiers. The ordering determines
            in which order values corresponding to each species are
            returned by methods of the object.
        subcommunity_order: Iterable
            Ordered unique subcommunity identifiers. The ordering
            determines in which order values corresponding to each
            species are returned by methods of the object.
        subcommunity_column: int
            Index of subcommunity identifier column in counts.
        species_column: int
            Index of species identifier column in counts.
        count_column: int
            Index of species count column in counts.
        """
        self.counts = counts
        self.subcommunity_column = subcommunity_column
        self.species_column = species_column
        self.count_column = count_column
        self.species_order, self.__species_unique_pos = unique_correspondence(
            items=self.counts[:, self.species_column],
            ordered_unique_items=species_order,
        )
        self.subcommunity_order, self.__subcommunity_unique_pos = unique_correspondence(
            items=self.counts[:, self.subcommunity_column],
            ordered_unique_items=subcommunity_order,
        )

    def __pivot_table(self):
        """Converts long to wide formatted counts.

        Returns
        -------
        A numpy.ndarray where rows correspond to species, columns to
        subcommunities and each element is the count of a species in a
        specific subcommunity.
        """
        table = zeros(
            (len(self.species_order), len(self.subcommunity_order)), dtype=float64
        )
        table[self.__species_unique_pos, self.__subcommunity_unique_pos] = self.counts[
            :, self.count_column
        ].astype(float64)
        return table

    @cached_property
    def subcommunity_abundance(self):
        """Calculates the relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        The row ordering is established by the species_to_row attribute.
        """
        metacommunity_counts = self.__pivot_table()
        total_abundance = metacommunity_counts.sum()
        return metacommunity_counts / total_abundance

    @cached_property
    def metacommunity_abundance(self):
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity. The row ordering is
        established by the species_to_row attribute.
        """
        return self.subcommunity_abundance.sum(axis=1, keepdims=True)

    @cached_property
    def subcommunity_normalizing_constants(self):
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=0)

    @cached_property
    def normalized_subcommunity_abundance(self):
        """Calculates normalized relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the subcommunity size. The row
        ordering is established by the species_to_row attribute.
        """
        return self.subcommunity_abundance / self.subcommunity_normalizing_constants


class Similarity(ABC):
    """Interface for classes computing weighted similarities."""

    def __init__(self, species_order):
        self.species_order = species_order

    @abstractmethod
    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Parameters
        ----------
        relative_abundances: numpy.ndarray
            Array of shape (n_species, n_communities), where rows
            correspond to unique species, columns correspond to
            (meta-/sub-)communities and each element is the relative
            abundance of a species in a (meta-/sub-)community.

        Returns
        -------
        A 2-d numpy.ndarray of shape (n_species, n_communities), where
        rows correspond to unique species, columns correspond to
        (meta-/sub-) communities and each element is a sum of
        similarities to one species weighted by the similarities stored
        in the similarities file.
        """
        pass


class SimilarityFromFile(Similarity):
    """Implements Similarity by using similarities stored in file."""

    def __init__(self, similarity_matrix_filepath):
        """Initializes object.

        similarity_matrix_filepath: str
            Path to similarities file. If similarity_function is None
            the file must exist and contain a square matrix of
            similarities between species, together with a header
            containing the unique species names in the matrix's row and
            column ordering. If similarity_function is also specified,
            the file must not exist, but will instead be generated as
            soon as it is needed in subsequent computations.
        """
        self.similarity_matrix_filepath = Path(similarity_matrix_filepath)
        self._delimiter = get_file_delimiter(self.similarity_matrix_filepath)
        self.species_order = self.__get_species_order()

    def __get_species_order(self):
        """The species ordering used in similarity matrix file."""
        with open(self.similarity_matrix_filepath, "r") as file:
            species_order = array(next(reader(file, delimiter=self._delimiter)))
        return species_order

    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Similarities are read from similarities file referred to by
        object's similarity_matrix_filepath attribute.

        See diversity.metacommunity.Similarity.calculate_weighted_similarities
        for complete specification.
        """
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
        with open(self.similarity_matrix_filepath, "r") as file:
            next(reader(file, delimiter=self._delimiter))
            for i, row in enumerate(reader(file, delimiter=self._delimiter)):
                similarities_row = array(row, dtype=float64)
                weighted_similarities[i, :] = dot(similarities_row, relative_abundances)
        return weighted_similarities


class SimilarityFromFunction(Similarity):
    """Implements Similarity using a similarity function."""

    def __init__(self, similarity_function, features, species_order):
        """Initializes object.

        Parameters
        ----------
        similarity_function: Callable
            Callable to determine similarity between species. Must take
            two items from the features argument and return a numeric
            similarity value.
        features: numpy.ndarray
            A numpy.ndarray where each item along axis 0 comprises the
            features of a species that are passed to
            similarity_function. The order of features is determined
            by species_order.
        species_order: Iterable
            The unique species in desired order. Row and column ordering
            in similarity matrix calculations is determined by this
            argument.
        """

        self.similarity_function = similarity_function
        self.features = features
        super().__init__(species_order)

    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Similarities are calculated using object's similarity_function
        attribute.

        See diversity.metacommunity.Similarity.calculate_weighted_similarities
        for complete specification.
        """
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
        similarities_row = empty(len(self.species_order), dtype=float64)
        for i, species_i in enumerate(self.features):
            for j, species_j in enumerate(self.features):
                similarities_row[j] = self.similarity_function(species_i, species_j)
            weighted_similarities[i, :] = dot(similarities_row, relative_abundances)
        return weighted_similarities


class SimilarityFromMemory(Similarity):
    """Implements Similarity using similarities stored in memory."""

    def __init__(self, similarity_matrix, species_order):
        """Initializes object.

        similarity_matrix: numpy.ndarray
            2-d array of similarities between species. Ordering of rows
            and columns must correspond to species_order argument.
        species_order: Iterable
            The unique species in desired order.
        """
        self.similarity_matrix = similarity_matrix
        super().__init__(species_order)

    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Similarities are calculated using object's similarity_matrix
        attribute.

        See diversity.metacommunity.Similarity.calculate_weighted_similarities
        for complete specification.
        """
        return dot(self.similarity_matrix, relative_abundances)


def make_similarity(
    similarity_matrix=None,
    species_order=None,
    similarity_matrix_filepath=None,
    similarity_function=None,
    features=None,
):
    """Creates a Similarity object from specified parameter combination.

    Valid parameter combinations and the returned types:
        - similarity_matrix, species_order -> SimilarityFromMemory
        - similarity_matrix_filepath -> SimilarityFromFile
        - species_order, similarity_matrix_filepath,
          similarity_function, features -> SimilarityFromFunction

    Parameters
    ----------
    similarity_matrix: numpy.ndarray
        Used by diversity.metacommunity.SimilarityFromMemory.
    species_order: Iterable
        Used by diversity.metacommunity.SimilarityFromMemory, or
        diversity.metacommunity.SimilarityFromFunction.
    similarity_matrix_filepath: str
        Used by diversity.metacommunity.SimilarityFromFile, or
        diversity.metacommunity.SimilarityFromFunction.
    similarity_function: Callable
        Used by diversity.metacommunity.SimilarityFromFunction.
    features: numpy.ndarray
        Used by diversity.metacommunity.SimilarityFromFunction.

    Returns
    -------
    An object whose type is the implementation of the abstract base
    Similarity according to parameter specification.
    """
    from_memory_parameters = (similarity_matrix, species_order)
    from_function_parameters = (
        similarity_function,
        features,
        species_order,
    )
    from_file_parameters = (similarity_matrix_filepath,)

    are_all_not_none = lambda parameters: all(p is not None for p in parameters)
    if are_all_not_none(from_memory_parameters):
        return SimilarityFromMemory(*from_memory_parameters)
    elif are_all_not_none(from_function_parameters):
        return SimilarityFromFunction(*from_function_parameters)
    elif are_all_not_none(from_file_parameters):
        return SimilarityFromFile(*from_file_parameters)
    else:
        raise InvalidArgumentError(
            "Invalid argument combination. See the documentation for"
            " valid argument combinations."
        )


class Metacommunity:
    """Class for metacommunities and calculating their diversity.

    All diversities computed by objects of this class are
    similarity-sensitive. See https://arxiv.org/abs/1404.6520 for
    precise definitions of the various diversity measures.
    """

    def __init__(self, similarity, abundance):
        """Initializes object.

        Parameters
        ----------
        similarity: diversity.metacommunity.Similarity
            Object for calculating abundance-weighted similarities.
        abundance: diversity.metacommunity.Abundance
            Object whose (sub-/meta-)community species abundances are
            used.
        """
        self.similarity = similarity
        self.abundance = abundance

    @cached_property
    def __metacommunity_similarity(self):
        """Sums of similarities weighted by metacommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def __subcommunity_similarity(self):
        """Sums of similarities weighted by subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def __normalized_subcommunity_similarity(self):
        """Sums of similarities weighted by the normalized subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def subcommunity_alpha(self, viewpoint):
        """Calculates alpha class diversities of subcommunities.

        Corresponds roughly to the diversities of subcommunities
        relative to the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__subcommunity_measure(viewpoint, 1, self.__subcommunity_similarity)

    def subcommunity_rho(self, viewpoint):
        """Calculates rho class diversities of subcommunities.

        Corresponds roughly to how redundant each subcommunity's classes
        are in the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__subcommunity_measure(
            viewpoint,
            self.__metacommunity_similarity,
            self.__subcommunity_similarity,
        )

    def subcommunity_beta(self, viewpoint):
        """Calculates beta class diversities of subcommunities.

        Corresponds roughly to how distinct each subcommunity's classes
        are from all classes in metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only
            the most frequent species.
        """
        return 1 / self.subcommunity_rho(viewpoint)

    def subcommunity_gamma(self, viewpoint):
        """Calculates gamma class diversities of subcommunities.

        Corresponds roughly to how much each subcommunity contributes
        towards the metacommunity diversity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only
            the most frequent species.
        """
        denominator = broadcast_to(
            self.__metacommunity_similarity,
            self.abundance.normalized_subcommunity_abundance.shape,
        )
        return self.__subcommunity_measure(viewpoint, 1, denominator)

    def normalized_subcommunity_alpha(self, viewpoint):
        """Calculates normalized alpha class diversities of subcommunities.

        Corresponds roughly to the diversities of subcommunities in
        isolation.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__subcommunity_measure(
            viewpoint, 1, self.__normalized_subcommunity_similarity
        )

    def normalized_subcommunity_rho(self, viewpoint):
        """Calculates normalized rho class diversities of subcommunities.

        Corresponds roughly to the representativeness of subcommunities.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__subcommunity_measure(
            viewpoint,
            self.__metacommunity_similarity,
            self.__normalized_subcommunity_similarity,
        )

    def normalized_subcommunity_beta(self, viewpoint):
        """Calculates normalized rho class diversities of subcommunities.

        Corresponds roughly to average diversity of subcommunities in
        the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return 1 / self.normalized_subcommunity_rho(viewpoint)

    def metacommunity_alpha(self, viewpoint):
        """Calculates alpha class diversity of metacommunity.

        Corresponds roughly to the average diversity of subcommunities
        relative to the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(viewpoint, self.subcommunity_alpha)

    def metacommunity_rho(self, viewpoint):
        """Calculates rho class diversitiy of metacommunity.

        Corresponds roughly to the average redundancy of subcommunities
        in the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(viewpoint, self.subcommunity_rho)

    def metacommunity_beta(self, viewpoint):
        """Calculates beta class diversity of metacommunity.

        Corresponds roughly to the average distinctness of
        subcommunities within the metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(viewpoint, self.subcommunity_beta)

    def metacommunity_gamma(self, viewpoint):
        """Calculates gamma class diversity of metacommunity.

        Corresponds roughly to the class diversity of the unpartitioned
        metacommunity.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(viewpoint, self.subcommunity_gamma)

    def normalized_metacommunity_alpha(self, viewpoint):
        """Calculates alpha class diversity of metacommunity.

        Corresponds roughly to the average diversity of subcommunities
        in isolation.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(
            viewpoint, self.normalized_subcommunity_alpha
        )

    def normalized_metacommunity_rho(self, viewpoint):
        """Calculates rho class diversitiy of metacommunity.

        Corresponds roughly to the average representativeness of
        subcommunities.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(viewpoint, self.normalized_subcommunity_rho)

    def normalized_metacommunity_beta(self, viewpoint):
        """Calculates beta class diversity of metacommunity.

        Corresponds roughly to the effective number of distinct
        subcommunities.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return self.__metacommunity_measure(
            viewpoint, self.normalized_subcommunity_beta
        )

    def __subcommunity_measure(self, viewpoint, numerator, denominator):
        """Calculates subcommunity diversity measures."""
        similarities = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        return power_mean(
            1 - viewpoint,
            self.abundance.normalized_subcommunity_abundance,
            similarities,
        )

    def __metacommunity_measure(self, viewpoint, subcommunity_function):
        """Calculates metcommunity diversity measures."""
        subcommunity_measure = subcommunity_function(viewpoint)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants,
            subcommunity_measure,
        )

    def subcommunities_to_dataframe(self, viewpoint):
        """Table containing all subcommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        return DataFrame(
            {
                "community": self.abundance.subcommunity_order,
                "viewpoint": viewpoint,
                "alpha": self.subcommunity_alpha(viewpoint),
                "rho": self.subcommunity_rho(viewpoint),
                "beta": self.subcommunity_beta(viewpoint),
                "gamma": self.subcommunity_gamma(viewpoint),
                "normalized_alpha": self.normalized_subcommunity_alpha(viewpoint),
                "normalized_rho": self.normalized_subcommunity_rho(viewpoint),
                "normalized_beta": self.normalized_subcommunity_beta(viewpoint),
            }
        )

    def metacommunity_to_dataframe(self, viewpoint):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species."""
        return DataFrame(
            {
                "community": "metacommunity",
                "viewpoint": viewpoint,
                "alpha": self.metacommunity_alpha(viewpoint),
                "rho": self.metacommunity_rho(viewpoint),
                "beta": self.metacommunity_beta(viewpoint),
                "gamma": self.metacommunity_gamma(viewpoint),
                "normalized_alpha": self.normalized_metacommunity_alpha(viewpoint),
                "normalized_rho": self.normalized_metacommunity_rho(viewpoint),
                "normalized_beta": self.normalized_metacommunity_beta(viewpoint),
            },
            index=[0],
        )


def make_metacommunity(
    counts,
    similarity_matrix=None,
    similarity_matrix_filepath=None,
    similarity_function=None,
    features=None,
    species_order=None,
):
    """Builds a Metacommunity object from specified parameters.

    Valid parameter combinations and the corresponding Similarity
    implementations used are:
        - similarity_matrix, species_order -> SimilarityFromMemory
        - similarity_matrix_filepath -> SimilarityFromFile
        - species_order, similarity_matrix_filepath,
          similarity_function, features -> SimilarityFromFunction

    Parameters
    ----------
    counts: numpy.ndarray or pandas.DataFrame
        See diversity.metacommunity.Abundance. If the object is a
        pandas.DataFrame, its to_numpy method should return the expected
        numpy.ndarray.
    similarity_matrix: numpy.ndarray
        See diversity.metacommunity.SimilarityFromMemory.
    similarity_matrix_filepath: str
        See diversity.metacommunity.SimilarityFromFile
    similarity_function: Callable
        See diversity.metacommunity.SimilarityFromFunction.
    features: numpy.ndarray
        See diversity.metacommunity.SimilarityFromFunction.
    species_order: Iterable
        See diversity.metacommunity.SimilarityFromMemory, or
        diversity.metacommunity.SimilarityFromFunction.

    Returns
    -------
    A diversity.metacommunity.Metacommunity object build according to
    parameter specification.
    """

    if_dataframe_to_numpy = lambda x: x.to_numpy() if isinstance(x, DataFrame) else x
    similarity_matrix = if_dataframe_to_numpy(similarity_matrix)
    features = if_dataframe_to_numpy(features)
    similarity = make_similarity(
        similarity_matrix,
        species_order,
        similarity_matrix_filepath,
        similarity_function,
        features,
    )
    counts = if_dataframe_to_numpy(counts)
    abundance = Abundance(counts, similarity.species_order)
    return Metacommunity(similarity, abundance)
