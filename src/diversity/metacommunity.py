"""Module for metacommunity and subcommunity diversity measures.

Classes
-------

Abundance:
    Species abundances in metacommunity.

Similarity:
    Species similarities weighted by relative abundance.

Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity subcommunity diversity measures.
"""
from abc import ABC, abstractmethod
from csv import reader
from functools import cached_property
from pathlib import Path
from warnings import warn

from pandas import DataFrame
from numpy import dot, array, empty, zeros, broadcast_to, divide, float64

from diversity.utilities import get_file_delimiter, power_mean, unique_correspondence


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


class SimilarityStrategy(ABC):
    @abstractmethod
    def calculate_weighted_similarities(self):
        """Abstract method"""


class SimilarityFileStrategy(SimilarityStrategy):
    def __init__(self, similarity_matrix_filepath):
        """
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
        self.delimiter = get_file_delimiter(self.similarity_matrix_filepath)
        with open(self.similarity_matrix_filepath, "r") as file:
            self.species_order = array(next(reader(file, delimiter=self.delimiter)))

    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that
        similarities are read from similarities file referred to by
        object's similarity_matrix_filepath attribute.
        """
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
        with open(self.similarity_matrix_filepath, "r") as file:
            next(reader(file, delimiter=self.delimiter))
            for i, row in enumerate(reader(file, delimiter=self.delimiter)):
                similarities_row = array(row, dtype=float64)
                weighted_similarities[i, :] = dot(similarities_row, relative_abundances)
        return weighted_similarities


class SimilarityFunctionStrategy(SimilarityFileStrategy):
    def __init__(
        self, similarity_matrix_filepath, similarity_function, features, species_order
    ):
        """
        Parameters
        ----------
        similarity_function: Callable
            Callable to determine similarity between species. Must take
            two numpy.ndarray objects containing species features as
            arguments and return a numeric similarity value. Must be
            specified together with similarity_matrix_filepath, a path to a
            non-existing filename to store similarities in, which will
            be generated when needed.
        features: numpy.ndarray
            A 2d numpy.ndarray where rows are species and columns
            correspond to features. The order of features is inferred
            from species_order.
        species_order: numpy.ndarray
            The unique species in desired order. Must be specified when
            similarity_matrix, or similarity_function are provided. Must
            not be specified, when similarity_matrix_filepath but not
            similarity_function as provided. Row and column ordering in
            similarity matrix calculations is determined by this
            argument.
        """
        super().__init__(similarity_matrix_filepath)
        self.similarity_function = similarity_function
        self.features = features
        self.species_order = species_order

    def format_row(self, row):
        return "\t".join(map(lambda x: format(x, ".3e"), row)) + "\n"

    def __write_similarity_matrix(self):
        """Writes species similarity matrix into file.

        The matrix is computed using the object's similarity_function
        attribute applied to the object's features attribute. The result
        is written into file referred to by object's
        similarity_matrix_filepath attribute together with a header
        establishing row and column ordering. Any existing contents are
        overwritten.
        """
        row_i = empty(len(self.species_order), dtype=float64)
        with open(self.similarity_matrix_filepath, "w") as file:
            file.write("\t".join(map(str, self.species_order)) + "\n")
            for features_i in self.features:
                for j, features_j in enumerate(self.features):
                    row_i[j] = self.similarity_function(features_i, features_j)
                formatted_row_i = self.format_row(row_i)
                file.write(formatted_row_i)

    def calculate_weighted_similarities(self, relative_abundances):
        if not self.similarity_matrix_filepath.is_file():
            self.__write_similarity_matrix()
        return super().calculate_weighted_similarities(relative_abundances)


class SimilarityInMemoryStrategy(SimilarityStrategy):
    def __init__(self, similarity_matrix, species_order):
        """
        similarity_matrix: numpy.ndarray
            2-d array of similarities between species. Must be specified
            together with species_order, which establishes row and
            column ordering. If None, a similarity function or a file
            containing similarities must be specified via the
            similarity_matrix_filepath and similarity_function parameters.
        """
        self.similarity_matrix = similarity_matrix
        self.species_order = species_order

    def calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that the object's
        similarity_matrix attribute is used.
        """
        return dot(self.similarity_matrix, relative_abundances)


def similarity_strategy_factory(
    similarity_matrix=None,
    species_order=None,
    similarity_matrix_filepath=None,
    similarity_function=None,
    features=None,
):
    in_memory_parameters = (similarity_matrix, species_order)
    function_parameters = (
        similarity_matrix_filepath,
        similarity_function,
        features,
        species_order,
    )
    file_parameters = (similarity_matrix_filepath,)

    if all(p is not None for p in in_memory_parameters):
        return SimilarityInMemoryStrategy(*in_memory_parameters)
    elif all(p is not None for p in function_parameters):
        return SimilarityFunctionStrategy(*function_parameters)
    elif all(p is not None for p in file_parameters):
        return SimilarityFileStrategy(*file_parameters)
    else:
        # FIXME need to refer to correct documentation
        raise Exception(
            "Invalid argument combination. See the documentation for valid argument combinations"
        )


class Metacommunity:
    """Class for metacommunities and calculating their diversity.

    Attributes
    ----------
    abundance: Abundance
    similarity: Similarity
    """

    def __init__(self, similarity, abundance):
        self.similarity = similarity
        self.abundance = abundance

    @cached_property
    def metacommunity_similarity(self):
        """Calculates the sums of similarities weighted by the metacommunity
        abundance of each species.
        """
        return self.similarity.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the subcommunity
        abundance of each species.
        """
        return self.similarity.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the normalized
        subcommunity abundance of each species.
        """
        return self.similarity.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def subcommunity_alpha(self, viewpoint):
        return self.subcommunity_measure(viewpoint, 1, self.subcommunity_similarity)

    def subcommunity_rho(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint,
            self.metacommunity_similarity,
            self.subcommunity_similarity,
        )

    def subcommunity_beta(self, viewpoint):
        return 1 / self.subcommunity_rho(viewpoint)

    def subcommunity_gamma(self, viewpoint):
        denominator = broadcast_to(
            self.metacommunity_similarity,
            self.abundance.normalized_subcommunity_abundance.shape,
        )
        return self.subcommunity_measure(viewpoint, 1, denominator)

    def normalized_subcommunity_alpha(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint, 1, self.normalized_subcommunity_similarity
        )

    def normalized_subcommunity_rho(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint,
            self.metacommunity_similarity,
            self.normalized_subcommunity_similarity,
        )

    def normalized_subcommunity_beta(self, viewpoint):
        return 1 / self.normalized_subcommunity_rho(viewpoint)

    def metacommunity_alpha(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.subcommunity_alpha)

    def metacommunity_rho(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.subcommunity_rho)

    def metacommunity_beta(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.subcommunity_beta)

    def metacommunity_gamma(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.subcommunity_gamma)

    def normalized_metacommunity_alpha(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_subcommunity_alpha)

    def normalized_metacommunity_rho(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_subcommunity_rho)

    def normalized_metacommunity_beta(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_subcommunity_beta)

    def subcommunity_measure(self, viewpoint, numerator, denominator):
        similarities = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        return power_mean(
            1 - viewpoint,
            self.abundance.normalized_subcommunity_abundance,
            similarities,
        )

    def metacommunity_measure(self, viewpoint, subcommunity_function):
        subcommunity_measure = subcommunity_function(viewpoint)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants,
            subcommunity_measure,
        )

    def subcommunities_to_dataframe(self, viewpoint):
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
    if isinstance(similarity_matrix, DataFrame):
        similarity_matrix = similarity_matrix.to_numpy()
    similarity = similarity_strategy_factory(
        similarity_matrix,
        species_order,
        similarity_matrix_filepath,
        similarity_function,
        features,
    )
    if isinstance(counts, DataFrame):
        counts = counts.to_numpy()
    abundance = Abundance(counts, similarity.species_order)
    return Metacommunity(similarity, abundance)
