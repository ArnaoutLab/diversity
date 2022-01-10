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
from collections.abc import Iterable
from csv import reader, writer
from functools import cached_property
from pathlib import Path
from typing import Callable

from pandas import DataFrame
from numpy import dot, array, empty, zeros, unique, broadcast_to, divide, float64

from metacommunity.utilities import (
    InvalidArgumentError,
    power_mean,
    unique_correspondence,
)


class Abundance:
    """Relative abundances of species in a metacommunity.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities.

    Attributes
    ----------
    counts: numpy.ndarray
        A 2-d numpy.ndarray with subcommunity identifiers, species
        identifiers and number of appearances of the row's species in
        the row's subcommunity. The column ordering is determined by the
        subcommunity_column, species_column, and counts_column
        attributes. Each combination of species and subcommunity must
        appear no more than once.
    species_order: Iterable
        Ordered unique species identifiers. The ordering determines in
        which order values corresponding to each species are returned by
        methods the object.
    subcommunity_column: int
        Index of subcommunity identifier column in counts.
    species_column: int
        Index of species identifier column in counts.
    count_column: int
        Index of species count column in counts.
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
        self.counts = counts
        self.species_order, self.__species_unique_pos = unique_correspondence(
            items=self.counts[:, self.species_column],
            ordered_items=species_order,
        )
        self.subcommunity_order, self.__subcommunity_unique_pos = unique_correspondence(
            items=self.counts[:, self.subcommunity_column],
            ordered_items=subcommunity_order,
        )
        self.subcommunity_column = subcommunity_column
        self.species_column = species_column
        self.count_column = count_column

    def __pivot_table(self):
        table = zeros((len(), len(self.subcommunity_order)), dtype=float64)
        table[self.__species_unique_pos, self.__subcommunity_unique_pos] = self.counts[
            :, self.counts_column
        ]
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


class Similarity:
    """Species similarities weighted by meta- and subcommunity abundance.

    Attributes
    ----------
    abundance: Abundance
        Relative species abundances in metacommunity and its
        subcommunities.
    similarities_filepath: str
        Path to file containing species similarity matrix. If it doesn't
        exist, the write_similarity_matrix method generates one. File
        must have a header listing the species names according to the
        column ordering of the matrix. Column and row ordering must be
        the same.
    similarity_function: Callable
        Similarity function used to generate similarity matrix file.
    features: numpy.ndarray
        A 2d numpy.ndarray where rows are species and columns correspond
        to features. The order of features corresponds to the species
        argument.
    species: numpy.ndarray
        A 1d numpy.nds array of unique species corresponding to the rows
        in features.
    species_to_idx: FrozenDict
        Maps species names uniquely to integers between 0 and n_species - 1.
    """

    def __init__(
        self,
        counts,
        similarity_matrix=None,
        similarities_filepath=None,
        similarity_function=None,
        features=None,
        species_order=None,
    ):
        if species_order is None:
            species_order = self.__get_species_order()
        self.abundance = Abundance(counts, species_order)
        self.similarity_matrix = similarity_matrix
        self.similarities_filepath = similarities_filepath
        self.similarity_function = similarity_function
        self.features = features
        self.__validate_features()

    def __get_species_order(self):
        if self.similarities_filepath is None:
            raise InvalidArgumentError(
                "Unable to determine species ordering to correspond between"
                " species counts and similarities. If no similarity matrix"
                " filepath is provided, then the species order must be"
                " specified."
            )
        else:
            with open(self.similarities_filepath, "r") as file:
                return next(reader(file))

    def __validate_features(self):
        if (
            self.similarity_matrix,
            self.similarities_filepath,
            self.similarity_function,
        ) == (None, None, None):
            raise InvalidArgumentError(
                "At least one of similarity_matrix, similarities_filepath, and"
                " similarity_function must be specified to initialize a"
                " Similarity object."
            )
        if self.similarity_matrix is None and self.similarities_filepath is None:
            raise InvalidArgumentError(
                "Must specify similarities_filepath if not using" " similarity_matrix."
            )
        if self.similarity_function is not None and self.features is None:
            raise InvalidArgumentError(
                "Must specify features if similarity_function is provided."
            )
        if self.features is not None:
            if self.features.shape[0] != len(self.species_order):
                raise InvalidArgumentError(
                    "Number of entries in features array doesn't match number"
                    " of species."
                )
        if self.similarity_matrix is not None:
            if (
                len(self.species_order) != self.similarity_matrix.shape[0]
                or len(self.species_order) != self.similarity_matrix.shape[1]
            ):
                raise InvalidArgumentError(
                    "Dimensions of similarity matrix don't correspond to the"
                    " number of species."
                )

    @cached_property
    def metacommunity_similarity(self):
        """Calculates the sums of similarities weighted by the metacommunity
        abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the subcommunity
        abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the normalized
        subcommunity abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def __write_similarity_matrix(self):
        """Writes species similarity matrix into file.

        The matrix is written into file referred to by object's
        similarities_filepath attribute. Any existing contents are
        overwritten.
        """
        row_i = empty(self.species_order.shape, dtype=float64)
        with open(self.similarities_filepath, "w") as file:
            csv_writer = writer(file)
            csv_writer.writerow(self.species_order)  # write header
            for features_i in self.features:
                for j, features_j in enumerate(self.features):
                    row_i[j] = self.similarity_function(features_i, features_j)
                csv_writer.writerow(row_i)

    def __weighted_similarities_from_file(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that similarities
        are read from similarities file referred to by object's
        similarities_filepath attribute.
        """
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
        with open(self.similarities_filepath, "r") as file:
            next(reader(file))
            for i, row in enumerate(reader(file)):
                similarities_row = array(row, dtype=float64)
                weighted_similarities[i, :] = dot(similarities_row, relative_abundances)
        return weighted_similarities

    def __weighted_similarities_from_array(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that the object's
        similarity_matrix attribute is used.
        """
        return dot(self.similarity_matrix, relative_abundances)

    def __calculate_weighted_similarities(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Attempts to read similarities from the file at object's
        similarities_filepath attribute, if it exists. If it doesn't
        exist, a file is generated using the object's
        similarity_function attribute.

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
        if self.similarity_matrix is not None:
            return self.__weighted_similarities_from_array(relative_abundances)
        if not self.similarities_filepath.is_file():
            self.__write_similarity_matrix()
        return self.__weighted_similarities_from_file(relative_abundances)


class Metacommunity:
    """Class for metacommunities and calculating their diversity.

    Attributes
    ----------
    counts: numpy.ndarray
    viewpoint: float
    similarities_filepath: str
    similarity_function: Callable
    features: np.ndarray
    abundance: Abundance
    similarity: Similarity
    """

    def __init__(
        self,
        counts,
        similarities_filepath,
        similarity_matrix=None,
        similarity_function=None,
        features=None,
        species_order=None,
    ):
        if isinstance(counts, DataFrame):
            counts = counts.to_numpy()
        if isinstance(similarity_matrix, DataFrame):
            similarity_matrix = similarity_matrix.to_numpy()
        if similarities_filepath:
            similarities_filepath = Path(similarities_filepath)
        self.similarity = Similarity(
            counts=counts,
            similarity_matrix=similarity_matrix,
            similarities_filepath=similarities_filepath,
            similarity_function=similarity_function,
            features=features,
            species_order=species_order,
        )

    # FIXME repr is potientially large for datalasses, right? could be slow
    def __hash__(self):
        return hash(repr(self))

    @cache
    def alpha(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint, 1, self.similarity.subcommunity_similarity
        )

    @cache
    def rho(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint,
            self.similarity.metacommunity_similarity,
            self.similarity.subcommunity_similarity,
        )

    @cache
    def beta(self, viewpoint):
        return 1 / self.rho(viewpoint)

    @cache
    def gamma(self, viewpoint):
        denominator = broadcast_to(
            self.similarity.metacommunity_similarity,
            self.similarity.abundance.normalized_subcommunity_abundance.shape,
        )
        return self.subcommunity_measure(viewpoint, 1, denominator)

    @cache
    def normalized_alpha(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint, 1, self.similarity.normalized_subcommunity_similarity
        )

    @cache
    def normalized_rho(self, viewpoint):
        return self.subcommunity_measure(
            viewpoint,
            self.similarity.metacommunity_similarity,
            self.similarity.normalized_subcommunity_similarity,
        )

    @cache
    def normalized_beta(self, viewpoint):
        return 1 / self.normalized_rho(viewpoint)

    @cache
    def A(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.alpha)

    @cache
    def R(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.rho)

    @cache
    def B(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.beta)

    @cache
    def G(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.gamma)

    @cache
    def normalized_A(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_alpha)

    @cache
    def normalized_R(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_rho)

    @cache
    def normalized_B(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_beta)

    def subcommunity_measure(self, viewpoint, numerator, denominator):
        similarities = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        return power_mean(
            1 - viewpoint,
            self.similarity.abundance.normalized_subcommunity_abundance,
            similarities,
        )

    def metacommunity_measure(self, viewpoint, subcommunity_function):
        subcommunity_measure = subcommunity_function(viewpoint)
        return power_mean(
            1 - viewpoint,
            self.similarity.abundance.subcommunity_normalizing_constants,
            subcommunity_measure,
        )

    def subcommunities_to_dataframe(self, viewpoint):
        return DataFrame(
            {
                "community": self.similarity.abundance.subcommunity_names,
                "viewpoint": viewpoint,
                "alpha": self.alpha(viewpoint),
                "rho": self.rho(viewpoint),
                "beta": self.beta(viewpoint),
                "gamma": self.gamma(viewpoint),
                "normalized_alpha": self.normalized_alpha(viewpoint),
                "normalized_rho": self.normalized_rho(viewpoint),
                "normalised_beta": self.normalized_beta(viewpoint),
            }
        )

    def metacommunity_to_dataframe(self, viewpoint):
        return DataFrame(
            {
                "community": "metacommunity",
                "viewpoint": viewpoint,
                "A": self.A(viewpoint),
                "R": self.R(viewpoint),
                "B": self.B(viewpoint),
                "G": self.G(viewpoint),
                "normalized_A": self.normalized_A(viewpoint),
                "normalized_R": self.normalized_R(viewpoint),
                "normalized_B": self.normalized_B(viewpoint),
            },
            index=[0],
        )
