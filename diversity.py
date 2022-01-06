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
from csv import reader, writer
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Callable

from pandas import DataFrame
from numpy import dot, array, empty, lexsort, zeros, argsort, divide, float64

from utilities import InvalidArgumentError, pivot_table, power_mean, reorder_rows


@dataclass
class Abundance:
    """Relative abundances of species in a metacommunity.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities.

    Attributes
    ----------
    counts: array
        A 2-d structured numpy.array with species names in the first
        column, number of appearances in the second column and
        subcommunity names in the third column.
    unique_species_correspondence: UniqueRowsCorrespondence
        Correspondence between rows in counts data with a unique
        ordering of the unique species listed in its first column.
    """

    counts: array
    species_simiarlity_order: list
    subcommunity_names: array = field(init=False)

    def __post_init__(self):
        # These properties are used for every diversity measure, so they should be calculated on initialization
        self.subcommunity_abundance
        self.subcommunity_normalizing_constants
        self.normalized_subcommunity_abundance

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
        # FIXME This function still assumes columns in the input file are ordered: subcommunity, species, count
        metacommunity_counts, self.subcommunity_names = pivot_table(
            self.counts, columns_index=0, indices_index=1, values_index=2)
        metacommunity_counts = reorder_rows(
            metacommunity_counts, self.species_simiarlity_order)
        total_abundance = metacommunity_counts.sum()
        return metacommunity_counts / total_abundance

    @ cached_property
    def subcommunity_normalizing_constants(self):
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=0)

    @ cached_property
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
        return (self.subcommunity_abundance
                / self.subcommunity_normalizing_constants)


@ dataclass
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
    """

    abundance: Abundance
    similarity_matrix: array = None
    similarities_filepath: str = None
    similarity_function: Callable = None
    features: array = None
    species_order: array = None

    def __post_init__(self):
        """Validates attributes."""
        self.validate_features()

    def validate_features(self):
        if self.features is not None:
            if self.species_order is None:
                raise InvalidArgumentError(
                    'If features argument is provided, then species must be'
                    ' provided to establish the similarity matrix row and column ordering.')
            elif self.features.shape[0] != len(self.species_order):
                raise InvalidArgumentError(
                    'Invalid species array shape. Expected 1-d array of'
                    ' length equal to number of rows in features')

    @ cached_property
    def metacommunity_similarity(self):
        """Calculates the sums of similarities weighted by the metacommunity
        abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance)

    @ cached_property
    def subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the subcommunity
        abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance)

    @ cached_property
    def normalized_subcommunity_similarity(self):
        """Calculates the sums of similarities weighted by the normalized
        subcommunity abundance of each species.
        """
        return self.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance)

    def write_similarity_matrix(self):
        """Writes species similarity matrix into file.

        The matrix is written into file referred to by object's
        similarities_filepath attribute. Any existing contents are
        overwritten.
        """
        row_i = empty(self.species_order.shape, dtype=float64)
        with open(self.similarities_filepath, 'w') as file:
            csv_writer = writer(file)
            csv_writer.writerow(self.species_order)  # write header
            for features_i in self.features:
                for j, features_j in enumerate(self.features):
                    row_i[j] = self.similarity_function(features_i, features_j)
                csv_writer.writerow(row_i)

    def weighted_similarities_from_file(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that similarities
        are read from similarities file referred to by object's
        similarities_filepath attribute.
        """
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
        # species_order = self.abundance.species_simiarlity_order
        with open(self.similarities_filepath, 'r') as file:
            header = next(reader(file))  # skip header
            order = argsort(header)
            for i, row in enumerate(reader(file)):
                similarities_row = array(row, dtype=float64)
                weighted_similarities[i, :] = dot(similarities_row,
                                                  relative_abundances[order, :])
        return weighted_similarities

    def weighted_similarities_from_array(self, relative_abundances):
        """Calculates weighted sums of similarities to each species.

        Same as calculate_weighted_similarities, except that the object's
        similarity_matrix attribute is used.
        """
        return dot(self.similarity_matrix, relative_abundances)

    def calculate_weighted_similarities(self, relative_abundances):
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
            return self.weighted_similarities_from_array(relative_abundances)
        if not self.similarities_filepath.is_file():
            self.write_similarity_matrix()
        return self.weighted_similarities_from_file(relative_abundances)


@ dataclass
class Metacommunity:
    """Class for metacommunities and calculation their diversity.

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

    counts: array
    similarities_filepath: str = None
    similarity_matrix: array = None
    similarity_function: Callable = None
    features: array = None
    species_order: list = None
    abundance: Abundance = field(init=False)
    similarity: Similarity = field(init=False)

    def __post_init__(self):
        # sort by subcommunity and species # FIXME assumes input file columns are ordered: subcommunity, species, count
        self.counts = self.counts[lexsort(
            (self.counts[:, 0], self.counts[:, 1]))]
        if not self.species_order:
            self.species_order = self.get_species_order()
        if self.similarities_filepath:
            self.similarities_filepath = Path(self.similarities_filepath)
        self.abundance = Abundance(self.counts, self.species_order)
        self.similarity = Similarity(
            self.abundance,
            similarity_matrix=self.similarity_matrix,
            similarities_filepath=self.similarities_filepath,
            similarity_function=self.similarity_function,
            features=self.features,
            species_order=self.species_order)

    # FIXME repr is potientially large for datalasses, right? could be slow
    def __hash__(self):
        return hash(repr(self))

    def get_species_order(self):
        if self.similarities_filepath:
            with open(self.similarities_filepath, 'r') as file:
                return argsort(next(reader(file)))

    @ cache
    def alpha(self, viewpoint):
        return self.subcommunity_measure(viewpoint, 1, self.similarity.subcommunity_similarity)

    @ cache
    def rho(self, viewpoint):
        return self.subcommunity_measure(viewpoint, self.similarity.metacommunity_similarity, self.similarity.subcommunity_similarity)

    @ cache
    def beta(self, viewpoint):
        return 1 / self.rho(viewpoint)

    @ cache
    def gamma(self, viewpoint):
        return self.subcommunity_measure(viewpoint, 1, self.similarity.metacommunity_similarity)

    @ cache
    def normalized_alpha(self, viewpoint):
        return self.subcommunity_measure(viewpoint, 1, self.similarity.normalized_subcommunity_similarity)

    @ cache
    def normalized_rho(self, viewpoint):
        return self.subcommunity_measure(viewpoint, self.similarity.metacommunity_similarity, self.similarity.normalized_subcommunity_similarity)

    @ cache
    def normalized_beta(self, viewpoint):
        return 1 / self.normalized_rho(viewpoint)

    @ cache
    def A(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.alpha)

    @ cache
    def R(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.rho)

    @ cache
    def B(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.beta)

    @ cache
    def G(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.gamma)

    @ cache
    def normalized_A(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_alpha)

    @ cache
    def normalized_R(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_rho)

    @ cache
    def normalized_B(self, viewpoint):
        return self.metacommunity_measure(viewpoint, self.normalized_beta)

    def subcommunity_measure(self, viewpoint, numerator, denominator):
        similarities = divide(numerator, denominator, out=zeros(
            denominator.shape), where=denominator != 0)
        return power_mean(1 - viewpoint, self.abundance.normalized_subcommunity_abundance, similarities)

    def metacommunity_measure(self, viewpoint, subcommunity_function):
        subcommunity_measure = subcommunity_function(viewpoint)
        return power_mean(1 - viewpoint, self.abundance.subcommunity_normalizing_constants, subcommunity_measure)

    def subcommunities_to_dataframe(self, viewpoint):
        return DataFrame({
            'subcommunity': self.abundance.subcommunity_names,
            'viewpoint': viewpoint,
            'alpha': self.alpha(viewpoint),
            'rho': self.rho(viewpoint),
            'beta': self.beta(viewpoint),
            'gamma': self.gamma(viewpoint),
            'normalized_alpha': self.normalized_alpha(viewpoint),
            'normalized_rho': self.normalized_rho(viewpoint),
            'normalised_beta': self.normalized_beta(viewpoint)
        })

    def metacommunity_to_dataframe(self, viewpoint):
        return DataFrame({
            'viewpoint': viewpoint,
            'A': self.A(viewpoint),
            'R': self.R(viewpoint),
            'B': self.B(viewpoint),
            'G': self.G(viewpoint),
            'normalized_A': self.normalized_A(viewpoint),
            'normalized_R': self.normalized_R(viewpoint),
            'normalized_B': self.normalized_B(viewpoint)
        }, index=['Metacommunity'])
