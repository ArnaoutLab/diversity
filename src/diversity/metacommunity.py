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
SimilarityFromMemory
    Implements Similarity by storing similarities in memory.
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity subcommunity diversity measures.

Functions
---------
make_metacommunity
    Builds diversity.metacommunity.Metacommunity object according to
    parameter specification.
"""

from abc import ABC, abstractmethod
from functools import cache

from pandas import DataFrame, concat, read_csv, unique
from numpy import empty, flatnonzero, zeros, broadcast_to, divide, float64 

from diversity.log import LOGGER
from diversity.utilities import (
    get_file_delimiter,
    InvalidArgumentError,
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
        subcommunity_column="subcommunity",
        species_column="species",
        count_column="count",
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
        LOGGER.debug(
            "Abundance(%s, species_order=%s, subcommunity_order=%s,"
            "subcommunity_column=%s, species_column=%s, count_column=%s"
            % (
                counts,
                species_order,
                subcommunity_order,
                subcommunity_column,
                species_column,
                count_column,
            )
        )
        self.counts = counts
        self.subcommunity_column = subcommunity_column
        self.species_column = species_column
        self.count_column = count_column
        self.species_order, self.__species_unique_pos = unique_correspondence(
            items=self.counts[self.species_column].to_numpy(),
            ordered_unique_items=species_order,
        )
        self.subcommunity_order, self.__subcommunity_unique_pos = unique_correspondence(
            items=self.counts[self.subcommunity_column].to_numpy(),
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
        LOGGER.debug("Abundance.__pivot_table(%s)" % self)
        table = zeros(
            (len(self.species_order), len(self.subcommunity_order)), dtype=float64
        )
        table[self.__species_unique_pos, self.__subcommunity_unique_pos] = (
            self.counts[self.count_column].to_numpy().astype(float64)
        )
        return table

    @cache
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
        counts = self.__pivot_table()
        total_abundance = counts.sum()
        relative_abundances = empty(shape=counts.shape, dtype=float64)
        relative_abundances[:] = counts / total_abundance
        return relative_abundances

    @cache
    def metacommunity_abundance(self):
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity. The row ordering is
        established by the species_to_row attribute.
        """
        return self.subcommunity_abundance().sum(axis=1, keepdims=True)

    @cache
    def subcommunity_normalizing_constants(self):
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        return self.subcommunity_abundance().sum(axis=0)

    @cache
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
        return self.subcommunity_abundance() / self.subcommunity_normalizing_constants()


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
    def species_order(self):
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
        species_subset: Set
            The species to include. Only similarities from columns and
            rows corresponding to these species are used.
        chunk_size: int
            Number of rows to read from similarity matrix at a time.
        """
        self.similarity_matrix = similarity_matrix
        self.__delimiter = get_file_delimiter(self.similarity_matrix)
        self.chunk_size = chunk_size
        (
            self.__species_order,
            self.__usecols,
            self.__skiprows,
        ) = self.__get_species_order(species_subset)

    def __get_species_order(self, species_subset):
        """The species ordering used in similarity matrix file.

        Parameters
        ----------
        # FIXME this may not be a set
        species: Set
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
        LOGGER.debug("SimilarityFromFile.__get_species_order(%s, %s)" % (self, species_subset))
        with read_csv(
            self.similarity_matrix, delimiter=self.__delimiter, chunksize=1
        ) as similarity_matrix_chunks:
            species = (
                next(similarity_matrix_chunks)
                .columns
                .astype(str)
            )
        species_subset_indices = species.isin(species_subset)
        species_order = species[species_subset_indices]
        usecols = flatnonzero(species_subset_indices)
        skiprows = flatnonzero(~species_subset_indices) + 1
        return species_order, usecols, skiprows

    @property
    def species_order(self):
        return self.__species_order

    def calculate_weighted_similarities(self, relative_abundances):
        weighted_similarities = empty(relative_abundances.shape, dtype=float64)
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
        self.__species_order = self.__get_species_order(similarity_matrix, species_subset)
        self.similarity_matrix = self.__reindex_similarity_matrix(similarity_matrix)

    @property
    def species_order(self):
        return self.__species_order

    def __get_species_order(self, similarity_matrix, species_subset):
        species = similarity_matrix.columns.astype(str)
        species_subset_indices = species.isin(species_subset)
        return species[species_subset_indices]

    def __reindex_similarity_matrix(self, similarity_matrix):
        return similarity_matrix.reindex(index=self.species_order, columns=self.species_order, copy=False)

    def calculate_weighted_similarities(self, relative_abundances):
        return self.similarity_matrix.to_numpy() @ relative_abundances


class Metacommunity:
    """Class for metacommunities and calculating their diversity.

    All diversities computed by objects of this class are
    similarity-sensitive. See https://arxiv.org/abs/1404.6520 for
    precise definitions of the various diversity measures.
    """

    def __init__(self, abundance, similarity=None):
        """Initializes object.

        Parameters
        ----------
        similarity: diversity.metacommunity.Similarity
            Object for calculating abundance-weighted similarities.
        abundance: diversity.metacommunity.Abundance
            Object whose (sub-/meta-)community species abundances are
            used.
        """
        self.abundance = abundance
        self.similarity = similarity

        if self.similarity is not None:
            self.measure_components = { 
                    "alpha": (1, self.subcommunity_similarity),
                    "rho": (self.metacommunity_similarity, self.subcommunity_similarity),
                    "beta": (self.metacommunity_similarity, self.subcommunity_similarity),
                    "gamma": (1, self.metacommunity_similarity),
                    "normalized_alpha": (1, self.normalized_subcommunity_similarity),
                    "normalized_rho": (
                        self.metacommunity_similarity,
                        self.normalized_subcommunity_similarity,
                    ),
                    "normalized_beta": (
                        self.metacommunity_similarity,
                        self.normalized_subcommunity_similarity,
                    ),
                }
        else:
            self.measure_components = {
                "alpha": (1, self.abundance.subcommunity_abundance),
                "rho": (
                    self.abundance.metacommunity_abundance,
                    self.abundance.subcommunity_abundance,
                ),
                "beta": (
                    self.abundance.metacommunity_abundance,
                    self.abundance.subcommunity_abundance,
                ),
                "gamma": (1, self.abundance.metacommunity_abundance),
                "normalized_alpha": (
                    1,
                    self.abundance.normalized_subcommunity_abundance,
                ),
                "normalized_rho": (
                    self.abundance.metacommunity_abundance,
                    self.abundance.normalized_subcommunity_abundance,
                ),
                "normalized_beta": (
                    self.abundance.metacommunity_abundance,
                    self.abundance.normalized_subcommunity_abundance,
                ),
            }

    @cache
    def metacommunity_similarity(self):
        """Sums of similarities weighted by metacommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance()
        )

    @cache
    def subcommunity_similarity(self):
        """Sums of similarities weighted by subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance()
        )

    @cache
    def normalized_subcommunity_similarity(self):
        """Sums of similarities weighted by the normalized subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance()
        )

    @cache
    def subcommunity_measure(self, viewpoint, measure):
        """Calculates subcommunity diversity measures."""
        numerator, denominator = self.measure_components[measure]
        if callable(numerator):
            numerator = numerator()
        denominator = denominator()
        if measure == "gamma":
            denominator = broadcast_to(
                denominator, self.abundance.subcommunity_abundance().shape
            )
        community_ratio = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        result = power_mean(
            1 - viewpoint,
            self.abundance.normalized_subcommunity_abundance(),
            community_ratio,
        )
        if measure in ["beta", "normalized_beta"]:
            return 1 / result
        return result

    def metacommunity_measure(self, viewpoint, measure):
        """Calculates metcommunity diversity measures."""
        subcommunity_measure = self.subcommunity_measure(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants(),
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
        df = DataFrame(
        {key: self.subcommunity_measure(viewpoint, key)
            for key in self.measure_components.keys()})
        df.insert(0, 'viewpoint', viewpoint)
        df.insert(0, 'community', self.abundance.subcommunity_order)
        return df

    def metacommunity_to_dataframe(self, viewpoint):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species."""
        df = DataFrame(
        {key: self.metacommunity_measure(viewpoint, key)
            for key in self.measure_components.keys()},
            index=["metacommunity"]
            )
        df.insert(0, 'viewpoint', viewpoint)
        df.reset_index(level=0)
        return df

def make_similarity(similarity_matrix, species_subset, chunk_size):
    similarity_type = type(similarity_matrix)
    if similarity_type not in {DataFrame, str}:
        raise InvalidArgumentError(
            "similarity_matrix must be a str or a pandas.DataFrame, but"
            f"was: {similarity_type}."
        )
    similarity_classes = {
        DataFrame: SimilarityFromMemory, 
        str: SimilarityFromFile
    }
    similarity_arguments = {
        DataFrame: (similarity_matrix, species_subset),
        str: (similarity_matrix, species_subset, chunk_size),
    }
    similarity_class = similarity_classes[similarity_type]
    initializer_arguments = similarity_arguments[similarity_type]
    return similarity_class(*initializer_arguments)


def subset_subcommunities(counts, subcommunities, subcommunity_column):
    if subcommunities is None:
        return counts
    return counts[counts[subcommunity_column].isin(subcommunities)]


def make_metacommunity(
    counts,
    similarity_matrix=None,
    subcommunities=None,
    chunk_size=1,
    subcommunity_column="subcommunity",
    species_column="species",
    count_column="count",
):
    """Builds a Metacommunity object from specified parameters.

    Parameters
    ----------
    counts: numpy.ndarray or pandas.DataFrame
        See diversity.metacommunity.Abundance. If the object is a
        pandas.DataFrame, its to_numpy method should return the expected
        numpy.ndarray.
    similarity_matrix: pandas.DataFrame, or str
        For data frame, see diversity.metacommunity.SimilarityFromMemory,
        and for str, see diversity.metacommunity.SimilarityFromFile.
    subcommunities: Set
        Names of subcommunities to include. Their union is the
        metacommunity, and data for all other subcommunities is ignored.
    chunk_size: int
        Optional. See diversity.metacommunity.SimilarityFromFile.
    subcommunity_column, species_column, count_column: str
        Used to specify non-default column headers. See
        diversity.metacommunity.Abundance.

    Returns
    -------
    A diversity.metacommunity.Metacommunity object built according to
    parameter specification.
    """
    LOGGER.debug(
        "make_metacommunity(%s, %s, subcommunities=%s, chunk_size=%s,"
        " subcommunity_column=%s, species_column=%s, count_column=%s"
        % (
            counts,
            similarity_matrix,
            subcommunities,
            chunk_size,
            subcommunity_column,
            species_column,
            count_column,
        )
    )

    counts_subset = subset_subcommunities(counts, subcommunities, subcommunity_column)
    species_subset = unique(counts_subset[species_column])
    abundance = Abundance(
        counts_subset,
        species_subset,
        subcommunity_column=subcommunity_column,
        species_column=species_column,
        count_column=count_column,
    )
    if similarity_matrix is not None:
        similarity = make_similarity(similarity_matrix, species_subset, chunk_size)
        arguements = (abundance, similarity)
    else:
        arguements = (abundance,)
    return Metacommunity(*arguements)


def make_pairwise_metacommunities(counts, similarity_matrix, subcommunity_column, **kwargs):
    subcommunties_groups = counts.groupby(subcommunity_column)
    pairwise_metacommunities = []
    for i, (_, group_i) in enumerate(subcommunties_groups):
            for j, (_, group_j) in enumerate(subcommunties_groups):
                if j > i:
                    counts = concat([group_i, group_j])
                    pair_ij = make_metacommunity(
                        counts, 
                        similarity_matrix, 
                        subcommunity_column=subcommunity_column, 
                        **kwargs)
                    pairwise_metacommunities.append(pair_ij)
    return pairwise_metacommunities