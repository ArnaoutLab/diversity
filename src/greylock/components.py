from functools import cached_property
from numpy import ndarray

from greylock.abundance import Abundance
from greylock.similarity import Similarity


class Components:
    """Dispatches diversity components based on specified measure.
    If the similarity matrix is not the identity matrix, these
    will be similarity-sensitive diversity components."""

    def __init__(self, abundance: Abundance, similarity: Similarity) -> None:
        self.abundance = abundance

        """Create the ordinariness vectors by multipying the
        similarity matrix with each of the metacommunity abundance vector,
        the subcommunity abundance vectors, and the normalized
        subcommunity vectors.
        (See Leinster book* page 174 for discussion of ordinariness.
        * https://arxiv.org/pdf/2012.02113)
        Of course, for IdentitySimilarity, this multiplication would be
        a no-op (and thus is not actually performed).
        """
        (
            self.metacommunity_ordinariness,
            self.subcommunity_ordinariness,
            self.normalized_subcommunity_ordinariness,
        ) = self.abundance.premultiply_by(similarity)

        self.numerators = {
            **dict.fromkeys(["alpha", "gamma", "normalized_alpha"], 1),
            **dict.fromkeys(
                [
                    "beta",
                    "rho",
                    "normalized_beta",
                    "normalized_rho",
                    "beta_hat",
                    "rho_hat",
                ],
                self.metacommunity_ordinariness,
            ),
        }
        self.denominators = {
            **dict.fromkeys(
                ["alpha", "beta", "rho", "beta_hat", "rho_hat"],
                self.subcommunity_ordinariness,
            ),
            **dict.fromkeys(
                ["normalized_alpha", "normalized_beta", "normalized_rho"],
                self.normalized_subcommunity_ordinariness,
            ),
            "gamma": self.metacommunity_ordinariness,
        }
