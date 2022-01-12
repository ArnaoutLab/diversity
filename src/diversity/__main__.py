from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv, concat

from diversity.metacommunity import Abundance, Metacommunity, Similarity
from diversity.parameters import configure_arguments
from diversity.log import LOG_HANDLER, LOGGER

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger("py.warnings").addHandler(LOG_HANDLER)

########################################################################


def main():
    # Parse and validate arguments
    parser = configure_arguments()
    args = parser.parse_args()

    LOGGER.setLevel(args.log_level)
    LOGGER.info(" ".join([f"python{python_version()}", *argv]))
    LOGGER.debug(f"args: {args}")

    species_counts = read_csv(args.input_file, sep="\t")

    LOGGER.debug(f"data: {species_counts}")

    features = None  # FIXME read features in separately

    similarity = Similarity(similarities_filepath=args.similarity_matrix_file)
    abundance = Abundance(
        counts=species_counts.to_numpy(), species_order=similarity.species_order
    )
    meta = Metacommunity(similarity=similarity, abundance=abundance)

    community_views = []
    for view in args.viewpoint:
        community_views.append(meta.subcommunities_to_dataframe(view))
        community_views.append(meta.metacommunity_to_dataframe(view))

    community_views = concat(community_views, ignore_index=True)
    community_views.to_csv(args.output_file, sep="\t", float_format="%.4f", index=False)

    LOGGER.info("Done!")


########################################################################
if __name__ == "__main__":
    main()
