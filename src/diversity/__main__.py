from collections import defaultdict
from json import load
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv, concat, DataFrame

from diversity.metacommunity import make_metacommunity
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

    species_counts = read_csv(args.input_file, sep=",").to_numpy()
    print(species_counts)

    LOGGER.debug(f"data: {species_counts}")

    meta = make_metacommunity(
        species_counts, similarity_matrix_filepath=args.similarity_matrix_filepath
    )

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
