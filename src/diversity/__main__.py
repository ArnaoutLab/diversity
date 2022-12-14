"""Main module for executing diversity on command-line.

Functions
---------
main
    Calculates diversities according to command-line specifications.
"""
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger

from pandas import read_csv, concat

from diversity.log import LOG_HANDLER, LOGGER
from diversity.metacommunity import Metacommunity
from diversity.parameters import configure_arguments
from diversity.utilities import get_file_delimiter

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger("py.warnings").addHandler(LOG_HANDLER)


def main(args):
    """Calculates diversity from species counts and similarities.

    Parameters
    ----------
    args: argparse.Namespace
        Return object of argparse.ArgumentParser object created by
        diversity.parameters.configure_arguments and applied to command
        line arguments.
    """
    LOGGER.setLevel(args.log_level)
    LOGGER.info(" ".join([f"python{python_version()}", *argv]))
    LOGGER.debug(f"args: {args}")

    delimiter = get_file_delimiter(args.input_filepath)
    counts = read_csv(args.input_filepath, sep=delimiter, dtype=int)
    LOGGER.debug(f"data: {counts}")
    meta = Metacommunity(
        counts=counts,
        similarity=args.similarity,
        chunk_size=args.chunk_size,
    )
    community_views = []
    for view in args.viewpoint:
        community_views.append(meta.subcommunities_to_dataframe(view))
        community_views.append(meta.metacommunity_to_dataframe(view))

    community_views = concat(community_views, ignore_index=True)
    community_views.viewpoint = community_views.viewpoint.map(
        lambda v: format(v, ".2f")
    )
    community_views.to_csv(
        args.output_filepath, sep="\t", float_format="%.4f", index=False
    )
    LOGGER.info("Done!")


if __name__ == "__main__":
    parser = configure_arguments()
    args = parser.parse_args()
    main(args)
