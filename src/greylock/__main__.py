"""Main module for executing diversity on command-line.

Functions
---------
main
    Calculates diversities according to command-line specifications.
"""
from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger
from numpy import int64

from pandas import read_csv

from greylock.log import LOG_HANDLER, LOGGER
from greylock import Metacommunity
from greylock.parameters import configure_arguments

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

    counts = read_csv(args.input_filepath, sep=None, engine="python", dtype=int64)
    LOGGER.debug(f"data: {counts}")
    metacommunity = Metacommunity(
        counts=counts,
        similarity=args.similarity,
        chunk_size=args.chunk_size,
    )
    community_views = metacommunity.to_dataframe(viewpoint=args.viewpoint)
    community_views.to_csv(
        args.output_filepath, sep="\t", float_format="%.4f", index=False
    )
    LOGGER.info("Done!")


if __name__ == "__main__":
    parser = configure_arguments()
    args = parser.parse_args()
    main(args)
